"""OFFLINE replay/visualisation of a master or export. Reads files only.

This module intentionally imports no robot adapters and has no code path that sends commands.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .util import iter_jsonl, read_json


def depth_to_preview(depth: np.ndarray) -> np.ndarray:
    """Colourised copy for display only; numeric depth is never replaced by this."""
    d = depth.astype(np.float32)
    valid = depth > 0
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1), 0, 1)
    img = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def iter_frames(path: str | Path):
    """Yield dicts {image_bgr, depth_preview, text} for a master episode or an export."""
    p = Path(path)
    man = read_json(p / "manifest.json")
    if man["kind"] == "export" and man["profile"] != "full_master":
        arr = {f.stem: np.load(f) for f in (p / "arrays").glob("*.npy")}
        for s in iter_jsonl(p / "samples.jsonl"):
            cam, ce = next(((c, e) for c, e in s["cameras"].items() if e.get("image_path")), (None, None))
            if cam is None:
                continue
            img = cv2.imread(str(p / ce["image_path"]))
            dp = None
            for k in ("depth_aligned_path", "depth_native_path"):
                if ce.get(k):
                    dp = depth_to_preview(cv2.imread(str(p / ce[k]), cv2.IMREAD_UNCHANGED))
                    break
            st = s["state"]
            ang = st.get("joint_angles_rad") or st.get("joint_angles_rad_optional")
            text = f"#{s['sample_index']} state={st['association_status']} " + (
                "angles_rad=" + ",".join(f"{a:.2f}" for a in ang) if ang else
                ("raw=" + str(st.get("motor_positions_raw")) if st.get("motor_positions_raw") else ""))
            yield {"image_bgr": img, "depth_preview": dp, "text": text}
        return
    # master or full_master export
    meta = read_json(p / "episode.json")
    rows = list(iter_jsonl(p / "sync" / "associations.jsonl"))
    state = list(iter_jsonl(p / "telemetry" / "follower_state.jsonl"))
    cams = {c: list(iter_jsonl(p / "streams" / c / "frames.jsonl")) for c in meta.get("streams", {})}
    for row in rows:
        c = row["ref_camera"]
        f = cams[c][row["frame_index"]]
        img = cv2.imread(str(p / "streams" / c / f["color_file"]))
        dp = None
        for k in ("depth_aligned_file", "depth_native_file"):
            if f.get(k):
                dp = depth_to_preview(cv2.imread(str(p / "streams" / c / f[k]), cv2.IMREAD_UNCHANGED))
                break
        sa = row["follower_state"]
        raw = state[sa["index"]]["raw"] if sa["status"] == "matched" else None
        yield {"image_bgr": img, "depth_preview": dp,
               "text": f"#{row['frame_index']} state={sa['status']} raw={raw}"}


def replay(path: str | Path, display: bool = True, fps: float = 30.0, max_frames: int | None = None) -> int:
    n = 0
    for fr in iter_frames(path):
        n += 1
        if display:
            img = fr["image_bgr"].copy()
            cv2.putText(img, fr["text"], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow("replay (offline; nothing is sent to the robot)", img)
            if fr["depth_preview"] is not None:
                cv2.imshow("depth preview (colourised copy)", fr["depth_preview"])
            if cv2.waitKey(int(1000 / fps)) in (27, ord("q")):
                break
        if max_frames and n >= max_frames:
            break
    if display:
        cv2.destroyAllWindows()
    return n
