"""Timestamp association (derived tables). Raw streams remain the source of truth.

Method ``nearest_host_monotonic``: every sample is placed on the host CLOCK_MONOTONIC axis
(camera: receipt time; robot state: midpoint of the bus read; commands: time stamped just before send).
Device timestamps (e.g. RealSense hardware clock) are NOT compared with host times; only an informational
offset estimate (median host - device) per camera is reported. No hardware synchronisation is claimed.
Latency between exposure and receipt is not removed (unknown unless measured).
"""
from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from .util import iter_jsonl, read_json

METHOD = "nearest_host_monotonic"


def depth_age_ms(frames: list[dict], status: list[str]) -> list[float | None]:
    """Host-receipt age of the depth frame attached to each colour frame, in ms (None where depth is missing).

    age = colour frame host receipt time - host receipt time of the FIRST frameset that carried this depth frame number.
    A fresh depth frame has age 0 by this definition. It is a lower bound of the true age: exposure-to-receipt latency of
    the depth sensor is unknown and not included. It is computed from host monotonic receipt times only (device clocks
    are never compared with host clocks).
    """
    first_seen: dict[int, int] = {}
    out: list[float | None] = []
    for f, s in zip(frames, status):
        n = f.get("depth_frame_number")
        if s == "missing" or n is None or f.get("host_mono_ns") is None:
            out.append(None)
            continue
        first_seen.setdefault(n, f["host_mono_ns"])
        out.append((f["host_mono_ns"] - first_seen[n]) / 1e6)
    return out


def depth_freshness(frames: list[dict]) -> dict:
    """Per-frame depth status for a camera with depth, from the recorded frame records (never from pixel content).

    A RealSense frameset can carry the SAME depth frame again (same frame number and device timestamp) next to a new colour
    frame. The recorder keeps the original depth frame number/timestamp, so this is detectable: status is
    ``fresh`` | ``repeated_previous_depth`` (stale by about one colour period) | ``missing`` (no depth frame or file).
    Aligned depth is derived from the same frameset's depth frame, so it inherits that frame's status.
    """
    status, prev = [], None
    for f in frames:
        n = f.get("depth_frame_number")
        if n is None or not f.get("depth_native_file"):
            status.append("missing")
        elif prev is not None and n == prev:
            status.append("repeated_previous_depth")
        else:
            status.append("fresh")
        if n is not None:
            prev = n
    nums = [f["depth_frame_number"] for f in frames if f.get("depth_frame_number") is not None]
    dts = [f["depth_device_timestamp_ms"] for f in frames if f.get("depth_device_timestamp_ms") is not None]
    off = [abs(f["depth_device_timestamp_ms"] - f["device_timestamp_ms"]) for f in frames
           if f.get("depth_device_timestamp_ms") is not None and f.get("device_timestamp_ms") is not None]
    age = depth_age_ms(frames, status)
    ages = [a for a in age if a is not None]
    run = longest = 0
    for s in status:
        run = run + 1 if s == "repeated_previous_depth" else 0
        longest = max(longest, run)
    counts = {k: status.count(k) for k in ("fresh", "repeated_previous_depth", "missing")}
    with_depth = counts["fresh"] + counts["repeated_previous_depth"]
    return {"status": status, "age_ms": age,
            "age_ms_stats": ({"max": max(ages), "mean": sum(ages) / len(ages)} if ages else None),
            "longest_repeat_run": longest,
            "repeat_fraction": (counts["repeated_previous_depth"] / with_depth) if with_depth else None,
            "counts": counts,
            "depth_frame_number_skipped": sum(b - a - 1 for a, b in zip(nums, nums[1:]) if b - a > 1),
            "depth_timestamp_non_monotonic": sum(1 for a, b in zip(dts, dts[1:]) if b < a),
            "depth_frame_number_non_monotonic": sum(1 for a, b in zip(nums, nums[1:]) if b < a),
            "color_depth_device_timestamp_offset_ms_max": max(off) if off else None}


def state_time(r: dict) -> int:
    return (r["t_start_ns"] + r["t_end_ns"]) // 2


def match_nearest(ref_ns: int, times: list[int], tol_ns: int, valid: list[bool] | None = None, direction: str = "nearest"):
    """Return (index, residual_ns, status). residual = sample_time - ref_time.

    direction: 'nearest' | 'before' (latest sample at or before ref) | 'after'.
    Samples further than tol are reported as unmatched (never reused as stale values).
    """
    if not times:
        return None, None, "no_samples"
    i = bisect.bisect_left(times, ref_ns)
    if direction == "before":
        cands = [k for k in (bisect.bisect_right(times, ref_ns) - 1,) if k >= 0]
    elif direction == "after":
        cands = [i] if i < len(times) else []
    else:
        cands = [k for k in (i - 1, i) if 0 <= k < len(times)]
    if valid is not None:
        cands = [k for k in cands if valid[k]] or []
    if not cands:
        return None, None, "no_valid_candidate"
    k = min(cands, key=lambda c: abs(times[c] - ref_ns))
    res = times[k] - ref_ns
    if abs(res) > tol_ns:
        return None, res, "unmatched_beyond_tolerance"
    return k, res, "matched"


def build_associations(ep: str | Path, sync_cfg: dict, reference_camera: str | None = None) -> tuple[list[dict], dict]:
    ep = Path(ep)
    meta = read_json(ep / "episode.json")
    cams = sorted(meta.get("streams", {}))
    ref = reference_camera or sync_cfg.get("reference_camera") or next(
        (c for c in cams if meta["streams"][c].get("role") == "primary"), cams[0] if cams else None)
    if ref is None:
        return [], {"method": METHOD, "reference_camera": None, "note": "no camera streams"}
    frames = {c: list(iter_jsonl(ep / "streams" / c / "frames.jsonl")) for c in cams}
    state = list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))
    leader = list(iter_jsonl(ep / "telemetry" / "leader_state.jsonl"))
    cmds = list(iter_jsonl(ep / "telemetry" / "commands.jsonl"))
    s_t, s_ok = [state_time(r) for r in state], [bool(r.get("read_ok")) for r in state]
    l_t, l_ok = [state_time(r) for r in leader], [bool(r.get("read_ok", True)) for r in leader]
    c_t = [r["t_start_ns"] for r in cmds]
    tol_s = int(sync_cfg["max_state_tolerance_ms"] * 1e6)
    tol_c = int(sync_cfg["max_command_tolerance_ms"] * 1e6)
    tol_cam = int(sync_cfg["max_camera_tolerance_ms"] * 1e6)
    cam_t = {c: [f["host_mono_ns"] for f in frames[c]] for c in cams}
    rows, stats = [], {"state_residuals_ns": [], "unmatched": {"state": 0, "command": 0, "leader": 0}}
    for fi, f in enumerate(frames.get(ref, [])):
        t = f["host_mono_ns"]
        si, sr, ss = match_nearest(t, s_t, tol_s, s_ok)
        ci, cr, cs = match_nearest(t, c_t, tol_c, None, "before")
        cni, cnr, cns = match_nearest(t, c_t, tol_c, None, "after")
        li, lr, ls = match_nearest(t, l_t, tol_s, l_ok) if leader else (None, None, "source_absent")
        row = {"ref_camera": ref, "frame_index": fi, "recv_index": f["recv_index"], "frame_number": f.get("frame_number"),
               "t_host_mono_ns": t,
               "follower_state": {"index": si, "residual_ns": sr, "status": ss if state else "source_absent"},
               "command_prev": {"index": ci, "residual_ns": cr, "status": cs if cmds else "source_absent"},
               "command_next": {"index": cni, "residual_ns": cnr, "status": cns if cmds else "source_absent"},
               "leader_state": {"index": li, "residual_ns": lr, "status": ls},
               "cameras": {}}
        for c in cams:
            if c == ref:
                continue
            ii, rr, st = match_nearest(t, cam_t[c], tol_cam)
            row["cameras"][c] = {"index": ii, "residual_ns": rr, "status": st}
        rows.append(row)
        if state:
            if ss == "matched":
                stats["state_residuals_ns"].append(sr)
            else:
                stats["unmatched"]["state"] += 1
        stats["unmatched"]["command"] += int(bool(cmds) and cs != "matched")
        stats["unmatched"]["leader"] += int(bool(leader) and ls != "matched")
    res = np.abs(np.array(stats["state_residuals_ns"], dtype=float)) / 1e6 if stats["state_residuals_ns"] else np.array([])
    offsets = {}
    for c in cams:
        d = [f["host_mono_ns"] / 1e6 - f["device_timestamp_ms"] for f in frames[c] if f.get("device_timestamp_ms") is not None]
        offsets[c] = ({"median_host_minus_device_ms": float(np.median(d)), "spread_ms": float(np.ptp(d)),
                       "domain": frames[c][0].get("timestamp_domain"), "used_for_association": False}
                      if d else {"note": "no device timestamps", "used_for_association": False})
    report = {
        "method": METHOD, "reference_camera": ref, "num_reference_frames": len(rows),
        "tolerances_ms": {"state": sync_cfg["max_state_tolerance_ms"], "command": sync_cfg["max_command_tolerance_ms"],
                          "camera": sync_cfg["max_camera_tolerance_ms"]},
        "interpolation": "none (originals only; no interpolated values stored)",
        "hardware_synchronisation": False,
        "unmatched": stats["unmatched"],
        "state_residual_abs_ms": ({"mean": float(res.mean()), "p95": float(np.percentile(res, 95)), "max": float(res.max())}
                                  if res.size else None),
        "device_clock_offset_estimates": offsets,
        "notes": "Residual = sample time - frame receipt time. Camera receipt time includes unknown capture latency.",
    }
    return rows, report
