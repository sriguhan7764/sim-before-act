"""Recover an interrupted (``.inprogress``) episode. The result is flagged as interrupted - never 'complete'."""
from __future__ import annotations

import json
from pathlib import Path

from .storage import STATUS_INTERRUPTED, finalize_directory
from .util import jsonl_line, read_json, wall_ns, write_json_atomic


def _repair_jsonl(path: Path, file_exists) -> dict:
    """Keep only intact lines (and, for frame indexes, only lines whose image files exist)."""
    good, bad = [], 0
    with open(path, "rb") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if file_exists is not None and not file_exists(rec):
                    raise ValueError("missing file")
                good.append(rec)
            except Exception:
                bad += 1
    if bad:
        path.rename(path.with_name(path.name + ".corrupt"))
        with open(path, "wb") as f:
            for r in good:
                f.write(jsonl_line(r))
    return {"kept": len(good), "discarded": bad}


def recover(inprog: str | Path, sync_cfg: dict | None = None) -> Path:
    inprog = Path(inprog)
    if not inprog.name.endswith(".inprogress"):
        raise ValueError("not an .inprogress directory")
    final = inprog.with_name(inprog.name[: -len(".inprogress")])
    if final.exists():
        raise FileExistsError(final)
    try:
        meta = read_json(inprog / "episode.json")
    except Exception:
        meta = {"episode_id": final.name, "streams": {}, "quality_flags": {}}
    report = {}
    for c in sorted(p.name for p in (inprog / "streams").glob("*") if p.is_dir()):
        fp = inprog / "streams" / c / "frames.jsonl"
        if fp.exists():
            base = inprog / "streams" / c

            def ok(rec, base=base):
                return all((base / rec[k]).exists() for k in ("color_file", "depth_native_file", "depth_aligned_file") if rec.get(k))

            report[c] = _repair_jsonl(fp, ok)
        meta.setdefault("streams", {}).setdefault(c, {"role": "unknown", "depth": False})
    for t in ("follower_state", "leader_state", "commands"):
        fp = inprog / "telemetry" / f"{t}.jsonl"
        if fp.exists():
            report[t] = _repair_jsonl(fp, None)
    for tmp in inprog.rglob("*.tmp"):
        tmp.unlink()
    meta.update(status=STATUS_INTERRUPTED, outcome="aborted", end_wall_ns=meta.get("end_wall_ns") or wall_ns(),
                aborted_reason="recording interrupted; recovered after the fact")
    meta.setdefault("quality_flags", {}).update(recovered=True, partial=True, recovery_report=report)
    meta.setdefault("episode_id", final.name)
    write_json_atomic(inprog / "episode.json", meta)
    sync_cfg = sync_cfg or (meta.get("config_snapshot") or {}).get("sync") or {
        "max_state_tolerance_ms": 20.0, "max_command_tolerance_ms": 50.0, "max_camera_tolerance_ms": 20.0}
    return finalize_directory(inprog, final, sync_cfg, meta.get("reference_camera"))


def find_interrupted(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("*.inprogress"))
