"""Integrity validation of master episodes and exports (read-only)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .util import iter_jsonl, read_json, sha256_file


def _monotonic(vals: list, name: str, errs: list) -> None:
    if any(b < a for a, b in zip(vals, vals[1:])):
        errs.append(f"{name}: timestamps not monotonic")


def _duplicates(vals: list, name: str, warns: list) -> None:
    n = sum(1 for a, b in zip(vals, vals[1:]) if b == a)
    if n:
        warns.append(f"{name}: {n} duplicate consecutive timestamps")


def validate_master(ep: str | Path, check_hashes: bool = True) -> dict:
    ep = Path(ep)
    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {}
    if ep.name.endswith(".inprogress"):
        errors.append("directory is an unfinalised (in-progress) episode; run `recover`")
        return {"ok": False, "errors": errors, "warnings": warnings, "info": info}
    try:
        man = read_json(ep / "manifest.json")
        meta = read_json(ep / "episode.json")
    except Exception as e:
        return {"ok": False, "errors": [f"cannot read manifest/episode.json: {e}"], "warnings": [], "info": {}}
    if check_hashes:
        for rel, rec in man["files"].items():
            p = ep / rel
            if not p.exists():
                errors.append(f"missing file {rel}")
            elif p.stat().st_size != rec["bytes"] or sha256_file(p) != rec["sha256"]:
                errors.append(f"hash mismatch {rel}")
        on_disk = {p.relative_to(ep).as_posix() for p in ep.rglob("*") if p.is_file() and p.name != "manifest.json"}
        extra = on_disk - set(man["files"])
        if extra:
            errors.append(f"files not in manifest: {sorted(extra)[:5]}")
    info["status"] = meta.get("status")
    if meta.get("status") not in ("complete",):
        warnings.append(f"episode status is '{meta.get('status')}'")
    if meta.get("quality_flags", {}).get("mock_data"):
        warnings.append("MOCK (synthetic) data")
    info["streams"] = {}
    for c in sorted(meta.get("streams", {})):
        fr = list(iter_jsonl(ep / "streams" / c / "frames.jsonl"))
        _monotonic([f["host_mono_ns"] for f in fr], f"camera {c}", errors)
        _duplicates([f["host_mono_ns"] for f in fr], f"camera {c}", warnings)
        _dd = [f["device_timestamp_ms"] for f in fr if f.get("device_timestamp_ms") is not None]
        _monotonic(_dd, f"camera {c} device", warnings)  # device clock anomalies are reported, not fatal
        _duplicates(_dd, f"camera {c} device", warnings)
        idx = [f["recv_index"] for f in fr]
        gaps = (max(idx) + 1 - len(idx)) if idx else 0
        fn = [f["frame_number"] for f in fr if f.get("frame_number") is not None]
        dev_gaps = sum(b - a - 1 for a, b in zip(fn, fn[1:]) if b - a > 1)
        info["streams"][c] = {"frames": len(fr), "missing_recv_indices": gaps, "device_frame_number_gaps": dev_gaps}
        if gaps:
            warnings.append(f"camera {c}: {gaps} frames dropped by the recorder")
        if dev_gaps:
            warnings.append(f"camera {c}: {dev_gaps} device frame numbers missing (dropped upstream)")
        for f in fr[:: max(1, len(fr) // 5)]:  # spot-check decodability and depth dtype
            for key in ("depth_native_file", "depth_aligned_file"):
                if f.get(key):
                    a = cv2.imread(str(ep / "streams" / c / f[key]), cv2.IMREAD_UNCHANGED)
                    if a is None or a.dtype != np.uint16:
                        errors.append(f"{c}/{f[key]}: not a readable uint16 image")
            if f.get("color_file") and cv2.imread(str(ep / "streams" / c / f["color_file"])) is None:
                errors.append(f"{c}/{f['color_file']}: unreadable")
    for k in ("follower_state", "leader_state", "commands"):
        recs = list(iter_jsonl(ep / "telemetry" / f"{k}.jsonl"))
        info[k] = len(recs)
        _monotonic([r["t_start_ns"] for r in recs], k, errors)
        _duplicates([r["t_start_ns"] for r in recs], k, warnings)
        seq = [r["seq"] for r in recs]
        if seq and seq != list(range(seq[0], seq[0] + len(seq))):
            warnings.append(f"{k}: sequence gaps (dropped records)")
        bad = sum(1 for r in recs if r.get("read_ok") is False)
        if bad:
            warnings.append(f"{k}: {bad} failed reads flagged read_ok=false")
    sync = read_json(ep / "sync" / "sync_report.json") if (ep / "sync" / "sync_report.json").exists() else None
    if sync:
        info["sync"] = {"unmatched": sync["unmatched"], "state_residual_abs_ms": sync["state_residual_abs_ms"],
                        "hardware_synchronisation": sync["hardware_synchronisation"]}
        if any(sync["unmatched"].values()):
            warnings.append(f"unmatched associations: {sync['unmatched']}")
    for stream, n in meta.get("counts", {}).get("dropped", {}).items():
        if n:
            warnings.append(f"{stream}: {n} items dropped by the recorder (queue overflow / write error)")
    for m in meta.get("quality_flags", {}).get("missing_modalities", []):
        warnings.append(f"missing modality: {m}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info}


def validate_export(path: str | Path) -> dict:
    p = Path(path)
    errors = []
    man = read_json(p / "manifest.json")
    for rel, rec in man["files"].items():
        f = p / rel
        if not f.exists() or sha256_file(f) != rec["sha256"]:
            errors.append(f"bad or missing {rel}")
    if not man.get("complete", True):
        errors.append(f"export incomplete, missing: {man.get('missing_required_signals')}")
    return {"ok": not errors, "errors": errors, "warnings": [], "info": {"profile": man["profile"], "parent": man["parent"]}}


def validate(path: str | Path) -> dict:
    p = Path(path)
    man = read_json(p / "manifest.json") if (p / "manifest.json").exists() else {}
    return validate_export(p) if man.get("kind") == "export" else validate_master(p)
