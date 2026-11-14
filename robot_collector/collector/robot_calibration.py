"""Read-only robot range capture -> project-local calibration JSON (LeRobot-compatible keys).

The operator moves each joint by hand through its range while this tool only READS Present_Position (and the stored
Homing_Offset). Nothing is written to the motors, torque is untouched, and no file outside the project is modified.
Output: <out_dir>/<name>.json (refuses to overwrite, then made read-only) + <name>.meta.json (provenance).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from .adapters.feetech_readonly import FeetechReadOnly
from .util import sha256_file, wall_ns, write_json_atomic

FULL_TURN = (0, 4095)  # LeRobot convention for the continuous wrist_roll joint


def capture_ranges(cfg: dict, role: str, seconds: float, out_dir: str | Path, name: str, progress=print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite a calibration file")
    src = FeetechReadOnly(cfg, role)
    src.open()
    try:
        offsets = {j["name"]: src.read_homing_offset(j["id"]) for j in cfg["joints"]}
        # Raw ticks wrap at 4096. The motors' existing homing offsets may put the wrap point inside a joint's travel, so
        # track the UNWRAPPED excursion from the first sample instead of raw min/max.
        first, prev, u, lo, hi, n = {}, {}, {}, {}, {}, 0
        t_end, last = time.monotonic() + seconds, 0.0
        while time.monotonic() < t_end:
            for rec in src.poll():
                if not rec["read_ok"]:
                    continue
                n += 1
                for k, v in rec["raw"].items():
                    if k not in first:
                        first[k], prev[k], u[k], lo[k], hi[k] = v, v, 0, 0, 0
                        continue
                    u[k] += ((v - prev[k] + 2048) % 4096) - 2048
                    prev[k] = v
                    lo[k], hi[k] = min(lo[k], u[k]), max(hi[k], u[k])
            if time.monotonic() - last > 1.0:
                last = time.monotonic()
                progress("t-%4.0fs  " % (t_end - last) + "  ".join(f"{k}:{first[k]+lo[k]}..{first[k]+hi[k]}" for k in lo))
    finally:
        src.close()
    if n < 50:
        raise RuntimeError("too few valid reads to build a calibration")
    MIN_SPAN = 300  # ticks (~26 deg): anything smaller means the joint was not moved through its range
    unmoved = [k for k in lo if k != "wrist_roll" and hi[k] - lo[k] < MIN_SPAN]
    if any(hi[k] - lo[k] >= 4000 for k in lo if k != "wrist_roll"):
        raise RuntimeError("a joint travelled >= ~360 deg; this convention cannot represent that; nothing saved")
    if unmoved:
        raise RuntimeError(f"joints not moved through a usable range (span < {MIN_SPAN} ticks): {unmoved}; nothing saved")
    cal = {}
    for j in cfg["joints"]:
        k = j["name"]
        rmin, rmax = (FULL_TURN if k == "wrist_roll" else (first[k] + lo[k], first[k] + hi[k]))
        cal[k] = {"id": j["id"], "drive_mode": 0, "homing_offset": offsets[k], "range_min": rmin, "range_max": rmax}
    write_json_atomic(path, cal)
    os.chmod(path, 0o444)
    meta = {"name": name, "role": role, "port": cfg["port"], "created_wall_ns": wall_ns(), "samples": n,
            "method": "operator moved joints by hand; unwrapped min/max of Present_Position (read-only); range_min/max may be <0 or >4095 when the motor wrap point lies inside the travel. wrist_roll set to full turn 0..4095 "
                      "like LeRobot. Homing_Offset read from the motors, never written.",
            "valid_only_while": "motor homing offsets still equal the stored values (re-read and compared by validate)",
            "sha256": sha256_file(path)}
    write_json_atomic(out / f"{name}.meta.json", meta)
    os.chmod(out / f"{name}.meta.json", 0o444)
    return path
