"""Read-only importer: LeRobot v3.0 dataset -> ReplayRecord.

Takes the measured state at one frame and the next H recorded (executed) actions. It NEVER writes into the dataset.
Object poses, the task predicate target, outcome labels and calibration refs are NOT in a LeRobot recording; they may be
supplied only by an operator sidecar JSON. Without them the verifier returns INVALID_INPUT / INVALID_STATE, which is the
honest state of the real data (no object poses, no outcome labels, no validated camera-to-robot transform).

Assumptions flagged on the record: values are LeRobot `use_degrees=true` (degrees, gripper 0..100) - info.json does not
state units; the clock is the dataset's episode-relative timestamp; the action 'issue time' is the state frame time, so
state age is not measurable (reported as 0 with a limitation).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..robot.action_contract import JOINT_NAMES, REPRESENTATION, UNITS
from ..schemas.record import RECORD_SCHEMA_VERSION


def import_lerobot(dataset_dir: str | Path, episode_index: int = 0, start_frame: int = 0, horizon: int = 30,
                   sidecar: str | Path | None = None) -> dict:
    import pyarrow.parquet as pq

    root = Path(dataset_dir)
    info = json.loads((root / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"unsupported LeRobot codebase_version {info.get('codebase_version')!r} (expected v3.x)")
    names = info["features"]["action"]["names"]
    if [n.removesuffix(".pos") for n in names] != list(JOINT_NAMES) or info["features"]["observation.state"]["names"] != names:
        raise ValueError(f"unexpected joint names {names}")
    table = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    idx = [i for i, e in enumerate(table["episode_index"]) if e == episode_index]
    if not idx:
        raise ValueError(f"episode {episode_index} not found")
    fi = [table["frame_index"][i] for i in idx]
    if start_frame not in fi or start_frame + horizon > len(idx):
        raise ValueError(f"start_frame {start_frame} + horizon {horizon} outside episode of {len(idx)} frames")
    base = idx[0] + fi.index(start_frame)
    state = [float(x) for x in table["observation.state"][base]]
    chunk = [[float(x) for x in table["action"][base + k]] for k in range(horizon)]
    ts = float(table["timestamp"][base])
    sc = json.loads(Path(sidecar).read_text()) if sidecar else {}
    rid = f"lerobot_{root.name}_ep{episode_index}_f{start_frame}"
    lim = ["Units ASSUMED degrees (arm) and 0..100 (gripper) from LeRobot use_degrees default; info.json does not state units.",
           "Action 'issue time' equals the state frame time; state age is not measurable from this dataset.",
           "Actions are the recorded teleoperation LEADER-derived commands that were executed; not a policy output."]
    if not sc:
        lim.append("No operator sidecar: object poses, task target and outcome label are absent (not fabricated).")
    return {
        "schema_version": RECORD_SCHEMA_VERSION, "record_id": rid, "episode_id": f"{root.name}_ep{episode_index}",
        "session_id": root.name, "chunk_id": f"f{start_frame}_h{horizon}", "split": None,
        "source_dataset": {"type": "lerobot_v3.0", "path": str(root), "fps": info.get("fps"), "robot_type": info.get("robot_type")},
        "timestamp_wall_iso": None, "synthetic": False, "synthetic_label": None,
        "state": {"joint_names": list(JOINT_NAMES), "positions": state, "units": list(UNITS), "timestamp_s": ts,
                  "clock_domain": "lerobot_episode_relative_s", "velocities": None},
        "objects": sc.get("objects", []),
        "action": {"chunk": chunk, "joint_names": list(JOINT_NAMES), "units": list(UNITS), "representation": REPRESENTATION,
                   "control_hz": float(info["fps"]), "issue_timestamp_s": ts, "policy_id": None, "policy_confidence": None},
        "task": sc.get("task", {"name": "unlabelled", "instruction": "", "target_xyz_m": None, "tolerance_m": None,
                                "target_source": None}),
        "outcome": sc.get("outcome", {}), "calibration_ref": sc.get("calibration_ref"),
        "image_refs": [f"videos/{k}" for k in ("observation.images.front", "observation.images.front_depth", "observation.images.wrist")],
        "limitations": lim,
    }
