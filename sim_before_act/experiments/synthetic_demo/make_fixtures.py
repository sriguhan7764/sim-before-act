"""Generate the three SYNTHETIC demo records (pipeline validation only).

The reach target of the 'safe' record is the NOMINAL simulator end-effector position of that same action, so the
'safe' record passes by construction. That is circular and proves only that the pipeline runs; it says nothing about
the physical SO-101. Records carry synthetic=true and a label that the CLI prints.

Run:  /home/apic3/miniforge3/envs/guhan_c/bin/python experiments/synthetic_demo/make_fixtures.py
Refuses to overwrite existing files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sim_before_act.config import load_config  # noqa: E402
from sim_before_act.robot.action_contract import JOINT_NAMES, REPRESENTATION, UNITS  # noqa: E402
from sim_before_act.runtime.offline import build_stack  # noqa: E402
from sim_before_act.schemas.record import RECORD_SCHEMA_VERSION  # noqa: E402
from sim_before_act.util import write_json_exclusive  # noqa: E402

START = [0.0, -30.0, 30.0, 20.0, 0.0, 50.0]
SAFE_GOAL = [10.0, -20.0, 40.0, 20.0, 0.0, 50.0]
FLOOR_GOAL = [0.0, 28.6, 45.8, 28.6, 0.0, 50.0]
LABEL = "SYNTHETIC FIXTURE - pipeline validation only (generic MuJoCo scene, uncalibrated, targets derived from the nominal sim)"


def chunk(goal, n=30):
    a = np.linspace(START, goal, n + 1)[1:]
    return a.tolist()


def record(rid, goal, target, tol, task_name, instr, outcome=None):
    return {
        "schema_version": RECORD_SCHEMA_VERSION, "record_id": rid, "episode_id": f"synthetic_{rid}", "session_id": "synthetic_session",
        "chunk_id": "c0", "split": None, "source_dataset": {"type": "synthetic_fixture", "path": None},
        "timestamp_wall_iso": None, "synthetic": True, "synthetic_label": LABEL,
        "state": {"joint_names": list(JOINT_NAMES), "positions": START, "units": list(UNITS), "timestamp_s": 10.0,
                  "clock_domain": "synthetic_s", "velocities": None},
        "objects": [{"name": "cube", "position_m": [0.25, 0.09, 0.02], "quat_wxyz": [1, 0, 0, 0], "frame": "robot_base",
                     "source": "synthetic_fixture", "position_sigma_m": None}],
        "action": {"chunk": chunk(goal), "joint_names": list(JOINT_NAMES), "units": list(UNITS), "representation": REPRESENTATION,
                   "control_hz": 30.0, "issue_timestamp_s": 10.05, "policy_id": None, "policy_confidence": None},
        "task": {"name": task_name, "instruction": instr, "target_xyz_m": target, "tolerance_m": tol,
                 "target_source": "synthetic: derived from nominal simulator end-effector position (circular)"},
        "outcome": outcome or {}, "calibration_ref": None, "image_refs": [],
        "limitations": ["Synthetic record: not a physical observation."],
    }


def main(out_dir: Path) -> None:
    cfg = load_config(ROOT / "configs" / "offline_demo.yaml")
    twin, contract, _, _ = build_stack(cfg)
    from sim_before_act.twin.perturbation import PerturbationSampler
    nominal = PerturbationSampler({"enabled": False}, 0, 1).sample(0, 30)
    objs = {"cube": {"position_m": [0.25, 0.09, 0.02], "quat_wxyz": [1, 0, 0, 0]}}
    ee = {}
    for k, goal in (("safe", SAFE_GOAL), ("floor", FLOOR_GOAL)):
        r = twin.run_rollout(contract, np.array(START), objs, np.array(chunk(goal)), nominal, 30.0, np.zeros(3), 1.0)
        assert r.valid, r
        ee[k] = r.ee_final_m
        print(f"nominal {k}: ee_final={np.round(r.ee_final_m, 4).tolist()} safety={r.safety_violations}")
    safe_t = [round(x, 4) for x in ee["safe"]]
    far_t = [round(safe_t[0] + 0.15, 4), safe_t[1], safe_t[2]]
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "safe_reach": record("synthetic_safe_reach", SAFE_GOAL, safe_t, 0.03, "reach", "synthetic: move the end-effector to the target"),
        "floor_contact": record("synthetic_floor_contact", FLOOR_GOAL, [round(x, 4) for x in ee["floor"]], 0.03, "reach",
                                "synthetic: known constraint violation (arm driven into the floor)"),
        "far_target": record("synthetic_far_target", SAFE_GOAL, far_t, 0.03, "reach",
                             "synthetic: target deliberately unreachable by this chunk (task failure)"),
    }
    for name, rec in specs.items():
        write_json_exclusive(out_dir / f"{name}.json", rec, readonly=False)
        print("wrote", out_dir / f"{name}.json")


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
