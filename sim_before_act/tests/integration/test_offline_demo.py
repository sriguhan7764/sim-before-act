"""Milestone J: the one-command offline demo runs end to end, is labelled synthetic, and opens NO hardware.

Hardware guard: pyserial, pyrealsense2 and cv2.VideoCapture are replaced (via PYTHONPATH, inherited by every child process the demo
spawns, including the collector CLI and the lerobot-env ACT smoke test) by shims that LOG and REFUSE any open attempt.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SHIM = Path(__file__).parent / "hw_guard_shim"
ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.simulation


def _env(log: Path) -> dict:
    return {**os.environ, "HW_GUARD_LOG": str(log), "PYTHONPATH": f"{SHIM}{os.pathsep}{ROOT / 'src'}", "PYTHONDONTWRITEBYTECODE": "1"}


def test_guard_shims_actually_catch_hardware_attempts(tmp_path):
    """The guard is only meaningful if it fires: prove it logs and refuses serial, RealSense and camera opens."""
    log = tmp_path / "hw.log"
    code = ("import serial, pyrealsense2, cv2\n"
            "for f in (lambda: serial.Serial('/dev/ttyFAKE'), lambda: pyrealsense2.pipeline(), lambda: cv2.VideoCapture(0),\n"
            "          lambda: __import__('serial.tools.list_ports', fromlist=['x']).comports()):\n"
            "    try:\n        f()\n    except Exception as e:\n        print('refused', type(e).__name__)\n")
    p = subprocess.run([sys.executable, "-c", code], env=_env(log), capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert p.stdout.count("refused") == 4
    text = log.read_text()
    assert "serial: Serial" in text and "pyrealsense2: pipeline" in text and "cv2.VideoCapture" in text and "comports" in text


def test_offline_demo_end_to_end_synthetic_and_hardware_free(tmp_path):
    log, out = tmp_path / "hw.log", tmp_path / "demo"
    p = subprocess.run([sys.executable, "-m", "sim_before_act.demo", "--out", str(out), "-n", "5", "--no-render"],
                       env=_env(log), capture_output=True, text=True, timeout=600, cwd=str(tmp_path))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert not log.exists() or log.read_text() == "", f"hardware access attempted: {log.read_text()}"
    s = json.loads((out / "demo_summary.json").read_text())
    assert s["data_origin"] == "synthetic" and s["recorded_physical_data_used"] is False and s["physically_evaluated"] is False
    assert s["hardware"] == {"serial_opened": False, "cameras_opened": False, "robot_session": False}
    assert s["decision_meaning"].startswith("WOULD_EXECUTE / WOULD_REJECT only")
    # corrected gate: unknown-only evidence must never be approved, and the summary/report/trace must agree
    ver = json.loads((out / "04_verification" / "verification.json").read_text())
    assert s["verification_decision"] == ver["decision"] == "REJECT" and s["offline_decision_label"] == "WOULD_REJECT"
    assert s["gate_status"] == ver["status"] and s["approval_scope"] == "none" and ver["simulated_failure_fraction"] is None
    assert s["fixture_verification_all_expectations_met"] is True
    fxv = json.loads((out / "05b_synthetic_verification_fixtures" / "fixture_verification.json").read_text())
    got = {c["case"]: (c["observed"]["status"], c["observed"]["decision"], c["observed"]["approval_scope"]) for c in fxv["cases"]}
    assert got["B1_hold_pose_safety_only_scope"] == ("COMPLETED", "WOULD_EXECUTE", "safety_only")
    assert got["B2_hold_pose_task_horizon_reached"][:2] == ("COMPLETED", "REJECT")
    assert got["B3_known_constraint_violation"][:2] == ("COMPLETED", "REJECT")
    assert got["B4_insufficient_evidence"] == ("INSUFFICIENT_EVIDENCE", "REJECT", "none")
    assert got["B5_all_rollouts_invalid"][:2] == ("INSUFFICIENT_VALID_ROLLOUTS", "REJECT")
    assert "NOT ACT" in fxv["label"].upper() or "Not ACT" in fxv["label"]
    assert (out / "04_verification" / "outcome_audit.json").exists()
    from sim_before_act.demo import LEROBOT_PY, LEROBOT_SRC
    if LEROBOT_PY.exists() and LEROBOT_SRC.exists():
        assert s["policy"].startswith("ACT smoke test")          # the guarded run really exercised the lerobot-env child process
    text = (out / "DEMO_REPORT.md").read_text()
    assert "SYNTHETIC" in text and "not a thesis finding" in text and "trained_for_task` is **false**" in text
    man = json.loads((out / "01_fixture_episodes" / "manifest.json").read_text())
    assert {e["role"] for e in man["entries"]} == {"synthetic_fixture"} and all(not e["training_eligible"] for e in man["entries"])
    rep = (out / "07_report" / "review_report.md").read_text()
    assert "NO physical shadow trials exist" in rep
    assert (out / "04_verification" / "verification.json").exists() and (out / "04_verification" / "rollout0_plot.png").exists()
    for lg in (out / "logs").glob("*.log"):
        assert "--allow-hardware" not in lg.read_text().split("--- stdout ---")[0]
    q = subprocess.run([sys.executable, "-m", "sim_before_act.demo", "--out", str(out)], env=_env(log), capture_output=True, text=True,
                       cwd=str(tmp_path))
    assert q.returncode != 0 and "refusing to overwrite" in (q.stdout + q.stderr)
