"""Trace builder/writer: one JSON per verification attempt (including rejected attempts and verification failures).

Traces are independent of the decision: logging is not a decision outcome. Files are written exclusively (never
overwritten) and made read-only. Non-finite floats are stored as null, never as 0.
"""
from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
import uuid
from importlib import metadata
from pathlib import Path

from ..robot.action_contract import ActionContract
from ..runtime.gate import Decision, ExecutionOutcome
from ..schemas.record import ReplayRecord
from ..util import to_jsonable, write_json_exclusive

TRACE_SCHEMA_VERSION = "sba-trace/0.1"

STATIC_LIMITATIONS = [
    "Offline replay only: no policy was run and no robot command was sent.",
    "The MuJoCo scene is a generic SO-101 model; fidelity to the physical robot/calibration is NOT validated.",
    "Perturbation distributions are placeholders unless config marks them calibrated; they are not measured.",
    "The simulated failure fraction is not a calibrated probability of physical failure; the interval covers Monte-Carlo error only.",
    "Gripper 0-100 -> MJCF joint mapping is an unvalidated assumption.",
    "Prediction horizon is the action chunk plus settle time, not full-task success.",
    "The decision threshold is only meaningful if threshold_calibrated is true.",
    "The Wilson interval describes sampling uncertainty under the chosen rollout model only; it excludes simulator/calibration mismatch and does not demonstrate physical safety. The gate does not use it.",
    "Timings are PRELIMINARY: single machine, no warm-up or repetition protocol, boundaries not formally defined; not an E3 measurement.",
]


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def reproducibility(cfg_hash: str, asset_hash: str | None, cwd: Path) -> dict:
    return {"git_commit": git_commit(cwd), "git_note": "null: project directory is not a git repository" if git_commit(cwd) is None else None,
            "config_hash": cfg_hash, "robot_asset_hash": asset_hash, "policy_checkpoint": None,
            "python": sys.version.split()[0], "os": platform.platform(),
            "packages": {p: _pkg(p) for p in ("numpy", "mujoco", "pyyaml", "pyarrow")},
            "determinism_note": "Fixed seeds do not guarantee bitwise-identical results across hardware/library versions."}


def build_trace(rec: ReplayRecord, contract: ActionContract, ver, dec: Decision, exe: ExecutionOutcome, cfg, timings: dict,
                project_dir: Path) -> dict:
    o = rec.outcome
    obs_failure = None if o.observed_success is None else (not o.observed_success)
    synthetic = rec.synthetic
    lim = list(STATIC_LIMITATIONS) + list(rec.limitations)
    if rec.state.velocities is None:
        lim.append("Robot velocities were not recorded; the simulator starts at rest.")
    if o.observed_success is None:
        lim.append("No observed outcome label for the recorded action; no prediction-vs-outcome comparison is possible.")
    if synthetic:
        lim.append("SYNTHETIC record: not evidence of SO-101 digital-twin fidelity or real-world predictive validity.")
    est = ver.estimate
    return to_jsonable({
        "schema_version": TRACE_SCHEMA_VERSION,
        "attempt_id": str(uuid.uuid4()),
        "created_wall_iso": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": "offline_replay",
        "identity": {"record_id": rec.record_id, "episode_id": rec.episode_id, "session_id": rec.session_id,
                     "chunk_id": rec.chunk_id, "split": rec.split, "source_dataset": rec.source_dataset,
                     "action_id": ver.action_id, "record_schema_version": rec.schema_version},
        "synthetic": synthetic, "synthetic_label": rec.synthetic_label,
        "observation_and_state": {
            "instruction": rec.task.instruction, "image_refs": rec.image_refs, "calibration_ref": rec.calibration_ref,
            "robot_state": {"joint_names": rec.state.joint_names, "positions": rec.state.positions, "units": rec.state.units,
                            "velocities": rec.state.velocities, "timestamp_s": rec.state.timestamp_s,
                            "clock_domain": rec.state.clock_domain, "state_age_s": ver.state_age_s},
            "objects": [{"name": x.name, "position_m": x.position_m, "quat_wxyz": x.quat_wxyz, "frame": x.frame,
                         "source": x.source, "position_sigma_m": x.position_sigma_m} for x in rec.objects],
            "state_flags": ver.state_flags, "state_hash": ver.state_hash,
            "estimated_state_is_not_ground_truth": True},
        "action": {"chunk": rec.action.chunk, "contract": contract.describe(), "policy_id": rec.action.policy_id,
                   "policy_confidence": rec.action.policy_confidence,
                   "policy_confidence_note": "null: not exposed/meaningful; never fabricated" if rec.action.policy_confidence is None else None,
                   "task_predicate": {"target_xyz_m": rec.task.target_xyz_m, "tolerance_m": rec.task.tolerance_m,
                                      "source": rec.task.target_source}},
        "verification": {
            "status": ver.status, "problems": ver.problems, "verification_config_hash": ver.verification_config_hash,
            "requested": est.requested if est else None, "attempted": est.attempted if est else 0,
            "valid": est.valid if est else 0, "failed_valid": est.failed if est else 0,
            "counts": None if est is None else {"task_failure": est.task_failures, "safety_violation": est.safety_violations,
                                                "simulator_error": est.simulator_errors, "invalid_init": est.invalid_init,
                                                "timeout": est.timeouts, "unknown": est.unknown},
            "violation_breakdown": None if est is None else est.violation_breakdown,
            "simulated_failure_fraction": ver.simulated_failure_fraction,
            "task_failure_fraction": est.task_failure_fraction if est and ver.status == "COMPLETED" else None,
            "safety_violation_fraction": est.safety_violation_fraction if est and ver.status == "COMPLETED" else None,
            "interval": ver.interval,
            "interval_note": "Wilson interval: Monte-Carlo sampling uncertainty under the chosen rollout/perturbation model only; excludes model mismatch; not used by the gate",
            "timing_note": "preliminary: hardware, repetitions, warm-up and timing boundaries not documented; not an E3 measurement",
            "estimator": "failed_valid / valid (failed = task failure OR safety violation)",
            "seeds": ver.seeds, "perturbation": ver.perturbation, "rollouts": ver.rollouts,
            "horizon": ver.horizon, "twin": ver.twin, "timings_s": {**ver.timings, **timings}},
        "decision": {"decision": dec.decision, "reason_codes": dec.reason_codes, "reason": dec.reason,
                     "risk_threshold": dec.risk_threshold, "threshold_calibrated": dec.threshold_calibrated,
                     "decision_rule": dec.decision_rule,
                     "approval": None if dec.approval is None else {
                         "action_id": dec.approval.action_id, "state_hash": dec.approval.state_hash,
                         "state_timestamp_s": dec.approval.state_timestamp_s, "validity_s": dec.approval.expires_at - dec.approval.issued_at,
                         "permitted_state_deviation_deg": dec.approval.permitted_state_deviation_deg},
                     "hold_stop_behaviour": dec.hold_stop_behaviour},
        "execution": {"executed": exe.executed, "reason_codes": exe.reason_codes,
                      "note": "Offline milestone: no execution adapter is connected; EXECUTE means 'would execute'."},
        "physical_outcome": {
            "execution_occurred_in_this_attempt": False,
            "recorded_action_outcome": {"observed_success": o.observed_success, "observed_failure": obs_failure,
                                        "failure_type": o.failure_type, "label_source": o.label_source,
                                        "horizon": o.horizon, "applies_to": o.applies_to},
            "label_available": o.observed_success is not None,
            "note": "Labels the recorded action only. Never assigned to a different action."},
        "reproducibility": reproducibility(cfg.hash, ver.twin.get("robot_asset_hash") if ver.twin else None, project_dir),
        "limitations": lim,
    })


def write_trace(trace: dict, out_dir: Path) -> Path:
    rid = trace["identity"]["record_id"]
    p = Path(out_dir) / f"trace_{rid}_{trace['attempt_id'][:8]}.json"
    write_json_exclusive(p, trace, readonly=True)
    return p
