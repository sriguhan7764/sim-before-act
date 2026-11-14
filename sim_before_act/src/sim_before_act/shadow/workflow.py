"""Shadow evaluation: prediction-first records bound to a trial, and operator annotations of what actually happened.

OFFLINE software only. Nothing here imports a robot adapter, opens a device, or launches execution: no function accepts an
executor. A prediction is made BEFORE the outcome is known, is written once (exclusive, read-only) and carries the persisted
prediction timestamp. What physically happened is recorded later by the operator as an append-only annotation; it never rewrites
the prediction.

Honesty rules enforced here:
* "EXECUTE" offline means WOULD_EXECUTE only (nothing was sent anywhere).
* A full-task outcome never labels a short action segment; only an operator-supplied ``segment`` horizon does.
* If replanning / temporal aggregation / deviations / a different executed command are present or NOT ASSESSED, the original
  chunk comparison is flagged as not exact.
* Synthetic integration trials are marked and can never be counted as physical trials.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from ..util import hash_obj, write_json_exclusive

PRED_SCHEMA = "shadow-prediction-v0.1"
ANN_SCHEMA = "shadow-annotation-v0.1"
ORIGINS = ("synthetic_integration_test", "physical_trial")
DECISIONS = ("WOULD_EXECUTE", "REJECT")
EXECUTION = ("executed", "not_executed", "unknown")
OBSERVED = ("success", "failure", "aborted", "unknown")
LABEL_SOURCES = ("operator_visual", "operator_measured")
TRIAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class ShadowError(ValueError):
    pass


def _need(d: dict, keys, where: str) -> None:
    miss = [k for k in keys if k not in d or d[k] is None]
    if miss:
        raise ShadowError(f"{where}: missing {miss}")


def prediction_body_hash(pred: dict) -> str:
    return hash_obj({k: v for k, v in pred.items() if k != "prediction_sha256"})


def record_prediction(root: str | Path, *, trial_id: str, data_origin: str, initial_state: dict, action: dict, policy: dict,
                      verification: dict, execution_horizon: dict, timings: dict | None = None, hardware: dict | None = None,
                      provenance_label: str | None = None, clock_wall_ns: Callable[[], int] = time.time_ns) -> Path:
    """Persist a prediction for ``trial_id``. The timestamp is taken HERE, not by the caller. Never overwrites.

    initial_state: state_values, state_sha256, state_timestamp, clock_domain (+ optional objects with provenance, may be unknown).
    action: segment_id, action_id, action_sha256, chunk_length.
    policy: kind, trained_for_task, checkpoint_id (+ checkpoint_sha256).
    verification: see prediction_from_objects().
    execution_horizon: {type: 'segment'|'full_task', steps}: what the prediction is ABOUT.
    """
    if not TRIAL_ID.match(trial_id or ""):
        raise ShadowError(f"trial_id {trial_id!r} must match {TRIAL_ID.pattern}")
    if data_origin not in ORIGINS:
        raise ShadowError(f"data_origin must be one of {ORIGINS}")
    if data_origin == "synthetic_integration_test" and not (provenance_label or "").startswith("SYNTHETIC"):
        raise ShadowError("synthetic trials need a provenance_label starting with 'SYNTHETIC'")
    _need(initial_state, ("state_values", "state_sha256", "state_timestamp", "clock_domain"), "initial_state")
    _need(action, ("segment_id", "action_id", "action_sha256", "chunk_length"), "action")
    _need(policy, ("kind", "checkpoint_id", "trained_for_task"), "policy")
    _need(verification, ("config_hash", "status", "decision", "n_rollouts", "base_seed", "counts", "threshold_calibrated"),
          "verification")
    if verification["decision"] not in DECISIONS:
        raise ShadowError(f"offline decision must be one of {DECISIONS} (EXECUTE means WOULD_EXECUTE only)")
    _need(execution_horizon, ("type", "steps"), "execution_horizon")
    if execution_horizon["type"] not in ("segment", "full_task"):
        raise ShadowError("execution_horizon.type must be 'segment' or 'full_task'")
    if execution_horizon["steps"] > action["chunk_length"]:
        raise ShadowError("execution horizon longer than the action chunk")
    if data_origin == "physical_trial" and policy["trained_for_task"] is not True:
        raise ShadowError("a physical trial prediction needs trained_for_task=true (smoke/fixture policies are synthetic only)")
    pred = {
        "schema": PRED_SCHEMA, "trial_id": trial_id, "data_origin": data_origin,
        "provenance_label": provenance_label or "physical shadow-evaluation trial prediction",
        "initial_state": initial_state, "action": action, "policy": policy, "verification": verification,
        "execution_horizon": execution_horizon, "timings": timings or {}, "hardware": hardware or {},
        "predicted_at_wall_ns": int(clock_wall_ns()),
        "note": "prediction made before the outcome was known; persisted once; WOULD_EXECUTE is not a command to execute",
    }
    pred["prediction_sha256"] = prediction_body_hash(pred)
    path = Path(root) / trial_id / "prediction.json"
    try:
        write_json_exclusive(path, pred)
    except FileExistsError as e:
        raise ShadowError(f"a prediction for trial {trial_id!r} already exists; it is immutable") from e
    return path


def verification_block(verification_result, decision) -> dict:
    """Build the ``verification`` block from a VerificationResult + gate Decision (duck-typed)."""
    est = verification_result.estimate
    counts = {} if est is None else {"task_failures": est.task_failures, "safety_violations": est.safety_violations,
                                     "invalid": est.simulator_errors + est.invalid_init, "timeouts": est.timeouts,
                                     "unknown": est.unknown, "valid": est.valid, "attempted": est.attempted}
    return {"config_hash": verification_result.verification_config_hash, "status": verification_result.status,
            "decision": "WOULD_EXECUTE" if decision.decision == "EXECUTE" else "REJECT",
            "reason_codes": list(decision.reason_codes), "n_rollouts": est.requested if est else 0,
            "base_seed": verification_result.seeds.get("base_seed"),
            "simulated_failure_fraction": verification_result.simulated_failure_fraction,
            "interval": verification_result.interval, "counts": counts, "threshold": decision.risk_threshold,
            "threshold_calibrated": decision.threshold_calibrated,
            "horizon_claim": verification_result.horizon.get("claim")}


def load_prediction(root: str | Path, trial_id: str) -> dict:
    pred = json.loads((Path(root) / trial_id / "prediction.json").read_text())
    if pred.get("prediction_sha256") != prediction_body_hash(pred):
        raise ShadowError(f"{trial_id}: prediction.json hash mismatch (edited after persistence)")
    return pred


def annotate_trial(root: str | Path, trial_id: str, *, execution: str, annotated_by: str, observed_outcome: str | None = None,
                   outcome_horizon: str | None = None, label_source: str | None = None, abort_reason: str | None = None,
                   executed_command_ref: dict | None = None, deviations_from_verified_action: list[str] | None = None,
                   replanning_or_temporal_aggregation: bool | None = None, notes: str = "",
                   clock_wall_ns: Callable[[], int] = time.time_ns) -> Path:
    """Append an operator annotation (v001, v002, ...). Fields not assessed stay None and keep the comparison 'not exact'."""
    pred = load_prediction(root, trial_id)
    if not annotated_by:
        raise ShadowError("annotated_by (operator name) is required")
    if execution not in EXECUTION:
        raise ShadowError(f"execution must be one of {EXECUTION}")
    if execution == "executed":
        if observed_outcome not in OBSERVED:
            raise ShadowError(f"an executed trial needs observed_outcome in {OBSERVED}")
        if observed_outcome in ("success", "failure"):
            if outcome_horizon not in ("full_task", "segment"):
                raise ShadowError("success/failure needs outcome_horizon 'full_task' or 'segment' (what the label refers to)")
            if label_source not in LABEL_SOURCES:
                raise ShadowError(f"success/failure needs label_source in {LABEL_SOURCES}")
        else:
            if label_source or outcome_horizon:
                raise ShadowError(f"outcome '{observed_outcome}' cannot carry a label source / horizon")
            if observed_outcome == "aborted" and not abort_reason:
                raise ShadowError("aborted needs abort_reason")
    else:
        if observed_outcome not in (None, "unknown"):
            raise ShadowError(f"execution '{execution}' cannot have an observed outcome (got {observed_outcome!r})")
        if label_source or outcome_horizon:
            raise ShadowError("a trial that was not executed (or unknown) carries no outcome label/horizon")
        observed_outcome = None if execution == "not_executed" else "unknown"
    if executed_command_ref is not None and not isinstance(executed_command_ref, dict):
        raise ShadowError("executed_command_ref must be a dict (e.g. episode id, command range, action_sha256)")
    now = int(clock_wall_ns())
    if now < pred["predicted_at_wall_ns"]:
        raise ShadowError("annotation timestamp precedes the persisted prediction timestamp")
    d = Path(root) / trial_id / "annotations"
    v = len(list(d.glob("v*.json"))) + 1 if d.is_dir() else 1
    ann = {"schema": ANN_SCHEMA, "trial_id": trial_id, "version": v, "prediction_sha256": pred["prediction_sha256"],
           "execution": execution, "observed_outcome": observed_outcome, "outcome_horizon": outcome_horizon,
           "label_source": label_source, "abort_reason": abort_reason, "executed_command_ref": executed_command_ref,
           "deviations_from_verified_action": deviations_from_verified_action,
           "replanning_or_temporal_aggregation": replanning_or_temporal_aggregation,
           "annotated_by": annotated_by, "annotated_wall_ns": now, "notes": notes}
    path = d / f"v{v:03d}.json"
    write_json_exclusive(path, ann)
    return path


def load_annotations(root: str | Path, trial_id: str) -> list[dict]:
    d = Path(root) / trial_id / "annotations"
    return [json.loads(p.read_text()) for p in sorted(d.glob("v*.json"))] if d.is_dir() else []


def match_trial(pred: dict, ann: dict | None) -> dict:
    """Join a prediction with the LATEST annotation; states what the physical result can and cannot be compared to."""
    out = {"trial_id": pred["trial_id"], "physical": pred["data_origin"] == "physical_trial", "data_origin": pred["data_origin"],
           "provenance_label": pred["provenance_label"], "predicted_decision": pred["verification"]["decision"],
           "predicted_failure_fraction": pred["verification"].get("simulated_failure_fraction"),
           "verification_status": pred["verification"]["status"], "prediction_horizon": pred["execution_horizon"]["type"],
           "annotated": ann is not None, "execution": None, "observed_outcome": None, "label_horizon": None,
           "segment_label": None, "comparison_exact": False, "not_exact_reasons": [], "excluded_reason": None}
    if ann is None:
        out["excluded_reason"] = "no operator annotation yet (outcome unknown)"
        return out
    out.update(execution=ann["execution"], observed_outcome=ann["observed_outcome"], label_horizon=ann["outcome_horizon"])
    if ann["execution"] != "executed":
        out["excluded_reason"] = f"execution: {ann['execution']}"
        return out
    if ann["observed_outcome"] in ("aborted", "unknown"):
        out["excluded_reason"] = f"observed outcome {ann['observed_outcome']}" + (
            f" ({ann['abort_reason']})" if ann["abort_reason"] else "")
        return out
    why: list[str] = []
    if ann["replanning_or_temporal_aggregation"] is None:
        why.append("replanning/temporal aggregation not assessed")
    elif ann["replanning_or_temporal_aggregation"]:
        why.append("replanning or temporal aggregation changed the executed commands: original chunk comparison is not exact")
    devs = ann["deviations_from_verified_action"]
    if devs is None:
        why.append("deviations from the verified action not assessed")
    elif devs:
        why.append(f"operator-reported deviations: {devs}")
    ref = ann["executed_command_ref"]
    if not ref:
        why.append("no executed-command reference recorded")
    elif ref.get("action_sha256") and ref["action_sha256"] != pred["action"]["action_sha256"]:
        why.append("executed command hash differs from the verified action hash")
    out["not_exact_reasons"] = why
    out["comparison_exact"] = not why
    if ann["outcome_horizon"] == "segment" and pred["execution_horizon"]["type"] == "segment":
        out["segment_label"] = ann["observed_outcome"]
    elif ann["outcome_horizon"] == "full_task":
        out["excluded_reason"] = ("full-task outcome cannot label a short segment prediction "
                                  "(needs an operator segment-horizon label)")
    else:
        out["excluded_reason"] = "label horizon does not match the prediction horizon"
    return out


def load_trials(root: str | Path) -> list[dict]:
    """All trials under ``root`` as match_trial() dicts plus the fields the reports need (latest annotation wins)."""
    out = []
    for d in sorted(p for p in Path(root).glob("*") if (p / "prediction.json").exists()):
        pred = load_prediction(root, d.name)
        anns = load_annotations(root, d.name)
        m = match_trial(pred, anns[-1] if anns else None)
        v = pred["verification"]
        m.update(annotation_versions=len(anns), timings=pred.get("timings", {}), n_rollouts=v.get("n_rollouts"),
                 threshold=v.get("threshold"), threshold_calibrated=v.get("threshold_calibrated"),
                 policy={k: pred["policy"].get(k) for k in ("kind", "checkpoint_id", "trained_for_task")},
                 hardware=pred.get("hardware", {}), counts=v.get("counts", {}), interval=v.get("interval"),
                 predicted_at_wall_ns=pred["predicted_at_wall_ns"])
        out.append(m)
    return out
