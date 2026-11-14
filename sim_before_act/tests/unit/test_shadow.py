"""Shadow-evaluation workflow tests. Every trial here is SYNTHETIC (explicitly labelled); nothing touches hardware."""
import json

import pytest

from sim_before_act.shadow.workflow import (ShadowError, annotate_trial, load_annotations, load_prediction, load_trials,
                                            match_trial, record_prediction)

SYN = "SYNTHETIC integration-test trial: not a physical result"


def kw(trial="t001", origin="synthetic_integration_test", decision="WOULD_EXECUTE", horizon="segment", steps=10, trained=False):
    return dict(
        trial_id=trial, data_origin=origin, provenance_label=SYN if origin.startswith("synthetic") else None,
        initial_state={"state_values": [0.0] * 6, "state_sha256": "a" * 64, "state_timestamp": 123, "clock_domain": "host_monotonic_ns"},
        action={"segment_id": "seg1", "action_id": "act1", "action_sha256": "b" * 64, "chunk_length": 100},
        policy={"kind": "fixture_stub", "checkpoint_id": "none", "trained_for_task": trained},
        verification={"config_hash": "c" * 64, "status": "COMPLETED", "decision": decision, "n_rollouts": 5, "base_seed": 1,
                      "counts": {"task_failures": 1}, "threshold_calibrated": False, "simulated_failure_fraction": 0.2,
                      "threshold": 0.3},
        execution_horizon={"type": horizon, "steps": steps})


def test_prediction_persisted_once_with_timestamp_and_hash(tmp_path):
    p = record_prediction(tmp_path, clock_wall_ns=lambda: 1000, **kw())
    pred = load_prediction(tmp_path, "t001")
    assert pred["predicted_at_wall_ns"] == 1000 and pred["verification"]["config_hash"] == "c" * 64
    with pytest.raises(ShadowError, match="immutable"):
        record_prediction(tmp_path, **kw())
    p.chmod(0o644)
    d = json.loads(p.read_text())
    d["verification"]["decision"] = "REJECT"
    p.write_text(json.dumps(d))
    with pytest.raises(ShadowError, match="hash mismatch"):
        load_prediction(tmp_path, "t001")


def test_decision_vocabulary_is_would_execute_only(tmp_path):
    with pytest.raises(ShadowError, match="WOULD_EXECUTE"):
        record_prediction(tmp_path, **kw(decision="EXECUTE"))


def test_synthetic_must_be_labelled_and_smoke_policy_cannot_be_physical(tmp_path):
    k = kw()
    k["provenance_label"] = "looks real"
    with pytest.raises(ShadowError, match="SYNTHETIC"):
        record_prediction(tmp_path, **k)
    with pytest.raises(ShadowError, match="trained_for_task"):
        record_prediction(tmp_path, **kw("p1", origin="physical_trial", trained=False))
    record_prediction(tmp_path, **kw("p2", origin="physical_trial", trained=True))


def test_horizon_cannot_exceed_chunk_and_trial_id_is_safe(tmp_path):
    with pytest.raises(ShadowError, match="longer"):
        record_prediction(tmp_path, **kw(steps=101))
    with pytest.raises(ShadowError, match="trial_id"):
        record_prediction(tmp_path, **kw("../evil"))


def test_annotation_validation(tmp_path):
    record_prediction(tmp_path, clock_wall_ns=lambda: 10, **kw())
    c = dict(annotated_by="op", clock_wall_ns=lambda: 20)
    with pytest.raises(ShadowError, match="outcome_horizon"):
        annotate_trial(tmp_path, "t001", execution="executed", observed_outcome="success", **c)
    with pytest.raises(ShadowError, match="label_source"):
        annotate_trial(tmp_path, "t001", execution="executed", observed_outcome="failure", outcome_horizon="segment", **c)
    with pytest.raises(ShadowError, match="abort_reason"):
        annotate_trial(tmp_path, "t001", execution="executed", observed_outcome="aborted", **c)
    with pytest.raises(ShadowError, match="cannot have an observed outcome"):
        annotate_trial(tmp_path, "t001", execution="not_executed", observed_outcome="success", **c)
    with pytest.raises(ShadowError, match="precedes"):
        annotate_trial(tmp_path, "t001", execution="not_executed", annotated_by="op", clock_wall_ns=lambda: 5)
    assert load_annotations(tmp_path, "t001") == []


def test_annotations_append_and_latest_wins_without_touching_prediction(tmp_path):
    record_prediction(tmp_path, clock_wall_ns=lambda: 10, **kw())
    before = (tmp_path / "t001" / "prediction.json").read_bytes()
    c = dict(annotated_by="op", clock_wall_ns=lambda: 20)
    annotate_trial(tmp_path, "t001", execution="unknown", **c)
    annotate_trial(tmp_path, "t001", execution="executed", observed_outcome="success", outcome_horizon="segment",
                   label_source="operator_visual", executed_command_ref={"action_sha256": "b" * 64},
                   deviations_from_verified_action=[], replanning_or_temporal_aggregation=False, **c)
    assert [a["version"] for a in load_annotations(tmp_path, "t001")] == [1, 2]
    t = load_trials(tmp_path)[0]
    assert t["observed_outcome"] == "success" and t["comparison_exact"] and t["segment_label"] == "success"
    assert (tmp_path / "t001" / "prediction.json").read_bytes() == before


def _executed(tmp_path, **over):
    record_prediction(tmp_path, clock_wall_ns=lambda: 10, **kw())
    a = dict(execution="executed", observed_outcome="success", outcome_horizon="segment", label_source="operator_visual",
             executed_command_ref={"action_sha256": "b" * 64}, deviations_from_verified_action=[],
             replanning_or_temporal_aggregation=False, annotated_by="op", clock_wall_ns=lambda: 20)
    a.update(over)
    annotate_trial(tmp_path, "t001", **a)
    return load_trials(tmp_path)[0]


def test_full_task_outcome_never_labels_a_segment(tmp_path):
    t = _executed(tmp_path, outcome_horizon="full_task")
    assert t["segment_label"] is None and "short segment" in t["excluded_reason"]


@pytest.mark.parametrize("over,needle", [
    ({"replanning_or_temporal_aggregation": True}, "not exact"),
    ({"replanning_or_temporal_aggregation": None}, "not assessed"),
    ({"deviations_from_verified_action": ["operator paused 2 s"]}, "deviations"),
    ({"deviations_from_verified_action": None}, "not assessed"),
    ({"executed_command_ref": None}, "no executed-command reference"),
    ({"executed_command_ref": {"action_sha256": "d" * 64}}, "differs"),
])
def test_comparison_flagged_not_exact(tmp_path, over, needle):
    t = _executed(tmp_path, **over)
    assert not t["comparison_exact"] and any(needle in r for r in t["not_exact_reasons"])


def test_unannotated_abort_and_unknown_are_excluded_not_labelled(tmp_path):
    record_prediction(tmp_path, clock_wall_ns=lambda: 10, **kw("a"))
    record_prediction(tmp_path, clock_wall_ns=lambda: 10, **kw("b"))
    record_prediction(tmp_path, clock_wall_ns=lambda: 10, **kw("c"))
    annotate_trial(tmp_path, "b", execution="executed", observed_outcome="aborted", abort_reason="operator stop", annotated_by="o",
                   clock_wall_ns=lambda: 20)
    annotate_trial(tmp_path, "c", execution="executed", observed_outcome="unknown", annotated_by="o", clock_wall_ns=lambda: 20)
    by = {t["trial_id"]: t for t in load_trials(tmp_path)}
    assert "no operator annotation" in by["a"]["excluded_reason"]
    assert "aborted" in by["b"]["excluded_reason"] and by["b"]["segment_label"] is None
    assert "unknown" in by["c"]["excluded_reason"] and by["c"]["segment_label"] is None


def test_shadow_workflow_runs_without_importing_any_runtime_or_serial_modules(tmp_path):
    import subprocess
    import sys
    code = ("import sys; import sim_before_act.shadow.workflow as w; "
            "bad=[m for m in sys.modules if m.split('.')[0] in ('serial','pyrealsense2','lerobot','torch','cv2') "
            "or m.startswith('sim_before_act.runtime')]; assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": str(__import__('pathlib').Path(__file__).resolve().parents[2] / "src")})
    assert r.returncode == 0, r.stderr
