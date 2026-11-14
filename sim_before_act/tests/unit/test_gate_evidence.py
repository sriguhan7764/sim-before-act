"""Gate evidence semantics: a completed simulation is not a valid outcome observation. Pure logic, no simulator, no hardware.

Rollout rows are produced by the REAL ``combine_outcome`` from hand-constructed predicate values (SYNTHETIC; no physics).
Thresholds are UNCALIBRATED placeholders and are not what is being tested here.
"""
import pytest

from sim_before_act.integration.offline_pipeline import assess_rollouts
from sim_before_act.twin.scene.outcomes import combine_outcome

TASK_H = 8.0


def _preds(value, evaluable=True, violation=(), unsafe=()):
    p = {"value": value, "evaluable": evaluable}
    return {"constraints": {"invalid_items": [], "violation_items": list(violation)}, "contacts": {"unsafe_hit": list(unsafe)},
            "lift": dict(p), "inside_tray": dict(p), "release_settle": dict(p)}


def row(kind, i=0):
    if kind == "unknown":              # clean, short (segment) horizon: unmet predicates are inconclusive
        o = combine_outcome(_preds(False), 1.8, TASK_H)
    elif kind == "success":
        o = combine_outcome(_preds(True), TASK_H, TASK_H)
    elif kind == "task_failure":
        o = combine_outcome(_preds(False), TASK_H, TASK_H)
    elif kind == "violation":          # known constraint violation; the task outcome stays unknown at a segment horizon
        o = combine_outcome(_preds(False, violation=["joint_limit:elbow_flex"]), 1.8, TASK_H)
    elif kind == "invalid":
        o = combine_outcome(_preds(False), 1.8, TASK_H, invalid_reason="simulator instability")
    elif kind == "timeout":
        o = combine_outcome(_preds(False), 1.8, TASK_H, timed_out=True)
    else:
        raise AssertionError(kind)
    return {"index": i, "category": o["category"], "task_outcome": o["task_outcome"]}


def rows(**spec):
    out = []
    for k, c in spec.items():
        out += [row(k, len(out) + j) for j in range(c)]
    return out


def gate(r, n=None, required=("safety", "task"), thr=0.3, mvf=0.8):
    return assess_rollouts(r, n if n is not None else len(r), threshold=thr, min_valid_fraction=mvf, required=required)


def test_row_helper_categories_are_what_combine_outcome_says():
    assert row("unknown")["category"] == "unknown" and row("success")["category"] == "task_success"
    assert row("task_failure")["category"] == "task_failure" and row("violation")["category"] == "safety_violation"
    assert row("violation")["task_outcome"] == "unknown"
    assert row("invalid")["category"] == "invalid_rollout" and row("timeout")["category"] == "timeout"


def test_all_required_outcomes_unknown_is_insufficient_evidence_not_zero_risk():
    g = gate(rows(unknown=10))
    assert g["status"] == "INSUFFICIENT_EVIDENCE" and g["decision"] == "REJECT" and g["offline_decision_label"] == "WOULD_REJECT"
    assert g["simulated_failure_fraction"] is None and g["partial_fraction_over_determined"] is None and g["interval"] is None
    assert "task_outcome_unavailable" in g["reason_codes"] and "safety_outcome_unavailable" not in g["reason_codes"]
    assert g["criteria"]["task"]["failure_fraction"] is None and g["criteria"]["task"]["interval"] is None   # no estimate, not 0
    assert g["criteria"]["safety"]["evaluable"] == 10 and g["criteria"]["safety"]["violation_fraction"] == 0.0
    assert g["approval_scope"] == "none"


def test_all_simulator_errors():
    g = gate(rows(invalid=10))
    assert g["status"] == "INSUFFICIENT_VALID_ROLLOUTS" and g["decision"] == "REJECT" and g["simulated_failure_fraction"] is None
    assert {"safety_outcome_unavailable", "task_outcome_unavailable"} <= set(g["reason_codes"])
    assert g["criteria"]["safety"]["violation_fraction"] is None and g["denominators"]["valid"] == 0


@pytest.mark.parametrize("r, n", [([], 5), (rows(timeout=5), 5), (rows(invalid=2, timeout=3), 5)])
def test_zero_completed_rollouts(r, n):
    g = gate(r, n)
    assert g["status"] == "INSUFFICIENT_VALID_ROLLOUTS" and g["decision"] == "REJECT" and g["simulated_failure_fraction"] is None
    assert g["coverage"]["determined_over_all_rollouts"] == 0.0 and g["interval"] is None


def test_mixed_known_and_unknown_below_coverage_withholds_the_subset_fraction():
    g = gate(rows(success=6, task_failure=1, unknown=3))                 # determined 7/10 < required 8
    assert g["status"] == "INSUFFICIENT_EVIDENCE" and g["decision"] == "REJECT"
    assert g["simulated_failure_fraction"] is None and g["interval"] is None
    assert g["partial_fraction_over_determined"] == pytest.approx(1 / 7)          # only a labelled subset value
    assert g["denominators"]["determined"] == 7 and g["denominators"]["all_rollouts"] == 10
    assert g["denominators"]["used_for_fraction"].startswith("none")
    assert "task_evidence_below_required_coverage" in g["reason_codes"]


def test_mixed_with_sufficient_coverage_uses_determined_denominator_and_reports_the_rest():
    g = gate(rows(success=7, task_failure=1, unknown=2))                 # determined 8/10 meets the requirement
    assert g["status"] == "COMPLETED" and g["decision"] == "WOULD_EXECUTE" and g["approval_scope"] == "full"
    assert g["simulated_failure_fraction"] == pytest.approx(1 / 8)
    assert g["denominators"]["used_for_fraction"] == "determined" and g["denominators"]["excluded_from_fraction"]["undetermined_valid"] == 2
    assert g["coverage"]["determined_over_all_rollouts"] == pytest.approx(0.8)


def test_unknowns_are_never_zero_risk_even_when_one_rollout_is_known():
    g = gate(rows(task_failure=1, unknown=9))
    assert g["decision"] == "REJECT" and g["simulated_failure_fraction"] is None
    g2 = gate(rows(success=1, unknown=9))
    assert g2["decision"] == "REJECT" and g2["simulated_failure_fraction"] is None


def test_safety_known_but_task_unknown_preserves_the_distinction():
    r = rows(unknown=10)
    full, safety_only = gate(r), gate(r, required=("safety",))
    assert full["decision"] == "REJECT" and full["approval_scope"] == "none"
    assert safety_only["status"] == "COMPLETED" and safety_only["decision"] == "WOULD_EXECUTE"
    assert safety_only["approval_scope"] == "safety_only" and safety_only["simulated_failure_fraction"] == 0.0
    assert safety_only["criteria"]["task"]["available"] is False and safety_only["required_criteria"] == ["safety"]


def test_safety_cannot_be_dropped_from_required_criteria():
    for bad in (("task",), (), ("safety", "task", "extra")):
        with pytest.raises(ValueError):
            gate(rows(unknown=3), required=bad)


def test_known_constraint_violation_is_reported_and_task_stays_unknown():
    g = gate(rows(violation=10))
    assert g["status"] == "COMPLETED" and g["decision"] == "REJECT" and g["simulated_failure_fraction"] == 1.0
    assert "safety_violation_observed" in g["reason_codes"] and g["criteria"]["task"]["evaluable"] == 0
    g2 = gate(rows(violation=2, unknown=8))                              # violation is a known verdict but coverage is short
    assert g2["status"] == "INSUFFICIENT_EVIDENCE" and g2["decision"] == "REJECT" and "safety_violation_observed" in g2["reason_codes"]
    g3 = gate(rows(violation=3, unknown=7), required=("safety",))        # safety-only: any valid rollout is determined
    assert g3["status"] == "COMPLETED" and g3["simulated_failure_fraction"] == pytest.approx(0.3) and g3["decision"] == "REJECT"


def test_available_task_failure():
    g = gate(rows(task_failure=10))
    assert g["status"] == "COMPLETED" and g["simulated_failure_fraction"] == 1.0 and g["decision"] == "REJECT"
    assert g["criteria"]["task"]["failure"] == 10 and g["criteria"]["task"]["failure_fraction"] == 1.0
    assert "risk_at_or_above_threshold" in g["reason_codes"]


def test_all_task_success_is_the_only_full_scope_approval_path():
    g = gate(rows(success=10))
    assert g["decision"] == "WOULD_EXECUTE" and g["approval_scope"] == "full" and g["simulated_failure_fraction"] == 0.0


def test_threshold_equality_still_rejects_and_coverage_is_not_relaxed_by_n():
    assert gate(rows(success=7, task_failure=3), thr=0.3)["decision"] == "REJECT"          # 0.3 >= 0.3
    for n in (1, 5, 10):
        assert gate(rows(unknown=n))["decision"] == "REJECT"
