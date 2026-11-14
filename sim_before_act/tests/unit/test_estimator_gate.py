import math

import numpy as np
import pytest

from sim_before_act.runtime.gate import Gate
from sim_before_act.twin.mujoco_twin import RolloutResult
from sim_before_act.verification.estimator import estimate, wilson_interval
from sim_before_act.verification.verifier import VerificationResult


def R(i, outcome="ok", task=False, safety=()):
    return RolloutResult(index=i, outcome=outcome, task_failure=task if outcome == "ok" else None, safety_violations=list(safety))


# ---- risk calculation ---------------------------------------------------------------------------------------
def test_fraction_is_failed_over_valid():
    e = estimate([R(0), R(1, task=True), R(2, safety=["robot_floor_contact"]), R(3, task=True, safety=["x"])], 4)
    assert (e.valid, e.failed, e.task_failures, e.safety_violations) == (4, 3, 2, 2)
    assert e.simulated_failure_fraction == 0.75


def test_zero_valid_rollouts_is_unavailable_not_zero():
    e = estimate([R(0, "simulator_error"), R(1, "invalid_init")], 2)
    assert e.valid == 0 and e.simulated_failure_fraction is None and e.interval is None


def test_partial_failures_exclude_invalid_from_denominator_and_are_not_task_failures():
    e = estimate([R(0), R(1, task=True), R(2, "simulator_error"), R(3, "timeout"), R(4, "invalid_init"), R(5, "unknown")], 6)
    assert e.valid == 2 and e.simulated_failure_fraction == 0.5
    assert (e.simulator_errors, e.timeouts, e.invalid_init, e.unknown, e.task_failures) == (1, 1, 1, 1, 1)


def test_wilson_known_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and math.isclose(hi, 0.2775, abs_tol=1e-3)
    lo, hi = wilson_interval(5, 10)
    assert math.isclose(lo, 0.2366, abs_tol=1e-3) and math.isclose(hi, 0.7634, abs_tol=1e-3)
    assert wilson_interval(0, 0) is None


# ---- gate ---------------------------------------------------------------------------------------------------
def V(status="COMPLETED", frac=0.0, **kw):
    d = dict(status=status, problems=[], action_id="a" * 8, state_hash="s" * 8, verification_config_hash="h",
             simulated_failure_fraction=frac, estimate=None, interval=None, seeds={}, perturbation={}, rollouts=[], timings={},
             twin={}, horizon={}, state_age_s=0.0, action_validated=True, state_validated=True, independent_limits_ok=True)
    d.update(kw)
    return VerificationResult(**d)


def G(threshold=0.3, **kw):
    cfg = {"risk_threshold": threshold, "threshold_calibrated": False, "approval_validity_s": 1.0, **kw}
    return Gate(cfg, 3.0, clock=lambda: 100.0)


STATE = [0, 0, 0, 0, 0, 0]


def test_threshold_boundary_equality_rejects():
    assert G().decide(V(frac=0.3), 1.0, STATE).decision == "REJECT"
    assert G().decide(V(frac=0.3 - 1e-9), 1.0, STATE).decision == "EXECUTE"
    assert "RISK_AT_OR_ABOVE_THRESHOLD" in G().decide(V(frac=0.31), 1.0, STATE).reason_codes


def test_unconfigured_threshold_fails_closed():
    d = G(threshold=None).decide(V(frac=0.0), 1.0, STATE)
    assert d.decision == "REJECT" and "THRESHOLD_NOT_CONFIGURED" in d.reason_codes


@pytest.mark.parametrize("status", ["INVALID_INPUT", "INVALID_STATE", "SIMULATION_ERROR", "TIMEOUT", "INSUFFICIENT_VALID_ROLLOUTS"])
def test_failed_verification_always_rejects_even_with_low_fraction(status):
    d = G().decide(V(status=status, frac=0.0), 1.0, STATE)
    assert d.decision == "REJECT" and f"VERIFICATION_{status}" in d.reason_codes


def test_unavailable_risk_never_approves():
    assert G().decide(V(frac=None), 1.0, STATE).decision == "REJECT"


@pytest.mark.parametrize("flag", ["action_validated", "state_validated", "independent_limits_ok"])
def test_each_precondition_is_required(flag):
    assert G().decide(V(frac=0.0, **{flag: False}), 1.0, STATE).decision == "REJECT"


def test_bad_threshold_config_rejected():
    with pytest.raises(ValueError):
        G(threshold=1.5)


def test_uncalibrated_threshold_is_disclosed_in_reason():
    d = G().decide(V(frac=0.0), 1.0, STATE)
    assert "UNCALIBRATED" in d.reason and d.threshold_calibrated is False
