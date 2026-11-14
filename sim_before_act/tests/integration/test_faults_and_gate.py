"""Fault and gate tests: stale state, simulator failure, timeouts, missing poses, and approval binding."""
import copy

import numpy as np
import pytest

import sim_before_act.twin.mujoco_twin as mt
from sim_before_act.runtime.gate import GuardedExecutor
from sim_before_act.runtime.offline import run_attempt
from sim_before_act.twin.mujoco_twin import RolloutResult
from sim_before_act.verification.verifier import Verifier


class Spy:
    def __init__(self):
        self.sent = []

    def send_chunk(self, c):
        self.sent.append(c)


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _verify(stack, rec, **ov):
    cfg, (twin, contract, verifier, gate) = stack(**ov)
    return cfg, verifier.verify(rec), gate


# ---- state faults -------------------------------------------------------------------------------------------
def test_stale_state(stack, rec):
    _, v, gate = _verify(stack, rec("safe_reach", state__timestamp_s=9.0))      # age 1.05 s > 0.5
    assert v.status == "INVALID_STATE" and any("too old" in p for p in v.problems)
    assert gate.decide(v, 9.0, [0] * 6).decision == "REJECT"


def test_state_after_issue_time_invalid(stack, rec):
    _, v, _ = _verify(stack, rec("safe_reach", state__timestamp_s=11.0))
    assert v.status == "INVALID_STATE"


def test_missing_object_pose(stack, rec):
    _, v, gate = _verify(stack, rec("safe_reach", objects=[]))
    assert v.status == "INVALID_STATE" and any("pose missing" in p for p in v.problems)
    d = gate.decide(v, 10.0, [0] * 6)
    assert d.decision == "REJECT" and v.simulated_failure_fraction is None


def test_object_in_wrong_frame(stack, rec, rec_dict):
    d = rec_dict(); d["objects"][0]["frame"] = "mat"
    from sim_before_act.schemas.record import ReplayRecord
    _, v, _ = _verify(stack, ReplayRecord.from_dict(d))
    assert v.status == "INVALID_STATE"


def test_missing_task_predicate_is_invalid_input(stack, rec):
    _, v, _ = _verify(stack, rec("safe_reach", task__target_xyz_m=None))
    assert v.status == "INVALID_INPUT"


def test_nonfinite_action_is_invalid_input_and_rejected(stack, rec, rec_dict):
    d = rec_dict(); d["action"]["chunk"][3][2] = float("nan")
    from sim_before_act.schemas.record import ReplayRecord
    _, v, gate = _verify(stack, ReplayRecord.from_dict(d))
    assert v.status == "INVALID_INPUT" and gate.decide(v, 10.0, [0] * 6).decision == "REJECT"


def test_missing_object_pose_with_nan_values(stack, rec_dict):
    d = rec_dict(); d["objects"][0]["position_m"][0] = float("nan")
    from sim_before_act.schemas.record import ReplayRecord
    _, v, _ = _verify(stack, ReplayRecord.from_dict(d))
    assert v.status == "INVALID_STATE"


# ---- simulator faults ---------------------------------------------------------------------------------------
def test_simulator_exception_is_simulation_error_not_task_failure(stack, rec, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("injected engine failure")
    monkeypatch.setattr(mt.mujoco, "mj_step", boom)
    _, v, gate = _verify(stack, rec("safe_reach"))
    assert v.status == "SIMULATION_ERROR" and v.simulated_failure_fraction is None
    assert v.estimate.simulator_errors == 10 and v.estimate.task_failures == 0 and v.estimate.valid == 0
    assert gate.decide(v, 10.0, [0] * 6).decision == "REJECT"


def test_nan_state_detected_as_simulator_error(stack, rec, monkeypatch):
    real = mt.mujoco.mj_step

    def nan_step(m, d):
        real(m, d)
        d.qpos[0] = float("nan")
    monkeypatch.setattr(mt.mujoco, "mj_step", nan_step)
    _, v, _ = _verify(stack, rec("safe_reach"))
    assert v.status == "SIMULATION_ERROR" and "non-finite" in v.rollouts[0]["detail"]


def test_partial_simulator_errors_are_excluded_not_counted_as_failures(stack, rec, monkeypatch):
    cfg, (twin, contract, verifier, gate) = stack()
    real = twin.run_rollout

    def flaky(c, init, objs, arr, smp, hz, tgt, tol, dl=None):
        if smp.index % 2:
            return RolloutResult(index=smp.index, outcome="simulator_error", detail="injected")
        return real(c, init, objs, arr, smp, hz, tgt, tol, dl)
    monkeypatch.setattr(twin, "run_rollout", flaky)
    v = verifier.verify(rec("safe_reach"))
    assert v.status == "COMPLETED" and (v.estimate.valid, v.estimate.simulator_errors) == (5, 5)
    assert v.simulated_failure_fraction == 0.0 and v.estimate.task_failures == 0


def test_too_few_valid_rollouts_is_insufficient(stack, rec, monkeypatch):
    cfg, (twin, contract, verifier, gate) = stack()
    real = twin.run_rollout
    monkeypatch.setattr(twin, "run_rollout", lambda c, i, o, a, s, hz, t, tol, dl=None:
                        real(c, i, o, a, s, hz, t, tol, dl) if s.index == 0 else RolloutResult(index=s.index, outcome="invalid_init"))
    v = verifier.verify(rec("safe_reach"))
    assert v.status == "INSUFFICIENT_VALID_ROLLOUTS" and v.simulated_failure_fraction is None
    assert gate.decide(v, 10.0, [0] * 6).decision == "REJECT"


def test_verification_timeout(stack, rec):
    _, v, gate = _verify(stack, rec("safe_reach"), **{"verification.verification_timeout_s": 0.0})
    assert v.status == "TIMEOUT" and v.simulated_failure_fraction is None
    assert gate.decide(v, 10.0, [0] * 6).decision == "REJECT"


def test_per_rollout_timeout_reported_as_timeout(stack, rec):
    _, v, _ = _verify(stack, rec("safe_reach"), **{"rollout.rollout_timeout_s": 0.0})
    assert v.status == "TIMEOUT" and v.estimate.timeouts == 1 and v.estimate.task_failures == 0


# ---- approval binding ---------------------------------------------------------------------------------------
@pytest.fixture
def approved(stack, rec, contract):
    cfg, (twin, c, verifier, _) = stack()
    clk = Clock(100.0)
    from sim_before_act.runtime.gate import Gate
    gate = Gate(cfg.section("gate"), cfg.section("state")["permitted_state_deviation_deg"], clock=clk)
    r = rec("safe_reach")
    v = verifier.verify(r)
    d = gate.decide(v, r.state.timestamp_s, r.state.positions)
    assert d.decision == "EXECUTE"
    spy = Spy()
    return d, v, np.array(r.action.chunk), r, spy, GuardedExecutor(c, spy, clock=clk), clk


def test_exact_action_executes(approved):
    d, v, chunk, r, spy, ex, _ = approved
    out = ex.execute(d, chunk, r.state.positions, v.verification_config_hash)
    assert out.executed and len(spy.sent) == 1


def test_action_modified_after_verification_is_refused(approved):
    d, v, chunk, r, spy, ex, _ = approved
    mod = chunk.copy(); mod[-1, 0] += 0.5          # a small valid change is a NEW candidate
    out = ex.execute(d, mod, r.state.positions, v.verification_config_hash)
    assert not out.executed and "ACTION_MODIFIED_AFTER_VERIFICATION" in out.reason_codes and spy.sent == []


def test_approval_for_another_action_is_refused(approved, rec, stack):
    d, v, chunk, r, spy, ex, _ = approved
    other = np.array(rec("far_target").action.chunk)
    other = other + 0.25
    out = ex.execute(d, other, r.state.positions)
    assert not out.executed and "ACTION_MODIFIED_AFTER_VERIFICATION" in out.reason_codes


def test_expired_approval_refused(approved):
    d, v, chunk, r, spy, ex, clk = approved
    clk.t += 5.0
    out = ex.execute(d, chunk, r.state.positions)
    assert not out.executed and "APPROVAL_EXPIRED" in out.reason_codes and spy.sent == []


def test_state_divergence_refused(approved):
    d, v, chunk, r, spy, ex, _ = approved
    cur = np.array(r.state.positions); cur[0] += 10.0
    out = ex.execute(d, chunk, cur)
    assert not out.executed and "STATE_DIVERGED_FROM_APPROVED" in out.reason_codes


def test_verification_config_mismatch_refused(approved):
    d, v, chunk, r, spy, ex, _ = approved
    assert "VERIFICATION_CONFIG_MISMATCH" in ex.execute(d, chunk, r.state.positions, "other").reason_codes


def test_reject_decision_cannot_execute_even_with_forged_chunk(stack, rec, contract):
    cfg, (twin, c, verifier, gate) = stack()
    r = rec("floor_contact")
    d = gate.decide(verifier.verify(r), r.state.timestamp_s, r.state.positions)
    spy = Spy()
    out = GuardedExecutor(c, spy).execute(d, np.array(r.action.chunk), r.state.positions)
    assert not out.executed and spy.sent == []
