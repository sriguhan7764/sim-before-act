"""Milestone E: shared action contract + action-segment file validator (consumer side). All data here is SYNTHETIC / invented."""
import copy
import json

import numpy as np
import pytest

from sim_before_act.robot.action_contract import ActionContract
from sim_before_act.robot.segment import (ActionSegment, SegmentInvalid, actions_sha256, build_fixture_segment, load_action_segment,
                                          save_segment, to_sim_targets, validate_for_sim, validate_segment, write_fixture_segment)


def _actions(n=20):
    a = np.tile([0.0, -20.0, 30.0, 20.0, 0.0, 30.0], (n, 1))
    a[:, 0] += np.linspace(0.0, 5.0, n)
    return a


def _seg(contract, **kw):
    return ActionSegment(build_fixture_segment(_actions(), contract=contract, **kw), None)


def test_contract_id_chunk_duration_and_relative_rejected(base_cfg, twin, contract):
    assert len(contract.contract_id) == 64
    same = ActionContract.build(base_cfg.section("contract"), twin.joint_limits_deg, twin.gripper_range_rad)
    assert same.contract_id == contract.contract_id and same.describe()["contract_id"] == contract.contract_id
    other = ActionContract.build({**base_cfg.section("contract"), "control_hz": contract.control_hz / 2}, twin.joint_limits_deg,
                                 twin.gripper_range_rad)
    assert other.contract_id != contract.contract_id
    assert contract.chunk_duration_s(30) == pytest.approx(30 / contract.control_hz)
    assert contract.describe()["reference"] == "follower joint-position target space"
    with pytest.raises(ValueError, match="relative|delta"):
        ActionContract.build({**base_cfg.section("contract"), "representation": "relative_joint_position_deltas"},
                             twin.joint_limits_deg, twin.gripper_range_rad)


def test_valid_segment_and_provenance(contract):
    arr, problems = validate_segment(_seg(contract), contract)
    assert problems == [] and arr.shape == (20, 6) and arr.dtype == np.float64
    np.testing.assert_array_equal(arr, _actions())
    vs = validate_for_sim(_seg(contract), contract)
    assert vs.action_sha256 == actions_sha256(_actions()) and vs.contract_id == contract.contract_id
    prov = vs.provenance()
    assert prov["reported_as_trained"] is False and "NOT A TRAINED POLICY" in prov["policy_label"]
    assert prov["data_origin"] == "synthetic_fixture" and "SYNTHETIC" in prov["provenance_label"]
    assert any("untrained_policy" in f for f in vs.flags) and vs.exact_comparison
    assert not vs.actions.flags.writeable
    np.testing.assert_allclose(to_sim_targets(vs, contract), contract.to_sim_rad(_actions()))


def test_tamper_is_rejected_not_repaired(contract):
    raw = build_fixture_segment(_actions(), contract=contract)
    raw["actions"][3][1] += 0.5
    arr, problems = validate_segment(ActionSegment(raw, None), contract)
    assert arr is None and any("action_sha256 mismatch" in p for p in problems)
    with pytest.raises(SegmentInvalid):
        validate_for_sim(ActionSegment(raw, None), contract)


@pytest.mark.parametrize("edit,needle", [
    (lambda c: c["units"].__setitem__(0, "radians"), "unit"),
    (lambda c: c["joint_names"].reverse(), "joint"),
    (lambda c: c.__setitem__("control_hz", 99.0), "control_hz"),
    (lambda c: c.__setitem__("representation", "relative_joint_position_deltas"), "relative"),
    (lambda c: c.__setitem__("reference", "something else"), "reference"),
])
def test_contract_mismatches_rejected(contract, edit, needle):
    raw = build_fixture_segment(_actions(), contract=contract)
    edit(raw["contract"])
    arr, problems = validate_segment(ActionSegment(raw, None), contract)
    assert arr is None and any(needle in p.lower() for p in problems), problems


def test_nan_and_over_limit_are_invalid_and_untouched(contract):
    bad = _actions()
    bad[4, 2] = np.nan
    arr, problems = validate_segment(ActionSegment(build_fixture_segment(bad, contract=contract), None), contract)
    assert arr is None and problems
    over = _actions()
    over[5, 2] = 150.0                              # elbow far beyond the simulator limit
    seg = ActionSegment(build_fixture_segment(over, contract=contract), None)
    arr, problems = validate_segment(seg, contract)
    assert arr is None and any("limit" in p.lower() for p in problems)
    assert seg.raw["actions"][5][2] == 150.0         # not clipped / modified


def test_chunk_fields_flags_and_horizon(contract):
    seg = _seg(contract, replanning=True)
    arr, problems = validate_segment(seg, contract)
    assert problems == [] and arr is not None
    vs = validate_for_sim(seg, contract)
    assert not vs.exact_comparison and any("comparison not exact" in f for f in vs.flags)
    assert validate_for_sim(_seg(contract, temporal_ensembling=True), contract).exact_comparison is False
    raw = build_fixture_segment(_actions(), contract=contract)
    raw["chunk"]["execution_horizon_steps"] = 21
    assert any("execution_horizon_steps" in p for p in validate_segment(ActionSegment(raw, None), contract)[1])
    prefix = validate_for_sim(_seg(contract, execution_horizon_steps=8), contract)
    assert prefix.executed_actions.shape == (8, 6) and prefix.actions.shape == (20, 6)


def test_untrained_cannot_be_reported_as_trained(contract):
    raw = build_fixture_segment(_actions(), contract=contract)
    raw["policy"]["kind"], raw["data_origin"] = "trained", "recorded_physical"      # trained_for_task stays false
    arr, problems = validate_segment(ActionSegment(raw, None), contract)
    assert arr is None and any("cannot be reported as trained" in p for p in problems)
    raw2 = build_fixture_segment(_actions(), contract=contract)
    raw2["policy"]["kind"] = "smoke_random_init_few_steps"
    vs = validate_for_sim(ActionSegment(raw2, None), contract)
    assert vs.provenance()["reported_as_trained"] is False


def test_hash_stable_across_save_load_and_input_not_mutated(contract, tmp_path):
    a = _actions()
    before = a.copy()
    raw = write_fixture_segment(tmp_path / "s.json", a, contract=contract, created_wall_ns=1)
    np.testing.assert_array_equal(a, before)
    seg = load_action_segment(tmp_path / "s.json")
    snapshot = copy.deepcopy(seg.raw)
    arr, problems = validate_segment(seg, contract)
    assert problems == [] and seg.raw == snapshot
    assert actions_sha256(arr) == raw["action_sha256"] == seg.raw["action_sha256"]
    save_segment(tmp_path / "s2.json", seg.raw)
    assert load_action_segment(tmp_path / "s2.json").raw["action_sha256"] == raw["action_sha256"]
    assert json.loads((tmp_path / "s.json").read_text())["policy"]["trained_for_task"] is False
