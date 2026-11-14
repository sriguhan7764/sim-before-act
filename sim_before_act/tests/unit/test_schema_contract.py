import json
import math

import numpy as np
import pytest

from sim_before_act.config import ConfigError, load_config
from sim_before_act.schemas.record import RecordError, ReplayRecord


# ---- record schema ------------------------------------------------------------------------------------------
def test_record_loads_and_keeps_missing_fields_none(rec):
    r = rec()
    assert r.state.velocities is None and r.action.policy_confidence is None and r.outcome.observed_success is None


def test_record_rejects_wrong_schema_version(rec_dict):
    d = rec_dict(); d["schema_version"] = "other/9"
    with pytest.raises(RecordError, match="schema_version"):
        ReplayRecord.from_dict(d)


def test_record_reports_all_missing_fields(rec_dict):
    d = rec_dict(); del d["state"]["positions"]; del d["action"]["chunk"]
    with pytest.raises(RecordError) as e:
        ReplayRecord.from_dict(d)
    assert any("state.positions" in p for p in e.value.problems) and any("action.chunk" in p for p in e.value.problems)


# ---- action contract ----------------------------------------------------------------------------------------
def _chunk(n=5, v=0.0):
    return np.full((n, 6), v, dtype=float)


def test_valid_chunk_unchanged(contract):
    a = _chunk(); a[:, 5] = 50
    arr, errs = contract.validate_chunk(a)
    assert errs == [] and np.array_equal(arr, a)


@pytest.mark.parametrize("bad", [np.zeros((5, 5)), np.zeros(6), np.zeros((0, 6)), "abc", [[1, 2], [3]]])
def test_malformed_shapes_rejected(contract, bad):
    arr, errs = contract.validate_chunk(bad)
    assert errs and (arr is None or errs)


@pytest.mark.parametrize("v", [np.nan, np.inf, -np.inf])
def test_non_finite_rejected(contract, v):
    a = _chunk(); a[2, 3] = v
    arr, errs = contract.validate_chunk(a)
    assert arr is None and any("non-finite" in e for e in errs)


def test_out_of_range_never_clipped(contract):
    a = _chunk(); a[1, 1] = -103.3        # below the simulator shoulder_lift limit (-100)
    arr, errs = contract.validate_chunk(a)
    assert any("outside limits" in e for e in errs)
    assert a[1, 1] == -103.3              # input untouched


def test_gripper_range_is_0_100(contract):
    a = _chunk(); a[0, 5] = 101
    assert any("gripper" in e for e in contract.validate_chunk(a)[1])


def test_metadata_mismatches(contract):
    a = _chunk()
    _, errs = contract.validate_chunk(a, joint_names=list(reversed(contract.joint_names)), units=["radians"] * 6,
                                      representation="delta", control_hz=10)
    assert len(errs) == 4


def test_speed_check_reported_only_when_configured(contract):
    assert contract.describe()["speed_check"].startswith("NOT performed")


def test_unit_conversion_arm_degrees_to_radians_and_gripper_mapping(contract):
    v = np.array([90.0, -45.0, 0.0, 10.0, 180.0, 0.0])
    r = contract.to_sim_rad(v)
    assert np.allclose(r[:5], np.deg2rad(v[:5]))
    lo, hi = contract.gripper_range_rad
    assert math.isclose(r[5], lo)
    v2 = v.copy(); v2[5] = 100
    assert math.isclose(contract.to_sim_rad(v2)[5], hi)


def test_action_id_is_content_bound(contract):
    a = _chunk(); b = a.copy(); b[3, 2] += 1e-9
    assert contract.action_id(a) == contract.action_id(a.copy())
    assert contract.action_id(a) != contract.action_id(b)


# ---- config -------------------------------------------------------------------------------------------------
def test_config_rejects_nonstudy_budget(make_cfg):
    with pytest.raises(ConfigError, match="study budgets"):
        make_cfg(**{"verification.n_rollouts": 7})


def test_config_requires_perturbation_provenance(tmp_path, cfg_path):
    import yaml
    raw = yaml.safe_load(cfg_path.read_text()); raw["twin"]["scene_xml"] = str(load_config(cfg_path).scene_xml())
    del raw["perturbations"]["variables"]["object_xy_m"]["source"]
    p = tmp_path / "c.yaml"; p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="source"):
        load_config(p)
