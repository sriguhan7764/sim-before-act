import numpy as np
import pytest

from sim_before_act.twin.perturbation import PerturbationSampler

SPEC = {"enabled": True, "variables": {
    "robot_state_error_deg": {"dist": "normal", "sigma": 0.5, "bound": 1.0, "units": "deg", "source": "t", "calibrated": False},
    "object_xy_m": {"dist": "normal", "sigma": 0.01, "bound": 0.005, "units": "m", "source": "t", "calibrated": False},
    "object_yaw_deg": {"dist": "uniform", "low": -5, "high": 5, "units": "deg", "source": "t", "calibrated": False},
    "friction_scale": {"dist": "uniform", "low": 0.8, "high": 1.2, "units": "x", "source": "t", "calibrated": False},
    "control_noise_deg": {"dist": "normal", "sigma": 0.2, "bound": 0.4, "units": "deg", "source": "t", "calibrated": False}}}


def _flat(s):
    return np.concatenate([s.state_error_deg, s.object_xy_m, [s.object_yaw_deg, s.friction_scale], s.control_noise_deg.ravel()])


def test_same_seed_same_samples():
    a, b = PerturbationSampler(SPEC, 7, 10), PerturbationSampler(SPEC, 7, 10)
    assert all(np.array_equal(_flat(a.sample(i, 6)), _flat(b.sample(i, 6))) for i in range(10))


def test_different_seed_differs():
    assert not np.array_equal(_flat(PerturbationSampler(SPEC, 1, 3).sample(0, 4)), _flat(PerturbationSampler(SPEC, 2, 3).sample(0, 4)))


def test_sample_independent_of_call_order_and_budget():
    a = PerturbationSampler(SPEC, 7, 10)
    fwd = [_flat(a.sample(i, 6)) for i in range(5)]
    b = PerturbationSampler(SPEC, 7, 50)           # larger budget -> same first samples (paired budgets)
    rev = [_flat(b.sample(i, 6)) for i in reversed(range(5))][::-1]
    assert all(np.array_equal(x, y) for x, y in zip(fwd, rev))


def test_truncation_bounds_respected():
    s = PerturbationSampler(SPEC, 3, 200)
    samples = [s.sample(i, 3) for i in range(200)]
    assert max(np.abs(x.state_error_deg).max() for x in samples) <= 1.0
    assert max(np.abs(x.object_xy_m).max() for x in samples) <= 0.005
    assert all(0.8 <= x.friction_scale <= 1.2 for x in samples)


def test_disabled_returns_nominal_sample():
    s = PerturbationSampler({"enabled": False}, 1, 2).sample(0, 4)
    assert s.nominal and not s.state_error_deg.any() and s.friction_scale == 1.0 and not s.control_noise_deg.any()


def test_unsupported_variable_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        PerturbationSampler({"variables": {"camera_pixel_noise": {}}}, 1, 1)


def test_describe_flags_uncalibrated():
    d = PerturbationSampler(SPEC, 1, 1).describe()
    assert d["all_calibrated"] is False and any("pixel" in x for x in d["unsupported_by_design"])
