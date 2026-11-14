"""Simulation tests against the MuJoCo scene (generic SO-101 model; NOT validated against the physical robot).

They show the engine behaves as specified (predicates, reset, seeding). They are engineering tests, not support for any
research hypothesis, and they do not require every perturbation to increase risk.
"""
import numpy as np
import pytest

from sim_before_act.twin.perturbation import PerturbationSample, PerturbationSampler

pytestmark = pytest.mark.simulation

OBJS = {"cube": {"position_m": [0.25, 0.09, 0.02], "quat_wxyz": [1, 0, 0, 0]}}


def nominal(T=30):
    return PerturbationSampler({"enabled": False}, 0, 1).sample(0, T)


def custom(T=30, state=None, xy=None, yaw=0.0, fric=1.0, noise=None):
    z = np.zeros
    return PerturbationSample(0, 0, (0,), False, z(5) if state is None else np.asarray(state, float),
                              z(2) if xy is None else np.asarray(xy, float), yaw, fric, z((T, 5)) if noise is None else noise)


def chunk(rec):
    return np.array(rec.action.chunk)


def run(twin, contract, rec, smp=None, init=None, act=None):
    return twin.run_rollout(contract, np.array(rec.state.positions if init is None else init), OBJS,
                            chunk(rec) if act is None else act, smp or nominal(), 30.0,
                            np.array(rec.task.target_xyz_m), rec.task.tolerance_m)


def test_known_constraint_violation_detected(twin, contract, rec):
    r = run(twin, contract, rec("floor_contact"))
    assert r.valid and "robot_floor_contact" in r.safety_violations and r.failed


def test_known_non_violating_motion_passes(twin, contract, rec):
    r = run(twin, contract, rec("safe_reach"))
    assert r.valid and r.safety_violations == [] and r.task_failure is False and not r.failed
    assert r.ee_error_m < 1e-4                      # target = nominal run rounded to 0.1 mm (circular by construction)


def test_unreachable_target_is_task_failure_without_safety_violation(twin, contract, rec):
    r = run(twin, contract, rec("far_target"))
    assert r.valid and r.task_failure is True and r.safety_violations == []


def test_controlled_perturbation_shifts_endpoint_and_flips_predicate(twin, contract, rec):
    r = rec("safe_reach")
    noise = np.zeros((30, 5)); noise[:, 0] = 10.0          # constant 10 deg pan execution offset
    p = run(twin, contract, r, custom(noise=noise))
    assert p.valid and p.task_failure is True and p.ee_error_m > 0.03


def test_perturbation_not_forced_to_increase_risk(twin, contract, rec):
    r = rec("safe_reach")
    p = run(twin, contract, r, custom(fric=1.2, xy=[0.002, -0.002], yaw=3.0, state=[0.3, -0.3, 0.2, 0.1, 0.0]))
    assert p.valid and p.task_failure is False             # small perturbations should not change this outcome


def test_invalid_initialisation_cases(twin, contract, rec):
    r = rec("safe_reach")
    near_limit = list(r.state.positions); near_limit[1] = -99.0
    a = run(twin, contract, r, custom(state=[0, -5.0, 0, 0, 0]), init=near_limit)
    assert a.outcome == "invalid_init" and a.task_failure is None and not a.valid and not a.failed
    b = run(twin, contract, r, custom(xy=[1.0, 0.0]))
    assert b.outcome == "invalid_init"


def test_rollouts_are_fully_reset_between_runs(twin, contract, rec):
    safe = rec("safe_reach")
    a = run(twin, contract, safe)
    run(twin, contract, rec("floor_contact"), custom(fric=1.2))       # dirty the engine with a violent rollout
    b = run(twin, contract, safe)
    assert a.ee_final_m == b.ee_final_m and a.safety_violations == b.safety_violations
    assert np.array_equal(twin.model.geom_friction, twin._nominal_friction)          # friction restored


def test_order_independence_across_fresh_twin(twin, contract, rec, base_cfg):
    from sim_before_act.twin.mujoco_twin import MujocoTwin
    fresh = MujocoTwin(base_cfg.scene_xml(), base_cfg.section("twin"), base_cfg.section("rollout"))
    s = PerturbationSampler(base_cfg.section("perturbations"), 11, 5)
    r = rec("safe_reach")
    ours = [run(twin, contract, r, s.sample(i, 30)).ee_final_m for i in (3, 0, 4, 1, 2)]
    theirs = [run(fresh, contract, r, s.sample(i, 30)).ee_final_m for i in (0, 1, 2, 3, 4)]
    assert ours[1] == theirs[0] and ours[3] == theirs[1] and ours[4] == theirs[2] and ours[0] == theirs[3]


def test_seeded_verification_is_repeatable(stack, rec):
    _, (twin, c, verifier, g) = stack()
    a, b = verifier.verify(rec("safe_reach")), verifier.verify(rec("safe_reach"))
    strip = lambda v: [{k: x for k, x in r.items() if k != "wall_s"} for r in v.rollouts]
    assert strip(a) == strip(b) and a.simulated_failure_fraction == b.simulated_failure_fraction and a.action_id == b.action_id


def test_budgets_are_paired_prefixes(stack, rec):
    _, (twin, c, verifier, g) = stack()
    r = rec("safe_reach")
    small, big = verifier.verify(r, 3), verifier.verify(r, 10)
    for i in range(3):
        for k in ("ee_final_m", "outcome", "perturbation", "seed_spawn_key"):
            assert small.rollouts[i][k] == big.rollouts[i][k]


def test_linear_interpolation_option_runs_and_is_recorded(make_cfg, twin, rec):
    from sim_before_act.runtime.offline import build_stack
    cfg = make_cfg(**{"contract.interpolation": "linear"})
    twin_, contract, verifier, gate = build_stack(cfg, twin)
    v = verifier.verify(rec("safe_reach"))
    assert contract.interpolation == "linear" and v.status == "COMPLETED"


def test_rollout_cost_is_small(twin, contract, rec):
    import time
    t = time.monotonic(); run(twin, contract, rec("safe_reach")); assert time.monotonic() - t < 5.0
