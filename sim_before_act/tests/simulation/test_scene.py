"""Milestone F: configurable cube-to-tray scene, outcome predicates, headless rollouts, visuals. Everything is SYNTHETIC DEMO
(invented values, CAD geometry); nothing here is evidence about the physical robot."""
import copy
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest

from sim_before_act.cli import main as cli_main
from sim_before_act.robot.segment import (ActionSegment, build_fixture_segment, load_action_segment, policy_to_sim, validate_for_sim,
                                          write_fixture_segment)
from sim_before_act.twin.perturbation import PerturbationSample
from sim_before_act.twin.scene.builder import build_scene
from sim_before_act.twin.scene.config import SceneConfigError, SceneRefusal, load_scene_config
from sim_before_act.twin.scene.fixtures import make_fixture
from sim_before_act.twin.scene.outcomes import (CONTACT_CATEGORIES, TraceArrays, classify_pair, combine_outcome, contact_report,
                                                eval_constraints, eval_contacts, eval_inside_tray, eval_lift, eval_release_settle)
from sim_before_act.twin.scene.runner import check_sample_in_ranges, run_scene_rollout
from sim_before_act.twin.scene.viz import plot_rollout, render_contact_sheet
from sim_before_act.util import to_jsonable

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "configs" / "scene_synthetic_demo.yaml"
PHYS = ROOT / "configs" / "scene_physical_template.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_scene_config(SYN)


@pytest.fixture(scope="module")
def scene(cfg):
    return build_scene(cfg)


def _seg(scene, actions, tmp_path, name="seg.json", **kw):
    p = tmp_path / name
    write_fixture_segment(p, actions, contract=scene.contract, **kw)
    return p


def _state(scene, q_deg, cube=None, cube_z=None):
    d = scene.new_data()
    scene.set_robot_q(d, scene.contract.to_sim_rad(np.asarray(q_deg, float)))
    scene.place_cube(d, *(cube or (0.15, 0.05, 0.0)), z=cube_z)
    mujoco.mj_forward(scene.model, d)
    return d


# ---- build / config / refusal -----------------------------------------------------------------------------------------------
def test_scene_builds_and_is_labelled(scene):
    d = scene.describe()
    assert d["label"].startswith("SYNTHETIC DEMO SCENE (invented values, CAD geometry; not a calibrated twin)")
    assert "NOT a physically calibrated twin" in d["fidelity"]
    assert d["nu"] == 6 and d["controller"]["kp"][0] == pytest.approx(998.22)
    assert d["contract_id"] == scene.contract.contract_id and len(d["robot_asset_hash"]) == 64
    g = scene.geometry
    assert g["mat_size_m"][:2] == [0.4, 0.4] and g["tray_rim_top_z_m"] > g["tray_floor_top_z_m"]
    assert all(src for src in (scene.cfg.inputs[k]["source"] for k in ("cube", "tray", "base_to_mat")))
    assert all("INVENTED" in scene.cfg.inputs[k]["source"] for k in ("cube", "tray", "base_to_mat", "initial_robot"))


def test_synthetic_config_cannot_hide_invented_values():
    with pytest.raises(SceneConfigError):
        load_scene_config(SYN, {"inputs.cube.source": "measured with a ruler"})
    with pytest.raises(SceneConfigError):
        load_scene_config(SYN, {"outcome.thresholds_calibrated": True})


def test_physical_template_refuses_with_precise_list():
    cfg = load_scene_config(PHYS)
    with pytest.raises(SceneRefusal) as e:
        build_scene(cfg)
    miss = " ".join(e.value.missing)
    for needle in ("inputs.base_to_mat.x_m", "inputs.cube.x_m", "inputs.tray.interior_x_m", "inputs.tray.sigma_xy_m",
                   "calibration.reference.follower_calibration_sha256", "calibration.calibration_blocker_resolved_by"):
        assert needle in miss
    with pytest.raises(SceneRefusal):
        run_scene_rollout(cfg, np.zeros((5, 6)), seed=0)


def test_physical_refuses_while_calibration_blocker_unresolved():
    raw = copy.deepcopy(load_scene_config(SYN).raw)
    raw["mode"] = "physical"
    for g in ("base_to_mat", "cube", "tray", "initial_robot"):
        raw["inputs"][g]["source"] = "TEST-ONLY placeholder, not a measurement"
    cfg = load_scene_config(raw)
    for _ in range(3):                    # fill whatever is still reported (sigmas, calibration reference) with test-only values
        for item in cfg.missing_inputs():
            path = item.split(":")[0].split(".")
            if path[-1] == "calibration_blocker_resolved_by":
                continue
            cur = raw
            for k in path[:-1]:
                cur = cur.setdefault(k, {})
            cur[path[-1]] = 0.001 if path[-1].startswith("sigma") else "test-only"
        cfg = load_scene_config(raw)
    miss = cfg.missing_inputs()
    assert len(miss) == 1 and "calibration_blocker_resolved_by" in miss[0]
    with pytest.raises(SceneRefusal):
        build_scene(cfg)
    raw["calibration"]["calibration_blocker_resolved_by"] = "test-only evidence name"
    assert load_scene_config(raw).missing_inputs() == []


# ---- predicates on constructed states ---------------------------------------------------------------------------------------------
def _trace(cube_xyz, gripper=40.0, speed=0.0, dt=0.1, contacts=None):
    cube = np.asarray(cube_xyz, float).reshape(-1, 3)
    n = len(cube)
    pres = np.zeros((n, len(CONTACT_CATEGORIES)), bool)
    pen = np.zeros((n, len(CONTACT_CATEGORIES)))
    for cat, depth in (contacts or {}).items():
        pres[:, CONTACT_CATEGORIES.index(cat)] = True
        pen[:, CONTACT_CATEGORIES.index(cat)] = depth
    return TraceArrays(t_s=np.arange(n) * dt, q_rad=np.zeros((n, 6)), cube_pos=cube, cube_speed=np.full(n, speed) if np.isscalar(speed) else np.asarray(speed),
                       gripper_pct=np.full(n, gripper) if np.isscalar(gripper) else np.asarray(gripper), contact_present=pres, contact_pen=pen,
                       max_abs_qvel=np.zeros(n))


def test_inside_tray_predicate(scene, cfg):
    th, g = cfg.section("outcome"), scene.geometry
    tp = g["tray_pose"]
    inside = [tp["x_m"], tp["y_m"], g["tray_floor_top_z_m"] + 0.02]
    assert eval_inside_tray(_trace([inside]), g, th)["value"] is True
    outside = [tp["x_m"] + g["tray_interior_m"][0], tp["y_m"], 0.02]
    assert eval_inside_tray(_trace([outside]), g, th)["value"] is False
    above_rim = [tp["x_m"], tp["y_m"], g["tray_rim_top_z_m"] + 0.03]
    assert eval_inside_tray(_trace([above_rim]), g, th)["value"] is False
    near_wall = [tp["x_m"] + g["tray_interior_m"][0] / 2 - 0.001, tp["y_m"], 0.02]          # inside the margin -> not counted
    assert eval_inside_tray(_trace([near_wall]), g, th)["value"] is False


def test_lift_predicate(scene, cfg):
    th, g = cfg.section("outcome"), scene.geometry
    rest, high = 0.02, 0.02 + th["lift_height_m"]
    assert eval_lift(_trace([[0, 0.2, rest]] * 3), g, th)["value"] is False
    r = eval_lift(_trace([[0, 0.2, rest], [0, 0.2, high + 0.001]]), g, th)
    assert r["value"] is True and r["thresholds"]["lift_height_m"] == th["lift_height_m"]
    assert r["thresholds"]["threshold_status"] == "ASSUMED_UNCALIBRATED"


def test_release_and_settle_predicate(cfg):
    th = cfg.section("outcome")
    n = 10
    ok = eval_release_settle(_trace([[0, 0, 0.02]] * n, gripper=80.0, speed=0.0), th)
    assert ok["value"] is True
    assert eval_release_settle(_trace([[0, 0, 0.02]] * n, gripper=10.0, speed=0.0), th)["value"] is False        # not opened
    assert eval_release_settle(_trace([[0, 0, 0.02]] * n, gripper=80.0, speed=0.5), th)["value"] is False        # still moving
    assert eval_release_settle(_trace([[0, 0, 0.02]] * 2, gripper=80.0, speed=0.0), th)["value"] is False        # window too short


def test_contact_categories_from_constructed_states(scene):
    thr = 0.0
    ee = scene.ee_position(_state(scene, [0, -50, 40, 50, 0, 40]))
    held = _state(scene, [0, -50, 40, 50, 0, 40], cube=(ee[0], ee[1], 0.0), cube_z=ee[2])
    assert "gripper_cube" in contact_report(scene, held, thr)
    on_mat = contact_report(scene, _state(scene, [0, -50, 40, 50, 0, 40]), thr)
    assert set(on_mat) == {"cube_mat_table"}
    tp, g = scene.geometry["tray_pose"], scene.geometry
    in_tray = _state(scene, [0, -50, 40, 50, 0, 40], cube=(tp["x_m"], tp["y_m"], 0.0), cube_z=g["tray_floor_top_z_m"] + g["cube_side_m"] / 2 - 0.0005)
    assert "cube_tray" in contact_report(scene, in_tray, thr)
    assert "arm_mat_table" in contact_report(scene, _state(scene, [0, 90, 0, 0, 0, 40]), thr)
    assert "arm_self" in contact_report(scene, _state(scene, [0, -100, 90, 90, 0, 40]), thr)
    assert classify_pair("robot_gripper", "cube") == "gripper_cube" and classify_pair("robot_arm", "mat") == "arm_mat_table"
    assert classify_pair("tray", "mat") is None


def test_gripper_cube_contact_is_not_unsafe_but_arm_mat_is(cfg):
    th = cfg.section("outcome")
    expected = eval_contacts(_trace([[0, 0, 0.02]] * 3, contacts={"gripper_cube": 0.001, "cube_mat_table": 0.0}), th)
    assert expected["value"] is False and expected["unsafe_hit"] == []
    bad = eval_contacts(_trace([[0, 0, 0.02]] * 3, contacts={"arm_mat_table": 0.001}), th)
    assert bad["value"] is True and bad["unsafe_hit"] == ["arm_mat_table"]


# ---- outcome separation -------------------------------------------------------------------------------------------------------------
def _preds(scene, cfg, tr):
    th, g = cfg.section("outcome"), scene.geometry
    lo, hi = scene.model.jnt_range[scene.jnt_ids, 0], scene.model.jnt_range[scene.jnt_ids, 1]
    return {"constraints": eval_constraints(tr, lo, hi, th), "contacts": eval_contacts(tr, th), "lift": eval_lift(tr, g, th),
            "inside_tray": eval_inside_tray(tr, g, th), "release_settle": eval_release_settle(tr, th)}


def test_outcome_categories_are_separate(scene, cfg):
    g, th = scene.geometry, cfg.section("outcome")
    tp = g["tray_pose"]
    zin = g["tray_floor_top_z_m"] + 0.02
    path = [[-0.06, 0.22, 0.02], [-0.06, 0.22, 0.09], [tp["x_m"], tp["y_m"], 0.09]] + [[tp["x_m"], tp["y_m"], zin]] * 6
    ok_tr = _trace(path, gripper=[40, 40, 40, 80, 80, 80, 80, 80, 80], speed=[0, 0, 0, 0.5, 0.1, 0, 0, 0, 0], dt=0.1)
    succ = combine_outcome(_preds(scene, cfg, ok_tr), 8.0, 8.0)
    assert succ["category"] == "task_success" and not succ["safety_violation"]

    fail_tr = _trace([[-0.06, 0.22, 0.02]] * 10, gripper=40.0)
    fail = combine_outcome(_preds(scene, cfg, fail_tr), 8.0, 8.0)
    assert fail["category"] == "task_failure"
    assert set(fail["task_failure_reasons"]) == {"cube_not_lifted", "not_in_tray", "not_released"}

    short = combine_outcome(_preds(scene, cfg, fail_tr), 1.0, 8.0)                 # same data, short horizon: NOT a failure
    assert short["category"] == "unknown" and short["horizon_kind"] == "segment" and not short["task_failure_reasons"]

    unsafe_tr = _trace(path, gripper=[40, 40, 40, 80, 80, 80, 80, 80, 80], speed=[0, 0, 0, 0.5, 0.1, 0, 0, 0, 0],
                       contacts={"arm_mat_table": 0.001})
    unsafe = combine_outcome(_preds(scene, cfg, unsafe_tr), 8.0, 8.0)
    assert unsafe["category"] == "safety_violation" and unsafe["task_outcome"] == "task_success" and unsafe["safety_violation"]

    assert combine_outcome(_preds(scene, cfg, ok_tr), 8.0, 8.0, timed_out=True)["category"] == "timeout"
    inv = combine_outcome(_preds(scene, cfg, ok_tr), 8.0, 8.0, invalid_reason="NaN state")
    assert inv["category"] == "invalid_rollout" and inv["invalid_reason"] and not inv["safety_violation"]
    nan_tr = _trace(path)
    nan_tr.max_abs_qvel[2] = np.nan
    assert combine_outcome(_preds(scene, cfg, nan_tr), 8.0, 8.0)["category"] == "invalid_rollout"
    assert th["thresholds_calibrated"] is False


# ---- headless rollouts -----------------------------------------------------------------------------------------------------------------
def test_zero_motion_segment_end_to_end(scene, tmp_path):
    a = make_fixture(scene.cfg, "zero", 30)
    p = _seg(scene, a, tmp_path)
    r = run_scene_rollout(scene, p, seed=0)
    assert r.category == "unknown" and r.outcome["horizon_kind"] == "segment"          # a short rollout is not a full-task prediction
    assert r.outcome["task_failure_reasons"] == [] and not r.outcome["safety_violation"]
    assert r.labels["scene"].startswith("SYNTHETIC DEMO SCENE") and "NOT A TRAINED POLICY" in r.labels["policy"]
    seg_hash = validate_for_sim(load_action_segment(p), scene.contract).action_sha256
    assert r.action_record["action_sha256"] == seg_hash and r.action_record["unmodified_after_run"] is True
    assert r.predicates["lift"]["numbers"]["max_com_height_above_mat_m"] == pytest.approx(0.02, abs=2e-3)       # cube rests on the mat
    assert r.trace["q_rad"].shape[1] == 6 and np.all(np.isfinite(r.trace["qpos"]))
    np.testing.assert_allclose(np.rad2deg(r.trace["q_rad"][-1, :5]), a[-1, :5], atol=1.5)                       # PD target tracking


def test_small_motion_segment_end_to_end_and_adapter(scene, tmp_path):
    a = make_fixture(scene.cfg, "small_motion", 45)
    p = _seg(scene, a, tmp_path, execution_horizon_steps=30)
    r = policy_to_sim(p, scene, seed=1)
    assert r.category in ("unknown", "task_success", "safety_violation") and r.outcome["horizon_kind"] == "segment"
    assert r.category != "invalid_rollout", r.detail
    assert r.action_record["executed_steps"] == 30 and r.action_record["chunk_length"] == 45
    assert r.action_record["executed_prefix_sha256"] != r.action_record["action_sha256"]
    assert r.trace["gripper_pct"][-1] > r.trace["gripper_pct"][0]
    full = policy_to_sim(p, scene, seed=1, use_full_chunk=True)
    assert full.action_record["executed_steps"] == 45 and full.trace["t_s"][-1] > r.trace["t_s"][-1]


def test_invalid_inputs_give_invalid_rollout_not_clipping(scene, tmp_path):
    a = make_fixture(scene.cfg, "zero", 10)
    raw = build_fixture_segment(a, contract=scene.contract)
    raw["actions"][2][0] += 1.0                                                             # tamper after hashing
    r = run_scene_rollout(scene, ActionSegment(raw, None), seed=0)
    assert r.category == "invalid_rollout" and "action_sha256 mismatch" in r.detail and not r.trace
    over = a.copy()
    over[3, 2] = 200.0
    r2 = run_scene_rollout(scene, over, seed=0)
    assert r2.category == "invalid_rollout" and not r2.trace
    nan = a.copy()
    nan[1, 1] = np.nan
    assert run_scene_rollout(scene, nan, seed=0).category == "invalid_rollout"
    # a segment state outside the simulator limits (the documented sim-vs-calibration mismatch) is refused, never clipped
    seg = build_fixture_segment(a, contract=scene.contract, state_values=[0, -105.0, 30, 20, 0, 30])
    r3 = run_scene_rollout(scene, ActionSegment(seg, None), seed=0, init="segment_state")
    assert r3.category == "invalid_rollout" and "outside simulator joint limits" in r3.detail


def test_driving_into_the_mat_is_a_safety_violation(scene, tmp_path):
    a = np.tile(scene.cfg.inputs["initial_robot"]["values"], (40, 1)).astype(float)
    a[20:] = [0.0, 90.0, 0.0, 0.0, 0.0, 40.0]
    r = run_scene_rollout(scene, _seg(scene, a, tmp_path), seed=0)
    assert r.category == "safety_violation" and r.outcome["safety_violation"]
    assert any("arm_mat_table" in s or "penetration" in s for s in r.outcome["safety_reasons"])
    assert r.outcome["task_outcome"] == "unknown"           # separate from the task result


def test_timeout_category(scene):
    r = run_scene_rollout(scene, make_fixture(scene.cfg, "zero", 20), seed=0, deadline=time.monotonic() - 1.0)
    assert r.category == "timeout" and r.outcome["task_outcome"] == "unknown"


def test_determinism_and_recorded_perturbation(scene):
    a = make_fixture(scene.cfg, "small_motion", 20)
    pc = scene.cfg.perturbation_cfg()
    r1, r2 = run_scene_rollout(scene, a, seed=7, perturbation=pc), run_scene_rollout(scene, a, seed=7, perturbation=pc)
    for k in ("qpos", "cube_pos", "ctrl"):
        np.testing.assert_array_equal(r1.trace[k], r2.trace[k])
    assert r1.perturbation["ranges_checked"] and to_jsonable(r1.perturbation["sample"]) == to_jsonable(r2.perturbation["sample"])
    r3 = run_scene_rollout(scene, a, seed=8, perturbation=pc)
    assert to_jsonable(r3.perturbation["sample"]) != to_jsonable(r1.perturbation["sample"])
    nominal = run_scene_rollout(scene, a, seed=7)
    assert nominal.perturbation["mode"].startswith("none")
    assert r1.perturbation["all_calibrated"] is False


def test_out_of_range_perturbation_is_refused(scene):
    pc = scene.cfg.perturbation_cfg()
    s = PerturbationSample(index=0, seed_entropy=0, spawn_key=(), nominal=False, state_error_deg=np.zeros(5),
                           object_xy_m=np.array([0.5, 0.0]), object_yaw_deg=0.0, friction_scale=1.0, control_noise_deg=np.zeros((5, 5)))
    assert any("object_xy_m" in p for p in check_sample_in_ranges(s, pc))


# ---- visuals + CLI --------------------------------------------------------------------------------------------------------------------
def test_visual_outputs_are_written_and_labelled(scene, cfg, tmp_path):
    r = run_scene_rollout(scene, make_fixture(cfg, "small_motion", 30), seed=2)
    png = plot_rollout(r, tmp_path / "plot.png")
    img = cv2.imread(str(png))
    assert img is not None and img.shape[0] > 400 and png.stat().st_size > 10_000
    vis = render_contact_sheet(r, cfg, tmp_path, "t", video=False)
    if vis["renderer"] is None:
        pytest.skip(f"no MuJoCo GL backend worked: {vis['attempts']}")
    sheet = cv2.imread(str(vis["png"]))
    assert sheet is not None and sheet.shape[1] > 600


def test_cli_scene_commands(tmp_path, capsys):
    assert cli_main(["scene-check"]) == 0
    assert "SYNTHETIC DEMO SCENE" in capsys.readouterr().out
    assert cli_main(["scene-check", "--scene", str(PHYS)]) == 3
    assert "missing" in capsys.readouterr().out
    assert cli_main(["scene-rollout", "--scene", str(PHYS), "--fixture", "zero", "--out", str(tmp_path)]) == 3
    assert cli_main(["scene-rollout", "--fixture", "zero", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "unknown" in out and "SYNTHETIC TEST FIXTURE" in out
    assert (tmp_path / "rollout_fixture_zero_seed0_summary.json").exists()
