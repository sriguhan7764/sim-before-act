"""Milestone G: offline segment -> N seeded rollouts -> failure fraction. SYNTHETIC fixtures only; no hardware.

The 'unsafe' case drives the arm into the mat with INVENTED actions. Results say nothing about the physical robot.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sim_before_act.integration.offline_pipeline import (CATEGORIES, budget_study, decide, rollout_seeds, shadow_verification_block,
                                                         verify_segment)
from sim_before_act.robot.segment import write_fixture_segment
from sim_before_act.twin.scene.builder import build_scene
from sim_before_act.twin.scene.config import load_scene_config

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "configs" / "scene_synthetic_demo.yaml"
PHYS = ROOT / "configs" / "scene_physical_template.yaml"
pytestmark = pytest.mark.simulation


@pytest.fixture(scope="module")
def scene():
    return build_scene(load_scene_config(SCENE))


@pytest.fixture(scope="module")
def cases(scene, tmp_path_factory):
    d = tmp_path_factory.mktemp("cases")
    q0 = np.asarray(scene.cfg.inputs["initial_robot"]["values"], float)
    safe = np.tile(q0, (40, 1))
    unsafe = safe.copy()
    unsafe[20:] = [0.0, 90.0, 0.0, 0.0, 0.0, 40.0]
    outside = safe.copy()
    outside[30:, 2] = 120.0                      # outside the simulator limit: refused, never clipped
    out = {}
    for name, a, kw in (("safe", safe, {}), ("unsafe", unsafe, {}), ("outside", outside, {}),
                        ("bad_state", safe, {"state_values": [0, -105.0, 30, 20, 0, 30]})):
        p = d / f"{name}.json"
        write_fixture_segment(p, a, contract=scene.contract, segment_id=name, **kw)
        out[name] = p
    return out


def test_seeds_are_nested_prefixes_and_deterministic():
    a, b = rollout_seeds(3, 10), rollout_seeds(3, 5)
    assert a[:5] == b and rollout_seeds(3, 10) == a and rollout_seeds(4, 10) != a and len(set(a)) == 10


def test_safe_fixture_task_unknown_is_insufficient_evidence_not_approval(scene, cases):
    r = verify_segment(cases["safe"], scene, 5, 1)
    assert r["counts"]["unknown"] == 5 and r["counts"]["task_success"] == 0
    assert r["status"] == "INSUFFICIENT_EVIDENCE" and r["decision"] == "REJECT" and r["offline_decision_label"] == "WOULD_REJECT"
    assert r["simulated_failure_fraction"] is None and r["interval"] is None and r["approval_scope"] == "none"
    assert "task_outcome_unavailable" in r["reason_codes"]
    assert r["criteria"]["safety"]["available"] and r["criteria"]["safety"]["violations"] == 0      # safety evaluable and clean
    assert r["criteria"]["task"]["evaluable"] == 0 and not r["criteria"]["task"]["available"]
    assert r["denominators"]["determined"] == 0 and r["denominators"]["safety_evaluable"] == 5
    assert "NOT a full-task success prediction" in r["horizon_claim"] and r["threshold_calibrated"] is False


def test_safe_fixture_in_explicit_safety_only_scope_is_not_full_approval(scene, cases):
    r = verify_segment(cases["safe"], scene, 5, 1, required_criteria=("safety",))
    assert r["status"] == "COMPLETED" and r["decision"] == "WOULD_EXECUTE" and r["approval_scope"] == "safety_only"
    assert r["simulated_failure_fraction"] == 0.0 and r["criteria"]["task"]["evaluable"] == 0
    with pytest.raises(ValueError):
        verify_segment(cases["safe"], scene, 1, 1, required_criteria=("task",))           # safety can never be dropped


def test_real_rollout_reaching_task_horizon_gives_known_task_failure(cases):
    short = build_scene(load_scene_config(SCENE, overrides={"simulation.task_horizon_s": 1.5}))      # invented, synthetic-only
    r = verify_segment(cases["safe"], short, 4, 1, save_traces=False)
    assert r["counts"]["task_failure"] == 4 and r["criteria"]["task"]["evaluable"] == 4
    assert r["status"] == "COMPLETED" and r["simulated_failure_fraction"] == 1.0 and r["decision"] == "REJECT"
    assert r["rollouts"][0]["horizon_kind"] == "task" and r["rollouts"][0]["predicates_unmet_but_inconclusive"] == []


def test_trace_and_audit_agree_with_the_decision(scene, cases, tmp_path):
    from sim_before_act.integration.offline_pipeline import outcome_audit
    r = verify_segment(cases["safe"], scene, 3, 2, out_dir=tmp_path)
    on_disk = json.loads((tmp_path / "verification.json").read_text())
    a = outcome_audit(on_disk)
    assert on_disk["decision"] == r["decision"] == a["decision"] == "REJECT" and on_disk["status"] == a["status"] == "INSUFFICIENT_EVIDENCE"
    assert a["simulation_completed"] == 3 and all(p["simulation_completed"] and p["safety_status"] == "clean" for p in a["per_rollout"])
    assert all(p["task_status"].startswith("unknown") and "horizon_kind=segment" in p["why_task_unknown"] for p in a["per_rollout"])
    assert a["availability"]["safety"]["evaluable"] == 3 and a["availability"]["task"]["evaluable"] == 0
    assert r["segment"]["data_origin"] == "synthetic_fixture" and not r["segment"]["reported_as_trained"]


def test_unsafe_fixture_counts_safety_violations_and_rejects(scene, cases):
    r = verify_segment(cases["unsafe"], scene, 5, 1)
    assert r["counts"]["safety_violation"] == 5 and r["simulated_failure_fraction"] == 1.0
    assert r["decision"] == "REJECT" and r["reason_codes"] == ["risk_at_or_above_threshold", "safety_violation_observed"]
    assert r["status"] == "COMPLETED" and r["denominators"]["determined"] == 5 and r["criteria"]["task"]["evaluable"] == 0
    assert r["interval"]["lower"] > 0.4 and r["interval"]["scope"].startswith("Monte-Carlo")


def test_out_of_limit_actions_are_refused_not_clipped(scene, cases):
    r = verify_segment(cases["outside"], scene, 5, 1)
    assert r["status"] == "INVALID_INPUT" and r["decision"] == "REJECT" and any("limits" in p for p in r["problems"])
    assert r["simulated_failure_fraction"] is None


def test_invalid_rollouts_are_not_failures_and_not_zero_risk(scene, cases):
    r = verify_segment(cases["bad_state"], scene, 5, 1, init="segment_state")
    assert r["counts"]["invalid_rollout"] == 5 and r["ran_validly"] == 0
    assert r["simulated_failure_fraction"] is None and r["interval"] is None
    assert r["status"] == "INSUFFICIENT_VALID_ROLLOUTS" and r["decision"] == "REJECT"
    assert "safety_outcome_unavailable" in r["reason_codes"] and "task_outcome_unavailable" in r["reason_codes"]


def test_budget_study_same_cases_n_1_5_10(scene, cases):
    rows = budget_study(cases, scene, ns=(1, 5, 10), base_seed=2)
    assert len(rows) == 12 and {r["n_rollouts"] for r in rows} == {1, 5, 10}
    for r in rows:
        assert set(r["counts"]) == set(CATEGORIES)
        assert sum(r["counts"].values()) == (0 if r["status"] == "INVALID_INPUT" else r["n_rollouts"])
    assert {r["status"] for r in rows if r["case_id"] == "outside"} == {"INVALID_INPUT"}
    wide = {r["n_rollouts"]: r["interval"] for r in rows if r["case_id"] == "unsafe"}
    assert (wide[1]["upper"] - wide[1]["lower"]) > (wide[10]["upper"] - wide[10]["lower"])   # more rollouts, narrower interval
    from sim_before_act.evaluation.report import budget_comparison
    b = budget_comparison(rows)
    assert b["n_cases"] == 4 and b["rollout_budgets"] == [1, 5, 10]


def test_traces_are_written_per_rollout_and_result_is_reproducible(scene, cases, tmp_path):
    r1 = verify_segment(cases["safe"], scene, 3, 9, out_dir=tmp_path / "a")
    r2 = verify_segment(cases["safe"], scene, 3, 9, out_dir=tmp_path / "b")
    assert len(list((tmp_path / "a" / "rollouts").glob("*_trace.npz"))) == 3
    assert json.loads((tmp_path / "a" / "verification.json").read_text())["config_hash"] == r1["config_hash"] == r2["config_hash"]
    assert [x["seed"] for x in r1["rollouts"]] == [x["seed"] for x in r2["rollouts"]]
    assert [x["category"] for x in r1["rollouts"]] == [x["category"] for x in r2["rollouts"]]


def test_config_hash_binds_n_seed_and_threshold(scene, cases):
    h = lambda **k: verify_segment(cases["safe"], scene, k.pop("n", 1), k.pop("seed", 1), save_traces=False, **k)["config_hash"]  # noqa: E731
    assert len({h(), h(n=2), h(seed=2), h(threshold=0.1)}) == 4


def test_tampered_segment_is_refused_not_repaired(scene, cases, tmp_path):
    raw = json.loads(cases["safe"].read_text())
    raw["actions"][3][0] += 1.0
    p = tmp_path / "t.json"
    p.write_text(json.dumps(raw))
    r = verify_segment(p, scene, 3, 1)
    assert r["status"] == "INVALID_INPUT" and r["decision"] == "REJECT" and r["problems"]


def test_physical_scene_template_refuses(cases):
    r = verify_segment(cases["safe"], PHYS, 3, 1)
    assert r["status"] == "REFUSED_INPUTS" and r["decision"] == "REJECT" and r["refusal"]


def test_decision_rule_is_fail_closed_and_equality_rejects():
    assert decide("COMPLETED", 0.3, 0.3)[0] == "REJECT"
    assert decide("COMPLETED", 0.29, 0.3)[0] == "WOULD_EXECUTE"
    assert decide("COMPLETED", None, 0.3)[0] == "REJECT" and decide("TIMEOUT", 0.0, 0.3)[0] == "REJECT"


def test_shadow_block_roundtrip(scene, cases, tmp_path):
    from sim_before_act.shadow.workflow import load_prediction, record_prediction
    r = verify_segment(cases["safe"], scene, 3, 1, save_traces=False)
    record_prediction(
        tmp_path, trial_id="g1", data_origin="synthetic_integration_test", provenance_label="SYNTHETIC integration-test trial",
        initial_state={"state_values": [0.0] * 6, "state_sha256": "a" * 64, "state_timestamp": 1, "clock_domain": "none (synthetic)"},
        action={"segment_id": "safe", "action_id": r["contract_id"], "action_sha256": r["action_sha256"], "chunk_length": r["chunk_length"]},
        policy={"kind": "fixture_stub", "checkpoint_id": "none", "trained_for_task": False},
        verification=shadow_verification_block(r), execution_horizon={"type": "segment", "steps": r["executed_steps"]})
    assert load_prediction(tmp_path, "g1")["verification"]["config_hash"] == r["config_hash"]
    v = load_prediction(tmp_path, "g1")["verification"]                 # persisted record agrees with the gate result
    assert v["decision"] == r["decision"] == "REJECT" and v["status"] == r["status"] == "INSUFFICIENT_EVIDENCE"
    assert v["approval_scope"] == "none" and v["denominators"] == r["denominators"] and v["simulated_failure_fraction"] is None


@pytest.mark.skipif(not Path("/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python").exists()
                    or not Path("/home/apic3/lerobot_v6_2/lerobot/src").exists(), reason="lerobot env not available")
def test_cross_env_act_smoke_segment_is_accepted_and_verified(scene, tmp_path):
    """ACT smoke (random-init, 6 steps, torch env) writes a segment; the MuJoCo env verifies it. Software execution only."""
    collector = ROOT.parent / "robot_collector"
    env = {**os.environ, "PYTHONPATH": f"/home/apic3/lerobot_v6_2/lerobot/src:{collector}", "PYTHONDONTWRITEBYTECODE": "1"}
    out = tmp_path / "smoke"
    p = subprocess.run(["/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python", "-m", "collector.act", "smoke", "--out", str(out)],
                       cwd=collector, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    r = verify_segment(out / "action_segment.json", scene, 5, 1, save_traces=False)
    assert r["status"] not in ("INVALID_INPUT", "REFUSED_INPUTS"), r.get("problems")          # the segment is accepted and simulated
    assert r["decision"] == "REJECT" and r["approval_scope"] == "none"       # random-init output can never be approved offline either
    assert r["status"] in ("INSUFFICIENT_EVIDENCE", "COMPLETED") and r["criteria"]["task"]["success"] == 0
    assert r["segment"]["policy_kind"] == "smoke_random_init_few_steps" and r["segment"]["trained_for_task"] is False
    assert "NOT A TRAINED POLICY" in r["segment"]["policy_label"]
