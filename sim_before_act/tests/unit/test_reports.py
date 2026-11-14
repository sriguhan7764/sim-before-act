"""Report logic tests.

The 'physical' trials here are LOGIC FIXTURES in tmp dirs that exercise the statistics code paths (class-composition gating,
pooling rules). They are not data, and nothing from them is ever written outside pytest's tmp_path.
"""
import json

import pytest

from sim_before_act.evaluation.report import (MIN_PER_CLASS, auc, budget_comparison, build_report, render_dataset_card,
                                              render_markdown, wilson, write_report)
from sim_before_act.shadow.workflow import annotate_trial, load_trials, record_prediction

from test_shadow import SYN, kw


def make(root, tid, *, origin="physical_trial", decision="REJECT", frac=0.5, outcome=None, horizon="segment", timings=None,
         execution="executed", **ann):
    k = kw(tid, origin=origin, decision=decision, trained=origin == "physical_trial")
    k["verification"]["simulated_failure_fraction"] = frac
    k["timings"] = timings or {}
    record_prediction(root, clock_wall_ns=lambda: 1, **k)
    if outcome or execution != "executed":
        base = dict(replanning_or_temporal_aggregation=False, deviations_from_verified_action=[],
                    executed_command_ref={"action_sha256": "b" * 64})
        if outcome in ("success", "failure"):
            base.update(outcome_horizon=horizon, label_source="operator_visual")
        base.update(ann)
        annotate_trial(root, tid, execution=execution, observed_outcome=outcome, annotated_by="op", clock_wall_ns=lambda: 2, **base)


def trials(root, physical):
    return [t for t in load_trials(root) if t["physical"] == physical]


def test_no_physical_trials_gives_no_results(tmp_path):
    rep = build_report(manifest=None, split=None, physical_trials=[], synthetic_trials=[])
    assert rep["physical_trials"]["n_trials"] == 0 and "NO physical" in rep["physical_trials"]["note"]
    assert "NO physical shadow trials" in render_markdown(rep)


def test_only_successes_or_only_failures_withhold_discrimination(tmp_path):
    for i in range(7):
        make(tmp_path, f"s{i}", outcome="success", decision="WOULD_EXECUTE", frac=0.1)
    pv = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=[])["physical_trials"][
        "prediction_vs_outcome"]
    assert pv["roc_auc"] is None and "no observed failures" in pv["withheld"] and pv["observed_failures"] == 0
    for i in range(7):
        make(tmp_path / "f", f"f{i}", outcome="failure", frac=0.9)
    pv = build_report(manifest=None, split=None, physical_trials=trials(tmp_path / "f", True), synthetic_trials=[])[
        "physical_trials"]["prediction_vs_outcome"]
    assert pv["roc_auc"] is None and "no observed successes" in pv["withheld"]


def test_small_sample_withholds_rates_and_auc(tmp_path):
    for i in range(MIN_PER_CLASS - 1):
        make(tmp_path, f"s{i}", outcome="success", decision="WOULD_EXECUTE", frac=0.1)
        make(tmp_path, f"f{i}", outcome="failure", decision="REJECT", frac=0.9)
    pv = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=[])["physical_trials"][
        "prediction_vs_outcome"]
    assert pv["roc_auc"] is None and "fewer than" in pv["withheld"] and "sensitivity_for_failures" not in pv
    assert pv["confusion"]["reject_and_failed"] == MIN_PER_CLASS - 1


def test_both_classes_adequate_gives_metrics_with_caveat(tmp_path):
    for i in range(MIN_PER_CLASS):
        make(tmp_path, f"s{i}", outcome="success", decision="WOULD_EXECUTE", frac=0.1)
        make(tmp_path, f"f{i}", outcome="failure", decision="REJECT", frac=0.9)
    pv = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=[])["physical_trials"][
        "prediction_vs_outcome"]
    assert pv["roc_auc"]["value"] == 1.0 and "not a validated" in pv["roc_auc"]["note"]
    assert pv["sensitivity_for_failures"]["value"] == 1.0 and pv["false_rejection_rate"]["value"] == 0.0


def test_missing_numeric_score_withholds_auc(tmp_path):
    for i in range(MIN_PER_CLASS):
        make(tmp_path, f"s{i}", outcome="success", decision="WOULD_EXECUTE", frac=0.1)
        make(tmp_path, f"f{i}", outcome="failure", decision="REJECT", frac=None if i == 0 else 0.9)
    pv = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=[])["physical_trials"][
        "prediction_vs_outcome"]
    assert pv["roc_auc"] is None and "no numeric" in pv["withheld"]


def test_unknown_abort_unannotated_and_full_task_are_not_in_comparison(tmp_path):
    make(tmp_path, "u")
    make(tmp_path, "a", outcome="aborted", abort_reason="stop")
    make(tmp_path, "k", outcome="unknown")
    make(tmp_path, "n", execution="not_executed")
    make(tmp_path, "ft", outcome="success", horizon="full_task")
    p = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=[])["physical_trials"]
    assert p["n_unannotated"] == 1 and p["n_aborted"] == 1 and p["n_unknown"] == 1 and p["n_not_executed"] == 1
    assert p["prediction_vs_outcome"]["n_evaluable"] == 0
    assert p["policy_full_task_outcomes"]["n_labelled"] == 1 and p["policy_full_task_outcomes"]["successes"] == 1
    assert {e["trial_id"] for e in p["excluded"]} == {"u", "a", "k", "n", "ft"}


def test_inexact_comparisons_are_listed(tmp_path):
    make(tmp_path, "x", outcome="failure", replanning_or_temporal_aggregation=True)
    p = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=[])["physical_trials"]
    assert p["segment_matched"]["not_exact_trial_ids"] == ["x"]


def test_physical_and_synthetic_never_pooled(tmp_path):
    make(tmp_path, "syn1", origin="synthetic_integration_test", outcome="failure")
    make(tmp_path, "ph1", outcome="success", decision="WOULD_EXECUTE")
    allt = load_trials(tmp_path)
    with pytest.raises(ValueError):
        build_report(manifest=None, split=None, physical_trials=allt, synthetic_trials=[])
    with pytest.raises(ValueError):
        build_report(manifest=None, split=None, physical_trials=[], synthetic_trials=allt)
    rep = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=trials(tmp_path, False))
    assert rep["physical_trials"]["n_trials"] == 1 and rep["synthetic_trials"]["n_trials"] == 1
    assert "never evidence" in rep["synthetic_trials"]["label"]


def test_latency_summary_and_missing(tmp_path):
    make(tmp_path, "a", timings={"inference_s": 0.2, "verification_s": 3.0})
    make(tmp_path, "b", timings={"inference_s": 0.4})
    lat = build_report(manifest=None, split=None, physical_trials=trials(tmp_path, True), synthetic_trials=[])["physical_trials"][
        "latency"]
    assert lat["inference"]["n"] == 2 and lat["inference"]["median_s"] == pytest.approx(0.3)
    assert lat["verification"]["n"] == 1


def test_statistics_helpers():
    assert auc([0.9, 0.8], [0.1, 0.8]) == pytest.approx(0.875)
    w = wilson(0, 5)
    assert w["lower"] == 0.0 and 0.4 < w["upper"] < 0.6 and wilson(0, 0) is None


def test_budget_comparison_same_cases():
    rows = [{"case_id": c, "n_rollouts": n, "simulated_failure_fraction": f, "status": "COMPLETED", "decision": d,
             "interval": {"lower": 0.0, "upper": 0.5}}
            for c, n, f, d in [("c1", 1, 0.0, "WOULD_EXECUTE"), ("c1", 5, 0.4, "REJECT"), ("c2", 1, 0.0, "WOULD_EXECUTE"),
                               ("c2", 5, 0.0, "WOULD_EXECUTE"), ("c2", 10, 0.1, "WOULD_EXECUTE")]]
    b = budget_comparison(rows)
    assert b["rollout_budgets"] == [1, 5, 10] and b["n_cases"] == 2 and b["n_cases_decision_changes"] == 1
    assert [r["by_n"]["10"] for r in b["cases"]][0] is None


def test_dataset_section_and_card_use_manifest_not_plan(tmp_path):
    man = {"dataset_name": "d", "dataset_version": "0.1", "content_sha256": "f" * 64, "plan": {"train": 25, "val": 5},
           "inventory": {"n_entries": 2, "by_role": {"unclassified": 2}, "by_data_origin": {"recorded_physical": 2},
                         "training_eligible": 0, "planned_successful_demonstrations": 30, "shortfall_vs_plan": 30,
                         "successful_demonstrations_blocked_by": {}}}
    rep = build_report(manifest=man, split=None, physical_trials=[], synthetic_trials=[])
    md = render_markdown(rep)
    assert "training-eligible demonstrations: 0" in md and "shortfall 30" in md
    paths = write_report(tmp_path / "out", rep, man)
    assert "NOT claimed" in render_dataset_card(man) or "NOT claimed" in open(paths["dataset_card"]).read()
    assert json.load(open(paths["json"]))["schema"] == "review-report-v0.1"
