import json
import os

import numpy as np
import pytest

from sim_before_act.cli import main as cli_main
from sim_before_act.runtime.offline import run_attempt


class SpyAdapter:
    def __init__(self):
        self.sent = []

    def send_chunk(self, chunk):
        self.sent.append(np.array(chunk))


def test_record_to_twin_to_risk_to_decision_to_trace(stack, rec, tmp_path):
    cfg, st = stack()
    res = run_attempt(cfg, rec("safe_reach"), tmp_path, st)
    v, t = res.verification, res.trace
    assert v.status == "COMPLETED" and v.estimate.valid == 10
    assert res.decision.decision == "EXECUTE" and res.execution.executed is False   # offline: no adapter
    assert t["verification"]["status"] == "COMPLETED" and len(t["verification"]["rollouts"]) == 10
    assert t["decision"]["decision"] == "EXECUTE" and t["execution"]["executed"] is False
    assert t["synthetic"] is True and t["verification"]["perturbation"]["all_calibrated"] is False
    assert t["reproducibility"]["config_hash"] == cfg.hash and t["reproducibility"]["robot_asset_hash"] == st[0].asset_hash
    assert res.trace_path.exists() and not os.access(res.trace_path, os.W_OK)
    assert json.loads(res.trace_path.read_text())["identity"]["action_id"] == v.action_id


def test_rejected_and_failed_attempts_still_write_traces(stack, rec, tmp_path):
    cfg, st = stack()
    a = run_attempt(cfg, rec("floor_contact"), tmp_path, st)
    b = run_attempt(cfg, rec("safe_reach", state__timestamp_s=0.0), tmp_path, st)    # stale
    assert a.decision.decision == "REJECT" and b.verification.status == "INVALID_STATE" and b.decision.decision == "REJECT"
    assert a.trace_path.exists() and b.trace_path.exists() and a.trace_path != b.trace_path
    assert a.trace["verification"]["safety_violation_fraction"] == 1.0
    assert b.trace["verification"]["simulated_failure_fraction"] is None


def test_trace_never_overwrites(stack, rec, tmp_path):
    from sim_before_act.util import write_json_exclusive
    p = tmp_path / "x.json"; write_json_exclusive(p, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json_exclusive(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}


def test_rejected_actions_never_reach_adapter(stack, rec, tmp_path):
    cfg, st = stack()
    spy = SpyAdapter()
    for name in ("floor_contact", "far_target"):
        res = run_attempt(cfg, rec(name), tmp_path, st, adapter=spy)
        assert res.decision.decision == "REJECT" and not res.execution.executed
    assert spy.sent == []


def test_approved_action_reaches_adapter_only_through_guard(stack, rec, tmp_path):
    cfg, st = stack()
    spy = SpyAdapter()
    res = run_attempt(cfg, rec("safe_reach"), tmp_path, st, adapter=spy)
    assert res.execution.executed and len(spy.sent) == 1
    assert np.allclose(spy.sent[0], np.array(rec("safe_reach").action.chunk))


def test_missing_outcome_label_is_reported_not_invented(stack, rec, tmp_path):
    cfg, st = stack()
    t = run_attempt(cfg, rec("safe_reach"), tmp_path, st).trace
    po = t["physical_outcome"]
    assert po["label_available"] is False and po["recorded_action_outcome"]["observed_success"] is None
    assert po["recorded_action_outcome"]["observed_failure"] is None
    assert any("No observed outcome label" in x for x in t["limitations"])


def test_recorded_outcome_labels_recorded_action_only_and_does_not_alter_verification(stack, rec, tmp_path):
    cfg, st = stack()
    a = run_attempt(cfg, rec("safe_reach"), tmp_path, st)
    labelled = rec("safe_reach", outcome={"observed_success": False, "failure_type": "x", "label_source": "test",
                                          "horizon": "chunk", "applies_to": "recorded_action_only"})
    b = run_attempt(cfg, labelled, tmp_path, st)
    assert a.verification.simulated_failure_fraction == b.verification.simulated_failure_fraction
    o = b.trace["physical_outcome"]
    assert o["recorded_action_outcome"]["observed_failure"] is True and o["recorded_action_outcome"]["applies_to"] == "recorded_action_only"


def test_unavailable_policy_confidence_stays_null(stack, rec, tmp_path):
    cfg, st = stack()
    t = run_attempt(cfg, rec("safe_reach"), tmp_path, st).trace
    assert t["action"]["policy_confidence"] is None and "never fabricated" in t["action"]["policy_confidence_note"]


def test_cli_verify_end_to_end(cfg_path, tmp_path, capsys):
    from conftest import FIX_DIR
    rc = cli_main(["verify", "--config", str(cfg_path), "--record", str(FIX_DIR / "floor_contact.json"), "--n", "5", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "SYNTHETIC" in out and "REJECT" in out and len(list(tmp_path.glob("trace_*.json"))) == 1


def test_cli_returns_error_code_on_bad_input(cfg_path, tmp_path, capsys):
    assert cli_main(["verify", "--config", str(cfg_path), "--record", str(tmp_path / "nope.json"), "--out", str(tmp_path)]) == 2


def test_gate_uses_point_estimate_not_wilson_bound_and_trace_states_rule(stack, rec, tmp_path):
    # 0/10 failures: point estimate 0.0 < 0.1, but the Wilson upper bound (~0.28) is above 0.1.
    cfg, st = stack(**{"gate.risk_threshold": 0.1})
    t = run_attempt(cfg, rec("safe_reach"), tmp_path, st).trace
    assert t["verification"]["interval"]["high"] > 0.1
    assert t["decision"]["decision"] == "EXECUTE"
    rule = t["decision"]["decision_rule"]
    assert rule["uses_wilson_upper_bound"] is False and rule["threshold"] == 0.1
    assert rule["threshold_calibrated"] is False and "strict" in rule["comparison"]
    assert "Wilson" in t["verification"]["interval_note"] and "preliminary" in t["verification"]["timing_note"]
    assert any("PRELIMINARY" in x for x in t["limitations"])
