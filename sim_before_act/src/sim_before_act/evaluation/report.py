"""Review reports built from shadow-trial records, dataset manifests and verification summaries.

Rules:
* Physical and synthetic trials are NEVER pooled. Synthetic trials appear in their own labelled section.
* Metrics are produced only when the sample composition supports them; otherwise the field says why it is withheld.
* ROC-AUC needs >= MIN_PER_CLASS observed physical successes AND failures with a numeric score; otherwise it is not computed.
* Thresholds are read from the persisted predictions. This module never searches or tunes a threshold.
* Dataset counts come from the manifest, never from the plan.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MIN_PER_CLASS = 5
REPORT_SCHEMA = "review-report-v0.1"


def wilson(k: int, n: int, z: float = 1.959964) -> dict | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"method": "wilson", "confidence": 0.95, "lower": max(0.0, c - h), "upper": min(1.0, c + h)}


def _lat(trials: list[dict], key: str) -> dict:
    vals = sorted(float(t["timings"][key]) for t in trials if t.get("timings", {}).get(key) is not None)
    if not vals:
        return {"n": 0, "note": f"no '{key}' timings recorded"}
    mid = vals[len(vals) // 2] if len(vals) % 2 else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2])
    return {"n": len(vals), "median_s": mid, "max_s": vals[-1], "min_s": vals[0]}


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Mann-Whitney AUC; positive = observed failure. Ties count one half."""
    w = sum((p > n) + 0.5 * (p == n) for p in scores_pos for n in scores_neg)
    return w / (len(scores_pos) * len(scores_neg))


def trial_section(trials: list[dict], min_per_class: int = MIN_PER_CLASS) -> dict:
    """Summary for ONE homogeneous group of trials (call separately for physical and synthetic)."""
    n = len(trials)
    ann = [t for t in trials if t["annotated"]]
    ex = [t for t in ann if t["execution"] == "executed"]
    aborted = [t for t in ex if t["observed_outcome"] == "aborted"]
    unknown = [t for t in ex if t["observed_outcome"] == "unknown"] + [t for t in ann if t["execution"] == "unknown"]
    not_exec = [t for t in ann if t["execution"] == "not_executed"]
    full = [t for t in ex if t["observed_outcome"] in ("success", "failure") and t["label_horizon"] == "full_task"]
    succ = sum(t["observed_outcome"] == "success" for t in full)
    out = {
        "n_trials": n, "n_unannotated": n - len(ann), "n_executed": len(ex), "n_not_executed": len(not_exec),
        "n_aborted": len(aborted), "n_unknown": len(unknown),
        "excluded": [{"trial_id": t["trial_id"], "reason": t["excluded_reason"]} for t in trials
                     if t["excluded_reason"] and t["segment_label"] is None],
        "policy_full_task_outcomes": {"n_labelled": len(full), "successes": succ, "failures": len(full) - succ,
                                      "success_rate": succ / len(full) if full else None,
                                      "interval": wilson(succ, len(full)),
                                      "note": None if full else "no full-task labelled executed trials: no policy success rate"},
        "latency": {"inference": _lat(trials, "inference_s"), "verification": _lat(trials, "verification_s")},
        "threshold_values_seen": sorted({t["threshold"] for t in trials if t.get("threshold") is not None}),
        "thresholds_calibrated": sorted({bool(t["threshold_calibrated"]) for t in trials}),
        "threshold_note": "thresholds are read from the persisted predictions; none were tuned on these trials",
        "policy_identities": sorted({json.dumps(t["policy"], sort_keys=True) for t in trials}),
        "hardware": sorted({json.dumps(t["hardware"], sort_keys=True) for t in trials}),
    }
    seg = [t for t in ex if t["segment_label"] in ("success", "failure")]
    inexact = [t["trial_id"] for t in seg if not t["comparison_exact"]]
    out["segment_matched"] = {"n": len(seg), "n_not_exact": len(inexact), "not_exact_trial_ids": inexact}
    out["prediction_vs_outcome"] = _pred_vs_out(seg, min_per_class)
    return out


def _pred_vs_out(seg: list[dict], min_per_class: int) -> dict:
    res: dict = {"n_evaluable": len(seg)}
    if not seg:
        res["withheld"] = "no executed trials with a segment-horizon outcome label: no prediction-vs-outcome comparison"
        return res
    res["rows"] = [{"trial_id": t["trial_id"], "predicted_decision": t["predicted_decision"],
                    "predicted_failure_fraction": t["predicted_failure_fraction"], "observed": t["segment_label"],
                    "comparison_exact": t["comparison_exact"], "verification_status": t["verification_status"]} for t in seg]
    pos = [t for t in seg if t["segment_label"] == "failure"]
    neg = [t for t in seg if t["segment_label"] == "success"]
    res["observed_failures"], res["observed_successes"] = len(pos), len(neg)
    c = {"reject_and_failed": sum(t["predicted_decision"] == "REJECT" for t in pos),
         "would_execute_and_failed": sum(t["predicted_decision"] == "WOULD_EXECUTE" for t in pos),
         "reject_and_succeeded": sum(t["predicted_decision"] == "REJECT" for t in neg),
         "would_execute_and_succeeded": sum(t["predicted_decision"] == "WOULD_EXECUTE" for t in neg)}
    res["confusion"] = c
    why = []
    if not pos:
        why.append("no observed failures: failure detection cannot be assessed")
    if not neg:
        why.append("no observed successes: false-rejection cannot be assessed")
    if pos and neg and (len(pos) < min_per_class or len(neg) < min_per_class):
        why.append(f"fewer than {min_per_class} per class; rates/AUC would be unstable and are withheld")
    sc_p = [t["predicted_failure_fraction"] for t in pos if t["predicted_failure_fraction"] is not None]
    sc_n = [t["predicted_failure_fraction"] for t in neg if t["predicted_failure_fraction"] is not None]
    if pos and neg and (len(sc_p) < len(pos) or len(sc_n) < len(neg)):
        why.append("some trials have no numeric predicted failure fraction (invalid verification)")
    res["roc_auc"] = None
    if why:
        res["withheld"] = "; ".join(why)
        return res
    res["sensitivity_for_failures"] = {"value": c["reject_and_failed"] / len(pos), "interval": wilson(c["reject_and_failed"], len(pos))}
    res["false_rejection_rate"] = {"value": c["reject_and_succeeded"] / len(neg),
                                   "interval": wilson(c["reject_and_succeeded"], len(neg))}
    res["roc_auc"] = {"value": auc(sc_p, sc_n), "n_pos": len(pos), "n_neg": len(neg),
                      "note": "small-sample estimate on this trial set; not a validated discrimination claim"}
    return res


def budget_comparison(rows: list[dict]) -> dict:
    """rows: {case_id, n_rollouts, simulated_failure_fraction, interval, status, decision}. The same cases at different N."""
    by_case: dict = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[int(r["n_rollouts"])] = r
    ns = sorted({int(r["n_rollouts"]) for r in rows})
    table = []
    for cid, d in sorted(by_case.items()):
        row = {"case_id": cid, "by_n": {}}
        for n in ns:
            r = d.get(n)
            if r is None:
                row["by_n"][str(n)] = None
                continue
            iv = r.get("interval")
            row["by_n"][str(n)] = {"status": r["status"], "decision": r.get("decision"),
                                   "approval_scope": r.get("approval_scope", "none"), "reason_codes": r.get("reason_codes"),
                                   "failure_fraction": r.get("simulated_failure_fraction"),
                                   "interval_width": None if not iv else iv.get("upper", iv.get("high")) - iv.get("lower", iv.get("low")),
                                   "counts": r.get("counts")}
        decs = {v["decision"] for v in row["by_n"].values() if v}
        row["decision_changes_with_n"] = len(decs) > 1
        table.append(row)
    return {"rollout_budgets": ns, "n_cases": len(by_case), "cases": table,
            "n_cases_decision_changes": sum(r["decision_changes_with_n"] for r in table),
            "caveat": "fractions at small N have wide Wilson intervals; this compares simulated verification only "
                      "and says nothing about physical outcomes"}


def dataset_section(manifest: dict, split: dict | None = None) -> dict:
    out = {"manifest_content_sha256": manifest.get("content_sha256"), "dataset_version": manifest.get("dataset_version"),
           "inventory": manifest["inventory"], "plan": manifest.get("plan"), "split": None}
    if split:
        out["split"] = {k: split[k] for k in ("status", "frozen", "actual", "plan", "shortfall")}
    return out


def build_report(*, manifest: dict | None, split: dict | None, physical_trials: list[dict], synthetic_trials: list[dict],
                 budget_rows: list[dict] | None = None, collector_acceptance: dict | None = None,
                 min_per_class: int = MIN_PER_CLASS) -> dict:
    if any(not t["physical"] for t in physical_trials):
        raise ValueError("physical_trials must contain only physical trials")
    if any(t["physical"] for t in synthetic_trials):
        raise ValueError("synthetic_trials must not contain physical trials")
    return {
        "schema": REPORT_SCHEMA,
        "dataset": dataset_section(manifest, split) if manifest else {"note": "no manifest supplied: dataset counts unavailable"},
        "physical_trials": trial_section(physical_trials, min_per_class) if physical_trials else {
            "n_trials": 0, "note": "NO physical shadow trials exist: no policy success rate, no prediction-vs-outcome result."},
        "synthetic_trials": ({**trial_section(synthetic_trials, min_per_class),
                              "label": "SYNTHETIC integration-test trials only; a software execution check, never evidence "
                                       "about the robot or the method"} if synthetic_trials else {"n_trials": 0}),
        "rollout_budget_comparison": budget_comparison(budget_rows) if budget_rows else {"note": "no budget comparison rows supplied"},
        "collector_acceptance": collector_acceptance or {"note": "not supplied; see `collector accept`: the three verdicts are separate"},
        "limits": [
            "Offline EXECUTE is WOULD_EXECUTE only.",
            "The simulated failure fraction is a chunk-horizon simulator-model estimate, not a physical failure probability.",
            "Gate thresholds are uncalibrated unless a trial states otherwise; none were tuned here.",
            "CAD geometry is not a physically calibrated twin; the calibration mismatch remains a blocker until measured evidence resolves it.",
            "ACT is a pilot policy, not a language-conditioned VLA.",
        ],
    }


def render_markdown(rep: dict) -> str:
    L = ["# Review report (generated; every number comes from persisted manifests/trials)", "", "## Dataset (from manifest, not the plan)"]
    ds = rep["dataset"]
    if "inventory" in ds:
        inv = ds["inventory"]
        L += [f"- manifest sha256: `{ds['manifest_content_sha256']}`", f"- entries: {inv['n_entries']}; by role: {inv['by_role']}",
              f"- by data origin: {inv['by_data_origin']}",
              f"- training-eligible demonstrations: {inv['training_eligible']} (plan {inv['planned_successful_demonstrations']}; "
              f"shortfall {inv['shortfall_vs_plan']})",
              f"- blocked demonstrations: {inv['successful_demonstrations_blocked_by'] or 'none'}"]
        sp = ds["split"]
        L.append(f"- split: {sp['status']} actual={sp['actual']} plan={sp['plan']}" if sp else "- split: none supplied")
    else:
        L.append(f"- {ds['note']}")
    L += ["", "## Physical shadow trials"]
    p = rep["physical_trials"]
    if p["n_trials"] == 0:
        L.append(f"- {p['note']}")
    else:
        o = p["policy_full_task_outcomes"]
        iv = o["interval"]
        L += [f"- trials {p['n_trials']}: executed {p['n_executed']}, not executed {p['n_not_executed']}, aborted {p['n_aborted']}, "
              f"unknown {p['n_unknown']}, unannotated {p['n_unannotated']}",
              f"- policy full-task outcomes: {o['successes']}/{o['n_labelled']} successes"
              + (f" (Wilson 95%: {iv['lower']:.2f}-{iv['upper']:.2f})" if iv else f" ({o['note']})")]
        pv = p["prediction_vs_outcome"]
        L.append(f"- prediction vs outcome: {pv.get('withheld') or 'see JSON'}")
        if "confusion" in pv:
            L.append(f"  confusion: {pv['confusion']}")
        L.append(f"- latency inference: {p['latency']['inference']}; verification: {p['latency']['verification']}")
        L.append(f"- threshold(s): {p['threshold_values_seen']} calibrated={p['thresholds_calibrated']} ({p['threshold_note']})")
        if p["excluded"]:
            L.append(f"- excluded trials: {p['excluded']}")
    s = rep["synthetic_trials"]
    L += ["", "## Synthetic integration trials (not evidence)", f"- {s['label']}; n={s['n_trials']}" if s["n_trials"] else "- none"]
    b = rep["rollout_budget_comparison"]
    L += ["", "## Rollout-budget comparison (simulated only)"]
    if "cases" in b:
        L += [f"- budgets {b['rollout_budgets']} over {b['n_cases']} cases; decision changes with N in {b['n_cases_decision_changes']}",
              f"- caveat: {b['caveat']}"]
    else:
        L.append(f"- {b['note']}")
    L += ["", "## Limits"] + [f"- {x}" for x in rep["limits"]]
    return "\n".join(L) + "\n"


DATASET_CARD = """# Dataset card: {name} ({version})

> Generated from manifest `{sha}`. Counts below are of episodes that EXIST in the manifest, not the plan.

## Contents
- Entries: {n_entries}; by role: {by_role}; by data origin: {by_origin}
- Training-eligible successful demonstrations: {eligible} (plan {planned}; shortfall {shortfall})
- Blocked demonstrations (reason: count): {blocked}
- Split: {split}

## What is captured
Per episode: RGB, native depth and aligned depth (separately), measured follower state, leader measurements, clipped follower
targets, submitted raw Goal_Position values (submitted, not confirmed executed), timestamps with domains, calibration references
and hashes, quality flags. See `docs/DATASET_V0_1.md` and `docs/SCHEMA.md` in the collector.

## What is NOT claimed
- Not every future experiment is supported. Object and tray poses are present only where the operator measured them
  (nullable otherwise); manipulation-data readiness is separate from collector integrity.
- Joint signs, gripper mapping, base alignment and the calibration mismatch are not physically validated by this card.
- Synthetic fixtures and simulated rollouts are never counted as demonstrations.
"""


def render_dataset_card(manifest: dict, split: dict | None = None) -> str:
    inv = manifest["inventory"]
    sp = "none supplied" if not split else f"{split['status']} (actual {split['actual']}, plan {split['plan']})"
    return DATASET_CARD.format(name=manifest.get("dataset_name"), version=manifest.get("dataset_version"),
                               sha=manifest.get("content_sha256"), n_entries=inv["n_entries"], by_role=inv["by_role"],
                               by_origin=inv["by_data_origin"], eligible=inv["training_eligible"],
                               planned=inv["planned_successful_demonstrations"], shortfall=inv["shortfall_vs_plan"],
                               blocked=inv["successful_demonstrations_blocked_by"] or "none", split=sp)


def write_report(out_dir: str | Path, rep: dict, manifest: dict | None = None, split: dict | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "review_report.json", "markdown": out / "review_report.md"}
    paths["json"].write_text(json.dumps(rep, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")
    paths["markdown"].write_text(render_markdown(rep))
    if manifest:
        paths["dataset_card"] = out / "dataset_card.md"
        paths["dataset_card"].write_text(render_dataset_card(manifest, split))
    return {k: str(v) for k, v in paths.items()}
