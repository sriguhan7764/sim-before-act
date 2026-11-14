"""Offline verification of one fixed action segment: N seeded scene rollouts -> counts, failure fraction, interval, decision.

record/state -> ACT inference (separate env; produces the segment file) -> THIS MODULE -> trace + visualisation.

Semantics (documented, uncalibrated). A COMPLETED SIMULATION IS NOT A VALID OUTCOME OBSERVATION.
* Every rollout lands in exactly one category: task_success, task_failure, safety_violation, unknown, timeout, invalid_rollout.
  Categories are never merged away: they are all reported.
* Two criteria have SEPARATE availability, counted per rollout:
    safety evaluable : the rollout ran validly (not invalid, not timed out). Violated or clean, the safety verdict is known.
    task   evaluable : task_outcome is task_success or task_failure. ``unknown`` (segment horizon shorter than the assumed task
                       horizon, so unmet predicates are inconclusive; or invalid/timeout) is NOT evaluable.
* A rollout is DETERMINED when its verdict for the REQUIRED criteria is known: for required ("safety","task") that is a known safety
  violation, or clean safety AND a known task outcome; for required ("safety",) it is any valid rollout.
* ``simulated_failure_fraction`` = (safety_violation + task_failure) / DETERMINED rollouts, reported ONLY when evidence coverage is
  sufficient (status COMPLETED). Unknown rollouts are never counted as success, never as zero risk, and never silently dropped to
  make the remainder stand for all N: coverage (determined / N) is reported and enforced, and every denominator is listed under
  ``denominators``. If coverage is short, the fraction is None and the subset value is only given as ``partial_fraction_over_determined``.
* Gate (fail-closed): WOULD_EXECUTE requires status COMPLETED, i.e. determined >= ceil(min_valid_fraction * N) AND fraction < threshold.
  Otherwise REJECT (offline label WOULD_REJECT) with reason codes; zero evaluable outcomes for a mandatory criterion gives
  INSUFFICIENT_EVIDENCE and ``task_outcome_unavailable`` / ``safety_outcome_unavailable``.
* ``required_criteria`` defaults to ("safety","task"). ("safety",) is an explicit, labelled SAFETY-ONLY scope: its WOULD_EXECUTE
  is recorded with ``approval_scope: safety_only`` and is NOT a full-action approval. Safety can never be dropped.
* The Wilson interval covers Monte-Carlo sampling error only (not simulator model error or sim-to-real gap).
* Threshold and min_valid_fraction are UNCALIBRATED placeholders, and are not relaxed in demo mode.
* Seeds are SeedSequence(base_seed).spawn(N)[i], so the rollouts for N=1, 5, 10 are nested prefixes of one sequence.
No hardware module is imported or touched.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ..robot.segment import SegmentInvalid, load_action_segment, validate_for_sim
from ..twin.scene.builder import build_scene
from ..twin.scene.config import SceneRefusal, load_scene_config
from ..util import hash_obj, to_jsonable
from ..verification.estimator import wilson_interval

SCHEMA = "offline-verification-v0.1"
CATEGORIES = ("task_success", "task_failure", "safety_violation", "unknown", "timeout", "invalid_rollout")
DEFAULT_THRESHOLD = 0.3
DEFAULT_MIN_VALID_FRACTION = 0.8
UNCALIBRATED = "UNCALIBRATED placeholder: not derived from physical data"


def rollout_seeds(base_seed: int, n: int) -> list[int]:
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(int(base_seed)).spawn(int(n))]


ALLOWED_REQUIRED = (("safety", "task"), ("safety",))


def decide(status: str, fraction: float | None, threshold: float) -> tuple[str, list[str]]:
    if status != "COMPLETED":
        return "REJECT", [f"status_{status}"]
    if fraction is None:
        return "REJECT", ["no_valid_rollouts"]
    if fraction >= threshold:
        return "REJECT", ["risk_at_or_above_threshold"]
    return "WOULD_EXECUTE", ["risk_below_threshold"]


def assess_rollouts(rows: list[dict], n: int, *, threshold: float, min_valid_fraction: float,
                    required: tuple[str, ...] = ("safety", "task")) -> dict:
    """Pure evidence/gate logic over per-rollout rows (category, task_outcome). No simulator access, so it is directly testable.

    Returns counts, per-criterion availability, denominators, status, decision and reason codes. See the module docstring.
    """
    required = tuple(required)
    if required not in ALLOWED_REQUIRED:
        raise ValueError(f"required_criteria must be one of {ALLOWED_REQUIRED}; safety can never be dropped")
    need = math.ceil(min_valid_fraction * n)
    cat = lambda r: r["category"]  # noqa: E731
    invalid = sum(cat(r) == "invalid_rollout" for r in rows)
    timeouts = sum(cat(r) == "timeout" for r in rows)
    valid = [r for r in rows if cat(r) not in ("invalid_rollout", "timeout")]
    sv = sum(cat(r) == "safety_violation" for r in valid)
    t_known = [r for r in valid if r["task_outcome"] in ("task_success", "task_failure")]
    t_fail = sum(r["task_outcome"] == "task_failure" for r in t_known)
    t_succ = sum(r["task_outcome"] == "task_success" for r in t_known)
    if "task" in required:
        determined = [r for r in valid if cat(r) == "safety_violation" or r["task_outcome"] in ("task_success", "task_failure")]
    else:
        determined = valid
    failed = sum(cat(r) == "safety_violation" or r["task_outcome"] == "task_failure" for r in determined) if "task" in required \
        else sv
    d = len(determined)
    iv = lambda k, m: None if m == 0 else dict(zip(("lower", "upper"), wilson_interval(k, m)),  # noqa: E731
                                              method="wilson", confidence=0.95,
                                              scope="Monte-Carlo sampling error only; excludes simulator model error")
    criteria = {
        "safety": {"evaluable": len(valid), "of": n, "coverage": len(valid) / n, "available": len(valid) > 0, "violations": sv,
                   "violation_fraction": (sv / len(valid)) if valid else None, "interval": iv(sv, len(valid)),
                   "meaning": "violation of constraints / unsafe contacts among validly-run rollouts"},
        "task": {"evaluable": len(t_known), "of": n, "coverage": len(t_known) / n, "available": len(t_known) > 0,
                 "success": t_succ, "failure": t_fail, "unknown": len(valid) - len(t_known),
                 "failure_fraction": (t_fail / len(t_known)) if t_known else None, "interval": iv(t_fail, len(t_known)),
                 "meaning": "task_success / task_failure; unknown (segment horizon shorter than task horizon, or invalid/timeout) "
                            "is NOT evaluable"}}
    reasons: list[str] = []
    if len(valid) < need or not valid:
        status = "INSUFFICIENT_VALID_ROLLOUTS"
    elif d < need or d == 0:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "COMPLETED"
    if status != "COMPLETED":
        for crit in required:
            c = criteria[crit]
            if c["evaluable"] == 0:
                reasons.append(f"{crit}_outcome_unavailable")
            elif c["evaluable"] < need:
                reasons.append(f"{crit}_evidence_below_required_coverage")
        reasons.append(f"status_{status}")
    frac = (failed / d) if (status == "COMPLETED" and d) else None
    decision, why = decide(status, frac, threshold) if status == "COMPLETED" else ("REJECT", reasons)
    if sv and "safety_violation_observed" not in why:
        why = list(why) + ["safety_violation_observed"]
    interval = iv(failed, d) if frac is not None else None
    scope = "none" if decision != "WOULD_EXECUTE" else ("full" if "task" in required else "safety_only")
    return {"status": status, "decision": decision, "offline_decision_label": "WOULD_EXECUTE" if decision == "WOULD_EXECUTE" else "WOULD_REJECT",
            "reason_codes": why, "approval_scope": scope, "required_criteria": list(required),
            "ran_validly": len(valid), "determined": d, "failed": failed, "simulated_failure_fraction": frac, "interval": interval,
            "partial_fraction_over_determined": (failed / d) if (frac is None and d) else None,
            "coverage": {"determined_over_all_rollouts": d / n, "required_minimum": min_valid_fraction, "required_count": need,
                         "note": "UNCALIBRATED placeholder; unknown/invalid/timeout rollouts are never counted as success or zero risk"},
            "denominators": {"all_rollouts": n, "valid": len(valid), "safety_evaluable": len(valid), "task_evaluable": len(t_known),
                             "determined": d, "used_for_fraction": "determined" if frac is not None else "none (fraction withheld)",
                             "excluded_from_fraction": {"invalid": invalid, "timeout": timeouts,
                                                        "undetermined_valid": len(valid) - d}},
            "criteria": criteria,
            "evidence_summary": (f"safety evaluable {len(valid)}/{n} ({sv} violating); task evaluable {len(t_known)}/{n} "
                                 f"({t_succ} success, {t_fail} failure, {len(valid) - len(t_known)} unknown); "
                                 f"determined for required {list(required)}: {d}/{n}; need >= {need}")}


def _empty_counts() -> dict:
    return {c: 0 for c in CATEGORIES}


def verify_segment(segment, scene, n: int, base_seed: int = 0, *, out_dir: str | Path | None = None, threshold: float = DEFAULT_THRESHOLD,
                   min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION, init: str = "scene", save_traces: bool = True,
                   keep_first_result: bool = False, required_criteria: tuple[str, ...] = ("safety", "task")) -> dict:
    """Run ``n`` seeded rollouts of ``segment`` (path or ActionSegment) in ``scene`` (path/SceneConfig/Scene).

    ``required_criteria`` are the mandatory verification criteria; the default requires BOTH safety and task evidence.
    ("safety",) is an explicit safety-only scope whose WOULD_EXECUTE is not a full-action approval.
    """
    if tuple(required_criteria) not in ALLOWED_REQUIRED:
        raise ValueError(f"required_criteria must be one of {ALLOWED_REQUIRED}")
    from ..twin.scene.builder import Scene
    from ..twin.scene.runner import run_scene_rollout

    if n < 1:
        raise ValueError("n must be >= 1")
    t0 = time.monotonic()
    cfg = scene.cfg if isinstance(scene, Scene) else (scene if hasattr(scene, "raw") else load_scene_config(scene))
    base = {"schema": SCHEMA, "n_rollouts": n, "base_seed": base_seed, "threshold": threshold, "threshold_calibrated": False,
            "threshold_note": UNCALIBRATED, "min_valid_fraction": min_valid_fraction, "min_valid_note": UNCALIBRATED,
            "required_criteria": list(required_criteria), "approval_scope": "none", "offline_decision_label": "WOULD_REJECT",
            "scene_label": cfg.label, "scene_mode": cfg.mode}
    try:
        sc = scene if isinstance(scene, Scene) else build_scene(cfg, init_state_available=(init == "segment_state"))
    except SceneRefusal as e:
        return {**base, "status": "REFUSED_INPUTS", "decision": "REJECT", "reason_codes": ["scene_refused"], "refusal": list(e.missing),
                "counts": _empty_counts(), "simulated_failure_fraction": None, "interval": None, "config_hash": None,
                "rollouts": [], "wall_s": time.monotonic() - t0}
    seg = load_action_segment(segment) if isinstance(segment, (str, Path)) else segment
    try:
        vs = validate_for_sim(seg, sc.contract)
    except SegmentInvalid as e:
        return {**base, "status": "INVALID_INPUT", "decision": "REJECT", "reason_codes": ["segment_invalid"], "problems": e.problems,
                "counts": _empty_counts(), "simulated_failure_fraction": None, "interval": None, "config_hash": None,
                "rollouts": [], "wall_s": time.monotonic() - t0}
    pert = cfg.perturbation_cfg() or None
    config_hash = hash_obj({"scene": cfg.raw, "contract_id": sc.contract.contract_id, "n": n, "base_seed": base_seed, "init": init,
                            "rule": {"threshold": threshold, "min_valid_fraction": min_valid_fraction,
                                     "required_criteria": list(required_criteria)},
                            "perturbation": pert})
    counts, rows, first = _empty_counts(), [], None
    out = Path(out_dir) if out_dir else None
    for i, seed in enumerate(rollout_seeds(base_seed, n)):
        res = run_scene_rollout(sc, vs, seed=seed, perturbation=pert, init=init)
        counts[res.category] = counts.get(res.category, 0) + 1
        o = res.outcome
        valid_run = res.category not in ("invalid_rollout", "timeout")
        rows.append({"index": i, "seed": seed, "category": res.category, "task_outcome": o["task_outcome"],
                     "safety_reasons": o["safety_reasons"], "invalid_reason": o["invalid_reason"],
                     "safety_evaluable": valid_run, "task_evaluable": valid_run and o["task_outcome"] in ("task_success", "task_failure"),
                     "horizon_kind": o.get("horizon_kind"), "horizon_s": o.get("horizon_s"), "task_horizon_s": o.get("task_horizon_s"),
                     "predicates_observed_true": o.get("predicates_observed_true"),
                     "predicates_not_evaluable": o.get("predicates_not_evaluable"),
                     "predicates_unmet_but_inconclusive": o.get("predicates_unmet_but_inconclusive"), "outcome_note": o.get("note"),
                     "wall_s": res.wall_s})
        if first is None:
            first = res
        if out and save_traces:
            res.save(out / "rollouts", f"rollout{i:03d}_seed{seed}")
    ass = assess_rollouts(rows, n, threshold=threshold, min_valid_fraction=min_valid_fraction, required=tuple(required_criteria))
    result = {**base, **ass, "counts": counts,
              "config_hash": config_hash, "segment": vs.provenance(), "action_sha256": vs.action_sha256,
              "contract_id": sc.contract.contract_id, "chunk_length": int(vs.actions.shape[0]),
              "executed_steps": int(vs.execution_horizon_steps), "rollouts": rows,
              "horizon_claim": "segment-horizon safety/constraint observations only; NOT a full-task success prediction",
              "meaning": ("offline decision: WOULD_EXECUTE means the uncalibrated gate would pass for the stated approval_scope; nothing was "
                          "executed or sent. WOULD_REJECT (decision REJECT) includes insufficient evidence. A completed simulation is not "
                          "a valid outcome observation: see criteria/denominators and sim_before_act/integration/offline_pipeline.py."),
              "wall_s": time.monotonic() - t0}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "verification.json").write_text(json.dumps(to_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    if keep_first_result:
        result["_first_result"] = first
    return result


def budget_study(cases: dict[str, str | Path], scene, ns=(1, 5, 10), base_seed: int = 0, **kw) -> list[dict]:
    """Same cases at each N, same base seed. Returns rows for evaluation.report.budget_comparison."""
    rows = []
    for cid, seg in sorted(cases.items()):
        for n in ns:
            r = verify_segment(seg, scene, n, base_seed, save_traces=False, **kw)
            rows.append({"case_id": cid, "n_rollouts": n, "simulated_failure_fraction": r["simulated_failure_fraction"],
                         "interval": r["interval"], "status": r["status"], "decision": r["decision"], "counts": r["counts"],
                         "approval_scope": r.get("approval_scope", "none"), "reason_codes": r["reason_codes"],
                         "denominators": r.get("denominators"), "wall_s": r["wall_s"]})
    return rows


def shadow_verification_block(result: dict) -> dict:
    """Map a verify_segment result to the ``verification`` block expected by shadow.workflow.record_prediction."""
    return {"config_hash": result["config_hash"], "status": result["status"], "decision": result["decision"],
            "offline_decision_label": result.get("offline_decision_label"), "approval_scope": result.get("approval_scope", "none"),
            "required_criteria": result.get("required_criteria"), "denominators": result.get("denominators"),
            "coverage": result.get("coverage"), "evidence_summary": result.get("evidence_summary"),
            "reason_codes": result["reason_codes"], "n_rollouts": result["n_rollouts"], "base_seed": result["base_seed"],
            "simulated_failure_fraction": result["simulated_failure_fraction"], "interval": result["interval"],
            "counts": {"task_failures": result["counts"]["task_failure"], "safety_violations": result["counts"]["safety_violation"],
                       "invalid": result["counts"]["invalid_rollout"], "timeouts": result["counts"]["timeout"],
                       "unknown": result["counts"]["unknown"], "task_success": result["counts"]["task_success"]},
            "threshold": result["threshold"], "threshold_calibrated": False, "horizon_claim": result["horizon_claim"]}


def outcome_audit(result: dict) -> dict:
    """Explain, per rollout, why each outcome is what it is. Derived only from a verify_segment result (no re-simulation)."""
    per = []
    for r in result.get("rollouts", []):
        done = r["category"] not in ("invalid_rollout", "timeout")
        safety = ("not_evaluable" if not done else ("violated" if r["category"] == "safety_violation" else "clean"))
        task = r["task_outcome"] if r["task_evaluable"] else "unknown (not evaluable)"
        why = None
        if r["task_outcome"] == "unknown":
            if not done:
                why = f"rollout {r['category']}: {r['invalid_reason'] or 'timed out'}"
            else:
                why = (f"horizon_kind={r['horizon_kind']} (rollout {r['horizon_s']:.3f} s vs assumed task horizon {r['task_horizon_s']} s); "
                       f"predicates unmet but inconclusive: {r['predicates_unmet_but_inconclusive']}; "
                       f"not evaluable: {r['predicates_not_evaluable']}")
        per.append({"index": r["index"], "seed": r["seed"], "simulation_completed": done, "category": r["category"],
                    "safety_status": safety, "safety_reasons": r["safety_reasons"], "task_status": task, "why_task_unknown": why})
    return {"predicates_evaluated": ["constraints", "contacts", "lift", "inside_tray", "release_settle"],
            "n": result.get("n_rollouts"), "simulation_completed": sum(p["simulation_completed"] for p in per), "per_rollout": per,
            "availability": {"safety": (result.get("criteria") or {}).get("safety"), "task": (result.get("criteria") or {}).get("task")},
            "denominators": result.get("denominators"), "status": result.get("status"), "decision": result.get("decision"),
            "offline_decision_label": result.get("offline_decision_label"), "approval_scope": result.get("approval_scope"),
            "reason_codes": result.get("reason_codes"), "evidence_summary": result.get("evidence_summary")}
