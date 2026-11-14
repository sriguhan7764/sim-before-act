"""One-command OFFLINE demonstration of the whole software chain. SYNTHETIC data only. Opens no hardware.

    python -m sim_before_act.demo --out /path/to/NEW_DIR

Chain (every artefact is labelled synthetic / smoke-test; nothing here is a thesis finding):
  1. collector: write the SYNTHETIC fixture episodes + manifest + provisional split (collector.act fixture)
  2. collector: validate, inventory, replay offline, export BOTH lightweight profiles
  3. ACT smoke policy (random-init, a handful of CPU steps, ``trained_for_task=false``) -> action segment   [lerobot env]
     fallback (clearly labelled) if that environment is unavailable: an invented fixture segment
  4. Demo A (ACT smoke integration): validate the segment against the action contract, run N seeded MuJoCo rollouts (synthetic demo
     scene), per-criterion evidence, outcome audit and gate. REJECT / insufficient evidence is an honest result here.
  5. trace (npz) + rollout plot (+ rendered contact sheet when a GL backend works)
  6. rollout-budget comparison N=1/5/10 over {policy smoke segment, safe fixture, unsafe invented fixture}
  6b. Demo B: five deterministic SYNTHETIC verification fixtures with independently specified expectations (NOT the ACT policy)
  7. shadow-evaluation record (synthetic integration trial) + operator annotation
  8. review report + dataset card + DEMO_REPORT.md

Hardware boundary: no serial port, camera or robot session is opened; no ``--allow-hardware`` flag is passed anywhere. Offline
``WOULD_EXECUTE`` means the uncalibrated demo gate would pass; nothing is executed or sent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .evaluation.report import build_report, write_report
from .integration.offline_pipeline import budget_study, outcome_audit, shadow_verification_block, verify_segment
from .robot.segment import load_action_segment, write_fixture_segment
from .shadow.workflow import annotate_trial, load_trials, record_prediction
from .twin.scene.builder import build_scene
from .twin.scene.config import load_scene_config
from .twin.scene.viz import plot_rollout, render_contact_sheet
from .util import to_jsonable

PKG_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_ROOT = PKG_ROOT.parent / "robot_collector"
LEROBOT_PY = Path("/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python")
LEROBOT_SRC = Path("/home/apic3/lerobot_v6_2/lerobot/src")
SCENE = PKG_ROOT / "configs" / "scene_synthetic_demo.yaml"

BANNER = ("SYNTHETIC OFFLINE DEMONSTRATION - invented fixture data and a random-init smoke-test policy. Not recorded physical data, "
          "not physically evaluated, not a thesis finding. No hardware was opened.")


def _run(cmd: list[str], env_extra: dict, log: Path, cwd: Path) -> int:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **env_extra}
    if "PYTHONPATH" in env_extra and os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = env_extra["PYTHONPATH"] + os.pathsep + os.environ["PYTHONPATH"]      # keep guards/shims inherited
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(cwd))
    log.write_text(f"$ {' '.join(cmd)}\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}\nexit={p.returncode}\n")
    return p.returncode


def _collector(args: list[str], log: Path) -> int:
    """Collector CLI, run in this environment. Never given hardware flags: the collector defaults to offline."""
    return _run([sys.executable, "-m", "collector", *args], {"PYTHONPATH": str(COLLECTOR_ROOT)}, log, COLLECTOR_ROOT)


def _step(steps: list[dict], name: str, status: str, detail: str, **extra) -> None:
    steps.append({"step": name, "status": status, "detail": detail, **extra})
    print(f"[demo] {name}: {status} - {detail}", flush=True)


def _fixture_cases(cfg, scene) -> list[dict]:
    """Demo B: deterministic SYNTHETIC verification fixtures. ``expect`` is specified HERE, independently of the gate's output,
    from the fixture construction itself (hold the start pose / drive into the mat / invalid start state) and the scene assumptions."""
    q0 = np.asarray(cfg.inputs["initial_robot"]["values"], float)
    hold = np.tile(q0, (40, 1))                               # 40 steps @30 Hz = 1.333 s (+0.5 s settle = 1.834 s rollout)
    unsafe = hold.copy()
    unsafe[20:] = [0.0, 90.0, 0.0, 0.0, 0.0, 40.0]            # INVENTED: drives the arm into the mat -> constraint violation
    return [
        {"id": "B1_hold_pose_safety_only_scope", "segment": ("hold_pose", hold, {}), "kw": {"required_criteria": ("safety",)},
         "scene": "default (task horizon 8.0 s)",
         "why": "arm holds its start pose: no joint-limit or unsafe-contact violation, so safety is evaluable and clean. Task is NOT "
                "evaluated: this is an explicitly safety-only scope, not a full-action approval",
         "expect": {"status": "COMPLETED", "decision": "WOULD_EXECUTE", "approval_scope": "safety_only", "safety_violation": 0,
                    "task_evaluable": 0, "fraction": 0.0}},
        {"id": "B2_hold_pose_task_horizon_reached", "segment": ("hold_pose", hold, {}), "kw": {}, "task_horizon_s": 1.5,
         "scene": "synthetic variant: simulation.task_horizon_s=1.5 (invented) so the 1.83 s rollout reaches the task horizon",
         "why": "arm holds its pose; the cube is never touched, so lift/inside_tray/release_settle are evaluable and false at the task "
                "horizon: task_failure, with no safety violation",
         "expect": {"status": "COMPLETED", "decision": "REJECT", "approval_scope": "none", "safety_violation": 0,
                    "task_failure": "all", "task_evaluable": "all", "fraction": 1.0}},
        {"id": "B3_known_constraint_violation", "segment": ("drive_into_mat", unsafe, {}), "kw": {}, "scene": "default",
         "why": "invented actions drive the arm into the mat: constraint/contact violation is a known safety outcome even though the "
                "task outcome stays unknown",
         "expect": {"status": "COMPLETED", "decision": "REJECT", "approval_scope": "none", "safety_violation": "all",
                    "task_evaluable": 0, "fraction": 1.0}},
        {"id": "B4_insufficient_evidence", "segment": ("hold_pose", hold, {}), "kw": {}, "scene": "default (task horizon 8.0 s)",
         "why": "same safe hold, full (safety+task) scope: safety is clean but the 1.83 s rollout is below the 8.0 s task horizon, so "
                "no task outcome exists for ANY rollout",
         "expect": {"status": "INSUFFICIENT_EVIDENCE", "decision": "REJECT", "approval_scope": "none", "safety_violation": 0,
                    "task_evaluable": 0, "fraction": None}},
        {"id": "B5_all_rollouts_invalid", "segment": ("bad_start_state", hold, {"state_values": [0, -105.0, 30, 20, 0, 30]}),
         "kw": {"init": "segment_state"}, "scene": "default",
         "why": "the segment's start state is outside the simulator joint limits: every rollout is invalid and refused, never clipped",
         "expect": {"status": "INSUFFICIENT_VALID_ROLLOUTS", "decision": "REJECT", "approval_scope": "none", "invalid": "all",
                    "fraction": None}},
    ]


def run_fixture_verification(out: Path, n: int, base_seed: int) -> dict:
    """Demo B: five SYNTHETIC fixtures with independently specified expectations. The ACT policy is NOT involved."""
    out.mkdir(parents=True)
    base_cfg = load_scene_config(SCENE)
    base_scene = build_scene(base_cfg)
    results, ok_all = [], True
    for c in _fixture_cases(base_cfg, base_scene):
        name, acts, kw = c["segment"]
        short = c.get("task_horizon_s")
        cfg = load_scene_config(SCENE, overrides={"simulation.task_horizon_s": short}) if short else base_cfg
        scene = build_scene(cfg, init_state_available=(c["kw"].get("init") == "segment_state")) if short else base_scene
        sp = out / f"{c['id']}_segment.json"
        write_fixture_segment(sp, acts, contract=scene.contract, segment_id=name, **kw)
        r = verify_segment(sp, scene, n, base_seed, out_dir=out / c["id"], **c["kw"])
        (out / c["id"] / "outcome_audit.json").write_text(json.dumps(to_jsonable(outcome_audit(r)), indent=2, sort_keys=True,
                                                                     allow_nan=False) + "\n")
        exp, checks = c["expect"], {}
        sub = lambda v: n if v == "all" else v  # noqa: E731
        checks["status"] = r["status"] == exp["status"]
        checks["decision"] = r["decision"] == exp["decision"]
        checks["approval_scope"] = r["approval_scope"] == exp["approval_scope"]
        checks["fraction"] = (r["simulated_failure_fraction"] == exp["fraction"]) if "fraction" in exp else True
        if "safety_violation" in exp:
            checks["safety_violation"] = r["counts"]["safety_violation"] == sub(exp["safety_violation"])
        if "task_failure" in exp:
            checks["task_failure"] = r["counts"]["task_failure"] == sub(exp["task_failure"])
        if "task_evaluable" in exp:
            checks["task_evaluable"] = r["denominators"]["task_evaluable"] == sub(exp["task_evaluable"]) if r.get("denominators") else False
        if "invalid" in exp:
            checks["invalid"] = r["counts"]["invalid_rollout"] == sub(exp["invalid"])
        ok = all(checks.values())
        ok_all &= ok
        results.append({"case": c["id"], "scene": c["scene"], "why": c["why"], "expected": exp, "observed": {
            "status": r["status"], "decision": r["decision"], "offline_decision_label": r["offline_decision_label"],
            "approval_scope": r["approval_scope"], "counts": r["counts"], "fraction": r["simulated_failure_fraction"],
            "denominators": r.get("denominators"), "reason_codes": r["reason_codes"], "evidence_summary": r.get("evidence_summary")},
            "checks": checks, "expectation_met": ok, "data_origin": "synthetic_fixture"})
    doc = {"label": "SYNTHETIC deterministic verification fixtures (invented actions and an invented task horizon). Not ACT output, "
                    "not recorded data, not physically evaluated. A case meeting its expectation shows the GATE LOGIC, nothing else.",
           "n_rollouts": n, "base_seed": base_seed, "all_expectations_met": ok_all, "cases": results}
    (out / "fixture_verification.json").write_text(json.dumps(to_jsonable(doc), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return doc


def run_demo(out: str | Path, n: int = 10, base_seed: int = 0, render: bool = True, use_act_smoke: bool = True) -> dict:
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite (give a NEW directory)")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    steps: list[dict] = []
    logs = out / "logs"
    logs.mkdir()

    # 1-2. collector side ----------------------------------------------------------------------------------------------------
    fx_dir = out / "01_fixture_episodes"
    rc = _run([sys.executable, "-m", "collector.act", "fixture", "--out", str(fx_dir)], {"PYTHONPATH": str(COLLECTOR_ROOT)},
              logs / "01_fixture.log", COLLECTOR_ROOT)
    if rc != 0:
        raise SystemExit(f"fixture step failed; see {logs / '01_fixture.log'}")
    manifest_p, split_p = fx_dir / "manifest.json", fx_dir / "split.json"
    man, split = json.loads(manifest_p.read_text()), json.loads(split_p.read_text())
    entry = man["entries"][0]["entry_id"]
    _step(steps, "fixture", "OK", f"{len(man['entries'])} SYNTHETIC fixture episodes; roles={sorted({e['role'] for e in man['entries']})}; "
          f"split data_origin={split['data_origin']} frozen={split['frozen']}")
    coll = {}
    for name, args in (("validate", ["dataset", "validate", "--manifest", str(manifest_p)]),
                       ("inventory", ["dataset", "inventory", "--manifest", str(manifest_p)]),
                       ("replay", ["dataset", "replay", "--manifest", str(manifest_p), "--entry", entry]),
                       ("export_rgb_joint_angles", ["dataset", "export", "--manifest", str(manifest_p), "--entry", entry,
                                                    "--profile", "rgb_joint_angles"]),
                       ("export_rgbd_motor_positions", ["dataset", "export", "--manifest", str(manifest_p), "--entry", entry,
                                                        "--profile", "rgbd_motor_positions"])):
        coll[name] = _collector(args, logs / f"02_{name}.log")
    bad = [k for k, v in coll.items() if v != 0]
    _step(steps, "collector_validate_replay_export", "OK" if not bad else "FAILED",
          "validate, inventory, offline replay and both export profiles on a SYNTHETIC fixture episode" if not bad else f"failed: {bad}",
          exit_codes=coll)

    # 3. policy segment ------------------------------------------------------------------------------------------------------
    cfg = load_scene_config(SCENE)
    scene = build_scene(cfg)
    seg_path, policy_source = None, None
    if use_act_smoke and LEROBOT_PY.exists() and LEROBOT_SRC.exists():
        rc = _run([str(LEROBOT_PY), "-m", "collector.act", "smoke", "--out", str(out / "03_act_smoke")],
                  {"PYTHONPATH": f"{LEROBOT_SRC}{os.pathsep}{COLLECTOR_ROOT}"}, logs / "03_act_smoke.log", COLLECTOR_ROOT)
        cand = out / "03_act_smoke" / "action_segment.json"
        if rc == 0 and cand.exists():
            seg_path, policy_source = cand, "ACT smoke test (random-init, few CPU steps, trained_for_task=false)"
        else:
            _step(steps, "act_smoke", "FAILED", f"see {logs / '03_act_smoke.log'}; falling back to an invented fixture segment")
    if seg_path is None:
        q0 = np.asarray(cfg.inputs["initial_robot"]["values"], float)
        a = np.tile(q0, (40, 1))
        a[:, 0] += 8.0 * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, 40)))
        seg_path = out / "03_fixture_segment" / "action_segment.json"
        seg_path.parent.mkdir()
        write_fixture_segment(seg_path, a, contract=scene.contract, segment_id="demo_fallback_fixture_segment")
        policy_source = "INVENTED fixture actions (ACT smoke unavailable); not policy output"
    seg = load_action_segment(seg_path)
    _step(steps, "policy_segment", "OK", policy_source, segment=str(seg_path), policy_kind=seg.raw["policy"]["kind"],
          trained_for_task=seg.raw["policy"]["trained_for_task"])

    # 4-5. verification ------------------------------------------------------------------------------------------------------
    vdir = out / "04_verification"
    res = verify_segment(seg_path, scene, n, base_seed, out_dir=vdir, keep_first_result=True)
    first = res.pop("_first_result", None)
    viz = {"plot": None, "contact_sheet": None}
    if first is not None:
        viz["plot"] = str(plot_rollout(first, vdir / "rollout0_plot.png"))
        if render:
            cs = render_contact_sheet(first, cfg, vdir, "rollout0", video=False, timeout_s=90.0)
            viz["contact_sheet"] = {"renderer": cs["renderer"], "png": None if cs["png"] is None else str(cs["png"]),
                                    "attempts": cs["attempts"]}
    (vdir / "verification.json").write_text(json.dumps(to_jsonable(res), indent=2, sort_keys=True, allow_nan=False) + "\n")
    audit = outcome_audit(res)
    (vdir / "outcome_audit.json").write_text(json.dumps(to_jsonable(audit), indent=2, sort_keys=True, allow_nan=False) + "\n")
    _step(steps, "A_act_smoke_verification", "OK",
          f"integration ran (policy: {policy_source}). Gate status={res['status']} decision={res['decision']} "
          f"({res['offline_decision_label']}, scope={res['approval_scope']}); {res['evidence_summary']}; reasons={res['reason_codes']}. "
          "A rejection or insufficient-evidence result here is an honest integration outcome, not a step failure",
          gate_status=res["status"], visualisation=viz)

    # 6. budget study --------------------------------------------------------------------------------------------------------
    q0 = np.asarray(cfg.inputs["initial_robot"]["values"], float)
    safe = np.tile(q0, (40, 1))
    unsafe = safe.copy()
    unsafe[20:] = [0.0, 90.0, 0.0, 0.0, 0.0, 40.0]
    cdir = out / "05_budget_cases"
    cdir.mkdir()
    write_fixture_segment(cdir / "fixture_safe.json", safe, contract=scene.contract, segment_id="fixture_safe")
    write_fixture_segment(cdir / "fixture_unsafe_invented.json", unsafe, contract=scene.contract, segment_id="fixture_unsafe_invented")
    cases = {"policy_segment": seg_path, "fixture_safe": cdir / "fixture_safe.json", "fixture_unsafe_invented": cdir / "fixture_unsafe_invented.json"}
    rows = budget_study(cases, scene, (1, 5, 10), base_seed)
    (out / "05_budget_rows.json").write_text(json.dumps(to_jsonable(rows), indent=2, sort_keys=True, allow_nan=False) + "\n")
    _step(steps, "budget_study", "OK", "N=1/5/10, same cases and base seed, seeds nested", rows=len(rows))

    # 6b. Demo B: deterministic synthetic verification fixtures (NOT the ACT policy) ---------------------------------------------
    fxv = run_fixture_verification(out / "05b_synthetic_verification_fixtures", n, base_seed)
    _step(steps, "B_synthetic_verification_fixtures", "OK" if fxv["all_expectations_met"] else "FAILED",
          "; ".join(f"{c['case']}: {c['observed']['status']}/{c['observed']['decision']}"
                    f"{'' if c['expectation_met'] else ' (EXPECTATION NOT MET)'}" for c in fxv["cases"])
          + ". SYNTHETIC fixtures, not ACT output", fixture_file=str(out / "05b_synthetic_verification_fixtures" / "fixture_verification.json"))

    # 7. shadow record -------------------------------------------------------------------------------------------------------
    sroot = out / "06_shadow"
    r = seg.raw
    record_prediction(
        sroot, trial_id="SYNTHETIC_demo_trial_001", data_origin="synthetic_integration_test",
        provenance_label="SYNTHETIC integration-test trial: offline demo; not a physical result",
        initial_state={"state_values": r["input"]["state_values"], "state_sha256": r["input"]["state_sha256"],
                       "state_timestamp": r["created_wall_ns"], "clock_domain": "wall_ns of SYNTHETIC segment creation (no real state)"},
        action={"segment_id": r["segment_id"], "action_id": res["segment"]["action_sha256"][:16], "action_sha256": res["action_sha256"],
                "chunk_length": res["chunk_length"]},
        policy={"kind": r["policy"]["kind"], "checkpoint_id": r["policy"]["checkpoint_id"], "trained_for_task": False},
        verification=shadow_verification_block(res),
        execution_horizon={"type": "segment", "steps": res["executed_steps"]},
        timings={"verification_wall_s": res["wall_s"]}, hardware={"hardware_used": False})
    annotate_trial(sroot, "SYNTHETIC_demo_trial_001", execution="not_executed", annotated_by="demo script (synthetic)",
                   notes="offline demo: nothing was executed; there is no observed outcome")
    _step(steps, "shadow_record", "OK", "synthetic prediction persisted + 'not_executed' annotation; no outcome exists")

    # 8. reports -------------------------------------------------------------------------------------------------------------
    trials = load_trials(sroot)
    rep = build_report(manifest=man, split=split, physical_trials=[t for t in trials if t["physical"]],
                       synthetic_trials=[t for t in trials if not t["physical"]], budget_rows=rows)
    paths = write_report(out / "07_report", rep, man, split)
    _step(steps, "report", "OK", f"review report + dataset card written ({paths['markdown']})")

    summary = {"banner": BANNER, "output_dir": str(out), "wall_s": time.monotonic() - t0, "n_rollouts": n, "base_seed": base_seed,
               "data_origin": "synthetic", "recorded_physical_data_used": False, "physically_evaluated": False,
               "policy": policy_source, "verification_decision": res["decision"],
               "offline_decision_label": res["offline_decision_label"], "approval_scope": res["approval_scope"],
               "gate_status": res["status"], "evidence_summary": res["evidence_summary"], "denominators": res["denominators"],
               "decision_meaning": "WOULD_EXECUTE / WOULD_REJECT only; nothing executed",
               "fixture_verification_all_expectations_met": fxv["all_expectations_met"],
               "counts": res["counts"], "visualisation": viz, "steps": steps, "hardware": {"serial_opened": False, "cameras_opened": False,
                                                                                         "robot_session": False}}
    (out / "demo_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "DEMO_REPORT.md").write_text(_demo_markdown(summary, res, rows, fxv))
    return summary


def _demo_markdown(s: dict, res: dict, rows: list[dict], fxv: dict) -> str:
    dn = res["denominators"]
    L = [f"# Offline demo report", "", f"> **{BANNER}**", "",
         f"- Data origin: **synthetic** (fixture episodes written by `collector.act.synthetic`; invented values).",
         f"- Scene: `{res['scene_label']}`.", "",
         "## A. ACT smoke integration (honest integration test)", "",
         f"- Policy: {s['policy']}. `trained_for_task` is **false**; this is not a trained policy and says nothing about ACT performance.",
         f"- Gate status **{res['status']}**, decision **{res['decision']}** (offline label {res['offline_decision_label']}, approval scope "
         f"`{res['approval_scope']}`). Reasons: {res['reason_codes']}. Nothing was executed or sent; threshold {res['threshold']} and "
         f"coverage requirement {res['min_valid_fraction']} are UNCALIBRATED.",
         f"- Rollouts: N={res['n_rollouts']}, base seed {res['base_seed']}; counts {res['counts']}.",
         f"- Evidence: {res['evidence_summary']}.",
         f"- Denominators: {dn}. Risk estimate: {res['simulated_failure_fraction']} (None = withheld, not zero). "
         f"Interval: {res['interval']}.",
         f"- Horizon: {res['horizon_claim']}",
         "- A completed simulation is not a valid outcome observation. Safety was evaluable for every rollout; the task outcome was "
         "not evaluable because the rollout is shorter than the assumed task horizon. Per-rollout reasons: "
         "`04_verification/outcome_audit.json`.", "",
         "## B. Deterministic SYNTHETIC verification fixtures (NOT the ACT policy)", "",
         f"> {fxv['label']}", "", f"All expectations met: **{fxv['all_expectations_met']}**", "",
         "| case | scene | expected status/decision/scope | observed status/decision/scope | counts | fraction | met |",
         "|---|---|---|---|---|---|---|"]
    for c in fxv["cases"]:
        e, o = c["expected"], c["observed"]
        L.append(f"| {c['case']} | {c['scene']} | {e['status']} / {e['decision']} / {e['approval_scope']} | "
                 f"{o['status']} / {o['decision']} / {o['approval_scope']} | {o['counts']} | {o['fraction']} | {c['expectation_met']} |")
    L += ["", "Why each fixture is expected to behave so:", ""] + [f"- **{c['case']}**: {c['why']}" for c in fxv["cases"]]
    L += ["", "## Steps", ""]
    for st in s["steps"]:
        L.append(f"- **{st['step']}** - {st['status']}: {st['detail']}")
    L += ["", "## Rollout-budget comparison (SYNTHETIC cases, default full scope)", "",
          "| case | N | failure fraction | status | decision | counts |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['case_id']} | {r['n_rollouts']} | {r['simulated_failure_fraction']} | {r['status']} | {r['decision']} | {r['counts']} |")
    L += ["", "## Not claimed", "",
          "- No recorded physical demonstration was used; no physical trial exists; nothing was physically evaluated.",
          "- The simulator is CAD geometry with invented scene values, not a physically calibrated twin; the calibration mismatch "
          "remains a blocker.",
          "- A random-init ACT smoke checkpoint says nothing about ACT performance. ACT is a pilot policy, not a language-conditioned VLA.",
          "- Fixture cases demonstrate gate logic only. No fixture result is ACT task success or a prediction of physical outcome.",
          "- `unknown` is never counted as success or zero risk; WOULD_EXECUTE needs evaluable results for every required criterion.",
          "", "## Outputs", "", "- `01_fixture_episodes/` SYNTHETIC masters, manifest, split (fixture split, never frozen)",
          "- `04_verification/` Demo A: verification.json, outcome_audit.json, per-rollout traces, plot, contact sheet when a GL backend worked",
          "- `05_budget_rows.json`, `05b_synthetic_verification_fixtures/` (Demo B), `06_shadow/`, `07_report/`, `logs/`", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_before_act.demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="NEW output directory (never overwritten)")
    ap.add_argument("-n", "--n-rollouts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true", help="skip the rendered contact sheet (plots are still written)")
    ap.add_argument("--no-act-smoke", action="store_true", help="skip the lerobot-env ACT smoke test; use an invented fixture segment")
    a = ap.parse_args(argv)
    s = run_demo(a.out, a.n_rollouts, a.seed, render=not a.no_render, use_act_smoke=not a.no_act_smoke)
    print(f"\n{BANNER}\nreport: {Path(a.out) / 'DEMO_REPORT.md'}")
    return 0 if all(st["status"] in ("OK", "COMPLETED") for st in s["steps"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
