"""Command line interface.

  python -m sim_before_act verify --config CFG --record REC [--n N] [--out DIR]
  python -m sim_before_act import-lerobot --dataset DIR [--episode 0] [--start 0] [--horizon 30] [--sidecar JSON] --out REC.json
  python -m sim_before_act scene-check [--scene CFG] [--segment SEG.json]            (default scene: SYNTHETIC DEMO; physical refuses, exit 3)
  python -m sim_before_act scene-rollout [--scene CFG] (--segment SEG.json | --fixture zero|small_motion) [--seed N] [--perturb]
                                         [--init scene|segment_state] --out DIR [--render] [--video]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .schemas.record import RecordError, load_record
from .util import write_json_exclusive

DEFAULT_SCENE = str(Path(__file__).resolve().parents[2] / "configs" / "scene_synthetic_demo.yaml")


def _cmd_verify(a) -> int:
    from .runtime.offline import run_attempt

    cfg = load_config(a.config, {"verification.n_rollouts": a.n, "verification.allow_nonstandard_budget": True} if a.n else None)
    rec = load_record(a.record)
    if rec.synthetic:
        print(f"*** SYNTHETIC RECORD: {rec.synthetic_label or 'no label'} ***")
        print("*** Not evidence of SO-101 twin fidelity or real-world predictive validity. ***")
    res = run_attempt(cfg, rec, Path(a.out))
    v, d = res.verification, res.decision
    print(f"record       : {rec.record_id}")
    print(f"status       : {v.status}")
    if v.problems:
        for p in v.problems:
            print(f"  problem    : {p}")
    est = v.estimate
    if est:
        print(f"rollouts     : requested={est.requested} attempted={est.attempted} valid={est.valid} failed={est.failed} "
              f"(task={est.task_failures}, safety={est.safety_violations}, sim_err={est.simulator_errors}, "
              f"invalid_init={est.invalid_init}, timeout={est.timeouts})")
    f = v.simulated_failure_fraction
    print(f"sim failure fraction : {'unavailable' if f is None else f'{f:.3f}'}"
          + (f"  Wilson {v.interval['confidence']:.0%} CI [{v.interval['low']:.3f}, {v.interval['high']:.3f}] (Monte-Carlo only)" if v.interval else ""))
    print(f"decision     : {d.decision}  ({d.reason})")
    print(f"executed     : {res.execution.executed}  (offline: no adapter; EXECUTE means 'would execute')")
    print(f"timing       : verification {v.timings['verification_total_s']:.3f}s, end-to-end {res.trace['verification']['timings_s']['end_to_end_s']:.3f}s")
    print(f"trace        : {res.trace_path}")
    return 0


def _cmd_import(a) -> int:
    from .data.lerobot_import import import_lerobot

    d = import_lerobot(a.dataset, a.episode, a.start, a.horizon, a.sidecar)
    write_json_exclusive(Path(a.out), d, readonly=False)
    print(f"wrote record {a.out} (read-only import; dataset untouched)")
    return 0


def _cmd_scene_check(a) -> int:
    from .twin.scene.builder import build_scene
    from .twin.scene.config import SceneRefusal, load_scene_config

    cfg = load_scene_config(a.scene)
    print(f"scene config : {cfg.path}")
    print(f"mode         : {cfg.mode}")
    print(f"label        : {cfg.label}")
    try:
        scene = build_scene(cfg, init_state_available=a.segment is not None)
    except SceneRefusal as e:
        print("REFUSED: physical mode needs the following before it can run:")
        for m in e.missing:
            print(f"  missing    : {m}")
        return 3
    d = scene.describe()
    print(f"built        : ngeom={d['ngeom']} nbody={d['nbody']} nu={d['nu']} timestep={d['timestep_s']}s")
    print(f"contract_id  : {d['contract_id']}")
    print(f"controller   : {d['controller']}")
    if a.segment:
        from .robot.segment import load_action_segment, validate_segment

        seg = load_action_segment(a.segment)
        arr, problems = validate_segment(seg, scene.contract)
        print(f"segment      : {seg.segment_id}  valid={arr is not None}")
        for p in problems:
            print(f"  problem    : {p}")
        return 0 if arr is not None else 2
    return 0


def _cmd_scene_rollout(a) -> int:
    from .twin.scene.builder import build_scene
    from .twin.scene.config import SceneRefusal, load_scene_config
    from .twin.scene.fixtures import FIXTURE_LABEL, make_fixture
    from .twin.scene.runner import run_scene_rollout
    from .twin.scene.viz import plot_rollout, render_contact_sheet

    cfg = load_scene_config(a.scene)
    print(f"label        : {cfg.label}")
    try:
        scene = build_scene(cfg, init_state_available=(a.init == "segment_state"))
    except SceneRefusal as e:
        print("REFUSED: physical mode needs the following before it can run:")
        for m in e.missing:
            print(f"  missing    : {m}")
        return 3
    if (a.segment is None) == (a.fixture is None):
        raise ValueError("give exactly one of --segment or --fixture")
    if a.fixture:
        from .robot.segment import write_fixture_segment

        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        seg_path = out / f"fixture_{a.fixture}.json"
        write_fixture_segment(seg_path, make_fixture(cfg, a.fixture), contract=scene.contract, segment_id=f"fixture_{a.fixture}",
                              provenance_label=FIXTURE_LABEL.format(name=a.fixture))
    else:
        seg_path = Path(a.segment)
    res = run_scene_rollout(scene, seg_path, seed=a.seed, perturbation=cfg.perturbation_cfg() if a.perturb else None, init=a.init)
    o = res.outcome
    print(f"outcome      : {o['category']}  (task_outcome={o['task_outcome']}, horizon={o['horizon_kind']} {o['horizon_s']:.2f}s)")
    print(f"note         : {o['note']}")
    for key in ("task_failure_reasons", "safety_reasons", "invalid_reason", "predicates_observed_true", "predicates_unmet_but_inconclusive",
                "predicates_not_evaluable"):
        if o.get(key):
            print(f"  {key}: {o[key]}")
    print(f"action sha256: {res.action_record.get('action_sha256')}  contract_id={res.action_record.get('contract_id')}")
    print(f"segment      : {res.labels['segment']}")
    stem = f"rollout_{Path(seg_path).stem}_seed{a.seed}"
    files = res.save(a.out, stem)
    print(f"summary      : {files['summary']}")
    if a.render:
        plot = plot_rollout(res, Path(a.out) / f"{stem}_plot.png")
        print(f"plot (cv2)   : {plot}")
        vis = render_contact_sheet(res, cfg, a.out, stem, video=a.video)
        print(f"renderer     : {vis['renderer'] or 'none worked'}  attempts={vis['attempts']}")
        if vis["png"]:
            print(f"contact sheet: {vis['png']}")
    return 0 if o["category"] != "invalid_rollout" else 4


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sim_before_act", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("--config", required=True); v.add_argument("--record", required=True)
    v.add_argument("--n", type=int, default=None, help="rollout budget (study budgets: 1,3,5,10,25,50)")
    v.add_argument("--out", default="traces_out")
    i = sub.add_parser("import-lerobot"); i.add_argument("--dataset", required=True); i.add_argument("--episode", type=int, default=0)
    i.add_argument("--start", type=int, default=0); i.add_argument("--horizon", type=int, default=30)
    i.add_argument("--sidecar"); i.add_argument("--out", required=True)
    sc = sub.add_parser("scene-check", help="validate a scene config / build the scene; physical mode refuses without measured inputs")
    sc.add_argument("--scene", default=DEFAULT_SCENE); sc.add_argument("--segment", default=None)
    sr = sub.add_parser("scene-rollout", help="run an action segment (or an INVENTED fixture) in the cube-to-tray scene")
    sr.add_argument("--scene", default=DEFAULT_SCENE); sr.add_argument("--segment", default=None)
    sr.add_argument("--fixture", choices=("zero", "small_motion"), default=None)
    sr.add_argument("--init", choices=("scene", "segment_state"), default="scene")
    sr.add_argument("--seed", type=int, default=0); sr.add_argument("--perturb", action="store_true")
    sr.add_argument("--out", default="scene_out"); sr.add_argument("--render", action="store_true"); sr.add_argument("--video", action="store_true")
    a = p.parse_args(argv)
    try:
        return {"verify": _cmd_verify, "import-lerobot": _cmd_import, "scene-check": _cmd_scene_check,
                "scene-rollout": _cmd_scene_rollout}[a.cmd](a)
    except (ConfigError, RecordError, FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
