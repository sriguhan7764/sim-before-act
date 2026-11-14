"""Bounded SYNTHETIC smoke test of the whole offline chain (convert -> train few steps -> resume -> freeze -> infer -> evaluate).

Everything it produces is labelled ``SYNTHETIC SMOKE TEST - software execution only, not a thesis finding``. The checkpoint is a
randomly initialised ACT trained for a handful of CPU steps on fabricated fixture data: ``trained_for_task`` is false.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..util import write_json_atomic
from .contract import SMOKE_KIND, SMOKE_LABEL, SYNTHETIC_DATA_LABEL
from .convert import ConvertConfig, convert_dataset, verify_lineage
from .segment import validate_action_segment_dict
from .synthetic import build_fixture
from .train import TrainConfig, env_info, freeze, train, verify_frozen

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "act_smoke.yaml"


def run_smoke(out_dir: str | Path, config: str | Path = DEFAULT_CONFIG, stop_after: int = 3, log=print) -> dict:
    import yaml

    from .infer import evaluate_frozen, infer_from_converted

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} is not empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    doc = yaml.safe_load(Path(config).read_text())
    ccfg = ConvertConfig.from_dict(doc["convert"])
    tcfg = TrainConfig.from_yaml(config)
    if tcfg.kind != "smoke":
        raise RuntimeError("the smoke test runs kind=smoke configs only")
    t0 = time.monotonic()
    fx = build_fixture(out / "fixture")
    log(f"[smoke] fixture written ({SYNTHETIC_DATA_LABEL})")
    conv = convert_dataset(fx["manifest"], fx["split"], out / "converted", ccfg, allow_provisional_split=True)
    lin = verify_lineage(out / "converted", fx["manifest"])
    if not lin["ok"]:
        raise RuntimeError(f"lineage check failed: {lin['errors'][:3]}")
    log(f"[smoke] converted {conv['counts']}; lineage ok ({lin['n_rows']} rows, {lin['n_images_checked']} images)")

    straight = train(tcfg, out / "converted", out / "run_straight", log=log)
    train(tcfg, out / "converted", out / "run_resumed", stop_after_steps=stop_after, log=log)
    resumed = train(tcfg, out / "converted", out / "run_resumed", resume=True, log=log)
    same = straight["last_model_sha256"] == resumed["last_model_sha256"]
    log(f"[smoke] straight vs stop-and-resume final weights identical: {same}")

    fz = freeze(out / "run_resumed", out / "converted", out / "frozen", which="best")
    problems = verify_frozen(out / "frozen")
    if problems:
        raise RuntimeError(f"frozen checkpoint failed verification: {problems}")
    seg = infer_from_converted(out / "frozen", out / "converted", out / "action_segment.json", split="val", row=0)
    seg_problems = validate_action_segment_dict(json.loads((out / "action_segment.json").read_text()))
    ev = evaluate_frozen(out / "frozen", out / "converted", "val", out / "offline_eval.json")
    env = env_info()
    report = {
        "label": SMOKE_LABEL, "kind": SMOKE_KIND, "trained_for_task": False, "data_origin": "synthetic_fixture",
        "fixture_label": SYNTHETIC_DATA_LABEL, "provisional_split_flag_used": True,
        "steps": tcfg.steps, "stop_and_resume_at": stop_after, "straight": straight, "resumed": resumed,
        "resume_matches_straight_final_weights": same,
        "frozen_checkpoint_id": fz["checkpoint_id"], "frozen_selected": fz["selected"], "frozen_selected_step": fz["selected_step"],
        "frozen_verify_problems": problems, "segment_id": seg["segment_id"], "segment_problems": seg_problems,
        "segment_actions_shape": [seg["chunk"]["length"], len(seg["actions"][0])], "offline_eval": ev,
        "lineage": {k: lin[k] for k in ("ok", "n_rows", "n_images_checked")},
        "env": env, "device_used": "cpu", "seconds": round(time.monotonic() - t0, 2),
        "does_not_prove": ["the policy can do the task", "anything about GPU/XPU", "that pretrained ImageNet weights work offline",
                           "any simulator or hardware agreement"],
    }
    write_json_atomic(out / "smoke_report.json", report)
    return report
