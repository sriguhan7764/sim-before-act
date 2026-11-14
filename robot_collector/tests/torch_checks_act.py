"""Torch-side checks for the ACT pipeline (Milestone D). Plain asserts, no pytest: the lerobot env has no pytest.

Run from the robot_collector root in the lerobot env:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/apic3/lerobot_v6_2/lerobot/src:$PWD \
        /home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python tests/torch_checks_act.py [scratch_dir]

Everything is SYNTHETIC SMOKE TEST - software execution only, not a thesis finding. CPU only. No hardware is touched.
The scratch dir (default .work_act/torch_checks, inside the project) must not exist yet and is removed afterwards.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[str] = []


def check(cond, name: str):
    if not cond:
        raise AssertionError(name)
    RESULTS.append(name)


def raises(exc, fn, name: str, match: str | None = None):
    try:
        fn()
    except exc as e:
        if match is not None and match not in str(e):
            raise AssertionError(f"{name}: wrong message: {e}") from e
        RESULTS.append(name)
        return
    raise AssertionError(f"{name}: no {exc.__name__} raised")


def main(scratch: Path) -> int:
    import torch

    from collector.act import adapter
    from collector.act.contract import SMOKE_KIND, SMOKE_LABEL
    from collector.act.infer import FrozenPolicy, evaluate_frozen, infer_from_converted
    from collector.act.load import Converted
    from collector.act.segment import load_action_segment, validate_action_segment_dict
    from collector.act.smoke import DEFAULT_CONFIG, run_smoke
    from collector.act.train import OFFLINE_LABEL, TrainConfig, TrainingRefused, build_act_config, train, verify_frozen
    from collector.util import read_json

    if scratch.exists():
        raise SystemExit(f"{scratch} already exists; refusing to overwrite")
    cfg_path = DEFAULT_CONFIG

    # ---- ACT defaults we rely on (installed lerobot 0.6.1)
    from lerobot.policies.act.configuration_act import ACTConfig

    d = ACTConfig()
    check((d.chunk_size, d.n_action_steps, d.n_obs_steps) == (100, 100, 1), "ACT defaults: chunk_size=100 n_action_steps=100 n_obs_steps=1")
    check(d.use_vae and d.kl_weight == 10.0 and d.vision_backbone == "resnet18", "ACT defaults: use_vae, kl_weight=10, resnet18")
    check(d.normalization_mapping["ACTION"].value == "MEAN_STD" and d.normalization_mapping["STATE"].value == "MEAN_STD",
          "ACT defaults: MEAN_STD for state/action")

    # ---- configs parse
    for name in ("act_smoke.yaml", "act_pilot.yaml"):
        tc = TrainConfig.from_yaml(ROOT / "configs" / name)
        check(tc.device == "cpu", f"configs/{name} parses and is cpu")
    check(TrainConfig.from_yaml(ROOT / "configs" / "act_pilot.yaml").kind == "main", "act_pilot.yaml is kind=main")
    check(TrainConfig.from_yaml(cfg_path).kind == "smoke", "act_smoke.yaml is kind=smoke")
    raises(TrainingRefused, lambda: TrainConfig.from_dict({"device": "cuda"}), "device other than cpu is refused", "cpu")
    raises(TrainingRefused, lambda: TrainConfig.from_dict({"model": {"not_a_key": 1}}), "unknown model key is refused", "unsupported")

    # ---- full smoke (fixture -> convert -> train -> stop/resume -> freeze -> infer -> evaluate)
    try:
        rep = run_smoke(scratch, cfg_path, stop_after=3, log=lambda *_: None)
        conv_dir = scratch / "converted"
        conv = Converted(conv_dir)

        check(rep["kind"] == SMOKE_KIND and rep["trained_for_task"] is False and rep["data_origin"] == "synthetic_fixture",
              "smoke report is labelled smoke / not trained / synthetic")
        check(rep["resume_matches_straight_final_weights"] is True, "stop-and-resume gives the same final weights as a straight run")
        check(rep["frozen_verify_problems"] == [] and rep["segment_problems"] == [], "frozen checkpoint verifies and the segment validates")
        check(rep["device_used"] == "cpu", "device used is cpu")
        for run in ("run_straight", "run_resumed"):
            for which in ("best", "last"):
                ident = read_json(scratch / run / which / "identity.json")
                check(ident["kind"] == "smoke_random_init_few_steps" and ident["trained_for_task"] is False
                      and ident["data_origin"] == "synthetic_fixture" and "SYNTHETIC SMOKE TEST" in ident["label"],
                      f"{run}/{which}/identity.json carries the smoke labels")
        ident = read_json(scratch / "run_resumed" / "best" / "identity.json")
        check(ident["label"] == SMOKE_LABEL or "software execution only, not a thesis finding" in ident["label"], "identity label is the loud smoke label")

        # model config deviations from ACT defaults are recorded
        tcj = read_json(scratch / "run_resumed" / "train_config.json")
        check(tcj["config"]["model"].get("pretrained_backbone_weights", "x") is None, "smoke config uses pretrained_backbone_weights=None")
        act_cfg = build_act_config(TrainConfig.from_dict(tcj["config"]), conv)
        check(act_cfg.push_to_hub is False and act_cfg.device == "cpu", "ACTConfig has push_to_hub False and device cpu")
        check(act_cfg.chunk_size == conv.conv["chunk_size"], "ACT chunk_size follows the conversion")

        # train history: validation-based selection recorded
        state = read_json(scratch / "run_straight" / "last" / "train_state.json")
        check(state["best_val_loss"] is not None and state["best_step"] is not None, "best val loss and step are recorded")
        evs = state["history"]["eval"]
        check(min(e["val_loss_l1_normalised"] for e in evs) == state["best_val_loss"], "best checkpoint is the minimum val loss")

        # ---- freeze content
        fz = read_json(scratch / "frozen" / "FREEZE.json")
        for k in ("checkpoint_sha256", "config_sha256", "preprocessing_sha256", "normalization_sha256", "train_config_sha256",
                  "freeze_sha256", "seed", "versions", "device", "cuda_available", "xpu_available", "git_commit"):
            check(k in fz, f"FREEZE.json records {k}")
        check(fz["dataset"]["split_sha256"] and fz["dataset"]["manifest_content_sha256"] and fz["dataset"]["conversion_sha256"],
              "FREEZE.json records dataset, manifest and split hashes")
        check(fz["dataset"]["allow_provisional_split_flag"] is True, "FREEZE.json records the provisional-split flag")
        check(fz["trained_for_task"] is False and fz["kind"] == SMOKE_KIND, "FREEZE.json is labelled smoke / not trained")
        check(fz["versions"]["lerobot"] == "0.6.1" and fz["versions"]["torch"].startswith("2."), "FREEZE.json records lerobot and torch versions")

        # ---- tampering is detected (on copies: the frozen dir is read-only)
        for fname, mutate in (("model.safetensors", lambda p: p.write_bytes(p.read_bytes() + b"x")),
                              ("normalization_stats.json", lambda p: p.write_text(p.read_text().replace("0", "1", 1))),
                              ("preprocessor.json", lambda p: p.write_text(p.read_text() + " "))):
            cp = scratch / f"tamper_{fname}"
            shutil.copytree(scratch / "frozen", cp)
            target = cp / fname
            target.chmod(0o644)
            mutate(target)
            check(len(verify_frozen(cp)) >= 1, f"verify_frozen detects a modified {fname}")
            raises(TrainingRefused, lambda cp=cp: FrozenPolicy(cp), f"FrozenPolicy refuses a tampered {fname}", "verification")

        # ---- inference: deterministic, unmodified, segment spec
        fp = FrozenPolicy(scratch / "frozen")
        state_v, images, ref = None, None, None
        from collector.act.infer import row_inputs

        state_v, images, ref = row_inputs(conv, int(conv.row_indices("val")[0]))
        a1, a2 = fp.predict(state_v, images), fp.predict(state_v, images)
        check(a1.dtype == np.float64 and a1.shape == (conv.conv["chunk_size"], 6), "predicted chunk is float64 (chunk_size, 6)")
        check(np.array_equal(a1, a2) and np.isfinite(a1).all(), "same state and images give identical finite actions")
        raises(ValueError, lambda: fp.predict(state_v[:5], images), "wrong-length state is refused", "state must be")
        bad_img = {k: v.astype(np.float32) for k, v in images.items()}
        raises(ValueError, lambda: fp.predict(state_v, bad_img), "float images are refused", "uint8")
        raises(ValueError, lambda: fp.predict(state_v, {}), "missing camera is refused", "missing image")

        seg_path = scratch / "seg2.json"
        infer_from_converted(scratch / "frozen", conv_dir, seg_path, split="val", row=0)
        seg, seg_actions = load_action_segment(seg_path)
        check(validate_action_segment_dict(seg) == [], "action segment from infer validates")
        row0_state, row0_images, _ = row_inputs(conv, int(conv.row_indices("val")[0]))
        check(np.array_equal(seg_actions, fp.predict(row0_state, row0_images)), "segment actions are the unmodified denormalised output (no clipping)")
        check(seg["policy"]["trained_for_task"] is False and seg["policy"]["kind"] == SMOKE_KIND and seg["data_origin"] == "synthetic_fixture"
              and "SYNTHETIC SMOKE" in seg["provenance_label"], "segment is labelled smoke / not trained / synthetic")
        check(seg["policy"]["checkpoint_sha256"] == fz["checkpoint_sha256"] and seg["policy"]["device"] == "cpu", "segment carries the frozen checkpoint hash")
        check(seg["contract"]["control_hz"] == conv.conv["control_hz"], "segment control_hz equals the conversion control rate")

        # ---- offline evaluation is labelled as prediction error, not success
        ev = evaluate_frozen(scratch / "frozen", conv_dir, "val", scratch / "eval2.json")
        text = json.dumps(ev)
        check(ev["label"] == OFFLINE_LABEL and "NOT task success" in text and ev["trained_for_task"] is False,
              "offline evaluation is labelled offline prediction error, not task success")

        # ---- guards
        main_cfg = TrainConfig.from_dict({**tcj["config"], "kind": "main"})
        raises(TrainingRefused, lambda: train(main_cfg, conv_dir, scratch / "run_main", log=lambda *_: None),
               "kind=main is refused on synthetic / provisional data", "kind=main refused")
        check(not (scratch / "run_main" / "last").exists(), "refused main run wrote no checkpoint")
        # a weights file that is NOT in the local torch hub cache must be refused rather than downloaded
        uncached = "ResNet50_Weights.IMAGENET1K_V1"
        from collector.act.train import _pretrained_cached

        check(not _pretrained_cached(uncached), "precondition: ResNet50 ImageNet weights are not cached locally")
        pre_cfg = TrainConfig.from_dict({**tcj["config"], "model": {**tcj["config"]["model"], "pretrained_backbone_weights": uncached}})
        raises(TrainingRefused, lambda: train(pre_cfg, conv_dir, scratch / "run_pre", log=lambda *_: None),
               "uncached pretrained weights are refused instead of downloaded", "download")
        check(not (scratch / "run_pre").exists(), "refused pretrained run created no run dir")
        other = TrainConfig.from_dict({**tcj["config"], "batch_size": tcj["config"]["batch_size"] + 1})
        raises(TrainingRefused, lambda: train(other, conv_dir, scratch / "run_resumed", resume=True, log=lambda *_: None),
               "resume with a different config is refused", "resume refused")
        raises(TrainingRefused, lambda: train(TrainConfig.from_dict(tcj["config"]), conv_dir, scratch / "run_resumed", log=lambda *_: None),
               "a non-empty run dir is never overwritten", "not empty")
        raises(TrainingRefused, lambda: __import__("collector.act.train", fromlist=["freeze"]).freeze(
            scratch / "run_resumed", conv_dir, scratch / "frozen", which="best"), "an existing frozen dir is never overwritten", "never overwritten")

        # ---- adapter: ACT batch keys, shapes and masks; deterministic data order
        ds = adapter.ActStepsDataset(conv, "train", conv.conv["chunk_size"], require_wrist=True)
        batch = ds.batch(np.arange(4))
        need = {"observation.state", "observation.images.front", "observation.images.wrist", "action", "action_is_pad"}
        check(need <= set(batch), "batch has the keys ACT expects")
        cs = conv.conv["chunk_size"]
        check(tuple(batch["observation.state"].shape) == (4, 6) and tuple(batch["action"].shape) == (4, cs, 6)
              and tuple(batch["action_is_pad"].shape) == (4, cs) and batch["action_is_pad"].dtype == torch.bool,
              "batch shapes and the bool action_is_pad mask")
        check(float(batch["observation.images.front"].min()) >= 0.0 and float(batch["observation.images.front"].max()) <= 1.0,
              "images are float in [0,1]")
        s1 = [adapter.index_stream(100, 8, 1000, k).tolist() for k in range(5)]
        s2 = [adapter.index_stream(100, 8, 1000, k).tolist() for k in range(5)]
        s3 = [adapter.index_stream(100, 8, 1001, k).tolist() for k in range(5)]
        check(s1 == s2 and s1 != s3, "data order is a pure function of (seed, step)")
        # the val split is never used by training batches
        val_rows = set(conv.row_indices("val").tolist())
        train_rows = set(conv.row_indices("train").tolist())
        check(not (val_rows & train_rows) and len(ds) == len(train_rows), "train dataset contains only train rows")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"torch checks: {len([r for r in RESULTS if not r.startswith('SKIPPED')])} passed")
    for r in RESULTS:
        if r.startswith("SKIPPED"):
            print(r)
    return 0


if __name__ == "__main__":
    scratch_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".work_act" / "torch_checks"
    sys.exit(main(scratch_dir))
