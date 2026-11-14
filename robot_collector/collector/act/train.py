"""ACT training on a converted dataset (CPU-safe, offline; torch/lerobot are imported lazily inside functions).

* trains on the TRAIN split only; the VAL split drives checkpoint selection (best eval-mode L1 in normalised space, recorded);
* deterministic: ``torch.manual_seed``, numpy ``default_rng([seed, epoch])`` data order (a pure function of the step, so a resumed run
  sees exactly the batches an uninterrupted run would), torch RNG state saved/restored on resume;
* every checkpoint directory carries ``identity.json``; synthetic data or a provisional split can only produce a SMOKE checkpoint
  (``kind: smoke_random_init_few_steps``, ``trained_for_task: false``);
* ``freeze`` copies checkpoint + preprocessing + config into ``frozen/`` with sha256 of everything (``FREEZE.json``).

Nothing here touches hardware or the network. ``push_to_hub`` is forced to False.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ..util import sha256_file, write_json_atomic
from .adapter import ActStepsDataset, index_stream, stats_tensors
from .contract import (FRAMEWORK, JOINT_NAMES, KEY_IMAGE_PREFIX, KEY_STATE, SMOKE_KIND, SMOKE_LABEL, UNITS, sha256_obj)
from .load import Converted

RUN_KINDS = ("smoke", "main")
MODEL_KEYS = ("chunk_size", "n_action_steps", "vision_backbone", "pretrained_backbone_weights", "replace_final_stride_with_dilation",
              "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers", "n_decoder_layers", "n_vae_encoder_layers", "latent_dim",
              "use_vae", "kl_weight", "dropout", "feedforward_activation", "pre_norm")
FREEZE_SCHEMA = "act-frozen-checkpoint-v0.1"
OFFLINE_LABEL = "OFFLINE prediction error against recorded commands - NOT task success, NOT physical or simulated execution"
FROZEN_FILES = ("model.safetensors", "config.json", "normalization_stats.json", "preprocessor.json", "train_config.json",
                "conversion_ref.json", "identity.json")


class TrainingRefused(RuntimeError):
    pass


@dataclass
class TrainConfig:
    kind: str = "smoke"                 # smoke | main
    name: str = "act_run"
    seed: int = 1000
    steps: int = 100000
    batch_size: int = 8
    eval_every: int = 1000
    device: str = "cpu"
    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    require_wrist: bool = True
    model: dict = field(default_factory=dict)

    def validate(self) -> "TrainConfig":
        if self.kind not in RUN_KINDS:
            raise TrainingRefused(f"kind must be one of {RUN_KINDS}")
        bad = sorted(set(self.model) - set(MODEL_KEYS))
        if bad:
            raise TrainingRefused(f"unsupported model keys {bad}; allowed: {list(MODEL_KEYS)}")
        for k in ("steps", "batch_size", "eval_every"):
            if isinstance(getattr(self, k), bool) or not isinstance(getattr(self, k), int) or getattr(self, k) < 1:
                raise TrainingRefused(f"{k} must be a positive int")
        if self.device != "cpu":
            raise TrainingRefused("only device=cpu is supported/validated here (no GPU/XPU claims are made)")
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = set(cls.__dataclass_fields__)
        bad = sorted(set(d) - known)
        if bad:
            raise TrainingRefused(f"unknown train config keys: {bad}")
        return cls(**d).validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        doc = yaml.safe_load(Path(path).read_text()) or {}
        d = dict(doc.get("train", {}))
        d["model"] = dict(doc.get("model", {}))
        return cls.from_dict(d)


# ------------------------------------------------------------------ environment / identity
def env_info() -> dict:
    import torch

    def ver(mod):
        try:
            return importlib.import_module(mod).__version__
        except Exception:                                    # noqa: BLE001 - report absence, not fail
            return None
    xpu = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    commit = None
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=5)
        commit = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "torchvision": ver("torchvision"),
            "lerobot": ver("lerobot"), "numpy": np.__version__, "git_commit": commit, "torch_threads": torch.get_num_threads(),
            "cuda_available": bool(torch.cuda.is_available()), "xpu_available": xpu, "device_used": "cpu"}


def build_act_config(cfg: TrainConfig, conv: Converted):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

    cams = conv.conv["cameras"]
    feats = {KEY_STATE: PolicyFeature(FeatureType.STATE, (len(JOINT_NAMES),))}
    for c, v in cams.items():
        h, w, _ = v["shape_hwc"]
        feats[KEY_IMAGE_PREFIX + c] = PolicyFeature(FeatureType.VISUAL, (3, h, w))
    model = dict(cfg.model)
    model.setdefault("chunk_size", conv.conv["chunk_size"])
    model.setdefault("n_action_steps", conv.conv["n_action_steps"])
    if (model["chunk_size"], model["n_action_steps"]) != (conv.conv["chunk_size"], conv.conv["n_action_steps"]):
        raise TrainingRefused("model chunk_size/n_action_steps differ from the conversion's; re-convert (the audit's padding counts "
                              "are computed for the converted chunk_size)")
    model.setdefault("pretrained_backbone_weights", None)
    return ACTConfig(device=cfg.device, push_to_hub=False, input_features=feats,
                     output_features={"action": PolicyFeature(FeatureType.ACTION, (len(JOINT_NAMES),))},
                     optimizer_lr=cfg.optimizer_lr, optimizer_lr_backbone=cfg.optimizer_lr_backbone,
                     optimizer_weight_decay=cfg.optimizer_weight_decay, **model)


def deviations_from_act_defaults(cfg: TrainConfig, act_cfg) -> dict:
    from lerobot.policies.act.configuration_act import ACTConfig

    defaults = {f.name: f.default for f in dataclasses.fields(ACTConfig)}
    names = list(MODEL_KEYS) + ["optimizer_lr", "optimizer_lr_backbone", "optimizer_weight_decay"]
    return {k: {"act_default": defaults.get(k), "used": getattr(act_cfg, k)} for k in names
            if k in defaults and getattr(act_cfg, k) != defaults[k]}


def make_processors(act_cfg, stats_doc: dict):
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    return make_act_pre_post_processors(act_cfg, stats_tensors(stats_doc))


def _pretrained_cached(name: str) -> bool:
    import torch
    import torchvision.models as tm
    from urllib.parse import urlparse

    cls, member = name.split(".")
    w = getattr(tm, cls)[member]
    return (Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(urlparse(w.url).path)).exists()


def guard_run(cfg: TrainConfig, conv: Converted, act_cfg) -> None:
    src_split = conv.conv["source"]["split"]
    weights = act_cfg.pretrained_backbone_weights
    if weights is not None and not _pretrained_cached(weights):
        raise TrainingRefused(f"pretrained_backbone_weights={weights!r} is not in the local torch hub cache; it would need a download, "
                              "which this offline workflow never attempts. Use null for smoke runs.")
    if cfg.kind == "main":
        why = []
        if conv.data_origin != "recorded_physical":
            why.append(f"data_origin is {conv.data_origin!r}")
        if not src_split.get("frozen") or src_split.get("provisional"):
            why.append("the split is not FROZEN")
        if src_split.get("allow_provisional_split_flag"):
            why.append("the conversion used --allow-provisional-split")
        if not conv.conv.get("suitable_for_main_training"):
            why.append("conversion says suitable_for_main_training=false")
        if why:
            raise TrainingRefused("kind=main refused: " + "; ".join(why) + ". Only a smoke run is allowed on this data.")
    elif cfg.kind == "smoke" and conv.data_origin == "recorded_physical" and src_split.get("frozen"):
        pass                                                  # a smoke run on real frozen data is allowed but stays labelled smoke


def identity_doc(cfg: TrainConfig, conv: Converted, act_cfg, steps_completed: int, extra: dict | None = None) -> dict:
    smoke = cfg.kind == "smoke"
    complete = steps_completed >= cfg.steps
    doc = {
        "kind": SMOKE_KIND if smoke else "trained",
        "trained_for_task": False if smoke else bool(complete),
        "data_origin": conv.data_origin,
        "label": SMOKE_LABEL if smoke else ("TRAINED on the pilot task data (complete run); offline artefact - says nothing about task success"
                                            if complete else "INCOMPLETE main run - not a trained policy"),
        "run_name": cfg.name, "seed": cfg.seed, "steps_configured": cfg.steps, "steps_completed": steps_completed,
        "framework": FRAMEWORK, "device": cfg.device,
        "pretrained_backbone_weights": act_cfg.pretrained_backbone_weights,
        "deviations_from_act_defaults": deviations_from_act_defaults(cfg, act_cfg),
        "conversion_sha256": conv.sha256, "normalization_sha256": conv.stats_doc["sha256"],
        "split_status": conv.conv["source"]["split"]["status"],
        "note": ("Randomly initialised weights, a handful of CPU steps, synthetic fixture data: software execution only."
                 if smoke else "Not validated in simulation or on hardware."),
    }
    doc.update(extra or {})
    return doc


# ------------------------------------------------------------------ evaluation (val)
def evaluate(policy, pre, post, ds: ActStepsDataset, batch_size: int = 16) -> dict:
    """Eval-mode val loss (L1 in normalised space, padded positions masked) and OFFLINE prediction error in contract units."""
    import torch

    was_training = policy.training
    policy.eval()
    n = len(ds)
    loss_sum, abs_sum, cnt = 0.0, np.zeros(len(JOINT_NAMES)), 0.0
    with torch.no_grad():
        for s in range(0, n, batch_size):
            idx = list(range(s, min(n, s + batch_size)))
            raw = ds.batch(idx)
            batch = pre(raw)
            loss, _ = policy(batch)
            loss_sum += float(loss) * len(idx)
            obs = {k: v for k, v in batch.items() if k.startswith("observation.")}
            pred = post(policy.predict_action_chunk(obs)).double().cpu().numpy()
            for j, i in enumerate(idx):
                act, pad = ds.raw_actions(i)
                m = ~pad
                abs_sum += np.abs(pred[j] - act)[m].sum(axis=0)
                cnt += float(m.sum())
    policy.train(was_training)
    mae = abs_sum / max(cnt, 1.0)
    return {"n_rows": n, "val_loss_l1_normalised": loss_sum / max(n, 1), "label": OFFLINE_LABEL, "unit_per_joint": list(UNITS),
            "offline_mae_per_joint": dict(zip(JOINT_NAMES, [float(x) for x in mae])), "offline_mae_mean": float(mae.mean()),
            "n_valid_action_positions": int(cnt),
            "note": "mean over the 6 joints mixes degrees and gripper percent; read the per-joint values"}


# ------------------------------------------------------------------ checkpoints
def _set_seeds(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_ckpt(path: Path, policy, opt, state: dict, identity: dict) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    for f in path.iterdir():
        if f.is_file():
            f.unlink()
    policy.save_pretrained(path)
    torch.save({"optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state()}, path / "trainer_state.pt")
    write_json_atomic(path / "train_state.json", state)
    write_json_atomic(path / "identity.json", identity)


def train(cfg: TrainConfig, converted_dir: str | Path, run_dir: str | Path, *, resume: bool = False,
          stop_after_steps: int | None = None, log=print) -> dict:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg.validate()
    conv = Converted(converted_dir)
    act_cfg = build_act_config(cfg, conv)
    guard_run(cfg, conv, act_cfg)
    run = Path(run_dir)
    cfg_doc = cfg.to_dict()
    comparable = {k: v for k, v in cfg_doc.items() if k != "steps"}
    if resume:
        last = run / "last"
        if not (last / "model.safetensors").exists():
            raise TrainingRefused(f"nothing to resume in {run}")
        prev = json.loads((run / "train_config.json").read_text())
        if {k: v for k, v in prev["config"].items() if k != "steps"} != comparable or prev["conversion_sha256"] != conv.sha256:
            raise TrainingRefused("resume refused: train config or converted dataset differs from the original run")
        state = json.loads((last / "train_state.json").read_text())
    else:
        if run.exists() and any(run.iterdir()):
            raise TrainingRefused(f"{run} is not empty; refusing to overwrite a run")
        run.mkdir(parents=True, exist_ok=True)
        write_json_atomic(run / "train_config.json", {"config": cfg_doc, "config_sha256": sha256_obj(cfg_doc), "conversion_sha256": conv.sha256,
                                                      "env": env_info()})
        state = {"step": 0, "best_val_loss": None, "best_step": None, "history": {"train": [], "eval": []}}

    _set_seeds(cfg.seed)
    if resume:
        policy = ACTPolicy.from_pretrained(run / "last", config=act_cfg)
    else:
        policy = ACTPolicy(act_cfg)
    policy.to(cfg.device)
    policy.train()
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    if resume:
        ts = torch.load(run / "last" / "trainer_state.pt", map_location="cpu", weights_only=True)
        opt.load_state_dict(ts["optimizer"])
        torch.set_rng_state(ts["torch_rng"])
    pre, post = make_processors(act_cfg, conv.stats_doc)
    ds_tr = ActStepsDataset(conv, "train", act_cfg.chunk_size, cfg.require_wrist)
    ds_va = ActStepsDataset(conv, "val", act_cfg.chunk_size, cfg.require_wrist)
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise TrainingRefused(f"empty split (train {len(ds_tr)}, val {len(ds_va)} rows)")

    end = min(cfg.steps, stop_after_steps) if stop_after_steps else cfg.steps
    t0 = time.monotonic()
    for step in range(state["step"], end):
        batch = pre(ds_tr.batch(index_stream(len(ds_tr), cfg.batch_size, cfg.seed, step)))
        loss, ld = policy(batch)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        opt.step()
        done = step + 1
        state["step"] = done
        state["history"]["train"].append({"step": done, "loss": float(loss.detach()), "l1_loss": float(ld["l1_loss"]),
                                          "kld_loss": float(ld.get("kld_loss", 0.0)), "grad_norm": float(gn)})
        if done % cfg.eval_every == 0 or done == end:
            ev = evaluate(policy, pre, post, ds_va)
            ev["step"] = done
            state["history"]["eval"].append(ev)
            improved = state["best_val_loss"] is None or ev["val_loss_l1_normalised"] < state["best_val_loss"]
            if improved:
                state["best_val_loss"], state["best_step"] = ev["val_loss_l1_normalised"], done
            ident = identity_doc(cfg, conv, act_cfg, done, {"selected_by": "best val L1 (normalised space)", "eval": ev})
            if improved:
                _save_ckpt(run / "best", policy, opt, {**state}, ident)
            _save_ckpt(run / "last", policy, opt, {**state}, ident)
            log(f"step {done}/{cfg.steps} train_loss={float(loss.detach()):.4f} val_l1={ev['val_loss_l1_normalised']:.4f}"
                f"{' *best*' if improved else ''}")
    return {"run_dir": str(run), "steps_completed": state["step"], "steps_configured": cfg.steps,
            "best_step": state["best_step"], "best_val_loss_l1_normalised": state["best_val_loss"],
            "seconds": round(time.monotonic() - t0, 3), "last_model_sha256": sha256_file(run / "last" / "model.safetensors"),
            "kind": SMOKE_KIND if cfg.kind == "smoke" else "trained", "device": cfg.device}


# ------------------------------------------------------------------ freeze
def freeze(run_dir: str | Path, converted_dir: str | Path, frozen_dir: str | Path, which: str = "best") -> dict:
    if which not in ("best", "last"):
        raise ValueError("which must be 'best' or 'last'")
    run, out = Path(run_dir), Path(frozen_dir)
    ck = run / which
    if not (ck / "model.safetensors").exists():
        raise TrainingRefused(f"{ck} has no checkpoint")
    if out.exists() and any(out.iterdir()):
        raise TrainingRefused(f"{out} exists and is not empty; a frozen checkpoint is never overwritten")
    conv = Converted(converted_dir)
    tc = json.loads((run / "train_config.json").read_text())
    if tc["conversion_sha256"] != conv.sha256:
        raise TrainingRefused("the converted dataset is not the one this run was trained on")
    ident = json.loads((ck / "identity.json").read_text())
    state = json.loads((ck / "train_state.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ck / "model.safetensors", out / "model.safetensors")
    shutil.copyfile(ck / "config.json", out / "config.json")
    shutil.copyfile(Path(converted_dir) / "normalization_stats.json", out / "normalization_stats.json")
    cfg_doc = tc["config"]
    cam_shapes = {c: v["shape_hwc"] for c, v in conv.conv["cameras"].items()}
    act_cfg_doc = json.loads((ck / "config.json").read_text())
    pre_spec = {"normalization_stats_sha256": conv.stats_doc["sha256"], "normalization_mapping": act_cfg_doc.get("normalization_mapping"),
                "state_key": KEY_STATE, "image_keys": [KEY_IMAGE_PREFIX + c for c in cam_shapes], "image_shapes_hwc": cam_shapes,
                "image_scale": "uint8/255 -> [0,1], then MEAN_STD", "joint_names": list(JOINT_NAMES), "units": list(UNITS),
                "postprocess": "MEAN_STD unnormalise of the action chunk; float32 from torch, widened to float64; NO clipping"}
    write_json_atomic(out / "preprocessor.json", pre_spec)
    write_json_atomic(out / "train_config.json", tc)
    src = conv.conv["source"]
    ref = {"conversion_sha256": conv.sha256, "manifest_content_sha256": src["manifest_content_sha256"],
           "split_sha256": src["split"]["split_sha256"], "split_status": src["split"]["status"], "split_provisional": src["split"]["provisional"],
           "allow_provisional_split_flag": src["split"]["allow_provisional_split_flag"], "control_hz": conv.conv["control_hz"],
           "data_origin": conv.data_origin, "chunk_size": conv.conv["chunk_size"], "n_action_steps": conv.conv["n_action_steps"],
           "cameras": cam_shapes}
    write_json_atomic(out / "conversion_ref.json", ref)
    write_json_atomic(out / "identity.json", ident)
    files = {f: sha256_file(out / f) for f in FROZEN_FILES}
    env = env_info()
    doc = {"schema": FREEZE_SCHEMA, "checkpoint_id": f"{ident['run_name']}-{which}-{files['model.safetensors'][:12]}", "selected": which,
           "selected_step": state["step"] if which == "last" else state["best_step"], "best_val_loss_l1_normalised": state["best_val_loss"],
           "kind": ident["kind"], "trained_for_task": ident["trained_for_task"], "data_origin": ident["data_origin"], "label": ident["label"],
           "checkpoint_sha256": files["model.safetensors"], "config_sha256": files["config.json"],
           "preprocessing_sha256": sha256_obj(pre_spec), "normalization_sha256": conv.stats_doc["sha256"],
           "train_config_sha256": tc["config_sha256"], "dataset": ref, "seed": tc["config"]["seed"], "files": files,
           "versions": {k: env[k] for k in ("python", "torch", "torchvision", "lerobot", "numpy")}, "git_commit": env["git_commit"],
           "device": "cpu", "cuda_available": env["cuda_available"], "xpu_available": env["xpu_available"], "platform": env["platform"],
           "framework": FRAMEWORK, "created_wall_ns": time.time_ns()}
    doc["freeze_sha256"] = sha256_obj(doc)
    write_json_atomic(out / "FREEZE.json", doc)
    for f in (*FROZEN_FILES, "FREEZE.json"):
        os.chmod(out / f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return doc


def verify_frozen(frozen_dir: str | Path) -> list[str]:
    """Recompute every recorded hash of a frozen checkpoint. Empty list = intact."""
    d = Path(frozen_dir)
    p: list[str] = []
    try:
        doc = json.loads((d / "FREEZE.json").read_text())
    except (OSError, ValueError) as e:
        return [f"cannot read FREEZE.json: {e}"]
    if doc.get("schema") != FREEZE_SCHEMA:
        p.append("unknown FREEZE schema")
    body = {k: v for k, v in doc.items() if k != "freeze_sha256"}
    if sha256_obj(body) != doc.get("freeze_sha256"):
        p.append("FREEZE.json content hash mismatch")
    for f, h in doc.get("files", {}).items():
        if not (d / f).exists() or sha256_file(d / f) != h:
            p.append(f"{f}: missing or hash differs from FREEZE.json")
    if doc.get("checkpoint_sha256") != doc.get("files", {}).get("model.safetensors"):
        p.append("checkpoint_sha256 != model.safetensors hash")
    if (d / "preprocessor.json").exists() and sha256_obj(json.loads((d / "preprocessor.json").read_text())) != doc.get("preprocessing_sha256"):
        p.append("preprocessing_sha256 mismatch")
    if (d / "normalization_stats.json").exists():
        from .load import stats_hash

        ns = json.loads((d / "normalization_stats.json").read_text())
        if stats_hash(ns) != ns.get("sha256") or ns.get("sha256") != doc.get("normalization_sha256"):
            p.append("normalization stats hash mismatch")
    return p
