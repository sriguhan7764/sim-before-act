"""Offline inference from a FROZEN checkpoint -> action segment file; plus offline evaluation (torch/lerobot imported lazily).

The denormalised action chunk is written UNMODIFIED: no clipping, no offsets, no smoothing (float32 model output widened to float64).
Nothing is executed on a robot or in a simulator here. Offline evaluation reports prediction error against RECORDED commands only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import JOINT_NAMES, KEY_IMAGE_PREFIX, KEY_STATE, SEGMENT_SMOKE_PROVENANCE, SMOKE_KIND
from .load import Converted
from .segment import image_sha256, write_action_segment
from .train import OFFLINE_LABEL, TrainingRefused, evaluate, verify_frozen


class FrozenPolicy:
    """A verified frozen checkpoint, loaded for inference (eval mode, CPU)."""

    def __init__(self, frozen_dir: str | Path):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.dir = Path(frozen_dir)
        problems = verify_frozen(self.dir)
        if problems:
            raise TrainingRefused("frozen checkpoint failed verification: " + "; ".join(problems))
        self.freeze = json.loads((self.dir / "FREEZE.json").read_text())
        self.ref = json.loads((self.dir / "conversion_ref.json").read_text())
        self.stats_doc = json.loads((self.dir / "normalization_stats.json").read_text())
        self.identity = json.loads((self.dir / "identity.json").read_text())
        cfg = PreTrainedConfig.from_pretrained(self.dir)
        cfg.device, cfg.push_to_hub = "cpu", False
        self.policy = ACTPolicy.from_pretrained(self.dir, config=cfg)
        self.policy.to("cpu").eval()
        from .train import make_processors

        self.pre, self.post = make_processors(cfg, self.stats_doc)
        self.cfg = cfg
        self.cams = list(self.ref["cameras"])
        self.torch = torch

    def predict(self, state, images: dict) -> np.ndarray:
        """state: 6 floats in contract units (degrees x5, gripper 0..100). images: {cam: HxWx3 uint8 RGB at the training size}.
        Returns the (chunk_size, 6) float64 action chunk, unmodified."""
        torch = self.torch
        s = np.asarray(state, dtype=np.float64)
        if s.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(s)):
            raise ValueError(f"state must be {len(JOINT_NAMES)} finite floats in joint order {JOINT_NAMES}")
        obs = {KEY_STATE: torch.from_numpy(s.astype(np.float32))[None]}
        for cam in self.cams:
            img = images.get(cam)
            h, w, _ = self.ref["cameras"][cam]
            if img is None:
                raise ValueError(f"missing image for camera {cam!r}")
            if img.dtype != np.uint8 or img.shape != (h, w, 3):
                raise ValueError(f"image {cam!r} must be uint8 {(h, w, 3)} RGB (the training size); got {img.dtype} {img.shape}")
            obs[KEY_IMAGE_PREFIX + cam] = torch.from_numpy(np.ascontiguousarray(img.astype(np.float32).transpose(2, 0, 1) / 255.0))[None]
        with torch.no_grad():
            batch = self.pre(obs)
            chunk = self.policy.predict_action_chunk({k: v for k, v in batch.items() if k.startswith("observation.")})
            out = self.post(chunk)
        return out[0].double().cpu().numpy()

    def policy_block(self) -> dict:
        f = self.freeze
        return {"kind": f["kind"], "trained_for_task": bool(f["trained_for_task"]), "checkpoint_id": f["checkpoint_id"],
                "checkpoint_sha256": f["checkpoint_sha256"], "preprocessing_sha256": f["preprocessing_sha256"],
                "config_sha256": f["config_sha256"], "framework": f["framework"], "device": "cpu", "seed": f["seed"]}

    def provenance_label(self) -> str:
        if self.freeze["kind"] == SMOKE_KIND:
            return SEGMENT_SMOKE_PROVENANCE
        if self.freeze["data_origin"] == "synthetic_fixture":
            return "SYNTHETIC-DATA POLICY OUTPUT - not a thesis finding"
        return (f"OFFLINE POLICY OUTPUT of frozen checkpoint {self.freeze['checkpoint_id']} - not executed on hardware, "
                "not validated in simulation")


def write_segment(fp: FrozenPolicy, out_path: str | Path, state, images: dict, state_ref: dict | None = None) -> dict:
    actions = fp.predict(state, images)
    imgs = {c: {"sha256": image_sha256(v), "shape": list(v.shape)} for c, v in images.items() if c in fp.cams}
    return write_action_segment(
        out_path, actions, state_values=np.asarray(state, dtype=np.float64), control_hz=fp.ref["control_hz"],
        data_origin=fp.freeze["data_origin"], provenance_label=fp.provenance_label(), policy=fp.policy_block(),
        execution_horizon_steps=fp.ref["n_action_steps"], input_images=imgs, state_ref=state_ref)


def row_inputs(conv: Converted, row: int) -> tuple[np.ndarray, dict, dict]:
    """Measured follower state, converted images and the lineage reference of one converted row."""
    import cv2

    state = conv.a["state"][row].astype(np.float64)
    images = {}
    for cam in conv.cameras:
        p = conv.image_path(cam, row)
        if p is None:
            raise ValueError(f"row {row} has no {cam} image")
        images[cam] = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ep = conv.episode(int(conv.a["episode_idx"][row]))["episode_id"]
    ref = {"episode_id": ep, "frame_index": int(conv.a["front_frame_index"][row]), "t_host_mono_ns": int(conv.a["t_ref_ns"][row])}
    return state, images, ref


def infer_from_converted(frozen_dir, converted_dir, out_path, split: str = "val", row: int = 0) -> dict:
    fp = FrozenPolicy(frozen_dir)
    conv = Converted(converted_dir)
    if conv.sha256 != fp.ref["conversion_sha256"]:
        raise TrainingRefused("the converted dataset is not the one the frozen checkpoint was trained on")
    rows = conv.row_indices(split)
    if not 0 <= row < len(rows):
        raise ValueError(f"row must be in [0, {len(rows)}) for split {split!r}")
    state, images, ref = row_inputs(conv, int(rows[row]))
    return write_segment(fp, out_path, state, images, ref)


def evaluate_frozen(frozen_dir, converted_dir, split: str = "val", out_path: str | Path | None = None) -> dict:
    from .adapter import ActStepsDataset

    fp = FrozenPolicy(frozen_dir)
    conv = Converted(converted_dir)
    if conv.sha256 != fp.ref["conversion_sha256"]:
        raise TrainingRefused("the converted dataset is not the one the frozen checkpoint was trained on")
    ds = ActStepsDataset(conv, split, fp.cfg.chunk_size, require_wrist=True)
    res = evaluate(fp.policy, fp.pre, fp.post, ds)
    res.update({"split": split, "checkpoint_id": fp.freeze["checkpoint_id"], "policy_kind": fp.freeze["kind"],
                "trained_for_task": fp.freeze["trained_for_task"], "data_origin": fp.freeze["data_origin"], "label": OFFLINE_LABEL})
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return res
