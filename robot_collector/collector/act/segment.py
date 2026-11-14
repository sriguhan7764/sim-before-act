"""``action-segment-v0.1``: the hand-off file between offline ACT inference (torch environment) and MuJoCo verification.

Pure python + numpy. No torch, no lerobot, no MuJoCo. The loader on the simulator side re-implements its own reader to the
same spec; :func:`validate_action_segment_dict` here is self-contained so it can also be used to test that loader.

Hash recipes (exact):
  action_sha256 = sha256 of ``np.ascontiguousarray(actions, dtype='<f8').tobytes()`` for the (T, 6) array
  state_sha256  = same recipe for the (6,) state vector
  image sha256  = sha256 of the C-contiguous uint8 (h, w, 3) RGB array bytes
  contract_id   = sha256 of canonical JSON (sorted keys, compact separators) of ``contract`` (incl. ``control_hz``)
  segment_id    = first 16 hex characters of sha256( action_sha256 + contract_id )   (ascii string concatenation)
The actions are the DENORMALISED policy output exactly as computed (float32 network output widened to float64): never
clipped, offset or smoothed. They are position targets in follower normalised units, NOT confirmed physical execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .contract import (FRAMEWORK, JOINT_NAMES, N_JOINTS, SEGMENT_CONTRACT, SEGMENT_SCHEMA, SEGMENT_SMOKE_PROVENANCE, SMOKE_KIND,
                       canonical, contract_id, sha256_text)

DATA_ORIGINS = ("synthetic_fixture", "recorded_physical")
POLICY_KINDS = (SMOKE_KIND, "trained", "fixture_stub")
POLICY_KEYS = ("kind", "trained_for_task", "checkpoint_id", "checkpoint_sha256", "preprocessing_sha256", "config_sha256", "framework",
               "device", "seed")
TOP_KEYS = ("schema", "segment_id", "data_origin", "provenance_label", "contract", "actions", "action_sha256", "chunk", "policy",
            "input", "created_wall_ns")


def array_sha256(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a), dtype="<f8").tobytes()).hexdigest()


def image_sha256(img) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)).tobytes()).hexdigest()


def make_segment_id(action_sha256: str, control_hz: float) -> str:
    return sha256_text(action_sha256 + contract_id(control_hz))[:16]


def build_action_segment(actions, *, state_values, control_hz: float, data_origin: str, provenance_label: str, policy: dict,
                         execution_horizon_steps: int | None = None, input_images: dict | None = None, state_ref: dict | None = None,
                         created_wall_ns: int | None = None) -> dict:
    a = np.ascontiguousarray(np.asarray(actions, dtype="<f8"))
    if a.ndim != 2 or a.shape[1] != N_JOINTS or a.shape[0] < 1:
        raise ValueError(f"actions must have shape (T>=1, {N_JOINTS}); got {a.shape}")
    s = np.ascontiguousarray(np.asarray(state_values, dtype="<f8"))
    if s.shape != (N_JOINTS,):
        raise ValueError(f"state_values must have shape ({N_JOINTS},); got {s.shape}")
    T = int(a.shape[0])
    asha = array_sha256(a)
    pol = {"framework": FRAMEWORK, "device": "cpu", "seed": None, "checkpoint_sha256": None, "preprocessing_sha256": None,
           "config_sha256": None, **policy}
    images = {}
    for cam, v in (input_images or {}).items():
        images[cam] = {"sha256": v["sha256"], "shape": [int(x) for x in v["shape"]]}
    seg = {
        "schema": SEGMENT_SCHEMA,
        "segment_id": make_segment_id(asha, control_hz),
        "data_origin": data_origin,
        "provenance_label": provenance_label,
        "contract": {**SEGMENT_CONTRACT, "control_hz": float(control_hz)},
        "actions": a.tolist(),
        "action_sha256": asha,
        "chunk": {"length": T, "start_step": 0, "execution_horizon_steps": int(execution_horizon_steps or T),
                  "temporal_ensembling": False, "replanning": False},
        "policy": {k: pol.get(k) for k in POLICY_KEYS},
        "input": {"state_values": s.tolist(), "state_sha256": array_sha256(s),
                  "state_ref": {"episode_id": None, "frame_index": None, "t_host_mono_ns": None, **(state_ref or {})},
                  "images": images},
        "created_wall_ns": int(created_wall_ns if created_wall_ns is not None else time.time_ns()),
    }
    return seg


def write_action_segment(path: str | Path, actions, **kw) -> dict:
    """Build, validate and atomically write an action segment. Returns the segment dict. Refuses an invalid segment."""
    seg = build_action_segment(actions, **kw)
    problems = validate_action_segment_dict(seg)
    if problems:
        raise ValueError("refusing to write an invalid action segment: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(seg, indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)
    return seg


def load_action_segment(path: str | Path) -> tuple[dict, np.ndarray]:
    """Read + validate. Returns (segment dict, actions as a (T,6) float64 array). Raises ValueError when invalid."""
    seg = json.loads(Path(path).read_text())
    problems = validate_action_segment_dict(seg)
    if problems:
        raise ValueError(f"{path}: invalid action segment: " + "; ".join(problems))
    return seg, np.asarray(seg["actions"], dtype="<f8")


def _is_hex(s, n=None) -> bool:
    return isinstance(s, str) and (n is None or len(s) == n) and len(s) > 0 and all(c in "0123456789abcdef" for c in s)


def validate_action_segment_dict(d) -> list[str]:
    """Structural + integrity problems of an ``action-segment-v0.1`` dict (empty list = valid).

    Checks the schema, contract block, finite float actions of shape (T,6), the action/state/segment hashes, chunk bookkeeping,
    and the honesty rules: a smoke or synthetic policy can never claim ``trained_for_task`` and must carry its label.
    """
    p: list[str] = []
    if not isinstance(d, dict):
        return ["segment is not a JSON object"]
    p += [f"missing key {k!r}" for k in TOP_KEYS if k not in d]
    if p:
        return p
    if d["schema"] != SEGMENT_SCHEMA:
        p.append(f"schema {d['schema']!r} != {SEGMENT_SCHEMA!r}")
    if d["data_origin"] not in DATA_ORIGINS:
        p.append(f"data_origin {d['data_origin']!r} not in {DATA_ORIGINS}")
    if not isinstance(d["provenance_label"], str) or not d["provenance_label"].strip():
        p.append("provenance_label must be a non-empty string")
    c = d["contract"]
    if not isinstance(c, dict):
        return p + ["contract is not an object"]
    for k, want in SEGMENT_CONTRACT.items():
        if c.get(k) != want:
            p.append(f"contract.{k} != {want!r}")
    hz = c.get("control_hz")
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not np.isfinite(hz) or hz <= 0:
        p.append("contract.control_hz must be a positive finite number")
        hz = None
    # actions
    try:
        a = np.asarray(d["actions"], dtype="<f8")
    except (TypeError, ValueError):
        return p + ["actions is not a numeric array"]
    if a.ndim != 2 or a.shape[1] != N_JOINTS or a.shape[0] < 1:
        return p + [f"actions must have shape (T>=1, {N_JOINTS}); got {a.shape}"]
    if not np.all(np.isfinite(a)):
        p.append("actions contain NaN/inf")
    elif d["action_sha256"] != array_sha256(a):
        p.append("action_sha256 does not match the actions array")
    T = a.shape[0]
    if not _is_hex(d["action_sha256"], 64):
        p.append("action_sha256 is not 64 lowercase hex characters")
    if hz is not None and _is_hex(d["action_sha256"], 64):
        if d["segment_id"] != make_segment_id(d["action_sha256"], hz):
            p.append("segment_id != first 16 hex of sha256(action_sha256 + contract_id)")
    ch = d["chunk"]
    if not isinstance(ch, dict):
        p.append("chunk is not an object")
    else:
        if ch.get("length") != T:
            p.append(f"chunk.length {ch.get('length')!r} != T {T}")
        if ch.get("start_step") != 0:
            p.append("chunk.start_step must be 0")
        h = ch.get("execution_horizon_steps")
        if isinstance(h, bool) or not isinstance(h, int) or not 1 <= h <= T:
            p.append(f"chunk.execution_horizon_steps must be an int in [1, {T}]")
        for k in ("temporal_ensembling", "replanning"):
            if ch.get(k) is not False:
                p.append(f"chunk.{k} must be false")
    pol = d["policy"]
    if not isinstance(pol, dict):
        p.append("policy is not an object")
    else:
        p += [f"missing policy.{k}" for k in POLICY_KEYS if k not in pol]
        if pol.get("kind") not in POLICY_KINDS:
            p.append(f"policy.kind {pol.get('kind')!r} not in {POLICY_KINDS}")
        if not isinstance(pol.get("trained_for_task"), bool):
            p.append("policy.trained_for_task must be a bool")
        if not isinstance(pol.get("checkpoint_id"), str):
            p.append("policy.checkpoint_id must be a string")
        for k in ("checkpoint_sha256", "preprocessing_sha256", "config_sha256"):
            if pol.get(k) is not None and not _is_hex(pol[k], 64):
                p.append(f"policy.{k} must be null or 64 hex characters")
        if pol.get("kind") in (SMOKE_KIND, "fixture_stub") and pol.get("trained_for_task") is not False:
            p.append(f"policy.kind {pol.get('kind')!r} can never be trained_for_task")
        if pol.get("kind") == "trained" and pol.get("trained_for_task") is True and not pol.get("checkpoint_sha256"):
            p.append("a trained policy must carry checkpoint_sha256")
        if d["data_origin"] == "synthetic_fixture" and pol.get("trained_for_task") is not False:
            p.append("a synthetic_fixture segment can never claim trained_for_task")
        if pol.get("kind") in (SMOKE_KIND, "fixture_stub") or d["data_origin"] == "synthetic_fixture":
            if "SYNTHETIC" not in str(d["provenance_label"]).upper():
                p.append("synthetic / smoke segments must carry a SYNTHETIC provenance_label")
        if pol.get("kind") == SMOKE_KIND and d["provenance_label"] != SEGMENT_SMOKE_PROVENANCE:
            p.append("smoke policy output must carry the standard SYNTHETIC SMOKE-TEST provenance_label")
        if pol.get("seed") is not None and (isinstance(pol["seed"], bool) or not isinstance(pol["seed"], int)):
            p.append("policy.seed must be null or int")
    inp = d["input"]
    if not isinstance(inp, dict):
        p.append("input is not an object")
    else:
        try:
            s = np.asarray(inp.get("state_values"), dtype="<f8")
            if s.shape != (N_JOINTS,) or not np.all(np.isfinite(s)):
                p.append(f"input.state_values must be {N_JOINTS} finite floats")
            elif inp.get("state_sha256") != array_sha256(s):
                p.append("input.state_sha256 does not match input.state_values")
        except (TypeError, ValueError):
            p.append("input.state_values is not numeric")
        ref = inp.get("state_ref")
        if not isinstance(ref, dict) or any(k not in ref for k in ("episode_id", "frame_index", "t_host_mono_ns")):
            p.append("input.state_ref needs episode_id, frame_index, t_host_mono_ns (null allowed)")
        imgs = inp.get("images")
        if not isinstance(imgs, dict):
            p.append("input.images must be an object")
        else:
            for cam, v in imgs.items():
                if not (isinstance(v, dict) and _is_hex(v.get("sha256"), 64) and isinstance(v.get("shape"), list)
                        and len(v["shape"]) == 3 and v["shape"][2] == 3 and all(isinstance(x, int) and x > 0 for x in v["shape"])):
                    p.append(f"input.images[{cam!r}] needs sha256 (64 hex) and shape [h, w, 3]")
    if isinstance(d["created_wall_ns"], bool) or not isinstance(d["created_wall_ns"], int):
        p.append("created_wall_ns must be an int")
    if not _is_hex(d["segment_id"]):
        p.append("segment_id must be lowercase hex")
    try:
        canonical(d)  # NaN/inf anywhere (not just in actions) makes the file non-portable JSON
    except ValueError:
        p.append("segment contains NaN/inf")
    return p


def stub_policy(checkpoint_id: str = "fixture_stub") -> dict:
    """Policy block for hand-made test segments (e.g. a scripted ramp). Never trained."""
    return {"kind": "fixture_stub", "trained_for_task": False, "checkpoint_id": checkpoint_id, "framework": "none (scripted stub)"}


__all__ = ["JOINT_NAMES", "array_sha256", "build_action_segment", "image_sha256", "load_action_segment", "make_segment_id",
           "stub_policy", "validate_action_segment_dict", "write_action_segment"]
