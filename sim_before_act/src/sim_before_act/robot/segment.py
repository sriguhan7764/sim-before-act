"""Action segment files ("action-segment-v0.1"): loader, validator and policy-to-simulator adapter.

A segment is the unmodified output of a policy (absolute joint-position targets in the follower target space) plus provenance.
The consumer's job is to refuse anything that was changed after hashing, never to repair it:

* the sha256 of the (T, 6) '<f8' array is RECOMPUTED and must equal ``action_sha256`` (tamper / silent-modification detection);
* values are validated, never clipped (out-of-limit, non-finite, wrong shape, wrong units/order -> INVALID);
* ``policy.kind`` / ``trained_for_task`` / ``data_origin`` are carried into every verification output; an untrained segment
  can never be reported as trained;
* ``chunk.replanning`` / ``temporal_ensembling`` => the executed commands may differ from this chunk (comparison not exact);
* ``execution_horizon_steps <= length`` is enforced.

``policy_to_sim`` converts a VALIDATED segment to simulator joint targets with ``ActionContract.to_sim_rad`` and runs it in the
scene twin. ``state_sha256`` and image hashes are only format-checked here (their preimage is defined by the producer).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .action_contract import GRIPPER_SEMANTICS, JOINT_NAMES, REFERENCE, REPRESENTATION, UNITS, ActionContract

SEGMENT_SCHEMA = "action-segment-v0.1"
DATA_ORIGINS = ("synthetic_fixture", "recorded_physical")
POLICY_KINDS = ("smoke_random_init_few_steps", "trained", "fixture_stub")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SegmentError(ValueError):
    """The file cannot even be parsed as a segment (not JSON / not an object). Content problems are returned by validate_segment."""


def actions_sha256(actions) -> str:
    """sha256 of the C-contiguous '<f8' bytes of the (T, 6) array (the definition used by ``action_sha256``)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(actions, dtype="<f8")).tobytes()).hexdigest()


def contract_block(contract: ActionContract) -> dict:
    """The exact ``contract`` object a producer must write for this contract."""
    return {"joint_names": list(contract.joint_names), "units": list(contract.units), "representation": contract.representation,
            "control_hz": contract.control_hz, "interpolation": contract.interpolation, "reference": contract.reference,
            "gripper": contract.gripper_semantics}


@dataclass(frozen=True)
class ActionSegment:
    raw: dict
    path: Path | None = None

    def get(self, *keys, default=None):
        cur = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @property
    def segment_id(self):
        return self.get("segment_id")


def load_action_segment(path: str | Path) -> ActionSegment:
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SegmentError(f"cannot read segment {p}: {e}") from e
    if not isinstance(raw, dict):
        raise SegmentError(f"segment {p} is not a JSON object")
    return ActionSegment(raw=raw, path=p)


def save_segment(path: str | Path, raw: dict) -> None:
    """Write a segment dict as JSON (floats use repr, so float64 values round-trip exactly)."""
    Path(path).write_text(json.dumps(raw, indent=1, sort_keys=True))


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _hex_ok(v) -> bool:
    return isinstance(v, str) and bool(_HEX64.match(v))


def policy_label(seg: ActionSegment) -> str:
    """Honest label for reports. Never says 'trained' unless kind == trained AND trained_for_task is true."""
    kind, tft = seg.get("policy", "kind"), seg.get("policy", "trained_for_task")
    if kind == "trained" and tft is True:
        return f"TRAINED POLICY for this task (checkpoint_id={seg.get('policy', 'checkpoint_id')})"
    return f"NOT A TRAINED POLICY: kind={kind}, trained_for_task={tft} (smoke/fixture output; no task competence implied)"


def reported_as_trained(seg: ActionSegment) -> bool:
    return seg.get("policy", "kind") == "trained" and seg.get("policy", "trained_for_task") is True


def provenance_label(seg: ActionSegment) -> str:
    return f"[data_origin={seg.get('data_origin')}] {seg.get('provenance_label')} | {policy_label(seg)}"


def segment_flags(seg: ActionSegment) -> list[str]:
    """Non-blocking caveats that must travel with every result computed from this segment."""
    f: list[str] = []
    pol, ch = seg.get("policy") or {}, seg.get("chunk") or {}
    if not reported_as_trained(seg):
        f.append("untrained_policy: output of a smoke/fixture policy; must NOT be reported as a trained policy")
    if seg.get("data_origin") == "synthetic_fixture":
        f.append("synthetic_fixture: invented/synthetic data; not evidence about the physical robot")
    if ch.get("replanning") is True or ch.get("temporal_ensembling") is True:
        f.append("replanning/temporal_ensembling: executed commands may differ from this chunk: comparison not exact")
    if _is_int(ch.get("execution_horizon_steps")) and _is_int(ch.get("length")) and ch["execution_horizon_steps"] < ch["length"]:
        f.append(f"execution_horizon_steps={ch['execution_horizon_steps']} < length={ch['length']}: only the first "
                 "execution_horizon_steps targets are executed/simulated; the hash still covers the full chunk")
    if pol.get("kind") == "trained" and not _hex_ok(pol.get("checkpoint_sha256")):
        f.append("trained_claim_unverifiable: policy.kind=trained but checkpoint_sha256 missing/invalid")
    if _is_int(ch.get("start_step")) and ch["start_step"] != 0:
        f.append(f"chunk.start_step={ch['start_step']} (not 0)")
    return f


def validate_segment(seg: ActionSegment, contract: ActionContract) -> tuple[np.ndarray | None, list[str]]:
    """Returns (float64 (T,6) array or None, blocking problems). The array is None whenever there is any problem (fail closed).

    Nothing is modified or clipped. Flags (non-blocking) come from ``segment_flags``.
    """
    p: list[str] = []
    raw = seg.raw
    if not isinstance(raw, dict):
        return None, ["segment is not an object"]
    if raw.get("schema") != SEGMENT_SCHEMA:
        p.append(f"schema must be {SEGMENT_SCHEMA!r}, got {raw.get('schema')!r}")
    for k in ("segment_id", "data_origin", "provenance_label", "contract", "actions", "action_sha256", "chunk", "policy", "input", "created_wall_ns"):
        if k not in raw:
            p.append(f"missing field '{k}'")
    if not (isinstance(raw.get("segment_id"), str) and raw.get("segment_id")):
        p.append("segment_id must be a non-empty string")
    if raw.get("data_origin") not in DATA_ORIGINS:
        p.append(f"data_origin must be one of {DATA_ORIGINS}")
    if not (isinstance(raw.get("provenance_label"), str) and raw["provenance_label"].strip()):
        p.append("provenance_label must be a non-empty string")
    if not _is_int(raw.get("created_wall_ns")):
        p.append("created_wall_ns must be an integer")

    # contract block ------------------------------------------------------------------------------------------------------
    cb = raw.get("contract")
    names = units = rep = hz = None
    if isinstance(cb, dict):
        names, units, rep, hz = cb.get("joint_names"), cb.get("units"), cb.get("representation"), cb.get("control_hz")
        if not isinstance(names, list):
            p.append("contract.joint_names must be a list")
            names = None
        if not isinstance(units, list):
            p.append("contract.units must be a list")
            units = None
        if not _is_num(hz):
            p.append("contract.control_hz must be a number")
            hz = None
        if not isinstance(rep, str):
            p.append("contract.representation must be a string")
            rep = None
        if isinstance(rep, str) and (rep != REPRESENTATION) and any(w in rep.lower() for w in ("relative", "delta", "residual")):
            p.append(f"relative/delta representation {rep!r} rejected: only {REPRESENTATION!r} is supported (no conversion is attempted)")
            rep = None                       # avoid a second, redundant mismatch message
        if cb.get("interpolation") != contract.interpolation:
            p.append(f"interpolation mismatch: {cb.get('interpolation')!r} != {contract.interpolation!r}")
        if cb.get("reference") != contract.reference:
            p.append(f"reference mismatch: {cb.get('reference')!r} != {contract.reference!r}")
        if cb.get("gripper") != contract.gripper_semantics:
            p.append(f"gripper semantics mismatch: {cb.get('gripper')!r} != {contract.gripper_semantics!r}")
    else:
        p.append("contract must be an object")

    # actions ----------------------------------------------------------------------------------------------------------------
    acts = raw.get("actions")
    arr = None
    typed_ok = (isinstance(acts, np.ndarray) and acts.dtype.kind in "fiu") or (
        isinstance(acts, list) and all(isinstance(r, list) and all(_is_num(v) for v in r) for r in acts))
    if not typed_ok:
        p.append("actions must be a list of lists of numbers (T x 6)")
    else:
        arr, errs = contract.validate_chunk(acts, names, units, rep, hz)
        p += errs
        if arr is not None:
            h = raw.get("action_sha256")
            if not _hex_ok(h):
                p.append("action_sha256 missing or not 64 lowercase hex characters")
            else:
                got = actions_sha256(arr)
                if got != h:
                    p.append(f"action_sha256 mismatch: file says {h[:12]}..., recomputed {got[:12]}...: the actions were modified "
                             "after hashing (an action changed after verification is a NEW candidate and must be re-verified)")
    ch = raw.get("chunk")
    if isinstance(ch, dict):
        T = arr.shape[0] if arr is not None else (len(acts) if isinstance(acts, list) else None)
        if not _is_int(ch.get("length")):
            p.append("chunk.length must be an integer")
        elif T is not None and ch["length"] != T:
            p.append(f"chunk.length={ch['length']} != number of action rows {T}")
        if not (_is_int(ch.get("start_step")) and ch["start_step"] >= 0):
            p.append("chunk.start_step must be an integer >= 0")
        h_ = ch.get("execution_horizon_steps")
        if not _is_int(h_) or h_ < 1:
            p.append("chunk.execution_horizon_steps must be an integer >= 1")
        elif _is_int(ch.get("length")) and h_ > ch["length"]:
            p.append(f"chunk.execution_horizon_steps={h_} exceeds chunk.length={ch['length']}")
        for k in ("temporal_ensembling", "replanning"):
            if not isinstance(ch.get(k), bool):
                p.append(f"chunk.{k} must be a boolean")
    else:
        p.append("chunk must be an object")

    pol = raw.get("policy")
    if isinstance(pol, dict):
        if pol.get("kind") not in POLICY_KINDS:
            p.append(f"policy.kind must be one of {POLICY_KINDS}")
        if not isinstance(pol.get("trained_for_task"), bool):
            p.append("policy.trained_for_task must be a boolean")
        elif pol.get("kind") == "trained" and not pol["trained_for_task"]:
            p.append("policy.kind=trained but trained_for_task=false: an untrained segment cannot be reported as trained")
        elif pol.get("kind") in ("smoke_random_init_few_steps", "fixture_stub") and pol["trained_for_task"]:
            p.append(f"policy.kind={pol.get('kind')} cannot have trained_for_task=true")
        if pol.get("kind") == "fixture_stub" and raw.get("data_origin") != "synthetic_fixture":
            p.append("policy.kind=fixture_stub requires data_origin=synthetic_fixture")
        if not isinstance(pol.get("checkpoint_id"), str):
            p.append("policy.checkpoint_id must be a string")
        for k in ("checkpoint_sha256", "preprocessing_sha256", "config_sha256"):
            if pol.get(k) is not None and not _hex_ok(pol.get(k)):
                p.append(f"policy.{k} must be null or 64 hex characters")
    else:
        p.append("policy must be an object")

    inp = raw.get("input")
    if isinstance(inp, dict):
        sv = inp.get("state_values")
        if not (isinstance(sv, list) and len(sv) == 6 and all(_is_num(v) and math.isfinite(v) for v in sv)):
            p.append("input.state_values must be 6 finite numbers")
        if not _hex_ok(inp.get("state_sha256")):
            p.append("input.state_sha256 must be 64 hex characters")
        if not isinstance(inp.get("state_ref"), dict):
            p.append("input.state_ref must be an object")
        imgs = inp.get("images")
        if not isinstance(imgs, dict):
            p.append("input.images must be an object")
        else:
            for cam, im in imgs.items():
                if not (isinstance(im, dict) and _hex_ok(im.get("sha256")) and isinstance(im.get("shape"), list) and len(im["shape"]) == 3):
                    p.append(f"input.images[{cam!r}] needs sha256 (64 hex) and a 3-element shape")
    else:
        p.append("input must be an object")
    return (None if p else arr), p


class SegmentInvalid(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("action segment INVALID (not clipped, not repaired): " + "; ".join(problems))


@dataclass(frozen=True)
class ValidatedSegment:
    segment_id: str
    actions: np.ndarray                 # read-only (T, 6) float64, contract units, exactly as in the file
    action_sha256: str
    contract_id: str
    execution_horizon_steps: int
    state_values: tuple[float, ...]
    data_origin: str
    provenance_label: str
    policy_kind: str
    trained_for_task: bool
    policy_label: str
    flags: tuple[str, ...]
    exact_comparison: bool
    source_path: str | None = None

    @property
    def executed_actions(self) -> np.ndarray:
        return self.actions[:self.execution_horizon_steps]

    def provenance(self) -> dict:
        return {"segment_id": self.segment_id, "data_origin": self.data_origin, "provenance_label": self.provenance_label,
                "policy_kind": self.policy_kind, "trained_for_task": self.trained_for_task, "policy_label": self.policy_label,
                "reported_as_trained": self.policy_kind == "trained" and self.trained_for_task, "flags": list(self.flags),
                "comparison_exact": self.exact_comparison, "action_sha256": self.action_sha256, "contract_id": self.contract_id,
                "source_path": self.source_path}


def validate_for_sim(seg: ActionSegment, contract: ActionContract) -> ValidatedSegment:
    arr, problems = validate_segment(seg, contract)
    if problems or arr is None:
        raise SegmentInvalid(problems or ["unknown validation failure"])
    a = np.array(arr, dtype=np.float64, copy=True)
    a.setflags(write=False)
    ch = seg.raw["chunk"]
    flags = segment_flags(seg)
    return ValidatedSegment(
        segment_id=seg.raw["segment_id"], actions=a, action_sha256=actions_sha256(a), contract_id=contract.contract_id,
        execution_horizon_steps=int(ch["execution_horizon_steps"]), state_values=tuple(float(v) for v in seg.raw["input"]["state_values"]),
        data_origin=seg.raw["data_origin"], provenance_label=provenance_label(seg), policy_kind=seg.raw["policy"]["kind"],
        trained_for_task=bool(seg.raw["policy"]["trained_for_task"]), policy_label=policy_label(seg), flags=tuple(flags),
        exact_comparison=not (ch["replanning"] or ch["temporal_ensembling"]), source_path=str(seg.path) if seg.path else None)


def to_sim_targets(vs: ValidatedSegment, contract: ActionContract, executed_only: bool = True) -> np.ndarray:
    """Simulator joint-position targets (radians; gripper mapped linearly into the MJCF joint range) via ``to_sim_rad``."""
    return contract.to_sim_rad(vs.executed_actions if executed_only else vs.actions)


def policy_to_sim(segment, scene, seed: int = 0, perturbation=None, init: str = "scene", use_full_chunk: bool = False,
                  scene_cfg=None):
    """Validate a segment against the scene's contract, convert it to simulator targets and run it in the twin.

    ``segment``: ActionSegment | ValidatedSegment | path. ``scene``: built ``Scene`` (or a SceneConfig to build one).
    ``init``: 'scene' (initial configuration from the scene config) or 'segment_state' (the segment's input.state_values; a
    state outside the simulator limits is INVALID, never clipped). Returns a ``SceneRolloutResult`` whose ``action_record`` holds the
    exact action hash used; the hash is re-checked after the run so a modification during the run cannot go unnoticed.
    A segment that fails validation yields an ``invalid_rollout`` result (no simulation is run), never a repaired action.
    """
    from ..twin.scene.runner import run_scene_rollout  # lazy: keeps the robot package free of simulator imports

    return run_scene_rollout(scene if scene_cfg is None else scene_cfg, segment, seed, perturbation, init=init,
                             use_full_chunk=use_full_chunk)


# ---- fixture writer (TESTS ONLY) -----------------------------------------------------------------------------------------------------
def build_fixture_segment(actions, *, segment_id: str = "fixture_segment", contract: ActionContract | None = None, control_hz: float = 30.0,
                          state_values=None, execution_horizon_steps: int | None = None, replanning: bool = False,
                          temporal_ensembling: bool = False,
                          provenance_label: str = "SYNTHETIC TEST FIXTURE: invented actions written by write_fixture_segment; not policy output",
                          created_wall_ns: int | None = None) -> dict:
    a = np.array(actions, dtype=np.float64)
    T = int(a.shape[0])
    cb = contract_block(contract) if contract is not None else {
        "joint_names": list(JOINT_NAMES), "units": list(UNITS), "representation": REPRESENTATION, "control_hz": float(control_hz),
        "interpolation": "zero_order_hold", "reference": REFERENCE, "gripper": GRIPPER_SEMANTICS}
    sv = [float(v) for v in (a[0] if state_values is None else state_values)]
    return {
        "schema": SEGMENT_SCHEMA, "segment_id": segment_id, "data_origin": "synthetic_fixture", "provenance_label": provenance_label,
        "contract": cb, "actions": a.tolist(), "action_sha256": actions_sha256(a),
        "chunk": {"length": T, "start_step": 0, "execution_horizon_steps": T if execution_horizon_steps is None else int(execution_horizon_steps),
                  "temporal_ensembling": bool(temporal_ensembling), "replanning": bool(replanning)},
        "policy": {"kind": "fixture_stub", "trained_for_task": False, "checkpoint_id": "none (fixture)", "checkpoint_sha256": None,
                   "preprocessing_sha256": None, "config_sha256": None, "framework": "none", "device": "none", "seed": None},
        "input": {"state_values": sv, "state_sha256": hashlib.sha256(np.asarray(sv, dtype="<f8").tobytes()).hexdigest(),
                  "state_ref": {"episode_id": None, "frame_index": None, "t_host_mono_ns": None}, "images": {}},
        "created_wall_ns": time.time_ns() if created_wall_ns is None else int(created_wall_ns)}


def write_fixture_segment(path: str | Path, actions, **kw) -> dict:
    """TEST HELPER: write a synthetic fixture segment (policy.kind=fixture_stub, trained_for_task=false, synthetic_fixture)."""
    raw = build_fixture_segment(actions, **kw)
    save_segment(path, raw)
    return copy.deepcopy(raw)
