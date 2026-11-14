"""Replay record schema (what the offline verifier consumes).

A record holds ONE recorded pre-action state, the action chunk that was actually executed from that state,
and (optionally) the observed outcome of that same action. The outcome labels the recorded action only.
Nothing here fabricates data: missing object poses, outcomes, velocities or confidence stay None/empty and are
reported by the verifier.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

RECORD_SCHEMA_VERSION = "sba-record/0.1"


class RecordError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class ObjectState:
    name: str
    position_m: list[float]
    quat_wxyz: list[float]
    frame: str                       # must be the robot-base frame for this slice
    source: str                      # provenance: e.g. "synthetic_fixture", "operator_measured", "fiducial:<id>"
    position_sigma_m: float | None = None   # estimate uncertainty if known (None = unknown, not zero)


@dataclass
class RobotState:
    joint_names: list[str]
    positions: list[float]
    units: list[str]
    timestamp_s: float
    clock_domain: str
    velocities: list[float] | None = None   # None = not available (simulator starts at rest, flagged)


@dataclass
class ActionSpec:
    chunk: list[list[float]]
    joint_names: list[str]
    units: list[str]
    representation: str
    control_hz: float
    issue_timestamp_s: float
    policy_id: str | None = None
    policy_confidence: float | None = None   # nullable: never fabricated


@dataclass
class TaskSpec:
    name: str
    instruction: str
    target_xyz_m: list[float] | None         # chunk-horizon predicate target for the end-effector, robot-base frame
    tolerance_m: float | None
    target_source: str | None = None


@dataclass
class OutcomeLabel:
    observed_success: bool | None = None
    failure_type: str | None = None
    label_source: str | None = None
    horizon: str | None = None
    applies_to: str = "recorded_action_only"


@dataclass
class ReplayRecord:
    schema_version: str
    record_id: str
    episode_id: str
    session_id: str
    chunk_id: str
    split: str | None
    source_dataset: dict
    timestamp_wall_iso: str | None
    synthetic: bool
    synthetic_label: str | None
    state: RobotState
    objects: list[ObjectState]
    action: ActionSpec
    task: TaskSpec
    outcome: OutcomeLabel = field(default_factory=OutcomeLabel)
    calibration_ref: dict | None = None
    image_refs: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "ReplayRecord":
        problems: list[str] = []

        def need(obj, key, types, where):
            if not isinstance(obj, dict) or key not in obj:
                problems.append(f"{where}.{key}: missing")
                return None
            v = obj[key]
            if v is not None and not isinstance(v, types):
                problems.append(f"{where}.{key}: expected {types}, got {type(v).__name__}")
                return None
            return v

        if not isinstance(d, dict):
            raise RecordError(["record must be a JSON object"])
        if d.get("schema_version") != RECORD_SCHEMA_VERSION:
            problems.append(f"schema_version: expected {RECORD_SCHEMA_VERSION}, got {d.get('schema_version')!r}")
        for k in ("record_id", "episode_id", "session_id", "chunk_id"):
            need(d, k, str, "record")
        syn = need(d, "synthetic", bool, "record")
        st, ac, tk = d.get("state"), d.get("action"), d.get("task")
        for k in ("joint_names", "positions", "units"):
            need(st, k, list, "state")
        for k in ("timestamp_s",):
            need(st, k, (int, float), "state")
        need(st, "clock_domain", str, "state")
        for k in ("chunk", "joint_names", "units"):
            need(ac, k, list, "action")
        for k in ("representation",):
            need(ac, k, str, "action")
        for k in ("control_hz", "issue_timestamp_s"):
            need(ac, k, (int, float), "action")
        need(tk, "name", str, "task")
        need(tk, "instruction", str, "task")
        if problems:
            raise RecordError(problems)
        objs = []
        for i, o in enumerate(d.get("objects") or []):
            for k, t in (("name", str), ("position_m", list), ("quat_wxyz", list), ("frame", str), ("source", str)):
                need(o, k, t, f"objects[{i}]")
        if problems:
            raise RecordError(problems)
        for o in d.get("objects") or []:
            objs.append(ObjectState(o["name"], [float(x) for x in o["position_m"]], [float(x) for x in o["quat_wxyz"]],
                                    o["frame"], o["source"], o.get("position_sigma_m")))
        out = d.get("outcome") or {}
        return ReplayRecord(
            schema_version=d["schema_version"], record_id=d["record_id"], episode_id=d["episode_id"],
            session_id=d["session_id"], chunk_id=d["chunk_id"], split=d.get("split"),
            source_dataset=d.get("source_dataset") or {}, timestamp_wall_iso=d.get("timestamp_wall_iso"),
            synthetic=bool(syn), synthetic_label=d.get("synthetic_label"),
            state=RobotState(list(st["joint_names"]), list(st["positions"]), list(st["units"]), float(st["timestamp_s"]),
                             st["clock_domain"], st.get("velocities")),
            objects=objs,
            action=ActionSpec(list(ac["chunk"]), list(ac["joint_names"]), list(ac["units"]), ac["representation"],
                              float(ac["control_hz"]), float(ac["issue_timestamp_s"]), ac.get("policy_id"),
                              ac.get("policy_confidence")),
            task=TaskSpec(tk["name"], tk["instruction"], tk.get("target_xyz_m"), tk.get("tolerance_m"),
                          tk.get("target_source")),
            outcome=OutcomeLabel(out.get("observed_success"), out.get("failure_type"), out.get("label_source"),
                                 out.get("horizon"), out.get("applies_to", "recorded_action_only")),
            calibration_ref=d.get("calibration_ref"), image_refs=list(d.get("image_refs") or []),
            limitations=list(d.get("limitations") or []))


def load_record(path: str | Path) -> ReplayRecord:
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RecordError([f"cannot read record {path}: {e}"]) from e
    return ReplayRecord.from_dict(d)


def all_finite(values) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False
