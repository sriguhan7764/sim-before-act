"""Offline replay pipeline: record -> verify -> gate -> trace. Execution is only possible via GuardedExecutor."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..robot.action_contract import ActionContract
from ..schemas.record import ReplayRecord
from ..traces.writer import build_trace, write_trace
from ..twin.mujoco_twin import MujocoTwin
from ..verification.verifier import Verifier
from .gate import Decision, ExecutionAdapter, ExecutionOutcome, Gate, GuardedExecutor

PROJECT_DIR = Path(__file__).resolve().parents[3]


def build_stack(cfg: Config, twin=None):
    if twin is not None and getattr(twin, "rcfg", cfg.section("rollout")) != cfg.section("rollout"):
        raise ValueError("supplied twin was built with a different 'rollout' config than cfg; build a matching twin")
    twin = twin or MujocoTwin(cfg.scene_xml(), cfg.section("twin"), cfg.section("rollout"))
    contract = ActionContract.build(cfg.section("contract"), twin.joint_limits_deg, twin.gripper_range_rad)
    return twin, contract, Verifier(twin, contract, cfg), Gate(cfg.section("gate"), cfg.section("state")["permitted_state_deviation_deg"])


@dataclass
class AttemptResult:
    verification: object
    decision: Decision
    execution: ExecutionOutcome
    trace_path: Path | None
    trace: dict


def run_attempt(cfg: Config, rec: ReplayRecord, out_dir: Path | None, stack=None, n_rollouts: int | None = None,
                adapter: ExecutionAdapter | None = None, write: bool = True) -> AttemptResult:
    """One attempt. If an adapter is supplied (tests only) an approved action may reach it, solely via GuardedExecutor."""
    t0 = time.monotonic()
    twin, contract, verifier, gate = stack or build_stack(cfg)
    ver = verifier.verify(rec, n_rollouts)
    t1 = time.monotonic()
    dec = gate.decide(ver, rec.state.timestamp_s, rec.state.positions)
    t2 = time.monotonic()
    exe = ExecutionOutcome(False, ["OFFLINE_NO_ADAPTER"])
    if adapter is not None:
        exe = GuardedExecutor(contract, adapter).execute(dec, np.asarray(rec.action.chunk, dtype=float), rec.state.positions,
                                                         ver.verification_config_hash)
    timings = {"decision_s": t2 - t1, "verification_stage_s": t1 - t0, "end_to_end_s": time.monotonic() - t0}
    trace = build_trace(rec, contract, ver, dec, exe, cfg, timings, PROJECT_DIR)
    path = write_trace(trace, out_dir) if (write and out_dir is not None) else None
    return AttemptResult(ver, dec, exe, path, trace)
