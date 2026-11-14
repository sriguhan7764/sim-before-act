"""Sequential counterfactual verifier.

Keeps the proposed action FIXED, samples initial-state/dynamics uncertainty, runs the same chunk in each reset
simulator instance, and summarises. Verification status is kept separate from the EXECUTE/REJECT decision (see runtime.gate).
A recorded outcome is never used here; it only labels the recorded action in the trace.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import ALLOWED_BUDGETS, Config
from ..robot.action_contract import ActionContract
from ..schemas.record import ReplayRecord, all_finite
from ..twin.perturbation import PerturbationSampler
from ..util import hash_obj, to_jsonable
from .estimator import RiskEstimate, estimate

STATUSES = ("COMPLETED", "INVALID_INPUT", "INVALID_STATE", "SIMULATION_ERROR", "TIMEOUT", "INSUFFICIENT_VALID_ROLLOUTS")


@dataclass
class VerificationResult:
    status: str
    problems: list[str]
    action_id: str | None
    state_hash: str | None
    verification_config_hash: str
    simulated_failure_fraction: float | None      # None unless status == COMPLETED
    estimate: RiskEstimate | None                 # diagnostic counts (also for failed statuses when available)
    interval: dict | None
    seeds: dict
    perturbation: dict
    rollouts: list[dict]
    timings: dict
    twin: dict
    horizon: dict
    state_age_s: float | None
    state_flags: list[str] = field(default_factory=list)
    independent_limits_ok: bool = False
    action_validated: bool = False
    state_validated: bool = False


def state_snapshot_hash(rec: ReplayRecord) -> str:
    return hash_obj({"names": rec.state.joint_names, "q": rec.state.positions, "units": rec.state.units,
                     "t": rec.state.timestamp_s, "clock": rec.state.clock_domain, "v": rec.state.velocities,
                     "objects": [(o.name, o.position_m, o.quat_wxyz, o.frame, o.source) for o in rec.objects]})


class Verifier:
    def __init__(self, twin, contract: ActionContract, cfg: Config):
        self.twin, self.contract, self.cfg = twin, contract, cfg
        v = cfg.section("verification")
        self.n = int(v["n_rollouts"])
        self.base_seed = int(v["base_seed"])
        self.min_valid_fraction = float(v.get("min_valid_fraction", 0.5))
        self.timeout_s = float(v.get("verification_timeout_s", 120.0))
        self.interval_cfg = v.get("interval", {"method": "wilson", "confidence": 0.95})
        self.max_rollouts = max(self.n, max(v.get("allowed_budgets", ALLOWED_BUDGETS)))
        self.config_hash = hash_obj({k: cfg.raw[k] for k in ("twin", "contract", "rollout", "state", "perturbations", "verification")}
                                    | {"asset_hash": getattr(twin, "asset_hash", None)})

    # ------------------------------------------------------------------------------------------------------
    def _check_input(self, rec: ReplayRecord) -> tuple[np.ndarray | None, list[str]]:
        a = rec.action
        arr, errs = self.contract.validate_chunk(a.chunk, a.joint_names, a.units, a.representation, a.control_hz)
        if rec.task.target_xyz_m is None or rec.task.tolerance_m is None:
            errs.append("task target_xyz_m / tolerance_m missing: no outcome predicate can be evaluated")
        elif not (all_finite(rec.task.target_xyz_m) and len(rec.task.target_xyz_m) == 3 and math.isfinite(rec.task.tolerance_m)
                  and rec.task.tolerance_m > 0):
            errs.append("task target must be 3 finite numbers and tolerance a positive finite number")
        return arr, errs

    def _check_state(self, rec: ReplayRecord) -> tuple[list[str], list[str], float | None]:
        errs, flags = [], []
        s, sc = rec.state, self.cfg.section("state")
        if tuple(s.joint_names) != self.contract.joint_names:
            errs.append("state joint names/order do not match the action contract")
        if tuple(s.units) != self.contract.units:
            errs.append("state units do not match the action contract")
        if len(s.positions) != len(self.contract.joint_names) or not all_finite(s.positions):
            errs.append("state positions malformed or non-finite")
        else:
            lo, hi = np.asarray(self.contract.limits_lo), np.asarray(self.contract.limits_hi)
            p = np.asarray(s.positions, dtype=float)
            tol = float(self.cfg.section("rollout").get("init_limit_tolerance_deg", 2.0))
            if np.any(p < lo - tol) or np.any(p > hi + tol):
                errs.append("measured state outside simulator joint limits (beyond tolerance)")
        if s.velocities is None:
            flags.append("velocities_not_recorded: simulator starts at rest")
        age = rec.action.issue_timestamp_s - s.timestamp_s
        if not math.isfinite(age):
            errs.append("state/issue timestamps non-finite")
            age = None
        elif age < 0:
            errs.append(f"state timestamp is after the action issue time (age {age:.3f}s)")
        elif age > float(sc["max_state_age_s"]):
            errs.append(f"state too old: {age:.3f}s > max_state_age_s {sc['max_state_age_s']}")
        names = {o.name: o for o in rec.objects}
        for req in sc.get("require_objects", []):
            o = names.get(req)
            if o is None:
                errs.append(f"required object pose missing: '{req}'")
                continue
            if o.frame != sc.get("object_frame", "robot_base"):
                errs.append(f"object '{req}' frame {o.frame!r} != {sc.get('object_frame')!r}")
            if not (len(o.position_m) == 3 and all_finite(o.position_m) and len(o.quat_wxyz) == 4 and all_finite(o.quat_wxyz)
                    and np.linalg.norm(o.quat_wxyz) > 1e-6):
                errs.append(f"object '{req}' pose malformed or non-finite")
            if o.position_sigma_m is None:
                flags.append(f"object '{req}' pose uncertainty unknown (not assumed zero)")
        return errs, flags, age

    def _result(self, status, problems, t0, **kw) -> VerificationResult:
        base = dict(action_id=None, state_hash=None, simulated_failure_fraction=None, estimate=None, interval=None,
                    seeds={"base_seed": self.base_seed}, perturbation={}, rollouts=[], twin=self._twin_desc(),
                    horizon=self._horizon(), state_age_s=None)
        base.update(kw)
        base["timings"] = {**kw.get("timings", {}), "verification_total_s": time.monotonic() - t0}
        return VerificationResult(status=status, problems=problems, verification_config_hash=self.config_hash, **base)

    def _twin_desc(self) -> dict:
        return self.twin.describe() if hasattr(self.twin, "describe") else {}

    def _horizon(self) -> dict:
        return {"type": "action_chunk_plus_settle", "settle_s": self.cfg.section("rollout").get("settle_s"),
                "claim": "chunk-horizon end-effector/safety predicates only; NOT a full-task success prediction"}

    # ------------------------------------------------------------------------------------------------------
    def verify(self, rec: ReplayRecord, n_rollouts: int | None = None) -> VerificationResult:
        t0 = time.monotonic()
        n = int(n_rollouts or self.n)
        arr, in_errs = self._check_input(rec)
        if in_errs or arr is None:
            return self._result("INVALID_INPUT", in_errs, t0, state_hash=state_snapshot_hash(rec))
        action_id = self.contract.action_id(arr)
        st_errs, flags, age = self._check_state(rec)
        shash = state_snapshot_hash(rec)
        if st_errs:
            return self._result("INVALID_STATE", st_errs, t0, action_id=action_id, state_hash=shash, state_age_s=age,
                                state_flags=flags, action_validated=True)

        sampler = PerturbationSampler(self.cfg.section("perturbations"), self.base_seed, max(self.max_rollouts, n))
        objs = {o.name: {"position_m": o.position_m, "quat_wxyz": o.quat_wxyz} for o in rec.objects}
        init = np.asarray(rec.state.positions, dtype=float)
        target = np.asarray(rec.task.target_xyz_m, dtype=float)
        deadline = t0 + self.timeout_s
        results, summaries, times = [], [], []
        timed_out = False
        t_roll = time.monotonic()
        for i in range(n):
            if time.monotonic() > deadline:
                timed_out = True
                break
            smp = sampler.sample(i, arr.shape[0])
            r = self.twin.run_rollout(self.contract, init, objs, arr, smp, rec.action.control_hz, target,
                                      rec.task.tolerance_m, deadline)
            results.append(r)
            times.append(r.wall_s)
            summaries.append({"index": i, "outcome": r.outcome, "task_failure": r.task_failure,
                              "safety_violations": r.safety_violations, "ee_final_m": r.ee_final_m,
                              "ee_error_m": r.ee_error_m, "detail": r.detail, "wall_s": r.wall_s,
                              "perturbation": to_jsonable(smp.summary()), "seed_spawn_key": list(smp.spawn_key)})
            if r.outcome == "timeout":
                timed_out = True
                break
        est = estimate(results, n, float(self.interval_cfg.get("confidence", 0.95)), self.interval_cfg.get("method", "wilson"))
        rollout_s = time.monotonic() - t_roll

        problems: list[str] = []
        if timed_out:
            status = "TIMEOUT"
            problems.append(f"verification timed out after {len(results)}/{n} rollouts; no estimate reported")
        elif est.valid == 0:
            if est.simulator_errors == est.attempted:
                status, problems = "SIMULATION_ERROR", [f"all {est.attempted} rollouts raised simulator errors: " + results[0].detail]
            elif est.invalid_init == est.attempted:
                status, problems = "INVALID_STATE", ["every sampled initial state was invalid"]
            else:
                status, problems = "INSUFFICIENT_VALID_ROLLOUTS", ["no valid rollouts completed; risk unavailable (not zero)"]
        elif est.valid / n < self.min_valid_fraction:
            status = "INSUFFICIENT_VALID_ROLLOUTS"
            problems = [f"valid rollouts {est.valid}/{n} below min_valid_fraction {self.min_valid_fraction}"]
        else:
            status = "COMPLETED"
        ok = status == "COMPLETED"
        return self._result(
            status, problems, t0, action_id=action_id, state_hash=shash, state_age_s=age, state_flags=flags,
            simulated_failure_fraction=est.simulated_failure_fraction if ok else None, estimate=est,
            interval=est.interval if ok else None,
            seeds={"base_seed": self.base_seed, "scheme": "SeedSequence(base_seed).spawn(max_rollouts)[i]",
                   "max_rollouts": sampler.max_rollouts, "spawn_keys": [s["seed_spawn_key"] for s in summaries]},
            perturbation=sampler.describe(), rollouts=summaries,
            timings={"rollout_total_s": rollout_s, "rollout_mean_s": float(np.mean(times)) if times else None,
                     "rollout_max_s": float(np.max(times)) if times else None, "n_attempted": len(results)},
            action_validated=True, state_validated=True, independent_limits_ok=True)
