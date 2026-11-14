"""Decision gate and guarded execution.

Verification status (verifier) is separate from the decision (EXECUTE/REJECT). The gate is fail-closed: anything
missing, unconfigured, stale or unverified results in REJECT. Equality with the threshold rejects.

An approval is bound to: action id (hash of the exact chunk), state snapshot hash + timestamp, verification config
hash, validity period, and a permitted state deviation. `GuardedExecutor` is the ONLY path to an ExecutionAdapter and
re-checks all of these. A changed action is a new candidate and must be re-verified; nothing here clips or repairs.

Offline milestone: no hardware adapter exists. EXECUTE means "would execute". The hold/stop behaviour applied on
REJECT for a real controller is NOT defined here and must be specified and validated before any hardware mode.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..robot.action_contract import ActionContract
from ..verification.verifier import VerificationResult


@dataclass(frozen=True)
class Approval:
    action_id: str
    state_hash: str
    state_timestamp_s: float
    verification_config_hash: str
    issued_at: float
    expires_at: float
    permitted_state_deviation_deg: float
    approved_state_deg: tuple[float, ...]


@dataclass
class Decision:
    decision: str                   # EXECUTE | REJECT
    reason_codes: list[str]
    reason: str
    risk_threshold: float | None
    threshold_calibrated: bool
    simulated_failure_fraction: float | None
    approval: Approval | None = None
    decision_rule: dict = field(default_factory=dict)
    hold_stop_behaviour: str = "NOT DEFINED offline: rejected actions are simply not sent; controller hold/stop behaviour must be specified and validated before any hardware mode"


class ExecutionAdapter(Protocol):
    def send_chunk(self, chunk: np.ndarray) -> None: ...


@dataclass
class ExecutionOutcome:
    executed: bool
    reason_codes: list[str] = field(default_factory=list)


class Gate:
    def __init__(self, gate_cfg: dict, permitted_state_deviation_deg: float, clock: Callable[[], float] = time.monotonic):
        self.cfg = gate_cfg
        thr = gate_cfg.get("risk_threshold")
        if thr is not None and not (isinstance(thr, (int, float)) and math.isfinite(thr) and 0.0 <= thr <= 1.0):
            raise ValueError("gate.risk_threshold must be null or a number in [0, 1]")
        self.threshold = thr
        self.calibrated = bool(gate_cfg.get("threshold_calibrated", False))
        self.validity_s = float(gate_cfg.get("approval_validity_s", 1.0))
        self.sv_limit = gate_cfg.get("reject_on_safety_violation_fraction")
        self.permitted_dev = float(permitted_state_deviation_deg)
        self.clock = clock

    def rule(self) -> dict:
        return {
            "statistic_used": "point estimate of simulated_failure_fraction (failed_valid / valid; failed = task failure OR safety violation)",
            "comparison": "EXECUTE only if fraction < threshold (strict); equality rejects",
            "uses_wilson_upper_bound": False,
            "threshold": self.threshold,
            "threshold_calibrated": self.calibrated,
            "threshold_note": "demo value, NOT selected from calibration data" if not self.calibrated else "selected on calibration data",
            "separate_safety_violation_rule": None if self.sv_limit is None else
                f"also REJECT if safety_violation_fraction >= {self.sv_limit}",
            "other_required_conditions": ["valid action", "valid and fresh state", "verification status COMPLETED",
                                          "independent limits satisfied", "binding ids present"],
        }

    def decide(self, v: VerificationResult, state_timestamp_s: float, approved_state_deg) -> Decision:
        codes: list[str] = []
        if not v.action_validated:
            codes.append("ACTION_INVALID")
        if not v.state_validated:
            codes.append("STATE_INVALID_OR_STALE")
        if v.status != "COMPLETED":
            codes.append(f"VERIFICATION_{v.status}")
        if not v.independent_limits_ok:
            codes.append("INDEPENDENT_LIMITS_NOT_SATISFIED")
        frac = v.simulated_failure_fraction
        if v.status == "COMPLETED" and frac is None:
            codes.append("RISK_UNAVAILABLE")
        if self.threshold is None:
            codes.append("THRESHOLD_NOT_CONFIGURED")
        elif frac is not None and not frac < self.threshold:
            codes.append("RISK_AT_OR_ABOVE_THRESHOLD")
        if self.sv_limit is not None and v.estimate is not None and v.estimate.safety_violation_fraction is not None \
                and v.estimate.safety_violation_fraction >= float(self.sv_limit):
            codes.append("SAFETY_VIOLATION_FRACTION_AT_OR_ABOVE_LIMIT")
        if v.action_id is None or v.state_hash is None:
            codes.append("MISSING_BINDING_IDS")
        if codes:
            return Decision("REJECT", codes, "; ".join(codes) + (" | " + " | ".join(v.problems) if v.problems else ""),
                            self.threshold, self.calibrated, frac, decision_rule=self.rule())
        now = self.clock()
        appr = Approval(v.action_id, v.state_hash, float(state_timestamp_s), v.verification_config_hash, now,
                        now + self.validity_s, self.permitted_dev, tuple(float(x) for x in approved_state_deg))
        why = f"simulated failure fraction {frac:.4f} < threshold {self.threshold}" + \
              ("" if self.calibrated else " (threshold is an UNCALIBRATED demo value)")
        return Decision("EXECUTE", ["APPROVED"], why, self.threshold, self.calibrated, frac, appr, self.rule())


class GuardedExecutor:
    """Sole path to an adapter. Re-checks approval binding at execution time."""

    def __init__(self, contract: ActionContract, adapter: ExecutionAdapter, clock: Callable[[], float] = time.monotonic):
        self._contract, self._adapter, self._clock = contract, adapter, clock

    def execute(self, decision: Decision, chunk, current_state_deg, verification_config_hash: str | None = None) -> ExecutionOutcome:
        a = decision.approval
        if decision.decision != "EXECUTE" or a is None:
            return ExecutionOutcome(False, ["NOT_APPROVED"])
        codes = []
        arr, errs = self._contract.validate_chunk(chunk)
        if errs or arr is None:
            return ExecutionOutcome(False, ["ACTION_INVALID"])
        if self._contract.action_id(arr) != a.action_id:
            codes.append("ACTION_MODIFIED_AFTER_VERIFICATION")
        if verification_config_hash is not None and verification_config_hash != a.verification_config_hash:
            codes.append("VERIFICATION_CONFIG_MISMATCH")
        if self._clock() > a.expires_at:
            codes.append("APPROVAL_EXPIRED")
        cur = np.asarray(current_state_deg, dtype=float)
        ref = np.asarray(a.approved_state_deg, dtype=float)
        if cur.shape != ref.shape or not np.all(np.isfinite(cur)):
            codes.append("CURRENT_STATE_INVALID")
        elif np.max(np.abs(cur[:5] - ref[:5])) > a.permitted_state_deviation_deg:
            codes.append("STATE_DIVERGED_FROM_APPROVED")
        if codes:
            return ExecutionOutcome(False, codes)
        self._adapter.send_chunk(arr)
        return ExecutionOutcome(True, ["EXECUTED"])
