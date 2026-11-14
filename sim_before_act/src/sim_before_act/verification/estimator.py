"""Risk estimator: simulated failure fraction with a Wilson interval.

simulated_failure_fraction = failed_valid_rollouts / valid_rollouts
  "failed" = task failure OR safety-constraint violation among VALID rollouts.
Simulator errors, invalid initialisations, timeouts and unknown outcomes are NOT failures and NOT valid rollouts;
they are counted separately. With zero valid rollouts the fraction is None (unavailable), never 0.

This is a simulated failure fraction, not a calibrated probability of physical failure. The Wilson interval
quantifies Monte-Carlo sampling error given the simulator and perturbation model only; it says nothing about
simulator model error or sim-to-real gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..twin.mujoco_twin import RolloutResult

Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    if n <= 0:
        return None
    z = Z.get(round(confidence, 2))
    if z is None:
        raise ValueError(f"confidence must be one of {sorted(Z)}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class RiskEstimate:
    requested: int
    attempted: int
    valid: int
    failed: int                    # failed among valid (task OR safety)
    task_failures: int
    safety_violations: int
    simulator_errors: int
    invalid_init: int
    timeouts: int
    unknown: int
    simulated_failure_fraction: float | None
    task_failure_fraction: float | None
    safety_violation_fraction: float | None
    interval: dict | None
    violation_breakdown: dict = field(default_factory=dict)


def estimate(results: list[RolloutResult], requested: int, confidence: float = 0.95, method: str = "wilson") -> RiskEstimate:
    if method != "wilson":
        raise ValueError("only the Wilson score interval is implemented")
    valid = [r for r in results if r.valid]
    n = len(valid)
    tf = sum(1 for r in valid if r.task_failure)
    sv = sum(1 for r in valid if r.safety_violations)
    failed = sum(1 for r in valid if r.failed)
    breakdown: dict[str, int] = {}
    for r in valid:
        for v in r.safety_violations:
            breakdown[v] = breakdown.get(v, 0) + 1
    cnt = lambda o: sum(1 for r in results if r.outcome == o)
    frac = (lambda k: k / n) if n else (lambda k: None)
    iv = wilson_interval(failed, n, confidence)
    return RiskEstimate(
        requested=requested, attempted=len(results), valid=n, failed=failed, task_failures=tf, safety_violations=sv,
        simulator_errors=cnt("simulator_error"), invalid_init=cnt("invalid_init"), timeouts=cnt("timeout"),
        unknown=cnt("unknown"), simulated_failure_fraction=frac(failed), task_failure_fraction=frac(tf),
        safety_violation_fraction=frac(sv),
        interval=None if iv is None else {"method": "wilson", "confidence": confidence, "low": iv[0], "high": iv[1],
                                          "scope": "Monte-Carlo sampling error only; excludes simulator model error"},
        violation_breakdown=breakdown)
