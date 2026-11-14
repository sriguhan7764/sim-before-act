"""Time-alignment rules for the ACT conversion (pure python). All times are host CLOCK_MONOTONIC nanoseconds.

Rules (documented in docs/ACT_CONVERSION.md; every one is CAUSAL and never interpolates):

* reference = the front camera frame receipt time ``t_ref`` (one control step per front frame);
* observation state = the LATEST VALID follower state sample whose time (midpoint of the bus read) is <= ``t_ref``,
  accepted only if ``t_ref - t_state <= max_state_age``. A newer sample that failed to read is skipped, never repaired;
* wrist image = the latest wrist frame at or before ``t_ref`` with age <= ``wrist_max_age``; one wrist frame may serve at most
  ``wrist_max_reuse`` steps; otherwise the policy ``drop_step`` | ``mark_missing_with_mask`` applies;
* action = the FIRST command stamped (before send) at or after ``t_ref`` with ``t_cmd - t_ref <= max_command_lag``.
  If that command record is invalid the step is dropped (the next command is NOT used instead).
"""
from __future__ import annotations

import bisect

DROP_REASONS = (
    "no_state_at_or_before_ref", "state_too_old",
    "no_command_at_or_after_ref", "command_lag_too_large", "command_invalid",
    "front_image_missing", "front_image_unreadable",
    "wrist_stream_absent", "wrist_no_frame_at_or_before_ref", "wrist_frame_too_old", "wrist_reuse_exceeded", "wrist_image_unreadable",
)


def latest_valid_at_or_before(times: list[int], valid: list[bool], t: int, max_age_ns: int):
    """(index, age_ns, status, n_invalid_skipped). ``times`` must be sorted ascending. status 'ok' or a drop reason suffix."""
    i = bisect.bisect_right(times, t) - 1
    skipped = 0
    while i >= 0 and not valid[i]:
        i -= 1
        skipped += 1
    if i < 0:
        return None, None, "none_at_or_before", skipped
    age = t - times[i]
    if age > max_age_ns:
        return None, age, "too_old", skipped
    return i, age, "ok", skipped


def first_at_or_after(times: list[int], t: int, max_lag_ns: int):
    """(index, lag_ns, status). The FIRST sample with time >= t; refused if its lag exceeds ``max_lag_ns``."""
    i = bisect.bisect_left(times, t)
    if i >= len(times):
        return None, None, "none_at_or_after"
    lag = times[i] - t
    if lag > max_lag_ns:
        return None, lag, "lag_too_large"
    return i, lag, "ok"


def is_non_decreasing(times: list[int]) -> bool:
    return all(b >= a for a, b in zip(times, times[1:]))


def stats(vals: list[float]) -> dict | None:
    """min/mean/median/p95/max of a list (None when empty). Plain python, deterministic."""
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)

    def pct(q):
        k = (n - 1) * q
        lo, hi = int(k), min(int(k) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)

    return {"n": n, "min": s[0], "mean": sum(s) / n, "median": pct(0.5), "p95": pct(0.95), "max": s[-1]}
