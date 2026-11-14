# Configurable depth-age and repeat-rate checks

Config section `quality_checks` (frozen into the episode's `config_snapshot`). **All defaults are `null` = not checked.**
No default threshold is physically validated; a configured threshold is an *operator-chosen* criterion, and the
acceptance report says so. Override per run: `collector acceptance --max-depth-repeat-fraction F --max-depth-age-ms MS
--max-consecutive-repeated-depth N`.

| key | semantics | PASS / FAIL |
|---|---|---|
| `max_depth_repeat_fraction` (0..1) | repeated-previous-depth colour frames / colour frames that have depth | FAIL if above |
| `max_depth_age_ms` (>= 0) | max over frames of *depth age* (below) | FAIL if above |
| `max_consecutive_repeated_depth` (int >= 0) | longest run of consecutive repeated-depth colour frames | FAIL if above |

## Definitions

* A colour frame's depth is **fresh** if its depth frame number differs from the previous colour frame's, **repeated_previous_depth**
  if the same depth frame number is paired again, **missing** if no depth was paired. Missing is never counted as a valid zero.
* **Depth age** = colour-frame host-monotonic receipt time minus the host receipt time of the *first* colour frame that
  carried that depth frame number. Fresh frames have age 0. It is a **lower bound** on true staleness (the depth frame may
  have been captured earlier than first receipt). It is computed on the host-monotonic domain only; device clock domains are
  never mixed. Frames without `host_mono_ns` get age `null` and are excluded (and the check is NOT_CHECKED if none remain).
* A depth frame number that goes backwards is counted (`depth_frame_number_non_monotonic`) and reported.

## Verdicts

* threshold null -> `NOT_CHECKED`
* nothing measurable (no depth stream / no host timestamps) -> `NOT_CHECKED`
* configured and measurable -> `PASS` or `FAIL`
* the pre-existing "depth frames fresh" check is kept: `FAIL` on missing depth, `WARN` on repeats.

Tests: `tests/test_milestone_a.py`.
