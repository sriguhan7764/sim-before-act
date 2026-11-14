# Action contract (v0.1)

The single definition of what an action chunk means. Replay, simulation and any future physical execution must share it.
Implemented in `src/sim_before_act/robot/action_contract.py`; defaults in `configs/offline_demo.yaml`.

| Item | Definition |
|---|---|
| Representation | `absolute_joint_position_targets` (not deltas, not torques) |
| Joint order | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| Units | arm joints: **degrees**; gripper: **0-100 normalised** (LeRobot convention; 0 = closed). Never radians |
| Control frequency | 30 Hz default; must equal the record's `control_hz`, else INVALID_INPUT |
| Chunk length | 1-200 steps (configurable); duration = length / control_hz |
| Interpolation | `zero_order_hold` (default: each target held for its control period) or `linear` (from previous target; first period starts from the measured state). The last target is held through `settle_s` |
| Limits | arm: MJCF joint range in degrees (**simulator** limits, not hardware limits; overridable via `limits_override_deg`); gripper 0-100 |
| Speed limit | `max_joint_speed_deg_s`; `null` = check NOT performed (reported in every trace) |
| Normalisation | none in this slice: values are already in physical units. Policy (SmolVLA) de/normalisation is NOT implemented |
| Clipping | **never.** Out-of-range, non-finite or mis-shaped chunks are INVALID_INPUT. A changed action is a new candidate and must be re-verified |
| Action id | sha256 over (contract identity, shape, float64 bytes). Any value change changes the id |

## Extended fields (Milestone E)

`ActionContract.describe()` now states every item explicitly; `contract_id` = sha256 of that description (without the id itself),
alongside `action_id` (which hashes the contract identity + the actions).

| Field | Value |
|---|---|
| `joint_names` / `units` | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper / `degrees` x5, `percent_0_100` |
| `reference` | `follower joint-position target space` (the target the LeRobot follower is commanded to; same space as `action`) |
| `representation` | `absolute_joint_position_targets` only. `relative`/`delta` is **rejected** (`ActionContract.build` and segment validation); no conversion is attempted |
| `gripper_semantics` | `normalized_0_100_0_closed` (UNVALIDATED mapping to the MJCF gripper joint range, see below) |
| `control_hz`, `chunk_duration_s(T)` | 30 Hz default; `T / control_hz` |
| `interpolation` | `zero_order_hold` or `linear` |
| `limits_lo/hi`, `limits_source` | **simulator** (CAD-derived MJCF) limits and where they came from; they are NOT the user's follower calibration limits. Resolving that mismatch is the documented blocker (`STATUS.md`, `robot_collector/docs/CALIBRATION.md`) |

**Targets are not torques.** ACT outputs are absolute joint-position targets in follower target space. The simulator maps them through
`ActionContract.to_sim_rad` into `data.ctrl` of MuJoCo *position* actuators (PD: `kp*(target-q) - kv*qdot`, clipped to `forcerange`,
gains read back from the model and recorded in every result). That is **not** the Feetech STS3215 controller (its internal PID,
speed/acceleration profile, load behaviour and communication latency are not modelled).

## Action segment file (`action-segment-v0.1`, JSON) - consumer: `robot/segment.py`

Producers (e.g. the ACT adapter in the LeRobot env) write; the simulator side only reads. Key fields: `schema`, `segment_id`,
`data_origin` (`synthetic_fixture` | `recorded_physical`), `provenance_label`, `contract` {joint_names, units, representation,
control_hz, interpolation, `reference`, `gripper`}, `actions` (T x 6, unmodified policy output), `action_sha256`, `chunk`
{length, start_step, execution_horizon_steps, temporal_ensembling, replanning}, `policy` {kind, trained_for_task, checkpoint_*,
...}, `input` {state_values, state_sha256, state_ref, images}, `created_wall_ns`.

* **Exact strings.** `contract.reference` must be exactly `follower joint-position target space`, `contract.gripper` exactly
  `normalized_0_100_0_closed`, `interpolation` equal to the simulator contract's. Anything else is reported as a mismatch.
* **Hash rule.** `action_sha256` = sha256 of the C-contiguous little-endian float64 bytes of the (T,6) array. The loader recomputes it;
  a mismatch is INVALID ("modified after hashing: a changed action is a new candidate and must be re-verified"). The hash is stable
  across save/load and is stored in every rollout result (`action_record.action_sha256`, plus the hash of the executed prefix).
* **Validation, never repair.** `validate_segment(seg, contract) -> (array | None, problems)` checks schema, joint order, units,
  representation, `control_hz`, `chunk.length == T`, `execution_horizon_steps <= length`, finite values, limits, and speed (only if
  configured). The array is `None` whenever there is any problem. Out-of-limit values make the segment INVALID; nothing is clipped.
* **Provenance travels with results.** `data_origin`, `provenance_label`, `policy.kind` / `trained_for_task` are copied into every
  rollout result. `trained_for_task: false` (or kind `smoke_random_init_few_steps` / `fixture_stub`) is labelled
  `NOT A TRAINED POLICY` and can never be reported as trained; `kind: trained` with `trained_for_task: false` is invalid.
* **Replanning / temporal ensembling** set `exact_comparison = false` with the flag "executed commands may differ from this chunk:
  comparison not exact".
* **Execution horizon.** By default `policy_to_sim` simulates the first `execution_horizon_steps` targets (the hash covers the whole chunk);
  `use_full_chunk=True` simulates all.
* **Initial state.** `init="scene"` (scene-config initial configuration) or `init="segment_state"` (`input.state_values`). A state outside the
  simulator limits (e.g. the real recorded shoulder_lift of about -103 deg) makes the rollout INVALID; it is never clipped or offset.
* `write_fixture_segment(path, actions, ...)` is a TEST helper: `fixture_stub`, `trained_for_task=false`, `synthetic_fixture`.

## Contract -> MuJoCo controller

The MJCF has six position actuators named like the joints (`ctrlrange` = joint range). Arm targets: `deg -> rad`,
written directly to `ctrl`. Gripper: linear map 0..100 -> [-0.175, 1.745] rad (the MJCF joint range).

**Unvalidated assumptions (must be checked before relying on results):**
1. The gripper 0..100 -> joint-range mapping and its direction (0 = closed = lower limit). The MJCF has no 0..100 joint
   and the existing sim scripts apply `deg2rad` to every joint including the gripper.
2. The MJCF zero pose corresponds to the physical SO-101 calibration used in LeRobot. The first real recorded frame has
   shoulder_lift about -103 deg, outside the model's +/-100 deg, which suggests an offset/limit difference.
3. Actuator gains/`forcerange` of the MJCF match the Feetech controller response. Not measured.

## Rollout semantics

* Reset every rollout: `mj_resetData`, nominal friction restored, warning counters cleared, state re-initialised from the record.
* Initial state: record joint positions + sampled measurement error (arm only); velocities = 0 when not recorded (flagged).
* Control period: substeps follow cumulative rounding of `1/(control_hz * timestep)` (16.67 substeps at 30 Hz / 2 ms).
* Predicates (every `predicate_check_every_substeps`, default 5 = 10 ms): `robot_floor_contact`, `robot_bin_collision`,
  `ee_outside_workspace`, optional `unintended_displacement_<obj>` (disabled by default). Transient contacts between
  checks can be missed; this is a documented sampling limitation.
* Task predicate (chunk horizon): fail if `||ee_final - target|| > tolerance`. Horizon = chunk + `settle_s`; this is NOT a full-task success prediction.
* Outcomes per rollout: `ok`, `simulator_error`, `invalid_init`, `timeout`, `unknown`. Only `ok` rollouts are valid.

## Perturbations (all placeholders until measured)

| Variable | Distribution | Units | Applies to | Calibrated |
|---|---|---|---|---|
| robot_state_error_deg | truncated normal sigma 0.5, bound 1.5 | deg | 5 arm joints at init | no |
| object_xy_m | truncated normal sigma 0.003, bound 0.01 | m | object x,y | no |
| object_yaw_deg | truncated normal sigma 3, bound 10 | deg | object yaw | no |
| friction_scale | uniform 0.8-1.2 | x | cube sliding friction | no |
| control_noise_deg | truncated normal sigma 0.2, bound 0.6 | deg | per-step arm target noise | no |

Sources are recorded in the config and copied into every trace. **Not implemented by design:** camera-pixel noise (needs a
perception pipeline), action perturbation (separate candidates, separate risk estimates), actuator response (not in this slice).

Seeding: `SeedSequence(base_seed).spawn(max_rollouts)[i]`, so rollout `i` is identical for every budget N > i (paired
comparisons) and independent of execution order. Fixed seeds do not guarantee bitwise-identical results across hardware.

## Estimator and gate

* `simulated_failure_fraction = failed / determined`; failed = task failure OR safety violation. Task and safety availability, counts and
  fractions are reported separately (`criteria`, `denominators`); see the docstring of `integration/offline_pipeline.py`.
* A completed simulation is not a valid outcome observation. **Safety evaluable** = rollout ran validly. **Task evaluable** = `task_success`
  or `task_failure`; `unknown` (segment horizon shorter than the assumed task horizon, invalid, timeout) is NOT evaluable and is never
  counted as success or zero risk. **Determined** (required criteria known): a known safety violation, or clean safety AND a known task
  outcome (safety-only scope: any valid rollout).
* Coverage: `determined >= ceil(min_valid_fraction * N)` (UNCALIBRATED 0.8) else the fraction is withheld (`null`; the subset value appears only as
  `partial_fraction_over_determined`) and the status is `INSUFFICIENT_EVIDENCE` (`INSUFFICIENT_VALID_ROLLOUTS` when too few ran validly)
  with reason codes such as `task_outcome_unavailable` / `safety_outcome_unavailable`. The decision is REJECT (offline label WOULD_REJECT).
* `required_criteria` defaults to `("safety","task")`. `("safety",)` is an explicit safety-only scope whose WOULD_EXECUTE is recorded as
  `approval_scope: safety_only`, not a full-action approval. Safety can never be dropped. Demo mode does not bypass any of this.
* Simulator errors, invalid initialisations and timeouts are never failures and never evidence.
* Zero determined rollouts -> fraction `null`, never 0. TIMEOUT, SIMULATION_ERROR, INSUFFICIENT_VALID_ROLLOUTS, INSUFFICIENT_EVIDENCE also give `null`.
* Interval: Wilson score (default 95%). Covers Monte-Carlo sampling error only, not simulator model error.
* Gate (fail closed): EXECUTE only if action valid, state valid and fresh, status COMPLETED, limits ok, threshold configured
  and `fraction < threshold` (equality rejects). `risk_threshold: null` always rejects. The demo threshold (0.3) is **uncalibrated**
  and is labelled as such in every decision. Thresholds must be chosen on calibration data, never on held-out test outcomes.
