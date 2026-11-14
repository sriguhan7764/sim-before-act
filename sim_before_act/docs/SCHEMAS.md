# Schemas

## Input: `sba-record/0.1` (`schemas/record.py`)

One recorded pre-action state, the action chunk actually executed from it, and optionally the observed outcome **of that
same action**. Required: ids (record/episode/session/chunk), `synthetic`, `state` (joint_names, positions, units, timestamp_s,
clock_domain, velocities nullable), `action` (chunk, joint_names, units, representation, control_hz, issue_timestamp_s,
policy_id nullable, policy_confidence nullable), `task` (name, instruction, `target_xyz_m`/`tolerance_m` nullable).
Optional: `objects[]` (name, position_m, quat_wxyz, frame=`robot_base`, source, position_sigma_m nullable), `outcome`
(observed_success nullable, failure_type, label_source, horizon, `applies_to: recorded_action_only`), `calibration_ref`,
`image_refs`, `split`, `limitations`.

Missing fields stay null/empty and are reported by the verifier; nothing is fabricated. State age = `issue_timestamp_s - state.timestamp_s`
(both must be in the state's clock domain; the verifier cannot check that).

## Output: `sba-trace/0.1` (`traces/writer.py`), one JSON per attempt

`identity` | `observation_and_state` (state, objects, uncertainty availability, flags) | `action` (chunk, contract, nullable confidence,
task predicate) | `verification` (status, problems, counts: requested/attempted/valid/failed + task_failure, safety_violation,
simulator_error, invalid_init, timeout, unknown; fraction nullable; interval; seeds; perturbation spec; per-rollout results;
horizon; twin + asset hash; timings) | `decision` (EXECUTE/REJECT, reason codes, threshold + calibrated flag, approval binding,
hold/stop behaviour = "not defined offline") | `execution` (`executed:false`) | `physical_outcome` (recorded outcome of the recorded
action, nullable) | `reproducibility` (git commit or null, config hash, asset hash, python/os/package versions) | `limitations`.

Traces are written exclusively (never overwritten) and made read-only. Non-finite floats are stored as `null`.

## Relation to the planned SO101-TraceCF dataset

Implemented now: per-attempt JSON traces carrying the identity, observation/state, action, verification, outcome and
reproducibility blocks listed in the brief.

**Not implemented yet** (deliberately; no second storage backend without need): Parquet tables aggregating traces across
attempts, dataset split assignment, image/video asset references beyond the LeRobot paths, policy checkpoint reference,
estimated object state beyond operator-supplied poses. The planned approach is JSON per attempt (source of truth) plus a
derived Parquet table for evaluation (E1-E4), both versioned.

## LeRobot import (`data/lerobot_import.py`)

Reads `data/chunk-000/file-000.parquet` and `meta/info.json` (v3.0), takes the state at `--start` and the next `--horizon`
recorded actions. Writes only the output record file, never the dataset. Objects/task target/outcome/calibration ref come only from an
operator sidecar JSON (`{"objects": [...], "task": {...}, "outcome": {...}, "calibration_ref": {...}}`). Assumed, and flagged on the record:
degrees (arm) / 0-100 (gripper) units, because `info.json` does not state them; the dataset clock is episode-relative; issue time = state time.
Recorded actions are teleoperation commands that were executed, not policy outputs. Recorded depth is log-encoded 12-bit video,
so exact sensor values are not guaranteed.
