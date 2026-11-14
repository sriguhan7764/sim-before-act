# Data schema (schema_version 1.0.0)

## What is captured / not captured / derivable

| Captured (master) | NOT captured | Derivable later |
|---|---|---|
| Colour PNG (lossless), native depth uint16 PNG, colour-aligned depth uint16 PNG, wrist RGB | Camera pose in the **robot-base** frame (robot-base-to-mat is never inferred) | Camera-to-mat pose is stored (calibration snapshot); 3D points in mat frame via depth + intrinsics |
| Per-frame device frame number, device timestamp + domain, host monotonic receipt, host wall time | True exposure time / capture latency (not exposed or measured) | Association tables with other tolerances (re-run from raw streams) |
| Follower raw motor positions as returned (encoder ticks), velocity/load/current/voltage/temperature/status/moving (raw register units) | Torque/PID configuration, motor homing offsets read back from motors | Calibrated joint angles (rad) and gripper 0-100, if a calibration file was configured |
| Commands (only if the teleop process taps them), leader state (own read-only port) | Command acknowledgements (not available over the Feetech bus tap) | - |
| Camera intrinsics/distortion/extrinsics/depth scale/settings, robot calibration copy + sha256, config snapshot | Object poses unless operator-entered; success labels unless operator-entered | - |
| Task, instruction, scene text, operator, outcome, failure category, notes, versions, quality flags | | |

This dataset is not claimed to suit every experiment. It lacks, e.g., force/torque, tactile data, end-effector pose
(needs a kinematic model + robot-base-to-mat calibration), and sub-frame synchronisation.

## Master layout

```
<id>/
  episode.json            metadata, status, outcome, counts, quality_flags, calibration refs, config snapshot
  manifest.json           sha256 + size for every file; files are chmod 0444 after finalisation
  events.jsonl            queue overflows, write errors, low-disk events
  quality_report.json     per-stream frame/gap accounting + sync summary
  streams/<cam>/camera.json     device identity, profile, intrinsics, distortion model, extrinsics, depth_scale, settings
  streams/<cam>/frames.jsonl    one line per saved frame (written after its files exist)
  streams/<cam>/color/%06d.png          RGB (PNG viewed as normal colour image)
  streams/<cam>/depth_native/%06d.png   uint16, depth-sensor geometry
  streams/<cam>/depth_aligned/%06d.png  uint16, colour-image geometry
  telemetry/follower_state.jsonl  measured state (raw)
  telemetry/commands.jsonl        commands sent to the follower (separate stream)
  telemetry/leader_state.jsonl    leader state (separate stream)
  telemetry/*_source.json         adapter descriptions (units/meaning of raw fields)
  calibration/camera_to_mat.json | robot_motor_calibration.json | joints.json
  sync/associations.jsonl, sync/sync_report.json     DERIVED tables (raw streams stay the source of truth)
```

### frames.jsonl fields
`recv_index` (per-episode receipt counter; gaps = drops), `frame_number` (device), `device_timestamp_ms`,
`timestamp_domain`, `host_mono_ns`, `host_wall_ns`, `*_file`, depth frame number/timestamp/domain.

### Depth conventions
uint16 PNG, `metres = value * depth_scale` (`camera.json`). **0 = no measurement** (RealSense convention) - invalid
pixels are never turned into a valid distance. Depth is never stored as a colour image; replay colourises a copy only.

### Telemetry record
`seq`, `t_start_ns`/`t_end_ns` (host monotonic around the bus read), `wall_ns`, `read_ok`, `errors`, `raw` (ordered by
`joints.json`), `velocity_raw`, `load_raw`, `current_raw`, `voltage_raw`, `temperature_c`, `status`, `moving`.
`raw` = Feetech Present_Position register, encoder ticks 0-4095 **after the motor's stored homing offset**. It is not
normalised and not radians. Failed reads are kept with `read_ok: false` and never treated as valid.

### Joint angle semantics
`joint_angles_rad = (raw - (range_min+range_max)/2) * 2*pi/4095` using the calibration JSON, i.e. angle relative to the
middle of the calibrated range (LeRobot `DEGREES` convention, lerobot 0.6.1). Valid only if that file matches the
homing offsets stored in the motors. **Gripper** is `normalized_0_100`, never radians. No calibration file => the
`rgb_joint_angles` export is blocked (or `--allow-partial` produces a manifest with `complete: false` and the missing list).

### Time and synchronisation
All association uses host `CLOCK_MONOTONIC` ns (comparable across processes on one Linux host). Camera timestamp =
receipt time (includes unknown capture latency); robot state = midpoint of the bus read; commands = stamp before send.
Device timestamps are recorded with their domain but are **not** compared with host time; `sync_report.json` lists an
informational median(host - device) offset only (`used_for_association: false`). `associations.jsonl` row per
reference-camera frame: `follower_state`, `command_prev`, `command_next`, `leader_state`, other `cameras`, each with
`index`, signed `residual_ns` (sample - frame) and `status` (`matched` | `unmatched_beyond_tolerance` | `no_valid_candidate` |
`no_samples` | `source_absent`). Unmatched samples are never replaced by stale ones. Interpolation is not implemented
(`sync.interpolation` only accepts `none`); original samples are always the only stored values.

### Status values
`complete`, `complete_with_drops`, `aborted`, `aborted_writer_overflow`, `aborted_low_disk`, `failed_write_error`,
`interrupted_recovered`. Outcome (operator label): `success|failure|aborted|unlabelled` - never auto-assigned.

## Export profiles (`<id>.exports/<profile>__v1/`)

Common files: `manifest.json` (profile, parent episode id + master manifest sha256, modalities, file hashes,
`content_sha256`, `complete`, `missing_required_signals`), `schema.json` (units), `task.json`, `cameras.json`,
`calibration_refs.json`, `samples.jsonl` (one row per reference frame), `arrays/*.npy` (`t_host_mono_ns`, `state_valid`,
`state_residual_ns`, ...), image directories, `provenance.json` (generation time; excluded from the hash).

| Profile | Payload | Excluded |
|---|---|---|
| `rgb_joint_angles` | colour images, `joint_angles_rad` (revolute order in schema), gripper `{value, unit}`, timestamps + residuals, task/outcome, calibration refs | depth, raw motor positions |
| `rgbd_motor_positions` | colour, `depth_native` and `depth_aligned` (separately named), depth scale, intrinsics/extrinsics, `motor_positions_raw` in SDK units + names, residuals, optional `joint_angles_rad_optional` | - |
| `full_master` | byte copy (or `--link` hard-links) of every master file + `master_manifest.json` | - |

Unmatched states appear as `state.association_status != matched`, `state_valid=false`, NaN angles / -1 raw.

## Pilot additions (schema additive; `meta["pilot"]`, `meta["controller"]`)

- `meta.pilot`: `role` (setup|demonstration|evaluation), `cube_position_id`, `cube_start_pose_mat_frame`,
  `tray_pose_mat_frame` (+ provenance/uncertainty from `configs/pilot_layout.yaml`; **null stays null, never defaulted**),
  `robot_start_configuration` (only if a state <1 s old existed at start), `attempt_index`, `outcome_label_source`
  (`operator_visual|operator_measured|none`). Quality flags: `pilot_metadata_missing`, `cube_pose_missing_fields`,
  `tray_pose_missing_fields`, `robot_start_configuration_missing`, `controller_session_missing`,
  `follower_calibration_hash_matches_controller`.
- `meta.controller` / `telemetry/*_source.json`: bridge `session.json` (LeRobot version, control loop rate, `max_relative_target`,
  calibration paths + sha256, normalisation modes, signal semantics) and its sha256.
- LeRobot-bridge command signals (separate and never interchangeable): `commands.values` = follower target actually sent
  (after the relative-target clip, LeRobot-normalised units: degrees / gripper 0-100); `raw_goal` = Goal_Position ticks computed with
  the bus' `_unnormalize` (**not read back**; no acknowledgement; `ack: null`); `leader_measurement_normalized`; `clipped`.
  Leader raw ticks use the LEADER calibration (`calibration/leader_motor_calibration.json`).
- Lightweight exports additionally carry, per sample, `commands.{prev,next}` (target, raw_goal, validity, residual) and `leader`,
  plus `master_frame_index`, `master_recv_index`, `master_state_index` (source-frame associations). Arrays:
  `command_{prev,next}_{target,raw_goal,valid,residual_ns}`; `target_rad` only in `rgb_joint_angles`.
  Command signals are kept in both exports. Exports are views of one episode, not extra episodes.
- Reports: `<id>.reports/quality_report.{json,md}` and `acceptance.{json,md}`. Annotations: `<id>.annotations/vNNN.json`.
- The joint-angle zero is the calibration-range midpoint, **not** a validated physical zero; signs and gripper mapping are unvalidated.

## Pilot-preparation additions (640x480@30 setup pilot)
- `streams/<cam>/camera.json`: `negotiation` = `{requested, actual, mismatches, matches_request, silent_substitution:false}` (colour and
  depth requested vs the profile the device reported after the stream started; a mismatch refuses the episode), `depth_native`
  (width, height, format z16, geometry "depth-sensor image", intrinsics, extrinsics_to_color, scale, units, invalid_value 0) and
  `depth_aligned` (colour geometry and intrinsics; "derived by rs.align, resampled"). Both are also copied into `cameras.json` of the
  `rgbd_motor_positions` export (`negotiation` in both exports). `depth_intrinsics`/`extrinsics_depth_to_color` are retained.
  Mock cameras mark their negotiation `synthetic: true`. The OpenCV wrist camera has no device timestamp/frame counter.
- `config/effective_config.json`: the exact config used + `config_sha256` (sha256 of canonical JSON: `sort_keys`, compact separators,
  `default=str`), the config file path/hash, the calibration files in use (path + sha256) and the registry; `episode.json` carries
  `effective_config_sha256` and `stale_calibration_in_use`. `calibration_refs.json` of both exports carries the same config hash.
- `episode.json: protocol`: `{name, nominal segments, cues_issued[{label, cue_host_mono_ns}]}` for `record --protocol`; prompts only.
- Commands from the bridge carry `submission_status: submitted_not_confirmed`.
- `acceptance.json`: `verdicts.{collector_integrity, physical_calibration, manipulation_readiness}`, each check has a `category` and
  status PASS | FAIL | WARN | NOT_CHECKED; `overall` equals the collector-integrity verdict only.
