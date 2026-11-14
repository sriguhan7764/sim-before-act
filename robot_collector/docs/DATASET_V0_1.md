# Dataset v0.1 (schema `dataset-v0.1`)

Code: `collector/dataset.py`. CLI: `python -m collector dataset <manifest|validate|inventory|quality|split|verify-split|replay|export>`.
Everything is **offline and read-only over masters** (tests assert masters are byte-identical afterwards, and that importing the
module loads no device/robot code).

## Roles (one per entry)

| role | derived from | counts as a training demonstration? |
|---|---|---|
| `setup_pilot` | `pilot.role=setup` | no |
| `successful_demonstration` | `pilot.role=demonstration` + outcome `success` | **only if no exclusions** (below) |
| `unsuccessful_attempt` | demonstration + outcome `failure` / `aborted` | no |
| `evaluation_trial` | `pilot.role=evaluation` | no (independent evaluation, never train/val) |
| `synthetic_fixture` | `quality_flags.mock_data` or `quality_flags.synthetic_fixture` | no (data_origin `synthetic`) |
| `simulated_rollout` | produced by the MuJoCo verifier | no (data_origin `simulated`; no physical label) |
| `unclassified` | no pilot metadata, or demonstration with no outcome label | no - needs an operator label |

`data_origin` is `recorded_physical`, `synthetic` or `simulated`. The latest `<episode>.annotations/vNNN.json` (curator) overrides the
recorded outcome; the master is never edited.

## Entry fields

`entry_id/episode_id/session_id`, `episode_path`, `episode_json_sha256`, `master_manifest_sha256`, `role` + `role_basis`, `data_origin`,
`split`, `task` (name, instruction, fixed target red cube -> blue tray), `cube_position_id`, `cube_start_pose_mat_frame`,
`tray_pose_mat_frame` (**null until measured; never defaulted**) + `poses_measured`, `outcome` {value, label_source, horizon
(`full_task` or null), annotation_version, failure_category, accepted_for_dataset}, `cameras` (profile, device, depth scale,
camera.json hash, frame count), `calibration` (camera-to-mat, robot motor calibration sha256, controller session sha256, per-file
hashes, `registry_status`, `physically_validated: false`), `semantics`, `quality` (validation, flags, counts, missing modalities),
`exclusions`, `training_eligible`, `lineage` (exports with manifest hashes; master is source of truth).

### Action/state semantics (what is recorded)
* State = follower **measured** raw ticks (+ calibrated radians when a calibration is attached; gripper 0-100, never radians).
* Action candidates, kept separately: `commands.values` = controller's clipped follower target (LeRobot normalised units),
  `commands.raw_goal` = raw Goal_Position ticks **submitted** (`submitted_not_confirmed`; no register readback).
* Leader is a separate stream; never substituted for state or commands.

## Training eligibility (policy recorded in the manifest)

A `successful_demonstration` is eligible only if: master validates; status `complete`; not mock; cube position id present;
**cube and tray poses measured** (`require_measured_poses`; relax with `--allow-unmeasured-poses` - values stay null);
no missing modality; no dropped samples; **curator acceptance** (`annotate --accepted-for-dataset yes` - never inferred); and
not recorded under a stale calibration (`--calibration-registry`).

## Splits

`dataset split --seed S` assigns **whole episodes** by hash order of `seed:id`. Plan 25 train / 5 val is a **target**.
Below 30 eligible episodes the file is `PROVISIONAL` (proportional counts, explicit shortfall, `frozen: false`) and may be re-derived;
only at 30 does it become `FROZEN`, after which `write_split` refuses different assignments. Surplus episodes are listed in
`surplus_unassigned`, not silently added. Evaluation trials, synthetic fixtures and simulated rollouts are never in train/val.
`verify-split` checks hashes, eligibility, leakage and manifest hash.

## Status
Implemented; tested with synthetic fixtures; manifest/inventory/stale-calibration tagging run read-only on the 5 earlier real recordings
(all `unclassified`: no pilot metadata, no labels; 3 flagged stale). No real demonstrations exist yet, so no real split exists.
