# ACT dataset conversion (Milestone C)

Code: `collector/act/{contract,align,load,convert,synthetic}.py`. CLI: `python -m collector.act ...` (run from `robot_collector/`).
Status: **IMPL + SYN**. **REC no, PHYS no**: no recorded episode currently qualifies (see "Real data" below).
ACT is a single-task imitation policy on RGB + joint state. It is **not** a language-conditioned VLA. CAD geometry is not a
calibrated twin. The follower/leader calibration mismatch documented in the repo is an open **BLOCKER**; the converter records
it in every audit (it warns whenever `follower_calibration_matches_controller` is not `true`) and does not "fix" it.

## What is built (and what is not)

Input: a dataset v0.1 `manifest.json` + a split file (`collector.dataset`). Output directory:

| file | content |
|---|---|
| `steps.npz` | one row per kept control step (arrays below); hashed by array content (`arrays_sha256`) |
| `conversion.json` | config, counts, signals, units, source lineage, labels, hashes |
| `normalization_stats.json` | mean/std fitted on TRAIN rows only, with its own `sha256` |
| `image_index.json` | sha256 of every written PNG + shape |
| `images/<cam>/<episode_id>/<frame_index>.png` | RGB images at the stored size (uint8) |
| `conversion_audit.json` / `.md` | per-episode kept/dropped by reason, residuals, depth/wrist handling, lineage |

**This is the converter's own documented layout, NOT a LeRobotDataset v3 on disk.** Building a v3 dataset (video encoding,
parquet + metadata) offline was judged impractical/unnecessary. A thin adapter (`collector/act/adapter.py`,
`ActStepsDataset`) feeds `ACTPolicy.forward(batch)` directly with the keys ACT expects: `observation.state`,
`observation.images.<cam>`, `action`, `action_is_pad`. A later LeRobotDataset export can be written from `steps.npz` without
changing the semantics.

Per-row arrays in `steps.npz`: `episode_idx, split_code (0 train, 1 val), grid_index, t_ref_ns, state (N,6), action (N,6),
raw_goal (N,6 int ticks), leader_measurement (N,6), leader_valid, command_seq, state_seq, front_frame_index,
front_recv_index, front_frame_number, wrist_frame_index, wrist_recv_index, wrist_valid, state_t_ns, command_t_ns,
state_age_ns, command_lag_ns, wrist_age_ns`.

## 1. Observations

* `observation.state`: measured follower state, 6 floats.
* `observation.images.front`: front D435 **RGB** (the manifest/episode `reference_camera`, `front_rgbd`).
* `observation.images.wrist`: optional wrist RGB (`wrist_stream: wrist`; absent allowed, see section 8).
* **Depth is NOT fed to the baseline ACT.** Depth files are never opened by the converter; the audit records
  `depth.fed_to_model: false` and the depth freshness counts from `frames.jsonl`.
* Resolution: `image_size: [H, W]` resizes (cv2 `INTER_AREA`) every camera to one size (ACT needs identical shapes);
  `null` keeps native size and refuses if cameras differ. Images are stored uint8 RGB, scaled to [0,1] by the adapter, then
  per-channel MEAN_STD normalised using TRAIN-only image stats.

## 2. State units and joint order

`JOINT_NAMES = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`.
State = `follower_state.jsonl -> normalized` of the **measured** follower: LeRobot normalised units, **degrees** for the 5
joints and **0..100** for the gripper, the same space as the action. Only `read_ok` rows are eligible. Measured state is never
replaced by leader state or commanded values. The state time is the midpoint of the bus read (`(t_start_ns + t_end_ns) / 2`).
Records without a `normalized` block (raw ticks only) are refused, not converted by a guess.

## 3. Action signal

Default `action_signal: commands.values`: the **clipped follower target the controller asked the follower to reach**
(`commands.jsonl -> values`, LeRobot normalised, degrees for 5 joints, 0..100 gripper). This follows the bridge contract:
`raw_goal` (Goal_Position ticks submitted) and `leader_measurement` are different quantities and are kept only as separate
retained columns (`raw_goal`, `leader_measurement`); `submission_status` is `submitted_not_confirmed`, so a command is **not**
evidence the follower reached it. The converter applies no clipping or offset of its own; the only clipping is the
controller's, already contained in `values`. Commands with `units` other than the bridge's normalised format (e.g. older
tick-format recordings) are refused.

Substitution (training on `leader_measurement_normalized`) needs `allow_signal_substitution: true`; it is then recorded in
`conversion.json -> signals.action.substitution = true` and printed as SUBSTITUTED in the audit. Measured follower state is
never accepted as the action.

## 4. Temporal alignment (exact rule)

Control reference = each **front camera frame** (`t_ref = host_mono_ns` receipt time), `grid_index = recv_index`. All times are
host `CLOCK_MONOTONIC` ns. For each front frame, in this check order:

1. **state**: the latest valid (`read_ok`) follower state with `state_t <= t_ref` and `t_ref - state_t <= max_state_age_ms` (40);
   else drop `state_too_old` / `no_state_at_or_before_ref`.
2. **action**: the first command with `t_start_ns >= t_ref` (stamp before send) and `lag <= max_command_lag_ms` (50);
   else drop `command_lag_too_large` / `no_command_at_or_after_ref`.
3. front image readable (else `front_image_unreadable`).
4. wrist frame per section 8.

Dropped steps are **dropped and counted, never filled**; nothing is interpolated, and nothing bridges an episode boundary or an
invalid sample. The state used is always causal (`state_t <= t_ref`) and the command is the one stamped at/after the
observation (`command_t >= t_ref`); both are asserted by tests.

## 5. Rate, chunking, padding

The effective control rate is the front-camera rate (30 Hz nominal). `conversion.json -> control_hz` is derived as
`1000 / median front-camera frame period (consecutive recv_index)` over the episodes and is recorded with its basis. Chunk length `chunk_size` and `n_action_steps` default to ACT's 100/100 (smoke uses 8).
Action chunk for a row at grid g: position j is the kept action of grid g+j **of the same episode**. If that step does not
exist (episode end, or it was dropped) the position is **padded with the last valid action and `action_is_pad = True`**; this
matches ACT's own convention (`F.l1_loss` is masked by `~action_is_pad`). Episodes are never crossed and no step is invented.
`n_obs_steps = 1`.

## 6. Gripper

0..100 LeRobot-normalised gripper, in both state and action, never radians. The simulator mapping is unvalidated.

## 7. Normalisation

`normalization_stats.json`: mean/std (ddof 0) of `observation.state` and `action` over **train rows only** plus per-channel
image mean/std (on [0,1]) over train images only; `sha256` over the canonical content. Val rows are normalised with the same
numbers (tests: changing val data leaves the stats and hash unchanged). A zero std is an error, not clamped. ACT uses
MEAN_STD processors (`make_act_pre_post_processors`) built from these stats. The hash is recorded in the audit, the training
config and the frozen checkpoint.

## 8. Wrist camera

Optional, ~26.7 Hz measured (lower than the front camera). Matching: latest wrist frame with `recv <= t_ref` and
`age <= wrist_max_age_ms` (50). A frame can serve at most `wrist_max_reuse` (2) consecutive steps. A staler frame is **never**
silently copied. Missing behaviour (`wrist_missing_policy`):

* `drop_step` (default): the step is dropped (`wrist_no_frame_at_or_before_ref`, `wrist_frame_too_old`, `wrist_reuse_exceeded`).
* `mark_missing_with_mask`: the row is kept with `wrist_valid = False`, `wrist_frame_index = -1`, `wrist_age_ns = -1`. The
  baseline trainer **does not feed masked rows**: `ActStepsDataset(require_wrist=True)` (default) excludes them, and with
  `require_wrist=False` loading a masked row raises (no zero/stale image is substituted). The mask exists so the counts and the
  rows stay auditable. A stream with no accepted image at all is refused (nothing is invented).

The audit reports reuse counts (`max_steps_served_by_one_wrist_frame`) and masked rows per episode.

## 9. Audit and lineage

`conversion_audit.json/.md`: per episode `n_front_frames`, `n_kept`, `n_dropped`, `dropped_by_reason`, the dropped steps list,
association residuals (state age, command lag, wrist age: min/median/p95/max), wrist and depth handling, which signal is the
action, units, split, normalisation hash. Lineage: source episode ids, `episode_json_sha256` per episode, manifest content
hash, split hash (`split_sha256`), master-file hash verification (`verify_master_files`).

`verify_lineage(converted_dir, manifest)` (also `python -m collector.act verify-lineage`) re-reads the **source** episodes and
checks every converted row: episode id and manifest/episode hashes, command seq and state seq exist in the source,
`values`/`raw_goal`/leader measurement/state equal the source, the front (and wrist) frame records and times equal the source,
the state/command/wrist association rules still hold when re-derived from the source, no episode is in both splits, and the
stored PNGs equal the freshly converted source frames (a modified image or source is reported). Returns `{ok, errors, n_rows,
n_images_checked}`.

## 10. Refusals

* provisional / not-frozen split: `ProvisionalSplitError`, unless `--allow-provisional-split` (smoke only). When used, the flag
  is stored in `conversion.json (source.split.allow_provisional_split_flag)`, the audit and the frozen checkpoint, and
  `suitable_for_main_training` becomes false (main training is then refused).
* a manifest entry that is not `training_eligible`: `NotTrainingEligibleError`.
* split tampering (hash mismatch), source file hash mismatch, non-bridge command units, missing measured state, mixed synthetic
  and non-synthetic episodes in one split.

## Synthetic fixture

`python -m collector.act fixture --out DIR` (`collector/act/synthetic.py`) writes 4 **SYNTHETIC FIXTURE** episodes
(`synthetic_fixture_ep_000..`, `quality_flags.synthetic_fixture = true`), a manifest and a **synthetic-fixture split**
(`make_split(..., synthetic_fixture=True)`: 3 train / 1 val, `data_origin: synthetic_fixture`, never frozen, can never verify as a
physical split). Manifest entries are role `synthetic_fixture`, origin `synthetic`, `training_eligible: false`; they carry
`synthetic_fixture_usable` (the same curator-acceptance / valid-master / no-dropped-samples gates a real demonstration needs), which
is the only thing that lets the converter accept them - and only inside a synthetic-fixture split. Poses, images and timing are
fabricated; they are **not** demonstrations and prove only that the software runs. The
converter relabels the output `data_origin: synthetic_fixture` and all audits carry "SYNTHETIC FIXTURE DATA". Fault knobs
(`state_invalid_frames`, `command_gap_frames`, `wrist_gap_frames`, `with_wrist`) exist for the drop/pad tests.

## Real data later (documented, NOT run: nothing qualifies today)

Requirements: episodes with bridge-format commands (`units` normalised, `submission_status`), `training_eligible: true`
manifest entries, and a FROZEN split.

```bash
cd /home/apic3/guhan_c/robot_collector
python -m collector.act convert --manifest <manifest.json> --split <frozen_split.json> \
    --out data/act/<name> --config configs/act_convert.yaml
python -m collector.act verify-lineage --converted data/act/<name> --manifest <manifest.json>
```

## Limitations

Custom layout instead of LeRobotDataset v3; the control grid assumes a steady front-camera rate (irregularities show up as
drops/residuals, not corrections); image resize is the only image transform; float32 inside the model; the package list in
`pyproject.toml` was not edited, so run from the `robot_collector` root (or put it on `PYTHONPATH`).
