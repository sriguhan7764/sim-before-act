# ACT training, freezing and offline inference (Milestone D)

Code: `collector/act/{adapter,train,infer,segment,smoke,__main__}.py`; configs `configs/act_{smoke,pilot,convert}.yaml`.
Status: **IMPL + SYN** (smoke only). **REC no, PHYS no.** **No policy has been trained for the task, and none exists.**
ACT is a single-task imitation policy (RGB + joint state), **not** a language-conditioned VLA. Everything below runs in the
`guhan_c_lerobot` env (torch + LeRobot 0.6.1, CPU), run from `robot_collector/`:

```bash
export PYTHONPATH=/home/apic3/lerobot_v6_2/lerobot/src:$PWD        # LeRobot source is read-only
PY=/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python
```

All torch/lerobot imports are lazy, so the numpy-only `guhan_c` env can import the package, run the converter, and
validate action segments (`python -m collector.act validate-segment seg.json`).

## Commands

```bash
# 0. bounded SYNTHETIC smoke of the whole chain (fixture -> convert -> train 6 steps -> stop/resume -> freeze -> infer -> eval)
$PY -m collector.act smoke --out .work_act/smoke            # out dir must not exist / be empty; ~6 s on CPU

# later, REAL data (not run: no qualifying data, see ACT_CONVERSION.md). kind=main is refused on synthetic or provisional data
$PY -m collector.act train    --config configs/act_pilot.yaml --converted data/act/<name> --run runs/act_pilot_001
$PY -m collector.act resume   --config configs/act_pilot.yaml --converted data/act/<name> --run runs/act_pilot_001
$PY -m collector.act freeze   --run runs/act_pilot_001 --converted data/act/<name> --out frozen/act_pilot_001 --which best
$PY -m collector.act evaluate --frozen frozen/act_pilot_001 --converted data/act/<name> --split val --out frozen_eval.json
$PY -m collector.act infer    --frozen frozen/act_pilot_001 --converted data/act/<name> --split val --row 0 --out segment.json
```

`infer` reads a state (+ images) from a converted row; programmatic use: `FrozenPolicy(frozen_dir).predict(state, images)` and
`write_segment(...)` (`collector/act/infer.py`) for an arbitrary 6-float state in contract units + uint8 RGB images at the
training size.

## Training

* Trains on the **train split only**; the val split is used for evaluation and checkpoint selection. Normalisation stats come
  from the conversion (train-only, hashed).
* Loss: ACT's own (masked L1 on `action` with `action_is_pad`, plus `kl_weight` x KLD in train mode). Defaults verified in the
  installed code: `chunk_size=100, n_action_steps=100, n_obs_steps=1, resnet18, use_vae=True, kl_weight=10`, MEAN_STD
  normalisation.
* Optimiser AdamW (lr / backbone lr / weight decay / grad-clip from the config). `push_to_hub=False` always.
* Checkpoint selection: every `eval_every` steps the val L1 (normalised space) is computed; `best/` holds the lowest, `last/`
  the latest. Both carry `identity.json`, `train_state.json` (history, best step/loss) and `model.safetensors`.
* Determinism: `torch.manual_seed`, numpy seed, and data order = pure function of `(seed, step)` (`adapter.index_stream`), so
  **resume is exact**: in the smoke run the stop-at-3-then-resume run ended with weights sha256-identical to the straight run
  (observed on this CPU; not claimed for other hardware/thread counts). Resume refuses if the config (other than `steps`) or
  converted dataset differs.
* Guards (`TrainingRefused`): `device` other than `cpu`; unknown model keys; non-empty run dir; `pretrained_backbone_weights`
  not in the local torch hub cache (nothing is downloaded); `kind=main` on synthetic data, a non-frozen split, a conversion made
  with `--allow-provisional-split`, or `suitable_for_main_training=false`.

## Freeze

`freeze` copies `model.safetensors`, `config.json`, `normalization_stats.json`, `train_config.json`, `identity.json`,
`preprocessor.json` (state/image keys, shapes, scaling, "unnormalise, NO clipping") and `conversion_ref.json` into a new
read-only directory and writes `FREEZE.json`: checkpoint / config / preprocessing / normalisation / train-config sha256,
dataset (conversion, manifest content, split) hashes, split status and the provisional flag, seed, python / torch /
torchvision / lerobot / numpy versions, `git_commit` (null here: neither tree is a git repo), device, `cuda_available`,
`xpu_available`, platform, and a self hash. `verify_frozen(dir)` recomputes everything; `FrozenPolicy` refuses to load a
frozen dir that fails verification. A frozen dir is never overwritten.

## Offline inference and the action segment

`FrozenPolicy.predict(state, images)` -> `(chunk_size, 6)` **float64 action chunk, MEAN_STD-unnormalised and unmodified (no
clipping, no offsets, no limits widened)**. `write_segment` writes an **`action-segment-v0.1`** JSON
(`collector/act/segment.py`, pure python/numpy, shared with the simulator-side loader):

* `contract`: joint names, units `[degrees x5, percent_0_100]`, `absolute_joint_position_targets`, `control_hz`,
  `zero_order_hold`, reference = follower joint-position target space (NOT torques, NOT confirmed physical execution).
* `actions` (T,6) float64; `action_sha256` = sha256 of the C-contiguous `<f8` bytes.
* `contract_id` = sha256 of the canonical JSON (sorted keys, compact separators) of the `contract` block incl. `control_hz`;
  `segment_id` = first 16 hex chars of `sha256(action_sha256 + contract_id)` (ASCII concatenation).
* `policy` (kind, `trained_for_task`, checkpoint id/sha256, preprocessing/config sha256, framework `lerobot 0.6.1 ACT`, device,
  seed), `input` (state values + sha256 + `state_ref`, per-camera image sha256/shape), `chunk` (`execution_horizon_steps` <= T,
  `temporal_ensembling: false`, `replanning: false`), `data_origin`, `provenance_label`, `created_wall_ns`.

`validate_action_segment_dict(d)` returns a list of problems (empty = valid) and also rejects smoke policies that claim
`trained_for_task: true`. `load_action_segment(path)` reads + validates.

`evaluate` reports **OFFLINE prediction error** vs the recorded commands (val L1 in normalised space and per-joint MAE in
degrees / gripper percent over non-padded positions). It is **not task success** and not physical or simulated execution.
Note the per-joint units: the mixed mean over joints is only a convenience.

## What the smoke test does and does not prove

Observed run (this machine, synthetic fixture, `configs/act_smoke.yaml`): 4 fabricated episodes (3 train / 1 val,
provisional split, `--allow-provisional-split` recorded), 48x64 images, ACT `chunk_size 8`, `dim_model 32`, 1 encoder layer, 6
steps, batch 4, `pretrained_backbone_weights=None`, device **cpu** (`torch 2.15.0+cpu`, `torch.cuda.is_available()` False, XPU
not available), ~6 s. Best val L1 (normalised) 1.28 at step 6; offline MAE 9.3 (mixed units; meaningless).
Every artefact carries `kind: smoke_random_init_few_steps`, `trained_for_task: false`, `data_origin: synthetic_fixture` and the
label **"SYNTHETIC SMOKE TEST - software execution only, not a thesis finding"**. Deviations from ACT defaults (tiny model,
short chunk, larger lr, no pretrained weights, tiny images) are listed under `deviations_from_act_defaults` in `identity.json`.

It proves only that the software chain executes, that stop/resume reproduces the final weights, that freeze/verify detect
tampering, and that a valid segment is produced. It proves **nothing** about task performance, about GPU/XPU, about real data,
or about simulator or hardware agreement. The policy output of a 6-step random-init network is noise.

## Outstanding issues

* ResNet18 ImageNet weights: the pilot config needs them; this code only uses a file already in the torch hub cache and never
  downloads. A `resnet18-f37072fd.pth` was observed in `~/.cache/torch/hub/checkpoints` (provenance not verified). The smoke
  test does not use it.
* **No GPU/XPU**: only CPU is validated; `device != cpu` is refused. No Intel XPU claim is made.
* No real qualifying data; the calibration-mismatch BLOCKER is unresolved; a pilot policy would be trained on commands whose
  follower response is unvalidated.
* Custom converted layout, not LeRobotDataset v3 (see ACT_CONVERSION.md).
* `pytest` is absent from the lerobot env: torch checks are a plain script, `tests/torch_checks_act.py`
  (`$PY tests/torch_checks_act.py`), also run by `tests/test_act_torch_smoke.py` as a subprocess (skipped with a reason when the
  env is missing).
* `pyproject.toml` package list was not edited (out of scope); run from the `robot_collector` root or via `PYTHONPATH`.
