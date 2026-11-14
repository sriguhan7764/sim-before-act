# Sim-Before-Act (offline milestone 1)

Experimental research code for the M.Tech project *Physics-Grounded Counterfactual Verification of
Vision-Language-Action Policies for Safe Robotic Manipulation* (Guhan, Amrita School of Computing).
**Not a certified robot-safety system.** It supplements, never replaces, physical safeguards and independent robot limits.

## What this milestone is

An offline vertical slice:

```
record (state + objects + executed action chunk [+ outcome label])
  -> validate (action contract, state, freshness, object poses)
  -> MuJoCo twin: N seeded rollouts of the SAME fixed action under sampled initial-state/dynamics uncertainty
  -> simulated failure fraction (+ Wilson interval)        verification status: COMPLETED / INVALID_INPUT / ...
  -> decision gate (fail closed)                           decision: EXECUTE / REJECT  (separate from status)
  -> trace JSON (written for every attempt, incl. rejections and verification failures)
```

No policy, no robot, no hardware, no camera. "EXECUTE" offline means *would execute*: no execution adapter exists.
Execution is only possible through `GuardedExecutor` (approval bound to action hash, state hash/time, verification
config hash, expiry and permitted state deviation); a modified action must be re-verified.

## Layout

```
configs/offline_demo.yaml        all tunables; uncalibrated placeholders are marked calibrated:false
src/sim_before_act/
  schemas/record.py              ReplayRecord (input schema, sba-record/0.1)
  robot/action_contract.py       units, limits, validation, action id, contract->sim mapping
  twin/mujoco_twin.py            MuJoCo adapter, reset, rollout, outcome predicates
  twin/perturbation.py           seeded, documented perturbation sampling
  verification/{estimator,verifier}.py
  runtime/{gate,offline}.py      decision gate, guarded executor, pipeline
  traces/writer.py               sba-trace/0.1 JSON, exclusive + read-only
  data/lerobot_import.py         read-only LeRobot v3.0 -> record
  cli.py                         verify | import-lerobot
experiments/synthetic_demo/      3 SYNTHETIC records + generator
tests/{unit,integration,simulation}
docs/                            ACTION_CONTRACT.md, SCHEMAS.md, STATUS.md
```

The SO-101 MJCF is the user's existing `../so101_mujoco/scene.xml`, referenced read-only (hash stored as
`robot_asset_hash`). Nothing is copied or modified.

## Commands (conda env `guhan_c`)

```bash
cd /home/apic3/guhan_c/sim_before_act
PY=/home/apic3/miniforge3/envs/guhan_c/bin/python
export PYTHONPATH=src

# tests
$PY -m pytest -q                       # all; add -m "not simulation" to skip MuJoCo tests

# verify a synthetic record (N in {1,3,5,10,25,50}); traces go to --out
$PY -m sim_before_act verify --config configs/offline_demo.yaml --record experiments/synthetic_demo/safe_reach.json --n 10 --out traces_out
$PY -m sim_before_act verify --config configs/offline_demo.yaml --record experiments/synthetic_demo/floor_contact.json --n 10 --out traces_out

# import a window from the real LeRobot recording (read-only) and verify it
$PY -m sim_before_act import-lerobot --dataset ../lerobot_setup/datasets/g_c_20261113_174844 --start 0 --horizon 30 --out real_rec.json
$PY -m sim_before_act verify --config configs/offline_demo.yaml --record real_rec.json --out traces_out

# one-command OFFLINE demo of the whole chain (SYNTHETIC fixtures + random-init ACT smoke policy; opens no hardware; dir must be new)
$PY -m sim_before_act.demo --out /tmp/offline_demo_NEW
```

Sprint handoff (Milestones A-J, exact test results, measurements still required): `../robot_collector/docs/HANDOFF.md`.

## Results actually observed (this machine, mujoco 3.15.0)

* `pytest`: **89 passed** in ~2.8 s (unit 44, integration incl. fault/gate 33, simulation 12).
* Synthetic CLI, N=10: `safe_reach` -> fraction 0.000, Wilson 95% CI [0.000, 0.278], EXECUTE (demo threshold 0.3,
  uncalibrated); `floor_contact` -> 1.000 (10/10 `robot_floor_contact`), REJECT; `far_target` -> 1.000 (10/10 task failure), REJECT.
* Rollout cost: ~8 ms per rollout (30-step chunk + 0.5 s settle, 750 physics steps); ~0.08-0.1 s verification and
  ~0.2 s end-to-end per attempt for N=10 including trace writing. Single machine, no warm-up protocol: **not an E3 measurement**.
* Real LeRobot window (`g_c_20261113_174844`, frames 0-29): **INVALID_INPUT -> REJECT**, because (a) shoulder_lift is
  about -103 deg against the simulator's +/-100 deg limit (the MJCF zero/limits do not obviously match the user's
  calibration) and (b) there are no object poses, task target or outcome label in a LeRobot recording.

## Visualising rollouts (synthetic scenes)

`experiments/synthetic_demo/show_rollout.py` renders a rollout as video (`--video`) or a live viewer (`--live`,
needs `MUJOCO_GL=glfw` and a display). It only observes the twin through a read-only per-step callback.

## Audit and data readiness

* `docs/CALIBRATION_AUDIT.md`: read-only simulator-vs-calibration audit (why the recording shows shoulder_lift -103 deg vs the MJCF +/-100 deg).
* `docs/DATA_READINESS.md`: checklist for object poses, camera-to-robot calibration, task targets, outcome labels, and the next step.
* Traces record the gate's statistic and rule (`decision.decision_rule`): point estimate, strict `<`, Wilson bound not used.

## What these results do and do not show

The synthetic numbers show the pipeline, predicates, reset, seeding and gate behave as specified. The synthetic
`safe_reach` target is derived from the nominal simulator output (circular by construction). They are **not** evidence of
SO-101 digital-twin fidelity, of real-world predictive validity, or of any research hypothesis.

See `docs/STATUS.md` for limitations, blockers and the next stages.
