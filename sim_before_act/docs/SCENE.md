# Cube-to-tray scene (Milestone F)

Code: `src/sim_before_act/twin/scene/` (`config.py`, `builder.py`, `outcomes.py`, `runner.py`, `viz.py`, `render_worker.py`,
`fixtures.py`). Configs: `configs/scene_synthetic_demo.yaml`, `configs/scene_physical_template.yaml`.

> **SYNTHETIC DEMO SCENE (invented values, CAD geometry; not a calibrated twin).** This label is attached to every result, summary,
> plot and rendered image of the demo scene. CAD geometry (SO-101 MJCF from onshape-to-robot + simple primitives) is **not** a
> physically calibrated twin of the user's robot, mat, cube or tray.

## Frames and geometry

World frame = mat frame: origin at the midpoint of the mat edge nearest the robot, +X right (seen from the robot), +Y away across
the mat, +Z up. Mat 400 x 400 mm (thin box, top surface at z = 0), X in [-0.2, 0.2], Y in [0, 0.4]. The robot model is **composed**
in code with MuJoCo `MjSpec` from the read-only protected file `so101_mujoco/so101_new_calib.xml`; nothing is written back and no
XML is authored (the scene is generated in memory, so `assets/scene_demo/` only holds demo outputs). Added: table plane, mat, red cube
(free joint), blue tray (floor + four walls, static), cameras `overview`/`front`/`top`.
The MJCF robot looks along its own +X, so a base yaw of +90 deg faces +Y. Which physical point of the real robot the MJCF `base` origin
corresponds to is **not calibrated**: the robot-base-to-mat transform is an *explicit invented value* in synthetic mode (base (0, 0.03, 0),
yaw 90 deg) and a *required measured input* in physical mode.

Synthetic demo values (all INVENTED): cube 40 mm at (-0.06, 0.22) m, tray centre (0.10, 0.23) m, interior 0.12 x 0.10 m, rim 45 mm,
initial arm configuration [0, -50, 40, 50, 0, 40] (deg / gripper %).

## Modes

| Mode | Behaviour |
|---|---|
| `synthetic_demo` (default) | invented values; every source must say `INVENTED`; `outcome.thresholds_calibrated` must stay `false` |
| `physical` | requires measured base-to-mat transform, cube pose/size, tray pose/dims (each with sigma and a non-synthetic source), the initial robot state (or a segment's state) and a calibration reference. **Refuses** with a precise list of missing items and also while `calibration.calibration_blocker_resolved_by` is null (the documented sim-vs-follower-calibration mismatch). The shipped template has everything null, so it always refuses. There are no speculative offsets, no widened limits and no clipping |

`python -m sim_before_act scene-check --scene configs/scene_physical_template.yaml` prints the refusal list and exits 3.

## Controller

Targets are absolute joint-position targets in follower target space (degrees; gripper 0..100). `to_sim_rad` maps them to
MuJoCo position actuators; the twin tracks them with the model's PD (kp 998.22, kv 2.731, forcerange 3.35 N m; read back from the
compiled model and stored in every result). Zero-order hold at `control_hz`, then `settle_s` holding the last target. This is **not** the
Feetech STS3215 controller; tracking error, latency, speed limits and load behaviour of the real servos are not modelled. The 0..100 gripper mapping is unvalidated.

## Outcome predicates (`outcomes.py`)

All thresholds are in the config and are **ASSUMED / uncalibrated**; each result lists the numbers used and the thresholds.
Predicates are evaluated on samples (every `record_every_substeps`, default 10 ms); short transients can be missed.

| Predicate | Definition |
|---|---|
| constraint violation | joint beyond the simulator limit by > tolerance; non-cube penetration > tolerance. NaN / non-finite / huge `qvel` / MuJoCo warnings make the rollout **invalid** instead |
| contact categories | by geom name: `gripper_cube`, `arm_cube`, `cube_tray`, `cube_mat_table`, `arm_mat_table`, `gripper_mat_table`, `arm_self`, `arm_tray`, `gripper_tray`. Unsafe by default: `arm_mat_table`, `arm_self`, `arm_tray` (configurable). Gripper-cube contact is expected and not penalised. "Gripper" = bodies `gripper` + `moving_jaw`; everything else is "arm" |
| lift | cube **centre of mass** at least `lift_height_m` above the mat top at some sample |
| inside tray | at the final sample, COM inside the tray interior footprint (minus margin) and between floor top and rim top |
| release and settle | gripper opening above `gripper_open_pct` and cube speed below `v_settle_m_s`, over the trailing window (reaching the final sample) of at least `t_settle_s` |

Rollout outcome categories are separate: `task_success`, `task_failure` (reasons `cube_not_lifted` / `not_in_tray` / `not_released`),
`safety_violation`, `invalid_rollout` (instability, NaN, refused input), `timeout`, `unknown`. Primary category precedence is
invalid > timeout > safety > task result; `task_outcome` is always reported separately. **A rollout shorter than `task_horizon_s` has horizon
`segment`:** unmet predicates are *not evaluable as failures* and the outcome is `unknown` (a success seen within a segment horizon is
labelled as such, not as a full-task prediction).

## Running

```
python -m sim_before_act scene-rollout --fixture small_motion --seed 3 --perturb --out OUT --render --video
python -m sim_before_act scene-rollout --segment SEG.json --out OUT          # validated action segment
```
Python API: `run_scene_rollout(scene_cfg, segment_or_actions, seed, perturbation=None)`; `policy_to_sim(segment, scene, ...)`.
Perturbations come from the existing `PerturbationSampler` (config `perturbations:`, all placeholders), are range-checked and recorded.
Results: `<stem>_summary.json`, `<stem>_trace.npz`, `<stem>_plot.png` (cv2 trajectory/predicate timeline; matplotlib is not installed),
`<stem>_contact_sheet.png` and optional `<stem>_replay.mp4` (MuJoCo offscreen; the GL backend is tried in a subprocess in the order egl, osmesa, glfw and the
winner is reported; if none works only the cv2 plot is written). Examples are in `assets/scene_demo/` (synthetic fixture; not policy output).

## Status

| Item | Status |
|---|---|
| Scene builder, config schema, two modes, refusals | Implemented |
| Outcome predicates, rollout runner, visuals, CLI | Implemented |
| Tested on synthetic data (unit + headless rollouts of invented fixtures) | Yes |
| Tested on real recorded data | **No** |
| Physically validated (geometry, contacts, controller, thresholds) | **No** |
| Success path end-to-end (a full pick-and-place producing `task_success`) | Not exercised by a rollout; predicates tested on constructed states only |
| Blocked | calibration mismatch (sim limits/zero vs follower calibration); measurements: base-to-mat transform, cube/tray poses and dimensions, calibration reference |
