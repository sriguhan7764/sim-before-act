# Sim-Before-Act for SO-101

An M.Tech research prototype for collecting SO-101 teleoperation demonstrations and evaluating proposed robot actions offline with MuJoCo. The collector stores RGB-D frames, joint telemetry, commands, and calibration metadata in a lossless master episode. Sim-Before-Act runs proposed action segments under sampled conditions, checks outcomes, and records whether the offline gate would accept or reject the segment.

The workflow has been exercised with synthetic and mock data. Physical validation is pending, and the follower-to-simulator calibration mismatch remains open. The ACT integration uses a randomly initialized policy for a smoke test. Training and evaluating a task policy remain future work. [Calibration audit](sim_before_act/docs/CALIBRATION_AUDIT.md) · [Simulation status](sim_before_act/docs/STATUS.md) · [Collector status](robot_collector/docs/STATUS.md)

## Run the offline demo

From the repository root, use Python 3.12 in an activated virtual environment. Both packages are needed because the demo runs the collector on mock inputs before passing a synthetic action segment to MuJoCo.

```bash
python -m pip install -e ./robot_collector -e "./sim_before_act[lerobot-import]" pytest
python -m sim_before_act.demo --out sim_before_act/outputs/local_demo --no-act-smoke --no-render
```

Use a fresh output path for each run. `--no-act-smoke` lets the demo run without a separate LeRobot environment. `--no-render` skips the contact sheet and still produces plots. Start with `DEMO_REPORT.md` in the output directory. [Reference demo report](sim_before_act/outputs/offline_demo_gate_fix_v1/DEMO_REPORT.md) · [Synthetic rollout assets](sim_before_act/assets/README.md)

The offline gate treats missing outcome evidence as `INSUFFICIENT_EVIDENCE` and returns `WOULD_REJECT`. Synthetic results exercise the pipeline and gate. Real-world predictive validity remains untested.

## Repository layout

| Directory | Contents |
| --- | --- |
| [`robot_collector/`](robot_collector/) | RGB-D and telemetry collection, episode validation, export profiles, and ACT conversion. |
| [`sim_before_act/`](sim_before_act/) | MuJoCo rollouts, action contracts, evidence gating, and synthetic demos. |
| [`lerobot_setup/`](lerobot_setup/) | Lab-specific Linux teleoperation wrapper and calibration files. |
| [`so101_mujoco/`](so101_mujoco/) | SO-101 MJCF model and meshes used by the simulator. |

Keep these directories side by side so the configuration files can resolve model and calibration paths. Raw recordings and training checkpoints are stored outside the repository. Hardware profiles in this checkout contain lab-specific device identifiers and paths.

## Run the tests

```bash
python -m pytest -q robot_collector
python -m pytest -q sim_before_act
```

Run the suites separately. ACT integration tests that require the separate LeRobot environment may skip when it is unavailable. [Environment and test details](robot_collector/docs/HANDOFF.md)

## Model attribution

The bundled SO-101 model is based on TheRobotStudio’s SO-ARM100 project. This checkout contains the MJCF model and a subset of the upstream assets. The upstream Apache 2.0 license applies to the upstream material. The project code has no repo-wide license. [Upstream SO-101 model](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) · [Bundled license](so101_mujoco/LICENSE)
