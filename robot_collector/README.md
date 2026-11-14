# Robot demonstration data collector

Records a **lossless canonical master episode** (RealSense RGB + native uint16 depth + colour-aligned depth, optional
wrist camera, follower telemetry, commands, leader state, calibration, metadata) and derives independent **export
profiles** from it. Nothing here trains models, simulates, or scores research hypotheses.

> **Status: software complete for the mock backend; NO hardware has been connected or tested.**
> See [docs/STATUS.md](docs/STATUS.md) for exactly what is implemented / unit-tested / hardware-tested / blocked.

## Setup (conda env `guhan_c`, already created)

```bash
conda activate guhan_c            # Python 3.12, numpy, OpenCV (aruco), PyYAML, pyserial, tk, pyrealsense2, pytest
cd /home/apic3/guhan_c/robot_collector
# to recreate:  conda env create -f environment.yml
python -m pytest -q               # 59 tests, synthetic fixtures only
```

What `inspect` found on this machine (read-only discovery): the robot stack is **LeRobot 0.6.1** (SO-101, Feetech
STS3215, motor ids 1-6, `lerobot/robots/so_follower/so_follower.py`); existing teleop in `so101_mujoco/`; calibration
JSONs under `~/.cache/huggingface/lerobot/calibration/` (which one belongs to your follower is **not inferred**).
The collector does not import or modify LeRobot and does not touch those files.

## Commands

```bash
python -m collector inspect                                   # environment/device discovery (never opens ports)
python -m collector preview   --config configs/setup.yaml     # desktop preview, no recording controls
python -m collector calibrate --config configs/setup.yaml     # ArUco multi-frame camera->mat calibration
python -m collector record    --config configs/setup.yaml --task pick_place     # desktop UI (or --headless/--duration N)
python -m collector validate  --episode data/episodes/<id>
python -m collector replay    --episode data/episodes/<id>    # offline visualisation only (also works on exports)
python -m collector export    --episode data/episodes/<id> --profile rgb_joint_angles
python -m collector export    --episode data/episodes/<id> --profile rgbd_motor_positions
python -m collector export    --episode data/episodes/<id> --profile full_master [--link]
python -m collector recover   --episode data/episodes/<id>.inprogress
python -m collector annotate  --episode data/episodes/<id> --outcome success --label-source operator_visual
python -m collector quality-report --episode data/episodes/<id>   # timing, drops, residuals, units, clock assumptions
python -m collector acceptance     --episode data/episodes/<id>   # first acceptance gate: 3 verdicts (integrity / physical calibration / manipulation readiness)
python -m collector preflight --config configs/setup_lerobot.yaml [--open-cameras]   # no motion; frozen config hash, active/stale calibration, D435 profile support
# pilot recording: add --role setup|demonstration|evaluation --cube-position-id P1 --layout configs/pilot_layout.yaml [--protocol setup_short]
# LeRobot teleop host: python -m collector.lerobot_bridge ... --check-only | --i-authorise-motion  (see docs/PILOT_PROCEDURE.md)
```

Dry run with explicit **mock** data (clearly flagged everywhere):
`python -m collector record --config configs/mock.yaml --task smoke --duration 3`

## Fill in `configs/setup.yaml` (all left unset on purpose)

`cameras[].serial` / wrist `device`, `robot.follower.port` (+ `calibration_file`), `markers.dictionary`,
marker ids and `orientation_deg`, `markers.size_excludes_white_margin_confirmed`, `robot.control_mode`.
Unset hardware values make that source report *not connected*; they are never guessed, and mock data is never
substituted silently (mock needs `backend: mock`).

## Design summary

- **Master first**, written to `<id>.inprogress/`, hashed, made read-only and atomically renamed to `<id>/`.
- Exports go to `<id>.exports/<profile>__v1/`, deterministic, byte-copied images, manifest naming the parent master.
- Acquisition workers (one thread per camera/telemetry source) -> bounded queue -> single dedicated writer.
  Queue overflow is counted per stream and either flags the episode (`complete_with_drops`) or aborts it
  (`recording.overflow_policy`). Requested FPS is not a guarantee; measured rates are shown and stored.
- Synchronisation = derived association table on the host monotonic clock with explicit tolerances, residuals and
  `unmatched` markers. **No hardware sync is claimed.** See [docs/SCHEMA.md](docs/SCHEMA.md).
- `diagnostic` mode records without calibration but flags it; `research` mode enforces `research_gates`.
- Robot access is **read-only** (PING/READ only). See [docs/COEXISTENCE.md](docs/COEXISTENCE.md).

## Docs

[SCHEMA](docs/SCHEMA.md) · [CALIBRATION](docs/CALIBRATION.md) · [COEXISTENCE with teleop](docs/COEXISTENCE.md) ·
[SMOKE TEST](docs/SMOKE_TEST.md) · [PILOT PROCEDURE](docs/PILOT_PROCEDURE.md) · [TROUBLESHOOTING & RECOVERY](docs/TROUBLESHOOTING.md) · [STATUS](docs/STATUS.md)
