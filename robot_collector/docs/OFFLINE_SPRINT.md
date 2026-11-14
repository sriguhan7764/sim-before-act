# Remote Development Sprint - dependency-aware checklist

Boundary: robots powered off, remote access. **No serial ports, no `--check-only`, no bridge, no camera start, no discovery
tools that open ports, no register writes/torque, no physical ACT.** Offline/mock is the default everywhere.

Status vocabulary (kept separate; "implemented" never implies a later one):
`IMPL` implemented - `SYN` tested with synthetic data - `REC` tested with real recorded data - `PHYS` physically validated -
`BLOCKED` blocked by measurement/data.

## Environments (software-only inspection)

* `guhan_c`: py3.12, mujoco 3.15.0, numpy 2.5.4, cv2 5.0.0, pyyaml, pytest, pyarrow. **No torch.**
* `guhan_c_lerobot`: py3.12, torch 2.15.0+cpu, torchvision, datasets, safetensors, draccus, pyarrow, av, pandas. **No mujoco. No CUDA/XPU.**
* LeRobot 0.6.1 source: `/home/apic3/lerobot_v6_2/lerobot/src` (PYTHONPATH, read-only use).
* Design consequence: ACT runs in `guhan_c_lerobot` and writes a hashed **action-segment file**; MuJoCo verification runs in
  `guhan_c` and consumes it. The two never need to share a process.

## Hardware gate (new)

`collector/hardware_gate.py` fails closed. Real backends (`realsense`, `opencv`, `feetech_readonly`), RealSense/V4L2
enumeration and `inspect` enumeration call `hardware_gate.require()` and raise `HardwareAccessDenied` unless the process was
authorised with `collector --allow-hardware` **and** a typed `OPEN HARDWARE` (tty) or `--hardware-confirmed-by NAME`. A config
file never enables hardware; mock/tap backends need no authorisation. The authorisation provenance is stored in
`episode.json -> hardware_authorisation`. `lerobot_bridge` keeps its own stricter flags (`--i-authorise-motion` + typed
`ENABLE TORQUE`) and is not reachable from tests.
`tests/conftest.py::hardware_guard` (autouse) spies on `serial.Serial`, `cv2.VideoCapture`, `pyrealsense2`, device-path
`open`/`os.open` and device tools for **every** test; `tests/test_offline_guards.py` adds targeted checks.

## Checklist

| id | milestone | depends on | IMPL | SYN | REC | PHYS | notes |
|---|---|---|---|---|---|---|---|
| audit | repo audit + checklist | - | yes | n/a | n/a | n/a | this file |
| A | stabilise collection software | audit | yes | yes | no | no | deterministic bridge fixture; depth age/repeat checks; hardware gate + guards |
| B | dataset v0.1 schema, manifests, splits, inventory | A | yes | yes | no (manifest/inventory/stale tagging run read-only on the 5 earlier recordings: all `unclassified`) | no | `docs/DATASET_V0_1.md`; no 25/5 split exists (no qualifying demos) |
| C | ACT dataset conversion + audit | B | yes | yes | no | no | `collector/act/`; own npz layout (not LeRobotDataset v3); docs/ACT_CONVERSION.md; no real data qualifies yet |
| D | ACT train / infer configs, smoke | C | yes | yes | no | no | smoke only; never "trained"; CPU only; docs/ACT_TRAIN_INFER.md; writes action-segment-v0.1 |
| E | action contract + policy->sim adapter | C | yes | yes | no | BLOCKED | calibration mismatch stays a blocker |
| F | configurable MuJoCo cube->tray scene + predicates | E | yes | yes | no | BLOCKED | CAD geometry, not a calibrated twin; physical scene refuses without measurements |
| G | offline ACT -> verification integration | D, F | yes | yes | no | no | N = 1, 5, 10 nested seeds; thresholds UNCALIBRATED; evidence-gated: unknown outcomes -> INSUFFICIENT_EVIDENCE -> WOULD_REJECT (corrected after audit) |
| H | shadow-evaluation software | G | yes | yes | no | no | no hardware launch; no executor exists |
| I | evaluation/review reports | H, B | yes | yes | no | no | no ROC-AUC without both classes |
| J | one-command offline demo | G, I | yes | yes | no | no | all outputs labelled synthetic; hardware-refusing shim test |

(Final handoff, commands and exact test results: [HANDOFF.md](HANDOFF.md).)

## Milestone A - what changed

* Flaky `test_synthetic_episode_records_separate_signals_and_pilot_metadata`: the fixture used a wall-clock thread and a
  time-based signal, and compared fixed record indices. It now steps the bridge synchronously from the test thread with a scripted
  leader ramp, waits on ingestion conditions (no fixed sleeps) and asserts exact semantics (follower lags the leader by one cycle,
  `raw_goal` tracks the leader, `submission_status == "submitted_not_confirmed"`). 12/12 repeated runs passed under 8 CPU hogs.
* Sleep-based tests in `test_master_exports.py` / `test_recorder_mock.py` replaced by joins and condition waits (the
  remaining `sleep` calls inject deliberate writer latency or poll loops, not synchronisation).
* New: depth-age / repeat-rate checks (`docs/QUALITY_CHECKS.md`), `tests/test_milestone_a.py` (association boundaries,
  non-monotonic/duplicate samples, repeated vs missing depth, native->aligned provenance, export lineage, spy-based cleanup,
  interrupted recordings), `tests/test_offline_guards.py`.
* Acceptance already reports three separate verdicts (collector integrity / physical calibration / manipulation readiness);
  missing physical evidence is `NOT_CHECKED`, never `PASS`.

Tests actually run: `pytest -q` in `robot_collector` under `guhan_c` -> 96 passed (three consecutive runs).
