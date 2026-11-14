# Implementation status

**Update 2026-11-14 (remote offline sprint, milestones A-J).** Hardware untouched. Final suites: `robot_collector` 134 passed, `sim_before_act`
163 passed, torch checks 69 passed (all synthetic; exact commands, statuses and measurements still required: [HANDOFF.md](HANDOFF.md)).
The earlier test counts below are historical.

**Update 2026-11-13 (pilot preparation, 640x480@30).** Still **no hardware acceptance run and no stream or bridge started**.
Tests run: `pytest -q` in `robot_collector` -> 72 passed (all synthetic/mock; 59 before this step + 13 new); `sim_before_act` -> 90 passed.
Query-only preflight run on this machine (no stream, no motion): D435 `844212071704` lists colour rgb8 and depth z16 at 640x480@30 and
`rs.config.can_resolve` accepts the combination; both calibration files are the registry's ACTIVE ones (hashes match).

| Added in the preparation step | Implemented | Tested (synthetic) | Hardware |
|---|---|---|---|
| Requested vs negotiated profile (RealSense, OpenCV, mock); `ProfileMismatch` refuses, never substitutes | ✔ | ✔ (OpenCV fake capture; negotiation record) | ✘ RealSense negotiation not run |
| Native depth and aligned depth described separately (size, geometry, intrinsics, scale) | ✔ | ✔ (mock) | ✘ |
| Calibration registry (active vs stale), recorder/acceptance refuse stale | ✔ | ✔ (real file hashes, read-only) | n/a |
| Frozen effective config + sha256 per episode | ✔ | ✔ incl. tamper -> FAIL | n/a |
| `preflight` (query-only) and `preflight --open-cameras` | ✔ | ✘ (needs device) | query-only ✔; `--open-cameras` ✘ not run |
| Bridge `--check-only`, start-gap guard, typed torque confirmation, register read-back | ✔ | ✔ (guard logic, fakes) | ✘ not run |
| `setup_short` protocol cues + per-segment statistics (no thresholds) | ✔ | ✔ | ✘ |
| Acceptance: three verdicts, `NOT_CHECKED` instead of `SKIP`, camera/config/depth checks | ✔ | ✔ | ✘ |

**Correction:** earlier docs said the bridge "never changes limits, offsets or PID". LeRobot's own `connect()` runs `configure()`, which does
write acceleration, operating mode, PID and gripper torque/current limits and enables torque (listed in PILOT_PROCEDURE.md). It does not
write homing offsets, position limits or calibration. The bridge itself writes only `Goal_Position`.

**Update 2026-11-13 (collector-readiness milestone).** Still **no hardware acceptance run**: the first acceptance gate
([PILOT_PROCEDURE.md](PILOT_PROCEDURE.md)) has NOT been executed on a real operator episode. Tests run at that point: `pytest -q` -> 59 passed
(45 earlier + 14 in `tests/test_bridge_pilot.py`), all synthetic/mock. Ad-hoc, no-hardware check: LeRobot `_normalize` (real calibration
files) equals the collector `AngleMapping` to ~1e-14 deg inside the calibrated range (differs by 360 deg outside it);
`collector.lerobot_bridge` imports in env `guhan_c_lerobot`.

| New in this milestone | Implemented | Tested (synthetic) | Hardware |
|---|---|---|---|
| LeRobot bridge (taps for follower/leader/commands + session.json) | ✔ | ✔ fake buses | ✘ not run |
| Pilot metadata (role, cube position id, poses, tray, start config, label source) | ✔ | ✔ | n/a |
| Commands + leader + source-frame indices in both lightweight exports | ✔ | ✔ | n/a |
| Duplicate / non-monotonic timestamp warnings | ✔ | ✔ | n/a |
| `quality-report`, `acceptance` | ✔ | ✔ pass + tampered-export fail | ✘ |
| Hardware synchronisation | not available (host-monotonic association only) | | |
| Wrist-camera device timestamp / frame counter | not available | | |
| Measured cube/tray/marker poses; validated mat->base transform | **missing** (operator measurement) | | |
| Validated zero offsets, signs, gripper mapping, joint limits | **missing** | | |

The older status table below predates this milestone (2026-11-11).

# Implementation status (as of 2026-11-11)

**No robot, RealSense or wrist camera was connected. Nothing is hardware-tested.** Mock/synthetic data only.

Legend: **I** implemented · **U** unit-tested (synthetic/mock) · **H** hardware-tested · **B** blocked

| Area | I | U | H | B / notes |
|---|---|---|---|---|
| Conda env `guhan_c`, environment.yml | ✔ | - | - | |
| Environment inspection (`inspect`) | ✔ | - (run manually) | ✔ partial: ran on this machine, no devices opened | Does not identify which LeRobot calibration file is the follower's |
| Config + units validation, unset-unknowns | ✔ | ✔ | - | |
| Marker geometry, corner order, transforms | ✔ | ✔ | - | Marker dictionary/ids/orientations unknown - must be configured |
| ArUco detect, multi-frame PnP, reproj/stability, immutable snapshots | ✔ | ✔ (synthetic projections, generated marker image) | ✘ | `calibrate` CLI interactive flow not unit-tested; needs camera |
| RealSense adapter (colour, native + aligned depth, intrinsics, extrinsics, scale, settings) | ✔ | ✘ | ✘ | Written against pyrealsense2 API; no device. `pyrealsense2` 2.59 installs/imports |
| OpenCV wrist-camera adapter | ✔ | ✘ | ✘ | device path unset |
| Feetech read-only adapter (PING/READ) | ✔ | ✔ (fake serial, packet codec, write refusal) | ✘ | Register layout from lerobot 0.6.1 tables; real baud/IDs/timing unverified |
| Tap adapter (teleop coexistence) | ✔ | ✘ | ✘ | Requires small hook in existing teleop script (not applied by me) |
| Leader state, commands | ✔ (via adapters) | partial (mock commands) | ✘ | Commands need teleop tap; acks unavailable |
| Mock backends (explicit, flagged) | ✔ | ✔ | n/a | |
| Master writer: bounded queue, writer thread, in-progress dir, atomic publish, hashes, read-only | ✔ | ✔ | ✘ | Throughput on real hardware unknown |
| Overflow accounting (flag/abort), disk monitor | ✔ | overflow ✔, disk monitor ✘ | ✘ | |
| Association/sync tables, stale rejection | ✔ | ✔ | ✘ | Host-receipt clock only; no hardware sync; interpolation not implemented |
| Exports: rgb_joint_angles, rgbd_motor_positions, full_master | ✔ | ✔ (round trip, determinism, missing modality, uint16) | ✘ | Joint-angle export needs your robot calibration file |
| validate / recover / replay (offline) / annotate | ✔ | ✔ (replay GUI window ✘) | ✘ | |
| Desktop UI (tkinter) | ✔ | ✘ | ✘ | Not exercised (no interaction run); headless CLI recording is exercised with mock |
| Recorder gates (diagnostic/research), no-write shutdown | ✔ | ✔ | ✘ | |
| Robot-base-to-mat calibration | ✘ | - | - | Out of scope/blocked: not inferred; stored as null |
| Smoke test (docs/SMOKE_TEST.md) | procedure written | - | **not run** | Needs hardware |

## Needed from you to unblock hardware integration
ArUco dictionary + ids + orientations; RealSense serial (optional); follower serial port; which LeRobot calibration JSON
belongs to the follower; whether teleop will tap state/commands; wrist camera device (if any).
