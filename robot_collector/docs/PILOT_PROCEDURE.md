# Recording ONE setup pilot (first acceptance gate) - 640x480 @ 30 FPS

**Status: prepared, NOT yet run on hardware.** Every test so far is synthetic (`tests/`). Nothing below starts until you
explicitly authorise each hardware step. A collector PASS says the *recorder and exports are internally consistent*; it does
**not** validate joint signs, zero offsets, gripper mapping, object poses or base alignment (see "Three verdicts").

Architecture: LeRobot owns both arm buses (`collector.lerobot_bridge`, motion) and publishes JSONL taps; the collector opens
**only the cameras** and reads the taps (`record --taps-dir`). They never share a serial port. A RealSense can be opened by one
process only - do not run `run.sh record` at the same time.

Every command runs from `/home/apic3/guhan_c/robot_collector`.
`C=/home/apic3/miniforge3/envs/guhan_c/bin/python` (collector), `L=/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python` (LeRobot).

## Requested configuration (`configs/setup_lerobot.yaml`)
| stream | requested | notes |
|---|---|---|
| D435 colour | 640x480 rgb8 @ 30 | |
| D435 native depth | 640x480 z16 @ 30 | depth sensor geometry; 0 = invalid |
| D435 aligned depth | colour geometry (640x480) | derived with `rs.align` (resampled); stored separately with colour intrinsics |
| wrist (OpenCV/V4L2) | 640x480 @ 30 | no device timestamp / frame counter available |

The negotiated profile is read from the device after the stream starts and compared with the request. A mismatch raises
`ProfileMismatch`, nothing is recorded, nothing is substituted. The effective config is frozen per episode in
`config/effective_config.json` with its sha256 (`episode.json: effective_config_sha256`).

## Calibration files (`configs/calibration_registry.yaml`)
| role | ACTIVE file (sha256 prefix) |
|---|---|
| follower | `lerobot_setup/calibration/follower/g_c_follower_calibration.json` (`99548f73`) |
| leader | `lerobot_setup/calibration/leader/g_c_leader_calibration.json` (`9eba342b`) |

Stale, preserved but excluded: `data/robot_calibration/follower_calibration_guhan_c.json`, `leader_guhan_c.json`, `follower.json`.
Preflight, the recorder and acceptance all refuse a config that points at (or hashes like) a stale file. The three early
hardware episodes (`ep_20261113T101447Z_83f1ba`, `..101628Z_f3cde4`, `..104200Z_e68da4`) used the stale file: never interpret
them with the active calibration. If calibration is in doubt, **stop before follower actuation** and report it.

## Hardware actions that need your explicit go-ahead
| # | Action | Effect |
|---|---|---|
| A | `preflight --open-cameras` (step 2) | Starts the cameras for a few seconds. No motion. |
| B | bridge `--check-only` (step 3) | Opens both serial ports and READS registers. No torque, no writes, no motion. |
| C | start the bridge (step 5) | **ENABLES FOLLOWER TORQUE; the follower tracks the leader.** Also runs LeRobot's `configure()` (below). |
| D | the recording (step 6) | Cameras + taps; you move the leader by hand. |
| E | physical measurements | Operator work. |

**What LeRobot's `connect()` writes** (verified in the LeRobot 0.6.1 source; listed in `lerobot_bridge.LEROBOT_CONNECT_WRITES` and
saved in `session.json`): `Return_Delay_Time=0`, `Maximum_Acceleration=254`, `Acceleration=254`, Phase bit 4 cleared,
`Operating_Mode=POSITION`, P/I/D = 16/0/32 on the follower, gripper `Max_Torque_Limit=500`, `Protection_Current=250`,
`Overload_Torque=25`, then torque is enabled. It does **not** write Homing_Offset, position limits or calibration. (An earlier
version of these docs said the bridge "never changes PID"; that was wrong for LeRobot's own `connect()`.) The bridge itself
only writes `Goal_Position` through `send_action`. What the servo firmware does with its old `Goal_Position` at torque-enable
is **unverified**, so the start-gap guard (step 3) and a hand on the follower power supply are the mitigation.
Nothing here homes, calibrates, runs ACT or moves toward a mechanical stop.

## 1. Pre-flight (no hardware access beyond USB enumeration)
```bash
$C -m pytest -q
$C -m collector preflight --config configs/setup_lerobot.yaml
```
Checks: frozen config hash; active/stale calibration; D435 supports colour rgb8 640x480@30 and depth z16 640x480@30 and the SDK
can resolve both together (`rs.config.can_resolve`, no streaming); wrist cam formats via `v4l2-ctl`. Saved to `data/preflight/`.
(Last result, 2026-11-13: supported and resolvable; the *negotiated* profile is not yet verified.)

## 2. Negotiated profiles (authorised: camera access)
```bash
$C -m collector preflight --config configs/setup_lerobot.yaml --open-cameras --seconds 3
```
Prints requested vs negotiated colour / native depth / aligned depth, intrinsics, extrinsics, depth scale. Any mismatch =>
stop and tell me; do not change the request to match.

Timing fields in the saved report (`opened_streams.<cam>.timing_boundaries`): `requested_capture_window_s` (the `--seconds`
argument), `loop_wall_s` (first read request to loop exit; stream open/warm-up excluded), `first_frame_latency_s`,
`first_to_last_frame_s`, and `measured_rate_hz = (frames-1) / first_to_last_frame_s`. The rate is NOT frames / capture wall
time. A 3 s preflight is a smoke test, not the recording acceptance test: sustained timing and association quality come from
`quality-report` on the recorded episode.

Depth freshness (`depth_freshness`, and in recorded episodes `quality`/`acceptance`/export rows): a frameset may carry the
previous depth frame again. The recorder keeps that frame's ORIGINAL depth frame number and device timestamp, so a repeat is
detectable (`repeated_previous_depth`); `missing` = no depth frame/file. Aligned depth is derived from the same frameset's native
depth frame and inherits its status; each export row states `depth_status`, `depth_frame_number`, `depth_device_timestamp_ms` and
`depth_aligned_source`. Repeats are reported (acceptance WARN), never silently treated as fresh; missing depth is a FAIL. No
threshold is applied to the repeat fraction.
Camera-state/command associations are separate: samples further than the documented tolerance are `unmatched_beyond_tolerance`,
never reused as stale values.

## 3. Operator safety check, then read-only motor check (authorised: serial read)
1. Clear the workspace of people, cables and objects; keep the mat clear of the arm's path.
2. Place leader and follower in a **similar, comfortable mid-range pose** (arm folded over the mat, gripper mid-open), away from stops.
3. One hand on the follower power supply. Know where the leader is held.
```bash
HERE=/home/apic3/guhan_c
L=/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python
BR="env PYTHONPATH=/home/apic3/lerobot_v6_2/lerobot/src:$HERE/robot_collector $L -m collector.lerobot_bridge \
  --follower-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14030118-if00 \
  --leader-port   /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46084050-if00 \
  --follower-id g_c_follower_calibration --leader-id g_c_leader_calibration \
  --follower-cal-dir $HERE/lerobot_setup/calibration/follower --leader-cal-dir $HERE/lerobot_setup/calibration/leader \
  --taps-dir /tmp/collector_taps"
$BR --check-only
```
It verifies that the motors' homing offsets/ranges match both active calibration files (exit 3 if not - never recalibrates)
and that no joint of leader and follower differs by more than `--max-start-gap` (default 15 degrees / gripper %). If it
refuses, nothing was enabled or written.

**Exact `--check-only` path** (`collector/lerobot_bridge.py: main`, LeRobot 0.6.1 source in `/home/apic3/lerobot_v6_2`):
1. Construct `SO101Follower`/`SO101Leader` (no bus traffic; reads the calibration JSON; `mkdir(exist_ok)` on the existing dir).
2. `fol.bus.connect()` and `lea.bus.connect()` - the BUS-level connect, **not** `SO101Follower.connect()`: open the port, then
   LeRobot's handshake = PING every expected motor id (model-number reply) + read Firmware_Major/Minor_Version per motor;
   then set the host-side packet timeout. No `configure()`, no torque change, no register write.
3. `startup_check`: `is_calibrated` = per motor READ of `Min_Position_Limit`, `Max_Position_Limit`, `Homing_Offset`, compared
   with the calibration file; then one `sync_read("Present_Position")` of each arm, normalised in software with the files'
   calibration.
4. `bus.disconnect(disable_torque=False)` for both = close the port only (no torque write).
5. Exit 0 (OK), 3 (refused; nothing written), or a traceback if a motor does not answer.
Bridge module is also covered by a static test (`test_bridge_module_has_no_unauthorised_motion_paths`: no `sync_write`, torque,
calibrate or `configure` names in the module; `send_action` and `connect(calibrate=False)` only after `--i-authorise-motion`).
It is a source check, not proof about the wire. Residual unknowns: opening a CH340-type USB serial port may toggle DTR/RTS
(unverified; ordinarily harmless on these bus adapters), and if torque was ALREADY enabled by an earlier process it stays
as it was - this check does not change it.

## 4. Authorisation
Reply "go" for step 5. Say "stop" at any time; Ctrl-C in the bridge terminal releases follower torque.

## 5. Start the bridge (terminal 1) - **ENABLES FOLLOWER TORQUE AND TRACKING**
```bash
$BR --seconds 120 --i-authorise-motion
```
It re-runs the check, prints a torque warning and requires you to type `ENABLE TORQUE` (needs a real terminal). Follower motion
is capped by `max_relative_target` (10 normalised units/cycle). The bridge writes `session.json` including the register
read-back after connect.

## 6. Record the setup episode (terminal 2, within the bridge's 120 s)
```bash
$C -m collector record --config configs/setup_lerobot.yaml --taps-dir /tmp/collector_taps \
   --task pick_red_cube_place_blue_tray --instruction "setup pilot: stationary, small supervised motion, stationary" \
   --role setup --attempt-index 1 --protocol setup_short
```
Timed cues (nominal, ~22 s): `stationary_start` 0-5 s (hands off both arms) -> `small_supervised_motion` 5-17 s (move the **leader**
slowly and gently, well inside its range; never push toward a stop; release it and stop if anything looks wrong) ->
`stationary_end` 17-22 s. Cue times are stored in `episode.json: protocol` and are prompts only. Then Ctrl-C the bridge.
The episode is recorded as `setup`, outcome `unlabelled`, with no cube/tray pose (nothing is invented).
**A pilot recording is refused (nothing recorded) if any source is down, a calibration file is stale, or a camera negotiated
a different profile.**

## 7. After the episode (no hardware)
```bash
EP=data/episodes/<episode id printed by record>
$C -m collector validate       --episode $EP
$C -m collector quality-report --episode $EP
$C -m collector replay         --episode $EP --no-display      # no robot, no camera
$C -m collector acceptance     --episode $EP                   # writes both exports, then checks everything
```
Exports (`$EP.exports/{rgb_joint_angles,rgbd_motor_positions}`) are derived only; the master is read-only and hashed. Thresholds
are fixed constants in `collector/acceptance.py` (state/command coverage >= 0.95, mapping diff <= 1e-3); they are not tuned to
get a PASS.

## Three verdicts (reported separately)
| verdict | question | possible values |
|---|---|---|
| `collector_integrity` | recorded, hashed, replayable, exportable, units and frame associations consistent, frozen config and calibration refs correct? | PASS / FAIL |
| `physical_calibration` | joint zero offsets, signs, gripper mapping, joint limits, camera-to-mat, base-to-mat physically validated? | PASS / FAIL / **NOT_VALIDATED** |
| `manipulation_readiness` | cube/tray poses measured, outcome label with source, eligible as accepted demonstration? | READY / **NOT_READY** |

Statuses: `PASS` (checked and met), `FAIL` (checked and not met), `WARN` (not fatal), `NOT_CHECKED` (nothing in the recording can
establish it; never a pass). A setup episode is expected to be `PASS / NOT_VALIDATED / NOT_READY`.

## Reading the data correctly
- `follower_state` = **measured**. `leader_state` = the operator's arm. `commands` = follower targets **submitted** for writing
  (clipped leader measurement) plus the raw `Goal_Position` ticks computed from them.
- `Goal_Position` is **not read back** and there is no acknowledgement: a submitted value is not confirmation of physical
  execution. The measured follower state (and the command-vs-state lag statistic) is the only evidence of motion.
- Joint angles: collector `AngleMapping` equals LeRobot's normalisation inside the calibrated range; outside it the collector
  wraps on the 4096-tick circle and LeRobot does not (360-degree difference, physically implausible).
- Zero = calibration-range midpoint, not a physically validated reference. Encoder range is not a safe range.
- Association is host-monotonic only; there is no hardware synchronisation. Native depth and aligned depth are different
  representations; aligned depth is resampled by the SDK.

## 8. Label afterwards (never inferred)
```bash
$C -m collector annotate --episode $EP --outcome unknown --notes "setup pilot"
```
Success/failure needs `--label-source operator_visual|operator_measured`.
