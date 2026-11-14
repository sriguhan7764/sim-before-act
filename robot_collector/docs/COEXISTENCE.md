## Update 2026-11-13: LeRobot bridge (preferred when LeRobot teleop is used)
`collector.lerobot_bridge` runs LeRobot's own SO-101 leader->follower loop (the only code that writes Goal_Position; refuses
without `--i-authorise-motion`, never recalibrates) and publishes follower/leader/command taps + `session.json`. The recorder
then opens only the cameras. The three command signals (leader measurement, clipped follower target, raw Goal_Position
ticks) are recorded separately; there are no torque/current commands and no acknowledgements. Commands are **submitted**
values: `Goal_Position` is not read back, so they do not confirm physical execution. Start the bridge BEFORE
the recorder (taps are read from end-of-file). Procedure: [PILOT_PROCEDURE.md](PILOT_PROCEDURE.md).

**Caveat:** the bridge's own loop writes only `Goal_Position`, but LeRobot's `connect()` runs `configure()` first, which writes
acceleration, operating mode, PID and gripper torque/current limits and enables follower torque (not homing offsets, limits or
calibration). A read-only `--check-only` mode and a start-gap guard run before that; the register read-back is stored in `session.json`.
The statement "never changes PID" in earlier versions of these docs was wrong for `connect()`.

The rest of this page (read-only Feetech adapter, hand-made taps) still applies if the bridge is not used.

# Coexisting with the existing teleoperation process

The recorder is a **passive observer**. It has no code path that sends goals, toggles torque, changes modes, homes or
calibrates. The Feetech adapter emits only PING (0x01) and READ (0x02) packets (tested: WRITE/REG_WRITE/SYNC_WRITE are
refused). `shutdown()` only closes the port/file - unlike LeRobot's `bus.disconnect()`, it never disables torque - and
quitting or aborting the recorder never touches your control loop.

## Serial bus sharing (important)
The Feetech bus is half-duplex; two processes polling the same port can corrupt each other's replies. The adapter opens
the port with `exclusive=True`, so if another process holds it (or locks it) the recorder fails to open instead of
interfering. Options:

1. **Dedicated telemetry port** (preferred when possible): e.g. record the *leader* arm through its own port with
   `robot.leader.backend: feetech_readonly`.
2. **Tap (recommended while teleop owns the follower port):** add a few lines to the existing teleop loop:
   ```python
   from collector.adapters.tap import TapWriter
   state_tap, cmd_tap = TapWriter("/tmp/follower_state.jsonl"), TapWriter("/tmp/commands.jsonl")
   # after reading Present_Position (raw=True values, i.e. encoder ticks):
   state_tap.write({"raw": {name: ticks, ...}})
   # right before sending the action:
   cmd_tap.write({"values": {name: goal, ...}, "units": "<state the real units>"})
   ```
   then set `robot.follower.backend: jsonl_tap` / `robot.commands.backend: jsonl_tap` with `path:`. `time.monotonic_ns`
   is shared between processes on the same host, so tap timestamps align with camera receipt times. The tap writes are
   a file append in your loop; the recorder reading them never feeds back. LeRobot's `get_observation()` returns
   *normalised* values (degrees / 0-100), not ticks - tap raw `bus.sync_read("Present_Position", normalize=False)` if you
   want raw ticks, or you must record what you actually write (the schema labels `raw` as encoder ticks).
3. Teleop stopped: use `feetech_readonly` on the follower port.

Commands are only recorded if the teleop side taps them; the recorder cannot observe them on the bus. Command
acknowledgements are not available. The commanded value is never used in place of measured state (separate files and
separate association fields).

Camera sharing: a RealSense device can be opened by only one process through the SDK; do not run LeRobot's RealSense
camera in the teleop process at the same time as the recorder.
