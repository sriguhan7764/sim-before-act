"""Leader->follower teleoperation for the SO-101 pair (project-local; the ONLY collector code that writes to motors).

Kept deliberately separate from the recorder: the recorder's adapters stay PING/READ only. This process owns BOTH
serial ports (half-duplex buses must have a single opener), reads follower and leader, writes follower goals, and
publishes everything to JSONL taps (collector.adapters.tap) that a recorder reads via ``jsonl_tap`` backends
(configs/setup_teleop.yaml). The recorder therefore never touches the buses.

Motor writes are restricted by an allowlist to:  Torque_Enable (addr 40) and Goal_Position (addr 42).
Nothing else is ever written (no offsets, modes, limits, PID, EEPROM).

Mapping (relative, because no leader calibration file exists): at start the follower's present position and the
leader's present position are captured as references; afterwards
    goal = clamp( follower_ref + sign * scale * (leader - leader_ref) )
so the follower NEVER jumps at start (goal == present position before torque is enabled). Goals are clamped to the
follower calibration range (+margin) when one is known, and slew-limited per cycle. Leader deltas are wrap-aware
(4096-tick circle). This assumes leader and follower joints turn in the same direction per tick (same assembly);
verify with --dry-run first and use --invert for any joint that goes the wrong way.

Command log units: raw Goal_Position register ticks (same representation as follower Present_Position).
"""
from __future__ import annotations

import json
import os
import signal
import time

from .adapters.feetech_readonly import FeetechReadOnly, parse_status
from .adapters.tap import TapWriter
from .util import mono_ns, wall_ns

ADDR_TORQUE, ADDR_GOAL = 40, 42
INSTR_WRITE, INSTR_SYNC_WRITE = 0x03, 0x83
_ALLOWED_ADDRS = {ADDR_TORQUE, ADDR_GOAL}
CMD_UNITS = "feetech_goal_position_register_ticks_after_motor_homing_offset"
TICKS = 4096


def wrap_delta(cur: int, prev: int) -> int:
    d = (cur - prev) % TICKS
    return d - TICKS if d > TICKS // 2 else d


def checksum_packet(body: bytes) -> bytes:
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def build_write(motor_id: int, addr: int, data: bytes) -> bytes:
    if addr not in _ALLOWED_ADDRS:
        raise PermissionError(f"teleop refuses to write register {addr}")
    return checksum_packet(bytes([motor_id, len(data) + 3, INSTR_WRITE, addr]) + data)


def build_sync_goal(goals: dict[int, int]) -> bytes:
    params = bytes([ADDR_GOAL, 2])
    for mid, g in goals.items():
        g = int(g) % TICKS
        params += bytes([mid, g & 0xFF, g >> 8])
    return checksum_packet(bytes([0xFE, len(params) + 2, INSTR_SYNC_WRITE]) + params)


def initial_unwrapped(raw: int, rng: tuple[int, int] | None, margin: int) -> int:
    """Pick raw+k*4096 that lies inside the calibration window (ranges may exceed 4095 when a joint crosses the wrap)."""
    if rng is None:
        return raw
    lo, hi = rng
    for k in (0, 1, -1):
        u = raw + k * TICKS
        if lo - margin <= u <= hi + margin:
            return u
    raise ValueError(f"present position {raw} is outside calibration range {lo}..{hi} (+/-{margin}); refusing to start")


def plan_goal(cmd_prev: float, f_ref: int, l_ref: int, l_unwrapped: int, sign: int, scale: float,
              rng: tuple[int, int] | None, margin: int, max_step: float) -> tuple[float, float]:
    """Returns (new_commanded_goal, unclamped_target). Pure function (unit-tested)."""
    target = f_ref + sign * scale * (l_unwrapped - l_ref)
    clamped = target if rng is None else min(max(target, rng[0] - margin), rng[1] + margin)
    step = min(max(clamped - cmd_prev, -max_step), max_step)
    return cmd_prev + step, target


class TeleopBus(FeetechReadOnly):
    """Reads like the recorder adapter; adds allowlisted writes. Only the follower instance ever writes."""

    def write_reg(self, mid: int, addr: int, data: bytes) -> None:
        self.ser.write(build_write(mid, addr, data))
        parse_status(self.ser.read(6), mid)

    def sync_goals(self, goals: dict[int, int]) -> None:
        self.ser.write(build_sync_goal(goals))

    def read_reg(self, mid: int, addr: int, n: int = 1) -> int:
        d = self._txrx(mid, 0x02, bytes([addr, n]))
        return int.from_bytes(d, "little")


def _rng_map(cal_path: str | None) -> dict[str, tuple[int, int]]:
    if not cal_path or not os.path.exists(cal_path):
        return {}
    with open(cal_path) as f:
        return {k: (v["range_min"], v["range_max"]) for k, v in json.load(f).items()}


def run_teleop(cfg: dict, hz: float = 100.0, max_step: float = 12.0, scale: float = 1.0, margin: int = 100,
               invert: list[str] | None = None, taps_dir: str = "/tmp/collector_taps", dry_run: bool = False,
               on_exit: str = "release", seconds: float | None = None, confirm=input, log=print) -> int:
    r = cfg["robot"]
    fcfg, lcfg = dict(r["follower"], poll_hz=10000.0), dict(r["leader"], poll_hz=10000.0)
    names = [j["name"] for j in fcfg["joints"]]
    ids = {j["name"]: j["id"] for j in fcfg["joints"]}
    sign = {n: (-1 if n in (invert or []) else 1) for n in names}
    ranges = _rng_map(r["follower"].get("calibration_file"))
    fol, lead = TeleopBus(fcfg, "follower"), TeleopBus(lcfg, "leader")
    stop = {"flag": False}
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.update(flag=True))
    torque_on = False
    try:
        lead.open()
        fol.open()
        # safety: identify roles by bus voltage again and verify the follower is in position mode
        fp, lp = fol.poll()[0], lead.poll()[0]
        if not (fp["read_ok"] and lp["read_ok"]):
            raise RuntimeError(f"initial read failed: {fp['errors']} {lp['errors']}")
        fv, lv = min(fp["voltage_raw"].values()), max(lp["voltage_raw"].values())
        if fv < 80 or lv > 80:
            raise RuntimeError(f"role/voltage check failed (follower {fv/10:.1f} V, leader {lv/10:.1f} V): ports swapped?")
        modes = {n: fol.read_reg(ids[n], 33) for n in names}
        if any(m != 0 for m in modes.values()):
            raise RuntimeError(f"follower not in position mode (Operating_Mode={modes}); refusing")
        if any(lead.read_reg(ids[n], ADDR_TORQUE) for n in names):
            raise RuntimeError("leader torque is enabled; refusing (leader must be free to move by hand)")
        f_ref = {n: initial_unwrapped(fp["raw"][n], ranges.get(n) if n != "wrist_roll" else None, margin) for n in names}
        l_ref = dict(lp["raw"])
        l_u = {n: l_ref[n] for n in names}
        l_prev = dict(l_ref)
        cmd = {n: float(f_ref[n]) for n in names}
        log(f"follower start (raw): {fp['raw']}\nleader start (raw):   {lp['raw']}\nmode: {'DRY-RUN (no writes)' if dry_run else 'LIVE'}")
        os.makedirs(taps_dir, exist_ok=True)
        paths = {k: os.path.join(taps_dir, f"{k}.jsonl") for k in ("follower_state", "leader_state", "commands")}
        for p in paths.values():
            open(p, "w").close()
        taps = {k: TapWriter(p) for k, p in paths.items()}
        log(f"taps: {taps_dir}")
        if not dry_run:
            log("SUPPORT THE FOLLOWER ARM. Torque will be enabled at its CURRENT pose (no jump), then it follows the leader.")
            if confirm("Type GO to enable follower torque: ").strip() != "GO":
                log("aborted by operator; nothing written")
                return 1
            fp, lp = fol.poll()[0], lead.poll()[0]  # re-capture references at the moment of GO (leader may have moved)
            if not (fp["read_ok"] and lp["read_ok"]):
                raise RuntimeError("read failed after confirmation; nothing written")
            f_ref = {n: initial_unwrapped(fp["raw"][n], ranges.get(n) if n != "wrist_roll" else None, margin) for n in names}
            l_ref, l_prev = dict(lp["raw"]), dict(lp["raw"])
            l_u = {n: l_ref[n] for n in names}
            cmd = {n: float(f_ref[n]) for n in names}
            for n in names:  # goal = present first, then torque
                fol.write_reg(ids[n], ADDR_GOAL, int(fp["raw"][n]).to_bytes(2, "little"))
            for n in names:
                fol.write_reg(ids[n], ADDR_TORQUE, b"\x01")
            torque_on = True
        period, t_next, t_end = 1.0 / hz, time.monotonic(), (time.monotonic() + seconds) if seconds else None
        n_cycles, bad = 0, 0
        while not stop["flag"] and (t_end is None or time.monotonic() < t_end):
            fp, lp = fol.poll()[0], lead.poll()[0]
            taps["follower_state"].write(fp)
            taps["leader_state"].write(lp)
            if not (fp["read_ok"] and lp["read_ok"]):
                bad += 1
                if bad > 20:
                    raise RuntimeError("20 consecutive failed reads; stopping")
            else:
                bad = 0
                goals = {}
                for n in names:
                    l_u[n] += wrap_delta(lp["raw"][n], l_prev[n])
                    l_prev[n] = lp["raw"][n]
                    rng = None if n == "wrist_roll" else ranges.get(n)
                    cmd[n], _ = plan_goal(cmd[n], f_ref[n], l_ref[n], l_u[n], sign[n], scale, rng, margin, max_step)
                    goals[ids[n]] = round(cmd[n])
                t = mono_ns()
                if not dry_run:
                    fol.sync_goals(goals)
                taps["commands"].write({"t_start_ns": t, "t_end_ns": t, "values": {n: round(cmd[n]) % TICKS for n in names},
                                        "units": CMD_UNITS, "dry_run": dry_run, "sent": not dry_run})
            n_cycles += 1
            if n_cycles % int(hz) == 0:
                log("leader d: " + " ".join(f"{n[:6]}={l_u[n]-l_ref[n]:+d}" for n in names) +
                    " | cmd d: " + " ".join(f"{round(cmd[n]-f_ref[n]):+d}" for n in names))
            t_next += period
            time.sleep(max(0.0, t_next - time.monotonic()))
        log(f"stopped after {n_cycles} cycles")
        return 0
    finally:
        try:
            if torque_on and on_exit == "release":
                for n in names:
                    fol.write_reg(ids[n], ADDR_TORQUE, b"\x00")
                log("follower torque released (arm is now free - it will sag if unsupported)")
            elif torque_on:
                log("follower torque LEFT ON (holding last goal); no process controls it now")
        except Exception as e:  # report, never hide
            log(f"WARNING: exit handling failed: {e}")
        for b in (fol, lead):
            b.close()
