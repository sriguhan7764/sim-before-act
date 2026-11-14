"""LeRobot-hosted teleoperation bridge: LeRobot owns the arm buses, the collector reads JSONL taps.

Why: LeRobot's SO-101 leader->follower teleoperation is the proven control path for this setup. This module runs
that same loop (read follower, read leader, ``follower.send_action(leader_measurement)``) and additionally
publishes, per cycle, exactly what was read and sent, so the recorder never opens an arm port:

  follower_state.jsonl : follower Present_Position, RAW ticks (as the SDK returns them, after the motor's
                         Homing_Offset) and the LeRobot-normalised value (arm: degrees about the calibration range
                         midpoint; gripper: 0..100)
  leader_state.jsonl   : same for the leader, normalised with the LEADER calibration
  commands.jsonl       : the Goal_Position target actually sent to the follower (after LeRobot's
                         max_relative_target clip), plus the leader measurement it was derived from and per-joint
                         clip flags.  ``raw_goal`` is computed with the bus' own _unnormalize (what sync_write sends);
                         it is NOT read back from the motor, so no write acknowledgement is claimed.
  session.json         : controller configuration, calibration file hashes and the semantics of every signal.

Command semantics (never interchangeable):
  leader measurement  = leader Present_Position, normalised (this is what ``lerobot_record`` stores as ``action``)
  follower target     = Goal_Position sent after the relative-target clip (``commands.values`` here)
  low-level command   = Goal_Position register ticks (``commands.raw_goal``); no torque/current/PWM commands exist.

The loop itself only reads Present_Position and writes Goal_Position, through LeRobot's own SO101 classes. It never
calibrates (``connect(calibrate=False)``; refuses to run if the motors do not match the calibration file) and sends nothing
before the first complete read of both arms. NOTE: LeRobot's ``connect()`` itself runs its standard ``configure()`` which
DOES write motor registers (acceleration, operating mode, PID, gripper torque/current limits) and ENABLES follower torque;
these are listed in ``LEROBOT_CONNECT_WRITES`` and recorded in session.json, with a read-back of the follower registers.
Before any of that, a read-only startup check verifies the motors match the calibration files and that leader and
follower start in a similar pose (``--max-start-gap``). ``--check-only`` stops after that check (no torque, no writes).
This module does NOT run automatically; hardware use needs the operator's explicit start (docs/PILOT_PROCEDURE.md).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from .adapters.tap import TapWriter
from .joints import LEROBOT_NORMALIZED_UNITS as NORMALIZED_UNITS
from .util import mono_ns, sha256_file, wall_ns

BRIDGE_VERSION = "0.1"

SEMANTICS = {
    "follower_state.raw": "Present_Position register ticks after the motor Homing_Offset (SDK value, unconverted)",
    "follower_state.normalized": NORMALIZED_UNITS,
    "leader_state.raw": "leader Present_Position register ticks after the leader motor Homing_Offset",
    "leader_state.normalized": "as follower_state.normalized but with the LEADER calibration file",
    "commands.values": "follower Goal_Position target actually sent via LeRobot send_action, AFTER the "
                       "max_relative_target clip, in follower-normalised units",
    "commands.raw_goal": "Goal_Position ticks computed with bus._unnormalize (what sync_write puts on the wire); "
                         "not read back from the motor",
    "commands.leader_measurement_normalized": "leader measurement that the target was derived from (leader "
                                              "calibration units); lerobot_record stores this as dataset 'action'",
    "commands.clipped": "true when the sent target differs from the leader measurement (relative-target clip)",
    "commands.ack": "none: sync_write has no acknowledgement; no status packet is read",
    "commands.submission_status": "SUBMITTED for writing only. Neither physical execution nor the Goal_Position register value "
                                  "is confirmed here; the measured follower_state is the only evidence of motion",
    "signal_kind": "follower position target (derived 1:1 from the leader measurement, optionally clipped); "
                   "NOT a torque/current/PWM command",
}

# What LeRobot's SO101Follower.connect(calibrate=False) -> configure() writes before the loop starts (lerobot 0.6.1, verified in source).
LEROBOT_CONNECT_WRITES = {
    "configure_motors (all 6)": "Return_Delay_Time=0, Maximum_Acceleration=254, Acceleration=254, Phase bit4 cleared",
    "configure (all 6)": "Operating_Mode=POSITION, P_Coefficient=16, I_Coefficient=0, D_Coefficient=32 (SO101FollowerConfig defaults)",
    "configure (gripper only)": "Max_Torque_Limit=500, Protection_Current=250, Overload_Torque=25",
    "torque": "Torque_Enable disabled during configure, then ENABLED (follower holds position / tracks the leader afterwards)",
    "not written": "Homing_Offset, Min/Max_Position_Limit, calibration (calibrate=False; motors must already match the file)",
}


class StartupRefused(RuntimeError):
    pass


def startup_gap(norm_f: dict[str, float], norm_l: dict[str, float]) -> dict[str, float]:
    """|leader - follower| per joint in LeRobot-normalised units (degrees / gripper %), from reads taken BEFORE torque is enabled."""
    return {n: abs(float(norm_l[n]) - float(norm_f[n])) for n in norm_f}


def startup_check(fol, lea, max_gap: float) -> dict:
    """Read-only: confirm the motors still match the calibration files and that both arms start in a similar pose."""
    chk = {"follower_matches_calibration": bool(fol.is_calibrated), "leader_matches_calibration": bool(lea.is_calibrated)}
    if not (chk["follower_matches_calibration"] and chk["leader_matches_calibration"]):
        raise StartupRefused("motor homing offsets/ranges do not match the calibration files "
                             f"(follower ok={chk['follower_matches_calibration']}, leader ok={chk['leader_matches_calibration']}). "
                             "The bridge never recalibrates; resolve this before any actuation.")
    _, nf = read_raw_and_normalized(fol.bus)
    _, nl = read_raw_and_normalized(lea.bus)
    gap = startup_gap(nf, nl)
    chk.update({"start_gap_normalized_units": gap, "max_start_gap_allowed": max_gap,
                "follower_pose_before_torque": nf, "leader_pose_before_torque": nl})
    worst = max(gap, key=gap.get)
    if gap[worst] > max_gap:
        raise StartupRefused(f"leader and follower start poses differ by {gap[worst]:.1f} (> {max_gap}) at '{worst}'; "
                             "enabling torque could make the follower move abruptly. Pose the arms similarly and retry.")
    return chk


def _by_name(bus, ids_values: dict[int, float]) -> dict[str, float]:
    id_to_name = {m.id: n for n, m in bus.motors.items()}
    return {id_to_name[i]: v for i, v in ids_values.items()}


def read_raw_and_normalized(bus) -> tuple[dict[str, int], dict[str, float]]:
    """One raw bus read, normalised with the bus' own calibration (identical to what get_observation computes)."""
    raw = bus.sync_read("Present_Position", normalize=False)
    ids = {bus.motors[n].id: int(v) for n, v in raw.items()}
    return {n: int(v) for n, v in raw.items()}, _by_name(bus, bus._normalize(ids))


def goal_ticks(bus, sent: dict[str, float]) -> dict[str, int]:
    ids = {bus.motors[n].id: float(v) for n, v in sent.items()}
    return {n: int(v) for n, v in _by_name(bus, bus._unnormalize(ids)).items()}


class Bridge:
    """One teleop cycle per ``step()``. ``follower``/``leader`` are LeRobot SO101 objects (or test doubles)."""

    def __init__(self, follower, leader, taps_dir: str | Path, fps: float = 30.0, session: dict | None = None,
                 max_consecutive_errors: int = 5):
        self.follower, self.leader, self.fps = follower, leader, float(fps)
        self.dir = Path(taps_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t_f = TapWriter(str(self.dir / "follower_state.jsonl"))
        self.t_l = TapWriter(str(self.dir / "leader_state.jsonl"))
        self.t_c = TapWriter(str(self.dir / "commands.jsonl"))
        self.session = session or {}
        self.max_err = max_consecutive_errors
        self.cycles = self.errors = self.consecutive_errors = self.overruns = 0
        self.max_cycle_s = 0.0

    def write_session(self, extra: dict | None = None) -> None:
        doc = {"bridge_version": BRIDGE_VERSION, "fps": self.fps, "semantics": SEMANTICS, **self.session, **(extra or {})}
        tmp = self.dir / "session.json.tmp"
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
        os.replace(tmp, self.dir / "session.json")

    def step(self) -> bool:
        """Read both arms, send the (clipped) target. Returns False when this cycle was skipped (read error)."""
        try:
            tf0 = mono_ns()
            raw_f, norm_f = read_raw_and_normalized(self.follower.bus)
            tf1 = mono_ns()
            tl0 = mono_ns()
            raw_l, norm_l = read_raw_and_normalized(self.leader.bus)
            tl1 = mono_ns()
        except Exception as e:  # nothing is sent on a failed read: never act on stale or partial data
            self.errors += 1
            self.consecutive_errors += 1
            t = mono_ns()
            self.t_f.write({"t_start_ns": t, "t_end_ns": t, "read_ok": False, "error": f"{type(e).__name__}: {e}"})
            if self.consecutive_errors >= self.max_err:
                raise RuntimeError(f"{self.consecutive_errors} consecutive read failures; stopping bridge") from e
            return False
        self.consecutive_errors = 0
        self.t_f.write({"t_start_ns": tf0, "t_end_ns": tf1, "raw": raw_f, "normalized": norm_f})
        self.t_l.write({"t_start_ns": tl0, "t_end_ns": tl1, "raw": raw_l, "normalized": norm_l})
        action = {f"{n}.pos": v for n, v in norm_l.items()}
        tc0 = mono_ns()
        sent_full = self.follower.send_action(action)
        tc1 = mono_ns()
        sent = {k.removesuffix(".pos"): float(v) for k, v in sent_full.items()}
        self.t_c.write({"t_start_ns": tc0, "t_end_ns": tc1, "values": sent, "raw_goal": goal_ticks(self.follower.bus, sent),
                        "units": NORMALIZED_UNITS, "leader_measurement_normalized": norm_l,
                        "leader_measurement_t_ns": (tl0 + tl1) // 2,
                        "clipped": {n: not math.isclose(sent[n], norm_l[n], abs_tol=1e-6) for n in sent},
                        "ack": None, "submission_status": "submitted_not_confirmed"})
        return True

    def run(self, seconds: float | None = None, stop=lambda: False, sleep=time.sleep, clock=time.perf_counter) -> dict:
        self.write_session({"started_wall_ns": wall_ns()})
        period, t_end = 1.0 / self.fps, (None if seconds is None else clock() + seconds)
        while not stop() and (t_end is None or clock() < t_end):
            t0 = clock()
            self.step()
            self.cycles += 1
            dt = clock() - t0
            self.max_cycle_s = max(self.max_cycle_s, dt)
            if dt > period:
                self.overruns += 1
            else:
                sleep(period - dt)
        stats = {"cycles": self.cycles, "read_errors": self.errors, "overruns": self.overruns,
                 "max_cycle_s": self.max_cycle_s, "ended_wall_ns": wall_ns()}
        self.write_session({"stats": stats})
        return stats


def _cal_info(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def main(argv=None) -> int:  # pragma: no cover - needs hardware and the LeRobot environment
    p = argparse.ArgumentParser(prog="collector.lerobot_bridge", description=__doc__.split("\n\n")[0])
    p.add_argument("--follower-port", required=True)
    p.add_argument("--leader-port", required=True)
    p.add_argument("--follower-id", required=True, help="calibration file stem, e.g. g_c_follower_calibration")
    p.add_argument("--leader-id", required=True)
    p.add_argument("--follower-cal-dir", required=True)
    p.add_argument("--leader-cal-dir", required=True)
    p.add_argument("--taps-dir", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-relative-target", type=float, default=10.0)
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--max-start-gap", type=float, default=15.0,
                   help="refuse to enable torque if any joint of leader and follower differ by more than this (degrees / gripper %%)")
    p.add_argument("--check-only", action="store_true",
                   help="READ-ONLY: verify motors match the calibration files and compare start poses, then exit. "
                        "No torque, no register writes, no motion")
    p.add_argument("--i-authorise-motion", action="store_true",
                   help="required for the real run: this process enables follower torque and sends Goal_Position through LeRobot")
    a = p.parse_args(argv)
    if not a.check_only and not a.i_authorise_motion:
        print("refusing to run: pass --i-authorise-motion (the follower will follow the leader), or --check-only for a read-only check")
        return 2
    import importlib.metadata as md

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    fol = SO101Follower(SO101FollowerConfig(port=a.follower_port, id=a.follower_id, calibration_dir=Path(a.follower_cal_dir),
                                            max_relative_target=a.max_relative_target, use_degrees=True))
    lea = SO101Leader(SO101LeaderConfig(port=a.leader_port, id=a.leader_id, calibration_dir=Path(a.leader_cal_dir),
                                        use_degrees=True))
    fol.bus.connect()
    lea.bus.connect()
    try:
        chk = startup_check(fol, lea, a.max_start_gap)   # reads only; torque is still off
    except StartupRefused as e:
        print("REFUSING TO START (no torque was enabled, nothing was written):", e)
        return 3
    finally:
        fol.bus.disconnect(False)
        lea.bus.disconnect(False)
    print("startup check OK: motors match both calibration files; start gap (normalised units):",
          {k: round(v, 1) for k, v in chk["start_gap_normalized_units"].items()})
    if a.check_only:
        print("--check-only: no torque enabled, no register written, nothing moved.")
        return 0
    print("\n" + "!" * 72 + "\n!!! STARTING THE BRIDGE ENABLES FOLLOWER TORQUE AND THE FOLLOWER WILL TRACK THE LEADER !!!\n"
          "!!! Clear the workspace; keep a hand on the follower power supply; never push toward a mechanical stop. !!!\n"
          + "!" * 72)
    if not sys.stdin.isatty() or input("Type ENABLE TORQUE to continue: ").strip() != "ENABLE TORQUE":
        print("not confirmed; nothing was written")
        return 4
    fol.connect(calibrate=False)
    lea.connect(calibrate=False)
    readback = {}
    for reg in ("Operating_Mode", "P_Coefficient", "I_Coefficient", "D_Coefficient", "Max_Torque_Limit", "Protection_Current",
                "Overload_Torque", "Torque_Enable", "Acceleration", "Return_Delay_Time"):
        try:
            readback[reg] = fol.bus.sync_read(reg, normalize=False)
        except Exception as e:
            readback[reg] = f"read failed: {type(e).__name__}: {e}"
    session = {
        "lerobot_version": md.version("lerobot") if _has(md, "lerobot") else "source tree via PYTHONPATH",
        "follower": {"port": a.follower_port, "calibration": _cal_info(Path(a.follower_cal_dir) / f"{a.follower_id}.json"),
                     "max_relative_target": a.max_relative_target,
                     "norm_modes": {n: str(m.norm_mode) for n, m in fol.bus.motors.items()}},
        "leader": {"port": a.leader_port, "calibration": _cal_info(Path(a.leader_cal_dir) / f"{a.leader_id}.json"),
                   "norm_modes": {n: str(m.norm_mode) for n, m in lea.bus.motors.items()}},
        "startup_check": chk,
        "lerobot_connect_writes": LEROBOT_CONNECT_WRITES,
        "follower_registers_read_back_after_connect": readback,
        "control_loop": "leader Present_Position (normalised) -> follower.send_action -> Goal_Position (relative-target clip)",
        "disconnect_behaviour": "LeRobot default: follower torque is DISABLED on disconnect (arm goes limp)",
    }
    br = Bridge(fol, lea, a.taps_dir, a.fps, session)
    try:
        stats = br.run(a.seconds)
        print("bridge stopped:", json.dumps(stats))
    except KeyboardInterrupt:
        print("bridge interrupted")
    finally:
        fol.disconnect()
        lea.disconnect()
    return 0


def _has(md, name: str) -> bool:
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    sys.exit(main())
