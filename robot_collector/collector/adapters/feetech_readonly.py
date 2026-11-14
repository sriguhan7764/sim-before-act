"""Read-only Feetech STS3215 telemetry adapter (pure pyserial).

Safety by construction: the only instructions ever put on the wire are PING (0x01) and READ (0x02).
There is no code path that writes registers, enables/disables torque, sets modes or homing.
``close()`` only closes the port (it never disables torque, unlike LeRobot ``bus.disconnect()``).

Register layout verified against lerobot 0.6.1 ``motors/feetech/tables.py`` (STS3215):
  56 Present_Position(2, sign-magnitude bit15)  58 Present_Velocity(2, bit15)  60 Present_Load(2, bit10)
  62 Present_Voltage(1)  63 Present_Temperature(1)  65 Status(1)  66 Moving(1)  69 Present_Current(2)
One READ of 15 bytes from address 56 returns all of them. Velocity/load/current/voltage are kept in
raw register units (unit conversion not asserted). Present_Position is the raw register value AFTER the
motor's stored homing offset (encoder ticks, 0..4095), not a normalised value.

IMPORTANT: Feetech buses are half-duplex. Do not open the same port from two processes at once.
The port is opened with exclusive=True so a conflicting second opener fails instead of corrupting traffic.
Use the tap adapter (adapters/tap.py) when an existing teleop process owns the port.
Not hardware-tested.
"""
from __future__ import annotations

import time

from ..util import mono_ns, wall_ns
from .base import StateSource

READ_ADDR, READ_LEN = 56, 15
INSTR_PING, INSTR_READ = 0x01, 0x02
_ALLOWED = {INSTR_PING, INSTR_READ}


def decode_sign_magnitude(v: int, bit: int) -> int:
    mag = v & ((1 << bit) - 1)
    return -mag if (v >> bit) & 1 else mag


def build_packet(motor_id: int, instruction: int, params: bytes = b"") -> bytes:
    if instruction not in _ALLOWED:
        raise PermissionError(f"read-only adapter refuses instruction 0x{instruction:02x}")
    body = bytes([motor_id, len(params) + 2, instruction]) + params
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def parse_status(pkt: bytes, motor_id: int) -> tuple[int, bytes]:
    if len(pkt) < 6 or pkt[:2] != b"\xff\xff":
        raise IOError("bad header")
    if pkt[2] != motor_id:
        raise IOError(f"unexpected id {pkt[2]}")
    if len(pkt) != pkt[3] + 4:
        raise IOError("bad length")
    if (~sum(pkt[2:-1])) & 0xFF != pkt[-1]:
        raise IOError("bad checksum")
    return pkt[4], bytes(pkt[5:-1])


def decode_block(data: bytes) -> dict:
    u16 = lambda o: data[o] | (data[o + 1] << 8)
    return {"raw_position": decode_sign_magnitude(u16(0), 15), "velocity_raw": decode_sign_magnitude(u16(2), 15),
            "load_raw": decode_sign_magnitude(u16(4), 10), "voltage_raw": data[6], "temperature_c": data[7],
            "status": data[9], "moving": data[10], "current_raw": u16(13)}


class FeetechReadOnly(StateSource):
    def __init__(self, cfg: dict, kind: str = "follower", serial_factory=None):
        self.cfg, self.kind = cfg, kind
        self.joints = cfg["joints"]
        self._factory, self.ser = serial_factory, None
        self._period = 1.0 / float(cfg.get("poll_hz", 100.0))
        self._next = 0.0

    def open(self):
        if not self.cfg.get("port"):
            raise RuntimeError(f"{self.kind}: robot port is unset in config; refusing to guess a serial device")
        if not self.joints:
            raise RuntimeError(f"{self.kind}: no joints configured")
        if self._factory:
            self.ser = self._factory(self.cfg["port"], self.cfg["baudrate"])
        else:
            from .. import hardware_gate

            hardware_gate.require(f"serial port open ({self.kind}: {self.cfg['port']})")
            import serial

            self.ser = serial.Serial(self.cfg["port"], self.cfg["baudrate"], timeout=0.02, exclusive=True)
        self.ser.reset_input_buffer()
        for j in self.joints:  # PING only, to verify presence
            self._txrx(j["id"], INSTR_PING)
        self._next = time.monotonic()

    def _txrx(self, mid: int, instr: int, params: bytes = b"") -> bytes:
        self.ser.write(build_packet(mid, instr, params))
        exp = 6 if instr == INSTR_PING else 6 + (params[1] if instr == INSTR_READ else 0)
        pkt = self.ser.read(exp)
        _, data = parse_status(pkt, mid)
        return data

    def read_homing_offset(self, motor_id: int) -> int:
        """READ-only: Homing_Offset register (addr 31, 2 bytes, sign-magnitude bit 11) as stored in the motor."""
        d = self._txrx(motor_id, INSTR_READ, bytes([31, 2]))
        return decode_sign_magnitude(d[0] | (d[1] << 8), 11)

    def close(self):
        if self.ser is not None:
            self.ser.close()  # no torque change, no writes
            self.ser = None

    def poll(self):
        wait = self._next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next + self._period, time.monotonic())
        t0 = mono_ns()
        blocks, errors = {}, []
        for j in self.joints:
            try:
                blocks[j["name"]] = decode_block(self._txrx(j["id"], INSTR_READ, bytes([READ_ADDR, READ_LEN])))
            except Exception as e:
                errors.append(f"{j['name']}: {e}")
        t1 = mono_ns()
        ok = not errors
        get = lambda k: {n: b[k] for n, b in blocks.items()} if ok else None
        return [{"t_start_ns": t0, "t_end_ns": t1, "wall_ns": wall_ns(), "read_ok": ok, "errors": errors,
                 "raw": get("raw_position"), "velocity_raw": get("velocity_raw"), "load_raw": get("load_raw"),
                 "current_raw": get("current_raw"), "temperature_c": get("temperature_c"),
                 "voltage_raw": get("voltage_raw"), "status": get("status"), "moving": get("moving")}]

    def describe(self):
        return {"backend": "feetech_readonly", "is_mock": False, "kind": self.kind, "port": self.cfg["port"],
                "baudrate": self.cfg["baudrate"], "joint_names": [j["name"] for j in self.joints],
                "motor_ids": [j["id"] for j in self.joints],
                "raw": {"representation": "feetech_sts3215_present_position_register",
                        "unit": "encoder_ticks_0_4095_after_motor_homing_offset"},
                "extra_fields": {"velocity_raw": "register units, sign-magnitude decoded", "load_raw": "register units",
                                 "current_raw": "register units", "voltage_raw": "register units (0.1 V typical)",
                                 "temperature_c": "degrees C", "status": "status register", "moving": "moving flag"},
                "write_access": "none (PING/READ only)"}
