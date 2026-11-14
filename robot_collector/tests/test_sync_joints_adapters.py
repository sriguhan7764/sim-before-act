"""Association, joint semantics, read-only Feetech adapter (fake serial; synthetic bytes)."""
import math

import pytest

from collector.adapters import feetech_readonly as fh
from collector.joints import AngleMapping
from collector.sync import match_nearest
from conftest import CAL, JOINTS, raw_for

MS = 1_000_000


def test_nearest_match_and_residual_sign():
    times = [0, 10 * MS, 20 * MS]
    i, res, st = match_nearest(11 * MS, times, 5 * MS)
    assert (i, res, st) == (1, -1 * MS, "matched")


def test_stale_sample_rejected_not_reused():
    i, res, st = match_nearest(100 * MS, [0, 10 * MS], 20 * MS)
    assert i is None and st == "unmatched_beyond_tolerance" and res == -90 * MS
    assert match_nearest(0, [], 20 * MS)[2] == "no_samples"


def test_invalid_samples_skipped():
    i, _, st = match_nearest(10 * MS, [9 * MS, 11 * MS], 5 * MS, valid=[False, True])
    assert i == 1 and st == "matched"
    assert match_nearest(10 * MS, [9 * MS], 5 * MS, valid=[False])[2] == "no_valid_candidate"


def test_command_direction_semantics():
    t = [0, 50 * MS]
    assert match_nearest(40 * MS, t, 100 * MS, direction="before")[0] == 0
    assert match_nearest(40 * MS, t, 100 * MS, direction="after")[0] == 1


def test_joint_order_is_stable_and_gripper_is_not_radians():
    m = AngleMapping(JOINTS, CAL)
    out = m.convert(raw_for(0))
    assert out["revolute_names"] == [j["name"] for j in JOINTS if j["type"] == "revolute"]
    assert "gripper" not in out["revolute_names"] and len(out["joint_angles_rad"]) == 5
    assert out["gripper"]["gripper"]["unit"] == "normalized_0_100"


def test_raw_ticks_vs_calibrated_angle():
    m = AngleMapping(JOINTS, CAL)
    raw = {j["name"]: 2000 for j in JOINTS}          # midpoint of range 1000..3000
    assert m.convert(raw)["joint_angles_rad"] == [0.0] * 5
    raw["shoulder_pan"] = 2000 + 1024                 # quarter turn of ticks
    a = m.convert(raw)["joint_angles_rad"][0]
    assert a == pytest.approx(1024 * 2 * math.pi / 4095)
    assert a != 1024                                  # radians are not raw ticks
    assert m.convert({**raw, "gripper": 3000})["gripper"]["gripper"]["value"] == 100.0


def test_calibration_must_match_motor_ids():
    bad = {**CAL, "elbow_flex": {**CAL["elbow_flex"], "id": 9}}
    with pytest.raises(ValueError):
        AngleMapping(JOINTS, bad)
    with pytest.raises(ValueError):
        AngleMapping(JOINTS, {k: v for k, v in CAL.items() if k != "gripper"})


class FakeSerial:
    def __init__(self, port, baud):
        self.sent, self.buf = [], b""

    def reset_input_buffer(self): pass
    def close(self): self.closed = True

    def write(self, pkt):
        self.sent.append(bytes(pkt))
        mid, instr = pkt[2], pkt[4]
        if instr == fh.INSTR_PING:
            self.buf = self._status(mid, b"")
        else:
            n = pkt[6]
            data = bytearray(n)
            pos = 2048 | 0  # raw ticks
            data[0], data[1] = pos & 0xFF, pos >> 8
            data[7] = 31  # temperature
            self.buf = self._status(mid, bytes(data))

    @staticmethod
    def _status(mid, params):
        body = bytes([mid, len(params) + 2, 0]) + params
        return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])

    def read(self, n):
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


def test_feetech_adapter_is_read_only_and_decodes_raw_ticks():
    cfg = {"port": "/dev/fake", "baudrate": 1_000_000, "joints": JOINTS, "poll_hz": 1000}
    ser = FakeSerial(None, None)
    a = fh.FeetechReadOnly(cfg, serial_factory=lambda p, b: ser)
    a.open()
    rec = a.poll()[0]
    assert rec["read_ok"] and rec["raw"]["shoulder_pan"] == 2048 and rec["temperature_c"]["gripper"] == 31
    assert {p[4] for p in ser.sent} <= {fh.INSTR_PING, fh.INSTR_READ}
    a.close()
    assert ser.closed


def test_feetech_refuses_write_instructions():
    for instr in (0x03, 0x04, 0x83):  # WRITE, REG_WRITE, SYNC_WRITE
        with pytest.raises(PermissionError):
            fh.build_packet(1, instr, b"\x28\x00")


def test_feetech_unset_port_is_an_error_not_a_guess():
    a = fh.FeetechReadOnly({"port": None, "baudrate": 1, "joints": JOINTS})
    with pytest.raises(RuntimeError, match="unset"):
        a.open()


def test_sign_magnitude():
    assert fh.decode_sign_magnitude(0x8005, 15) == -5 and fh.decode_sign_magnitude(5, 15) == 5


def test_angle_mapping_is_continuous_across_encoder_wrap():
    """SYNTHETIC: a joint whose travel crosses the 0/4095 wrap (range given unwrapped as -300..500)."""
    import math
    from collector.joints import AngleMapping
    m = AngleMapping([{"name": "j", "id": 1, "type": "revolute"}], {"j": {"id": 1, "range_min": -300, "range_max": 500}})
    up, down = m.convert({"j": 100})["joint_angles_rad"][0], m.convert({"j": 4096 - 150})["joint_angles_rad"][0]
    assert abs(up) < 1e-9  # midpoint
    assert abs(down - (-250) * 2 * math.pi / 4095) < 1e-9  # -150 raw == 3946 wraps to -250 ticks from mid
