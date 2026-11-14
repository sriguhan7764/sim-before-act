"""SYNTHETIC test fixtures. Everything here is fabricated for tests only and never used by real recording."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector.adapters.base import CameraFrame  # noqa: E402
from collector.joints import AngleMapping  # noqa: E402
from collector.storage import EpisodeWriter  # noqa: E402
from collector.util import write_json_atomic  # noqa: E402

JOINTS = [
    {"name": "shoulder_pan", "id": 1, "type": "revolute"},
    {"name": "shoulder_lift", "id": 2, "type": "revolute"},
    {"name": "elbow_flex", "id": 3, "type": "revolute"},
    {"name": "wrist_flex", "id": 4, "type": "revolute"},
    {"name": "wrist_roll", "id": 5, "type": "revolute"},
    {"name": "gripper", "id": 6, "type": "gripper"},
]
CAL = {j["name"]: {"id": j["id"], "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000}
       for j in JOINTS}
REC_CFG = {"queue_max_items": 1000, "overflow_policy": "flag_and_drop", "flush_interval_s": 0.2,
           "min_free_disk_gb": 0.0, "png_compression": 1}
SYNC_CFG = {"reference_camera": "cam", "max_state_tolerance_ms": 20.0, "max_command_tolerance_ms": 50.0,
            "max_camera_tolerance_ms": 20.0, "interpolation": "none"}
T0 = 10_000_000_000


def depth_image(i):
    d = (np.arange(24 * 32).reshape(24, 32) * 3 + i).astype(np.uint16)
    d[0, :4] = 0       # invalid pixels
    d[1, 0] = 65535    # max uint16
    d[1, 1] = 300      # > uint8
    return d


def frame(i, t=None):
    img = np.full((24, 32, 3), i % 255, np.uint8)
    img[..., 2] = 7
    d = depth_image(i)
    return CameraFrame(img, t if t is not None else T0 + i * 33_000_000, 0, frame_number=100 + i,
                       device_timestamp_ms=5000.0 + i * 33.3, timestamp_domain="hardware_clock",
                       depth_native=d, depth_aligned=d[::-1].copy(), depth_frame_number=100 + i,
                       depth_device_timestamp_ms=5000.0 + i * 33.3, depth_timestamp_domain="hardware_clock")


def frame_rec(i, t=None):
    f = frame(i, t)
    return f, {"recv_index": i, "frame_number": f.frame_number, "device_timestamp_ms": f.device_timestamp_ms,
               "timestamp_domain": f.timestamp_domain, "host_mono_ns": f.host_mono_ns, "host_wall_ns": 0,
               "depth_frame_number": f.depth_frame_number, "depth_device_timestamp_ms": 1.0,
               "depth_timestamp_domain": "hardware_clock", "extras": {}}


def raw_for(k):
    return {j["name"]: 1500 + 10 * k + 7 * n for n, j in enumerate(JOINTS)}


def new_writer(tmp, with_cal=True, rec_cfg=None, mock=False, meta_extra=None):
    meta = {"task_name": "synthetic", "instruction": "test", "reference_camera": "cam",
            "streams": {"cam": {"role": "primary", "depth": True}},
            "calibration": {"camera_to_mat": None,
                            "robot_motor_calibration_sha256": "abc" if with_cal else None, "robot_base_to_mat": None},
            "quality_flags": {"mock_data": mock, "synthetic_test_fixture": True}}
    for k, v in (meta_extra or {}).items():
        meta[k] = {**meta[k], **v} if k == "quality_flags" else v
    w = EpisodeWriter(tmp, meta, rec_cfg or REC_CFG, SYNC_CFG)
    (w.dir / "streams" / "cam").mkdir(exist_ok=True)
    write_json_atomic(w.dir / "streams" / "cam" / "camera.json",
                      {"depth_scale": 0.001, "depth_units": "metres_per_unit",
                       "color_intrinsics": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5}})
    (w.dir / "calibration").mkdir()
    write_json_atomic(w.dir / "calibration" / "joints.json", {"follower_joints": JOINTS})
    if with_cal:
        write_json_atomic(w.dir / "calibration" / "robot_motor_calibration.json",
                          AngleMapping(JOINTS, CAL, "synthetic.json", "deadbeef").describe())
    return w


def fill(w, n_frames=10, n_states=120, n_cmds=30, stream_cmd=True):
    for i in range(n_frames):
        w.submit("cam", "frame", frame_rec(i))
    for k in range(n_states):
        t = T0 + k * 3_000_000
        w.submit("follower_state", "telemetry", {"seq": k, "t_start_ns": t - 100_000, "t_end_ns": t + 100_000,
                                                 "wall_ns": 0, "read_ok": True, "raw": raw_for(k)})
    if stream_cmd:
        for k in range(n_cmds):
            t = T0 + k * 10_000_000
            w.submit("commands", "telemetry", {"seq": k, "t_start_ns": t, "t_end_ns": t, "wall_ns": 0,
                                               "values": {"shoulder_pan": 111.0 + k}, "units": "synthetic"})


def wait_until_ok(cond, what, timeout=10.0):
    """Condition wait (no fixed sleep): poll until cond() is true or fail with a clear message."""
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for: {what}")


HARDWARE_PATH_PREFIXES = ("/dev/ttyUSB", "/dev/ttyACM", "/dev/serial/", "/dev/video", "/dev/bus/usb", "/dev/media")
HARDWARE_TOOLS = ("v4l2-ctl", "rs-enumerate-devices", "lsusb", "fuser", "udevadm")


class HardwareAttempt(AssertionError):
    """Raised by the guard when code under test touches a device path, serial port, camera or enumeration tool."""


@pytest.fixture(autouse=True)
def hardware_guard(monkeypatch):
    """Behavioural guard for EVERY test: nothing may open a serial port, camera or device node, or enumerate hardware.

    Spies sit at the lowest levels (open/os.open on device paths, serial.Serial, cv2.VideoCapture, pyrealsense2, device
    tools via subprocess). An attempt is recorded and raised; because code under test may swallow exceptions, the
    recorded list is ALSO asserted empty at teardown. The process-wide hardware authorisation is reset before and after.
    """
    import builtins
    import io
    import subprocess
    import sys
    import types

    from collector import hardware_gate

    hardware_gate.revoke()
    hardware_gate._state["attempts_denied"].clear()
    attempts: list[str] = []

    def deny(what):
        attempts.append(what)
        raise HardwareAttempt(f"hardware access attempted by a test: {what}")

    def is_dev(p):
        return isinstance(p, (str, bytes, os.PathLike)) and os.fsdecode(p).startswith(HARDWARE_PATH_PREFIXES)

    real_open, real_osopen, real_run = builtins.open, os.open, subprocess.run

    def g_open(file, *a, **k):
        if is_dev(file):
            deny(f"open({file!r})")
        return real_open(file, *a, **k)

    def g_osopen(path, *a, **k):
        if is_dev(path):
            deny(f"os.open({path!r})")
        return real_osopen(path, *a, **k)

    def g_run(args, *a, **k):
        if args and os.path.basename(str(args[0] if isinstance(args, (list, tuple)) else args).split()[0]) in HARDWARE_TOOLS:
            deny(f"subprocess {args!r}")
        return real_run(args, *a, **k)

    monkeypatch.setattr(builtins, "open", g_open)
    monkeypatch.setattr(io, "open", g_open)
    monkeypatch.setattr(os, "open", g_osopen)
    monkeypatch.setattr(subprocess, "run", g_run)
    try:
        import serial

        monkeypatch.setattr(serial, "Serial", lambda *a, **k: deny("serial.Serial"))
    except ImportError:
        pass
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: deny("cv2.VideoCapture"))
    rs = types.ModuleType("pyrealsense2")
    rs.__version__ = "GUARD (no real pyrealsense2 is reachable from tests)"
    rs.__getattr__ = lambda name: (_ for _ in ()).throw(AttributeError(name)) if name.startswith("__") else deny(f"pyrealsense2.{name}")
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    yield attempts
    monkeypatch.undo()
    denied = hardware_gate.denied_attempts()
    hardware_gate.revoke()
    hardware_gate._state["attempts_denied"].clear()
    assert not attempts, f"test touched hardware: {attempts}"
    # attempts at the gate are legitimate only in tests that exercise the refusal; they must not leak real access
    assert all(isinstance(x, str) for x in denied)


@pytest.fixture
def hardware_authorised_for_fakes():
    """For tests that drive a real adapter class against a FAKE device object: authorise the gate, nothing else."""
    from collector import hardware_gate

    hardware_gate.authorise("pytest fake-device test")
    yield
    hardware_gate.revoke()


@pytest.fixture
def episode(tmp_path):
    w = new_writer(tmp_path / "eps")
    fill(w)
    return w.finalize("success")
