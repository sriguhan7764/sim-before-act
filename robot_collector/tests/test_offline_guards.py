"""Behavioural guards: the offline/mock path never opens serial ports, cameras or robot sessions.

The autouse ``hardware_guard`` fixture (conftest) spies on device-path opens, serial.Serial, cv2.VideoCapture,
pyrealsense2 and device tools for EVERY test; these tests add targeted checks of the fail-closed gate. SYNTHETIC only.
"""
import sys

import pytest

from collector import hardware_gate
from collector.adapters.factory import make_camera, make_state_source
from collector.cli import main as cli_main
from collector.config import load_config
from collector.hardware_gate import HardwareAccessDenied
from collector.recorder import Recorder
from conftest import HardwareAttempt, JOINTS, wait_until_ok


def real_cfg(tmp):
    return load_config({
        "paths": {"episodes_root": str(tmp / "eps"), "calibrations_root": str(tmp / "cal")},
        "cameras": [{"name": "rs", "backend": "realsense", "serial": "NOT-A-REAL-SERIAL", "width": 640, "height": 480, "fps": 30,
                     "depth": True},
                    {"name": "wrist", "backend": "opencv", "device": "/dev/video99", "role": "wrist"}],
        "robot": {"follower": {"backend": "feetech_readonly", "port": "/dev/ttyUSB99", "joints": [dict(j) for j in JOINTS]},
                  "leader": {"backend": "none", "joints": []}, "commands": {"backend": "none"}},
        "sync": {"reference_camera": "rs"}, "mode": "research"})


def test_guard_itself_catches_device_access(hardware_guard):
    with pytest.raises(HardwareAttempt):
        open("/dev/video0")
    with pytest.raises(HardwareAttempt):
        import cv2

        cv2.VideoCapture(0)
    with pytest.raises(HardwareAttempt):
        import pyrealsense2

        pyrealsense2.pipeline()
    assert len(hardware_guard) == 3
    hardware_guard.clear()                                   # expected here: this test exists to trigger the spies


def test_real_backends_are_refused_before_any_device_is_touched(tmp_path):
    cfg = real_cfg(tmp_path)
    for cam in cfg["cameras"]:
        with pytest.raises(HardwareAccessDenied):
            make_camera(cam).open()
    with pytest.raises(HardwareAccessDenied):
        make_state_source(cfg["robot"]["follower"], "follower").open()
    assert len(hardware_gate.denied_attempts()) == 3        # refused at the gate; the spies in hardware_guard saw nothing


def test_config_file_alone_never_enables_hardware(tmp_path):
    rec = Recorder(real_cfg(tmp_path))
    rec.connect()
    assert not any(rec.sources_ok.values())
    assert all("HardwareAccessDenied" in (s["error"] or "") for s in rec.status.values() if s["kind"] in ("camera", "robot"))
    with pytest.raises(RuntimeError, match="readiness gates"):
        rec.start_episode("t")                              # nothing connected => nothing is recorded, nothing is mocked in
    rec.shutdown()


def test_cli_hardware_flag_needs_confirmation_and_inspect_does_not_enumerate(capsys):
    assert cli_main(["--allow-hardware", "inspect"]) == 4    # pytest stdin is not a terminal: no silent authorisation
    assert not hardware_gate.is_authorised()
    assert "Nothing was opened" in capsys.readouterr().out
    assert cli_main(["inspect"]) == 0
    out = capsys.readouterr().out
    assert "device enumeration not performed" in out


def test_cli_confirmed_by_authorises_only_the_process_and_is_recorded(capsys):
    assert cli_main(["--allow-hardware", "--hardware-confirmed-by", "unit-test", "validate", "--episode", "/nonexistent"]) != 5
    assert hardware_gate.provenance() == {"hardware_authorised": True, "authorised_by": "unit-test"}   # conftest revokes afterwards


def test_bridge_refuses_without_explicit_flags_and_does_not_import_lerobot(capsys):
    from collector import lerobot_bridge

    sys.modules.pop("lerobot", None)
    rc = lerobot_bridge.main(["--follower-port", "/dev/null", "--leader-port", "/dev/null", "--follower-id", "f", "--leader-id", "l",
                              "--follower-cal-dir", "/nonexistent", "--leader-cal-dir", "/nonexistent", "--taps-dir", "/nonexistent"])
    assert rc == 2 and "refusing to run" in capsys.readouterr().out
    assert "lerobot" not in sys.modules


def test_mock_recording_pipeline_opens_no_real_backend(tmp_path, monkeypatch, hardware_guard):
    import yaml

    from collector.adapters import feetech_readonly, opencv_cam, realsense
    from collector.acceptance import run_acceptance
    from collector.config import load_config as lc
    from collector.replay import replay  # noqa: F401  (offline module must import without a device library)

    calls = []
    for cls, name in ((realsense.RealSenseCamera, "open"), (opencv_cam.OpenCVCamera, "open"), (feetech_readonly.FeetechReadOnly, "open")):
        monkeypatch.setattr(cls, name, lambda self, c=cls: calls.append(c.__name__))
    from pathlib import Path

    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/mock.yaml").read_text())
    raw["paths"] = {"episodes_root": str(tmp_path / "eps"), "calibrations_root": str(tmp_path / "cal")}
    cfg = lc(raw)
    rec = Recorder(cfg)
    rec.connect()
    rec.start_workers()
    rec.start_episode("mock")
    wait_until_ok(lambda: rec.status["mock_rgbd"]["count"] >= 5 and rec.status["follower"]["count"] >= 5, "mock samples")
    ep = rec.finish_episode("unlabelled")
    rec.shutdown()
    run_acceptance(ep)
    assert calls == [] and hardware_guard == [] and hardware_gate.denied_attempts() == []
    assert "lerobot" not in sys.modules and "serial" not in sys.modules or hardware_guard == []
