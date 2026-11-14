"""`collector inspect`: discover what exists on this machine. Never opens serial ports or moves anything."""
from __future__ import annotations

import glob
import importlib.metadata as md
import os
import platform
import sys
from pathlib import Path

from . import hardware_gate
from .util import repo_commit, sha256_file


def _pkg(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def inspect_environment(probe_realsense: bool = True) -> dict:
    rep: dict = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
                 "conda_env": os.environ.get("CONDA_DEFAULT_ENV"), "collector_repo_commit": repo_commit()}
    rep["packages"] = {p: _pkg(p) for p in ("numpy", "opencv-python", "opencv-python-headless", "pyrealsense2",
                                            "pyserial", "PyYAML", "lerobot", "feetech-servo-sdk")}
    try:
        import cv2
        rep["packages"]["cv2"] = cv2.__version__
        rep["aruco_available"] = hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")
    except Exception as e:
        rep["aruco_available"] = f"cv2 import failed: {e}"
    try:
        import tkinter  # noqa: F401
        rep["tkinter"] = True
    except Exception:
        rep["tkinter"] = False
    rep["display"] = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    rep["video_devices"] = sorted(glob.glob("/dev/video*"))
    rep["serial_devices"] = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    rep["serial_by_id"] = sorted(glob.glob("/dev/serial/by-id/*"))
    rep["note_devices"] = "listing only; ports are never opened by inspect. Device identity is unknown until you configure it."
    rs: dict = {"importable": False, "devices": []}
    if probe_realsense and not hardware_gate.is_authorised():
        rs["skipped"] = "device enumeration not performed: hardware not authorised in this process (--allow-hardware)"
    elif probe_realsense:
        try:
            import pyrealsense2 as r

            rs["importable"] = True
            for d in r.context().query_devices():
                rs["devices"].append({"name": d.get_info(r.camera_info.name),
                                      "serial": d.get_info(r.camera_info.serial_number)})
        except Exception as e:
            rs["error"] = f"{type(e).__name__}: {e}"
    rep["realsense"] = rs
    cal_root = Path.home() / ".cache/huggingface/lerobot/calibration"
    cals = []
    if cal_root.exists():
        for p in sorted(cal_root.rglob("*.json")):
            cals.append({"path": str(p), "sha256": sha256_file(p)[:16]})
    rep["existing_lerobot_calibration_files"] = cals
    rep["existing_lerobot_calibration_note"] = ("read-only listing; which file belongs to the follower is NOT inferred. "
                                                "Set robot.follower.calibration_file explicitly.")
    rep["known_robot_sdk"] = {"family": "SO-101 (Feetech STS3215, motor ids 1-6) via LeRobot 0.6.1 SOFollower; "
                                         "source: lerobot/robots/so_follower/so_follower.py",
                              "recorder_access": "feetech_readonly adapter (PING/READ only) or jsonl_tap"}
    return rep
