"""Build adapters from config. Mock backends are only created when explicitly requested."""
from __future__ import annotations

from .base import CameraSource, StateSource


def make_camera(c: dict) -> CameraSource:
    b = c["backend"]
    if b == "realsense":
        from .realsense import RealSenseCamera
        return RealSenseCamera(c)
    if b == "opencv":
        from .opencv_cam import OpenCVCamera
        return OpenCVCamera(c)
    if b == "mock":
        from .mock import MockCamera
        return MockCamera(c["name"], c.get("width") or 64, c.get("height") or 48, c.get("fps") or 30.0, c.get("depth", False))
    raise ValueError(f"unknown camera backend '{b}'")


def make_state_source(c: dict, kind: str) -> StateSource | None:
    b = c["backend"]
    if b == "none":
        return None
    if b == "feetech_readonly":
        from .feetech_readonly import FeetechReadOnly
        return FeetechReadOnly(c, kind)
    if b == "jsonl_tap":
        from .tap import TapSource
        return TapSource(c["path"], kind)
    if b == "mock":
        from .mock import MockStateSource
        return MockStateSource(c["joints"], c.get("poll_hz", 100.0), kind=kind, phase=0.0 if kind == "follower" else 0.1)
    raise ValueError(f"unknown {kind} backend '{b}'")


def make_command_source(c: dict, joints: list[dict]) -> StateSource | None:
    b = c["backend"]
    if b == "none":
        return None
    if b == "jsonl_tap":
        from .tap import TapSource
        return TapSource(c["path"], "command")
    if b == "mock":
        from .mock import MockCommandSource
        return MockCommandSource(joints)
    raise ValueError(f"unknown command backend '{b}'")
