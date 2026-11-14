"""Adapter interfaces. Every backend implements these; recorder code depends only on them."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraFrame:
    """One camera acquisition. ``color_rgb`` is HxWx3 uint8 in RGB channel order.

    depth_native: HxW uint16 in the depth sensor's own geometry, raw sensor units (see depth_scale).
    depth_aligned: HxW uint16 depth re-projected into the colour image geometry.
    0 in either depth image means "no valid measurement" (RealSense convention) - never distance zero.
    """
    color_rgb: np.ndarray | None
    host_mono_ns: int
    host_wall_ns: int
    frame_number: int | None = None
    device_timestamp_ms: float | None = None
    timestamp_domain: str | None = None  # e.g. "hardware_clock", "system_time", "global_time", None
    depth_native: np.ndarray | None = None
    depth_aligned: np.ndarray | None = None
    depth_frame_number: int | None = None
    depth_device_timestamp_ms: float | None = None
    depth_timestamp_domain: str | None = None
    extras: dict = field(default_factory=dict)


class CameraSource(abc.ABC):
    name: str
    is_mock: bool = False

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def read(self, timeout_s: float = 1.0) -> CameraFrame | None:
        """Block up to timeout_s; return None on timeout (not an error)."""

    @abc.abstractmethod
    def describe(self) -> dict:
        """Device identity, stream profile, intrinsics/distortion, extrinsics, depth scale, settings."""


class StateSource(abc.ABC):
    """Read-only source of robot records (follower state, leader state or command log).

    ``poll`` returns zero or more new records; each is a dict with at least
    ``t_start_ns``/``t_end_ns`` (host monotonic) and ``wall_ns``. Implementations must never move
    the robot, change torque, or write to the motor bus.
    """
    is_mock: bool = False

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release the port/file only; must not alter robot control state."""

    @abc.abstractmethod
    def poll(self) -> list[dict]: ...

    @abc.abstractmethod
    def describe(self) -> dict: ...
