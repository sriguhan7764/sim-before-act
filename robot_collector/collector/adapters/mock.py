"""EXPLICIT mock backends for tests/dry runs. Selected only with backend: mock; always flagged is_mock.

Mock data is synthetic and is marked in every manifest (quality flag ``mock_data``).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..negotiation import negotiation_record
from ..util import mono_ns, wall_ns
from .base import CameraFrame, CameraSource, StateSource


class MockCamera(CameraSource):
    is_mock = True

    def __init__(self, name="mock_cam", width=64, height=48, fps=30.0, depth=True, depth_scale=0.001):
        self.name, self.w, self.h, self.fps, self.depth, self.depth_scale = name, width, height, fps, depth, depth_scale
        self._n = 0
        self._t0 = None

    def open(self):
        self._t0 = time.monotonic()

    def close(self):
        pass

    def read(self, timeout_s=1.0):
        period = 1.0 / self.fps
        due = self._t0 + self._n * period
        delay = due - time.monotonic()
        if delay > timeout_s:
            time.sleep(timeout_s)
            return None
        if delay > 0:
            time.sleep(delay)
        n = self._n
        self._n += 1
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[..., 0] = np.linspace(0, 255, self.w, dtype=np.uint8)[None, :]
        img[..., 1] = (n * 7) % 256
        d = nat = None
        if self.depth:
            # values > 255 and a hole of zeros (= invalid) to exercise uint16 fidelity
            d = (1000 + (np.arange(self.h * self.w).reshape(self.h, self.w) % 3000) + n).astype(np.uint16)
            d[:4, :4] = 0
            nat = d
        return CameraFrame(img, mono_ns(), wall_ns(), frame_number=n, device_timestamp_ms=n * period * 1000,
                           timestamp_domain="mock_clock", depth_native=nat, depth_aligned=None if d is None else d.copy(),
                           depth_frame_number=n, depth_device_timestamp_ms=n * period * 1000,
                           depth_timestamp_domain="mock_clock")

    def describe(self):
        f = float(self.w)
        K = [[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]]
        return {"backend": "mock", "is_mock": True, "device": {"name": "MOCK", "serial": "MOCK"},
                "stream_profile": {"width": self.w, "height": self.h, "fps_configured": self.fps,
                                   "color_format": "rgb8", "depth_format": "z16" if self.depth else None},
                "color_intrinsics": {"width": self.w, "height": self.h, "K": K, "dist": [0, 0, 0, 0, 0],
                                     "distortion_model": "none"},
                "depth_intrinsics": ({"width": self.w, "height": self.h, "K": K, "dist": [0, 0, 0, 0, 0],
                                      "distortion_model": "none", "synthetic": True} if self.depth else None),
                "extrinsics_depth_to_color": ({"rotation_row_major": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation_m": [0, 0, 0],
                                               "synthetic": True} if self.depth else None),
                "depth_scale": self.depth_scale if self.depth else None,
                "depth_units": "metres_per_unit" if self.depth else None, "settings": {},
                "negotiation": negotiation_record(
                    {"color": {"width": self.w, "height": self.h, "fps": self.fps}},
                    {"color": {"width": self.w, "height": self.h, "fps": self.fps}}) | {"synthetic": True},
                **({"depth_native": {"width": self.w, "height": self.h, "format": "z16", "synthetic": True,
                                     "geometry": "depth-sensor image", "depth_scale": self.depth_scale},
                    "depth_aligned": {"width": self.w, "height": self.h, "format": "z16", "synthetic": True,
                                      "geometry": "colour image", "depth_scale": self.depth_scale}} if self.depth else {})}


class MockStateSource(StateSource):
    """Synthetic follower/leader. Raw values are fake encoder ticks in 0..4095."""
    is_mock = True

    def __init__(self, joints: list[dict], rate_hz=100.0, phase=0.0, kind="follower"):
        self.joints, self.rate, self.phase, self.kind = joints, rate_hz, phase, kind
        self._last = 0.0

    def open(self):
        self._last = time.monotonic()

    def close(self):
        pass

    def poll(self):
        period = 1.0 / self.rate
        dt = self._last + period - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        self._last = time.monotonic()
        t0 = mono_ns()
        t = time.monotonic()
        raw = {j["name"]: int(2048 + 800 * math.sin(t + k + self.phase)) for k, j in enumerate(self.joints)}
        t1 = mono_ns()
        return [{"t_start_ns": t0, "t_end_ns": t1, "wall_ns": wall_ns(), "read_ok": True, "raw": raw,
                 "velocity_raw": None, "load_raw": None, "current_raw": None, "temperature_c": None,
                 "voltage_raw": None, "status": None, "moving": None}]

    def describe(self):
        return {"backend": "mock", "is_mock": True, "kind": self.kind,
                "joint_names": [j["name"] for j in self.joints], "motor_ids": [j.get("id") for j in self.joints],
                "raw": {"representation": "synthetic_encoder_ticks", "unit": "ticks_0_4095"},
                "extra_fields": []}


class MockCommandSource(StateSource):
    is_mock = True

    def __init__(self, joints: list[dict], rate_hz=30.0):
        self.joints, self.rate = joints, rate_hz
        self._last = 0.0

    def open(self):
        self._last = time.monotonic()

    def close(self):
        pass

    def poll(self):
        dt = self._last + 1 / self.rate - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        self._last = time.monotonic()
        t0 = mono_ns()
        vals = {j["name"]: float(2048 + 700 * math.sin(time.monotonic() + k)) for k, j in enumerate(self.joints)}
        return [{"t_start_ns": t0, "t_end_ns": mono_ns(), "wall_ns": wall_ns(), "values": vals,
                 "units": "synthetic_ticks", "ack": None}]

    def describe(self):
        return {"backend": "mock", "is_mock": True, "kind": "command",
                "units": "synthetic_ticks", "ack_available": False}
