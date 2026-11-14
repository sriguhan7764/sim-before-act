"""Intel RealSense (pyrealsense2) camera adapter.

NOT validated on hardware (no device was connected when written). Uses documented pyrealsense2 calls:
rs.pipeline / rs.config.enable_device / enable_stream / rs.align / frame.get_frame_number /
frame.get_timestamp / frame.get_frame_timestamp_domain / profile.get_intrinsics /
get_extrinsics_to / depth_sensor.get_depth_scale.
Colour is requested as rgb8, depth as z16 (uint16). Depth 0 = invalid.
"""
from __future__ import annotations

import numpy as np

from ..negotiation import ProfileMismatch, negotiation_record
from ..util import mono_ns, wall_ns
from .base import CameraFrame, CameraSource


def _intr(i) -> dict:
    return {"width": i.width, "height": i.height,
            "K": [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]],
            "dist": list(i.coeffs), "distortion_model": str(i.model)}


class RealSenseCamera(CameraSource):
    def __init__(self, cfg: dict):
        self.cfg, self.name = cfg, cfg["name"]
        self.pipe = self.profile = self.align = None
        self._desc: dict = {}

    def open(self):
        from .. import hardware_gate

        hardware_gate.require(f"RealSense camera open ({self.name})")
        import pyrealsense2 as rs

        c = self.cfg
        rsc = rs.config()
        if c.get("serial"):
            rsc.enable_device(str(c["serial"]))
        w, h, fps = c.get("width") or 0, c.get("height") or 0, c.get("fps") or 0
        rsc.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        self._requested = {"color": {"width": w, "height": h, "fps": fps, "format": "rgb8"}}
        if c.get("depth"):
            rsc.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
            self._requested["depth"] = {"width": w, "height": h, "fps": fps, "format": "z16"}
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start(rsc)
        self.align = rs.align(rs.stream.color) if c.get("depth") and c.get("align_depth_to_color", True) else None
        self._desc = self._build_desc(rs)
        if not self._desc["negotiation"]["matches_request"]:
            self.close()
            raise ProfileMismatch(f"camera '{self.name}' negotiated a different profile than requested: "
                                  f"{self._desc['negotiation']['mismatches']}")

    def _actual(self, rs, stream) -> dict:
        p = self.profile.get_stream(stream).as_video_stream_profile()
        return {"width": p.width(), "height": p.height(), "fps": p.fps(), "format": str(p.format()).split(".")[-1]}

    def _build_desc(self, rs) -> dict:
        dev = self.profile.get_device()
        info = {k: dev.get_info(getattr(rs.camera_info, k)) for k in
                ("name", "serial_number", "firmware_version", "usb_type_descriptor", "physical_port", "product_id")
                if dev.supports(getattr(rs.camera_info, k))}
        cs = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        actual = {"color": self._actual(rs, rs.stream.color)}
        d = {"backend": "realsense", "is_mock": False, "device": info,
             "sdk": {"pyrealsense2": getattr(rs, "__version__", None)},
             "stream_profile": {"width": cs.width(), "height": cs.height(), "fps_configured": cs.fps(),
                                "color_format": "rgb8", "depth_format": None},
             "color_intrinsics": _intr(cs.get_intrinsics()), "depth_intrinsics": None,
             "extrinsics_depth_to_color": None, "depth_scale": None, "depth_units": None, "settings": {}}
        if self.cfg.get("depth"):
            ds = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            actual["depth"] = self._actual(rs, rs.stream.depth)
            e = ds.get_extrinsics_to(cs)
            d["depth_intrinsics"] = _intr(ds.get_intrinsics())
            d["extrinsics_depth_to_color"] = {"rotation_row_major": list(e.rotation), "translation_m": list(e.translation),
                                              "note": "maps depth-sensor-frame points into colour-sensor frame"}
            d["stream_profile"]["depth_format"] = "z16"
            sensor = dev.first_depth_sensor()
            d["depth_scale"] = sensor.get_depth_scale()
            d["depth_units"] = "metres_per_unit"
            # native and colour-aligned depth are different representations with their own geometry/calibration
            d["depth_native"] = {**actual["depth"], "geometry": "depth-sensor image", "intrinsics": d["depth_intrinsics"],
                                 "extrinsics_to_color": d["extrinsics_depth_to_color"], "depth_scale": d["depth_scale"],
                                 "units": "metres_per_unit", "invalid_value": 0}
            d["depth_aligned"] = {"width": cs.width(), "height": cs.height(), "format": "z16",
                                  "geometry": "colour image (rs.align(color) applied to the native frame)",
                                  "intrinsics": d["color_intrinsics"], "depth_scale": d["depth_scale"], "units": "metres_per_unit",
                                  "invalid_value": 0, "provenance": "derived from native depth by the RealSense SDK; "
                                  "resampled, so it is not an independent measurement. Intrinsics = colour stream intrinsics."}
        d["negotiation"] = negotiation_record(self._requested, actual)
        for s in dev.query_sensors():
            opts = {}
            for o in s.get_supported_options():
                try:
                    opts[str(o)] = s.get_option(o)
                except Exception:
                    pass
            d["settings"][s.get_info(rs.camera_info.name)] = opts
        return d

    def read(self, timeout_s=1.0):
        import pyrealsense2 as rs

        try:
            fs = self.pipe.wait_for_frames(int(timeout_s * 1000))
        except RuntimeError:
            return None
        t_mono, t_wall = mono_ns(), wall_ns()
        aligned_fs = self.align.process(fs) if self.align else None
        col = fs.get_color_frame()
        dep = fs.get_depth_frame() if self.cfg.get("depth") else None
        dal = aligned_fs.get_depth_frame() if aligned_fs else None
        return CameraFrame(
            color_rgb=np.asanyarray(col.get_data()).copy(), host_mono_ns=t_mono, host_wall_ns=t_wall,
            frame_number=col.get_frame_number(), device_timestamp_ms=col.get_timestamp(),
            timestamp_domain=str(col.get_frame_timestamp_domain()),
            depth_native=None if dep is None else np.asanyarray(dep.get_data()).astype(np.uint16, copy=True),
            depth_aligned=None if dal is None else np.asanyarray(dal.get_data()).astype(np.uint16, copy=True),
            depth_frame_number=None if dep is None else dep.get_frame_number(),
            depth_device_timestamp_ms=None if dep is None else dep.get_timestamp(),
            depth_timestamp_domain=None if dep is None else str(dep.get_frame_timestamp_domain()),
        )

    def close(self):
        if self.pipe is not None:
            self.pipe.stop()
            self.pipe = None

    def describe(self):
        return self._desc
