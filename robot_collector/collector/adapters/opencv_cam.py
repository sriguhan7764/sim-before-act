"""Generic OpenCV (V4L2/UVC) camera adapter, e.g. a wrist camera. Not hardware-tested.

No device timestamp or intrinsics are available from OpenCV: these are recorded as null, and
intrinsics may be supplied through config (``intrinsics: {K:..., dist:..., distortion_model:...}``).
"""
from __future__ import annotations

from ..negotiation import ProfileMismatch, negotiation_record
from ..util import mono_ns, wall_ns
from .base import CameraFrame, CameraSource


class OpenCVCamera(CameraSource):
    def __init__(self, cfg: dict):
        self.cfg, self.name = cfg, cfg["name"]
        self.cap = None
        self._n = 0
        self._described: dict = {}

    def open(self):
        from .. import hardware_gate

        dev = self.cfg.get("device")
        if dev is None:
            raise RuntimeError(f"camera '{self.name}': config 'device' is unset; refusing to guess a /dev/video index")
        hardware_gate.require(f"OpenCV/V4L2 camera open ({self.name})")
        import cv2

        self.cap = cv2.VideoCapture(dev)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera '{self.name}': cannot open {dev}")
        for prop, key in ((cv2.CAP_PROP_FRAME_WIDTH, "width"), (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
                          (cv2.CAP_PROP_FPS, "fps")):
            if self.cfg.get(key):
                self.cap.set(prop, float(self.cfg[key]))
        fc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        self._actual = {"width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": float(self.cap.get(cv2.CAP_PROP_FPS)),
                        "format": "".join(chr((fc >> (8 * i)) & 0xFF) for i in range(4)).strip() or None}
        self._negotiation = negotiation_record(
            {"color": {"width": self.cfg.get("width"), "height": self.cfg.get("height"), "fps": self.cfg.get("fps")}},
            {"color": self._actual})
        if not self._negotiation["matches_request"]:
            self.close()
            raise ProfileMismatch(f"camera '{self.name}' negotiated a different profile than requested: "
                                  f"{self._negotiation['mismatches']}")

    def read(self, timeout_s=1.0):
        import cv2

        ok, bgr = self.cap.read()
        if not ok:
            return None
        t_m, t_w = mono_ns(), wall_ns()
        n, self._n = self._n, self._n + 1
        return CameraFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_m, t_w, frame_number=None,
                           device_timestamp_ms=None, timestamp_domain=None, extras={"receipt_index": n})

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def describe(self):
        d = self._describe_open() if self.cap is not None else self._described
        return d

    def _describe_open(self):
        import cv2

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        d = {"backend": "opencv", "is_mock": False, "device": {"path_or_index": self.cfg.get("device")},
             "stream_profile": {"width": w, "height": h, "fps_configured": self.cfg.get("fps"),
                                "fps_reported": self.cap.get(cv2.CAP_PROP_FPS), "fourcc": self._actual["format"],
                                "color_format": "rgb8 (converted from BGR)", "depth_format": None},
             "color_intrinsics": self.cfg.get("intrinsics"), "depth_intrinsics": None,
             "extrinsics_depth_to_color": None, "depth_scale": None, "depth_units": None, "settings": {},
             "negotiation": self._negotiation,
             "timing_note": "OpenCV exposes no device timestamp or frame counter; only host receipt time is recorded"}
        self._described = d
        return d
