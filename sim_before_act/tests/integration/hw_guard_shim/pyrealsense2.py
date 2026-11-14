"""TEST GUARD: stands in for pyrealsense2. Any device/pipeline/context use is logged to $HW_GUARD_LOG and refused."""
import os


def _refuse(what):
    with open(os.environ["HW_GUARD_LOG"], "a") as f:
        f.write(f"pyrealsense2: {what}\n")
    raise RuntimeError("offline path tried to use a RealSense device")


class pipeline:
    def __init__(self, *a, **k):
        _refuse("pipeline()")


class context:
    def __init__(self, *a, **k):
        _refuse("context()")


class config:
    def __init__(self, *a, **k):
        _refuse("config()")
