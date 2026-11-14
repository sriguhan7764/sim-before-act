"""TEST GUARD: stands in for pyserial. Importing is harmless; OPENING a port is logged to $HW_GUARD_LOG and raises."""
import os


class HardwareAccessAttempt(RuntimeError):
    pass


def _log(what):
    with open(os.environ["HW_GUARD_LOG"], "a") as f:
        f.write(f"serial: {what}\n")


class Serial:
    def __init__(self, *a, **k):
        _log(f"Serial{a}{k}")
        raise HardwareAccessAttempt("offline path tried to open a serial port")


class SerialException(Exception):
    pass
