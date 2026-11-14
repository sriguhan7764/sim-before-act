"""TEST GUARD: port discovery is also hardware access; logged and refused."""
import os


def comports(*a, **k):
    with open(os.environ["HW_GUARD_LOG"], "a") as f:
        f.write("serial: list_ports.comports()\n")
    raise RuntimeError("offline path tried to enumerate serial ports")
