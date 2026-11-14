"""INVENTED synthetic action fixtures for the demo scene (tests / CLI smoke runs). Not policy output, not recorded data.

They are expressed in contract units (degrees, gripper 0..100), start from the scene's initial robot configuration and stay within
the simulator limits (nothing is clipped).
"""
from __future__ import annotations

import numpy as np

from .config import SceneConfig

FIXTURES = ("zero", "small_motion")
FIXTURE_LABEL = "SYNTHETIC TEST FIXTURE: invented actions ({name}); not policy output, not recorded data"


def make_fixture(cfg: SceneConfig, name: str, steps: int = 45) -> np.ndarray:
    q0 = np.asarray(cfg.inputs["initial_robot"]["values"], dtype=float)
    a = np.tile(q0, (steps, 1))
    if name == "zero":
        return a
    if name == "small_motion":
        s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, steps))      # smooth 0 -> 1
        a[:, 0] += 8.0 * s              # pan 8 deg
        a[:, 1] += 6.0 * s              # shoulder_lift
        a[:, 5] += 20.0 * s             # gripper opens by 20 (percent units)
        return a
    raise ValueError(f"unknown fixture {name!r}; choose from {FIXTURES}")
