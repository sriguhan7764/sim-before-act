"""Joint-name ordering and raw-vs-calibrated semantics.

Raw motor values are preserved exactly as the SDK returns them. Calibrated angles are DERIVED:

  revolute joints : angle_rad = (raw - (range_min + range_max)/2) * 2*pi / (resolution - 1)
                    i.e. angle relative to the middle of the calibrated range, matching the
                    LeRobot ``MotorNormMode.DEGREES`` convention (lerobot 0.6.1 motors_bus._normalize),
                    converted degrees->radians. Valid only if the calibration file corresponds to the
                    homing offsets currently stored in the motors (Present_Position is post-homing-offset).
                    The tick difference is taken on the 4096-tick circle, so a joint whose travel crosses the
                    0/4095 wrap point still yields continuous angles (travel must be < 360 deg).
  gripper         : normalised 0..100 over [range_min, range_max] (LeRobot ``RANGE_0_100``). NOT radians.
"""
from __future__ import annotations

import math
from pathlib import Path

from .util import read_json, sha256_file

TICKS_PER_REV = 4096  # STS3215 resolution; max index 4095

# Units of LeRobot-normalised values as published by collector.lerobot_bridge (follower/leader state and commands).
LEROBOT_NORMALIZED_UNITS = ("arm joints: degrees about the midpoint of the calibration range (LeRobot "
                            "MotorNormMode.DEGREES, unclamped); gripper: 0..100 over the calibrated range "
                            "(RANGE_0_100, bounded)")


class AngleMapping:
    def __init__(self, joints: list[dict], calibration: dict, source_path: str | None = None, sha256: str | None = None):
        self.joints = joints
        self.calibration = calibration
        self.source_path, self.sha256 = source_path, sha256
        for j in joints:
            c = calibration.get(j["name"])
            if c is None:
                raise ValueError(f"calibration has no entry for joint '{j['name']}'")
            if j.get("id") is not None and c.get("id") != j["id"]:
                raise ValueError(f"calibration id {c.get('id')} != configured motor id {j['id']} for {j['name']}")
            if c["range_max"] == c["range_min"]:
                raise ValueError(f"degenerate calibration range for {j['name']}")

    @classmethod
    def from_file(cls, joints: list[dict], path: str | Path) -> "AngleMapping":
        return cls(joints, read_json(path), str(path), sha256_file(path))

    @property
    def names(self) -> list[str]:
        return [j["name"] for j in self.joints]

    def convert(self, raw: dict[str, int]) -> dict:
        """Return {'joint_angles_rad': [...in joint order, revolute only...], 'revolute_names': [...],
        'gripper': {name: {'value':..,'unit':'normalized_0_100'}}}."""
        angles, names, grip = [], [], {}
        for j in self.joints:
            c, r = self.calibration[j["name"]], raw[j["name"]]
            if j["type"] == "revolute":
                mid = (c["range_min"] + c["range_max"]) / 2.0
                d = (r - mid + TICKS_PER_REV / 2) % TICKS_PER_REV - TICKS_PER_REV / 2  # shortest signed distance on the 4096-tick circle
                angles.append(d * 2 * math.pi / (TICKS_PER_REV - 1))
                names.append(j["name"])
            elif j["type"] == "gripper":
                lo, hi = c["range_min"], c["range_max"]
                v = (min(hi, max(lo, r)) - lo) / (hi - lo) * 100.0
                grip[j["name"]] = {"value": 100.0 - v if c.get("drive_mode") else v, "unit": "normalized_0_100"}
            else:
                raise ValueError(f"unknown joint type {j['type']}")
        return {"joint_angles_rad": angles, "revolute_names": names, "gripper": grip}

    def describe(self) -> dict:
        return {"mapping": "lerobot_feetech_midrange_degrees_to_radians", "calibration_file": self.source_path,
                "calibration_sha256": self.sha256, "calibration": self.calibration, "joints": self.joints,
                "gripper_unit": "normalized_0_100 (not radians)"}


def order_raw(joints: list[dict], raw: dict[str, int]) -> list[int]:
    return [raw[j["name"]] for j in joints]
