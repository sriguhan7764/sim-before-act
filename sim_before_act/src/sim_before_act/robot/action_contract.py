"""Action contract: the single definition of what an action chunk means.

Representation: ABSOLUTE joint position targets (not deltas, not torques). Order/units fixed below. The same
semantics must hold for replay, simulation and (later) physical execution. Nothing here clips: an out-of-range
chunk is INVALID and must be re-proposed (a changed action is a new candidate and is re-verified).

Units: arm joints in degrees, gripper in LeRobot's 0..100 normalised range (NOT radians, NOT degrees).
Gripper -> simulator mapping is an UNVALIDATED assumption (see ``gripper_mapping``); the MJCF has no 0..100 joint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..util import hash_obj

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
UNITS = ("degrees",) * 5 + ("percent_0_100",)
REPRESENTATION = "absolute_joint_position_targets"
INTERPOLATIONS = ("zero_order_hold", "linear")
REFERENCE = "follower joint-position target space"
GRIPPER_SEMANTICS = "normalized_0_100_0_closed"   # LeRobot RANGE_0_100; direction 0 = closed is an UNVALIDATED sim assumption
CONTROLLER_NOTE = ("Targets are absolute joint-position targets, NOT torques. The twin tracks them with MuJoCo position "
                   "actuators (kp/kv PD + forcerange from the CAD-derived MJCF); this is NOT the Feetech STS3215 controller dynamics.")


@dataclass(frozen=True)
class ActionContract:
    joint_names: tuple[str, ...]
    units: tuple[str, ...]
    representation: str
    control_hz: float
    interpolation: str
    min_chunk_len: int
    max_chunk_len: int
    limits_lo: tuple[float, ...]
    limits_hi: tuple[float, ...]
    limits_source: str
    max_joint_speed_deg_s: float | None          # None = speed check NOT performed (reported as such)
    gripper_range_rad: tuple[float, float]        # simulator joint range used by the gripper mapping
    gripper_mapping: dict = field(default_factory=dict)
    reference: str = REFERENCE
    gripper_semantics: str = GRIPPER_SEMANTICS

    def chunk_duration_s(self, n_steps: int) -> float:
        """Duration of a chunk of ``n_steps`` targets = T / control_hz."""
        return int(n_steps) / self.control_hz

    @property
    def contract_id(self) -> str:
        """Stable hash of the contract description (not of any chunk); changes if limits, units, order, hz etc. change."""
        d = self._describe_core()
        return hash_obj(d)

    @staticmethod
    def build(cfg: dict, joint_limits_deg: dict[str, tuple[float, float]], gripper_range_rad: tuple[float, float]):
        names = tuple(cfg.get("joint_names", JOINT_NAMES))
        units = tuple(cfg.get("units", UNITS))
        if names != JOINT_NAMES or units != UNITS:
            raise ValueError(f"contract joints/units must be {JOINT_NAMES} / {UNITS}")
        if cfg.get("interpolation", "zero_order_hold") not in INTERPOLATIONS:
            raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")
        rep = cfg.get("representation", REPRESENTATION)
        if rep != REPRESENTATION:
            raise ValueError(f"representation {rep!r} unsupported: only {REPRESENTATION!r} (absolute targets); "
                             "relative/delta actions are rejected, not converted")
        hz = float(cfg["control_hz"])
        if not (math.isfinite(hz) and hz > 0):
            raise ValueError("control_hz must be a positive number")
        lo, hi = [], []
        override = cfg.get("limits_override_deg") or {}
        for n in names:
            a, b = (0.0, 100.0) if n == "gripper" else tuple(override.get(n) or joint_limits_deg[n])
            lo.append(round(float(a), 6)); hi.append(round(float(b), 6))
        return ActionContract(names, units, cfg.get("representation", REPRESENTATION), hz,
                              cfg.get("interpolation", "zero_order_hold"), int(cfg.get("min_chunk_len", 1)),
                              int(cfg.get("max_chunk_len", 200)), tuple(lo), tuple(hi),
                              "override+mjcf_joint_range" if override else "mjcf_joint_range (simulator limits, NOT hardware limits)",
                              cfg.get("max_joint_speed_deg_s"), tuple(gripper_range_rad),
                              dict(cfg.get("gripper_mapping") or {}))

    def _describe_core(self) -> dict:
        return {"joint_names": list(self.joint_names), "units": list(self.units), "representation": self.representation,
                "reference": self.reference, "gripper_semantics": self.gripper_semantics,
                "absolute_only": True,
                "control_hz": self.control_hz, "interpolation": self.interpolation,
                "chunk_duration_rule": "length / control_hz",
                "chunk_len_range": [self.min_chunk_len, self.max_chunk_len],
                "limits_lo": list(self.limits_lo), "limits_hi": list(self.limits_hi), "limits_source": self.limits_source,
                "max_joint_speed_deg_s": self.max_joint_speed_deg_s,
                "speed_check": "performed" if self.max_joint_speed_deg_s is not None else "NOT performed (limit unset)",
                "gripper_mapping": self.gripper_mapping, "gripper_range_rad": list(self.gripper_range_rad),
                "clipping": "never (out-of-range actions are rejected as invalid)",
                "controller_note": CONTROLLER_NOTE}

    def describe(self) -> dict:
        return {**self._describe_core(), "contract_id": self.contract_id}

    def validate_chunk(self, chunk, joint_names=None, units=None, representation=None, control_hz=None
                       ) -> tuple[np.ndarray | None, list[str]]:
        """Returns (float64 array (T, 6) or None, list of problems). Never modifies values."""
        errs: list[str] = []
        if joint_names is not None and tuple(joint_names) != self.joint_names:
            errs.append(f"joint names/order mismatch: {list(joint_names)} != {list(self.joint_names)}")
        if units is not None and tuple(units) != self.units:
            errs.append(f"units mismatch: {list(units)} != {list(self.units)}")
        if representation is not None and representation != self.representation:
            errs.append(f"representation mismatch: {representation!r} != {self.representation!r}")
        if control_hz is not None and not (isinstance(control_hz, (int, float)) and math.isclose(control_hz, self.control_hz)):
            errs.append(f"control_hz mismatch: {control_hz!r} != {self.control_hz}")
        try:
            a = np.asarray(chunk, dtype=np.float64)
        except (TypeError, ValueError):
            return None, errs + ["chunk is not a numeric array"]
        if a.ndim != 2 or a.shape[1] != len(self.joint_names):
            return None, errs + [f"chunk shape {a.shape} != (T, {len(self.joint_names)})"]
        if not (self.min_chunk_len <= a.shape[0] <= self.max_chunk_len):
            errs.append(f"chunk length {a.shape[0]} outside [{self.min_chunk_len}, {self.max_chunk_len}]")
        if not np.all(np.isfinite(a)):
            return None, errs + ["chunk contains non-finite values"]
        lo, hi = np.asarray(self.limits_lo), np.asarray(self.limits_hi)
        bad = np.argwhere((a < lo) | (a > hi))
        if bad.size:
            t, j = bad[0]
            errs.append(f"{len(bad)} value(s) outside limits, first: step {t} {self.joint_names[j]}={a[t, j]:.3f} not in "
                        f"[{lo[j]:.3f}, {hi[j]:.3f}]")
        if self.max_joint_speed_deg_s is not None and a.shape[0] > 1:
            v = np.abs(np.diff(a[:, :5], axis=0)) * self.control_hz
            if np.any(v > self.max_joint_speed_deg_s):
                errs.append(f"joint speed {v.max():.1f} deg/s exceeds {self.max_joint_speed_deg_s}")
        return a, errs

    def action_id(self, chunk: np.ndarray) -> str:
        a = np.ascontiguousarray(np.asarray(chunk, dtype="<f8"))
        return hash_obj({"contract": [self.joint_names, self.units, self.representation, self.control_hz],
                         "shape": list(a.shape), "data": a.tobytes().hex()})

    def to_sim_rad(self, values: np.ndarray) -> np.ndarray:
        """(..., 6) contract units -> simulator radians. Gripper: linear 0..100 -> [range_lo, range_hi] (UNVALIDATED)."""
        v = np.asarray(values, dtype=np.float64).copy()
        out = v.copy()
        out[..., :5] = np.deg2rad(v[..., :5])
        lo, hi = self.gripper_range_rad
        out[..., 5] = lo + (v[..., 5] / 100.0) * (hi - lo)
        return out
