"""Seeded perturbation sampling for counterfactual rollouts.

The proposed action is NEVER perturbed here. Only initial-state and dynamics uncertainty is sampled
(measurement error, object pose, friction, controller execution noise). Every variable declares its distribution,
units, truncation bound, source and whether it is calibrated; the defaults in the demo config are explicitly
UNCALIBRATED placeholders. Observation (pixel) noise is deliberately unsupported: it would need a perception pipeline.

Seeding: rollout i uses SeedSequence(base_seed).spawn(max_rollouts)[i], so the first n samples are identical for any
budget >= n (paired budget comparisons) and independent of execution order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPPORTED = ("robot_state_error_deg", "object_xy_m", "object_yaw_deg", "friction_scale", "control_noise_deg")


@dataclass
class PerturbationSample:
    index: int
    seed_entropy: int
    spawn_key: tuple
    nominal: bool
    state_error_deg: np.ndarray      # (5,) added to the five arm joints at initialisation
    object_xy_m: np.ndarray          # (2,)
    object_yaw_deg: float
    friction_scale: float            # scales cube sliding friction
    control_noise_deg: np.ndarray    # (T, 5) added to the arm targets each control step (NOT a new action candidate)

    def summary(self) -> dict:
        return {"index": self.index, "nominal": self.nominal, "state_error_deg": self.state_error_deg,
                "object_xy_m": self.object_xy_m, "object_yaw_deg": self.object_yaw_deg,
                "friction_scale": self.friction_scale,
                "control_noise_deg_rms": float(np.sqrt(np.mean(self.control_noise_deg ** 2))) if self.control_noise_deg.size else 0.0}


def _trunc_normal(rng: np.random.Generator, sigma: float, bound: float, size) -> np.ndarray:
    """Zero-mean normal truncated to [-bound, bound] by rejection (deterministic given rng)."""
    out = np.zeros(size, dtype=np.float64)
    if sigma == 0:
        return out
    todo = np.ones(size, dtype=bool)
    while todo.any():
        draw = rng.normal(0.0, sigma, size)
        ok = todo & (np.abs(draw) <= bound)
        out[ok] = draw[ok]
        todo &= ~ok
    return out


def _draw(rng, spec: dict, size):
    if spec["dist"] == "normal":
        return _trunc_normal(rng, float(spec["sigma"]), float(spec["bound"]), size)
    return rng.uniform(float(spec["low"]), float(spec["high"]), size)


class PerturbationSampler:
    def __init__(self, cfg: dict, base_seed: int, max_rollouts: int):
        self.enabled = bool(cfg.get("enabled", True))
        self.vars = cfg.get("variables") or {}
        unknown = set(self.vars) - set(SUPPORTED)
        if unknown:
            raise ValueError(f"unsupported perturbation variables {sorted(unknown)}; supported: {SUPPORTED}")
        self.base_seed, self.max_rollouts = int(base_seed), int(max_rollouts)
        self._children = np.random.SeedSequence(self.base_seed).spawn(self.max_rollouts)

    def sample(self, index: int, chunk_len: int) -> PerturbationSample:
        if not 0 <= index < self.max_rollouts:
            raise IndexError(index)
        ss = self._children[index]
        rng = np.random.default_rng(ss)
        z = lambda *s: np.zeros(s)
        if not self.enabled:
            return PerturbationSample(index, ss.entropy, tuple(ss.spawn_key), True, z(5), z(2), 0.0, 1.0, z(chunk_len, 5))
        v = self.vars
        state_err = _draw(rng, v["robot_state_error_deg"], 5) if "robot_state_error_deg" in v else z(5)
        obj_xy = _draw(rng, v["object_xy_m"], 2) if "object_xy_m" in v else z(2)
        yaw = float(_draw(rng, v["object_yaw_deg"], 1)[0]) if "object_yaw_deg" in v else 0.0
        fric = float(_draw(rng, v["friction_scale"], 1)[0]) if "friction_scale" in v else 1.0
        noise = _draw(rng, v["control_noise_deg"], (chunk_len, 5)) if "control_noise_deg" in v else z(chunk_len, 5)
        return PerturbationSample(index, ss.entropy, tuple(ss.spawn_key), False, state_err, obj_xy, yaw, fric, noise)

    def describe(self) -> dict:
        return {"enabled": self.enabled, "base_seed": self.base_seed, "seeding": "SeedSequence(base_seed).spawn(max_rollouts)[i]",
                "variables": self.vars,
                "all_calibrated": bool(self.vars) and all(bool(s.get("calibrated")) for s in self.vars.values()),
                "unsupported_by_design": ["camera pixel noise (needs a perception pipeline)", "action perturbation (separate candidates)",
                                          "actuator gain/response (not implemented in this slice)"]}
