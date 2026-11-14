"""MuJoCo twin adapter for a rollout of ONE fixed action chunk from a recorded start state.

Scene: the user's existing SO-101 MJCF (read-only, referenced by path). It is a generic simulator scene; its
fidelity to the physical robot, the user's calibration, controller gains and objects has NOT been validated.

Reset: every rollout calls mj_resetData, restores nominal friction, and re-initialises state from the record, so no
state leaks between rollouts (tested). Velocities start at zero when none were recorded (reported by the verifier).
Control: position actuators receive absolute targets (contract -> radians). Substep schedule follows the cumulative
rounding of 1/(control_hz*timestep) so the control period is honoured on average when it is not an integer number of steps.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ..robot.action_contract import JOINT_NAMES, ActionContract
from ..util import hash_obj, sha256_file
from .perturbation import PerturbationSample

OUTCOMES = ("ok", "simulator_error", "invalid_init", "timeout", "unknown")
_WARNINGS = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL", "mjWARN_INERTIA")


class SimulatorError(RuntimeError):
    pass


@dataclass
class RolloutResult:
    index: int
    outcome: str                              # one of OUTCOMES
    task_failure: bool | None = None          # None unless the rollout finished validly
    safety_violations: list[str] = field(default_factory=list)
    ee_final_m: list[float] | None = None
    ee_error_m: float | None = None
    detail: str = ""
    wall_s: float = 0.0
    steps: int = 0

    @property
    def valid(self) -> bool:
        return self.outcome == "ok"

    @property
    def failed(self) -> bool:
        return self.valid and (bool(self.task_failure) or bool(self.safety_violations))


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


class MujocoTwin:
    def __init__(self, scene_xml: Path, twin_cfg: dict, rollout_cfg: dict):
        self.scene_xml = Path(scene_xml)
        if not self.scene_xml.is_file():
            raise FileNotFoundError(f"scene xml not found: {self.scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.cfg, self.rcfg = twin_cfg, rollout_cfg
        m = self.model
        self.joint_names = JOINT_NAMES
        self._qadr = np.array([m.joint(n).qposadr[0] for n in JOINT_NAMES])
        self._act = np.array([m.actuator(n).id for n in JOINT_NAMES])
        self.ee_site = m.site(twin_cfg["ee_site"]).id
        self.floor_geom = m.geom(twin_cfg["floor_geom"]).id
        root = m.body_rootid[m.joint(JOINT_NAMES[0]).bodyid[0]]
        self.robot_geoms = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root}
        self.objects = {}
        for name, spec in twin_cfg["objects"].items():
            jn = m.joint(spec["free_joint"])
            self.objects[name] = {"body": m.body(spec["body"]).id, "geom": m.geom(spec["geom"]).id,
                                  "qadr": int(jn.qposadr[0]), "quat_default_z": float(spec.get("z_hint", 0.0))}
        self.static_geoms = {}
        for b in twin_cfg.get("static_bodies", []):
            bid = m.body(b).id
            self.static_geoms[b] = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
        self._nominal_friction = m.geom_friction.copy()
        self._nominal_qpos0 = m.qpos0.copy()
        self.dt = float(m.opt.timestep)
        self.joint_limits_deg = {n: tuple(np.rad2deg(m.jnt_range[m.joint(n).id])) for n in JOINT_NAMES}
        self.gripper_range_rad = tuple(float(x) for x in m.jnt_range[m.joint("gripper").id])
        self.asset_hash, self.asset_files = self._hash_assets()

    # ---- identity -----------------------------------------------------------------------------------------
    def _hash_assets(self) -> tuple[str, dict]:
        root = self.scene_xml.parent
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".xml", ".stl", ".obj", ".png", ".urdf")
                       and "__pycache__" not in p.parts)
        # only files reachable from the scene directory's xml/asset tree; hash relative path + content
        entries = {str(p.relative_to(root)): sha256_file(p) for p in files}
        return hash_obj(entries), entries

    def describe(self) -> dict:
        m = self.model
        return {"engine": "mujoco", "mujoco_version": mujoco.__version__, "scene_xml": str(self.scene_xml),
                "robot_asset_hash": self.asset_hash, "n_asset_files": len(self.asset_files), "timestep_s": self.dt,
                "ngeom": int(m.ngeom), "nbody": int(m.nbody), "nu": int(m.nu), "ee_site": self.cfg["ee_site"],
                "fidelity": "UNVALIDATED generic SO-101 scene; not calibrated to the physical robot or the user's calibration"}

    def ee_position(self, q_rad: np.ndarray) -> np.ndarray:
        d = mujoco.MjData(self.model)
        d.qpos[:] = self._nominal_qpos0
        d.qpos[self._qadr] = q_rad
        mujoco.mj_forward(self.model, d)
        return d.site_xpos[self.ee_site].copy()

    # ---- helpers ------------------------------------------------------------------------------------------
    def _warning_hit(self) -> str | None:
        for name in _WARNINGS:
            idx = getattr(mujoco.mjtWarning, name, None)
            if idx is not None and self.data.warning[int(idx)].number > 0:
                return name
        return None

    def _contacts(self, thr: float, watch_bins: bool) -> list[str]:
        d, hits = self.data, []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > thr:
                continue
            a, b = int(c.geom1), int(c.geom2)
            ra, rb = a in self.robot_geoms, b in self.robot_geoms
            if ra == rb:
                continue
            other = b if ra else a
            if other == self.floor_geom:
                hits.append("robot_floor_contact")
            elif watch_bins:
                for name, gs in self.static_geoms.items():
                    if other in gs:
                        hits.append(f"robot_{name}_collision")
        return hits

    # ---- rollout ------------------------------------------------------------------------------------------
    def run_rollout(self, contract: ActionContract, init_state: np.ndarray, objects: dict, action: np.ndarray,
                    sample: PerturbationSample, control_hz: float, task_target: np.ndarray | None,
                    tolerance_m: float | None, deadline: float | None = None, step_cb=None) -> RolloutResult:
        """step_cb(model, data, step) is an optional read-only visualisation hook; it must not modify the simulation."""
        t0 = time.monotonic()
        m, d, rc = self.model, self.data, self.rcfg
        res = RolloutResult(index=sample.index, outcome="unknown")
        try:
            # complete reset
            mujoco.mj_resetData(m, d)
            m.geom_friction[:] = self._nominal_friction
            for i in range(len(d.warning)):
                d.warning[i].number = 0
            # initial state (+ sampled measurement error on arm joints only)
            q0 = contract.to_sim_rad(init_state)
            q0[:5] += np.deg2rad(sample.state_error_deg)
            tol = math.radians(float(rc.get("init_limit_tolerance_deg", 2.0)))
            lo, hi = m.jnt_range[[m.joint(n).id for n in JOINT_NAMES], 0], m.jnt_range[[m.joint(n).id for n in JOINT_NAMES], 1]
            if np.any(q0 < lo - tol) or np.any(q0 > hi + tol):
                res.outcome, res.detail = "invalid_init", "perturbed initial joint state outside simulator limits"
                return res
            d.qpos[self._qadr] = q0
            d.ctrl[self._act] = q0
            ws = rc["workspace_m"]
            for name, ob in objects.items():
                o = self.objects.get(name)
                if o is None:
                    res.outcome, res.detail = "invalid_init", f"object '{name}' not in twin"
                    return res
                pos = np.asarray(ob["position_m"], dtype=float).copy()
                pos[:2] += sample.object_xy_m
                if not (ws["x"][0] <= pos[0] <= ws["x"][1] and ws["y"][0] <= pos[1] <= ws["y"][1]):
                    res.outcome, res.detail = "invalid_init", f"object '{name}' outside workspace after perturbation"
                    return res
                half = 0.5 * math.radians(sample.object_yaw_deg)
                quat = quat_mul([math.cos(half), 0, 0, math.sin(half)], np.asarray(ob["quat_wxyz"], dtype=float))
                quat /= np.linalg.norm(quat)
                d.qpos[o["qadr"]:o["qadr"] + 3] = pos
                d.qpos[o["qadr"] + 3:o["qadr"] + 7] = quat
                m.geom_friction[o["geom"], 0] = self._nominal_friction[o["geom"], 0] * sample.friction_scale
            mujoco.mj_forward(m, d)
            pen = -float(rc.get("init_penetration_m", 0.002))
            for name in objects:
                for i in range(d.ncon):
                    c = d.contact[i]
                    g = {int(c.geom1), int(c.geom2)}
                    if self.objects[name]["geom"] in g and g & self.robot_geoms and c.dist < pen:
                        res.outcome, res.detail = "invalid_init", f"object '{name}' penetrates robot at init"
                        return res

            # control schedule
            T = action.shape[0]
            targets_deg = action.copy()
            targets_deg[:, :5] += sample.control_noise_deg
            targets = contract.to_sim_rad(targets_deg)
            per = 1.0 / (control_hz * self.dt)
            bounds = [0] + [int(round((t + 1) * per)) for t in range(T)]
            settle = int(round(float(rc.get("settle_s", 0.5)) / self.dt))
            check_every = max(1, int(rc.get("predicate_check_every_substeps", 5)))
            thr = float(rc.get("contact_dist_threshold_m", 0.0))
            watch_bins = bool(rc.get("watch_static_collisions", True))
            ws_z = ws["z"]
            wall = float(rc.get("rollout_timeout_s", 30.0))
            linear = contract.interpolation == "linear"
            violations: set[str] = set()
            prev = q0.copy()
            step = 0
            total = bounds[-1] + settle
            for t in range(T + 1):
                n = (bounds[t + 1] - bounds[t]) if t < T else settle
                tgt = targets[t] if t < T else targets[-1]
                for k in range(n):
                    d.ctrl[self._act] = prev + (tgt - prev) * ((k + 1) / n) if (linear and t < T) else tgt
                    mujoco.mj_step(m, d)
                    step += 1
                    if step_cb is not None:
                        step_cb(m, d, step)
                    if step % check_every == 0 or step == total:
                        if not np.all(np.isfinite(d.qpos)):
                            raise SimulatorError("non-finite qpos")
                        w = self._warning_hit()
                        if w:
                            raise SimulatorError(f"MuJoCo warning {w}")
                        violations.update(self._contacts(thr, watch_bins))
                        ee = d.site_xpos[self.ee_site]
                        if not (ws["x"][0] <= ee[0] <= ws["x"][1] and ws["y"][0] <= ee[1] <= ws["y"][1] and ws_z[0] <= ee[2] <= ws_z[1]):
                            violations.add("ee_outside_workspace")
                        if rc.get("check_object_displacement") is not None:
                            lim = float(rc["check_object_displacement"])
                            for name, ob in objects.items():
                                p = d.qpos[self.objects[name]["qadr"]:self.objects[name]["qadr"] + 2]
                                if np.linalg.norm(p - (np.asarray(ob["position_m"])[:2] + sample.object_xy_m)) > lim and name not in rc.get("task_objects", []):
                                    violations.add(f"unintended_displacement_{name}")
                    if (time.monotonic() - t0 > wall) or (deadline is not None and time.monotonic() > deadline):
                        res.outcome, res.detail = "timeout", "rollout wall-clock budget exceeded"
                        res.steps = step
                        return res
                if t < T:
                    prev = targets[t]
            mujoco.mj_forward(m, d)
            ee = d.site_xpos[self.ee_site].copy()
            res.ee_final_m = ee.tolist()
            if not np.all(np.isfinite(ee)):
                raise SimulatorError("non-finite end-effector position")
            res.outcome = "ok"
            res.safety_violations = sorted(violations)
            if task_target is not None and tolerance_m is not None:
                res.ee_error_m = float(np.linalg.norm(ee - np.asarray(task_target)))
                res.task_failure = bool(res.ee_error_m > tolerance_m)
            else:
                res.outcome, res.detail = "unknown", "no task predicate available"
            res.steps = step
            return res
        except SimulatorError as e:
            res.outcome, res.detail = "simulator_error", str(e)
            return res
        except Exception as e:  # any engine exception is a simulator error, NOT a task failure
            res.outcome, res.detail = "simulator_error", f"{type(e).__name__}: {e}"
            return res
        finally:
            res.wall_s = time.monotonic() - t0
            m.geom_friction[:] = self._nominal_friction
