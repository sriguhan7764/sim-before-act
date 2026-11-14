"""Headless rollout of an action chunk / validated action segment in the cube-to-tray scene.

``run_scene_rollout(scene_cfg, segment_or_actions, seed, perturbation=None)`` is deterministic given its inputs and seed (MuJoCo
on one machine; bitwise equality across machines/versions is not guaranteed).

Control path (NOT the real controller): contract units -> ``ActionContract.to_sim_rad`` -> ``data.ctrl`` of MuJoCo position
actuators (PD: kp*(target-q) - kv*qdot, clipped to forcerange). Zero-order hold (or linear interpolation) at ``control_hz``.
Policy outputs are position TARGETS, never torques; the real Feetech STS3215 position controller is not modelled.

Nothing is clipped: invalid segments/arrays and initial states outside the simulator limits give an ``invalid_rollout`` result.
Control noise from the perturbation sampler is the documented execution-noise perturbation (recorded), not a new action.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from ...robot.action_contract import CONTROLLER_NOTE
from ...robot.segment import (ActionSegment, SegmentInvalid, ValidatedSegment, actions_sha256, load_action_segment,
                              validate_for_sim)
from ...util import hash_obj, to_jsonable
from ..perturbation import PerturbationSample, PerturbationSampler
from .builder import Scene, build_scene
from .config import THRESHOLD_STATUS, SceneConfig, load_scene_config
from .outcomes import (CONTACT_CATEGORIES, TraceArrays, combine_outcome, contact_report, eval_constraints, eval_contacts,
                       eval_inside_tray, eval_lift, eval_release_settle)

_WARNINGS = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL", "mjWARN_INERTIA")


@dataclass
class SceneRolloutResult:
    outcome: dict
    predicates: dict | None
    trace: dict                          # name -> ndarray (n, ...)
    labels: dict
    action_record: dict
    scene: dict
    perturbation: dict | None
    seed: int
    detail: str = ""
    sim: dict = field(default_factory=dict)
    wall_s: float = 0.0

    @property
    def category(self) -> str:
        return self.outcome["category"]

    def summary(self) -> dict:
        return to_jsonable({"labels": self.labels, "outcome": self.outcome, "predicates": self.predicates, "action": self.action_record,
                            "scene": self.scene, "perturbation": self.perturbation, "seed": self.seed, "detail": self.detail,
                            "sim": self.sim, "wall_s": self.wall_s,
                            "trace_arrays": {k: list(v.shape) for k, v in self.trace.items()}})

    def save(self, out_dir: str | Path, stem: str) -> dict[str, Path]:
        """Write <stem>_trace.npz and <stem>_summary.json (derived artifacts; an existing file with the same name is replaced)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        npz, js = out / f"{stem}_trace.npz", out / f"{stem}_summary.json"
        np.savez_compressed(npz, **self.trace)
        js.write_text(json.dumps(self.summary(), indent=2, sort_keys=True, allow_nan=False) + "\n")
        return {"trace": npz, "summary": js}


def check_sample_in_ranges(sample: PerturbationSample, pcfg: dict) -> list[str]:
    """Every sampled perturbation must lie within its documented range (truncation bound / uniform limits)."""
    probs = []
    vars_ = (pcfg or {}).get("variables") or {}
    vals = {"robot_state_error_deg": sample.state_error_deg, "object_xy_m": sample.object_xy_m,
            "object_yaw_deg": np.asarray([sample.object_yaw_deg]), "friction_scale": np.asarray([sample.friction_scale]),
            "control_noise_deg": sample.control_noise_deg}
    for k, spec in vars_.items():
        v = np.asarray(vals[k], dtype=float)
        if not v.size:
            continue
        lo, hi = (-float(spec["bound"]), float(spec["bound"])) if spec["dist"] == "normal" else (float(spec["low"]), float(spec["high"]))
        if not (np.all(np.isfinite(v)) and v.min() >= lo - 1e-12 and v.max() <= hi + 1e-12):
            probs.append(f"perturbation '{k}' outside its documented range [{lo}, {hi}]")
    return probs


def _labels(cfg: SceneConfig, seg_prov: dict | None) -> dict:
    return {"scene": cfg.label, "mode": cfg.mode,
            "geometry": "CAD geometry; NOT a physically calibrated twin",
            "thresholds": f"{THRESHOLD_STATUS}: outcome thresholds are assumptions, not calibrated tolerances",
            "controller": CONTROLLER_NOTE,
            "segment": (seg_prov or {}).get("provenance_label", "no segment: raw action array supplied by the caller"),
            "policy": (seg_prov or {}).get("policy_label", "unknown (no segment provenance)"),
            "data_origin": (seg_prov or {}).get("data_origin"), "segment_flags": (seg_prov or {}).get("flags", [])}


def _invalid(scene: Scene, reasons: list[str], seg_prov: dict | None, action_record: dict, seed: int, pert: dict | None,
             t0: float) -> SceneRolloutResult:
    cfg = scene.cfg
    out = {"category": "invalid_rollout", "task_outcome": "unknown", "task_failure_reasons": [], "safety_violation": False,
           "safety_reasons": [], "invalid_reason": reasons, "timeout": False, "horizon_kind": None, "horizon_s": 0.0,
           "task_horizon_s": cfg.section("simulation")["task_horizon_s"], "predicates_observed_true": [],
           "predicates_not_evaluable": ["lift", "inside_tray", "release_settle"], "predicates_unmet_but_inconclusive": [],
           "note": "rollout not run or not usable; no task outcome", "threshold_status": THRESHOLD_STATUS}
    return SceneRolloutResult(outcome=out, predicates=None, trace={}, labels=_labels(cfg, seg_prov), action_record=action_record,
                              scene=scene.describe(), perturbation=pert, seed=seed, detail="; ".join(reasons),
                              wall_s=time.monotonic() - t0)


def _resolve_perturbation(perturbation, seed: int, n_steps: int):
    """-> (sample, description dict, problems)."""
    if perturbation is None:
        s = PerturbationSampler({"enabled": False}, seed, 1).sample(0, n_steps)
        return s, {"mode": "none (nominal)", "seed": seed}, []
    if isinstance(perturbation, PerturbationSample):
        probs = []
        if perturbation.control_noise_deg.shape != (n_steps, 5):
            probs.append(f"perturbation control_noise_deg shape {perturbation.control_noise_deg.shape} != ({n_steps}, 5)")
        return perturbation, {"mode": "explicit PerturbationSample (ranges not checked: no config supplied)", "seed": seed,
                              "sample": perturbation.summary()}, probs
    sampler = PerturbationSampler(perturbation, seed, 1)
    s = sampler.sample(0, n_steps)
    desc = sampler.describe()
    return s, {"mode": "sampled", "seed": seed, "sampler": desc, "sample": s.summary(), "ranges_checked": True,
               "all_calibrated": desc["all_calibrated"]}, check_sample_in_ranges(s, perturbation)


def run_scene_rollout(scene_or_cfg, segment_or_actions, seed: int = 0, perturbation=None, *, init: str = "scene",
                      use_full_chunk: bool = False, deadline: float | None = None, step_cb=None) -> SceneRolloutResult:
    """Run one rollout. Raises SceneRefusal (physical mode, missing inputs / open blocker): that is a refusal, not a result.

    ``scene_or_cfg``: Scene | SceneConfig | path. ``segment_or_actions``: ActionSegment | ValidatedSegment | segment path | (T,6) array.
    ``perturbation``: None (nominal) | perturbation config dict (see ``SceneConfig.perturbation_cfg()``) | PerturbationSample.
    ``init``: 'scene' | 'segment_state'. ``step_cb(model, data, step)`` is a read-only hook.
    """
    t0 = time.monotonic()
    if init not in ("scene", "segment_state"):
        raise ValueError("init must be 'scene' or 'segment_state'")
    if isinstance(scene_or_cfg, Scene):
        scene = scene_or_cfg
    else:
        cfg0 = scene_or_cfg if isinstance(scene_or_cfg, SceneConfig) else load_scene_config(scene_or_cfg)
        scene = build_scene(cfg0, init_state_available=(init == "segment_state"))
    cfg, contract, m = scene.cfg, scene.contract, scene.model
    sim = cfg.section("simulation")

    # ---- resolve + validate the action (never repaired) ----------------------------------------------------------------
    vs: ValidatedSegment | None = None
    seg_prov = None
    action_record: dict = {"source": None, "action_sha256": None, "contract_id": contract.contract_id}
    seg = segment_or_actions
    if isinstance(seg, (str, Path)):
        seg = load_action_segment(seg)
    if isinstance(seg, ActionSegment):
        try:
            vs = validate_for_sim(seg, contract)
        except SegmentInvalid as e:
            action_record.update(source="segment", segment_id=seg.segment_id, validation_problems=e.problems)
            return _invalid(scene, ["segment refused: " + p for p in e.problems], None, action_record, seed, None, t0)
    elif isinstance(seg, ValidatedSegment):
        vs = seg
        if vs.contract_id != contract.contract_id:
            action_record.update(source="validated_segment", segment_id=vs.segment_id)
            return _invalid(scene, [f"segment validated against contract {vs.contract_id[:12]}..., scene contract is "
                                    f"{contract.contract_id[:12]}...: re-validate"], vs.provenance(), action_record, seed, None, t0)
        if actions_sha256(vs.actions) != vs.action_sha256:
            return _invalid(scene, ["validated segment's action hash no longer matches its actions"], vs.provenance(),
                            action_record, seed, None, t0)
    if vs is not None:
        arr = vs.actions if use_full_chunk else vs.executed_actions
        seg_prov = vs.provenance()
        action_record.update(source="segment", segment_id=vs.segment_id, action_sha256=vs.action_sha256,
                             chunk_length=int(vs.actions.shape[0]), executed_steps=int(arr.shape[0]),
                             executed_prefix_sha256=actions_sha256(arr), comparison_exact=vs.exact_comparison,
                             execution_horizon_steps=vs.execution_horizon_steps)
        if not vs.exact_comparison:
            action_record["comparison_note"] = ("replanning/temporal ensembling: executed commands may differ from this chunk: "
                                                "comparison not exact")
    else:
        a, errs = contract.validate_chunk(seg)
        if errs or a is None:
            action_record.update(source="array", validation_problems=errs)
            return _invalid(scene, ["actions refused: " + e for e in errs], None, action_record, seed, None, t0)
        arr = a
        action_record.update(source="array", action_sha256=actions_sha256(a), chunk_length=int(a.shape[0]), executed_steps=int(a.shape[0]))
    n_steps = int(arr.shape[0])

    # ---- initial state ------------------------------------------------------------------------------------------------------
    if init == "segment_state":
        if vs is None:
            return _invalid(scene, ["init='segment_state' needs a segment"], seg_prov, action_record, seed, None, t0)
        init_deg = np.asarray(vs.state_values, dtype=float)
    else:
        init_deg = np.asarray(cfg.inputs["initial_robot"]["values"], dtype=float)
    action_record["initial_state_source"] = "segment input.state_values" if init == "segment_state" else "scene inputs.initial_robot.values"
    action_record["initial_state_deg"] = init_deg.tolist()

    sample, pert_desc, pprobs = _resolve_perturbation(perturbation, seed, n_steps)
    if pprobs:
        return _invalid(scene, pprobs, seg_prov, action_record, seed, pert_desc, t0)

    d = scene.new_data()
    th = cfg.section("outcome")
    nominal_friction = m.geom_friction.copy()
    geom = scene.geometry
    rows: dict[str, list] = {k: [] for k in ("t_s", "q_rad", "ctrl", "qpos", "cube_pos", "cube_quat", "cube_speed", "gripper_pct",
                                             "ee_pos", "contact_present", "contact_pen", "max_abs_qvel")}
    invalid_reason, timed_out, step = None, False, 0
    try:
        q0 = contract.to_sim_rad(init_deg)
        q0[:5] += np.deg2rad(sample.state_error_deg)
        tol = math.radians(float(sim["init_limit_tolerance_deg"]))
        lo, hi = m.jnt_range[scene.jnt_ids, 0], m.jnt_range[scene.jnt_ids, 1]
        if np.any(q0 < lo - tol) or np.any(q0 > hi + tol):
            return _invalid(scene, ["initial joint state outside simulator joint limits (beyond tolerance); not clipped. "
                                    "This is the documented sim-vs-calibration blocker when the state is a real recording"],
                            seg_prov, action_record, seed, pert_desc, t0)
        scene.set_robot_q(d, q0)
        c = cfg.inputs["cube"]
        cx, cy = c["x_m"] + float(sample.object_xy_m[0]), c["y_m"] + float(sample.object_xy_m[1])
        mx, my = geom["mat_x_range_m"], geom["mat_y_range_m"]
        if not (mx[0] <= cx <= mx[1] and my[0] <= cy <= my[1]):
            return _invalid(scene, ["cube outside the mat after perturbation"], seg_prov, action_record, seed, pert_desc, t0)
        scene.place_cube(d, cx, cy, c["yaw_deg"] + float(sample.object_yaw_deg))
        m.geom_friction[scene.cube_geom, 0] = nominal_friction[scene.cube_geom, 0] * sample.friction_scale
        mujoco.mj_forward(m, d)

        thr = float(th["contact_dist_threshold_m"])
        rep0 = contact_report(scene, d, thr)
        bad0 = [k for k in th["unsafe_contact_categories"] if k in rep0]
        deep0 = [k for k in ("gripper_cube", "arm_cube", "cube_tray") if rep0.get(k, {}).get("max_penetration_m", 0) > 0.002]
        if bad0 or deep0:
            return _invalid(scene, [f"initial state is in collision: {sorted(bad0 + deep0)}"], seg_prov, action_record, seed, pert_desc, t0)

        # control schedule: cumulative rounding of 1/(hz*dt), as the existing twin
        hz, dt = contract.control_hz, float(m.opt.timestep)
        targets_deg = arr.copy()
        targets_deg[:, :5] += sample.control_noise_deg
        targets = contract.to_sim_rad(targets_deg)
        ctrl_lo, ctrl_hi = m.actuator_ctrlrange[scene.act, 0], m.actuator_ctrlrange[scene.act, 1]
        clamped = bool(np.any(targets < ctrl_lo - 1e-12) or np.any(targets > ctrl_hi + 1e-12))
        per = 1.0 / (hz * dt)
        bounds = [0] + [int(round((t + 1) * per)) for t in range(n_steps)]
        settle = int(round(float(sim["settle_s"]) / dt))
        every = max(1, int(sim["record_every_substeps"]))
        linear = contract.interpolation == "linear"
        wall = float(sim["wall_timeout_s"])
        glo, ghi = contract.gripper_range_rad

        def sample_state() -> None:
            rep = contact_report(scene, d, thr)
            rows["t_s"].append(d.time)
            rows["q_rad"].append(d.qpos[scene.qadr].copy())
            rows["ctrl"].append(d.ctrl[scene.act].copy())
            rows["qpos"].append(d.qpos.copy())
            rows["cube_pos"].append(d.qpos[scene.cube_qadr:scene.cube_qadr + 3].copy())
            rows["cube_quat"].append(d.qpos[scene.cube_qadr + 3:scene.cube_qadr + 7].copy())
            rows["cube_speed"].append(float(np.linalg.norm(d.qvel[scene.cube_dofadr:scene.cube_dofadr + 3])))
            rows["gripper_pct"].append(float((d.qpos[scene.qadr[5]] - glo) / (ghi - glo) * 100.0))
            rows["ee_pos"].append(d.site_xpos[scene.ee_site].copy())
            rows["contact_present"].append([c_ in rep for c_ in CONTACT_CATEGORIES])
            rows["contact_pen"].append([rep.get(c_, {}).get("max_penetration_m", 0.0) for c_ in CONTACT_CATEGORIES])
            rows["max_abs_qvel"].append(float(np.max(np.abs(d.qvel))))

        sample_state()
        total = bounds[-1] + settle
        prev = q0.copy()
        done = False
        for t in range(n_steps + 1):
            n = (bounds[t + 1] - bounds[t]) if t < n_steps else settle
            tgt = targets[t] if t < n_steps else targets[-1]
            for k in range(n):
                d.ctrl[scene.act] = prev + (tgt - prev) * ((k + 1) / n) if (linear and t < n_steps) else tgt
                mujoco.mj_step(m, d)
                step += 1
                if step_cb is not None:
                    step_cb(m, d, step)
                if step % every == 0 or step == total:
                    if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                        invalid_reason, done = f"non-finite state at t={d.time:.3f}s (simulator instability)", True
                        break
                    w = next((nm for nm in _WARNINGS if getattr(mujoco.mjtWarning, nm, None) is not None
                              and d.warning[int(getattr(mujoco.mjtWarning, nm))].number > 0), None)
                    if w:
                        invalid_reason, done = f"MuJoCo warning {w} at t={d.time:.3f}s (simulator instability)", True
                        break
                    sample_state()
                    if (time.monotonic() - t0 > wall) or (deadline is not None and time.monotonic() > deadline):
                        timed_out, done = True, True
                        break
            if done:
                break
            if t < n_steps:
                prev = targets[t]
    finally:
        m.geom_friction[:] = nominal_friction

    trace = {k: np.asarray(v) for k, v in rows.items()}
    trace["t_s"] = trace["t_s"].astype(float)
    trace["contact_present"] = trace["contact_present"].astype(bool)
    ta = TraceArrays(t_s=trace["t_s"], q_rad=trace["q_rad"], cube_pos=trace["cube_pos"], cube_speed=trace["cube_speed"],
                     gripper_pct=trace["gripper_pct"], contact_present=trace["contact_present"], contact_pen=trace["contact_pen"],
                     max_abs_qvel=trace["max_abs_qvel"])
    preds = {"constraints": eval_constraints(ta, m.jnt_range[scene.jnt_ids, 0], m.jnt_range[scene.jnt_ids, 1], th),
             "contacts": eval_contacts(ta, th), "lift": eval_lift(ta, geom, th), "inside_tray": eval_inside_tray(ta, geom, th),
             "release_settle": eval_release_settle(ta, th)}
    horizon_s = float(trace["t_s"][-1])
    outcome = combine_outcome(preds, horizon_s, float(sim["task_horizon_s"]), invalid_reason, timed_out)
    action_record["unmodified_after_run"] = bool(vs is None or actions_sha256(vs.actions) == vs.action_sha256)
    action_record["ctrl_range_clamp_possible"] = clamped
    action_record["control_noise_applied"] = bool(np.any(sample.control_noise_deg != 0))
    action_record["sim_targets_sha256"] = hash_obj(targets.tolist())
    return SceneRolloutResult(
        outcome=outcome, predicates=preds, trace=trace, labels=_labels(cfg, seg_prov), action_record=action_record, scene=scene.describe(),
        perturbation=pert_desc, seed=seed, detail=invalid_reason or ("wall-clock budget exceeded" if timed_out else ""),
        sim={"substeps": step, "timestep_s": dt, "control_hz": hz, "interpolation": contract.interpolation, "n_samples": int(trace["t_s"].size),
             "sample_every_substeps": every, "sim_duration_s": horizon_s, "settle_s": sim["settle_s"],
             "tracking": "MuJoCo PD position actuators (NOT the real Feetech controller)"},
        wall_s=time.monotonic() - t0)
