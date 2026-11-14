"""Explicit outcome predicates for the cube-to-tray scene and their combination into SEPARATE outcome categories.

Every predicate returns a structured dict: ``name``, ``value`` (True/False/None), ``evaluable``, the ``numbers`` used, the
``thresholds`` applied (all ASSUMED, uncalibrated: see config ``outcome``), and a ``note``. Predicates are pure functions of
sampled arrays (``TraceArrays``) so they can be tested with constructed states.

Contact is NOT equated with failure. Gripper-cube, cube-tray and cube-mat contacts are expected. Only the categories listed in
``outcome.unsafe_contact_categories`` (default: non-gripper arm links vs mat/table, arm self-collision, non-gripper arm links vs
tray) are safety violations. Contacts are sampled (default every 10 ms), so shorter transients can be missed.

A rollout shorter than ``simulation.task_horizon_s`` has ``horizon_kind == "segment"``: observing a condition is evidence, but NOT
observing it is not evidence of failure, so task failure is never reported for segment horizons (outcome ``unknown``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import THRESHOLD_STATUS

CONTACT_CATEGORIES = ("gripper_cube", "arm_cube", "cube_tray", "cube_mat_table", "arm_mat_table", "gripper_mat_table",
                      "arm_self", "arm_tray", "gripper_tray")
OUTCOME_CATEGORIES = ("task_success", "task_failure", "safety_violation", "invalid_rollout", "timeout", "unknown")
FAILURE_REASONS = {"lift": "cube_not_lifted", "inside_tray": "not_in_tray", "release_settle": "not_released"}


def classify_pair(kind_a: str, kind_b: str) -> str | None:
    """Contact category of a geom pair from the geom 'kinds' (robot_arm, robot_gripper, cube, tray, mat, table); None = ignore."""
    s = {kind_a, kind_b}
    robot = {"robot_arm", "robot_gripper"}
    if "cube" in s and len(s) == 2:
        o = (s - {"cube"}).pop()
        return {"robot_gripper": "gripper_cube", "robot_arm": "arm_cube", "tray": "cube_tray", "mat": "cube_mat_table",
                "table": "cube_mat_table"}.get(o)
    if s <= robot:
        return "arm_self"
    if len(s & robot) == 1:
        r = (s & robot).pop()
        o = (s - robot).pop()
        if o in ("mat", "table"):
            return "arm_mat_table" if r == "robot_arm" else "gripper_mat_table"
        if o == "tray":
            return "arm_tray" if r == "robot_arm" else "gripper_tray"
    return None


def contact_report(scene, data, dist_thr: float = 0.0) -> dict[str, dict]:
    """Per-category {count, max_penetration_m} of the current contacts with geom distance <= dist_thr."""
    out: dict[str, dict] = {}
    kinds = scene.geom_kind
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > dist_thr:
            continue
        cat = classify_pair(kinds[int(c.geom1)], kinds[int(c.geom2)])
        if cat is None:
            continue
        e = out.setdefault(cat, {"count": 0, "max_penetration_m": 0.0})
        e["count"] += 1
        e["max_penetration_m"] = max(e["max_penetration_m"], max(0.0, -float(c.dist)))
    return out


@dataclass
class TraceArrays:
    t_s: np.ndarray                  # (n,) simulated time of each sample
    q_rad: np.ndarray                # (n,6) contract joints (simulator radians)
    cube_pos: np.ndarray             # (n,3) cube COM in the mat frame
    cube_speed: np.ndarray           # (n,) |linear velocity| m/s
    gripper_pct: np.ndarray          # (n,) gripper opening in contract units 0..100 (UNVALIDATED mapping)
    contact_present: np.ndarray      # (n, len(CONTACT_CATEGORIES)) bool
    contact_pen: np.ndarray          # (n, len(CONTACT_CATEGORIES)) max penetration depth in metres
    max_abs_qvel: np.ndarray         # (n,) rad/s over all DOFs
    warnings: list[str] = field(default_factory=list)


def _res(name, value, evaluable, numbers, thresholds, note="") -> dict:
    return {"name": name, "value": value, "evaluable": evaluable, "numbers": numbers,
            "thresholds": {**thresholds, "threshold_status": THRESHOLD_STATUS}, "note": note}


# ---- constraint violation -----------------------------------------------------------------------------------------------
def eval_constraints(tr: TraceArrays, joint_lo_rad: np.ndarray, joint_hi_rad: np.ndarray, th: dict) -> dict:
    """Splits into INVALID items (NaN, instability: the rollout is not usable) and VIOLATION items (joint limit, penetration)."""
    tol = math.radians(float(th["joint_limit_tolerance_deg"]))
    invalid, viol = [], []
    finite = bool(np.all(np.isfinite(tr.q_rad)) and np.all(np.isfinite(tr.cube_pos)) and np.all(np.isfinite(tr.max_abs_qvel)))
    if not finite:
        invalid.append("non_finite_state")
    invalid += [f"mujoco_warning:{w}" for w in tr.warnings]
    vmax = float(np.nanmax(tr.max_abs_qvel)) if tr.max_abs_qvel.size else 0.0
    if finite and vmax > float(th["max_abs_qvel_rad_s"]):
        invalid.append(f"instability: max |qvel| {vmax:.1f} rad/s > {th['max_abs_qvel_rad_s']}")
    over = np.maximum(tr.q_rad - joint_hi_rad, joint_lo_rad - tr.q_rad) if finite else np.zeros((0, 6))
    worst_rad = float(np.max(over)) if over.size else 0.0
    if over.size and worst_rad > tol:
        j = int(np.unravel_index(np.argmax(over), over.shape)[1])
        viol.append(f"joint_limit: joint {j} exceeds its simulator limit by {math.degrees(worst_rad):.2f} deg")
    # penetration is judged on non-cube contacts only: cube contacts are soft by design (solref) and reported separately
    idx = [i for i, c in enumerate(CONTACT_CATEGORIES) if not c.startswith(("gripper_cube", "arm_cube", "cube_"))]
    pen = float(np.max(tr.contact_pen[:, idx])) if tr.contact_pen.size else 0.0
    if pen > float(th["penetration_tolerance_m"]):
        viol.append(f"penetration: {pen*1000:.1f} mm > {float(th['penetration_tolerance_m'])*1000:.1f} mm (non-cube contact)")
    cube_pen = float(np.max(tr.contact_pen[:, [i for i in range(len(CONTACT_CATEGORIES)) if i not in idx]])) if tr.contact_pen.size else 0.0
    return {**_res("constraint_violation", bool(viol), True,
                   {"max_joint_limit_excess_deg": math.degrees(worst_rad), "max_noncube_penetration_m": pen,
                    "max_cube_contact_penetration_m": cube_pen, "max_abs_qvel_rad_s": vmax, "all_finite": finite},
                   {"joint_limit_tolerance_deg": th["joint_limit_tolerance_deg"], "penetration_tolerance_m": th["penetration_tolerance_m"],
                    "max_abs_qvel_rad_s": th["max_abs_qvel_rad_s"]}),
            "violation_items": viol, "invalid_items": invalid}


# ---- contacts -------------------------------------------------------------------------------------------------------------
def eval_contacts(tr: TraceArrays, th: dict) -> dict:
    unsafe_cfg = list(th["unsafe_contact_categories"])
    cats = {}
    for k, c in enumerate(CONTACT_CATEGORIES):
        pres = tr.contact_present[:, k] if tr.contact_present.size else np.zeros(0, bool)
        first = float(tr.t_s[np.argmax(pres)]) if pres.any() else None
        cats[c] = {"occurred": bool(pres.any()), "n_samples": int(pres.sum()), "first_time_s": first,
                   "max_penetration_m": float(tr.contact_pen[:, k].max()) if pres.any() else 0.0,
                   "unsafe_by_config": c in unsafe_cfg}
    unsafe_hit = sorted(c for c, v in cats.items() if v["occurred"] and v["unsafe_by_config"])
    return {**_res("contact_categories", bool(unsafe_hit), True, {"categories": cats, "unsafe_categories_hit": unsafe_hit},
                   {"unsafe_contact_categories": unsafe_cfg, "contact_dist_threshold_m": th["contact_dist_threshold_m"]},
                   "value = an UNSAFE category occurred; other contacts are reported, not penalised. Sampled, transients may be missed."),
            "unsafe_hit": unsafe_hit}


# ---- lift --------------------------------------------------------------------------------------------------------------------
def eval_lift(tr: TraceArrays, geom: dict, th: dict) -> dict:
    h = tr.cube_pos[:, 2] - geom["mat_top_z_m"] if tr.cube_pos.size else np.zeros(0)
    thr = float(th["lift_height_m"])
    ok = h >= thr
    return _res("cube_lifted", bool(ok.any()), bool(h.size),
                {"max_com_height_above_mat_m": float(h.max()) if h.size else None, "rest_com_height_m": geom["cube_side_m"] / 2,
                 "first_time_lifted_s": float(tr.t_s[np.argmax(ok)]) if ok.any() else None},
                {"lift_height_m": thr}, "cube COM above the mat top by >= lift_height at some sample")


# ---- inside tray -------------------------------------------------------------------------------------------------------------
def cube_in_tray_frame(p: np.ndarray, geom: dict) -> tuple[float, float, float]:
    tp = geom["tray_pose"]
    yaw = math.radians(tp["yaw_deg"])
    dx, dy = float(p[0]) - tp["x_m"], float(p[1]) - tp["y_m"]
    return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy, float(p[2]))


def point_inside_tray(p: np.ndarray, geom: dict, th: dict) -> tuple[bool, dict]:
    u, v, z = cube_in_tray_frame(p, geom)
    m = float(th["inside_tray_margin_m"])
    hx, hy = geom["tray_interior_m"][0] / 2 - m, geom["tray_interior_m"][1] / 2 - m
    zlo, zhi = geom["tray_floor_top_z_m"], geom["tray_rim_top_z_m"]
    inside = bool(abs(u) <= hx and abs(v) <= hy and zlo <= z <= zhi)
    return inside, {"u_m": u, "v_m": v, "z_m": z, "half_x_limit_m": hx, "half_y_limit_m": hy, "z_range_m": [zlo, zhi]}


def eval_inside_tray(tr: TraceArrays, geom: dict, th: dict) -> dict:
    if not tr.cube_pos.size:
        return _res("cube_inside_tray", None, False, {}, {"inside_tray_margin_m": th["inside_tray_margin_m"]})
    final, nums = point_inside_tray(tr.cube_pos[-1], geom, th)
    ever = any(point_inside_tray(p, geom, th)[0] for p in tr.cube_pos)
    return _res("cube_inside_tray", final, True, {**nums, "inside_at_end": final, "inside_at_any_sample": ever},
                {"inside_tray_margin_m": th["inside_tray_margin_m"]},
                "cube COM inside the tray interior footprint (minus margin) and below the rim top, evaluated at the final sample")


# ---- release and settle ---------------------------------------------------------------------------------------------------------
def eval_release_settle(tr: TraceArrays, th: dict) -> dict:
    thr = {"gripper_open_pct": th["gripper_open_pct"], "v_settle_m_s": th["v_settle_m_s"], "t_settle_s": th["t_settle_s"]}
    n = tr.t_s.size
    if n < 2:
        return _res("released_and_settled", None, False, {}, thr, "fewer than 2 samples")
    ok = (tr.gripper_pct > float(th["gripper_open_pct"])) & (tr.cube_speed < float(th["v_settle_m_s"]))
    dt = float(np.median(np.diff(tr.t_s)))
    run = 0
    while run < n and ok[n - 1 - run]:
        run += 1
    dur = run * dt
    gc = CONTACT_CATEGORIES.index("gripper_cube")
    in_contact = bool(tr.contact_present[n - run:, gc].any()) if run else False
    return _res("released_and_settled", bool(dur >= float(th["t_settle_s"])), True,
                {"trailing_window_s": dur, "gripper_pct_final": float(tr.gripper_pct[-1]), "cube_speed_final_m_s": float(tr.cube_speed[-1]),
                 "gripper_cube_contact_in_window": in_contact},
                thr, "gripper open beyond threshold AND cube speed below v_settle, for the trailing window of the rollout "
                     "(window must reach the final sample). Gripper opening uses the UNVALIDATED 0..100 mapping.")


# ---- combination -----------------------------------------------------------------------------------------------------------------
def combine_outcome(preds: dict, horizon_s: float, task_horizon_s: float, invalid_reason: str | None = None,
                    timed_out: bool = False) -> dict:
    """Separate categories; primary ``category`` precedence: invalid_rollout > timeout > safety_violation > task result.

    ``task_outcome`` (success / failure / unknown) is always reported separately so a rollout that reached the goal but violated
    safety is visible as both.
    """
    horizon_kind = "task" if horizon_s >= task_horizon_s else "segment"
    cons, cont = preds["constraints"], preds["contacts"]
    invalid_items = list(cons["invalid_items"]) + ([invalid_reason] if invalid_reason else [])
    safety = list(cons["violation_items"]) + [f"unsafe_contact:{c}" for c in cont["unsafe_hit"]]
    task_preds = {"lift": preds["lift"], "inside_tray": preds["inside_tray"], "release_settle": preds["release_settle"]}
    values = {k: p["value"] for k, p in task_preds.items()}
    not_evaluable = [k for k, p in task_preds.items() if not p["evaluable"]]
    reasons: list[str] = []
    if invalid_items or timed_out:
        task_outcome = "unknown"
        note = "no valid task outcome (rollout invalid or timed out)"
    elif all(v is True for v in values.values()):
        task_outcome = "task_success"
        note = ("all three task predicates observed true" + ("" if horizon_kind == "task" else
                " within a SEGMENT horizon (shorter than the assumed task horizon); not a full-task prediction"))
    elif horizon_kind == "task" and not not_evaluable:
        task_outcome = "task_failure"
        reasons = [FAILURE_REASONS[k] for k, v in values.items() if v is not True]
        note = "task horizon reached; at least one predicate not satisfied"
    else:
        task_outcome = "unknown"
        not_obs = [k for k, v in values.items() if v is not True]
        note = ("segment horizon: the rollout is shorter than the assumed task horizon, so unmet predicates "
                f"({', '.join(not_obs)}) are NOT evaluable as failures" if horizon_kind == "segment" else "predicates not evaluable")
    if invalid_items:
        category = "invalid_rollout"
    elif timed_out:
        category = "timeout"
    elif safety:
        category = "safety_violation"
    else:
        category = task_outcome
    return {"category": category, "task_outcome": task_outcome, "task_failure_reasons": reasons,
            "safety_violation": bool(safety) and not invalid_items, "safety_reasons": safety,
            "invalid_reason": invalid_items or None, "timeout": bool(timed_out), "horizon_kind": horizon_kind,
            "horizon_s": horizon_s, "task_horizon_s": task_horizon_s,
            "predicates_observed_true": [k for k, v in values.items() if v is True],
            "predicates_not_evaluable": not_evaluable,
            "predicates_unmet_but_inconclusive": ([k for k, v in values.items() if v is not True and k not in not_evaluable]
                                                  if task_outcome == "unknown" and horizon_kind == "segment" else []),
            "note": note, "threshold_status": THRESHOLD_STATUS}
