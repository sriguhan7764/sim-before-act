"""Visual outputs for a scene rollout, written without matplotlib (cv2 + numpy only).

* ``plot_rollout`` - trajectory / predicate-timeline plot drawn with cv2 (always available).
* ``render_contact_sheet`` - MuJoCo offscreen contact sheet (PNG, optional MP4) produced by a subprocess that tries the GL
  backends egl -> osmesa -> glfw; falls back to ``None`` (the cv2 plot is still written) when no backend works.

Every image carries the scene label (SYNTHETIC DEMO ... / PHYSICAL ...) and the not-a-calibrated-twin statement.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import SceneConfig
from .outcomes import CONTACT_CATEGORIES
from .runner import SceneRolloutResult

W, H = 1100, 820
BLACK, GREY, RED, GREEN, BLUE, ORANGE = (0, 0, 0), (200, 200, 200), (0, 0, 220), (40, 150, 40), (200, 80, 20), (0, 140, 255)
_SRC = Path(__file__).resolve().parents[3]          # .../src (contains the sim_before_act package)


def _text(img, s, org, scale=0.42, col=BLACK, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def _series_panel(img, rect, t, y, title, ylim, hlines=(), col=BLUE):
    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0, y0), (x1, y1), BLACK, 1)
    _text(img, title, (x0 + 4, y0 - 6))
    t0, t1 = float(t[0]), max(float(t[-1]), float(t[0]) + 1e-9)
    lo, hi = ylim
    px = lambda tt: int(x0 + (tt - t0) / (t1 - t0) * (x1 - x0))
    py = lambda yy: int(y1 - (min(max(yy, lo), hi) - lo) / (hi - lo) * (y1 - y0))
    for val, name, c in hlines:
        cv2.line(img, (x0, py(val)), (x1, py(val)), c, 1, cv2.LINE_AA)
        _text(img, f"{name}={val:g}", (x1 - 150, py(val) - 3), 0.35, c)
    pts = np.array([[px(a), py(b)] for a, b in zip(t, y)], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
    _text(img, f"{lo:g}", (x0 - 38, y1), 0.33)
    _text(img, f"{hi:g}", (x0 - 38, y0 + 10), 0.33)
    _text(img, f"t: {t0:.2f}..{t1:.2f} s", (x0 + 4, y1 + 13), 0.33)


def plot_rollout(result: SceneRolloutResult, path: str | Path) -> Path:
    """Trajectory + predicate-timeline plot. Works for any result that has a trace; for an invalid rollout a label/verdict page."""
    img = np.full((H, W, 3), 255, np.uint8)
    lab = result.labels
    cv2.rectangle(img, (0, 0), (W, 44), (230, 230, 255), -1)
    _text(img, lab["scene"][:170], (6, 17), 0.46, RED, 1)
    _text(img, "CAD geometry; thresholds ASSUMED/uncalibrated; MuJoCo PD position targets, NOT the real Feetech controller", (6, 36), 0.4, BLACK)
    o = result.outcome
    _text(img, f"outcome: {o['category']}   task_outcome: {o['task_outcome']}   horizon: {o['horizon_kind']} "
               f"({o['horizon_s']:.2f}s of assumed task {o['task_horizon_s']}s)   seed={result.seed}", (6, 62), 0.45, BLACK, 1)
    _text(img, f"segment: {str(lab['segment'])[:150]}", (6, 80), 0.36, GREEN)
    if not result.trace:
        _text(img, f"no trace: {result.detail[:200]}", (6, 110), 0.5, RED, 1)
        cv2.imwrite(str(path), img)
        return Path(path)

    tr, sc, p = result.trace, result.scene, result.predicates
    t, th = tr["t_s"], p["lift"]["thresholds"]
    g = sc["geometry"]
    _series_panel(img, (60, 110, 560, 230), t, tr["cube_pos"][:, 2], "cube COM height above mat [m] (z)",
                  (0.0, max(0.12, float(tr["cube_pos"][:, 2].max()) * 1.1)),
                  [(th["lift_height_m"], "lift_height", ORANGE), (g["tray_rim_top_z_m"], "tray_rim_top", BLUE)])
    ths = p["release_settle"]["thresholds"]
    _series_panel(img, (60, 270, 560, 390), t, tr["gripper_pct"], "gripper opening [%, UNVALIDATED 0..100 map]", (-15, 105),
                  [(ths["gripper_open_pct"], "open_thr", ORANGE)], GREEN)
    _series_panel(img, (60, 430, 560, 550), t, tr["cube_speed"], "cube COM speed [m/s]",
                  (0.0, max(0.1, float(tr["cube_speed"].max()) * 1.1)), [(ths["v_settle_m_s"], "v_settle", ORANGE)], RED)
    # contact timeline
    x0, y0, x1 = 60, 590, 560
    _text(img, "contact categories (sampled; red = UNSAFE by config)", (x0 + 4, y0 - 6))
    unsafe = set(p["contacts"]["thresholds"]["unsafe_contact_categories"])
    rh = 15
    t0, t1 = float(t[0]), max(float(t[-1]), float(t[0]) + 1e-9)
    for k, c in enumerate(CONTACT_CATEGORIES):
        yy = y0 + k * rh
        cv2.rectangle(img, (x0, yy), (x1, yy + rh - 2), GREY, 1)
        _text(img, c, (x1 + 6, yy + 11), 0.34, RED if c in unsafe else BLACK)
        pres = tr["contact_present"][:, k]
        for i in np.flatnonzero(pres):
            xa = int(x0 + (t[i] - t0) / (t1 - t0) * (x1 - x0))
            cv2.line(img, (xa, yy + 1), (xa, yy + rh - 3), RED if c in unsafe else BLUE, 1)

    # top-down trajectory (mat frame)
    ox, oy, px_per_m = 700, 110, 560.0

    def to_px(x, y):
        return int(ox + 160 + x * px_per_m), int(oy + 330 - y * px_per_m)
    _text(img, "top-down, mat frame (x right, y away from robot)", (ox - 5, oy - 6))
    a, b = to_px(*g["mat_x_range_m"][:1], 0.0), to_px(g["mat_x_range_m"][1], g["mat_y_range_m"][1])
    cv2.rectangle(img, (a[0], b[1]), (b[0], a[1]), (170, 170, 170), 1)
    tp, (ix, iy) = g["tray_pose"], g["tray_interior_m"]
    yaw = np.deg2rad(tp["yaw_deg"])
    corners = []
    for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        u, v = su * ix / 2, sv * iy / 2
        corners.append(to_px(tp["x_m"] + np.cos(yaw) * u - np.sin(yaw) * v, tp["y_m"] + np.sin(yaw) * u + np.cos(yaw) * v))
    cv2.polylines(img, [np.array(corners, np.int32)], True, BLUE, 2, cv2.LINE_AA)
    bx, by = to_px(g["base_to_mat"]["x_m"], g["base_to_mat"]["y_m"])
    cv2.drawMarker(img, (bx, by), BLACK, cv2.MARKER_TRIANGLE_UP, 12, 2)
    _text(img, "robot base (INVENTED pose)" if "SYNTHETIC" in lab["scene"] else "robot base", (bx + 8, by + 4), 0.33)
    cp = np.array([to_px(x, y) for x, y in tr["cube_pos"][:, :2]], np.int32)
    ep = np.array([to_px(x, y) for x, y in tr["ee_pos"][:, :2]], np.int32)
    cv2.polylines(img, [ep], False, GREEN, 1, cv2.LINE_AA)
    cv2.polylines(img, [cp], False, RED, 2, cv2.LINE_AA)
    cv2.circle(img, tuple(int(v) for v in cp[0]), 5, RED, 1)
    cv2.circle(img, tuple(int(v) for v in cp[-1]), 5, RED, -1)
    _text(img, "red: cube COM path (o start, filled end)", (ox - 5, oy + 360), 0.36, RED)
    _text(img, "green: end-effector site path; blue: tray interior", (ox - 5, oy + 376), 0.36, GREEN)

    # predicate verdicts
    y = 665
    for key in ("constraints", "contacts", "lift", "inside_tray", "release_settle"):
        r = p[key]
        verdict = "n/a" if not r["evaluable"] else str(r["value"])
        _text(img, f"{r['name']}: {verdict} (evaluable={r['evaluable']})", (700, y), 0.4, BLACK)
        y += 18
    _text(img, f"not evaluable as failure at this horizon: {o['predicates_unmet_but_inconclusive']}", (700, y + 4), 0.34, ORANGE)
    _text(img, f"safety_violation={o['safety_violation']}  reasons={o['safety_reasons']}"[:90], (700, y + 20), 0.34, RED)
    cv2.imwrite(str(path), img)
    return Path(path)


def _worker_cfg(cfg: SceneConfig) -> dict:
    raw = json.loads(json.dumps(cfg.raw))
    raw["robot"]["model_xml"] = str(cfg.robot_model_xml())
    return raw


def render_contact_sheet(result: SceneRolloutResult, cfg: SceneConfig, out_dir: str | Path, stem: str, *, video: bool = False,
                         backends=("egl", "osmesa", "glfw"), timeout_s: float = 120.0) -> dict:
    """-> {"renderer": backend|None, "png": Path|None, "video": Path|None, "attempts": [...]}. Never raises on a missing GL backend."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempts: list[str] = []
    if not result.trace:
        return {"renderer": None, "png": None, "video": None, "attempts": ["no trace to render"]}
    cfg_json, npz = out / f"{stem}_scenecfg.json", out / f"{stem}_trace.npz"
    cfg_json.write_text(json.dumps(_worker_cfg(cfg), indent=1, sort_keys=True))
    np.savez_compressed(npz, **result.trace)
    png, mp4 = out / f"{stem}_contact_sheet.png", out / f"{stem}_replay.mp4"
    for gl in backends:
        env = {**os.environ, "MUJOCO_GL": gl, "PYTHONPATH": f"{_SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        cmd = [sys.executable, "-m", "sim_before_act.twin.scene.render_worker", str(cfg_json), str(npz), str(png)]
        if video:
            cmd += ["--video", str(mp4)]
        try:
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s, cwd=str(out))
        except subprocess.TimeoutExpired:
            attempts.append(f"{gl}: timeout")
            continue
        if p.returncode == 0 and png.exists():
            attempts.append(f"{gl}: ok")
            cfg_json.unlink(missing_ok=True)
            return {"renderer": gl, "png": png, "video": mp4 if video and mp4.exists() else None, "attempts": attempts}
        attempts.append(f"{gl}: failed ({(p.stderr.strip().splitlines() or ['?'])[-1][:120]})")
    cfg_json.unlink(missing_ok=True)
    return {"renderer": None, "png": None, "video": None, "attempts": attempts}
