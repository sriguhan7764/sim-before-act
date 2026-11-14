"""ArUco detection and multi-frame camera-to-mat pose estimation.

Result transform: ``T_camera_from_mat`` (4x4) maps mat-frame points into camera coordinates.
Robot-base-to-mat calibration is NOT handled here and is never inferred.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import numpy as np

from . import geometry
from .config import markers_configured
from .util import SCHEMA_VERSION, canonical_json, new_id, read_json, sha256_bytes, utc_stamp, wall_ns


def make_detector(dictionary: str):
    import cv2

    if not hasattr(cv2.aruco, dictionary):
        raise ValueError(f"unknown ArUco dictionary '{dictionary}'")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def _detect_once(img, detector, scale: float = 1.0) -> dict[int, np.ndarray]:
    import cv2

    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = detector.detectMarkers(img)
    found: dict[int, list] = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2).astype(np.float64)
            found.setdefault(int(i), []).append((pts + 0.5) / scale - 0.5)
    return {i: v[0] for i, v in found.items() if len(v) == 1}


def detect_markers(image_bgr_or_gray: np.ndarray, detector, expected_ids=(), retry_scale: float = 3.0) -> dict[int, np.ndarray]:
    """Return {id: 4x2 float64 corners in ArUco order}. Duplicate ids are discarded as ambiguous.

    If some expected_ids are missing, one retry on an upscaled image recovers small/oblique markers; corners are
    mapped back to the original pixel grid. Retry results are only used for ids missing from the first pass.
    """
    import cv2

    img = image_bgr_or_gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = _detect_once(img, detector)
    if expected_ids and not set(expected_ids) <= set(found):
        for i, c in _detect_once(img, detector, retry_scale).items():
            if i in expected_ids and i not in found:
                found[i] = c
    return found


def draw_detections(image_bgr: np.ndarray, detections: dict[int, np.ndarray], expected_ids=()) -> np.ndarray:
    """Annotated COPY for previews; never used for stored raw data."""
    import cv2

    out = image_bgr.copy()
    for i, c in detections.items():
        colour = (0, 200, 0) if (not expected_ids or i in expected_ids) else (0, 165, 255)
        cv2.polylines(out, [c.astype(np.int32).reshape(-1, 1, 2)], True, colour, 2)
        cv2.putText(out, f"id {i}", tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return out


def _pnp(obj: np.ndarray, img: np.ndarray, K, dist):
    import cv2

    flag = cv2.SOLVEPNP_IPPE if len(obj) >= 4 else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    return rvec, tvec


def _reproj_errors(obj, img, rvec, tvec, K, dist) -> np.ndarray:
    import cv2

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def estimate_camera_from_mat(frames: list[dict[int, np.ndarray]], board: dict[int, np.ndarray], K, dist,
                             min_frames: int = 3) -> dict:
    """Joint PnP over every usable detected marker corner in every frame (camera/board are static).

    Returns dict with T_camera_from_mat, reprojection and per-frame pose-stability statistics.
    """
    import cv2

    K = np.asarray(K, np.float64)
    dist = np.asarray(dist, np.float64).reshape(-1)
    objs, imgs, per_frame, used_ids = [], [], [], set()
    for det in frames:
        ids = [i for i in det if i in board]
        if not ids:
            continue
        o = np.concatenate([board[i] for i in ids])
        p = np.concatenate([det[i] for i in ids])
        objs.append(o)
        imgs.append(p)
        used_ids.update(ids)
        if len(ids) == len(board):
            r = _pnp(o, p, K, dist)
            if r is not None:
                per_frame.append(geometry.to_T(*r))
    if not objs:
        return {"ok": False, "reason": "no configured markers detected in any frame"}
    obj_all, img_all = np.concatenate(objs), np.concatenate(imgs)
    sol = _pnp(obj_all, img_all, K, dist)
    if sol is None:
        return {"ok": False, "reason": "solvePnP failed"}
    rvec, tvec = sol
    T = geometry.to_T(rvec, tvec)
    err = _reproj_errors(obj_all, img_all, rvec, tvec, K, dist)
    per_marker = {}
    for i in sorted(used_ids):
        e = [_reproj_errors(board[i], d[i], rvec, tvec, K, dist) for d in frames if i in d]
        per_marker[str(i)] = float(np.sqrt(np.mean(np.concatenate(e) ** 2)))
    stab = {"frames_with_all_markers": len(per_frame)}
    if len(per_frame) >= 2:
        ts = np.array([p[:3, 3] for p in per_frame])
        stab["translation_std_mm"] = float(np.linalg.norm(ts.std(axis=0)) * 1000)
        stab["rotation_dev_deg_max"] = max(_rot_angle_deg(T[:3, :3], p[:3, :3]) for p in per_frame)
        stab["rotation_std_deg"] = float(np.std([_rot_angle_deg(T[:3, :3], p[:3, :3]) for p in per_frame]))
    cam_pos_mat = geometry.invert_T(T)[:3, 3]
    return {
        "ok": True, "T_camera_from_mat": T.tolist(), "naming": "p_camera = T_camera_from_mat @ p_mat",
        "camera_position_in_mat_m": cam_pos_mat.tolist(),
        "reprojection": {"rms_px": float(np.sqrt(np.mean(err ** 2))), "max_px": float(err.max()),
                         "num_corners": int(len(err)), "per_marker_rms_px": per_marker},
        "stability": stab, "num_frames_used": len(objs), "marker_ids_used": sorted(used_ids),
        "missing_marker_ids": sorted(set(board) - used_ids),
    }


def evaluate_acceptance(result: dict, cal_cfg: dict) -> list[str]:
    """Return a list of failed-check messages (empty = automatic checks pass)."""
    if not result.get("ok"):
        return [result.get("reason", "estimation failed")]
    fails = []
    if result["missing_marker_ids"]:
        fails.append(f"markers never detected: {result['missing_marker_ids']}")
    if result["reprojection"]["rms_px"] > cal_cfg["max_reprojection_rms_px"]:
        fails.append(f"reprojection RMS {result['reprojection']['rms_px']:.2f}px > {cal_cfg['max_reprojection_rms_px']}")
    st = result["stability"]
    if st["frames_with_all_markers"] < cal_cfg["min_frames_with_all_markers"]:
        fails.append(f"only {st['frames_with_all_markers']} frames with all markers (< {cal_cfg['min_frames_with_all_markers']})")
    if "translation_std_mm" in st:
        if st["translation_std_mm"] > cal_cfg["max_translation_std_mm"]:
            fails.append(f"translation std {st['translation_std_mm']:.2f}mm too high")
        if st["rotation_std_deg"] > cal_cfg["max_rotation_std_deg"]:
            fails.append(f"rotation std {st['rotation_std_deg']:.3f}deg too high")
    return fails


def _freeze(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def save_snapshot(root: str | Path, cfg: dict, camera_name: str, intrinsics: dict, result: dict,
                  auto_failures: list[str], operator_accepted: bool, operator_note: str = "") -> Path:
    """Write an immutable calibration snapshot; returns its directory. IDs embed a content hash."""
    body = {
        "schema_version": SCHEMA_VERSION, "kind": "camera_to_mat",
        "camera": camera_name, "intrinsics": intrinsics,
        "marker_config": {"dictionary": cfg["markers"]["dictionary"], "size_mm": cfg["markers"]["size_mm"],
                          "size_excludes_white_margin_confirmed":
                              cfg["markers"]["size_excludes_white_margin_confirmed"],
                          "corners": cfg["markers"]["corners"], "centers_m": cfg["mat"]["marker_centers_m"]},
        "result": result, "automatic_check_failures": auto_failures,
        "operator_accepted": bool(operator_accepted), "operator_note": operator_note,
        "created_wall_ns": wall_ns(),
        "robot_base_to_mat": None,
        "robot_base_to_mat_note": "NOT calibrated here; must be established separately and never inferred.",
    }
    h = sha256_bytes(canonical_json({k: v for k, v in body.items() if k != "created_wall_ns"}).encode())
    cal_id = f"cal_{utc_stamp()}_{h[:8]}"
    body["calibration_id"] = cal_id
    d = Path(root) / cal_id
    d.mkdir(parents=True, exist_ok=False)
    f = d / "calibration.json"
    f.write_text(canonical_json(body))
    body_hash = sha256_bytes(f.read_bytes())
    (d / "calibration.sha256").write_text(body_hash + "\n")
    _freeze(f)
    _freeze(d / "calibration.sha256")
    return d


def latest_snapshot(root: str | Path, require_accepted: bool = False) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    for d in sorted((p for p in root.iterdir() if (p / "calibration.json").exists()), reverse=True):
        if not require_accepted or read_json(d / "calibration.json").get("operator_accepted"):
            return d
    return None


def run_calibration(cfg: dict, camera, detector, n_frames: int | None = None) -> tuple[dict, dict]:
    """Collect frames from an opened camera adapter; return (result, intrinsics)."""
    missing = markers_configured(cfg)
    if missing:
        raise ValueError("cannot calibrate: unset config values: " + ", ".join(missing))
    board = geometry.board_from_config(cfg)
    desc = camera.describe()
    intr = desc.get("color_intrinsics")
    if not intr:
        raise ValueError("camera adapter provided no color intrinsics")
    K = np.array(intr["K"], float)
    dist = np.array(intr["dist"], float)
    dets = []
    for _ in range(n_frames or cfg["calibration"]["num_frames"]):
        fr = camera.read(timeout_s=2.0)
        if fr is not None:
            import cv2

            dets.append(detect_markers(cv2.cvtColor(fr.color_rgb, cv2.COLOR_RGB2BGR), detector, expected_ids=tuple(board)))
    return estimate_camera_from_mat(dets, board, K, dist), intr
