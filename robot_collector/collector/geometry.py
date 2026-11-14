"""Mat frame and ArUco marker geometry (metres).

Mat frame: origin at the midpoint of the mat edge nearest the robot; +X right (seen from the robot),
+Y away from the robot, +Z up. ``T_camera_from_mat`` maps mat-frame points into camera coordinates:
``p_cam = R @ p_mat + t``.
"""
from __future__ import annotations

import math

import numpy as np

from .config import CORNER_NAMES


def marker_object_points(center_m, size_m: float, orientation_deg: float) -> np.ndarray:
    """3D corners (4x3, mat frame) in OpenCV ArUco order: top-left, top-right, bottom-right, bottom-left.

    "Top" is the top edge of the marker as printed. ``orientation_deg`` is the counter-clockwise
    rotation (viewed from above, +Z towards the viewer) of the printed marker's "up" direction away
    from mat +Y. 0 means the printed top edge points to +Y.
    """
    h = size_m / 2.0
    local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]], dtype=np.float64)
    a = math.radians(orientation_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    xy = local @ r.T + np.array(center_m[:2])
    return np.column_stack([xy, np.full(4, float(center_m[2]))])


def board_from_config(cfg: dict) -> dict[int, np.ndarray]:
    """Map configured marker id -> 4x3 mat-frame corners. Raises if anything is unset."""
    mk = cfg["markers"]
    size = float(mk["size_mm"]) / 1000.0
    out: dict[int, np.ndarray] = {}
    for name in CORNER_NAMES:
        m = mk["corners"][name]
        if m["id"] is None or m["orientation_deg"] is None:
            raise ValueError(f"marker '{name}' id/orientation not configured")
        out[int(m["id"])] = marker_object_points(cfg["mat"]["marker_centers_m"][name], size, m["orientation_deg"])
    return out


def to_T(rvec, tvec) -> np.ndarray:
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R.T, -R.T @ t
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    return pts @ T[:3, :3].T + T[:3, 3]


def project(T_camera_from_mat: np.ndarray, K: np.ndarray, dist, pts_mat: np.ndarray) -> np.ndarray:
    import cv2

    rvec, _ = cv2.Rodrigues(T_camera_from_mat[:3, :3])
    img, _ = cv2.projectPoints(np.atleast_2d(pts_mat).astype(np.float64), rvec,
                               T_camera_from_mat[:3, 3], np.asarray(K, float), np.asarray(dist, float))
    return img.reshape(-1, 2)
