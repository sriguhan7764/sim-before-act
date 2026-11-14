"""Config, units, geometry, transforms, calibration math (synthetic geometry only)."""
import copy

import cv2
import numpy as np
import pytest

from collector import calibration as cal
from collector import geometry as g
from collector.config import ConfigError, DEFAULTS, load_config, markers_configured


def full_cfg():
    cfg = load_config({})
    cfg["markers"]["dictionary"] = "DICT_4X4_50"
    for k, (i, o) in {"near_left": (0, 0), "near_right": (1, 0), "far_left": (2, 0), "far_right": (3, 0)}.items():
        cfg["markers"]["corners"][k] = {"id": i, "orientation_deg": o}
    return cfg


def test_defaults_leave_unknowns_unset():
    cfg = load_config({})
    assert cfg["markers"]["dictionary"] is None
    assert all(c["id"] is None and c["orientation_deg"] is None for c in cfg["markers"]["corners"].values())
    assert "markers.dictionary" in markers_configured(cfg)
    assert cfg["robot"]["follower"]["port"] is None


def test_units_and_mat_bounds():
    cfg = load_config({})
    assert cfg["mat"]["width_m"] == 0.4 and cfg["markers"]["size_mm"] == 40.0
    with pytest.raises(ConfigError):  # centre given in mm by mistake -> outside the mat
        load_config({"mat": {"marker_centers_m": {"near_left": [-180, 20, 0]}}})
    with pytest.raises(ConfigError):  # 40 m marker (unit mistake)
        load_config({"markers": {"size_mm": 40000}})
    with pytest.raises(ConfigError):
        load_config({"not_a_key": 1})
    with pytest.raises(ConfigError):
        load_config({"markers": {"corners": {"near_left": {"id": 3, "orientation_deg": 45}}}})
    with pytest.raises(ConfigError):
        load_config({"markers": {"corners": {"near_left": {"id": 3, "orientation_deg": 0},
                                              "far_left": {"id": 3, "orientation_deg": 0}}}})


def test_marker_corner_ordering_and_orientation():
    p = g.marker_object_points((0.18, 0.38, 0), 0.04, 0)
    # OpenCV order TL, TR, BR, BL with printed top towards +Y
    np.testing.assert_allclose(p, [[0.16, 0.40, 0], [0.20, 0.40, 0], [0.20, 0.36, 0], [0.16, 0.36, 0]], atol=1e-12)
    p90 = g.marker_object_points((0, 0, 0), 0.04, 90)  # printed top now points to -X
    np.testing.assert_allclose(p90[0], [-0.02, -0.02, 0], atol=1e-12)
    np.testing.assert_allclose(p90[1], [-0.02, 0.02, 0], atol=1e-12)
    p180 = g.marker_object_points((0, 0, 0), 0.04, 180)
    np.testing.assert_allclose(p180[0], [0.02, -0.02, 0], atol=1e-12)
    assert np.allclose(np.linalg.norm(p[1] - p[0]), 0.04)


def test_board_uses_mat_frame_and_stays_inside_mat():
    board = g.board_from_config(full_cfg())
    allp = np.concatenate(list(board.values()))
    assert allp[:, 0].min() == pytest.approx(-0.2) and allp[:, 0].max() == pytest.approx(0.2)
    assert allp[:, 1].min() == pytest.approx(0.0) and allp[:, 1].max() == pytest.approx(0.4)  # no gap to mat edge
    assert np.all(allp[:, 2] == 0)
    with pytest.raises(ValueError):
        g.board_from_config(load_config({}))


K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
DIST = np.zeros(5)


def synthetic_pose():
    # camera ~0.6 m above the mat looking down, offset to +X side; only used as a synthetic pose
    R, _ = cv2.Rodrigues(np.array([np.pi * 0.95, 0.05, 0.02]))
    return g.to_T(cv2.Rodrigues(R)[0], np.array([-0.05, 0.12, 0.62]))


def test_transform_direction_and_inverse():
    T = synthetic_pose()
    p_mat = np.array([[0.1, 0.2, 0.0]])
    p_cam = g.transform_points(T, p_mat)
    np.testing.assert_allclose(T[:3, :3] @ p_mat[0] + T[:3, 3], p_cam[0])
    np.testing.assert_allclose(g.transform_points(g.invert_T(T), p_cam), p_mat, atol=1e-12)
    cam_in_mat = g.invert_T(T)[:3, 3]  # camera origin expressed in the mat frame
    np.testing.assert_allclose(g.transform_points(T, cam_in_mat), [[0, 0, 0]], atol=1e-12)
    assert p_cam[0, 2] > 0  # points in front of the camera


def test_pose_estimation_recovers_T_camera_from_mat():
    cfg = full_cfg()
    board = g.board_from_config(cfg)
    T = synthetic_pose()
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(12):
        frames.append({i: g.project(T, K, DIST, c) + rng.normal(0, 0.15, (4, 2)) for i, c in board.items()})
    res = cal.estimate_camera_from_mat(frames, board, K, DIST)
    assert res["ok"] and res["missing_marker_ids"] == []
    np.testing.assert_allclose(np.array(res["T_camera_from_mat"]), T, atol=2e-3)
    assert res["reprojection"]["rms_px"] < 0.5
    assert res["stability"]["frames_with_all_markers"] == 12 and res["stability"]["translation_std_mm"] < 3
    assert cal.evaluate_acceptance(res, cfg["calibration"]) == []
    assert res["camera_position_in_mat_m"][2] == pytest.approx(g.invert_T(T)[2, 3], abs=1e-3)


def test_wrong_marker_orientation_gives_bad_reprojection():
    cfg = full_cfg()
    board = g.board_from_config(cfg)
    T = synthetic_pose()
    frames = [{i: g.project(T, K, DIST, c) for i, c in board.items()}] * 12
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["markers"]["corners"]["near_left"]["orientation_deg"] = 90  # config disagrees with print
    res = cal.estimate_camera_from_mat(frames, g.board_from_config(bad_cfg), K, DIST)
    assert res["reprojection"]["rms_px"] > 1.5
    assert cal.evaluate_acceptance(res, cfg["calibration"])


def test_aruco_detection_roundtrip_and_orientation():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = cv2.aruco.generateImageMarker(d, 7, 100)
    canvas = np.full((300, 300), 255, np.uint8)
    canvas[100:200, 100:200] = m
    det = cal.detect_markers(canvas, cal.make_detector("DICT_4X4_50"))
    assert list(det) == [7]
    c = det[7]  # TL, TR, BR, BL of the printed marker
    assert c[0][0] < c[1][0] and c[0][1] < c[3][1]
    with pytest.raises(ValueError):
        cal.make_detector("DICT_NOPE")


def test_snapshot_is_immutable_and_versioned(tmp_path):
    cfg = full_cfg()
    res = {"ok": True, "T_camera_from_mat": np.eye(4).tolist()}
    d1 = cal.save_snapshot(tmp_path, cfg, "cam", {"K": K.tolist()}, res, [], True)
    f = d1 / "calibration.json"
    with pytest.raises(PermissionError):
        open(f, "w")
    assert cal.latest_snapshot(tmp_path, require_accepted=True) == d1
    d2 = cal.save_snapshot(tmp_path, cfg, "cam", {"K": K.tolist()}, res, ["x"], False)
    assert d2 != d1 and cal.latest_snapshot(tmp_path, require_accepted=True) == d1
