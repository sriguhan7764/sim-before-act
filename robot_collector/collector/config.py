"""Configuration loading and validation. All lengths are metres internally.

Config keys carry their unit in the name (``_mm``, ``_m``, ``_ms``, ``_gb``). Values that are
unknown for the physical setup (ArUco dictionary, marker IDs, device serials, ports) default to
``None`` and are never guessed.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CORNER_NAMES = ("near_left", "near_right", "far_left", "far_right")
MAT_SIZE_M = (0.400, 0.400)
DEFAULT_MARKER_CENTERS_M = {
    "near_left": (-0.180, 0.020, 0.0),
    "near_right": (0.180, 0.020, 0.0),
    "far_left": (-0.180, 0.380, 0.0),
    "far_right": (0.180, 0.380, 0.0),
}
VALID_ORIENTATIONS = (0, 90, 180, 270)

DEFAULTS: dict[str, Any] = {
    "config_version": 1,
    "mode": "diagnostic",  # diagnostic | research
    "paths": {"episodes_root": "data/episodes", "calibrations_root": "data/calibrations"},
    "cameras": [],
    "calibration_registry": None,   # path to configs/calibration_registry.yaml: active vs stale calibration files
    "robot": {
        "follower": {"backend": "none", "port": None, "baudrate": 1_000_000,
                     "joints": [], "calibration_file": None, "poll_hz": 100.0},
        "leader": {"backend": "none", "port": None, "baudrate": 1_000_000,
                   "joints": [], "calibration_file": None, "poll_hz": 100.0},
        "commands": {"backend": "none", "path": None},
        "control_mode": None,
        "controller": {},  # free-form description of the existing control loop (see configs/setup_lerobot.yaml)
    },
    "mat": {
        "width_m": MAT_SIZE_M[0], "length_m": MAT_SIZE_M[1],
        "marker_centers_m": {k: list(v) for k, v in DEFAULT_MARKER_CENTERS_M.items()},
    },
    "markers": {
        "dictionary": None,
        "size_mm": 40.0,
        "size_excludes_white_margin_confirmed": None,
        "corners": {k: {"id": None, "orientation_deg": None} for k in CORNER_NAMES},
    },
    "calibration": {
        "num_frames": 30, "min_frames_with_all_markers": 10,
        "max_reprojection_rms_px": 1.5, "max_translation_std_mm": 2.0, "max_rotation_std_deg": 0.3,
    },
    "sync": {
        "reference_camera": None,
        "max_state_tolerance_ms": 20.0,
        "max_command_tolerance_ms": 50.0,
        "max_camera_tolerance_ms": 20.0,
        "interpolation": "none",  # only "none" is implemented
    },
    "recording": {
        "queue_max_items": 256,
        "overflow_policy": "flag_and_drop",  # flag_and_drop | abort
        "flush_interval_s": 2.0,
        "min_free_disk_gb": 5.0,
        "png_compression": 1,
        "auto_export": ["rgb_joint_angles", "rgbd_motor_positions"],
    },
    # Depth quality thresholds. null = "not configured": the quantity is REPORTED and the threshold check is NOT_CHECKED.
    # Semantics are in docs/QUALITY_CHECKS.md. Any value set here is an operator choice, NOT physically validated.
    "quality_checks": {
        "max_depth_repeat_fraction": None,     # repeated_previous_depth / colour frames that carry depth
        "max_depth_age_ms": None,              # host-receipt age of the depth frame attached to a colour frame (lower bound)
        "max_consecutive_repeated_depth": None,  # longest run of colour frames reusing one depth frame
    },
    "research_gates": {
        "require_calibration_accepted": True,
        "require_marker_size_confirmed": True,
        "require_all_cameras_connected": True,
        "require_follower_connected": True,
        "require_robot_calibration_file": True,
        "require_depth": True,
        "forbid_mock": True,
    },
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise ConfigError(f"unknown config key: {path}{k}")
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict) and k not in ("marker_centers_m",):
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


CAMERA_KEYS = {"name", "backend", "serial", "device", "width", "height", "fps", "role", "enabled",
               "depth", "align_depth_to_color", "intrinsics"}


def load_config(path: str | Path | dict) -> dict:
    raw = path if isinstance(path, dict) else yaml.safe_load(Path(path).read_text()) or {}
    cfg = _merge(DEFAULTS, raw)
    cams = []
    for c in cfg["cameras"]:
        bad = set(c) - CAMERA_KEYS
        if bad:
            raise ConfigError(f"unknown camera keys: {sorted(bad)}")
        d = {"serial": None, "device": None, "role": "primary", "enabled": True, "depth": False,
             "align_depth_to_color": True, "intrinsics": None, "fps": None, "width": None, "height": None}
        d.update(c)
        for req in ("name", "backend"):
            if not d.get(req):
                raise ConfigError(f"camera missing '{req}': {c}")
        cams.append(d)
    cfg["cameras"] = cams
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["mode"] not in ("diagnostic", "research"):
        raise ConfigError("mode must be 'diagnostic' or 'research'")
    names = [c["name"] for c in cfg["cameras"]]
    if len(set(names)) != len(names):
        raise ConfigError("camera names must be unique")
    for c in cfg["cameras"]:
        if c["role"] not in ("primary", "wrist", "extra"):
            raise ConfigError(f"camera {c['name']}: bad role {c['role']}")
    ref = cfg["sync"]["reference_camera"]
    if ref is not None and ref not in names:
        raise ConfigError(f"sync.reference_camera '{ref}' is not a configured camera")
    if cfg["sync"]["interpolation"] != "none":
        raise ConfigError("sync.interpolation: only 'none' is implemented (originals are never interpolated)")
    for k in ("max_state_tolerance_ms", "max_command_tolerance_ms", "max_camera_tolerance_ms"):
        if not cfg["sync"][k] > 0:
            raise ConfigError(f"sync.{k} must be > 0")
    if cfg["recording"]["overflow_policy"] not in ("flag_and_drop", "abort"):
        raise ConfigError("recording.overflow_policy must be flag_and_drop|abort")
    if cfg["recording"]["queue_max_items"] < 1:
        raise ConfigError("recording.queue_max_items must be >= 1")
    for k, v in cfg["quality_checks"].items():
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0):
            raise ConfigError(f"quality_checks.{k} must be null or a number >= 0")
    if cfg["quality_checks"]["max_depth_repeat_fraction"] is not None and cfg["quality_checks"]["max_depth_repeat_fraction"] > 1:
        raise ConfigError("quality_checks.max_depth_repeat_fraction is a fraction in [0, 1]")
    mk = cfg["markers"]
    if not 5.0 <= float(mk["size_mm"]) <= 500.0:
        raise ConfigError("markers.size_mm outside plausible range (is the value in mm?)")
    half = float(mk["size_mm"]) / 2000.0
    mat = cfg["mat"]
    for name, c in mat["marker_centers_m"].items():
        if name not in CORNER_NAMES or len(c) != 3:
            raise ConfigError(f"bad marker centre '{name}'")
        if abs(c[0]) + half > mat["width_m"] / 2 + 1e-9 or not (half - 1e-9 <= c[1] <= mat["length_m"] - half + 1e-9):
            raise ConfigError(f"marker '{name}' extends outside the mat; check units (metres)")
    for name, m in mk["corners"].items():
        if name not in CORNER_NAMES:
            raise ConfigError(f"unknown marker corner '{name}'")
        if m["orientation_deg"] is not None and m["orientation_deg"] not in VALID_ORIENTATIONS:
            raise ConfigError(f"marker orientation must be one of {VALID_ORIENTATIONS} or null")
    ids = [m["id"] for m in mk["corners"].values() if m["id"] is not None]
    if len(set(ids)) != len(ids):
        raise ConfigError("marker IDs must be unique across corners")


def markers_configured(cfg: dict) -> list[str]:
    """Human-readable list of missing marker configuration (empty when complete)."""
    mk, missing = cfg["markers"], []
    if not mk["dictionary"]:
        missing.append("markers.dictionary")
    for k, m in mk["corners"].items():
        if m["id"] is None:
            missing.append(f"markers.corners.{k}.id")
        if m["orientation_deg"] is None:
            missing.append(f"markers.corners.{k}.orientation_deg")
    return missing
