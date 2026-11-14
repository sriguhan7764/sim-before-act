"""SYNTHETIC conversion fixture. Everything here is fabricated; none of it is a real demonstration.

Writes small master episodes (bridge-format telemetry for all six joints, a front RGB-D camera, an optional slower wrist
camera) with :class:`collector.storage.EpisodeWriter`, then a dataset v0.1 manifest and a split. Poses are INVENTED test
values. Episodes carry ``quality_flags.synthetic_test_fixture = True`` so the converter labels its output
``synthetic_fixture``. Nothing opens hardware; there is no torch/pytest import (the CLI uses it in the lerobot env).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import dataset as D
from ..adapters.base import CameraFrame
from ..joints import LEROBOT_NORMALIZED_UNITS, AngleMapping
from ..pilot import build_pilot_meta
from ..storage import EpisodeWriter
from ..util import write_json_atomic
from .contract import JOINT_NAMES, SYNTHETIC_DATA_LABEL

FRONT, WRIST = "front_rgbd", "wrist"
T0 = 20_000_000_000
INVENTED_POSE = {"x_m": 0.01, "y_m": 0.2, "yaw_deg": 0.0, "source": "manual_measurement", "sigma_mm": 2.0}   # INVENTED
JOINTS = [{"name": n, "id": i + 1, "type": "gripper" if n == "gripper" else "revolute"} for i, n in enumerate(JOINT_NAMES)]
CAL = {j["name"]: {"id": j["id"], "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000} for j in JOINTS}
REC_CFG = {"queue_max_items": 4000, "overflow_policy": "flag_and_drop", "flush_interval_s": 0.2, "min_free_disk_gb": 0.0,
           "png_compression": 1}
SYNC_CFG = {"reference_camera": FRONT, "max_state_tolerance_ms": 20.0, "max_command_tolerance_ms": 50.0,
            "max_camera_tolerance_ms": 20.0, "interpolation": "none"}


@dataclass
class EpisodeSpec:
    episode_id: str
    n_frames: int = 40
    size: tuple[int, int] = (48, 64)                 # (H, W) of the stored colour frames
    front_period_ns: int = 33_333_333
    wrist_period_ns: int = 37_500_000                # ~26.7 Hz, as measured on the real wrist camera
    state_period_ns: int = 3_000_000
    command_period_ns: int = 10_000_000
    with_wrist: bool = True
    offset: float = 0.0                              # added to every joint target/state (to make episodes differ)
    seed: int = 0
    state_invalid_frames: tuple = ()                 # front frame indices whose surrounding states are read_ok=False
    command_gap_frames: tuple = ()                   # front frame indices followed by an 80 ms command gap (> max_command_lag_ms)
    wrist_gap_frames: tuple = ()                     # front frame indices around which no wrist frame exists for > 2 wrist periods
    units: str = LEROBOT_NORMALIZED_UNITS
    accept: bool = True
    cube_id: str = "P1"
    notes: dict = field(default_factory=dict)


def _target(t_ns: int, spec: EpisodeSpec, phase: np.ndarray) -> np.ndarray:
    s = (t_ns - T0) * 1e-9
    base = np.array([10.0, -20.0, 30.0, -5.0, 15.0, 40.0])
    amp = np.array([8.0, 6.0, 9.0, 4.0, 7.0, 25.0])
    v = base + amp * np.sin(1.3 * s + phase) + spec.offset
    v[5] = float(np.clip(v[5], 0.0, 100.0))
    return v


def _color(kind: str, i: int, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 2] = 120 if kind == "front" else 40
    x = (3 + 2 * i) % max(w - 8, 1)
    img[h // 4: h // 4 + 6, x: x + 6, 0] = 220                        # "cube"-like red patch (fabricated pixels)
    img[h // 2:, : w // 3, 2] = 200                                    # "tray"-like blue patch
    img[..., 1] = (i * 5) % 255
    return img


def write_episode(root: Path, spec: EpisodeSpec) -> Path:
    phase = np.random.default_rng(spec.seed).uniform(0, 2 * np.pi, 6)
    pilot = build_pilot_meta("demonstration", cube_position_id=spec.cube_id, cube_pose=INVENTED_POSE, tray_pose=INVENTED_POSE)
    pilot["outcome_label_source"] = "operator_visual"
    streams = {FRONT: {"role": "primary", "depth": True}}
    if spec.with_wrist:
        streams[WRIST] = {"role": "wrist", "depth": False}
    meta = {"task_name": "SYNTHETIC FIXTURE", "instruction": SYNTHETIC_DATA_LABEL, "reference_camera": FRONT, "streams": streams,
            "calibration": {"camera_to_mat": None, "robot_motor_calibration_sha256": "fixture", "robot_base_to_mat": None},
            "quality_flags": {"mock_data": False, "synthetic_test_fixture": True, "synthetic_fixture": True},
            "pilot": pilot, "outcome": "success"}
    w = EpisodeWriter(root, meta, REC_CFG, SYNC_CFG, episode_id=spec.episode_id)
    for cam in streams:
        (w.dir / "streams" / cam).mkdir(exist_ok=True)
        write_json_atomic(w.dir / "streams" / cam / "camera.json",
                          {"depth_scale": 0.001, "depth_units": "metres_per_unit",
                           "color_intrinsics": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "dist": [0] * 5}})
    (w.dir / "calibration").mkdir()
    write_json_atomic(w.dir / "calibration" / "joints.json", {"follower_joints": JOINTS})
    write_json_atomic(w.dir / "calibration" / "robot_motor_calibration.json",
                      AngleMapping(JOINTS, CAL, "synthetic.json", "deadbeef").describe())

    t_end = T0 + spec.n_frames * spec.front_period_ns
    for i in range(spec.n_frames):
        t = T0 + i * spec.front_period_ns
        d = (np.arange(24 * 32).reshape(24, 32) + i).astype(np.uint16)
        f = CameraFrame(_color("front", i, spec.size), t, 0, frame_number=100 + i, device_timestamp_ms=5000.0 + i * 33.3,
                        timestamp_domain="hardware_clock", depth_native=d, depth_aligned=d.copy(), depth_frame_number=100 + i,
                        depth_device_timestamp_ms=5000.0 + i * 33.3, depth_timestamp_domain="hardware_clock")
        w.submit(FRONT, "frame", (f, {"recv_index": i, "frame_number": f.frame_number, "device_timestamp_ms": f.device_timestamp_ms,
                                      "timestamp_domain": "hardware_clock", "host_mono_ns": t, "host_wall_ns": 0,
                                      "depth_frame_number": f.depth_frame_number, "depth_device_timestamp_ms": 1.0,
                                      "depth_timestamp_domain": "hardware_clock", "extras": {}}))
    if spec.with_wrist:
        gaps = [(T0 + g * spec.front_period_ns - 2 * spec.wrist_period_ns, T0 + g * spec.front_period_ns + spec.front_period_ns)
                for g in spec.wrist_gap_frames]
        k = 0
        n_w = int((t_end - T0) // spec.wrist_period_ns)
        for j in range(n_w):
            t = T0 + j * spec.wrist_period_ns + 5_000_000
            if any(a <= t < b for a, b in gaps):
                continue
            f = CameraFrame(_color("wrist", j, spec.size), t, 0, frame_number=None)
            w.submit(WRIST, "frame", (f, {"recv_index": k, "frame_number": None, "device_timestamp_ms": None,
                                          "timestamp_domain": None, "host_mono_ns": t, "host_wall_ns": 0, "extras": {}}))
            k += 1

    bad_state = [(T0 + g * spec.front_period_ns - 60_000_000, T0 + g * spec.front_period_ns + 1_000_000) for g in spec.state_invalid_frames]
    cmd_gaps = [(T0 + g * spec.front_period_ns + 1, T0 + g * spec.front_period_ns + 80_000_000) for g in spec.command_gap_frames]
    n_states = int((t_end - T0) // spec.state_period_ns)
    for k in range(n_states):
        t = T0 + k * spec.state_period_ns
        meas = _target(t - 30_000_000, spec, phase)                    # measured lags the target (fabricated)
        ok = not any(a <= t < b for a, b in bad_state)
        raw = {n: int(2000 + 11 * v) for n, v in zip(JOINT_NAMES, meas)}
        rec = {"seq": k, "t_start_ns": t - 100_000, "t_end_ns": t + 100_000, "wall_ns": 0, "read_ok": ok, "raw": raw}
        if ok:
            rec["normalized"] = {n: float(v) for n, v in zip(JOINT_NAMES, meas)}
        w.submit("follower_state", "telemetry", rec)
    seq = 0
    for k in range(int((t_end - T0) // spec.command_period_ns)):
        t = T0 + k * spec.command_period_ns
        if any(a <= t < b for a, b in cmd_gaps):
            continue
        tgt = _target(t, spec, phase)
        lead = tgt + 0.7
        w.submit("commands", "telemetry", {
            "seq": seq, "t_start_ns": t, "t_end_ns": t, "wall_ns": 0,
            "values": {n: float(v) for n, v in zip(JOINT_NAMES, tgt)},
            "raw_goal": {n: int(2000 + 11 * v) for n, v in zip(JOINT_NAMES, tgt)},
            "units": spec.units, "leader_measurement_normalized": {n: float(v) for n, v in zip(JOINT_NAMES, lead)},
            "clipped": {n: False for n in JOINT_NAMES}, "submission_status": "submitted_not_confirmed"})
        seq += 1
    ep = w.finalize("success")
    if spec.accept:
        a = ep.parent / (ep.name + ".annotations")
        a.mkdir(exist_ok=True)
        (a / "v001.json").write_text(json.dumps({"version": 1, "outcome": "success", "label_source": "operator_visual",
                                                 "accepted_for_dataset": True, "failure_category": None}))
    return ep


def default_specs(n: int = 4, **common) -> list[EpisodeSpec]:
    return [EpisodeSpec(f"synthetic_fixture_ep_{i:03d}", offset=float(i), seed=100 + i, cube_id=f"P{i % 5 + 1}", **common)
            for i in range(n)]


def build_fixture(out_dir: str | Path, specs: list[EpisodeSpec] | None = None, split_seed: int = 7) -> dict:
    """Writes masters, ``manifest.json`` and ``split.json`` (PROVISIONAL: fewer than 30) side by side in ``out_dir``.

    ``episode_path`` in a manifest entry is relative to the manifest's directory, hence the shared directory.
    """
    out = Path(out_dir)
    root = out
    root.mkdir(parents=True, exist_ok=True)
    eps = [write_episode(root, s) for s in (specs or default_specs())]
    man = D.build_manifest(eps, name="SYNTHETIC_FIXTURE")
    split = D.make_split(man, split_seed, synthetic_fixture=True)
    mp, sp = out / "manifest.json", out / "split.json"
    write_json_atomic(mp, man)
    D.write_split(sp, split)
    return {"root": str(root), "manifest": str(mp), "split": str(sp), "episodes": [str(e) for e in eps],
            "label": SYNTHETIC_DATA_LABEL}
