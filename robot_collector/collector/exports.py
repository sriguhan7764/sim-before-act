"""Export profiles derived from a finalised master episode. The master is only ever read.

Profiles: rgb_joint_angles, rgbd_motor_positions, full_master. Output lives in
``<episode_dir>.exports/<profile>__v<EXPORT_VERSION>/`` and is deterministic: same master + same code
=> identical manifest ``content_sha256``. Wall-clock generation time lives only in provenance.json
(excluded from the hash). Images are copied byte-for-byte (no recompression).
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np

from .joints import LEROBOT_NORMALIZED_UNITS, AngleMapping
from .util import SCHEMA_VERSION, read_json, iter_jsonl, sha256_bytes, sha256_file, wall_ns, write_json_atomic, canonical_json, jsonl_line
from .sync import depth_freshness

EXPORT_VERSION = 1
PROFILES = ("rgb_joint_angles", "rgbd_motor_positions", "full_master")


class ExportBlocked(RuntimeError):
    def __init__(self, profile: str, missing: list[str]):
        self.profile, self.missing = profile, missing
        super().__init__(f"profile '{profile}' blocked, missing required signals: {missing}")


def export_dir_for(ep: Path, profile: str) -> Path:
    ep = Path(ep)
    return ep.parent / (ep.name + ".exports") / f"{profile}__v{EXPORT_VERSION}"


class _Master:
    def __init__(self, ep: Path):
        self.ep = ep
        self.meta = read_json(ep / "episode.json")
        self.manifest = read_json(ep / "manifest.json")
        self.cams = sorted(self.meta.get("streams", {}))
        self.frames = {c: list(iter_jsonl(ep / "streams" / c / "frames.jsonl")) for c in self.cams}
        self.cam_desc = {c: read_json(ep / "streams" / c / "camera.json") for c in self.cams
                         if (ep / "streams" / c / "camera.json").exists()}
        self.state = list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))
        self.commands = list(iter_jsonl(ep / "telemetry" / "commands.jsonl"))
        self.leader = list(iter_jsonl(ep / "telemetry" / "leader_state.jsonl"))
        self.rows = list(iter_jsonl(ep / "sync" / "associations.jsonl"))
        self.sync_report = read_json(ep / "sync" / "sync_report.json")
        p = ep / "calibration" / "robot_motor_calibration.json"
        self.robot_cal = read_json(p) if p.exists() else None
        pl = ep / "calibration" / "leader_motor_calibration.json"
        self.leader_cal = read_json(pl) if pl.exists() else None
        self.mapping = (AngleMapping(self.robot_cal["joints"], self.robot_cal["calibration"],
                                     self.robot_cal.get("calibration_file"), self.robot_cal.get("calibration_sha256"))
                        if self.robot_cal else None)
        self.joints = read_json(ep / "calibration" / "joints.json")["follower_joints"] \
            if (ep / "calibration" / "joints.json").exists() else []


def _check(m: _Master, profile: str) -> list[str]:
    missing = []
    if m.meta.get("status") not in ("complete", "complete_with_drops"):
        missing.append(f"episode status is '{m.meta.get('status')}', not a complete recording")
    if not m.cams:
        missing.append("camera streams")
    if profile in ("rgb_joint_angles", "rgbd_motor_positions"):
        if not m.state:
            missing.append("follower_state telemetry")
        if not any(r["follower_state"]["status"] == "matched" for r in m.rows):
            missing.append("any camera frame associated with a valid follower state")
    if profile == "rgb_joint_angles":
        if m.mapping is None:
            missing.append("calibrated joint angles: no robot motor calibration in master (joint-angle export blocked)")
        if not any(f.get("color_file") for fr in m.frames.values() for f in fr):
            missing.append("RGB frames")
    if profile == "rgbd_motor_positions":
        if not any(f.get("depth_native_file") or f.get("depth_aligned_file") for fr in m.frames.values() for f in fr):
            missing.append("depth frames")
        ref = m.sync_report.get("reference_camera")
        if ref and not m.cam_desc.get(ref, {}).get("depth_scale"):
            missing.append(f"depth scale for camera '{ref}'")
    return missing


def _copy(src: Path, dst: Path, link: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.link(src, dst)
    else:
        shutil.copyfile(src, dst)


def _task_block(m: _Master) -> dict:
    keys = ("episode_id", "task_name", "instruction", "operator", "scene", "initial_conditions", "object_poses",
            "object_poses_provenance", "outcome", "failure_category", "notes", "status", "quality_flags", "pilot")
    return {k: m.meta.get(k) for k in keys}


def export_episode(ep: str | Path, profile: str, allow_partial: bool = False, force: bool = False,
                   link: bool = False) -> Path:
    ep = Path(ep)
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}")
    m = _Master(ep)
    missing = _check(m, profile)
    if missing and not allow_partial:
        raise ExportBlocked(profile, missing)
    out = export_dir_for(ep, profile)
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists (use --force to regenerate; the master is never touched)")
        shutil.rmtree(out)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    if profile == "full_master":
        _full(m, tmp, link)
    else:
        _lightweight(m, tmp, profile)
    extra = {"complete": not missing, "missing_required_signals": missing}
    _finish(m, tmp, profile, extra)
    os.rename(tmp, out)
    return out


def _lightweight(m: _Master, d: Path, profile: str) -> None:
    rgbd = profile == "rgbd_motor_positions"
    rgb_angles = not rgbd
    ref = m.sync_report.get("reference_camera")
    variants = ["color"] + (["depth_native", "depth_aligned"] if rgbd else [])
    depth_status_by_cam = {c: depth_freshness(fr)["status"] for c, fr in m.frames.items()} if rgbd else {}
    state_by_idx = m.state
    n_rows = len(m.rows)
    J = [j["name"] for j in m.joints]
    rev = [j["name"] for j in m.joints if j["type"] == "revolute"]
    ang = np.full((n_rows, len(rev)), np.nan)
    raw = np.full((n_rows, len(J)), -1, dtype=np.int64)
    grip = np.full(n_rows, np.nan)
    valid = np.zeros(n_rows, dtype=bool)
    t_ref = np.zeros(n_rows, dtype=np.int64)
    resid = np.full(n_rows, np.iinfo(np.int64).min, dtype=np.int64)
    cmd_arrays = {}
    for w_ in ("prev", "next"):
        cmd_arrays[w_] = {"target": np.full((n_rows, len(J)), np.nan), "raw_goal": np.full((n_rows, len(J)), -1, dtype=np.int64),
                          "valid": np.zeros(n_rows, dtype=bool), "residual_ns": np.full(n_rows, np.iinfo(np.int64).min, dtype=np.int64),
                          "target_rad": np.full((n_rows, len(rev)), np.nan)}
    lines = []
    for r_i, row in enumerate(m.rows):
        t_ref[r_i] = row["t_host_mono_ns"]
        sa = row["follower_state"]
        entry = {"sample_index": r_i, "t_host_mono_ns": row["t_host_mono_ns"], "cameras": {}, "state": {
            "association_status": sa["status"], "residual_ns": sa["residual_ns"], "t_host_mono_ns": None}}
        if sa["status"] == "matched":
            s = state_by_idx[sa["index"]]
            valid[r_i], resid[r_i] = True, sa["residual_ns"]
            entry["state"]["t_host_mono_ns"] = (s["t_start_ns"] + s["t_end_ns"]) // 2
            entry["state"]["state_seq"] = s["seq"]
            entry["state"]["master_state_index"] = sa["index"]
            if rgbd:
                entry["state"]["motor_positions_raw"] = [s["raw"][n] for n in J]
                entry["state"]["motor_names"] = J
                raw[r_i] = entry["state"]["motor_positions_raw"]
            if m.mapping is not None:
                conv = m.mapping.convert(s["raw"])
                key = "joint_angles_rad_optional" if rgbd else "joint_angles_rad"
                entry["state"][key] = conv["joint_angles_rad"]
                entry["state"]["gripper"] = conv["gripper"]
                ang[r_i] = conv["joint_angles_rad"]
                g = next(iter(conv["gripper"].values()), None)
                grip[r_i] = g["value"] if g else np.nan
        entry["commands"] = _command_blocks(m, row, r_i, J, rev, cmd_arrays)
        entry["leader"] = _leader_block(m, row)
        cams = {ref: {"index": row["frame_index"], "residual_ns": 0, "status": "reference"}}
        cams.update(row["cameras"])
        for c, a in cams.items():
            if a["status"] not in ("reference", "matched"):
                entry["cameras"][c] = {"association_status": a["status"], "residual_ns": a["residual_ns"]}
                continue
            f = m.frames[c][a["index"]]
            ce = {"association_status": a["status"], "residual_ns": a["residual_ns"], "frame_number": f.get("frame_number"),
                  "master_frame_index": a["index"], "master_recv_index": f["recv_index"],
                  "device_timestamp_ms": f.get("device_timestamp_ms"), "timestamp_domain": f.get("timestamp_domain"),
                  "host_mono_ns": f["host_mono_ns"]}
            if rgbd and (f.get("depth_frame_number") is not None or f.get("depth_native_file")):
                ce["depth_frame_number"] = f.get("depth_frame_number")
                ce["depth_device_timestamp_ms"] = f.get("depth_device_timestamp_ms")
                ce["depth_status"] = depth_status_by_cam[c][a["index"]]
                ce["depth_aligned_source"] = "derived from the native depth frame above (same frameset, rs.align(color))"
            for v in variants:
                key = f"{v}_file"
                if f.get(key):
                    rel = f"{v}/{c}/{f['recv_index']:06d}.png" if v != "color" else f"images/{c}/{f['recv_index']:06d}.png"
                    _copy(m.ep / "streams" / c / f[key], d / rel)
                    ce[f"{v}_path" if v != "color" else "image_path"] = rel
            entry["cameras"][c] = ce
        lines.append(entry)
    with open(d / "samples.jsonl", "wb") as f:
        for e in lines:
            f.write(jsonl_line(e))
    arrays = {"t_host_mono_ns": t_ref, "state_valid": valid, "state_residual_ns": resid}
    for w_, ca in cmd_arrays.items():
        for k, v in ca.items():
            if k == "target_rad" and not rgb_angles:
                continue
            arrays[f"command_{w_}_{k}"] = v
    if m.mapping is not None:
        arrays["joint_angles_rad"] = ang
        arrays["gripper_normalized_0_100"] = grip
    if rgbd:
        arrays["motor_positions_raw"] = raw
    (d / "arrays").mkdir()
    for k, v in arrays.items():
        np.save(d / "arrays" / f"{k}.npy", v)  # .npy is byte-deterministic (npz embeds mtimes)
    schema = {
        "profile": profile, "schema_version": SCHEMA_VERSION,
        "units": {"time": "nanoseconds on host CLOCK_MONOTONIC (t_host_mono_ns); device_timestamp_ms in the device's own clock domain",
                  "joint_angles_rad": "radians, relative to middle of calibrated range (LeRobot midrange convention); that zero "
                                      "is NOT a physically validated reference and signs are unvalidated (see sim_before_act/docs/CALIBRATION_AUDIT.md)",
                  "gripper": "normalized_0_100 over calibrated range; NOT radians",
                  "motor_positions_raw": "encoder ticks as returned by the Feetech register (post motor homing offset)",
                  "depth": "uint16 PNG; metres = value * depth_scale; 0 = invalid / no measurement",
                  "residual_ns": "sample_time - reference_frame_receipt_time"},
        "arrays/*.npy": {"state_valid": "false where no follower state was within tolerance (values are NaN/-1)",
                       "joint_order": rev if not rgbd else None, "motor_order": J if rgbd else None},
        "commands": {
            "separation": "commands are follower position TARGETS sent by the teleoperation process; they are never "
                          "merged into measured state and are not the leader measurement",
            "arrays": "command_{prev,next}_{target,raw_goal,valid,residual_ns[,target_rad]}; prev = latest command at or "
                      "before the reference frame receipt time, next = first command after it; unmatched rows are NaN/-1",
            "command_*_target": "source-declared units (see samples.jsonl commands.*.units). For collector.lerobot_bridge: "
                                "follower-normalised degrees (arm) / 0..100 (gripper), after the relative-target clip",
            "command_*_target_rad": "rgb_joint_angles only: degrees->radians of the revolute targets, filled only when the "
                                    "source declares the LeRobot-normalised degree convention; gripper excluded",
            "command_*_raw_goal": "Goal_Position ticks computed from the target (not read back from the motor)",
            "joint_order": J, "temporal_semantics": "no interpolation; association residuals are in command_*_residual_ns",
        },
        "joint_order": rev, "motor_order": J if rgbd else None,
        "reference_camera": ref, "association_method": m.sync_report["method"],
        "hardware_synchronisation": False,
    }
    if rgbd:
        schema["depth_variants"] = {"depth_native": "depth sensor geometry (not pixel-aligned with colour)",
                                    "depth_aligned": "depth reprojected into colour image geometry"}
    write_json_atomic(d / "schema.json", schema)
    write_json_atomic(d / "task.json", _task_block(m))
    cams = {}
    for c in m.cams:
        ds = m.cam_desc.get(c, {})
        cams[c] = {k: ds.get(k) for k in ("device", "stream_profile", "color_intrinsics", "negotiation")}
        if rgbd:
            cams[c].update({k: ds.get(k) for k in ("depth_intrinsics", "extrinsics_depth_to_color", "depth_scale", "depth_units",
                                                   "depth_native", "depth_aligned")})
    write_json_atomic(d / "cameras.json", cams)
    refs = {"camera_to_mat_calibration": m.meta["calibration"]["camera_to_mat"],
            "robot_motor_calibration_sha256": m.meta["calibration"]["robot_motor_calibration_sha256"],
            "leader_motor_calibration_sha256": (m.leader_cal or {}).get("calibration_sha256"),
            "effective_config_sha256": m.meta.get("effective_config_sha256"),
            "controller": m.meta.get("controller"),
            "robot_base_to_mat": None}
    if m.robot_cal:
        refs["robot_motor_calibration"] = m.robot_cal if rgbd else {
            k: m.robot_cal[k] for k in ("mapping", "calibration_sha256", "gripper_unit")}
    write_json_atomic(d / "calibration_refs.json", refs)
    if rgbd and m.robot_cal is None:
        write_json_atomic(d / "motor_mapping.json", {"note": "no calibration mapping available in master",
                                                      "motor_order": J})


def _command_blocks(m: _Master, row: dict, r_i: int, J: list[str], rev: list[str], arrays: dict) -> dict:
    """Per-sample command associations (latest at-or-before 'prev' and first after 'next' the reference frame).

    Commands are kept SEPARATE from measured state. Units are whatever the source declared (never relabelled);
    ``target_rad`` is filled only for the LeRobot-normalised degree convention declared by collector.lerobot_bridge.
    """
    out = {}
    for w_ in ("prev", "next"):
        ca = row.get(f"command_{w_}") or {"status": "source_absent", "index": None, "residual_ns": None}
        blk = {"association_status": ca["status"], "residual_ns": ca["residual_ns"]}
        if ca["status"] == "matched":
            c = m.commands[ca["index"]]
            vals = c.get("values") or {}
            blk.update(master_command_index=ca["index"], command_seq=c["seq"], t_host_mono_ns=c["t_start_ns"],
                       units=c.get("units"), values=vals, raw_goal=c.get("raw_goal"),
                       leader_measurement_normalized=c.get("leader_measurement_normalized"), clipped=c.get("clipped"),
                       ack=c.get("ack"))
            a = arrays[w_]
            a["valid"][r_i], a["residual_ns"][r_i] = True, ca["residual_ns"]
            if all(n in vals for n in J):
                a["target"][r_i] = [float(vals[n]) for n in J]
                if c.get("units") == LEROBOT_NORMALIZED_UNITS:
                    a["target_rad"][r_i] = [math.radians(float(vals[n])) for n in rev]
            if c.get("raw_goal") and all(n in c["raw_goal"] for n in J):
                a["raw_goal"][r_i] = [int(c["raw_goal"][n]) for n in J]
        out[w_] = blk
    return out


def _leader_block(m: _Master, row: dict) -> dict:
    la = row.get("leader_state") or {"status": "source_absent", "index": None, "residual_ns": None}
    blk = {"association_status": la["status"], "residual_ns": la["residual_ns"]}
    if la["status"] == "matched":
        s = m.leader[la["index"]]
        blk.update(master_leader_index=la["index"], leader_seq=s["seq"], raw=s.get("raw"), normalized=s.get("normalized"),
                   note="leader measurement; never substituted for the measured follower state")
    return blk


def _full(m: _Master, d: Path, link: bool) -> None:
    for rel in m.manifest["files"]:
        _copy(m.ep / rel, d / rel, link)
    _copy(m.ep / "manifest.json", d / "master_manifest.json")


def _finish(m: _Master, d: Path, profile: str, extra: dict) -> None:
    files = {}
    for p in sorted(d.rglob("*")):
        if p.is_file():
            files[p.relative_to(d).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    content = sha256_bytes(canonical_json(files).encode())
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "export", "profile": profile, "export_version": EXPORT_VERSION,
                "parent": {"episode_id": m.meta["episode_id"], "manifest_sha256": sha256_file(m.ep / "manifest.json"),
                           "status": m.meta["status"], "mock": bool(m.manifest.get("mock"))},
                "modalities": _modalities(profile), "files": files, "content_sha256": content, **extra}
    write_json_atomic(d / "manifest.json", manifest)
    write_json_atomic(d / "provenance.json", {"generated_wall_ns": wall_ns(), "note": "excluded from content_sha256",
                                              "software": m.meta.get("software")})


def _modalities(profile: str) -> list[str]:
    return {"rgb_joint_angles": ["rgb", "joint_angles_rad", "gripper", "timestamps", "task_metadata"],
            "rgbd_motor_positions": ["rgb", "depth_native", "depth_aligned", "motor_positions_raw",
                                     "joint_angles_rad_optional", "timestamps", "task_metadata"],
            "full_master": ["everything in master"]}[profile]


def load_export(path: str | Path) -> dict:
    """Independent reader: needs only the export directory."""
    p = Path(path)
    out = {"manifest": read_json(p / "manifest.json")}
    if out["manifest"]["profile"] == "full_master":
        out["episode"] = read_json(p / "episode.json")
        return out
    out["schema"] = read_json(p / "schema.json")
    out["samples"] = list(iter_jsonl(p / "samples.jsonl"))
    out["arrays"] = {f.stem: np.load(f) for f in sorted((p / "arrays").glob("*.npy"))}
    out["task"] = read_json(p / "task.json")
    out["cameras"] = read_json(p / "cameras.json")
    return out
