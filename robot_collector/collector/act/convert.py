"""Dataset v0.1 manifest + split -> training-ready ACT dataset (OFFLINE; masters are only read).

Layout of ``<out>/`` (this package's OWN documented layout, NOT a LeRobotDataset v3 on disk; ``adapter.py`` feeds
``ACTPolicy.forward`` directly - see docs/ACT_CONVERSION.md):

  conversion.json           schema, config, lineage (manifest/split/episode hashes), labels, per-episode summary, file hashes
  steps.npz                 one row per KEPT control step (state, action, retained source columns, lineage indices)
  normalization_stats.json  mean/std fitted on the TRAIN split only (+ sha256)
  image_index.json          pixel-level sha256 of every converted image
  images/<cam>/<episode_id>/<frame_index>.png   RGB, optionally resized (lossless PNG)
  conversion_audit.json/.md per-episode kept/dropped counts by reason, residuals, depth/wrist handling, lineage

Nothing is interpolated, filled or clipped: a step with no valid match is DROPPED and counted; values are copied byte-exact
from the master telemetry (float64) in the order ``JOINT_NAMES``.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .. import dataset as D
from ..joints import LEROBOT_NORMALIZED_UNITS
from ..sync import depth_freshness, state_time
from ..util import canonical_json, iter_jsonl, read_json, sha256_bytes, sha256_file, write_json_atomic
from . import align
from .contract import (ACTION_SIGNAL_DEFAULT, ACTION_SIGNALS, CONVERSION_SCHEMA, JOINT_NAMES, KEY_ACTION, KEY_IMAGE_PREFIX,
                       KEY_STATE, N_JOINTS, PROVISIONAL_LABEL, SYNTHETIC_DATA_LABEL, UNITS, sha256_obj)
from .load import (CONVERSION_FILE, IMAGE_INDEX_FILE, SPLIT_CODES, STATS_FILE, STEPS_FILE, Converted, arrays_sha256, stats_hash)

WRIST_POLICIES = ("drop_step", "mark_missing_with_mask")
STATS_SCHEMA = "act-normalization-v0.1"
MIN_STD = 1e-6


class ConversionError(RuntimeError):
    """A refusal: the requested conversion would be unsound. Nothing is written."""


class ProvisionalSplitError(ConversionError):
    pass


class NotTrainingEligibleError(ConversionError):
    pass


@dataclass
class ConvertConfig:
    front_stream: str | None = None            # None -> episode.json reference_camera
    wrist_stream: str | None = None            # None -> no wrist camera in the converted dataset
    image_size: tuple[int, int] | None = None  # (H, W); None -> native size
    max_state_age_ms: float = 40.0
    max_command_lag_ms: float = 50.0
    wrist_max_age_ms: float = 50.0
    wrist_max_reuse: int = 2
    wrist_missing_policy: str = "drop_step"
    action_signal: str = ACTION_SIGNAL_DEFAULT
    allow_signal_substitution: bool = False
    chunk_size: int = 100
    n_action_steps: int = 100
    min_kept_steps_per_episode: int = 1
    verify_master_files: bool = True
    extra: dict = field(default_factory=dict)

    def validate(self) -> "ConvertConfig":
        for k in ("max_state_age_ms", "max_command_lag_ms", "wrist_max_age_ms"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ConversionError(f"{k} must be a number >= 0")
        if self.wrist_missing_policy not in WRIST_POLICIES:
            raise ConversionError(f"wrist_missing_policy must be one of {WRIST_POLICIES}")
        if self.action_signal not in ACTION_SIGNALS:
            raise ConversionError(f"action_signal must be one of {sorted(ACTION_SIGNALS)}")
        if self.action_signal != ACTION_SIGNAL_DEFAULT and not self.allow_signal_substitution:
            raise ConversionError(f"action_signal={self.action_signal!r} is a SUBSTITUTE for the commanded follower target; "
                                  "set allow_signal_substitution: true to proceed (it is recorded in the audit)")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ConversionError("need 1 <= n_action_steps <= chunk_size")
        if self.wrist_max_reuse < 1:
            raise ConversionError("wrist_max_reuse must be >= 1")
        if self.image_size is not None:
            h, w = self.image_size
            if int(h) < 1 or int(w) < 1:
                raise ConversionError("image_size must be positive (H, W)")
            self.image_size = (int(h), int(w))
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_size"] = list(self.image_size) if self.image_size else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConvertConfig":
        known = {k for k in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ConversionError(f"unknown conversion config keys: {unknown}")
        d = dict(d)
        if d.get("image_size") is not None:
            d["image_size"] = tuple(d["image_size"])
        return cls(**d).validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConvertConfig":
        doc = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(doc.get("convert", doc))


# ------------------------------------------------------------------ small helpers
def _vec(d, names=JOINT_NAMES):
    """6 finite floats in joint order, or None. Booleans and missing joints are invalid."""
    if not isinstance(d, dict):
        return None
    out = []
    for n in names:
        v = d.get(n)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            return None
        out.append(float(v))
    return out


def _ivec(d, names=JOINT_NAMES):
    if not isinstance(d, dict):
        return None
    out = []
    for n in names:
        v = d.get(n)
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        out.append(int(v))
    return out


def read_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(img: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size is None or img.shape[:2] == tuple(size):
        return img
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def write_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise ConversionError(f"cannot write {path}")


def pixel_sha256(img: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(img, dtype=np.uint8).tobytes())


class MasterFiles:
    """Verifies each master file we read against the master's own ``manifest.json`` (detects edits after finalisation)."""

    def __init__(self, ep_dir: Path, enabled: bool):
        self.dir, self.enabled, self._ok = ep_dir, enabled, set()
        self.files = read_json(ep_dir / "manifest.json")["files"] if enabled else {}

    def check(self, rel: str) -> None:
        if not self.enabled or rel in self._ok:
            return
        rec = self.files.get(rel)
        if rec is None:
            raise ConversionError(f"{self.dir.name}: {rel} is not listed in the master manifest")
        p = self.dir / rel
        if not p.exists() or sha256_file(p) != rec["sha256"]:
            raise ConversionError(f"{self.dir.name}: {rel} is missing or differs from the master manifest hash")
        self._ok.add(rel)


def _count(d: dict, k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


# ------------------------------------------------------------------ one episode
def _process_episode(er: D.EpisodeReader, entry: dict, split_name: str, ep_index: int, cfg: ConvertConfig, out: Path,
                     image_hashes: dict, img_sums: dict | None) -> tuple[dict, dict]:
    """Returns (rows as dict of lists, audit dict). Raises ConversionError for episode-level problems."""
    eid = entry["entry_id"]
    meta = er.meta
    mf = MasterFiles(er.dir, cfg.verify_master_files)
    streams = meta.get("streams") or {}
    front = cfg.front_stream or meta.get("reference_camera")
    if front is None or front not in streams:
        raise ConversionError(f"{eid}: front camera stream {front!r} not found among {sorted(streams)}")
    wrist = cfg.wrist_stream
    wrist_present = bool(wrist) and wrist in streams
    cams = {"front": front, **({"wrist": wrist} if wrist else {})}

    for rel in ("telemetry/follower_state.jsonl", "telemetry/commands.jsonl", f"streams/{front}/frames.jsonl"):
        mf.check(rel)
    if wrist_present:
        mf.check(f"streams/{wrist}/frames.jsonl")
    states, cmds = list(er.follower_states()), list(er.commands())
    fframes = list(er.frames(front))
    wframes = list(er.frames(wrist)) if wrist_present else []
    if not states or not cmds or not fframes:
        raise ConversionError(f"{eid}: missing follower_state / commands / front frames (have {len(states)}/{len(cmds)}/{len(fframes)})")

    units_seen = sorted({str(c.get("units")) for c in cmds})
    if units_seen != [LEROBOT_NORMALIZED_UNITS]:
        raise ConversionError(f"{eid}: commands are not in the bridge/LeRobot-normalised format (units seen: {units_seen}); "
                              "e.g. Feetech Goal_Position ticks from an older recording cannot be used as a degrees/0-100 target")
    status_seen = sorted({str(c.get("submission_status")) for c in cmds})

    s_t = [state_time(r) for r in states]
    c_t = [r["t_start_ns"] for r in cmds]
    f_t = [f["host_mono_ns"] for f in fframes]
    w_t = [f["host_mono_ns"] for f in wframes]
    for name, ts in (("follower_state", s_t), ("commands", c_t), ("front frames", f_t), ("wrist frames", w_t)):
        if not align.is_non_decreasing(ts):
            raise ConversionError(f"{eid}: {name} timestamps are not monotonic; refusing (no reordering/repair)")
    rec_idx = [f["recv_index"] for f in fframes]
    if any(b <= a for a, b in zip(rec_idx, rec_idx[1:])):
        raise ConversionError(f"{eid}: front recv_index not strictly increasing")

    s_vec = [_vec(r.get("normalized")) if r.get("read_ok") is True else None for r in states]
    s_valid = [v is not None for v in s_vec]
    if not any(s_valid):
        raise ConversionError(f"{eid}: no follower_state record has read_ok and a complete `normalized` vector; "
                              "the measured state is never replaced by leader/commanded values")
    sig_key = "values" if cfg.action_signal == ACTION_SIGNAL_DEFAULT else "leader_measurement_normalized"
    c_act = [_vec(r.get(sig_key)) for r in cmds]
    c_raw = [_ivec(r.get("raw_goal")) for r in cmds]
    c_lead = [_vec(r.get("leader_measurement_normalized")) for r in cmds]
    c_valid = [a is not None and g is not None for a, g in zip(c_act, c_raw)]
    w_valid = [bool(f.get("color_file")) for f in wframes]

    ms = 1e6
    max_s, max_c, max_w = int(cfg.max_state_age_ms * ms), int(cfg.max_command_lag_ms * ms), int(cfg.wrist_max_age_ms * ms)
    dropped: dict[str, int] = {}
    dropped_steps: list[list] = []
    masked_reasons: dict[str, int] = {}
    rows: dict[str, list] = {k: [] for k in (
        "episode_idx", "split_code", "grid_index", "t_ref_ns", "state", "action", "raw_goal", "leader_measurement", "leader_valid",
        "command_seq", "state_seq", "front_frame_index", "front_recv_index", "front_frame_number", "wrist_frame_index",
        "wrist_recv_index", "wrist_valid", "state_t_ns", "command_t_ns", "state_age_ns", "command_lag_ns", "wrist_age_ns")}
    wrist_use: dict[int, int] = {}
    state_age, cmd_lag, wrist_age = [], [], []
    n_state_fallback = 0
    img_cache: dict[tuple[str, int], np.ndarray | None] = {}
    committed: set[tuple[str, int]] = set()

    def load_image(cam: str, stream: str, frames: list, fi: int):
        """Read + resize one colour frame (nothing is written until the step is accepted). Keeps the newest frame per camera."""
        key = (cam, fi)
        if key in img_cache:
            return img_cache[key]
        rel = f"streams/{stream}/{frames[fi]['color_file']}"
        mf.check(rel)
        img = read_rgb(er.dir / rel)
        img = None if img is None else resize_rgb(img, cfg.image_size)
        for k in [k for k in img_cache if k[0] == cam]:
            del img_cache[k]
        img_cache[key] = img
        return img

    def commit_image(cam: str, stream: str, frames: list, fi: int):
        key = (cam, fi)
        if key in committed:
            return
        img = img_cache[key]
        dst = f"images/{cam}/{eid}/{fi:06d}.png"
        write_rgb(out / dst, img)
        image_hashes[dst] = {"pixel_sha256": pixel_sha256(img), "shape": list(img.shape),
                             "source_file": f"streams/{stream}/{frames[fi]['color_file']}"}
        if split_name == "train":
            x = img.astype(np.float64) / 255.0
            img_sums[(eid, cam, fi)] = (x.sum(axis=(0, 1)), (x * x).sum(axis=(0, 1)), img.shape[0] * img.shape[1])
        committed.add(key)

    def drop(fi, grid, reason):
        _count(dropped, reason)
        dropped_steps.append([int(grid), reason])

    for fi, f in enumerate(fframes):
        t, grid = f["host_mono_ns"], f["recv_index"]
        si, sage, sst, skipped = align.latest_valid_at_or_before(s_t, s_valid, t, max_s)
        if si is None:
            drop(fi, grid, "no_state_at_or_before_ref" if sst == "none_at_or_before" else "state_too_old")
            continue
        ci, clag, cst = align.first_at_or_after(c_t, t, max_c)
        if ci is None:
            drop(fi, grid, "no_command_at_or_after_ref" if cst == "none_at_or_after" else "command_lag_too_large")
            continue
        if not c_valid[ci]:
            drop(fi, grid, "command_invalid")
            continue
        if not f.get("color_file"):
            drop(fi, grid, "front_image_missing")
            continue
        if load_image("front", front, fframes, fi) is None:
            drop(fi, grid, "front_image_unreadable")
            continue
        wi, wage, wreason = None, None, None
        if wrist:
            if not wrist_present:
                wreason = "wrist_stream_absent"
            else:
                wi, wage, wst, _ = align.latest_valid_at_or_before(w_t, w_valid, t, max_w)
                if wi is None:
                    wreason = "wrist_no_frame_at_or_before_ref" if wst == "none_at_or_before" else "wrist_frame_too_old"
                elif wrist_use.get(wi, 0) >= cfg.wrist_max_reuse:
                    wi, wreason = None, "wrist_reuse_exceeded"
                elif load_image("wrist", wrist, wframes, wi) is None:
                    wi, wreason = None, "wrist_image_unreadable"
            if wreason and cfg.wrist_missing_policy == "drop_step":
                drop(fi, grid, wreason)
                continue
            if wreason:
                _count(masked_reasons, wreason)
                wi, wage = None, None
            else:
                wrist_use[wi] = wrist_use.get(wi, 0) + 1
        n_state_fallback += int(skipped > 0)
        state_age.append(sage / ms)
        cmd_lag.append(clag / ms)
        if wage is not None:
            wrist_age.append(wage / ms)
        commit_image("front", front, fframes, fi)
        if wi is not None:
            commit_image("wrist", wrist, wframes, wi)
        lead = c_lead[ci]
        rows["episode_idx"].append(ep_index)
        rows["split_code"].append(SPLIT_CODES[split_name])
        rows["grid_index"].append(grid)
        rows["t_ref_ns"].append(t)
        rows["state"].append(s_vec[si])
        rows["action"].append(c_act[ci])
        rows["raw_goal"].append(c_raw[ci])
        rows["leader_measurement"].append(lead if lead is not None else [float("nan")] * N_JOINTS)
        rows["leader_valid"].append(lead is not None)
        rows["command_seq"].append(int(cmds[ci]["seq"]))
        rows["state_seq"].append(int(states[si]["seq"]))
        rows["front_frame_index"].append(fi)
        rows["front_recv_index"].append(int(f["recv_index"]))
        rows["front_frame_number"].append(-1 if f.get("frame_number") is None else int(f["frame_number"]))
        rows["wrist_frame_index"].append(-1 if wi is None else wi)
        rows["wrist_recv_index"].append(-1 if wi is None else int(wframes[wi]["recv_index"]))
        rows["wrist_valid"].append(wi is not None)
        rows["state_t_ns"].append(s_t[si])
        rows["command_t_ns"].append(c_t[ci])
        rows["state_age_ns"].append(sage)
        rows["command_lag_ns"].append(clag)
        rows["wrist_age_ns"].append(-1 if wage is None else wage)

    n_kept = len(rows["grid_index"])
    if n_kept < cfg.min_kept_steps_per_episode:
        raise ConversionError(f"{eid}: only {n_kept} control steps survived alignment (dropped: {dropped})")

    st, ac = np.asarray(rows["state"]), np.asarray(rows["action"])
    dts = [(b - a) / ms for a, b, ia, ib in zip(f_t, f_t[1:], rec_idx, rec_idx[1:]) if ib == ia + 1]
    fn = [f["frame_number"] for f in fframes if f.get("frame_number") is not None]
    reused = {i: n for i, n in wrist_use.items() if n > 1}
    cseq = rows["command_seq"]
    audit = {
        "episode_id": eid, "split": split_name, "front_stream": front, "wrist_stream": wrist,
        "n_front_frames": len(fframes), "n_kept": n_kept, "n_dropped": sum(dropped.values()),
        "dropped_by_reason": dict(sorted(dropped.items())), "dropped_steps": dropped_steps,
        "kept_fraction": n_kept / len(fframes),
        "association_ms": {"state_age": align.stats(state_age), "command_lag": align.stats(cmd_lag),
                           "wrist_age": align.stats(wrist_age), "front_step_dt": align.stats(dts)},
        "state_rule": f"latest valid follower state at/before t_ref, age <= {cfg.max_state_age_ms} ms",
        "command_rule": f"first command stamped at/after t_ref, lag <= {cfg.max_command_lag_ms} ms",
        "steps_using_older_valid_state_because_newest_read_failed": n_state_fallback,
        "state_records": {"total": len(states), "invalid_or_failed": len(states) - sum(s_valid)},
        "command_records": {"total": len(cmds), "invalid": len(cmds) - sum(c_valid), "submission_status_seen": status_seen,
                            "kept_rows_sharing_a_command_with_previous_row": sum(1 for a, b in zip(cseq, cseq[1:]) if a == b),
                            "commands_skipped_between_kept_rows": sum(max(0, b - a - 1) for a, b in zip(cseq, cseq[1:]) if b > a)},
        "front_recv_index_gaps": (rec_idx[-1] + 1 - len(rec_idx)) if rec_idx else 0,
        "front_device_frame_number_gaps": sum(b - a - 1 for a, b in zip(fn, fn[1:]) if b - a > 1),
        "wrist": {"configured": bool(wrist), "stream_present": wrist_present, "missing_policy": cfg.wrist_missing_policy if wrist else None,
                  "max_age_ms": cfg.wrist_max_age_ms if wrist else None, "max_reuse": cfg.wrist_max_reuse if wrist else None,
                  "rows_with_wrist": int(sum(rows["wrist_valid"])), "rows_masked_missing": int(n_kept - sum(rows["wrist_valid"])) if wrist else 0,
                  "masked_reasons": dict(sorted(masked_reasons.items())),
                  "wrist_frames_used": len(wrist_use), "wrist_frames_reused_by_more_than_one_step": len(reused),
                  "rows_served_by_an_already_used_wrist_frame": int(sum(n - 1 for n in reused.values())),
                  "max_steps_served_by_one_wrist_frame": max(wrist_use.values()) if wrist_use else 0},
        "depth": {"fed_to_model": False, "files_opened": 0,
                  "reason": "baseline ACT takes RGB + joint state only; depth is not read or converted",
                  "front_depth_freshness_counts": (depth_freshness(fframes)["counts"] if streams[front].get("depth") else None)},
        "units": {"state": "LeRobot normalised follower MEASURED (degrees x5, gripper 0..100)", "action": f"{cfg.action_signal} (same units)"},
        "ranges": {"state_min": st.min(axis=0).tolist(), "state_max": st.max(axis=0).tolist(),
                   "action_min": ac.min(axis=0).tolist(), "action_max": ac.max(axis=0).tolist()},
        "gripper_outside_0_100": {"state_rows": int(((st[:, 5] < 0) | (st[:, 5] > 100)).sum()),
                                  "action_rows": int(((ac[:, 5] < 0) | (ac[:, 5] > 100)).sum())},
        "lineage": {"episode_json_sha256": entry["episode_json_sha256"], "master_manifest_sha256": entry.get("master_manifest_sha256"),
                    "episode_path": entry["episode_path"], "master_files_verified": cfg.verify_master_files},
        "calibration": {"follower_calibration_matches_controller": (entry.get("calibration") or {}).get("follower_calibration_matches_controller"),
                        "robot_motor_calibration_sha256": (entry.get("calibration") or {}).get("robot_motor_calibration_sha256"),
                        "physically_validated": (entry.get("calibration") or {}).get("physically_validated"),
                        "registry_status": (entry.get("calibration") or {}).get("registry_status")},
        "cameras": cams,
        "median_front_step_dt_ms": statistics.median(dts) if dts else None,
    }
    return rows, audit


# ------------------------------------------------------------------ statistics
def _mean_std(a: np.ndarray) -> dict:
    return {"mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist()}


def fit_stats(arrays: dict, train_episode_ids: list[str], img_sums: dict, cameras: list[str]) -> dict:
    """mean/std on TRAIN rows only (``arrays`` is the full table; the split mask is applied here)."""
    m = arrays["split_code"] == SPLIT_CODES["train"]
    if not m.any():
        raise ConversionError("the train split has no converted rows; cannot fit normalisation")
    stats = {KEY_STATE: _mean_std(arrays["state"][m]), KEY_ACTION: _mean_std(arrays["action"][m])}
    for cam in cameras:
        # every TRAIN row contributes its image once (an image reused by several rows counts once per row)
        idx = {}
        for r in np.nonzero(m)[0]:
            fi = int(arrays[f"{cam}_frame_index"][r])
            if fi >= 0:
                key = (arrays["_episode_ids"][int(arrays["episode_idx"][r])], cam, fi)
                idx[key] = idx.get(key, 0) + 1
        s = np.zeros(3)
        ss = np.zeros(3)
        n = 0
        for key, mult in idx.items():
            a, b, npix = img_sums[key]
            s += mult * a
            ss += mult * b
            n += mult * npix
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean * mean, 0.0))
        stats[KEY_IMAGE_PREFIX + cam] = {"mean": [[[float(v)]] for v in mean], "std": [[[float(v)]] for v in std]}
    for key in (KEY_STATE, KEY_ACTION):
        bad = [JOINT_NAMES[i] for i, v in enumerate(stats[key]["std"]) if v < MIN_STD]
        if bad:
            raise ConversionError(f"degenerate (zero-variance) {key} on the train split for {bad}: mean/std normalisation would "
                                  "divide by ~0. Not patched with a floor; collect motion in these joints or exclude them explicitly.")
    doc = {"schema": STATS_SCHEMA, "fitted_on": "train_split_only", "mode": "MEAN_STD", "std_ddof": 0,
           "train_episode_ids": sorted(train_episode_ids), "n_train_rows": int(m.sum()),
           "image_scale": "uint8 / 255 -> [0, 1] before normalisation; per-channel mean/std, shape (3,1,1)",
           "joint_names": list(JOINT_NAMES), "units": list(UNITS), "stats": stats}
    doc["sha256"] = stats_hash(doc)
    return doc


# ------------------------------------------------------------------ conversion
def _check_split(man: dict, split: dict, allow_provisional: bool) -> dict:
    """Refusals: tampered/inconsistent split, non-eligible entries, provisional split without the explicit flag."""
    expect = sha256_bytes(canonical_json({k: split[k] for k in ("seed", "plan", "assignments", "episode_json_sha256")}).encode())
    if split.get("split_sha256") != expect:
        raise ConversionError("split file content does not match its split_sha256 (edited after it was written)")
    by_id = {e["entry_id"]: e for e in man["entries"]}
    for eid in split["assignments"]:
        e = by_id.get(eid)
        if e is None:
            raise ConversionError(f"{eid}: assigned in the split but not in the manifest")
        is_fx = split.get("data_origin") == "synthetic_fixture" and e["role"] == "synthetic_fixture" and e.get("synthetic_fixture_usable")
        if not e["training_eligible"] and not is_fx:
            raise NotTrainingEligibleError(f"{eid}: not training_eligible ({e['exclusions']}); refusing to convert it")
        if e["source_type"] != "collector_episode":
            raise NotTrainingEligibleError(f"{eid}: {e['source_type']} has no master episode")
    problems = D.verify_split(split, man)
    if problems:
        raise ConversionError("split does not verify against the manifest: " + "; ".join(problems))
    frozen = bool(split.get("frozen"))
    if not frozen and not allow_provisional:
        raise ProvisionalSplitError(
            f"the split is not frozen ({split.get('status')}). Converting a provisional split is only allowed for software/smoke "
            "tests with --allow-provisional-split; the flag is recorded in the output.")
    n_tr = sum(1 for v in split["assignments"].values() if v == "train")
    if n_tr == 0:
        raise ConversionError("the split has no train episodes")
    return {"frozen": frozen, "status": split.get("status"), "provisional": not frozen, "allow_provisional_split_flag": bool(allow_provisional),
            "seed": split.get("seed"), "plan": split.get("plan"), "actual": split.get("actual"),
            "split_sha256": split["split_sha256"]}


def convert_dataset(manifest_path: str | Path, split_path: str | Path, out_dir: str | Path, config: ConvertConfig | dict | None = None,
                    allow_provisional_split: bool = False, root: str | Path | None = None) -> dict:
    """Convert the train/val whole-episode split of a dataset v0.1 manifest. Returns the conversion.json dict."""
    cfg = (config if isinstance(config, ConvertConfig) else ConvertConfig.from_dict(config or {})).validate()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ConversionError(f"{out} exists and is not empty; refusing to overwrite a converted dataset")
    reader = D.DatasetReader(manifest_path, root=root)          # verifies the manifest content hash
    man = reader.manifest
    split = read_json(split_path)
    split_info = _check_split(man, split, allow_provisional_split)
    entries = {e["entry_id"]: e for e in man["entries"]}
    order = sorted(split["assignments"], key=lambda i: (0 if split["assignments"][i] == "train" else 1, i))

    out.mkdir(parents=True, exist_ok=True)
    image_hashes: dict = {}
    img_sums: dict = {}
    all_rows: dict[str, list] = {}
    audits, episodes, fixture_flags = [], [], []
    for ei, eid in enumerate(order):
        entry = entries[eid]
        er = reader.episode(eid)                                 # verifies episode.json against the manifest hash
        qf = er.meta.get("quality_flags") or {}
        fixture_flags.append(bool(qf.get("synthetic_test_fixture") or qf.get("mock_data")))
        rows, audit = _process_episode(er, entry, split["assignments"][eid], ei, cfg, out, image_hashes, img_sums)
        for k, v in rows.items():
            all_rows.setdefault(k, []).extend(v)
        audits.append(audit)
        episodes.append({"index": ei, "episode_id": eid, "split": split["assignments"][eid], "episode_path": entry["episode_path"],
                         "episode_json_sha256": entry["episode_json_sha256"], "master_manifest_sha256": entry.get("master_manifest_sha256"),
                         "front_stream": audit["front_stream"], "wrist_stream": audit["wrist_stream"],
                         "n_front_frames": audit["n_front_frames"], "n_kept": audit["n_kept"],
                         "synthetic_fixture_flag": fixture_flags[-1]})
    if len(set(fixture_flags)) > 1:
        raise ConversionError("the split mixes synthetic-fixture episodes with non-fixture episodes; refusing")
    synthetic = fixture_flags[0]
    data_origin = "synthetic_fixture" if synthetic else "recorded_physical"

    arrays = {
        "episode_idx": np.asarray(all_rows["episode_idx"], np.int32), "split_code": np.asarray(all_rows["split_code"], np.int8),
        "grid_index": np.asarray(all_rows["grid_index"], np.int64), "t_ref_ns": np.asarray(all_rows["t_ref_ns"], np.int64),
        "state": np.asarray(all_rows["state"], np.float64), "action": np.asarray(all_rows["action"], np.float64),
        "raw_goal": np.asarray(all_rows["raw_goal"], np.int64),
        "leader_measurement": np.asarray(all_rows["leader_measurement"], np.float64),
        "leader_valid": np.asarray(all_rows["leader_valid"], bool),
        "command_seq": np.asarray(all_rows["command_seq"], np.int64), "state_seq": np.asarray(all_rows["state_seq"], np.int64),
        "front_frame_index": np.asarray(all_rows["front_frame_index"], np.int64),
        "front_recv_index": np.asarray(all_rows["front_recv_index"], np.int64),
        "front_frame_number": np.asarray(all_rows["front_frame_number"], np.int64),
        "wrist_frame_index": np.asarray(all_rows["wrist_frame_index"], np.int64),
        "wrist_recv_index": np.asarray(all_rows["wrist_recv_index"], np.int64),
        "wrist_valid": np.asarray(all_rows["wrist_valid"], bool),
        "state_t_ns": np.asarray(all_rows["state_t_ns"], np.int64), "command_t_ns": np.asarray(all_rows["command_t_ns"], np.int64),
        "state_age_ns": np.asarray(all_rows["state_age_ns"], np.int64), "command_lag_ns": np.asarray(all_rows["command_lag_ns"], np.int64),
        "wrist_age_ns": np.asarray(all_rows["wrist_age_ns"], np.int64),
    }
    cameras = ["front"] + (["wrist"] if cfg.wrist_stream else [])
    shapes = {}
    for cam in cameras:
        s = {tuple(v["shape"]) for k, v in image_hashes.items() if k.startswith(f"images/{cam}/")}
        if not s:
            raise ConversionError(f"camera {cam!r}: no image was accepted for any step; set that stream to null to train without it")
        if len(s) != 1:
            raise ConversionError(f"camera {cam!r}: images have {len(s)} different shapes {sorted(s)}; set image_size")
        shapes[cam] = list(next(iter(s)))
    if len({tuple(v) for v in shapes.values()}) > 1:
        raise ConversionError(f"ACT needs identical image shapes for all cameras; got {shapes}. Set image_size.")

    stats_arrays = {**arrays, "_episode_ids": [e["episode_id"] for e in episodes]}
    stats_doc = fit_stats(stats_arrays, [e["episode_id"] for e in episodes if e["split"] == "train"], img_sums, cameras)

    np.savez(out / STEPS_FILE, **arrays)
    write_json_atomic(out / STATS_FILE, stats_doc)
    write_json_atomic(out / IMAGE_INDEX_FILE, dict(sorted(image_hashes.items())))

    dts = [a["median_front_step_dt_ms"] for a in audits if a["median_front_step_dt_ms"]]
    med_dt = statistics.median(dts) if dts else None
    control_hz = round(1000.0 / med_dt, 3) if med_dt else None
    labels = []
    if synthetic:
        labels.append(SYNTHETIC_DATA_LABEL)
    if split_info["provisional"]:
        labels.append(PROVISIONAL_LABEL)
    files = {"steps_arrays_sha256": arrays_sha256(arrays), "normalization_stats_sha256": stats_doc["sha256"],
             "image_index_sha256": sha256_file(out / IMAGE_INDEX_FILE)}
    n_train, n_val = int((arrays["split_code"] == 0).sum()), int((arrays["split_code"] == 1).sum())
    conv = {
        "schema": CONVERSION_SCHEMA, "data_origin": data_origin, "labels": labels,
        "suitable_for_main_training": bool(split_info["frozen"] and not synthetic and not allow_provisional_split),
        "config": cfg.to_dict(), "config_sha256": sha256_obj(cfg.to_dict()),
        "source": {"manifest_file": Path(manifest_path).name, "manifest_content_sha256": man["content_sha256"],
                   "dataset_name": man.get("dataset_name"), "split_file": Path(split_path).name, "split": split_info},
        "signals": {
            "observation.state": "follower MEASURED state, telemetry/follower_state.jsonl `normalized` (LeRobot normalised: degrees x5, gripper 0..100); never leader or commanded values",
            "action": {"source": cfg.action_signal, "description": ACTION_SIGNALS[cfg.action_signal],
                       "substitution": cfg.action_signal != ACTION_SIGNAL_DEFAULT,
                       "retained_source_columns": ["raw_goal (Goal_Position ticks SUBMITTED, not confirmed)",
                                                   "leader_measurement (commands.leader_measurement_normalized, leader calibration units)"]},
            "images": {"front": "RGB colour only", "wrist": "RGB colour only (optional)", "depth": "NOT used by baseline ACT"}},
        "joint_names": list(JOINT_NAMES), "units": list(UNITS),
        "cameras": {c: {"source_stream": episodes[0]["front_stream"] if c == "front" else cfg.wrist_stream, "shape_hwc": shapes[c]}
                    for c in cameras},
        "control_hz": control_hz, "control_hz_basis": "1000 / median front-camera frame period (consecutive recv_index) over episodes",
        "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
        "episodes": episodes, "counts": {"train_rows": n_train, "val_rows": n_val,
                                         "train_episodes": sum(e["split"] == "train" for e in episodes),
                                         "val_episodes": sum(e["split"] == "val" for e in episodes)},
        "normalization_sha256": stats_doc["sha256"], "files": files, "created_wall_ns": time.time_ns(),
    }
    conv["conversion_sha256"] = sha256_obj({k: conv[k] for k in (
        "schema", "data_origin", "config_sha256", "source", "episodes", "files", "cameras", "control_hz", "signals")})
    write_json_atomic(out / CONVERSION_FILE, conv)

    audit_doc = build_audit(conv, audits, Converted(out), cfg)
    write_json_atomic(out / "conversion_audit.json", audit_doc)
    (out / "conversion_audit.md").write_text(render_audit_md(audit_doc))
    return conv


def build_audit(conv: dict, ep_audits: list[dict], data: Converted, cfg: ConvertConfig) -> dict:
    warnings = []
    for a in ep_audits:
        c = a["calibration"]["follower_calibration_matches_controller"]
        if c is not True:
            warnings.append(f"{a['episode_id']}: follower_calibration_matches_controller={c!r}. The documented follower/leader "
                            "calibration-mismatch BLOCKER is NOT resolved or hidden by conversion.")
        if a["gripper_outside_0_100"]["state_rows"] or a["gripper_outside_0_100"]["action_rows"]:
            warnings.append(f"{a['episode_id']}: gripper values outside 0..100 (not clipped): {a['gripper_outside_0_100']}")
        if a["n_dropped"]:
            warnings.append(f"{a['episode_id']}: {a['n_dropped']} control steps dropped {a['dropped_by_reason']}")
    for lab in conv["labels"]:
        warnings.append(lab)
    return {
        "schema": CONVERSION_SCHEMA + "/audit", "conversion_sha256": conv["conversion_sha256"], "data_origin": conv["data_origin"],
        "labels": conv["labels"], "suitable_for_main_training": conv["suitable_for_main_training"],
        "action_signal": conv["signals"]["action"], "state_signal": conv["signals"]["observation.state"],
        "joint_order": list(JOINT_NAMES), "units": list(UNITS), "split": conv["source"]["split"],
        "allow_provisional_split_flag_used": conv["source"]["split"]["allow_provisional_split_flag"],
        "normalization": {"sha256": conv["normalization_sha256"], "fitted_on": "train split only",
                          "stats_file": STATS_FILE, "n_train_rows": conv["counts"]["train_rows"]},
        "alignment_config": {k: getattr(cfg, k) for k in ("max_state_age_ms", "max_command_lag_ms", "wrist_max_age_ms",
                                                          "wrist_max_reuse", "wrist_missing_policy", "image_size")},
        "control_hz": conv["control_hz"], "chunk_padding": data.padding_summary(cfg.chunk_size),
        "counts": conv["counts"],
        "totals": {"front_frames": sum(a["n_front_frames"] for a in ep_audits), "kept": sum(a["n_kept"] for a in ep_audits),
                   "dropped_by_reason": _sum_dicts(a["dropped_by_reason"] for a in ep_audits)},
        "lineage": {"manifest_content_sha256": conv["source"]["manifest_content_sha256"], "split_sha256": conv["source"]["split"]["split_sha256"],
                    "source_episodes": [{"episode_id": a["episode_id"], "split": a["split"], **a["lineage"]} for a in ep_audits]},
        "files": conv["files"], "warnings": warnings, "episodes": ep_audits,
        "depth": "not fed to the baseline ACT; no depth file is opened by the conversion",
    }


def _sum_dicts(ds) -> dict:
    out: dict = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return dict(sorted(out.items()))


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def render_audit_md(a: dict) -> str:
    L = ["# ACT conversion audit", ""]
    for lab in a["labels"]:
        L += [f"> **{lab}**", ""]
    L += [f"* conversion_sha256: `{a['conversion_sha256']}`", f"* data_origin: `{a['data_origin']}`; suitable for main training: **{a['suitable_for_main_training']}**",
          f"* action signal: `{a['action_signal']['source']}` (substitution: {a['action_signal']['substitution']}) - {a['action_signal']['description']}",
          f"* state signal: {a['state_signal']}", f"* joint order: {', '.join(a['joint_order'])}; units: {', '.join(a['units'])}",
          f"* split: frozen={a['split']['frozen']}, status: {a['split']['status']}; `--allow-provisional-split` used: **{a['allow_provisional_split_flag_used']}**",
          f"* normalisation: train split only, sha256 `{a['normalization']['sha256']}` ({a['normalization']['n_train_rows']} train rows)",
          f"* manifest content sha256 `{a['lineage']['manifest_content_sha256']}`; split sha256 `{a['lineage']['split_sha256']}`",
          f"* control rate (measured front-frame rate): {a['control_hz']} Hz; chunk padding: {a['chunk_padding']}",
          f"* alignment: {a['alignment_config']}", f"* depth: {a['depth']}", "",
          f"Totals: {a['totals']['kept']} kept of {a['totals']['front_frames']} front frames; dropped by reason: {a['totals']['dropped_by_reason']}", "",
          "## Per episode", "",
          "| episode | split | frames | kept | dropped | state age ms (med/max) | cmd lag ms (med/max) | wrist rows / masked / reused rows | episode_json_sha256 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in a["episodes"]:
        m, w = e["association_ms"], e["wrist"]
        sa, cl = m["state_age"] or {}, m["command_lag"] or {}
        L.append(f"| {e['episode_id']} | {e['split']} | {e['n_front_frames']} | {e['n_kept']} | {e['dropped_by_reason'] or 0} | "
                 f"{_f(sa.get('median'))}/{_f(sa.get('max'))} | {_f(cl.get('median'))}/{_f(cl.get('max'))} | "
                 f"{w['rows_with_wrist']}/{w['rows_masked_missing']}/{w['rows_served_by_an_already_used_wrist_frame']} | `{e['lineage']['episode_json_sha256'][:16]}...` |")
    L += ["", "## Warnings", ""] + [f"* {w}" for w in a["warnings"]] + [""]
    return "\n".join(L)


# ------------------------------------------------------------------ lineage verification
def verify_lineage(converted_dir: str | Path, manifest: dict | str | Path, root: str | Path | None = None, check_images: bool = True) -> dict:
    """Re-read the SOURCE episodes and confirm every converted row maps back to them with identical values.

    ``manifest``: the dataset v0.1 manifest dict (then ``root`` = directory holding the episodes) or its path.
    Checks per row: command seq -> action / raw_goal / leader values equal, state seq -> state equal, front/wrist frame index ->
    recv_index / host time / frame number equal, the alignment rule re-derived independently (no newer valid state, no earlier
    command), the converted image's pixel hash equals a fresh conversion of the source PNG; and per episode: episode.json hash and the
    split assignment. Returns {ok, errors, n_rows, ...}.
    """
    conv_dir = Path(converted_dir)
    data = Converted(conv_dir)
    errs: list[str] = []
    if isinstance(manifest, dict):
        man = manifest
        if root is None:
            raise ValueError("root is required when the manifest is passed as a dict")
        if D.manifest_hash(man) != man["content_sha256"]:
            return {"ok": False, "errors": ["manifest content hash mismatch"], "n_rows": data.n}
        rd_root = Path(root)
    else:
        rd = D.DatasetReader(manifest)
        man, rd_root = rd.manifest, rd.root
    conv = data.conv
    if man["content_sha256"] != conv["source"]["manifest_content_sha256"]:
        errs.append("manifest content hash differs from the one recorded at conversion")
    by_id = {e["entry_id"]: e for e in man["entries"]}
    cfg = ConvertConfig.from_dict(conv["config"])
    ms = 1e6
    images = json.loads((conv_dir / IMAGE_INDEX_FILE).read_text())
    n_img = 0
    a = data.a
    sig_key = "values" if cfg.action_signal == ACTION_SIGNAL_DEFAULT else "leader_measurement_normalized"
    split_of = {}
    for ep in conv["episodes"]:
        eid = ep["episode_id"]
        e = by_id.get(eid)
        if e is None:
            errs.append(f"{eid}: not in manifest")
            continue
        if e["episode_json_sha256"] != ep["episode_json_sha256"]:
            errs.append(f"{eid}: episode_json_sha256 differs from the manifest")
        src = rd_root / e["episode_path"]
        if sha256_file(src / "episode.json") != ep["episode_json_sha256"]:
            errs.append(f"{eid}: source episode.json changed")
            continue
        split_of[eid] = ep["split"]
        er = D.EpisodeReader(src)
        states, cmds = list(er.follower_states()), list(er.commands())
        fr = list(er.frames(ep["front_stream"]))
        wr = list(er.frames(ep["wrist_stream"])) if ep["wrist_stream"] and ep["wrist_stream"] in er.cameras else []
        s_by_seq = {r["seq"]: i for i, r in enumerate(states)}
        c_by_seq = {r["seq"]: i for i, r in enumerate(cmds)}
        s_t = [state_time(r) for r in states]
        s_ok = [r.get("read_ok") is True and _vec(r.get("normalized")) is not None for r in states]
        c_t = [r["t_start_ns"] for r in cmds]
        for r in np.nonzero(a["episode_idx"] == ep["index"])[0]:
            r = int(r)
            tag = f"{eid} row {r}"
            ci, si = c_by_seq.get(int(a["command_seq"][r])), s_by_seq.get(int(a["state_seq"][r]))
            if ci is None or si is None:
                errs.append(f"{tag}: command seq {a['command_seq'][r]} / state seq {a['state_seq'][r]} not in source")
                continue
            if _vec(cmds[ci].get(sig_key)) != a["action"][r].tolist():
                errs.append(f"{tag}: action differs from source {sig_key}")
            if _ivec(cmds[ci].get("raw_goal")) != a["raw_goal"][r].tolist():
                errs.append(f"{tag}: raw_goal differs from source")
            lead = _vec(cmds[ci].get("leader_measurement_normalized"))
            if bool(a["leader_valid"][r]) != (lead is not None) or (lead is not None and lead != a["leader_measurement"][r].tolist()):
                errs.append(f"{tag}: leader measurement differs from source")
            if _vec(states[si].get("normalized")) != a["state"][r].tolist() or states[si].get("read_ok") is not True:
                errs.append(f"{tag}: state differs from source normalized / read_ok")
            fi = int(a["front_frame_index"][r])
            f = fr[fi]
            t = int(a["t_ref_ns"][r])
            if f["host_mono_ns"] != t or f["recv_index"] != a["front_recv_index"][r] or f["recv_index"] != a["grid_index"][r] \
                    or (-1 if f.get("frame_number") is None else f["frame_number"]) != a["front_frame_number"][r]:
                errs.append(f"{tag}: front frame record differs from source")
            if c_t[ci] != a["command_t_ns"][r] or s_t[si] != a["state_t_ns"][r]:
                errs.append(f"{tag}: stored times differ from source")
            # independent re-derivation of the alignment rule
            if not (s_t[si] <= t and t - s_t[si] <= cfg.max_state_age_ms * ms and not any(s_ok[k] for k in range(si + 1, len(states)) if s_t[k] <= t)):
                errs.append(f"{tag}: state is not the latest valid sample within max age")
            if not (c_t[ci] >= t and c_t[ci] - t <= cfg.max_command_lag_ms * ms and (ci == 0 or c_t[ci - 1] < t)):
                errs.append(f"{tag}: command is not the first one at/after t_ref within max lag")
            wi = int(a["wrist_frame_index"][r])
            if wi >= 0:
                w = wr[wi] if wi < len(wr) else None
                if w is None or w["host_mono_ns"] > t or t - w["host_mono_ns"] > cfg.wrist_max_age_ms * ms \
                        or w["recv_index"] != a["wrist_recv_index"][r]:
                    errs.append(f"{tag}: wrist frame violates the age rule or differs from source")
            elif bool(a["wrist_valid"][r]):
                errs.append(f"{tag}: wrist_valid without a wrist frame index")
            if check_images:
                for cam, frames, stream in (("front", fr, ep["front_stream"]), ("wrist", wr, ep["wrist_stream"])):
                    rel = data.image_rel(cam, r) if cam in data.cameras else None
                    if rel is None:
                        continue
                    n_img += 1
                    rec = images.get(rel)
                    got = read_rgb(conv_dir / rel)
                    srcimg = read_rgb(src / "streams" / stream / frames[int(a[f"{cam}_frame_index"][r])]["color_file"])
                    if rec is None or got is None or srcimg is None:
                        errs.append(f"{tag}: image {rel} missing in index/converted/source")
                    elif pixel_sha256(got) != rec["pixel_sha256"] or pixel_sha256(resize_rgb(srcimg, cfg.image_size)) != rec["pixel_sha256"]:
                        errs.append(f"{tag}: image {rel} does not equal the freshly converted source frame")
    # split leakage: an episode is in exactly one split and matches the split assignment recorded at conversion
    split_json = conv["source"]["split"]
    if len(split_of) != len(conv["episodes"]):
        errs.append("some episodes could not be verified")
    tr = {e["episode_id"] for e in conv["episodes"] if e["split"] == "train"}
    va = {e["episode_id"] for e in conv["episodes"] if e["split"] == "val"}
    if tr & va:
        errs.append(f"episodes in both train and val: {sorted(tr & va)}")
    for r in range(data.n):
        e = conv["episodes"][int(a["episode_idx"][r])]
        if int(a["split_code"][r]) != SPLIT_CODES[e["split"]]:
            errs.append(f"row {r}: split code differs from its episode's split")
            break
    return {"ok": not errs, "errors": errs[:100], "n_errors": len(errs), "n_rows": data.n, "n_images_checked": n_img,
            "n_episodes": len(conv["episodes"]), "split_status": split_json["status"]}
