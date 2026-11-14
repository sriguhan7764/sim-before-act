"""Dataset v0.1: versioned entry/manifest schema, inventory, quality summary, whole-episode split freeze, offline reader.

READ-ONLY over masters: nothing here writes into an episode directory, opens a device, or imports a robot adapter.
Nothing is estimated or invented: unmeasured poses, unknown outcomes and absent evidence stay ``null`` / flagged.

An *entry* describes one episode (or one simulated rollout). A *manifest* is a list of entries + the policy used to derive
them + a content hash. A *split manifest* assigns whole episodes (never frames) to train/val; it is frozen only when the
planned counts are actually reached - until then it is explicitly provisional.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pilot import eligibility, pose_missing
from .util import canonical_json, iter_jsonl, read_json, sha256_bytes, sha256_file, write_json_atomic

DATASET_SCHEMA = "dataset-v0.1"
DATASET_VERSION = "0.1"
ROLES = ("setup_pilot", "successful_demonstration", "unsuccessful_attempt", "evaluation_trial",
         "synthetic_fixture", "simulated_rollout", "unclassified")
DATA_ORIGINS = ("recorded_physical", "synthetic", "simulated")
OUTCOMES = ("success", "failure", "aborted", "unknown", "unlabelled")
LABEL_SOURCES = ("operator_visual", "operator_measured", "none")
HORIZONS = ("full_task", "segment", None)
SOURCE_TYPES = ("collector_episode", "sim_rollout")
DEFAULT_POLICY = {"require_measured_poses": True, "require_curator_acceptance": True, "require_valid_master": True}
PLAN = {"train": 25, "val": 5}        # the review plan, NOT a claim that 30 demonstrations exist

REQUIRED_ENTRY_KEYS = ("schema", "entry_id", "source_type", "data_origin", "role", "role_basis", "split", "task",
                       "cube_position_id", "cube_start_pose_mat_frame", "tray_pose_mat_frame", "outcome", "cameras",
                       "calibration", "semantics", "quality", "exclusions", "training_eligible", "lineage")

SEMANTICS = {
    "state": "follower MEASURED: raw Feetech Present_Position ticks (0-4095, after motor homing offset) + calibrated joint_angles_rad "
             "(radians relative to the calibrated range midpoint) when a calibration is attached; gripper normalised 0-100, never radians",
    "action_candidates": {
        "commands.values": "controller's CLIPPED follower target in LeRobot normalised units (degrees for 5 joints, 0-100 gripper); "
                           "this is what the follower controller was asked to reach",
        "commands.raw_goal": "raw Goal_Position ticks submitted for writing (submission_status=submitted_not_confirmed: "
                             "NOT confirmed physical execution, no register readback)"},
    "leader": "leader arm measured separately; never substituted for follower state or commands",
    "measured_and_commanded_separate": True,
}


class DatasetError(ValueError):
    pass


def _latest_annotation(ep: Path) -> dict | None:
    d = ep.parent / (ep.name + ".annotations")
    files = sorted(d.glob("v*.json")) if d.is_dir() else []
    return read_json(files[-1]) if files else None


def derive_role(meta: dict, annotation: dict | None) -> tuple[str, str, str]:
    """(role, data_origin, basis). The role follows recorded pilot metadata + curator outcome; never guessed."""
    qf = meta.get("quality_flags", {})
    pilot = meta.get("pilot") or {}
    if qf.get("mock_data") or qf.get("synthetic_fixture"):
        return "synthetic_fixture", "synthetic", "quality_flags.mock_data / synthetic_fixture"
    outcome = (annotation or {}).get("outcome", meta.get("outcome"))
    prole = pilot.get("role")
    if prole == "setup":
        return "setup_pilot", "recorded_physical", "pilot.role=setup"
    if prole == "evaluation":
        return "evaluation_trial", "recorded_physical", "pilot.role=evaluation"
    if prole == "demonstration":
        if outcome == "success":
            return "successful_demonstration", "recorded_physical", "pilot.role=demonstration + outcome=success"
        if outcome in ("failure", "aborted"):
            return "unsuccessful_attempt", "recorded_physical", f"pilot.role=demonstration + outcome={outcome}"
        return "unclassified", "recorded_physical", f"pilot.role=demonstration but outcome={outcome!r} (needs an operator label)"
    return "unclassified", "recorded_physical", "no pilot metadata recorded (episode predates the pilot schema or was not a pilot)"


def _camera_block(ep: Path, meta: dict) -> dict:
    out = {}
    for name, s in sorted((meta.get("streams") or {}).items()):
        cj = ep / "streams" / name / "camera.json"
        entry = {"role": s.get("role"), "depth": s.get("depth"), "camera_json_sha256": None, "stream_profile": None,
                 "device": None, "depth_scale": None, "n_frames": None}
        if cj.exists():
            c = read_json(cj)
            entry.update(camera_json_sha256=sha256_file(cj), stream_profile=c.get("stream_profile"), device=c.get("device"),
                         depth_scale=c.get("depth_scale"), is_mock=c.get("is_mock"))
        fj = ep / "streams" / name / "frames.jsonl"
        if fj.exists():
            entry["n_frames"] = sum(1 for _ in iter_jsonl(fj))
        out[name] = entry
    return out


def _calibration_block(ep: Path, meta: dict) -> dict:
    cal = meta.get("calibration") or {}
    files = {}
    for f in sorted((ep / "calibration").glob("*.json")) if (ep / "calibration").is_dir() else []:
        files[f.name] = sha256_file(f)
    ctl = meta.get("controller") or {}
    return {"camera_to_mat": cal.get("camera_to_mat"), "robot_base_to_mat": cal.get("robot_base_to_mat"),
            "robot_motor_calibration_sha256": cal.get("robot_motor_calibration_sha256"),
            "controller_session_sha256": ctl.get("published_session_sha256"),
            "follower_calibration_matches_controller": (meta.get("quality_flags") or {}).get(
                "follower_calibration_hash_matches_controller"),
            "episode_calibration_file_hashes": files,
            "physically_validated": False,
            "note": "hash references only; no physical validation of joint signs/gripper/base alignment is implied"}


def _lineage(ep: Path) -> dict:
    exp = ep.parent / (ep.name + ".exports")
    exports = []
    for d in sorted(exp.iterdir()) if exp.is_dir() else []:
        m = d / "manifest.json"
        if m.exists():
            mj = read_json(m)
            exports.append({"profile": mj.get("profile"), "path": d.name, "manifest_sha256": sha256_file(m),
                            "source_episode_id": (mj.get("source") or {}).get("episode_id", mj.get("source_episode_id"))})
    return {"derived_from": None, "exports": exports, "master_is_source_of_truth": True}


def entry_from_episode(ep_dir: str | Path, policy: dict | None = None, validate_hashes: bool = True,
                       stale_calibrations: dict[str, str] | None = None) -> dict:
    """``stale_calibrations``: {sha256: reason} from the calibration registry; an episode recorded with one is excluded."""
    from .validate import validate

    pol = {**DEFAULT_POLICY, **(policy or {})}
    ep = Path(ep_dir)
    meta = read_json(ep / "episode.json")
    ann = _latest_annotation(ep)
    role, origin, basis = derive_role(meta, ann)
    pilot = meta.get("pilot") or {}
    qf = meta.get("quality_flags", {})
    val = validate(ep) if validate_hashes else {"ok": None, "errors": [], "warnings": ["hash validation skipped"], "info": {}}
    outcome = (ann or {}).get("outcome", meta.get("outcome", "unlabelled"))
    label_source = (ann or {}).get("label_source", pilot.get("outcome_label_source", "none")) if outcome in ("success", "failure") else "none"
    el = eligibility(meta, ann)
    excl: list[str] = []
    if role == "successful_demonstration":
        excl = list(el["reasons"])
        if not pol["require_measured_poses"]:
            excl = [r for r in excl if "not measured" not in r]
        if pol["require_curator_acceptance"] and (ann or {}).get("accepted_for_dataset") is not True:
            excl.append("no curator acceptance (annotate --accepted-for-dataset yes)")
        if pol["require_valid_master"] and val["ok"] is not True:
            excl.append("master did not validate" + ("" if val["ok"] is False else " (not checked)"))
        if qf.get("data_dropped"):
            excl.append("writer dropped samples (quality_flags.data_dropped)")
    else:
        excl = [f"role is {role}: not a training demonstration"]
    fixture_gate: list[str] = []
    if role == "synthetic_fixture":
        # fixtures stay excluded from physical training, but software tests still need to refuse a bad fixture
        if (ann or {}).get("accepted_for_dataset") is not True:
            fixture_gate.append("no curator acceptance (annotate --accepted-for-dataset yes)")
        if val["ok"] is not True:
            fixture_gate.append("master did not validate" + ("" if val["ok"] is False else " (not checked)"))
        if qf.get("data_dropped"):
            fixture_gate.append("writer dropped samples (quality_flags.data_dropped)")
        excl += fixture_gate
    cal_sha = (meta.get("calibration") or {}).get("robot_motor_calibration_sha256")
    stale_reason = (stale_calibrations or {}).get(cal_sha) if cal_sha else None
    if stale_reason:
        excl.append(f"recorded under a STALE robot calibration ({stale_reason}); must not be interpreted with the active one")
    entry = {
        "schema": DATASET_SCHEMA,
        "entry_id": meta["episode_id"],
        "source_type": "collector_episode",
        "episode_id": meta["episode_id"],
        "session_id": meta.get("session_id") or ((meta.get("controller") or {}).get("published_session") or {}).get("session_id"),
        "episode_path": ep.name,
        "master_schema_version": meta.get("schema_version"),
        "episode_json_sha256": sha256_file(ep / "episode.json"),
        "master_manifest_sha256": sha256_file(ep / "manifest.json") if (ep / "manifest.json").exists() else None,
        "data_origin": origin,
        "role": role,
        "role_basis": basis,
        "split": None,
        "task": {"name": meta.get("task_name"), "instruction": meta.get("instruction") or None,
                 "target": {"object": "red cube", "receptacle": "blue tray"} if pilot else None,
                 "note": "ACT pilot task is fixed (not language-conditioned); instruction text is operator metadata only"},
        "cube_position_id": pilot.get("cube_position_id"),
        "cube_start_pose_mat_frame": pilot.get("cube_start_pose_mat_frame"),
        "tray_pose_mat_frame": pilot.get("tray_pose_mat_frame"),
        "poses_measured": {"cube": not pose_missing(pilot.get("cube_start_pose_mat_frame")),
                           "tray": not pose_missing(pilot.get("tray_pose_mat_frame"))},
        "outcome": {"value": outcome, "label_source": label_source, "horizon": "full_task" if outcome in ("success", "failure") else None,
                    "annotation_version": (ann or {}).get("version"), "failure_category": (ann or {}).get("failure_category") or meta.get("failure_category"),
                    "accepted_for_dataset": (ann or {}).get("accepted_for_dataset")},
        "cameras": _camera_block(ep, meta),
        "calibration": {**_calibration_block(ep, meta),
                        "registry_status": ("stale" if stale_reason else ("not on the stale list" if stale_calibrations is not None
                                                                         else "registry not supplied (stale check not performed)")),
                        "stale_reason": stale_reason},
        "semantics": SEMANTICS,
        "quality": {"status": meta.get("status"), "validation_ok": val["ok"], "validation_errors": val["errors"][:10],
                    "validation_warnings": val["warnings"][:10], "flags": qf,
                    "counts": meta.get("counts"), "missing_modalities": qf.get("missing_modalities", [])},
        "exclusions": sorted(set(excl)),
        "training_eligible": False,
        "lineage": _lineage(ep),
    }
    entry["training_eligible"] = role == "successful_demonstration" and not entry["exclusions"]
    if role == "synthetic_fixture":
        entry["synthetic_fixture_usable"] = not fixture_gate
    return entry


def sim_rollout_entry(rollout_id: str, trace_path: str | Path, generator: str, seed: int | None, outcome: str,
                      scene_config_sha256: str | None, source_action_hash: str | None = None) -> dict:
    """A SIMULATED rollout. It has no physical label and never enters a physical split."""
    tp = Path(trace_path)
    return {"schema": DATASET_SCHEMA, "entry_id": rollout_id, "source_type": "sim_rollout", "data_origin": "simulated",
            "role": "simulated_rollout", "role_basis": "produced by the MuJoCo verifier", "split": None,
            "task": {"name": "red_cube_to_blue_tray", "instruction": None, "target": {"object": "red cube", "receptacle": "blue tray"}},
            "cube_position_id": None, "cube_start_pose_mat_frame": None, "tray_pose_mat_frame": None,
            "outcome": {"value": outcome, "label_source": "simulator", "horizon": "segment",
                        "note": "simulated outcome; says nothing about the physical robot"},
            "cameras": {}, "calibration": {"physically_validated": False}, "semantics": {"simulated": True},
            "quality": {"seed": seed, "generator": generator, "scene_config_sha256": scene_config_sha256},
            "exclusions": ["simulated rollout: not a physical demonstration"], "training_eligible": False,
            "lineage": {"derived_from": source_action_hash, "trace_sha256": sha256_file(tp) if tp.exists() else None,
                        "trace_file": tp.name}}


def validate_entry(e: dict) -> list[str]:
    """Structural problems with one entry (empty = well-formed). Says nothing about physical validity."""
    p = [f"missing key {k}" for k in REQUIRED_ENTRY_KEYS if k not in e]
    if p:
        return p
    if e["schema"] != DATASET_SCHEMA:
        p.append(f"schema {e['schema']!r} != {DATASET_SCHEMA}")
    if e["role"] not in ROLES:
        p.append(f"unknown role {e['role']!r}")
    if e["data_origin"] not in DATA_ORIGINS:
        p.append(f"unknown data_origin {e['data_origin']!r}")
    if e["source_type"] not in SOURCE_TYPES:
        p.append(f"unknown source_type {e['source_type']!r}")
    o = e["outcome"]
    if o.get("value") not in OUTCOMES and e["source_type"] == "collector_episode":
        p.append(f"unknown outcome {o.get('value')!r}")
    if o.get("horizon") not in HORIZONS:
        p.append(f"unknown horizon {o.get('horizon')!r}")
    if e["source_type"] == "collector_episode" and o.get("value") in ("success", "failure") and o.get("label_source") in (None, "none"):
        p.append("success/failure outcome without a label source")
    if e["role"] == "synthetic_fixture" and e["data_origin"] != "synthetic":
        p.append("synthetic_fixture must have data_origin=synthetic")
    if e["role"] == "simulated_rollout" and e["data_origin"] != "simulated":
        p.append("simulated_rollout must have data_origin=simulated")
    if e["data_origin"] != "recorded_physical" and e["training_eligible"]:
        p.append("non-physical data marked training_eligible")
    if e["training_eligible"] and e["exclusions"]:
        p.append("training_eligible with exclusions")
    if e["split"] in ("train", "val") and not e["training_eligible"] and not (e["role"] == "synthetic_fixture" and e.get("synthetic_fixture_usable")):
        p.append("train/val assignment for an entry that is not training_eligible")
    return p


def build_manifest(episode_dirs: list[str | Path], policy: dict | None = None, extra_entries: list[dict] | None = None,
                   validate_hashes: bool = True, name: str = "dataset", stale_calibrations: dict[str, str] | None = None) -> dict:
    pol = {**DEFAULT_POLICY, **(policy or {})}
    entries = [entry_from_episode(d, pol, validate_hashes, stale_calibrations) for d in sorted(Path(x) for x in episode_dirs)]
    entries += list(extra_entries or [])
    ids = [e["entry_id"] for e in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise DatasetError(f"duplicate entry ids: {dup}")
    bad = {e["entry_id"]: validate_entry(e) for e in entries if validate_entry(e)}
    if bad:
        raise DatasetError(f"malformed entries: {bad}")
    entries.sort(key=lambda e: e["entry_id"])
    man = {"schema": DATASET_SCHEMA, "dataset_name": name, "dataset_version": DATASET_VERSION, "policy": pol, "plan": PLAN,
           "plan_note": "the 25/5 plan is a target; inventory reports what actually exists",
           "entries": entries}
    man["inventory"] = inventory(man)
    man["content_sha256"] = manifest_hash(man)
    return man


def manifest_hash(man: dict) -> str:
    return sha256_bytes(canonical_json({"policy": man["policy"], "entries": man["entries"]}).encode())


def inventory(man: dict) -> dict:
    ents = man["entries"]

    def count(key):
        c: dict = {}
        for e in ents:
            v = e[key] if not isinstance(e[key], dict) else e[key].get("value")
            c[str(v)] = c.get(str(v), 0) + 1
        return dict(sorted(c.items()))

    elig = [e for e in ents if e["training_eligible"]]
    planned = sum(PLAN.values())
    excl: dict = {}
    for e in ents:
        if e["role"] == "successful_demonstration":
            for r in e["exclusions"]:
                excl[r] = excl.get(r, 0) + 1
    phys = [e for e in ents if e["data_origin"] == "recorded_physical"]
    return {
        "n_entries": len(ents), "by_role": count("role"), "by_data_origin": count("data_origin"), "by_outcome": count("outcome"),
        "by_split": count("split"),
        "training_eligible": len(elig), "planned_successful_demonstrations": planned,
        "shortfall_vs_plan": max(0, planned - len(elig)),
        "plan_met": len(elig) >= planned,
        "recorded_physical_entries": len(phys),
        "successful_demonstrations_blocked_by": dict(sorted(excl.items())),
        "cube_position_ids_seen": sorted({e["cube_position_id"] for e in ents if e.get("cube_position_id")}),
        "statement": ("counts are of episodes present in this manifest. Synthetic fixtures and simulated rollouts are never counted as "
                      "recorded demonstrations. Nothing here is a physical-validation claim."),
    }


def quality_summary(man: dict) -> dict:
    rows = []
    for e in man["entries"]:
        if e["source_type"] != "collector_episode":
            continue
        q = e["quality"]
        rows.append({"entry_id": e["entry_id"], "role": e["role"], "data_origin": e["data_origin"], "status": q["status"],
                     "validation_ok": q["validation_ok"], "n_validation_errors": len(q["validation_errors"]),
                     "dropped": bool((q["counts"] or {}).get("dropped")),
                     "missing_modalities": q["missing_modalities"],
                     "cube_pose_measured": e["poses_measured"]["cube"], "tray_pose_measured": e["poses_measured"]["tray"],
                     "calibration_physically_validated": e["calibration"]["physically_validated"]})
    return {"rows": rows, "n_valid": sum(1 for r in rows if r["validation_ok"]), "n_invalid_or_unchecked": sum(1 for r in rows if r["validation_ok"] is not True)}


# ---------------------------------------------------------------- splits
def _order_key(seed: int, entry_id: str) -> str:
    return hashlib.sha256(f"{seed}:{entry_id}".encode()).hexdigest()


def make_split(man: dict, seed: int, plan: dict | None = None, synthetic_fixture: bool = False) -> dict:
    """Whole-episode train/val assignment from the manifest's training-eligible physical demonstrations.

    Deterministic (hash-ordered by seed+id). Frozen ONLY if the planned total is reached; otherwise provisional, with
    proportional counts and an explicit ``shortfall``. Surplus beyond the plan is listed, not silently added.
    Evaluation trials, synthetic fixtures and simulated rollouts are never placed in train/val of a physical split.
    ``synthetic_fixture=True`` builds the opposite, software-test-only split over synthetic fixture episodes: it is never frozen,
    carries ``data_origin: synthetic_fixture`` and can never be mistaken for (or verified as) a physical split.
    """
    plan = dict(plan or PLAN)
    total = sum(plan.values())
    if synthetic_fixture:
        pool = [e for e in man["entries"] if e["role"] == "synthetic_fixture" and e["data_origin"] == "synthetic"
                and e["source_type"] == "collector_episode" and e.get("synthetic_fixture_usable")]
    else:
        pool = [e for e in man["entries"] if e["training_eligible"] and e["data_origin"] == "recorded_physical"]
    elig = sorted(pool, key=lambda e: _order_key(seed, e["entry_id"]))
    n = min(len(elig), total)
    n_val = plan["val"] if n == total else (0 if n < 2 else max(1, round(n * plan["val"] / total)))
    chosen = elig[:n]
    assign = {e["entry_id"]: ("val" if i < n_val else "train") for i, e in enumerate(chosen)}
    hashes = {e["entry_id"]: e["episode_json_sha256"] for e in chosen}
    complete = len(elig) >= total and not synthetic_fixture
    sp = {
        "schema": DATASET_SCHEMA + "/split", "seed": seed, "plan": plan, "unit": "whole_episode",
        "data_origin": "synthetic_fixture" if synthetic_fixture else "recorded_physical",
        "manifest_content_sha256": man["content_sha256"],
        "actual": {"train": sum(v == "train" for v in assign.values()), "val": n_val},
        "complete": complete, "frozen": complete,
        "status": ("SYNTHETIC FIXTURE SPLIT (software tests only; not a physical split; never frozen)" if synthetic_fixture else
                   "FROZEN" if complete else "PROVISIONAL (planned counts not reached; re-derive when more demonstrations exist; do not train the main model on it)"),
        "shortfall": {"train": max(0, plan["train"] - sum(v == "train" for v in assign.values())), "val": max(0, plan["val"] - n_val)},
        "assignments": dict(sorted(assign.items())), "episode_json_sha256": dict(sorted(hashes.items())),
        "surplus_unassigned": sorted(e["entry_id"] for e in elig[total:]),
        "evaluation_trials": sorted(e["entry_id"] for e in man["entries"] if e["role"] == "evaluation_trial"),
        "not_in_physical_splits": sorted(e["entry_id"] for e in man["entries"] if e["data_origin"] != "recorded_physical"),
        "excluded_demonstration_like": {e["entry_id"]: e["exclusions"] for e in man["entries"]
                                       if e["role"] in ("successful_demonstration",) and not e["training_eligible"]},
    }
    sp["split_sha256"] = sha256_bytes(canonical_json({k: sp[k] for k in ("seed", "plan", "assignments", "episode_json_sha256")}).encode())
    return sp


def verify_split(split: dict, man: dict) -> list[str]:
    p: list[str] = []
    by_id = {e["entry_id"]: e for e in man["entries"]}
    for eid, s in split["assignments"].items():
        e = by_id.get(eid)
        if e is None:
            p.append(f"{eid}: not in manifest")
            continue
        if s not in ("train", "val"):
            p.append(f"{eid}: bad split {s!r}")
        if split.get("data_origin") == "synthetic_fixture":
            if not (e["role"] == "synthetic_fixture" and e["data_origin"] == "synthetic" and e.get("synthetic_fixture_usable")):
                p.append(f"{eid}: a synthetic-fixture split may only contain usable synthetic fixture episodes")
            if split["frozen"]:
                p.append("a synthetic-fixture split can never be frozen")
        elif not e["training_eligible"] or e["data_origin"] != "recorded_physical":
            p.append(f"{eid}: not a training-eligible physical demonstration")
        if e["episode_json_sha256"] != split["episode_json_sha256"].get(eid):
            p.append(f"{eid}: episode.json changed since the split was made")
    if set(split["evaluation_trials"]) & set(split["assignments"]):
        p.append("an evaluation trial is in train/val")
    if split["frozen"] and not (split["actual"]["train"] == split["plan"]["train"] and split["actual"]["val"] == split["plan"]["val"]):
        p.append("marked frozen but planned counts not met")
    if split["manifest_content_sha256"] != man["content_sha256"]:
        p.append("manifest content hash differs from the one the split was derived from")
    return p


def write_split(path: str | Path, split: dict) -> Path:
    """A FROZEN split file is never overwritten with different assignments."""
    path = Path(path)
    if path.exists():
        old = read_json(path)
        if old.get("frozen") and old.get("split_sha256") != split["split_sha256"]:
            raise DatasetError(f"{path} is a FROZEN split; refusing to overwrite with different assignments")
    write_json_atomic(path, split)
    return path


def apply_split(man: dict, split: dict) -> dict:
    """Return a copy of the manifest with ``split`` filled in per entry (assignment, or 'evaluation', else null)."""
    out = json.loads(json.dumps(man))
    for e in out["entries"]:
        e["split"] = split["assignments"].get(e["entry_id"]) or ("evaluation" if e["entry_id"] in split["evaluation_trials"] else None)
        if validate_entry(e):
            raise DatasetError(f"{e['entry_id']}: {validate_entry(e)}")
    out["inventory"] = inventory(out)
    out["content_sha256"] = manifest_hash(out)
    return out


# ---------------------------------------------------------------- offline reader
class EpisodeReader:
    """Lazy, read-only access to one master episode. No images are decoded unless asked; no device library is imported."""

    def __init__(self, ep_dir: str | Path):
        self.dir = Path(ep_dir)
        self.meta = read_json(self.dir / "episode.json")
        self.joints = read_json(self.dir / "calibration" / "joints.json") if (self.dir / "calibration" / "joints.json").exists() else None

    @property
    def cameras(self) -> list[str]:
        return sorted(self.meta.get("streams", {}))

    def follower_states(self):
        return iter_jsonl(self.dir / "telemetry" / "follower_state.jsonl")

    def leader_states(self):
        p = self.dir / "telemetry" / "leader_state.jsonl"
        return iter_jsonl(p) if p.exists() else iter(())

    def commands(self):
        p = self.dir / "telemetry" / "commands.jsonl"
        return iter_jsonl(p) if p.exists() else iter(())

    def frames(self, cam: str):
        return iter_jsonl(self.dir / "streams" / cam / "frames.jsonl")

    def associations(self):
        p = self.dir / "sync" / "associations.jsonl"
        return iter_jsonl(p) if p.exists() else iter(())

    def image_path(self, cam: str, frame: dict, kind: str = "color") -> Path | None:
        f = frame.get({"color": "color_file", "depth_native": "depth_native_file", "depth_aligned": "depth_aligned_file"}[kind])
        return self.dir / "streams" / cam / f if f else None


class DatasetReader:
    def __init__(self, manifest_path: str | Path, root: str | Path | None = None):
        self.manifest_path = Path(manifest_path)
        self.manifest = read_json(self.manifest_path)
        self.root = Path(root) if root else self.manifest_path.parent
        if self.manifest.get("schema") != DATASET_SCHEMA:
            raise DatasetError(f"unsupported manifest schema {self.manifest.get('schema')!r}")
        if manifest_hash(self.manifest) != self.manifest["content_sha256"]:
            raise DatasetError("manifest content hash mismatch (file edited after it was written)")

    def entries(self, role: str | None = None, split: str | None = None):
        return [e for e in self.manifest["entries"] if (role is None or e["role"] == role) and (split is None or e["split"] == split)]

    def episode(self, entry_id: str) -> EpisodeReader:
        e = next((x for x in self.manifest["entries"] if x["entry_id"] == entry_id), None)
        if e is None:
            raise KeyError(entry_id)
        if e["source_type"] != "collector_episode":
            raise DatasetError(f"{entry_id} is a {e['source_type']}; it has no master episode")
        d = self.root / e["episode_path"]
        if sha256_file(d / "episode.json") != e["episode_json_sha256"]:
            raise DatasetError(f"{entry_id}: episode.json differs from the hash in the manifest")
        return EpisodeReader(d)
