"""Milestone B: dataset v0.1 manifests, roles, inventory, whole-episode splits, reader, CLI. SYNTHETIC fixtures only.

Every episode here is a synthetic fixture written by the test writer; poses below are INVENTED test values, not measurements.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import fill, new_writer

from collector import dataset as D
from collector.cli import main as cli
from collector.pilot import build_pilot_meta
from collector.util import read_json, sha256_file

POSE = {"x_m": 0.01, "y_m": 0.2, "yaw_deg": 0.0, "source": "manual_measurement", "sigma_mm": 2.0}   # INVENTED for tests


def make_ep(tmp, role="demonstration", outcome="success", pose=True, accept=True, mock=False, label_source="operator_visual",
            cube_id="P1"):
    pilot = build_pilot_meta(role, cube_position_id=cube_id if role == "demonstration" else None,
                             cube_pose=POSE if pose else None, tray_pose=POSE if pose else None)
    pilot["outcome_label_source"] = label_source if outcome in ("success", "failure") else "none"
    w = new_writer(tmp, mock=mock, meta_extra={"pilot": pilot, "outcome": outcome})
    fill(w, n_frames=4, n_states=20, n_cmds=8)
    ep = w.finalize(outcome)
    if accept is not None and role == "demonstration":
        d = ep.parent / (ep.name + ".annotations")
        d.mkdir()
        (d / "v001.json").write_text(json.dumps({"version": 1, "outcome": outcome, "label_source": pilot["outcome_label_source"],
                                                 "accepted_for_dataset": accept, "failure_category": None}))
    return ep


def tree_hash(ep):
    return {p.relative_to(ep).as_posix(): sha256_file(p) for p in sorted(ep.rglob("*")) if p.is_file()}


def test_roles_and_origin_are_derived_not_guessed(tmp_path):
    cases = {
        "setup": make_ep(tmp_path, role="setup", outcome="unlabelled", accept=None),
        "good": make_ep(tmp_path),
        "bad": make_ep(tmp_path, outcome="failure"),
        "aborted": make_ep(tmp_path, outcome="aborted"),
        "unlabelled": make_ep(tmp_path, outcome="unlabelled", accept=None),
        "eval": make_ep(tmp_path, role="evaluation", outcome="success", accept=None),
        "mock": make_ep(tmp_path, mock=True),
    }
    e = {k: D.entry_from_episode(v) for k, v in cases.items()}
    assert (e["setup"]["role"], e["good"]["role"], e["bad"]["role"], e["aborted"]["role"], e["unlabelled"]["role"], e["eval"]["role"], e["mock"]["role"]) == \
        ("setup_pilot", "successful_demonstration", "unsuccessful_attempt", "unsuccessful_attempt", "unclassified", "evaluation_trial", "synthetic_fixture")
    assert e["mock"]["data_origin"] == "synthetic" and e["good"]["data_origin"] == "recorded_physical"
    assert e["good"]["training_eligible"] and not any(e[k]["training_eligible"] for k in e if k != "good")
    assert all(not D.validate_entry(x) for x in e.values())


def test_unmeasured_poses_stay_null_and_exclude_the_demonstration(tmp_path):
    e = D.entry_from_episode(make_ep(tmp_path, pose=False))
    assert e["cube_start_pose_mat_frame"] is None and e["tray_pose_mat_frame"] is None          # never defaulted or filled in
    assert e["poses_measured"] == {"cube": False, "tray": False}
    assert not e["training_eligible"] and any("not measured" in r for r in e["exclusions"])
    relaxed = D.entry_from_episode(make_ep(tmp_path, pose=False), {"require_measured_poses": False})
    assert relaxed["training_eligible"] and relaxed["cube_start_pose_mat_frame"] is None         # policy relaxed; value still null


def test_curator_acceptance_is_required_never_inferred(tmp_path):
    for acc in (None, False):
        e = D.entry_from_episode(make_ep(tmp_path, accept=acc))
        assert not e["training_eligible"] and any("curator" in r for r in e["exclusions"])


def test_stale_calibration_hash_excludes_the_episode_and_is_reported(tmp_path):
    ep = make_ep(tmp_path)                                    # synthetic fixture records robot calibration sha "abc"
    stale = D.entry_from_episode(ep, stale_calibrations={"abc": "synthetic stale example"})
    assert stale["calibration"]["registry_status"] == "stale" and not stale["training_eligible"]
    assert any("STALE" in r for r in stale["exclusions"])
    fresh = D.entry_from_episode(ep, stale_calibrations={"other": "x"})
    assert fresh["calibration"]["registry_status"] == "not on the stale list" and fresh["training_eligible"]
    assert D.entry_from_episode(ep)["calibration"]["registry_status"].startswith("registry not supplied")


def test_entry_records_camera_calibration_and_semantics(tmp_path):
    e = D.entry_from_episode(make_ep(tmp_path))
    assert e["cameras"]["cam"]["depth"] and e["cameras"]["cam"]["depth_scale"] == 0.001 and e["cameras"]["cam"]["n_frames"] == 4
    assert e["cameras"]["cam"]["camera_json_sha256"]
    assert e["calibration"]["robot_motor_calibration_sha256"] == "abc" and e["calibration"]["physically_validated"] is False
    assert "submitted_not_confirmed" in e["semantics"]["action_candidates"]["commands.raw_goal"]
    assert e["semantics"]["measured_and_commanded_separate"] is True
    assert e["quality"]["validation_ok"] is True and e["outcome"]["horizon"] == "full_task" and e["outcome"]["label_source"] == "operator_visual"


def test_manifest_inventory_counts_only_what_exists(tmp_path):
    eps = [make_ep(tmp_path), make_ep(tmp_path, outcome="failure"), make_ep(tmp_path, mock=True), make_ep(tmp_path, role="setup", outcome="unlabelled", accept=None)]
    sim = D.sim_rollout_entry("sim_0001", __file__, "mujoco", 7, "failure", "abc")
    man = D.build_manifest(eps, extra_entries=[sim])
    inv = man["inventory"]
    assert inv["training_eligible"] == 1 and inv["planned_successful_demonstrations"] == 30 and inv["shortfall_vs_plan"] == 29 and not inv["plan_met"]
    assert inv["by_data_origin"] == {"recorded_physical": 3, "simulated": 1, "synthetic": 1}
    assert inv["by_role"]["simulated_rollout"] == 1 and inv["by_role"]["synthetic_fixture"] == 1
    assert sim["training_eligible"] is False and not D.validate_entry(sim)
    q = D.quality_summary(man)
    assert q["n_valid"] == 4 and len(q["rows"]) == 4


def test_manifest_rejects_duplicates_and_malformed_entries(tmp_path):
    ep = make_ep(tmp_path)
    with pytest.raises(D.DatasetError, match="duplicate"):
        D.build_manifest([ep, ep])
    e = D.entry_from_episode(ep)
    e["data_origin"] = "synthetic"
    e["role"] = "simulated_rollout"
    assert any("simulated_rollout must have" in p for p in D.validate_entry(e))
    e2 = D.entry_from_episode(ep)
    e2["outcome"]["label_source"] = "none"
    assert any("without a label source" in p for p in D.validate_entry(e2))
    e3 = D.entry_from_episode(make_ep(tmp_path, mock=True))
    e3["training_eligible"] = True
    e3["exclusions"] = []
    assert any("non-physical" in p for p in D.validate_entry(e3))


def synth_manifest(n_eligible, n_eval=2):
    """Manifest built from copies of one validated entry; ids are invented. Used only to test split arithmetic."""
    base = None
    ents = []
    import tempfile

    with tempfile.TemporaryDirectory() as t:
        base = D.entry_from_episode(make_ep(Path(t)))
        ev = D.entry_from_episode(make_ep(Path(t), role="evaluation", outcome="success", accept=None))
    for i in range(n_eligible):
        e = copy.deepcopy(base)
        e["entry_id"] = e["episode_id"] = f"demo_{i:03d}"
        e["episode_json_sha256"] = f"{i:064x}"
        ents.append(e)
    for i in range(n_eval):
        e = copy.deepcopy(ev)
        e["entry_id"] = e["episode_id"] = f"eval_{i:03d}"
        ents.append(e)
    man = {"schema": D.DATASET_SCHEMA, "policy": dict(D.DEFAULT_POLICY), "entries": ents}
    man["inventory"] = D.inventory(man)
    man["content_sha256"] = D.manifest_hash(man)
    return man


def test_split_is_provisional_and_proportional_below_plan():
    man7 = synth_manifest(7)
    sp = D.make_split(man7, seed=1)
    assert sp["frozen"] is False and "PROVISIONAL" in sp["status"] and sp["complete"] is False
    assert sp["actual"] == {"train": 6, "val": 1} and sp["shortfall"] == {"train": 19, "val": 4}
    assert set(sp["assignments"].values()) == {"train", "val"} and not set(sp["assignments"]) & set(sp["evaluation_trials"])
    assert D.make_split(man7, seed=1) == sp                                    # deterministic
    assert any(D.make_split(man7, seed=s)["assignments"] != sp["assignments"] for s in range(2, 8))   # the seed matters
    assert D.make_split(synth_manifest(1), seed=1)["actual"] == {"train": 1, "val": 0}
    assert D.make_split(synth_manifest(0), seed=1)["assignments"] == {}


def test_split_freezes_only_at_25_5_and_never_overwrites_when_frozen(tmp_path):
    man = synth_manifest(32)
    sp = D.make_split(man, seed=3)
    assert sp["frozen"] and sp["actual"] == {"train": 25, "val": 5} and len(sp["surplus_unassigned"]) == 2
    assert not D.verify_split(sp, man)
    path = tmp_path / "split.json"
    D.write_split(path, sp)
    D.write_split(path, sp)                                                     # identical content is fine
    with pytest.raises(D.DatasetError, match="FROZEN"):
        D.write_split(path, D.make_split(man, seed=4))
    # provisional files may be rewritten
    prov = tmp_path / "prov.json"
    D.write_split(prov, D.make_split(synth_manifest(5), 1))
    D.write_split(prov, D.make_split(synth_manifest(6), 1))


def test_verify_split_detects_tampering_and_leakage():
    man = synth_manifest(8)
    sp = D.make_split(man, seed=1)
    bad = copy.deepcopy(sp)
    some = next(iter(bad["assignments"]))
    bad["episode_json_sha256"][some] = "f" * 64
    assert any("changed since" in p for p in D.verify_split(bad, man))
    leak = copy.deepcopy(sp)
    leak["assignments"]["eval_000"] = "train"
    leak["episode_json_sha256"]["eval_000"] = "x"
    assert any("not a training-eligible" in p for p in D.verify_split(leak, man))
    assert any("evaluation trial is in train/val" in p for p in D.verify_split(leak, man))
    fake_frozen = copy.deepcopy(sp)
    fake_frozen["frozen"] = True
    assert any("planned counts not met" in p for p in D.verify_split(fake_frozen, man))


def test_applied_split_marks_entries_and_evaluation_is_separate():
    man = synth_manifest(6)
    out = D.apply_split(man, D.make_split(man, 1))
    by = {e["entry_id"]: e["split"] for e in out["entries"]}
    assert by["eval_000"] == "evaluation" and sorted(v for k, v in by.items() if k.startswith("demo")).count("val") == 1
    assert out["content_sha256"] != man["content_sha256"]


def test_reader_detects_tampering_and_is_read_only(tmp_path):
    ep = make_ep(tmp_path)
    before = tree_hash(ep)
    man = D.build_manifest([ep])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    rd = D.DatasetReader(path, root=tmp_path)
    r = rd.episode(man["entries"][0]["entry_id"])
    assert len(list(r.follower_states())) == 20 and len(list(r.commands())) == 8 and r.cameras == ["cam"]
    fr = next(iter(r.frames("cam")))
    assert r.image_path("cam", fr, "depth_native").exists()
    assert tree_hash(ep) == before
    tam = json.loads(path.read_text())
    tam["entries"][0]["role"] = "evaluation_trial"
    (tmp_path / "t.json").write_text(json.dumps(tam))
    with pytest.raises(D.DatasetError, match="hash mismatch"):
        D.DatasetReader(tmp_path / "t.json")
    sim = D.build_manifest([], extra_entries=[D.sim_rollout_entry("s1", __file__, "mujoco", 1, "success", None)])
    (tmp_path / "s.json").write_text(json.dumps(sim))
    with pytest.raises(D.DatasetError, match="no master"):
        D.DatasetReader(tmp_path / "s.json").episode("s1")


def test_importing_the_dataset_module_loads_no_device_or_robot_code():
    code = "import sys, collector.dataset; bad=[m for m in sys.modules if m.startswith('collector.adapters') or m in ('serial','pyrealsense2','lerobot')]; print(bad); sys.exit(1 if bad else 0)"
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_dataset_commands_on_synthetic_episodes(tmp_path, capsys):
    eps = [make_ep(tmp_path) for _ in range(3)] + [make_ep(tmp_path, role="evaluation", outcome="success", accept=None)]
    before = {e.name: tree_hash(e) for e in eps}
    m, s, a = tmp_path / "m.json", tmp_path / "s.json", tmp_path / "ma.json"
    assert cli(["dataset", "manifest", "--episodes-root", str(tmp_path), "--out", str(m)]) == 0
    assert cli(["dataset", "validate", "--manifest", str(m)]) == 0
    assert cli(["dataset", "inventory", "--manifest", str(m)]) == 0
    assert cli(["dataset", "quality", "--manifest", str(m)]) == 0
    assert cli(["dataset", "split", "--manifest", str(m), "--seed", "5", "--out", str(s), "--apply-to", str(a)]) == 0
    assert "PROVISIONAL" in capsys.readouterr().out
    assert cli(["dataset", "verify-split", "--manifest", str(m), "--split", str(s)]) == 0
    assert cli(["dataset", "validate", "--manifest", str(a)]) == 0
    entry = read_json(m)["entries"][0]["entry_id"]
    for prof in ("rgb_joint_angles", "rgbd_motor_positions"):
        assert cli(["dataset", "export", "--manifest", str(m), "--entry", entry, "--profile", prof]) == 0
    assert cli(["dataset", "replay", "--manifest", str(m), "--entry", entry, "--max-frames", "2"]) == 0
    for e in eps:
        assert tree_hash(e) == before[e.name]                                                    # masters byte-identical
    assert len(list((tmp_path / (next(e for e in eps if e.name == entry).name + ".exports")).iterdir())) == 2
