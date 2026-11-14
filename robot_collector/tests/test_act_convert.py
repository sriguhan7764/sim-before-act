"""Milestone C: ACT dataset conversion, lineage, action-segment file. SYNTHETIC fixtures only (fabricated; not real demonstrations).

numpy-only (runs in the guhan_c env). The torch side is covered by tests/test_act_torch_smoke.py (subprocess, lerobot env).
"""
import copy
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from collector import dataset as D
from collector.act import align, segment
from collector.act import convert as C
from collector.act import synthetic as S
from collector.act.contract import (JOINT_NAMES, PROVISIONAL_LABEL, SEGMENT_SMOKE_PROVENANCE, SMOKE_KIND, SYNTHETIC_DATA_LABEL, UNITS)
from collector.act.load import Converted, ConvertedDatasetError
from collector.util import canonical_json, read_json, sha256_bytes

CFG = dict(wrist_stream="wrist", image_size=(24, 32), chunk_size=8, n_action_steps=8)


def build(tmp, specs=None, n=4):
    return S.build_fixture(tmp / "fx", specs or S.default_specs(n, n_frames=30))


def convert(fx, out, allow=True, **over):
    return C.convert_dataset(fx["manifest"], fx["split"], out, C.ConvertConfig(**{**CFG, **over}), allow_provisional_split=allow)


@pytest.fixture(scope="module")
def default(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("act_default")
    fx = build(tmp)
    return fx, convert(fx, tmp / "conv"), tmp / "conv"


def faulty_specs(frames=30):
    specs = S.default_specs(4, n_frames=frames)
    specs[0] = S.EpisodeSpec("synthetic_fixture_ep_000", n_frames=frames, offset=0.0, seed=100, state_invalid_frames=(10,),
                             command_gap_frames=(15,), wrist_gap_frames=(20,))
    return specs


def ep_audit(conv_dir, eid):
    return next(e for e in read_json(Path(conv_dir) / "conversion_audit.json")["episodes"] if e["episode_id"] == eid)


# ---------------------------------------------------------------- schema / shapes / units
def test_converted_schema_shapes_units_and_labels(default):
    fx, conv, d = default
    c = Converted(d)
    n = c.n
    assert n > 0 and c.a["state"].shape == (n, 6) and c.a["action"].shape == (n, 6) and c.a["raw_goal"].shape == (n, 6)
    assert c.a["state"].dtype == np.float64 and c.a["action"].dtype == np.float64 and c.a["raw_goal"].dtype.kind == "i"
    assert conv["joint_names"] == list(JOINT_NAMES) and conv["units"] == list(UNITS) and UNITS[-1] == "percent_0_100"
    assert conv["data_origin"] == "synthetic_fixture" and not conv["suitable_for_main_training"]
    assert SYNTHETIC_DATA_LABEL in conv["labels"] and PROVISIONAL_LABEL in conv["labels"]
    assert conv["signals"]["action"]["source"] == "commands.values" and conv["signals"]["action"]["substitution"] is False
    assert conv["cameras"]["front"]["shape_hwc"] == [24, 32, 3] and conv["cameras"]["wrist"]["shape_hwc"] == [24, 32, 3]
    assert abs(conv["control_hz"] - 30.0) < 0.5
    assert 0 <= c.a["action"][:, 5].min() and c.a["action"][:, 5].max() <= 100          # gripper in the 0..100 convention
    assert (c.a["state_age_ns"] >= 0).all() and (c.a["state_age_ns"] <= 40e6).all() and (c.a["command_lag_ns"] >= 0).all()
    audit = read_json(d / "conversion_audit.json")
    assert "not fed" in audit["depth"] and all(e["depth"]["fed_to_model"] is False and e["depth"]["files_opened"] == 0 for e in audit["episodes"])
    md = (d / "conversion_audit.md").read_text()
    assert "SYNTHETIC FIXTURE DATA" in md and "PROVISIONAL SPLIT" in md and "calibration-mismatch BLOCKER" in md
    assert audit["lineage"]["manifest_content_sha256"] and audit["lineage"]["split_sha256"]
    assert {e["episode_id"] for e in audit["lineage"]["source_episodes"]} == {e["episode_id"] for e in conv["episodes"]}
    assert all(len(e["episode_json_sha256"]) == 64 for e in audit["lineage"]["source_episodes"])
    assert audit["normalization"]["sha256"] == conv["normalization_sha256"]


def test_action_is_commanded_target_not_leader_and_state_is_measured(default):
    fx, conv, d = default
    c = Converted(d)
    er = D.DatasetReader(fx["manifest"]).episode("synthetic_fixture_ep_001")
    cmds = {r["seq"]: r for r in er.commands()}
    states = {r["seq"]: r for r in er.follower_states()}
    rows = np.nonzero(c.a["episode_idx"] == next(e["index"] for e in conv["episodes"] if e["episode_id"] == "synthetic_fixture_ep_001"))[0]
    for r in rows[:10]:
        cm, st = cmds[int(c.a["command_seq"][r])], states[int(c.a["state_seq"][r])]
        np.testing.assert_array_equal(c.a["action"][r], [cm["values"][j] for j in JOINT_NAMES])             # byte-exact, no clipping
        np.testing.assert_array_equal(c.a["raw_goal"][r], [cm["raw_goal"][j] for j in JOINT_NAMES])         # retained source column
        np.testing.assert_array_equal(c.a["leader_measurement"][r], [cm["leader_measurement_normalized"][j] for j in JOINT_NAMES])
        np.testing.assert_array_equal(c.a["state"][r], [st["normalized"][j] for j in JOINT_NAMES])
    assert np.abs(c.a["action"] - c.a["leader_measurement"]).min() > 0.5       # the fixture's leader differs by 0.7: not substituted
    assert np.abs(c.a["action"] - c.a["state"]).max() > 0.1                    # measured state is not the commanded target
    assert (c.a["command_t_ns"] >= c.a["t_ref_ns"]).all() and (c.a["state_t_ns"] <= c.a["t_ref_ns"]).all()   # causal rule


def test_signal_substitution_needs_the_flag_and_is_recorded(tmp_path):
    fx = build(tmp_path)
    with pytest.raises(C.ConversionError, match="allow_signal_substitution"):
        convert(fx, tmp_path / "a", action_signal="leader_measurement_normalized")
    conv = convert(fx, tmp_path / "b", action_signal="leader_measurement_normalized", allow_signal_substitution=True)
    assert conv["signals"]["action"]["substitution"] is True and conv["signals"]["action"]["source"] != "commands.values"
    c = Converted(tmp_path / "b")
    np.testing.assert_array_equal(c.a["action"], c.a["leader_measurement"])
    assert "SUBSTITUT" in (tmp_path / "b" / "conversion_audit.md").read_text().upper()
    assert C.verify_lineage(tmp_path / "b", fx["manifest"])["ok"]


# ---------------------------------------------------------------- lineage
def test_lineage_round_trip_and_tamper_detection(default, tmp_path):
    fx, conv, d = default
    r = C.verify_lineage(d, fx["manifest"])
    assert r["ok"] and r["n_rows"] == Converted(d).n and r["n_images_checked"] > 0
    assert r["n_episodes"] == 4
    # an edited converted image is caught by the pixel hash index
    import shutil
    import cv2

    cp = tmp_path / "conv_copy"
    shutil.copytree(d, cp)
    img = sorted((cp / "images" / "front").rglob("*.png"))[0]
    os.chmod(img, 0o644)
    a = cv2.imread(str(img))
    a[0, 0] ^= 255
    cv2.imwrite(str(img), a)
    bad = C.verify_lineage(cp, fx["manifest"])
    assert not bad["ok"] and any("image" in e for e in bad["errors"])
    # an edited steps array is caught by the content hash
    z = dict(np.load(cp / "steps.npz"))
    z["action"] = z["action"] + 0.5
    np.savez(cp / "steps.npz", **z)
    with pytest.raises(ConvertedDatasetError, match="steps.npz"):
        Converted(cp)


def test_lineage_detects_changed_manifest(default, tmp_path):
    fx, conv, d = default
    man = read_json(fx["manifest"])
    man["entries"][0]["episode_json_sha256"] = "0" * 64
    man["content_sha256"] = D.manifest_hash(man)
    res = C.verify_lineage(d, man, root=Path(fx["manifest"]).parent)
    assert not res["ok"] and any("episode_json_sha256" in e or "manifest" in e for e in res["errors"])


def test_converted_dir_is_never_overwritten(default):
    fx, conv, d = default
    with pytest.raises(C.ConversionError, match="not empty"):
        convert(fx, d)


# ---------------------------------------------------------------- dropped steps, padding, boundaries
def test_dropped_steps_are_counted_by_reason_and_never_filled(tmp_path):
    fx = build(tmp_path, faulty_specs())
    convert(fx, tmp_path / "c")
    c = Converted(tmp_path / "c")
    a = ep_audit(tmp_path / "c", "synthetic_fixture_ep_000")
    assert a["dropped_steps"] == [[0, "wrist_no_frame_at_or_before_ref"], [10, "state_too_old"], [15, "command_lag_too_large"],
                                  [19, "wrist_frame_too_old"], [20, "wrist_frame_too_old"], [21, "wrist_frame_too_old"]]
    assert a["n_kept"] == 24 and a["n_kept"] + a["n_dropped"] == a["n_front_frames"] == 30
    idx = next(e["index"] for e in c.conv["episodes"] if e["episode_id"] == "synthetic_fixture_ep_000")
    kept = set(c.a["grid_index"][c.a["episode_idx"] == idx].tolist())
    assert not kept & {0, 10, 15, 19, 20, 21} and len(kept) == 24                 # no row exists for a dropped step
    assert (c.a["state_age_ns"] <= 40e6).all() and (c.a["command_lag_ns"] <= 50e6).all()    # nothing stale was accepted
    r = {"dropped": a["dropped_by_reason"]}
    assert r["dropped"] == {"wrist_no_frame_at_or_before_ref": 1, "state_too_old": 1, "command_lag_too_large": 1, "wrist_frame_too_old": 3}


def test_chunk_padding_marks_dropped_steps_and_episode_end_and_never_crosses_episodes(tmp_path):
    fx = build(tmp_path, faulty_specs())
    convert(fx, tmp_path / "c")
    c = Converted(tmp_path / "c")
    idx = next(e["index"] for e in c.conv["episodes"] if e["episode_id"] == "synthetic_fixture_ep_000")
    ep_rows = {int(c.a["grid_index"][r]): int(r) for r in np.nonzero(c.a["episode_idx"] == idx)[0]}
    act, pad = c.chunk(ep_rows[8], 8)                                   # positions 8..15; 10 and 15 were dropped
    assert pad.tolist() == [False, False, True, False, False, False, False, True]
    np.testing.assert_array_equal(act[0], c.a["action"][ep_rows[8]])
    np.testing.assert_array_equal(act[3], c.a["action"][ep_rows[11]])
    np.testing.assert_array_equal(act[2], act[1])                       # padded value repeats the last valid action, not a neighbour's
    last = max(ep_rows)
    act, pad = c.chunk(ep_rows[last], 8)
    assert pad.tolist() == [False] + [True] * 7
    assert (act == c.a["action"][ep_rows[last]]).all()                  # episode end: padded, never the next episode's action
    for r in range(c.n):                                                # no chunk uses a row of another episode
        e, g = int(c.a["episode_idx"][r]), int(c.a["grid_index"][r])
        a2, p2 = c.chunk(r, 8)
        for j in np.nonzero(~p2)[0]:
            rr = [q for q in range(c.n) if c.a["episode_idx"][q] == e and c.a["grid_index"][q] == g + j]
            assert len(rr) == 1 and (a2[j] == c.a["action"][rr[0]]).all()
    ps = c.padding_summary(8)
    assert ps["rows_with_padding_from_dropped_steps"] > 0 and ps["rows_with_episode_end_padding"] > 0


# ---------------------------------------------------------------- wrist policy
def test_wrist_max_age_never_copies_a_stale_frame_and_reuse_is_bounded(tmp_path):
    fx = build(tmp_path, faulty_specs())
    convert(fx, tmp_path / "c")
    c = Converted(tmp_path / "c")
    assert (c.a["wrist_age_ns"][c.a["wrist_valid"]] <= 50e6).all() and c.a["wrist_valid"].all()
    from collections import Counter

    use = Counter(zip(c.a["episode_idx"].tolist(), c.a["wrist_frame_index"].tolist()))
    assert max(use.values()) <= 2
    tight = convert(fx, tmp_path / "t", wrist_max_reuse=1, wrist_max_age_ms=100.0)
    c2 = Converted(tmp_path / "t")
    use2 = Counter(zip(c2.a["episode_idx"].tolist(), c2.a["wrist_frame_index"].tolist()))
    assert max(use2.values()) == 1
    audit2 = read_json(tmp_path / "t" / "conversion_audit.json")
    assert audit2["totals"]["dropped_by_reason"].get("wrist_reuse_exceeded", 0) > 0
    assert tight["counts"]["train_rows"] < Converted(tmp_path / "c").row_indices("train").size


def test_wrist_missing_policy_mark_missing_with_mask_keeps_rows_without_stale_copies(tmp_path):
    fx = build(tmp_path, faulty_specs())
    convert(fx, tmp_path / "m", wrist_missing_policy="mark_missing_with_mask")
    c = Converted(tmp_path / "m")
    a = ep_audit(tmp_path / "m", "synthetic_fixture_ep_000")
    assert a["wrist"]["rows_masked_missing"] == 4 and a["n_kept"] == 28 and a["dropped_by_reason"] == {"state_too_old": 1, "command_lag_too_large": 1}
    assert not c.a["wrist_valid"].all()
    miss = ~c.a["wrist_valid"]
    assert (c.a["wrist_frame_index"][miss] == -1).all() and (c.a["wrist_age_ns"][miss] == -1).all()
    assert c.image_path("wrist", int(np.nonzero(miss)[0][0])) is None
    assert len(c.row_indices("val", require_wrist=True)) < len(c.row_indices("val", require_wrist=False))
    assert C.verify_lineage(tmp_path / "m", fx["manifest"])["ok"]


def test_absent_wrist_stream_drops_or_masks_instead_of_inventing(tmp_path):
    fx = build(tmp_path, S.default_specs(4, n_frames=20, with_wrist=False))
    with pytest.raises(C.ConversionError):
        convert(fx, tmp_path / "x")                                      # every step dropped -> nothing to train on
    convert(fx, tmp_path / "front_only", wrist_stream=None)
    assert "wrist" not in Converted(tmp_path / "front_only").cameras
    with pytest.raises(C.ConversionError, match="no image was accepted"):          # masking cannot invent a whole camera
        convert(fx, tmp_path / "masked", wrist_missing_policy="mark_missing_with_mask")


# ---------------------------------------------------------------- normalisation
def stats_and_hash(out):
    return read_json(Path(out) / "normalization_stats.json")


def test_normalisation_uses_train_split_only_and_matches_the_train_rows(default, tmp_path):
    fx, conv, d = default
    c = Converted(d)
    st = stats_and_hash(d)
    tr = c.a["split_code"] == 0
    np.testing.assert_allclose(st["stats"]["observation.state"]["mean"], c.a["state"][tr].mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(st["stats"]["action"]["std"], c.a["action"][tr].std(axis=0), rtol=1e-9)
    assert st["fitted_on"] == "train_split_only" and st["sha256"] == conv["normalization_sha256"]
    assert sorted(st["train_episode_ids"]) == sorted(k for k, v in read_json(fx["split"])["assignments"].items() if v == "train")
    val_id = next(k for k, v in read_json(fx["split"])["assignments"].items() if v == "val")

    def variant(tag, changed_id, offset):
        specs = S.default_specs(4, n_frames=30)
        specs = [S.EpisodeSpec(s.episode_id, n_frames=30, offset=offset if s.episode_id == changed_id else s.offset, seed=s.seed,
                               cube_id=s.cube_id) for s in specs]
        f = S.build_fixture(tmp_path / tag, specs)
        return f, convert(f, tmp_path / tag / "o")

    f_val, c_val = variant("v", val_id, 40.0)                           # only the VAL episode's data differs
    assert read_json(f_val["split"])["assignments"] == read_json(fx["split"])["assignments"]
    assert c_val["normalization_sha256"] == conv["normalization_sha256"]            # stats unchanged
    assert c_val["conversion_sha256"] != conv["conversion_sha256"]                  # but the dataset is different
    train_id = next(k for k, v in read_json(fx["split"])["assignments"].items() if v == "train")
    f_tr, c_tr = variant("t", train_id, 40.0)
    assert c_tr["normalization_sha256"] != conv["normalization_sha256"]             # a TRAIN change does move the stats


# ---------------------------------------------------------------- refusals
def test_provisional_split_refused_unless_flag_and_flag_is_recorded(tmp_path):
    fx = build(tmp_path)
    assert read_json(fx["split"])["frozen"] is False
    with pytest.raises(C.ProvisionalSplitError, match="allow-provisional-split"):
        convert(fx, tmp_path / "no", allow=False)
    assert not (tmp_path / "no").exists()                                # nothing written
    conv = convert(fx, tmp_path / "yes", allow=True)
    assert conv["source"]["split"]["allow_provisional_split_flag"] is True and conv["source"]["split"]["provisional"] is True
    assert PROVISIONAL_LABEL in conv["labels"] and not conv["suitable_for_main_training"]
    audit = read_json(tmp_path / "yes" / "conversion_audit.json")
    assert audit["allow_provisional_split_flag_used"] is True
    assert "allow-provisional-split` used: **True**" in (tmp_path / "yes" / "conversion_audit.md").read_text()


def test_non_training_eligible_entry_is_refused(tmp_path):
    specs = S.default_specs(4, n_frames=20)
    specs.append(S.EpisodeSpec("synthetic_fixture_ep_099", n_frames=20, offset=9.0, seed=9, accept=False))      # no curator acceptance
    fx = build(tmp_path, specs)
    man, split = read_json(fx["manifest"]), read_json(fx["split"])
    bad = next(e for e in man["entries"] if e["entry_id"] == "synthetic_fixture_ep_099")
    assert not bad["training_eligible"]
    split["assignments"][bad["entry_id"]] = "train"
    split["episode_json_sha256"][bad["entry_id"]] = bad["episode_json_sha256"]
    split["assignments"] = dict(sorted(split["assignments"].items()))
    split["episode_json_sha256"] = dict(sorted(split["episode_json_sha256"].items()))
    split["split_sha256"] = sha256_bytes(canonical_json({k: split[k] for k in ("seed", "plan", "assignments", "episode_json_sha256")}).encode())
    Path(fx["split"]).write_text(json.dumps(split))
    with pytest.raises(C.NotTrainingEligibleError, match="not training_eligible"):
        convert(fx, tmp_path / "o")


def test_synthetic_fixtures_are_never_counted_as_recorded_demonstrations(tmp_path):
    fx = build(tmp_path)
    man, split = read_json(fx["manifest"]), read_json(fx["split"])
    inv = D.inventory(man)
    assert all(e["role"] == "synthetic_fixture" and e["data_origin"] == "synthetic" and not e["training_eligible"]
               for e in man["entries"])
    assert split["data_origin"] == "synthetic_fixture" and split["frozen"] is False and "SYNTHETIC" in split["status"]
    assert inv["training_eligible"] == 0 and inv["recorded_physical_entries"] == 0
    assert set(inv["by_role"]) == {"synthetic_fixture"} and set(inv["by_data_origin"]) == {"synthetic"}
    # a physical split built from the same manifest has no eligible episodes to take
    assert D.make_split(man, 1)["assignments"] == {}


def test_edited_split_is_refused(tmp_path):
    fx = build(tmp_path)
    split = read_json(fx["split"])
    k = next(k for k, v in split["assignments"].items() if v == "train")
    split["assignments"][k] = "val"
    Path(fx["split"]).write_text(json.dumps(split))
    with pytest.raises(C.ConversionError, match="split_sha256"):
        convert(fx, tmp_path / "o")


def test_ticks_format_commands_and_missing_measured_state_are_refused(tmp_path):
    fx = build(tmp_path, S.default_specs(4, n_frames=20, units="feetech_goal_position_register_ticks (synthetic old format)"))
    with pytest.raises(C.ConversionError, match="LeRobot-normalised"):
        convert(fx, tmp_path / "o")


def test_edited_master_file_is_refused(tmp_path):
    fx = build(tmp_path)
    ep = Path(fx["root"]) / "synthetic_fixture_ep_001" / "telemetry" / "commands.jsonl"
    os.chmod(ep, 0o644)
    ep.write_bytes(ep.read_bytes() + b"\n")
    with pytest.raises(C.ConversionError, match="differs from the master manifest hash"):
        convert(fx, tmp_path / "o")


def test_config_validation():
    with pytest.raises(C.ConversionError):
        C.ConvertConfig(wrist_missing_policy="copy_last").validate()
    with pytest.raises(C.ConversionError):
        C.ConvertConfig(n_action_steps=200, chunk_size=100).validate()
    with pytest.raises(C.ConversionError):
        C.ConvertConfig(max_state_age_ms=-1).validate()
    for name in ("act_convert", "act_smoke"):
        C.ConvertConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml")


def test_alignment_primitives():
    t, v = [10, 20, 30, 40], [True, False, False, True]
    assert align.latest_valid_at_or_before(t, v, 35, 100)[:2] == (0, 25)                 # invalid records are skipped, never filled
    assert align.latest_valid_at_or_before(t, v, 35, 20)[0] is None                       # too old -> no match, no fallback
    assert align.first_at_or_after(t, 20, 5)[:2] == (1, 0) and align.first_at_or_after(t, 21, 5)[0] is None
    assert not align.is_non_decreasing([3, 2])


# ---------------------------------------------------------------- action segment (pure numpy)
def good_segment(**kw):
    acts = np.linspace(0, 1, 6 * 5).reshape(5, 6) * 10.0 - 3.0
    base = dict(state_values=[1, 2, 3, 4, 5, 50.0], control_hz=30.0, data_origin="synthetic_fixture",
                provenance_label=SEGMENT_SMOKE_PROVENANCE, policy={"kind": SMOKE_KIND, "trained_for_task": False, "checkpoint_id": "x"},
                execution_horizon_steps=3)
    base.update(kw)
    return segment.build_action_segment(acts, **base), acts


def test_action_segment_roundtrip_hash_recipe_and_validation(tmp_path):
    seg, acts = good_segment()
    assert segment.validate_action_segment_dict(seg) == []
    import hashlib

    assert seg["action_sha256"] == hashlib.sha256(np.ascontiguousarray(acts, dtype="<f8").tobytes()).hexdigest()
    import json

    # documented recipe: sorted keys, compact separators
    cid = hashlib.sha256(json.dumps(seg["contract"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert seg["segment_id"] == hashlib.sha256((seg["action_sha256"] + cid).encode()).hexdigest()[:16]
    assert seg["chunk"] == {"length": 5, "start_step": 0, "execution_horizon_steps": 3, "temporal_ensembling": False, "replanning": False}
    p = tmp_path / "s.json"
    written = segment.write_action_segment(p, acts, state_values=[1, 2, 3, 4, 5, 50.0], control_hz=30.0, data_origin="synthetic_fixture",
                                           provenance_label=SEGMENT_SMOKE_PROVENANCE,
                                           policy={"kind": SMOKE_KIND, "trained_for_task": False, "checkpoint_id": "x"})
    loaded, a2 = segment.load_action_segment(p)
    assert loaded == json.loads(p.read_text()) and np.array_equal(a2, acts) and written["segment_id"] == loaded["segment_id"]
    assert a2.dtype == np.dtype("<f8")                                   # values are written unmodified (not clipped/rounded)


def test_action_segment_rejects_tampering_and_dishonest_labels():
    seg, _ = good_segment()
    v = segment.validate_action_segment_dict
    t = copy.deepcopy(seg)
    t["actions"][0][0] += 1e-9
    assert any("action_sha256" in p for p in v(t))
    t = copy.deepcopy(seg)
    t["policy"]["trained_for_task"] = True
    assert any("trained_for_task" in p for p in v(t))
    t = copy.deepcopy(seg)
    t["provenance_label"] = "a trained policy"
    assert v(t)
    t = copy.deepcopy(seg)
    t["contract"]["units"][5] = "radians"
    assert any("contract.units" in p for p in v(t))
    t = copy.deepcopy(seg)
    t["chunk"]["execution_horizon_steps"] = 99
    assert v(t)
    t = copy.deepcopy(seg)
    t["actions"][1][2] = float("nan")
    assert v(t)
    t = copy.deepcopy(seg)
    t["input"]["state_values"][0] = 9.0
    assert any("state_sha256" in p for p in v(t))
    t = copy.deepcopy(seg)
    t["data_origin"] = "made_up"
    assert v(t)
    with pytest.raises(ValueError):
        segment.write_action_segment("unused.json", np.zeros((3, 5)), state_values=[0] * 6, control_hz=30.0, data_origin="synthetic_fixture",
                                     provenance_label=SEGMENT_SMOKE_PROVENANCE, policy=segment.stub_policy())
    stub, _ = good_segment(policy=segment.stub_policy(), provenance_label="SYNTHETIC scripted ramp - fixture stub")
    assert v(stub) == []


# ---------------------------------------------------------------- hardware boundary
def test_act_package_has_no_hardware_imports_and_no_import_time_torch():
    import ast
    import importlib

    pkg = Path(C.__file__).parent
    banned = ("serial", "pyrealsense2", "lerobot.robots", "lerobot.teleoperators", "collector.lerobot_bridge", "collector.adapters.realsense")
    for p in pkg.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                assert not any(n == b or n.startswith(b + ".") for b in banned), f"{p.name} imports {n}"
                if p.name not in ("train.py", "infer.py", "adapter.py", "smoke.py"):
                    assert n.split(".")[0] != "torch", f"{p.name} imports torch at module level or in a function"
    had_torch = "torch" in sys.modules
    for m in ("collector.act", "collector.act.contract", "collector.act.segment", "collector.act.align", "collector.act.load",
              "collector.act.convert", "collector.act.synthetic", "collector.act.train", "collector.act.infer"):
        importlib.import_module(m)
    assert ("torch" in sys.modules) == had_torch                       # importing the package never imports torch
