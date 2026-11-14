"""Master writer, exports, recovery, immutability (synthetic fixtures from conftest)."""
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from collector.exports import ExportBlocked, export_episode, load_export
from collector.recover import recover
from collector.util import iter_jsonl, read_json
from collector.validate import validate
from conftest import JOINTS, depth_image, fill, new_writer, raw_for, frame_rec, REC_CFG


def tree_hashes(p):
    from collector.util import sha256_file
    return {f.relative_to(p).as_posix(): sha256_file(f) for f in sorted(Path(p).rglob("*")) if f.is_file()}


def test_master_valid_complete_and_files_read_only(episode):
    r = validate(episode)
    assert r["ok"], r
    assert read_json(episode / "episode.json")["status"] == "complete"
    f = episode / "streams" / "cam" / "frames.jsonl"
    with pytest.raises(PermissionError):
        open(f, "ab")
    assert not list(episode.parent.glob("*.inprogress"))


def test_uint16_depth_preserved_native_and_aligned(episode):
    fr = list(iter_jsonl(episode / "streams" / "cam" / "frames.jsonl"))[3]
    nat = cv2.imread(str(episode / "streams" / "cam" / fr["depth_native_file"]), cv2.IMREAD_UNCHANGED)
    ali = cv2.imread(str(episode / "streams" / "cam" / fr["depth_aligned_file"]), cv2.IMREAD_UNCHANGED)
    assert nat.dtype == np.uint16 and np.array_equal(nat, depth_image(3))
    assert np.array_equal(ali, depth_image(3)[::-1])
    assert nat[0, 0] == 0 and nat[1, 0] == 65535  # invalid stays 0, max preserved


def test_commands_and_measured_state_are_separate_streams(episode):
    st = list(iter_jsonl(episode / "telemetry" / "follower_state.jsonl"))
    cm = list(iter_jsonl(episode / "telemetry" / "commands.jsonl"))
    assert "raw" in st[0] and "values" not in st[0]
    assert "values" in cm[0] and "raw" not in cm[0]
    row = list(iter_jsonl(episode / "sync" / "associations.jsonl"))[2]
    assert set(row) >= {"follower_state", "command_prev", "command_next"}
    assert [s["raw"] for s in st][5] == raw_for(5)  # measured value untouched by command data


def test_associations_have_residuals_and_unmatched_marked(tmp_path):
    w = new_writer(tmp_path)
    fill(w, n_frames=3, n_states=3)  # states only cover first ~6 ms; later frames have no fresh state
    ep = w.finalize()
    rows = list(iter_jsonl(ep / "sync" / "associations.jsonl"))
    assert rows[0]["follower_state"]["status"] == "matched" and rows[0]["follower_state"]["residual_ns"] is not None
    assert rows[2]["follower_state"]["status"] == "unmatched_beyond_tolerance"
    assert read_json(ep / "sync" / "sync_report.json")["hardware_synchronisation"] is False


def test_rgb_joint_angles_export_roundtrip(episode):
    out = export_episode(episode, "rgb_joint_angles")
    ex = load_export(out)
    assert ex["manifest"]["parent"]["episode_id"] == read_json(episode / "episode.json")["episode_id"]
    assert ex["manifest"]["profile"] == "rgb_joint_angles" and ex["manifest"]["complete"]
    assert not list(out.rglob("depth*")) and "motor_positions_raw" not in ex["arrays"]
    assert ex["schema"]["joint_order"] == [j["name"] for j in JOINTS if j["type"] == "revolute"]
    s0 = ex["samples"][0]
    img = cv2.imread(str(out / s0["cameras"]["cam"]["image_path"]))
    src = cv2.imread(str(episode / "streams" / "cam" / "color" / "000000.png"))
    assert np.array_equal(img, src)  # lossless copy
    assert validate(out)["ok"]
    master = list(iter_jsonl(episode / "telemetry" / "follower_state.jsonl"))
    k = s0["state"]["state_seq"]
    assert master[k]["raw"]["shoulder_pan"] > 0 and len(s0["state"]["joint_angles_rad"]) == 5
    assert ex["arrays"]["joint_angles_rad"].shape == (10, 5) and ex["arrays"]["state_valid"].all()
    assert validate(episode)["ok"]  # master untouched


def test_rgbd_export_has_both_depths_raw_motors_and_scale(episode):
    out = export_episode(episode, "rgbd_motor_positions")
    ex = load_export(out)
    s = ex["samples"][4]
    c = s["cameras"]["cam"]
    nat = cv2.imread(str(out / c["depth_native_path"]), cv2.IMREAD_UNCHANGED)
    ali = cv2.imread(str(out / c["depth_aligned_path"]), cv2.IMREAD_UNCHANGED)
    assert nat.dtype == ali.dtype == np.uint16 and np.array_equal(nat, depth_image(4))
    assert s["state"]["motor_positions_raw"] == [raw_for(s["state"]["state_seq"])[j["name"]] for j in JOINTS]
    assert s["state"]["motor_names"] == [j["name"] for j in JOINTS]
    assert "joint_angles_rad" not in s["state"] and "joint_angles_rad_optional" in s["state"]
    assert ex["cameras"]["cam"]["depth_scale"] == 0.001
    assert ex["arrays"]["motor_positions_raw"].dtype.kind == "i"
    assert "depth" in ex["schema"]["units"] and "0 = invalid" in ex["schema"]["units"]["depth"]


def test_full_master_export_contains_everything(episode):
    out = export_episode(episode, "full_master")
    assert validate(out)["ok"]
    master_files = set(read_json(episode / "manifest.json")["files"])
    assert master_files <= set(read_json(out / "manifest.json")["files"])


def test_export_deterministic_and_never_overwrites(episode):
    out = export_episode(episode, "rgbd_motor_positions")
    h1 = read_json(out / "manifest.json")["content_sha256"]
    with pytest.raises(FileExistsError):
        export_episode(episode, "rgbd_motor_positions")
    before = tree_hashes(episode)
    out = export_episode(episode, "rgbd_motor_positions", force=True)
    assert read_json(out / "manifest.json")["content_sha256"] == h1
    assert tree_hashes(episode) == before  # master immutable across exports


def test_missing_calibration_blocks_joint_angle_export(tmp_path):
    w = new_writer(tmp_path, with_cal=False)
    fill(w)
    ep = w.finalize()
    with pytest.raises(ExportBlocked) as e:
        export_episode(ep, "rgb_joint_angles")
    assert any("joint-angle export blocked" in m for m in e.value.missing)
    out = export_episode(ep, "rgbd_motor_positions")  # raw motor positions still possible
    ex = load_export(out)
    assert ex["manifest"]["complete"] and "joint_angles_rad_optional" not in ex["samples"][0]["state"]
    part = export_episode(ep, "rgb_joint_angles", allow_partial=True)
    m = read_json(part / "manifest.json")
    assert m["complete"] is False and m["missing_required_signals"]


def test_missing_depth_blocks_rgbd(tmp_path):
    from conftest import SYNC_CFG, new_writer as nw
    w = nw(tmp_path)
    for i in range(3):  # RGB-only frames
        fr, rec = frame_rec(i)
        fr.depth_native = fr.depth_aligned = None
        w.submit("cam", "frame", (fr, rec))
    for k in range(50):
        t = 10_000_000_000 + k * 3_000_000
        w.submit("follower_state", "telemetry", {"seq": k, "t_start_ns": t, "t_end_ns": t, "wall_ns": 0,
                                                 "read_ok": True, "raw": raw_for(k)})
    ep = w.finalize()
    with pytest.raises(ExportBlocked, match="depth"):
        export_episode(ep, "rgbd_motor_positions")
    assert export_episode(ep, "rgb_joint_angles")


def test_writer_overflow_flag_and_drop_is_counted_and_flagged(tmp_path, monkeypatch):
    from collector.storage import EpisodeWriter
    orig = EpisodeWriter._png
    monkeypatch.setattr(EpisodeWriter, "_png", lambda self, rel, arr: (time.sleep(0.05), orig(self, rel, arr))[1])
    w = new_writer(tmp_path, rec_cfg={**REC_CFG, "queue_max_items": 2})
    accepted = sum(w.submit("cam", "frame", frame_rec(i)) for i in range(20))
    ep = w.finalize()
    m = read_json(ep / "episode.json")
    assert accepted < 20 and m["status"] == "complete_with_drops"
    assert m["counts"]["dropped"]["cam"] == 20 - accepted and m["quality_flags"]["data_dropped"]
    v = validate(ep)
    assert v["ok"] and any("dropped" in x for x in v["warnings"])
    assert any(e["event"] == "queue_overflow" for e in iter_jsonl(ep / "events.jsonl"))
    with pytest.raises(ExportBlocked, match="follower_state"):  # missing signal reported, not silently dropped
        export_episode(ep, "rgb_joint_angles")


def test_writer_overflow_abort_policy(tmp_path, monkeypatch):
    from collector.storage import EpisodeWriter
    orig = EpisodeWriter._png
    monkeypatch.setattr(EpisodeWriter, "_png", lambda self, rel, arr: (time.sleep(0.05), orig(self, rel, arr))[1])
    w = new_writer(tmp_path, rec_cfg={**REC_CFG, "queue_max_items": 1, "overflow_policy": "abort"})
    for i in range(10):
        w.submit("cam", "frame", frame_rec(i))
    assert w.abort_reason
    ep = w.finalize("success")
    m = read_json(ep / "episode.json")
    assert m["status"] == "aborted_writer_overflow" and m["outcome"] == "aborted"
    with pytest.raises(ExportBlocked):
        export_episode(ep, "rgb_joint_angles")  # partial episode never presented as complete


def test_interrupted_episode_is_recoverable_and_flagged(tmp_path):
    w = new_writer(tmp_path)
    fill(w, n_frames=5, n_states=40, stream_cmd=False)
    w._closing.set()  # the writer drains its queue on closing; join() below is the deterministic barrier
    w.thread.join()
    for f in w._files.values():  # simulate crash: no finalize; torn last line + orphan temp file
        f.flush()
    with open(w.dir / "telemetry" / "follower_state.jsonl", "ab") as f:
        f.write(b'{"seq": 999, "t_sta')
    inprog = w.dir
    assert validate(inprog)["ok"] is False  # not presentable as complete
    ep = recover(inprog)
    m = read_json(ep / "episode.json")
    assert m["status"] == "interrupted_recovered" and m["quality_flags"]["recovered"]
    assert m["quality_flags"]["recovery_report"]["follower_state"]["discarded"] == 1
    assert len(list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl", tolerant=False))) == 40
    assert validate(ep)["ok"]
    with pytest.raises(ExportBlocked):
        export_episode(ep, "rgb_joint_angles")


def test_existing_episode_not_overwritten(tmp_path):
    w = new_writer(tmp_path)
    ep = w.finalize()
    from collector.storage import EpisodeWriter
    from conftest import SYNC_CFG
    with pytest.raises(FileExistsError):
        EpisodeWriter(tmp_path, {"streams": {}}, REC_CFG, SYNC_CFG, episode_id=ep.name)


def test_tamper_detection(episode):
    f = episode / "telemetry" / "follower_state.jsonl"
    os.chmod(f, 0o644)
    f.write_bytes(f.read_bytes() + b" ")
    assert not validate(episode)["ok"]


def test_replay_is_offline_only(episode):
    import collector.replay as r
    src = Path(r.__file__).read_text()
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not any("adapters" in ln or "serial" in ln or "recorder" in ln for ln in imports)
    assert "send_action" not in src and "Goal_Position" not in src
    out = export_episode(episode, "rgb_joint_angles")
    assert r.replay(out, display=False) == 10 and r.replay(episode, display=False) == 10
