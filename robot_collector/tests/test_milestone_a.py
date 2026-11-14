"""Milestone A: timestamp domains, association boundaries, depth freshness/age, thresholds, cleanup, lineage.

SYNTHETIC: every signal is generated from explicit integers; there is no wall-clock dependence in the assertions.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from collector.acceptance import run_acceptance
from collector.config import ConfigError, load_config
from collector.exports import export_episode, load_export
from collector.recorder import Recorder
from collector.sync import build_associations, depth_age_ms, depth_freshness, match_nearest
from collector.util import iter_jsonl, read_json, sha256_file
from collector.validate import validate
from conftest import JOINTS, SYNC_CFG, T0, fill, frame_rec, new_writer, raw_for

MS = 1_000_000


# ---------------------------------------------------------------- association boundaries
def test_match_tolerance_is_inclusive_and_beyond_is_unmatched_not_reused():
    t = [0, 10 * MS, 30 * MS]
    assert match_nearest(10 * MS + 5 * MS, t, 5 * MS)[2] == "matched"          # residual exactly at the tolerance
    k, res, st = match_nearest(10 * MS + 5 * MS + 1, t, 5 * MS)                  # one nanosecond further
    assert (k, st, res) == (None, "unmatched_beyond_tolerance", -(5 * MS + 1))
    assert match_nearest(100 * MS, t, 5 * MS)[0] is None                         # far-future reference: stale value never reused
    assert match_nearest(0, [], MS)[2] == "no_samples"


def test_match_directions_and_invalid_candidates_and_duplicate_times():
    t = [0, 10 * MS, 10 * MS, 20 * MS]
    k, res, st = match_nearest(10 * MS, t, MS)                                   # duplicate timestamps: deterministic pick
    assert st == "matched" and res == 0 and t[k] == 10 * MS
    assert match_nearest(15 * MS, t, 20 * MS, direction="before")[0] in (1, 2)   # latest at/before, never a future sample
    assert match_nearest(15 * MS, t, 20 * MS, direction="after")[0] == 3
    assert match_nearest(5 * MS, t, 20 * MS, direction="before")[0] == 0
    assert match_nearest(-1, t, 20 * MS, direction="before")[2] == "no_valid_candidate"
    assert match_nearest(25 * MS, t, 20 * MS, direction="after")[2] == "no_valid_candidate"
    ok = [True, False, False, True]                                              # invalid reads are skipped, not interpolated over
    k, _, st = match_nearest(10 * MS, t, 20 * MS, valid=ok)
    assert st == "matched" and k in (0, 3)
    assert match_nearest(10 * MS, t, 20 * MS, valid=[False] * 4)[2] == "no_valid_candidate"


def _episode(tmp, depth_numbers, host_step_ns=33 * MS, device_offset_ms=5000.0, host_times=None, n_states=120):
    w = new_writer(tmp)
    fill(w, n_frames=0, n_states=n_states)
    for i, dn in enumerate(depth_numbers):
        t = host_times[i] if host_times else T0 + i * host_step_ns
        f, r = frame_rec(i, t)
        r["device_timestamp_ms"] = f.device_timestamp_ms = device_offset_ms + i * 33.3
        if dn is None:
            f.depth_native = f.depth_aligned = None
            r["depth_frame_number"] = f.depth_frame_number = None
            r["depth_device_timestamp_ms"] = None
        else:
            f.depth_frame_number = r["depth_frame_number"] = dn
            r["depth_device_timestamp_ms"] = dn * 33.3
        w.submit("cam", "frame", (f, r))
    return w.finalize("unlabelled")


def test_association_depends_only_on_host_time_never_on_device_clock_domain(tmp_path):
    a = _episode(tmp_path / "a", list(range(8)), device_offset_ms=5_000.0)
    b = _episode(tmp_path / "b", list(range(8)), device_offset_ms=987_654_321.0)    # wildly different device clock domain
    ra, rep_a = build_associations(a, SYNC_CFG)
    rb, rep_b = build_associations(b, SYNC_CFG)
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("frame_number",)} for r in rows]
    assert strip(ra) == strip(rb)
    for rep in (rep_a, rep_b):
        assert rep["device_clock_offset_estimates"]["cam"]["used_for_association"] is False
        assert rep["hardware_synchronisation"] is False
    offs = [r["device_clock_offset_estimates"]["cam"]["median_host_minus_device_ms"] for r in (rep_a, rep_b)]
    assert offs[0] != offs[1]                                                         # reported as information only


def test_non_monotonic_host_timestamps_fail_validation_and_duplicates_warn(tmp_path):
    times = [T0, T0 + 33 * MS, T0 + 20 * MS, T0 + 99 * MS]                           # third frame goes backwards
    ep = _episode(tmp_path / "nm", [1, 2, 3, 4], host_times=times)
    res = validate(ep)
    assert not res["ok"] and any("camera cam: timestamps not monotonic" in e for e in res["errors"])
    dup = _episode(tmp_path / "dup", [1, 2, 3, 4], host_times=[T0, T0 + 33 * MS, T0 + 33 * MS, T0 + 99 * MS])
    res = validate(dup)
    assert res["ok"] and any("duplicate consecutive timestamps" in x for x in res["warnings"])


def test_missing_samples_are_flagged_by_sequence_gaps_not_filled(tmp_path):
    w = new_writer(tmp_path)
    fill(w, n_frames=0, n_states=0)
    for k in (0, 1, 2, 5, 6):                                                          # seq 3,4 never arrived
        t = T0 + k * 3 * MS
        w.submit("follower_state", "telemetry", {"seq": k, "t_start_ns": t, "t_end_ns": t, "wall_ns": 0, "read_ok": True,
                                                  "raw": raw_for(k)})
    for i in range(3):
        w.submit("cam", "frame", frame_rec(i))
    ep = w.finalize()
    res = validate(ep)
    assert any("follower_state: sequence gaps" in x for x in res["warnings"])
    assert len(list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))) == 5       # nothing invented to fill the gap


# ---------------------------------------------------------------- depth freshness, age, thresholds
def _frames(numbers, host_ms):
    return [{"depth_frame_number": n, "depth_native_file": None if n is None else "x", "host_mono_ns": int(h * MS),
             "depth_device_timestamp_ms": None if n is None else 33.3 * n} for n, h in zip(numbers, host_ms)]


def test_depth_age_is_host_receipt_based_and_repeat_statistics_are_exact():
    fr = _frames([10, 11, 11, 11, 12, None, 13, 13], [0, 33, 66, 99, 132, 165, 198, 231])
    d = depth_freshness(fr)
    assert d["status"] == ["fresh", "fresh", "repeated_previous_depth", "repeated_previous_depth", "fresh", "missing", "fresh",
                           "repeated_previous_depth"]
    assert d["age_ms"] == [0, 0, 33, 66, 0, None, 0, 33]
    assert d["age_ms_stats"]["max"] == 66 and d["longest_repeat_run"] == 2
    assert d["repeat_fraction"] == pytest.approx(3 / 7)                                # missing frames excluded from the denominator
    assert d["counts"]["missing"] == 1 and d["depth_frame_number_skipped"] == 0


def test_depth_frame_number_going_backwards_is_counted_not_hidden():
    d = depth_freshness(_frames([5, 6, 4, 7], [0, 33, 66, 99]))
    assert d["depth_frame_number_non_monotonic"] == 1
    assert d["status"][2] == "fresh"                                                   # not silently relabelled; the anomaly counter carries it
    assert depth_age_ms([{"depth_frame_number": 1}], ["fresh"]) == [None]              # no host time => no invented age


def _acc(ep, **ov):
    res = run_acceptance(ep, quality_overrides=ov or None)
    return {c["name"].split("depth quality ")[1]: c for c in res["checks"] if "depth quality" in c["name"]}, res


def test_depth_thresholds_unset_are_not_checked_and_never_pass(tmp_path):
    ep = _episode(tmp_path, [1, 2, 2, 2, 3, 4])
    chk, res = _acc(ep)
    assert {c["status"] for c in chk.values()} == {"NOT_CHECKED"}
    assert "no threshold configured" in chk["max_depth_repeat_fraction"]["detail"]
    assert chk["max_depth_age_ms"]["detail"].startswith("measured 66")                 # still reported


def test_depth_thresholds_configurable_pass_and_fail_and_documented_as_not_validated(tmp_path):
    ep = _episode(tmp_path, [1, 2, 2, 2, 3, 4])                                        # 2 repeats of 6, longest run 2, max age 66 ms
    chk, res = _acc(ep, max_depth_repeat_fraction=0.5, max_depth_age_ms=100.0, max_consecutive_repeated_depth=2)
    assert {c["status"] for c in chk.values()} == {"PASS"}
    assert "not physically validated" in chk["max_depth_age_ms"]["detail"]
    chk, res = _acc(ep, max_depth_repeat_fraction=0.2, max_depth_age_ms=50.0, max_consecutive_repeated_depth=1)
    assert {c["status"] for c in chk.values()} == {"FAIL"}
    assert res["verdicts"]["collector_integrity"]["verdict"] == "FAIL"                 # a configured threshold breach is a real FAIL
    assert res["verdicts"]["physical_calibration"]["verdict"] != "PASS"                # and a PASS here never implies physical validity


def test_missing_depth_frames_fail_regardless_of_thresholds(tmp_path):
    ep = _episode(tmp_path, [1, 2, None, 4])
    _, res = _acc(ep, max_depth_repeat_fraction=1.0)
    c = next(c for c in res["checks"] if "depth frames fresh" in c["name"])
    assert c["status"] == "FAIL"


def test_quality_check_config_validation():
    base = {"quality_checks": {"max_depth_age_ms": 80, "max_depth_repeat_fraction": 0.25}}
    assert load_config(base)["quality_checks"]["max_depth_age_ms"] == 80
    assert load_config({})["quality_checks"] == {"max_depth_repeat_fraction": None, "max_depth_age_ms": None,
                                                 "max_consecutive_repeated_depth": None}
    for bad in ({"max_depth_repeat_fraction": 1.5}, {"max_depth_age_ms": -1}, {"max_depth_age_ms": "fast"},
                {"max_depth_age_ms": True}, {"nonsense": 1}):
        with pytest.raises(ConfigError):
            load_config({"quality_checks": bad})


# ---------------------------------------------------------------- native -> aligned provenance, lineage, raw values
def test_aligned_depth_is_linked_to_its_native_source_in_exports_and_master(tmp_path):
    ep = _episode(tmp_path, [7, 8, 8, 9])
    frames = list(iter_jsonl(ep / "streams" / "cam" / "frames.jsonl"))
    assert [f["depth_frame_number"] for f in frames] == [7, 8, 8, 9]                   # repeated depth keeps its ORIGINAL number
    assert frames[1]["depth_device_timestamp_ms"] == frames[2]["depth_device_timestamp_ms"]
    out = export_episode(ep, "rgbd_motor_positions")
    cam = [s["cameras"]["cam"] for s in load_export(out)["samples"]]
    assert [c["depth_status"] for c in cam] == ["fresh", "fresh", "repeated_previous_depth", "fresh"]
    assert [c["depth_frame_number"] for c in cam] == [7, 8, 8, 9]
    assert all("native depth frame" in c["depth_aligned_source"] for c in cam)
    assert all(c["depth_native_path"] != c["depth_aligned_path"] for c in cam)          # stored separately


def test_export_lineage_points_back_to_master_records_and_raw_values_survive(tmp_path):
    ep = _episode(tmp_path, [1, 2, 3, 4, 5])
    out = export_episode(ep, "rgbd_motor_positions")
    ex = load_export(out)
    assert ex["manifest"]["parent"]["episode_id"] == ep.name
    assert ex["manifest"]["parent"]["manifest_sha256"] == sha256_file(ep / "manifest.json")
    master = list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))
    names = [j["name"] for j in JOINTS]
    for s in ex["samples"]:
        seq = s["state"]["state_seq"]
        assert s["state"]["motor_positions_raw"] == [master[seq]["raw"][n] for n in names]      # bit-exact SDK integers
        assert master[seq]["raw"] == raw_for(seq)                                              # master holds what the source gave
    assert ex["arrays"]["motor_positions_raw"].dtype.kind == "i"


# ---------------------------------------------------------------- interrupted recordings, safe cleanup
class SpySource:
    is_mock = True

    def __init__(self, label, fail_close=False, log=None):
        self.label, self.fail_close, self.log = label, fail_close, log

    def open(self): self.log.append((self.label, "open"))

    def close(self):
        self.log.append((self.label, "close"))
        if self.fail_close:
            raise OSError("simulated close failure")

    def poll(self):
        threading.Event().wait(0.005)
        return []

    def describe(self): return {"backend": "spy"}


def _mock_cfg(tmp):
    from conftest import JOINTS as J

    return load_config({
        "paths": {"episodes_root": str(tmp / "eps"), "calibrations_root": str(tmp / "cal")},
        "cameras": [], "robot": {"follower": {"backend": "mock", "joints": [dict(j) for j in J]},
                                 "leader": {"backend": "none", "joints": []}, "commands": {"backend": "none"}},
        "sync": {"reference_camera": None}, "recording": {"min_free_disk_gb": 0.0}})


def test_shutdown_releases_every_source_even_when_one_close_raises(tmp_path):
    log = []
    rec = Recorder(_mock_cfg(tmp_path), cameras={}, follower=SpySource("follower", fail_close=True, log=log),
                   leader=SpySource("leader", log=log), commands=SpySource("commands", log=log))
    rec.connect()
    rec.start_workers()
    rec.shutdown()
    assert sorted(x for x in log if x[1] == "close") == [("commands", "close"), ("follower", "close"), ("leader", "close")]
    assert all(not t.is_alive() for t in rec._threads)
    assert not any(x[1] not in ("open", "close") for x in log)                          # nothing but open/close was ever called


def test_shutdown_during_recording_leaves_a_flagged_aborted_episode_not_a_complete_one(tmp_path):
    log = []
    rec = Recorder(_mock_cfg(tmp_path), cameras={}, follower=SpySource("follower", log=log), leader=SpySource("leader", log=log),
                   commands=SpySource("commands", log=log))
    rec.connect()
    rec.start_workers()
    w = rec.start_episode("t")
    rec.shutdown()                                                                      # operator closes the app mid-episode
    eps = [p for p in (tmp_path / "eps").iterdir() if p.is_dir()]
    assert len(eps) == 1 and not eps[0].name.endswith(".inprogress")
    m = read_json(eps[0] / "episode.json")
    assert m["outcome"] == "aborted" and m["status"] != "complete" and "shutdown" in m["aborted_reason"]
    assert rec._writer is None and w.id == m["episode_id"]
    from collector.exports import ExportBlocked

    with pytest.raises(ExportBlocked):
        export_episode(eps[0], "rgb_joint_angles")                                      # a partial episode is never exported as usable
    assert os.access(eps[0] / "episode.json", os.W_OK) is False                          # frozen read-only after finalise
