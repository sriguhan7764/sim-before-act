"""Bridge, pilot metadata, quality report and acceptance gate.

EVERYTHING HERE IS SYNTHETIC: fake LeRobot-like buses (no serial port, no motors), the explicit mock camera and
synthetic calibration files. It validates software wiring only; it is not evidence about the physical robot.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from collector.acceptance import run_acceptance
from collector.adapters.tap import TapSource
from collector.cli import main as cli_main
from collector.config import load_config
from collector.joints import LEROBOT_NORMALIZED_UNITS
from collector.lerobot_bridge import Bridge, goal_ticks, read_raw_and_normalized
from collector.pilot import PilotError, build_pilot_meta, check_label, eligibility, load_layout, pose_missing
from collector.quality import quality_report
from collector.recorder import Recorder
from collector.util import iter_jsonl, read_json, sha256_file
from collector.validate import validate

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL = {n: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000} for i, n in enumerate(NAMES)}


class FakeBus:
    """Mimics the parts of LeRobot's FeetechMotorsBus the bridge touches (same normalisation formulas, copied)."""

    class M:
        def __init__(self, id_):
            self.id = id_

    def __init__(self, raw):
        self.motors = {n: self.M(i + 1) for i, n in enumerate(NAMES)}
        self.raw = dict(raw)
        self.fail = False

    def sync_read(self, reg, normalize=True):
        assert reg == "Present_Position" and normalize is False
        if self.fail:
            raise ConnectionError("simulated read failure")
        return dict(self.raw)

    def _normalize(self, ids_values):
        out = {}
        for i, v in ids_values.items():
            n = NAMES[i - 1]
            c = CAL[n]
            if n == "gripper":
                out[i] = (min(c["range_max"], max(c["range_min"], v)) - c["range_min"]) / (c["range_max"] - c["range_min"]) * 100
            else:
                out[i] = (v - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095
        return out

    def _unnormalize(self, ids_values):
        out = {}
        for i, v in ids_values.items():
            n = NAMES[i - 1]
            c = CAL[n]
            if n == "gripper":
                out[i] = int(round(v / 100 * (c["range_max"] - c["range_min"]) + c["range_min"]))
            else:
                out[i] = int(round(v * 4095 / 360 + (c["range_min"] + c["range_max"]) / 2))
        return out


class FakeFollower:
    def __init__(self, raw, max_rel=10.0):
        self.bus, self.max_rel, self.sent = FakeBus(raw), max_rel, []

    def send_action(self, action):
        _, present = read_raw_and_normalized(self.bus)
        sent = {}
        for k, v in action.items():
            n = k.removesuffix(".pos")
            sent[k] = min(present[n] + self.max_rel, max(present[n] - self.max_rel, v))
        self.sent.append(dict(sent))
        self.bus.raw = goal_ticks(self.bus, {k.removesuffix(".pos"): v for k, v in sent.items()})
        return sent


class FakeLeader:
    def __init__(self, raw):
        self.bus = FakeBus(raw)


def base_raw(k=0.0, amp=0.0):
    return {n: int(2000 + amp * math.sin(k + i)) for i, n in enumerate(NAMES)}


RAMP_TICKS_PER_CYCLE = 5   # ~0.44 normalised units/cycle: far below the 10-unit clip, so the fake follower tracks exactly
PRE_STEPS, EPISODE_STEPS = 5, 120


def ramp_raw(k):
    """Deterministic scripted leader: raw ticks are a pure function of the bridge cycle index k."""
    return {n: 2000 + RAMP_TICKS_PER_CYCLE * k + i for i, n in enumerate(NAMES)}


def ramp_index(raw):
    return (raw["shoulder_pan"] - 2000) // RAMP_TICKS_PER_CYCLE


def wait_until(cond, what, timeout=10.0):
    """Wait for a condition (never a fixed sleep); the deadline only bounds a hang."""
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.001)


# ---------------------------------------------------------------- bridge unit tests
def test_bridge_publishes_three_distinct_signals_and_sends_clipped_target(tmp_path):
    fol, lea = FakeFollower(base_raw()), FakeLeader({**base_raw(), "shoulder_pan": 2600})  # leader far from follower
    b = Bridge(fol, lea, tmp_path, fps=100, session={"follower": {"calibration": {"sha256": "x"}}})
    assert b.step() is True
    f, l, c = (list(iter_jsonl(tmp_path / f"{n}.jsonl"))[0] for n in ("follower_state", "leader_state", "commands"))
    assert f["raw"]["shoulder_pan"] == 2000 and l["raw"]["shoulder_pan"] == 2600
    assert l["normalized"]["shoulder_pan"] > f["normalized"]["shoulder_pan"] + 50     # leader measurement != follower state
    assert c["values"]["shoulder_pan"] == pytest.approx(f["normalized"]["shoulder_pan"] + 10)  # relative-target clip
    assert c["leader_measurement_normalized"]["shoulder_pan"] == pytest.approx(l["normalized"]["shoulder_pan"])
    assert c["clipped"]["shoulder_pan"] is True and c["clipped"]["shoulder_lift"] is False
    assert c["raw_goal"]["shoulder_pan"] == goal_ticks(fol.bus, c["values"])["shoulder_pan"]
    assert c["raw_goal"]["shoulder_pan"] == fol.bus.raw["shoulder_pan"]               # what the fake bus was told to go to
    assert c["units"] == LEROBOT_NORMALIZED_UNITS and c["ack"] is None
    for rec in (f, l, c):
        assert rec["t_start_ns"] <= rec["t_end_ns"]


def test_bridge_sends_nothing_on_read_failure_and_stops_after_repeated_failures(tmp_path):
    fol, lea = FakeFollower(base_raw()), FakeLeader(base_raw())
    b = Bridge(fol, lea, tmp_path, max_consecutive_errors=3)
    lea.bus.fail = True
    assert b.step() is False and b.step() is False
    assert fol.sent == []                                   # no command from stale/partial data
    with pytest.raises(RuntimeError, match="consecutive read failures"):
        b.step()
    first = list(iter_jsonl(tmp_path / "follower_state.jsonl"))[0]
    assert first["read_ok"] is False and "error" in first


def test_bridge_run_writes_session_with_semantics_and_stats(tmp_path):
    b = Bridge(FakeFollower(base_raw()), FakeLeader(base_raw()), tmp_path, fps=200, session={"follower": {"max_relative_target": 10}})
    stats = b.run(seconds=0.2)
    s = read_json(tmp_path / "session.json")
    assert stats["cycles"] >= 10 and s["stats"]["cycles"] == stats["cycles"]
    assert "NOT a torque" in s["semantics"]["signal_kind"] and s["semantics"]["commands.ack"].startswith("none")
    assert "lerobot_record stores this as dataset 'action'" in s["semantics"]["commands.leader_measurement_normalized"]


def test_tap_source_describe_includes_session_sidecar(tmp_path):
    (tmp_path / "commands.jsonl").write_text("")
    (tmp_path / "session.json").write_text(json.dumps({"fps": 30}))
    d = TapSource(str(tmp_path / "commands.jsonl"), "command").describe()
    assert d["session"] == {"fps": 30} and len(d["session_sha256"]) == 64
    assert TapSource(str(tmp_path / "other" / "x.jsonl"), "command").describe()["session"] is None


# ---------------------------------------------------------------- pilot metadata
def test_layout_ships_with_null_poses_and_never_defaults_them():
    layout = load_layout(ROOT / "configs/pilot_layout.yaml")
    assert all(v["x_m"] is None for v in layout["cube_start_positions"].values())
    p = build_pilot_meta("demonstration", layout, "P1")
    assert pose_missing(p["cube_start_pose_mat_frame"]) and pose_missing(p["tray_pose_mat_frame"])
    assert p["cube_start_pose_mat_frame"]["x_m"] is None
    assert eligibility({"pilot": p, "outcome": "success", "status": "complete", "quality_flags": {}})["eligible"] is False


def test_pilot_validation_rules():
    with pytest.raises(PilotError):
        build_pilot_meta("bogus")
    with pytest.raises(PilotError, match="cube-position-id"):
        build_pilot_meta("demonstration")
    layout = {"cube_start_positions": {"P1": {}}}
    with pytest.raises(PilotError, match="not in the layout"):
        build_pilot_meta("demonstration", layout, "P9")
    measured = {"x_m": 0.1, "y_m": 0.2, "yaw_deg": 0.0, "source": "manual_measurement", "sigma_mm": 2.0}
    assert pose_missing(measured) == []
    assert pose_missing({**measured, "source": "guess"}) == ["source"]
    check_label("success", "operator_visual")
    with pytest.raises(PilotError):
        check_label("success", "none")
    with pytest.raises(PilotError):
        check_label("unlabelled", "operator_visual")


def test_eligibility_accepts_only_complete_measured_successful_demonstration():
    measured = {"x_m": 0.1, "y_m": 0.2, "yaw_deg": 0.0, "source": "manual_measurement", "sigma_mm": 2.0}
    p = {"role": "demonstration", "cube_position_id": "P1", "cube_start_pose_mat_frame": measured, "tray_pose_mat_frame": measured}
    ok = {"pilot": p, "outcome": "success", "status": "complete", "quality_flags": {"missing_modalities": []}}
    assert eligibility(ok)["eligible"]
    assert not eligibility({**ok, "quality_flags": {"mock_data": True}})["eligible"]
    assert not eligibility({**ok, "outcome": "failure"})["eligible"]
    assert not eligibility({**ok, "pilot": {**p, "role": "setup"}})["eligible"]
    assert eligibility(ok, {"outcome": "failure"})["eligible"] is False       # a later annotation can demote


# ---------------------------------------------------------------- end-to-end (synthetic) through taps
@pytest.fixture
def synthetic_pilot_episode(tmp_path):
    cal_f, cal_l = tmp_path / "follower_cal.json", tmp_path / "leader_cal.json"
    cal_f.write_text(json.dumps(CAL))
    cal_l.write_text(json.dumps(CAL))
    taps = tmp_path / "taps"
    fol, lea = FakeFollower(base_raw()), FakeLeader(base_raw())
    session = {"follower": {"calibration": {"path": str(cal_f), "sha256": sha256_file(cal_f)}, "max_relative_target": 10},
               "leader": {"calibration": {"path": str(cal_l), "sha256": sha256_file(cal_l)}}, "synthetic": True}
    bridge = Bridge(fol, lea, taps, fps=100, session=session)
    bridge.write_session()
    raw = yaml.safe_load((ROOT / "configs/mock.yaml").read_text())
    raw["paths"] = {"episodes_root": str(tmp_path / "eps"), "calibrations_root": str(tmp_path / "cal")}
    tap = lambda n: {"backend": "jsonl_tap", "path": str(taps / f"{n}.jsonl")}
    raw["robot"]["follower"] = {**raw["robot"]["follower"], "calibration_file": str(cal_f)}
    raw["robot"]["leader"] = {"backend": "mock", "calibration_file": str(cal_l), "joints": raw["robot"]["follower"]["joints"]}
    raw["robot"]["controller"] = {"loop_hz": 100}
    cfg = load_config(raw)
    for k, n in (("follower", "follower_state"), ("leader", "leader_state")):   # same injection collector.cli does for --taps-dir
        cfg["robot"][k] = {**cfg["robot"][k], "backend": "jsonl_tap", "path": str(taps / f"{n}.jsonl")}
    cfg["robot"]["commands"] = tap("commands")
    rec = Recorder(cfg)
    rec.connect()           # tap sources seek to end-of-file here, so every later bridge cycle is ingested
    rec.start_workers()

    def cycle(k):           # scripted leader: step k of a slow ramp (well inside the follower clip), no wall-clock input
        lea.bus.raw = ramp_raw(k)
        assert bridge.step() is True

    def ingested(n):
        return all(rec.status[s]["count"] >= n for s in ("follower", "leader", "commands"))

    for k in range(PRE_STEPS):
        cycle(k)
    wait_until(lambda: ingested(PRE_STEPS), "pre-episode ingestion")
    layout = load_layout(ROOT / "configs/pilot_layout.yaml")
    rec.start_episode("pick_red_cube_place_in_blue_tray", "synthetic test", operator="test",
                      pilot=build_pilot_meta("setup", layout))
    for k in range(PRE_STEPS, PRE_STEPS + EPISODE_STEPS):    # paced by ingestion, not by sleeps
        cycle(k)
        wait_until(lambda k=k: ingested(k + 1), f"ingestion of cycle {k}")
    ep = rec.finish_episode("unlabelled")
    rec.shutdown()
    return ep, cal_f


def test_synthetic_episode_records_separate_signals_and_pilot_metadata(synthetic_pilot_episode):
    ep, cal_f = synthetic_pilot_episode
    meta = read_json(ep / "episode.json")
    assert meta["quality_flags"]["mock_data"] is True and "MOCK" in ep.name        # mock camera => clearly flagged
    assert meta["pilot"]["role"] == "setup" and meta["pilot"]["outcome_label_source"] == "none"
    start = meta["pilot"]["robot_start_configuration"]
    assert start and set(start["raw"]) == set(NAMES) and start["joint_angles_rad"] and "not commanded, not leader" in start["source"]
    assert meta["quality_flags"]["follower_calibration_hash_matches_controller"] is True
    assert meta["quality_flags"]["controller_session_missing"] is False
    assert meta["controller"]["published_session"]["synthetic"] is True
    assert meta["quality_flags"]["cube_pose_measured"] is False and meta["quality_flags"]["missing_modalities"] == []
    assert (ep / "calibration" / "leader_motor_calibration.json").exists()
    f = list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))
    l = list(iter_jsonl(ep / "telemetry" / "leader_state.jsonl"))
    c = list(iter_jsonl(ep / "telemetry" / "commands.jsonl"))
    assert len(f) == len(l) == len(c) == EPISODE_STEPS                 # every scripted cycle recorded exactly once
    lead = [ramp_index(r["raw"]) for r in l]
    assert lead == list(range(PRE_STEPS, PRE_STEPS + EPISODE_STEPS))   # leader stream == the script
    # the follower stream is the MEASURED state: it lags the leader by exactly one cycle (the previous submitted goal),
    # it is neither the leader reading of the same cycle nor the command of the same cycle
    assert [ramp_index(r["raw"]) for r in f] == [x - 1 for x in lead]
    assert all(fr["raw"] != lr["raw"] for fr, lr in zip(f, l))
    assert all(ramp_index(cr["raw_goal"]) == x for cr, x in zip(c, lead))    # submitted goal tracks the leader
    assert all(cr["submission_status"] == "submitted_not_confirmed" for cr in c)
    assert all("leader_measurement_normalized" in r for r in c)
    assert validate(ep)["ok"]


def test_acceptance_gate_passes_on_consistent_synthetic_episode_and_labels_it(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    res = run_acceptance(ep)
    failed = [c for c in res["checks"] if c["status"] == "FAIL"]
    assert res["overall"] == "PASS", failed
    assert res["synthetic_or_mock"] and "SYNTHETIC" in res["evidence_label"]
    names = " ".join(c["name"] for c in res["checks"])
    assert "rgb_joint_angles" in names and "rgbd_motor_positions" in names and "source-frame association" in names
    assert (ep.parent / (ep.name + ".reports") / "acceptance.md").exists()
    # both exports keep the command signal and the identity of the SAME episode
    from collector.exports import export_dir_for, load_export

    a, b = (load_export(export_dir_for(ep, p)) for p in ("rgb_joint_angles", "rgbd_motor_positions"))
    assert a["manifest"]["parent"]["episode_id"] == b["manifest"]["parent"]["episode_id"]
    assert a["arrays"]["command_prev_valid"].any() and b["arrays"]["command_prev_raw_goal"].max() > 0
    assert np.isfinite(a["arrays"]["command_prev_target_rad"][a["arrays"]["command_prev_valid"]]).all()
    assert "command_prev_target_rad" not in b["arrays"]        # raw-unit profile keeps SDK units, no rad conversion
    assert b["arrays"]["motor_positions_raw"].dtype == np.int64
    cam = next(s["cameras"]["mock_rgbd"] for s in b["samples"] if s["cameras"].get("mock_rgbd", {}).get("depth_status"))
    assert cam["depth_status"] == "fresh" and cam["depth_frame_number"] is not None and "native depth frame" in cam["depth_aligned_source"]


def test_acceptance_detects_tampered_export(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    assert run_acceptance(ep)["overall"] == "PASS"
    from collector.exports import export_dir_for

    p = export_dir_for(ep, "rgbd_motor_positions") / "arrays" / "state_valid.npy"
    v = np.load(p)
    v[0] = not v[0]
    np.save(p, v)
    res = run_acceptance(ep)
    assert res["overall"] == "FAIL"
    assert any(c["status"] == "FAIL" and "rgbd_motor_positions" in c["name"] for c in res["checks"])


def test_quality_report_contents_and_unit_crosscheck(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    rep = quality_report(ep)
    assert rep["data_origin"].startswith("SYNTHETIC")
    assert rep["telemetry"]["commands"]["ack_available"] is False
    assert rep["telemetry"]["commands"]["clipped_fraction"] is not None
    d = rep["units"]["collector_mapping_vs_lerobot_normalized"]
    assert d["max_abs_diff_arm_deg"] < 1e-6 and d["max_abs_diff_gripper_pct"] < 1e-6
    assert rep["association"]["hardware_synchronisation"] is False
    assert any("No hardware synchronisation" in a for a in rep["clock_domain_assumptions"])
    assert rep["command_tracking"] is not None
    assert rep["eligibility_as_accepted_demonstration"]["eligible"] is False


def test_finish_requires_label_source_and_cli_refuses_silent_labels(tmp_path, synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    assert cli_main(["annotate", "--episode", str(ep), "--outcome", "success"]) == 2
    assert cli_main(["annotate", "--episode", str(ep), "--outcome", "unknown", "--label-source", "operator_visual"]) == 2
    assert cli_main(["annotate", "--episode", str(ep), "--outcome", "success", "--label-source", "operator_visual",
                     "--accepted-for-dataset", "yes"]) == 0
    ann = read_json(ep.parent / (ep.name + ".annotations") / "v001.json")
    assert ann["label_source"] == "operator_visual" and ann["accepted_for_dataset"] is True
    assert read_json(ep / "episode.json")["outcome"] == "unlabelled"              # master immutable
    assert cli_main(["record", "--config", "x", "--task", "t", "--outcome", "success"]) == 2


def test_record_cli_rejects_bad_pilot_metadata_before_opening_devices(capsys):
    rc = cli_main(["record", "--config", str(ROOT / "configs/mock.yaml"), "--task", "t", "--role", "demonstration",
                   "--layout", str(ROOT / "configs/pilot_layout.yaml")])
    assert rc == 2 and "cube-position-id" in capsys.readouterr().out


def test_bridge_module_has_no_unauthorised_motion_paths():
    import ast
    src = (ROOT / "collector/lerobot_bridge.py").read_text()
    tree = ast.parse(src)
    # inspect code only (names, attributes, string literals outside docstrings), not prose
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    forbidden = {"sync_write", "enable_torque", "disable_torque", "set_half_turn_homings", "record_ranges_of_motion",
                 "write_calibration", "calibrate", "configure"}
    assert not (attrs & forbidden), attrs & forbidden
    writers = {n.value.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "write"
               and isinstance(n.value, ast.Attribute)}
    assert writers <= {"t_f", "t_l", "t_c"}, writers   # .write() only on the tap files, never on a bus
    assert "send_action" in attrs                      # the only motion path: LeRobot's own send_action, after authorisation
    assert "--i-authorise-motion" in src and "connect(calibrate=False)" in src


# ---------------------------------------------------------------- pilot preparation (all synthetic / no hardware)
def test_negotiation_mismatch_is_reported_never_substituted():
    from collector.negotiation import negotiation_record

    ok = negotiation_record({"color": {"width": 640, "height": 480, "fps": 30, "format": "rgb8"}},
                            {"color": {"width": 640, "height": 480, "fps": 30.0, "format": "rgb8"}})
    assert ok["matches_request"] and ok["silent_substitution"] is False
    bad = negotiation_record({"depth": {"width": 640, "height": 480, "fps": 30, "format": "z16"}},
                             {"depth": {"width": 848, "height": 480, "fps": 15, "format": "z16"}})
    assert not bad["matches_request"] and len(bad["mismatches"]["depth"]) == 2


def test_opencv_adapter_refuses_a_negotiated_profile_that_differs(monkeypatch, hardware_authorised_for_fakes):
    import cv2

    from collector.adapters.opencv_cam import OpenCVCamera
    from collector.negotiation import ProfileMismatch

    class FakeCap:                                    # device silently falls back to 320x240@15 (synthetic stand-in)
        def isOpened(self): return True
        def set(self, *_): return True
        def release(self): self.released = True
        def get(self, p):
            return {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240, cv2.CAP_PROP_FPS: 15,
                    cv2.CAP_PROP_FOURCC: 0x47504A4D}.get(p, 0)

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCap())
    cam = OpenCVCamera({"name": "wrist", "device": "/dev/null", "width": 640, "height": 480, "fps": 30})
    with pytest.raises(ProfileMismatch, match="width"):
        cam.open()
    assert cam.cap is None


def test_freeze_config_hash_is_order_independent_and_value_sensitive():
    from collector.preflight import freeze_config

    a = freeze_config({"x": 1, "y": {"b": 2, "a": 3}})[1]
    assert a == freeze_config({"y": {"a": 3, "b": 2}, "x": 1})[1]
    assert a != freeze_config({"x": 1, "y": {"b": 2, "a": 4}})[1]


def test_registry_accepts_active_and_refuses_stale_calibration_files():
    from collector.preflight import check_calibration_files, load_registry

    reg = ROOT / "configs/calibration_registry.yaml"
    doc = load_registry(reg)
    base = ROOT
    active = {r: str((base / doc["active"][r]["path"]).resolve()) for r in ("follower", "leader")}
    cfg = {"robot": {"follower": {"calibration_file": active["follower"]}, "leader": {"calibration_file": active["leader"]}}}
    assert [c["status"] for c in check_calibration_files(cfg, reg)] == ["PASS", "PASS"]
    stale = {s["path"]: s for s in doc["stale"]}
    for path in stale:
        cfg2 = {"robot": {"follower": {"calibration_file": str((base / path).resolve())},
                          "leader": {"calibration_file": active["leader"]}}}
        res = check_calibration_files(cfg2, reg)
        assert res[0]["status"] == "FAIL" and "STALE" in res[0]["detail"]


def test_startup_check_refuses_uncalibrated_motors_and_large_start_gap():
    from collector.lerobot_bridge import StartupRefused, startup_check

    class Arm(FakeFollower):
        def __init__(self, raw, cal=True):
            super().__init__(raw)
            self.is_calibrated = cal

    with pytest.raises(StartupRefused, match="calibration files"):
        startup_check(Arm(base_raw(), cal=False), Arm(base_raw()), 15.0)
    with pytest.raises(StartupRefused, match="differ"):
        startup_check(Arm(base_raw()), Arm({**base_raw(), "shoulder_lift": 2800}), 15.0)
    ok = startup_check(Arm(base_raw()), Arm({**base_raw(), "shoulder_lift": 2050}), 15.0)
    assert ok["follower_matches_calibration"] and ok["start_gap_normalized_units"]["shoulder_lift"] < 15


def test_bridge_commands_are_marked_submitted_not_confirmed(tmp_path):
    b = Bridge(FakeFollower(base_raw()), FakeLeader(base_raw()), tmp_path, fps=100)
    b.step()
    c = list(iter_jsonl(tmp_path / "commands.jsonl"))[0]
    assert c["submission_status"] == "submitted_not_confirmed" and c["ack"] is None


def test_acceptance_reports_three_separate_verdicts_and_never_validates_physical_calibration(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    res = run_acceptance(ep)
    v = res["verdicts"]
    assert set(v) == {"collector_integrity", "physical_calibration", "manipulation_readiness"}
    assert v["collector_integrity"]["verdict"] == "PASS" and res["overall"] == "PASS"
    assert v["physical_calibration"]["verdict"] == "NOT_VALIDATED"       # signs/zero/gripper/base alignment cannot be checked from data
    assert v["manipulation_readiness"]["verdict"] == "NOT_READY"         # setup episode: no measured poses or label
    phys = {c["name"].strip(): c["status"] for c in res["checks"] if c["category"] == "physical_calibration"}
    assert phys["joint rotation signs"] == "NOT_CHECKED" and phys["robot-base-to-mat alignment"] == "NOT_CHECKED"
    assert not any(c["status"] == "SKIP" for c in res["checks"])
    names = " ".join(c["name"] for c in res["checks"])
    assert "frozen config hash" in names and "negotiated profile == requested" in names and "depth_native stored size" in names
    assert "depth_aligned stored size" in names


def test_acceptance_cli_prints_three_verdicts(synthetic_pilot_episode, capsys):
    ep, _ = synthetic_pilot_episode
    assert cli_main(["acceptance", "--episode", str(ep)]) == 0
    out = capsys.readouterr().out
    for k in ("collector_integrity", "physical_calibration", "manipulation_readiness"):
        assert k in out
    assert "NOT_VALIDATED" in out and "NOT_READY" in out
    md = (ep.parent / (ep.name + ".reports") / "acceptance.md").read_text()
    assert "NOT_CHECKED" in md and "physical_calibration" in md


def test_acceptance_fails_when_frozen_config_hash_does_not_match(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    assert run_acceptance(ep)["overall"] == "PASS"
    fz = ep / "config" / "effective_config.json"
    doc = read_json(fz)
    doc["config"]["mode"] = "tampered"
    os.chmod(fz, 0o644)                                                      # masters are written read-only; the test tampers on a tmp copy
    fz.write_text(json.dumps(doc))
    res = run_acceptance(ep)
    assert res["overall"] == "FAIL"
    assert any(c["status"] == "FAIL" and "frozen config hash" in c["name"] for c in res["checks"])


def test_acceptance_fails_on_negotiation_mismatch_or_wrong_depth_size_in_record(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    cj = ep / "streams" / "mock_rgbd" / "camera.json"
    doc = read_json(cj)
    doc["negotiation"]["matches_request"] = False
    doc["depth_aligned"]["width"] = 1280                                     # description disagrees with the stored file
    os.chmod(cj, 0o644)
    cj.write_text(json.dumps(doc))
    res = run_acceptance(ep)
    bad = [c["name"].strip() for c in res["checks"] if c["status"] == "FAIL"]
    assert any("negotiated profile" in n for n in bad) and any("depth_aligned stored size" in n for n in bad)


def test_quality_report_has_negotiation_depth_shapes_and_submitted_wording(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    rep = quality_report(ep)
    cam = rep["cameras"]["mock_rgbd"]
    assert cam["negotiation"]["matches_request"] and cam["depth_native_file_size_dtype"][2] == "uint16"
    assert cam["depth_aligned_file_size_dtype"][2] == "uint16"
    assert "SUBMITTED" in rep["command_semantics"] and "not read back" in rep["command_semantics"]
    assert rep["effective_config_sha256"] and rep["stale_calibration_in_use"] == []


def test_protocol_segments_are_summarised_without_thresholds(synthetic_pilot_episode):
    from collector.quality import _protocol_section

    ep, _ = synthetic_pilot_episode
    f = list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))
    t0, t1 = f[0]["t_start_ns"], f[-1]["t_start_ns"]
    mid = (t0 + t1) // 2
    meta = {"protocol": {"name": "t", "nominal": {"segments": [{"label": "a"}, {"label": "b"}]},
                         "cues_issued": [{"label": "a", "cue_host_mono_ns": t0}, {"label": "b", "cue_host_mono_ns": mid}]}}
    out = _protocol_section(ep, meta)
    assert [s["label"] for s in out["segments"]] == ["a", "b"] and out["segments"][0]["follower_state_samples"] > 0
    assert "no pass/fail threshold" in out["note"]
    from collector.acceptance import _protocol_check

    assert _protocol_check(meta)["status"] == "PASS" and _protocol_check({})["status"] == "NOT_CHECKED"


def test_record_pilot_is_blocked_when_a_camera_negotiation_mismatches(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/mock.yaml").read_text())
    raw["paths"] = {"episodes_root": str(tmp_path / "eps"), "calibrations_root": str(tmp_path / "cal")}
    rec = Recorder(load_config(raw))
    rec.connect()
    rec.descr["mock_rgbd"]["negotiation"] = {"matches_request": False, "mismatches": {"color": ["width: requested 640, actual 320"]}}
    layout = load_layout(ROOT / "configs/pilot_layout.yaml")
    with pytest.raises(RuntimeError, match="pilot recording blocked.*negotiated profile differs"):
        rec.start_episode("t", pilot=build_pilot_meta("setup", layout))
    assert not list((tmp_path / "eps").glob("*")) or all(not p.is_dir() for p in (tmp_path / "eps").glob("ep_*"))
    rec.shutdown()


def test_depth_freshness_flags_repeated_and_missing_depth_and_keeps_original_numbers():
    from collector.sync import depth_freshness

    fr = [{"depth_frame_number": n, "depth_device_timestamp_ms": 33.3 * n, "device_timestamp_ms": 33.3 * k + 2, "depth_native_file": "x"}
          for k, n in enumerate([10, 11, 11, 13])]
    fr.append({"depth_frame_number": None, "depth_native_file": None})
    d = depth_freshness(fr)
    assert d["status"] == ["fresh", "fresh", "repeated_previous_depth", "fresh", "missing"]
    assert d["depth_frame_number_skipped"] == 1 and d["depth_timestamp_non_monotonic"] == 0
    assert d["color_depth_device_timestamp_offset_ms_max"] is not None


def test_repeated_depth_is_reported_by_quality_acceptance_and_export_rows(synthetic_pilot_episode):
    ep, _ = synthetic_pilot_episode
    fp = ep / "streams" / "mock_rgbd" / "frames.jsonl"
    rows = [json.loads(line) for line in fp.read_text().splitlines()]
    rows[3]["depth_frame_number"] = rows[2]["depth_frame_number"]            # the frameset re-delivered the previous depth frame
    rows[3]["depth_device_timestamp_ms"] = rows[2]["depth_device_timestamp_ms"]
    os.chmod(fp, 0o644)
    fp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    k = quality_report(ep)["cameras"]["mock_rgbd"]["depth_freshness"]["counts"]
    assert k["repeated_previous_depth"] == 1 and k["missing"] == 0
    res = run_acceptance(ep)
    chk = [c for c in res["checks"] if "depth frames fresh" in c["name"]]
    assert chk and chk[0]["status"] == "WARN"
