"""Recorder: independent acquisition workers, readiness gates, episode lifecycle.

The recorder never sends anything to the robot. Robot sources are read-only adapters; on shutdown they only
release their port/file (no torque or mode changes).
"""
from __future__ import annotations

import platform
import threading
import time
from pathlib import Path

from . import __version__, hardware_gate
from .adapters.base import CameraSource, StateSource
from .adapters.factory import make_camera, make_command_source, make_state_source
from .config import markers_configured
from .joints import AngleMapping
from .pilot import pilot_quality_flags, check_label
from .preflight import check_calibration_files, freeze_config, stale_hashes
from .storage import EpisodeWriter, disk_free_gb
from .util import mono_ns, read_json, repo_commit, sha256_file, write_json_atomic
from . import calibration as calmod


def check_readiness(cfg: dict, cal_dir: Path | None, cameras_ok: dict[str, bool], follower_ok: bool,
                    mock: bool) -> tuple[list[str], list[str]]:
    """Return (blocking_problems, warnings). Blocking only in research mode."""
    issues = []
    g = cfg["research_gates"]
    cal = read_json(cal_dir / "calibration.json") if cal_dir else None
    if g["require_calibration_accepted"] and not (cal and cal.get("operator_accepted")):
        issues.append("no operator-accepted camera-to-mat calibration")
    if g["require_marker_size_confirmed"] and cfg["markers"]["size_excludes_white_margin_confirmed"] is not True:
        issues.append("marker size (outer black square, excluding white margin) not confirmed in config")
    if g["require_all_cameras_connected"] and not all(cameras_ok.values()):
        issues.append(f"cameras not connected: {[k for k, v in cameras_ok.items() if not v]}")
    if g["require_follower_connected"] and not follower_ok:
        issues.append("follower telemetry not connected")
    if g["require_robot_calibration_file"] and not cfg["robot"]["follower"].get("calibration_file"):
        issues.append("robot.follower.calibration_file unset (joint-angle export would be blocked)")
    if g["require_depth"] and not any(c.get("depth") for c in cfg["cameras"] if c["enabled"]):
        issues.append("no depth-enabled camera configured")
    if g["forbid_mock"] and mock:
        issues.append("mock backend in use")
    if markers_configured(cfg) and g["require_calibration_accepted"]:
        issues.append("marker dictionary/ids/orientations unset: " + ", ".join(markers_configured(cfg)))
    if cfg["mode"] == "research":
        return issues, []
    return [], issues


class Recorder:
    def __init__(self, cfg: dict, cameras: dict[str, CameraSource] | None = None, follower: StateSource | None = None,
                 leader: StateSource | None = None, commands: StateSource | None = None, config_path: str | None = None):
        self.cfg, self.config_path = cfg, config_path
        self.cameras = cameras if cameras is not None else {c["name"]: make_camera(c) for c in cfg["cameras"] if c["enabled"]}
        r = cfg["robot"]
        self.follower = follower if follower is not None else make_state_source(r["follower"], "follower")
        self.leader = leader if leader is not None else make_state_source(r["leader"], "leader")
        self.commands = commands if commands is not None else make_command_source(r["commands"], r["follower"]["joints"])
        self.status: dict[str, dict] = {}
        self.latest: dict[str, object] = {}
        self.latest_state: dict | None = None
        self.mapping: AngleMapping | None = None
        self.leader_mapping: AngleMapping | None = None
        self._writer: EpisodeWriter | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.cal_dir: Path | None = None
        self.descr: dict[str, dict] = {}
        self.warnings: list[str] = []
        self.sources_ok: dict[str, bool] = {}

    @property
    def is_mock(self) -> bool:
        srcs = [*self.cameras.values(), self.follower, self.leader, self.commands]
        return any(getattr(s, "is_mock", False) for s in srcs if s is not None)

    # ---------- lifecycle ----------
    def connect(self) -> None:
        """Open every configured source; failures are recorded as status, not hidden or replaced by mock."""
        for name, cam in self.cameras.items():
            self._open(name, cam, "camera")
        for label, src in (("follower", self.follower), ("leader", self.leader), ("commands", self.commands)):
            if src is not None:
                self._open(label, src, "robot")
        fcfg = self.cfg["robot"]["follower"]
        if fcfg.get("calibration_file") and fcfg["joints"]:
            try:
                self.mapping = AngleMapping.from_file(fcfg["joints"], fcfg["calibration_file"])
            except Exception as e:
                self.warnings.append(f"follower calibration unusable: {e}")
        lcfg = self.cfg["robot"]["leader"]
        if lcfg.get("calibration_file") and lcfg["joints"]:
            try:
                self.leader_mapping = AngleMapping.from_file(lcfg["joints"], lcfg["calibration_file"])
            except Exception as e:
                self.warnings.append(f"leader calibration unusable: {e}")
        root = Path(self.cfg["paths"]["calibrations_root"])
        self.cal_dir = calmod.latest_snapshot(root, require_accepted=self.cfg["mode"] == "research")

    def _open(self, name, src, kind):
        st = self.status[name] = {"kind": kind, "connected": False, "error": None, "count": 0, "rate_hz": 0.0,
                                  "last_mono_ns": None, "_t": []}
        try:
            src.open()
            st["connected"] = True
            self.descr[name] = src.describe()
        except Exception as e:
            st["error"] = f"{type(e).__name__}: {e}"
        self.sources_ok[name] = st["connected"]

    def start_workers(self) -> None:
        for name, cam in self.cameras.items():
            if self.status[name]["connected"]:
                self._spawn(self._camera_loop, name, cam)
        for label, src, stream in (("follower", self.follower, "follower_state"), ("leader", self.leader, "leader_state"),
                                   ("commands", self.commands, "commands")):
            if src is not None and self.status[label]["connected"]:
                self._spawn(self._state_loop, label, src, stream)
        self._spawn(self._monitor)

    def _spawn(self, fn, *a):
        t = threading.Thread(target=fn, args=a, daemon=True, name=f"acq-{fn.__name__}")
        t.start()
        self._threads.append(t)

    def _tick(self, name: str) -> None:
        st = self.status[name]
        now = time.monotonic()
        st["count"] += 1
        st["_t"].append(now)
        st["_t"] = [t for t in st["_t"] if now - t < 2.0]
        st["rate_hz"] = len(st["_t"]) / 2.0 if now - st["_t"][0] > 1.0 else float(len(st["_t"]))
        st["last_mono_ns"] = mono_ns()

    def _camera_loop(self, name: str, cam: CameraSource) -> None:
        recv, cur = 0, None
        while not self._stop.is_set():
            try:
                fr = cam.read(timeout_s=0.5)
            except Exception as e:
                self.status[name]["error"] = f"{type(e).__name__}: {e}"
                time.sleep(0.2)
                continue
            if fr is None:
                continue
            self._tick(name)
            self.latest[name] = fr
            w = self._writer
            if w is not cur:
                cur, recv = w, 0  # indices restart per episode so gaps mean drops
            if w is not None:
                rec = {"recv_index": recv, "frame_number": fr.frame_number, "device_timestamp_ms": fr.device_timestamp_ms,
                       "timestamp_domain": fr.timestamp_domain, "host_mono_ns": fr.host_mono_ns,
                       "host_wall_ns": fr.host_wall_ns, "depth_frame_number": fr.depth_frame_number,
                       "depth_device_timestamp_ms": fr.depth_device_timestamp_ms,
                       "depth_timestamp_domain": fr.depth_timestamp_domain, "extras": fr.extras}
                w.submit(name, "frame", (fr, rec))
            recv += 1  # incremented even when the submit is dropped so gaps are visible

    def _state_loop(self, label: str, src: StateSource, stream: str) -> None:
        seq, cur = 0, None
        while not self._stop.is_set():
            try:
                recs = src.poll()
            except Exception as e:
                self.status[label]["error"] = f"{type(e).__name__}: {e}"
                time.sleep(0.1)
                continue
            for r in recs:
                self._tick(label)
                w = self._writer
                if w is not cur:
                    cur, seq = w, 0
                r["seq"] = seq
                seq += 1
                if stream == "follower_state":
                    self.latest_state = r
                if w is not None:
                    w.submit(stream, "telemetry", r)

    def _monitor(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.5)
            w = self._writer
            if w is None:
                continue
            free = disk_free_gb(w.root)
            if free < self.cfg["recording"]["min_free_disk_gb"]:
                w.abort_reason = f"low disk: {free:.2f} GB free"
                w.log_event("low_disk", free_gb=free)
            if w.abort_reason:
                w.log_event("auto_abort_requested", reason=w.abort_reason)

    def should_abort(self) -> str | None:
        w = self._writer
        return w.abort_reason if w else None

    def shutdown(self) -> None:
        """Stop workers and release ports. No robot control state is changed."""
        if self._writer is not None:
            self.abort_episode("recorder shutdown while recording")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        for src in [*self.cameras.values(), self.follower, self.leader, self.commands]:
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass

    # ---------- episodes ----------
    def pilot_blockers(self) -> list[str]:
        """Conditions under which a structured pilot episode must not start (no silent degradation)."""
        b = [f"{n} not connected: {s['error']}" for n, s in self.status.items() if not s["connected"]]
        reg = self.cfg.get("calibration_registry")
        if reg:
            b += [f"{c['name']}: {c['detail']}" for c in check_calibration_files(self.cfg, reg) if c["status"] != "PASS"]
        elif not self.is_mock:
            b.append("calibration_registry not configured (cannot prove stale calibration files are excluded)")
        if self.mapping is None:
            b.append("follower calibration unusable")
        if self.leader_mapping is None:
            b.append("leader calibration unusable")
        for n, d in self.descr.items():
            neg = d.get("negotiation")
            if n in self.cameras and neg is not None and not neg.get("matches_request"):
                b.append(f"camera {n}: negotiated profile differs from request: {neg.get('mismatches')}")
        return b

    def _freeze_config(self, w: EpisodeWriter) -> None:
        """Write the exact effective configuration used by this episode and its hash (also stored in episode.json)."""
        blob, sha = freeze_config(self.cfg)
        cal = {}
        for role, mp in (("follower", self.mapping), ("leader", self.leader_mapping)):
            if mp is not None and mp.source_path:
                cal[role] = {"path": str(Path(mp.source_path).resolve()), "sha256": mp.sha256}
        cfile = Path(self.config_path) if self.config_path else None
        stale = []
        reg = self.cfg.get("calibration_registry")
        if reg:
            sh = stale_hashes(reg)
            stale = [r for r, c in cal.items() if c["sha256"] in sh]
        (w.dir / "config").mkdir(exist_ok=True)
        write_json_atomic(w.dir / "config" / "effective_config.json", {
            "config_sha256": sha, "hash_definition": "sha256 of json.dumps(config, sort_keys=True, separators=(',',':'), default=str)",
            "config": self.cfg, "config_file": {"path": str(cfile.resolve()), "sha256": sha256_file(cfile)} if cfile and cfile.exists() else None,
            "calibration_files": cal, "stale_calibration_in_use": stale,
            "calibration_registry": str(Path(reg).resolve()) if reg else None})
        w.update_meta(effective_config_sha256=sha, stale_calibration_in_use=stale)

    def start_episode(self, task: str, instruction: str = "", scene: dict | None = None, operator: str = "",
                      initial_conditions: dict | None = None, object_poses: list | None = None,
                      pilot: dict | None = None) -> EpisodeWriter:
        if self._writer is not None:
            raise RuntimeError("an episode is already being recorded")
        problems, warns = check_readiness(self.cfg, self.cal_dir, {n: self.sources_ok.get(n, False) for n in self.cameras},
                                          self.sources_ok.get("follower", False), self.is_mock)
        if problems:
            raise RuntimeError("readiness gates failed: " + "; ".join(problems))
        if pilot is not None:
            blockers = self.pilot_blockers()
            if blockers:
                raise RuntimeError("pilot recording blocked (nothing recorded): " + "; ".join(blockers))
        root = Path(self.cfg["paths"]["episodes_root"])
        root.mkdir(parents=True, exist_ok=True)
        if disk_free_gb(root) < self.cfg["recording"]["min_free_disk_gb"]:
            raise RuntimeError("insufficient free disk space")
        missing = [f"camera:{n}" for n in self.cameras if not self.sources_ok.get(n)]
        if self.follower is None or not self.sources_ok.get("follower"):
            missing.append("follower_state")
        if self.leader is None or not self.sources_ok.get("leader"):
            missing.append("leader_state")
        if self.commands is None or not self.sources_ok.get("commands"):
            missing.append("commands")
        if not any(c.get("depth") for c in self.cfg["cameras"] if c["enabled"]):
            missing.append("depth")
        if self.mapping is None:
            missing.append("calibrated_joint_angles")
        pilot, controller_flags = self._pilot_block(pilot)
        cal_info = None
        if self.cal_dir:
            cj = read_json(self.cal_dir / "calibration.json")
            cal_info = {"calibration_id": cj["calibration_id"], "sha256": sha256_file(self.cal_dir / "calibration.json"),
                        "operator_accepted": cj["operator_accepted"]}
        meta = {
            "task_name": task, "instruction": instruction, "operator": operator, "scene": scene or {},
            "initial_conditions": initial_conditions or {}, "object_poses": object_poses or [],
            "object_poses_provenance": "operator-entered only; never estimated automatically",
            "mode": self.cfg["mode"], "reference_camera": self.cfg["sync"]["reference_camera"],
            "software": {"collector": __version__, "python": platform.python_version(), "platform": platform.platform(),
                         "repo_commit": repo_commit(), "packages": _versions()},
            "config_snapshot": self.cfg, "config_path": self.config_path,
            "calibration": {"camera_to_mat": cal_info,
                            "robot_motor_calibration_sha256": self.mapping.sha256 if self.mapping else None,
                            "robot_base_to_mat": None},
            "control_mode": self.cfg["robot"].get("control_mode"),
            "controller": self._controller_block(),
            "pilot": pilot,
            "hardware_authorisation": hardware_gate.provenance(),
            "quality_flags": {"mock_data": self.is_mock, "diagnostic": self.cfg["mode"] == "diagnostic",
                              "camera_calibration_missing": cal_info is None,
                              "camera_calibration_accepted": bool(cal_info and cal_info["operator_accepted"]),
                              "missing_modalities": missing, "readiness_warnings": warns,
                              **pilot_quality_flags(pilot), **controller_flags,
                              "synchronisation": "none (asynchronous host-timestamped streams)"},
            "streams": {n: {"role": c["role"], "depth": bool(c.get("depth"))} for c in self.cfg["cameras"]
                        if c["enabled"] and self.sources_ok.get(c["name"]) for n in [c["name"]]},
        }
        w = EpisodeWriter(root, meta, self.cfg["recording"], self.cfg["sync"])
        self._write_static(w)
        self._writer = w
        return w

    def _controller_block(self) -> dict:
        sess = (self.descr.get("commands") or {}).get("session")
        return {"configured": self.cfg["robot"].get("controller") or {}, "published_session": sess,
                "published_session_sha256": (self.descr.get("commands") or {}).get("session_sha256"),
                "note": "recorder is read-only; the control loop belongs to the teleoperation process"}

    def _pilot_block(self, pilot: dict | None) -> tuple[dict | None, dict]:
        """Attach the first measured follower state as robot_start_configuration and cross-check provenance."""
        flags: dict = {}
        sess = (self.descr.get("commands") or {}).get("session")
        tap = (self.descr.get("commands") or {}).get("backend") == "jsonl_tap"
        flags["controller_session_missing"] = bool(tap and sess is None)
        fsha = (((sess or {}).get("follower") or {}).get("calibration") or {}).get("sha256")
        flags["follower_calibration_hash_matches_controller"] = (
            None if not (fsha and self.mapping) else fsha == self.mapping.sha256)
        if pilot is not None:
            pilot = dict(pilot)
            s = self.latest_state
            fresh = s and (mono_ns() - s["t_end_ns"]) < 1_000_000_000
            if fresh and s.get("read_ok", True) and s.get("raw"):
                pilot["robot_start_configuration"] = {
                    "source": "last measured follower state before episode start (not commanded, not leader)",
                    "t_host_mono_ns": (s["t_start_ns"] + s["t_end_ns"]) // 2, "raw": s["raw"],
                    "lerobot_normalized": s.get("normalized"),
                    "joint_angles_rad": self.mapping.convert(s["raw"])["joint_angles_rad"] if self.mapping else None}
        return pilot, flags

    def _write_static(self, w: EpisodeWriter) -> None:
        d = w.dir
        for n, desc in self.descr.items():
            if n in self.cameras:
                (d / "streams" / n).mkdir(parents=True, exist_ok=True)
                write_json_atomic(d / "streams" / n / "camera.json", desc)
            else:
                write_json_atomic(d / "telemetry" / f"{n}_source.json", desc)
        (d / "calibration").mkdir()
        if self.cal_dir:
            write_json_atomic(d / "calibration" / "camera_to_mat.json", read_json(self.cal_dir / "calibration.json"))
        if self.mapping:
            write_json_atomic(d / "calibration" / "robot_motor_calibration.json", self.mapping.describe())
        if self.leader_mapping:
            write_json_atomic(d / "calibration" / "leader_motor_calibration.json", self.leader_mapping.describe())
        write_json_atomic(d / "calibration" / "joints.json", {"follower_joints": self.cfg["robot"]["follower"]["joints"]})
        self._freeze_config(w)

    def finish_episode(self, outcome="unlabelled", failure_category=None, notes="", label_source=None) -> Path:
        """label_source: who decided the outcome (operator_visual | operator_measured); required for success/failure."""
        if self._writer is None:
            raise RuntimeError("no active episode")
        label_source = label_source or ("none" if outcome in ("unlabelled", "aborted") else None)
        if label_source is None:
            raise ValueError("outcome 'success'/'failure' needs label_source (operator_visual | operator_measured)")
        if outcome in ("success", "failure", "unlabelled"):
            check_label(outcome, label_source)
        w, self._writer = self._writer, None
        w.update_meta(pilot={**(w.meta.get("pilot") or {}), "outcome_label_source": label_source})
        out = w.finalize(outcome, failure_category, notes)
        self._auto_export(out)
        return out

    def abort_episode(self, reason: str) -> Path:
        w, self._writer = self._writer, None
        if w is None:
            raise RuntimeError("no active episode")
        return w.finalize("aborted", None, "", aborted_reason=reason or "unspecified")

    def _auto_export(self, ep: Path) -> None:
        from .exports import ExportBlocked, export_episode

        for prof in self.cfg["recording"]["auto_export"]:
            try:
                export_episode(ep, prof)
            except ExportBlocked as e:
                print(f"[export:{prof}] BLOCKED: {e}")
            except Exception as e:  # never lose the master because an export failed
                print(f"[export:{prof}] FAILED: {type(e).__name__}: {e}")

    # ---------- stats ----------
    def stats(self) -> dict:
        w = self._writer
        s = {"streams": {k: {kk: vv for kk, vv in v.items() if kk != "_t"} for k, v in self.status.items()},
             "recording": w is not None, "episode_id": w.id if w else None,
             "mock": self.is_mock, "calibration": self.cal_dir.name if self.cal_dir else None}
        if w:
            s["queue"] = w.queue_stats()
            s["writer_rate_hz"] = w.writer_rate_hz()
            s["dropped"] = dict(w.counts["dropped"])
            s["abort_reason"] = w.abort_reason
        return s


def _versions() -> dict:
    out = {}
    for m in ("numpy", "cv2", "pyrealsense2", "yaml"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[m] = None
    return out
