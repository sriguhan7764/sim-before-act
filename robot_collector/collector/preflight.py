"""Pre-recording checks. Nothing here moves hardware, writes a motor register, or starts a camera stream.

- freeze_config: canonical JSON + sha256 of the effective configuration an episode is recorded with.
- check_calibration_files: the config must use the ACTIVE follower/leader calibration files from the registry; stale files
  are refused by path and by hash.
- probe_realsense: enumerate the D435's SUPPORTED stream profiles and ask the SDK whether the requested colour + depth
  combination can be resolved (``rs.config.can_resolve``/``resolve``: no streaming is started).
  The profile actually negotiated is only known after the stream starts; the adapter verifies it then and refuses a mismatch.
- probe_v4l2: list the wrist camera's formats with ``v4l2-ctl --list-formats-ext`` (enumeration only).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from .util import sha256_file


def freeze_config(cfg: dict) -> tuple[bytes, str]:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()
    return blob, hashlib.sha256(blob).hexdigest()


def load_registry(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _chk(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def check_calibration_files(cfg: dict, registry_path: str | Path) -> list[dict]:
    """PASS only if each configured calibration file is the registry's active file (same path and sha256)."""
    reg = load_registry(registry_path)
    base = Path(registry_path).resolve().parent.parent          # robot_collector/
    stale = {s["sha256"]: s for s in reg.get("stale", [])}
    stale_paths = {(base / s["path"]).resolve(): s for s in reg.get("stale", [])}
    out = []
    for role in ("follower", "leader"):
        want = reg["active"][role]
        cf = (cfg["robot"][role] or {}).get("calibration_file")
        label = f"{role} calibration file is the ACTIVE one"
        if not cf:
            out.append(_chk(label, "FAIL", "no calibration_file configured"))
            continue
        p = Path(cf)
        p = (p if p.is_absolute() else Path.cwd() / p).resolve()
        if p in stale_paths or not p.exists():
            why = stale_paths[p]["reason"] if p in stale_paths else "file not found"
            out.append(_chk(label, "FAIL", f"{p}: {'STALE - ' if p in stale_paths else ''}{why}"))
            continue
        sha = sha256_file(p)
        if sha in stale:
            out.append(_chk(label, "FAIL", f"{p} has the hash of a STALE file: {stale[sha]['reason']}"))
        elif p != (base / want["path"]).resolve():
            out.append(_chk(label, "FAIL", f"{p} is not the registry's active {role} file {want['path']}"))
        elif sha != want["sha256"]:
            out.append(_chk(label, "FAIL", f"{p} sha256 {sha[:16]}... != registry {want['sha256'][:16]}... (file changed?)"))
        else:
            out.append(_chk(label, "PASS", f"{p.name} sha256 {sha[:16]}..."))
    return out


def stale_hashes(registry_path: str | Path) -> dict[str, str]:
    return {s["sha256"]: s["reason"] for s in load_registry(registry_path).get("stale", [])}


def probe_realsense(serial: str | None, width: int, height: int, fps: int, depth: bool = True) -> dict:
    from . import hardware_gate

    hardware_gate.require("RealSense device enumeration (query_devices)")
    import pyrealsense2 as rs

    ctx = rs.context()
    dev = next((d for d in ctx.query_devices() if serial is None or d.get_info(rs.camera_info.serial_number) == str(serial)), None)
    if dev is None:
        return {"found": False, "serial_requested": serial, "supported": False, "detail": "device not present"}
    info = {k: dev.get_info(getattr(rs.camera_info, k)) for k in ("name", "serial_number", "firmware_version", "usb_type_descriptor",
                                                                    "physical_port") if dev.supports(getattr(rs.camera_info, k))}
    sup: dict[str, list] = {"color": [], "depth": []}
    for s in dev.query_sensors():
        for p in s.get_stream_profiles():
            kind = str(p.stream_type()).split(".")[-1]
            if kind in sup and p.is_video_stream_profile():
                v = p.as_video_stream_profile()
                sup[kind].append({"width": v.width(), "height": v.height(), "fps": v.fps(), "format": str(v.format()).split(".")[-1]})
    uniq = {k: sorted({json.dumps(x, sort_keys=True) for x in v}) for k, v in sup.items()}
    sup = {k: [json.loads(x) for x in v] for k, v in uniq.items()}
    want_c = {"width": width, "height": height, "fps": fps, "format": "rgb8"}
    want_d = {"width": width, "height": height, "fps": fps, "format": "z16"}
    res = {"found": True, "device": info, "sdk": {"pyrealsense2": getattr(rs, "__version__", None)},
           "requested": {"color": want_c, **({"depth": want_d} if depth else {})},
           "color_supported_alone": want_c in sup["color"], "depth_supported_alone": want_d in sup["depth"],
           "color_profiles_at_requested_resolution": [p for p in sup["color"] if (p["width"], p["height"]) == (width, height)],
           "depth_profiles_at_requested_resolution": [p for p in sup["depth"] if (p["width"], p["height"]) == (width, height)]}
    try:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(info["serial_number"])
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        wrapper = rs.pipeline_wrapper(pipe)
        res["combination_resolvable"] = bool(cfg.can_resolve(wrapper))
        if res["combination_resolvable"]:
            prof = cfg.resolve(wrapper)          # resolves against the device; does NOT start streaming
            resolved = {}
            for name, st in (("color", rs.stream.color), ("depth", rs.stream.depth)):
                try:
                    v = prof.get_stream(st).as_video_stream_profile()
                    resolved[name] = {"width": v.width(), "height": v.height(), "fps": v.fps(), "format": str(v.format()).split(".")[-1]}
                except Exception:
                    pass
            res["resolved_without_streaming"] = resolved
    except Exception as e:
        res["combination_resolvable"] = None
        res["resolve_error"] = f"{type(e).__name__}: {e}"
    res["supported"] = bool(res["color_supported_alone"] and (not depth or res["depth_supported_alone"])
                            and res.get("combination_resolvable"))
    res["note"] = ("supported-profile enumeration only; the profile actually negotiated is verified when the stream starts "
                   "and a mismatch aborts the episode")
    return res


def probe_v4l2(device: str, width: int, height: int, fps: float) -> dict:
    from . import hardware_gate

    hardware_gate.require(f"v4l2-ctl on {device}")
    try:
        txt = subprocess.run(["v4l2-ctl", "-d", str(device), "--list-formats-ext"], capture_output=True, text=True, timeout=15).stdout
    except Exception as e:
        return {"checked": False, "detail": f"{type(e).__name__}: {e}"}
    fmts, cur, size = {}, None, None
    for line in txt.splitlines():
        m = re.search(r"\[\d+\]: '(\w{4})'", line)
        if m:
            cur = m.group(1)
            fmts[cur] = {}
        m = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if m and cur:
            size = f"{m.group(1)}x{m.group(2)}"
            fmts[cur][size] = []
        m = re.search(r"\(([\d.]+) fps\)", line)
        if m and cur and size:
            fmts[cur][size].append(float(m.group(1)))
    key = f"{width}x{height}"
    ok = {f: key in s and any(abs(x - fps) < 0.5 for x in s[key]) for f, s in fmts.items()}
    return {"checked": bool(fmts), "formats": {f: {k: v for k, v in s.items() if k == key} for f, s in fmts.items()},
            "supported_by_format": ok, "supported": any(ok.values()),
            "note": "enumeration via v4l2-ctl; the negotiated format is recorded when the stream opens (OpenCV may pick any listed format)"}


def run_preflight(cfg: dict, registry_path: str | Path) -> dict:
    _, sha = freeze_config(cfg)
    res = {"effective_config_sha256": sha, "calibration_files": check_calibration_files(cfg, registry_path), "cameras": {}}
    for c in cfg["cameras"]:
        if not c["enabled"]:
            continue
        try:
            if c["backend"] == "realsense":
                res["cameras"][c["name"]] = probe_realsense(c.get("serial"), c["width"], c["height"], c["fps"], bool(c.get("depth")))
            elif c["backend"] == "opencv":
                res["cameras"][c["name"]] = probe_v4l2(c["device"], c["width"], c["height"], c["fps"])
            else:
                res["cameras"][c["name"]] = {"checked": False, "detail": f"backend {c['backend']}: no probe"}
        except Exception as e:
            res["cameras"][c["name"]] = {"checked": False, "error": f"{type(e).__name__}: {e}"}
    return res


def open_cameras_check(cfg: dict, seconds: float = 2.0) -> dict:
    """Start each enabled camera stream for a moment (camera access only; no robot code is imported or run) and report the
    NEGOTIATED profile, frame shapes/dtypes, valid-depth fraction and measured rate. A negotiation mismatch is reported, not fixed."""
    import time

    import numpy as np

    from .adapters.factory import make_camera
    from .negotiation import ProfileMismatch
    from .sync import depth_freshness
    out = {}
    for c in cfg["cameras"]:
        if not c["enabled"]:
            continue
        cam = make_camera(c)
        rep: dict = {"opened": False}
        try:
            cam.open()
            rep["opened"] = True
            d = cam.describe()
            rep["negotiation"] = d.get("negotiation")
            rep["device"] = d.get("device")
            rep["depth_native"], rep["depth_aligned"] = d.get("depth_native"), d.get("depth_aligned")
            n, t0, host = 0, time.monotonic(), []
            t0_ns = time.monotonic_ns()
            fnum, dts, dfnum, ddts, none_reads = [], [], [], [], 0
            shapes: dict = {}
            inv_n, inv_a = [], []
            domains = set()
            while time.monotonic() - t0 < seconds:
                f = cam.read(timeout_s=1.0)
                if f is None:
                    none_reads += 1
                    continue
                n += 1
                host.append(f.host_mono_ns)
                for lst, v in ((fnum, f.frame_number), (dts, f.device_timestamp_ms),
                               (dfnum, getattr(f, "depth_frame_number", None)), (ddts, getattr(f, "depth_device_timestamp_ms", None))):
                    if v is not None:
                        lst.append(v)
                domains.add(f.timestamp_domain)
                shapes["color"] = [list(f.color_rgb.shape), str(f.color_rgb.dtype)]
                if f.depth_native is not None:
                    shapes["depth_native"] = [list(f.depth_native.shape), str(f.depth_native.dtype)]
                    inv_n.append(float((f.depth_native == 0).mean()))
                if f.depth_aligned is not None:
                    shapes["depth_aligned"] = [list(f.depth_aligned.shape), str(f.depth_aligned.dtype)]
                    inv_a.append(float((f.depth_aligned == 0).mean()))
            def _seq(v):
                # frame counters: gaps = number of skipped counter values
                return {"n": len(v), "skipped_frames": sum(b - a - 1 for a, b in zip(v, v[1:]) if b - a > 1),
                        "duplicates": sum(1 for a, b in zip(v, v[1:]) if b == a),
                        "non_monotonic": sum(1 for a, b in zip(v, v[1:]) if b < a)}

            def _ts(v, scale):
                # timestamps: interval statistics only (a "gap" is not defined without a nominal period)
                d = np.diff(np.asarray(v, dtype=np.float64)) / scale if len(v) > 1 else np.array([])
                return {"n": len(v), "duplicates": int((d == 0).sum()), "non_monotonic": int((d < 0).sum()),
                        "dt_ms": {"p50": float(np.median(d)), "max": float(d.max())} if d.size else None}

            t_end_ns = time.monotonic_ns()
            dframes = [{"depth_frame_number": a, "depth_device_timestamp_ms": b, "device_timestamp_ms": c_, "depth_native_file": True}
                       for a, b, c_ in zip(dfnum, ddts, dts)] if dfnum else []
            rep["timing_boundaries"] = {
                "requested_capture_window_s": seconds,
                "loop_wall_s": (t_end_ns - t0_ns) / 1e9,
                "first_frame_latency_s": (host[0] - t0_ns) / 1e9 if host else None,
                "first_to_last_frame_s": (host[-1] - host[0]) / 1e9 if n > 1 else None,
                "note": "loop_wall_s = first read request to loop exit (stream warm-up/open excluded); first_frame_latency_s = wait "
                        "before the first frame arrived; measured_rate_hz = (frames-1)/first_to_last_frame_s, NOT frames/loop_wall_s"}
            if dframes:
                rep["depth_freshness"] = {k: v for k, v in depth_freshness(dframes).items() if k != "status"}
            rep.update({"read_timeouts_or_none": none_reads, "host_receipt": _ts(host, 1e6),
                        "color_frame_number": _seq(fnum), "color_device_timestamp_ms": _ts(dts, 1.0),
                        "depth_frame_number": _seq(dfnum), "depth_device_timestamp_ms": _ts(ddts, 1.0)})
            rep.update({"frames": n, "frame_shapes_dtypes": shapes, "timestamp_domains": sorted(map(str, domains)),
                        "measured_rate_hz": (n - 1) / ((host[-1] - host[0]) / 1e9) if n > 1 else None,
                        "invalid_depth_fraction_native_mean": float(np.mean(inv_n)) if inv_n else None,
                        "invalid_depth_fraction_aligned_mean": float(np.mean(inv_a)) if inv_a else None})
        except ProfileMismatch as e:
            rep["error"] = f"ProfileMismatch: {e}"
        except Exception as e:
            rep["error"] = f"{type(e).__name__}: {e}"
        finally:
            try:
                cam.close()
            except Exception:
                pass
        out[c["name"]] = rep
    return out
