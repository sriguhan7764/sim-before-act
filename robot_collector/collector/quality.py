"""Post-hoc quality report for a master episode (read-only; writes only to ``<episode>.reports/``).

Reports measurements and assumptions; it does not repair, interpolate or relabel anything. Thresholds that turn a
measurement into a pass/fail live in ``acceptance.py`` and are stated there.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .joints import LEROBOT_NORMALIZED_UNITS, AngleMapping
from .pilot import eligibility, pose_missing
from .sync import depth_freshness
from .util import iter_jsonl, read_json, write_json_atomic

CLOCK_ASSUMPTIONS = [
    "All association uses host CLOCK_MONOTONIC (camera receipt time; robot state = midpoint of the bus read; command = "
    "time stamped just before send_action). Processes on one Linux host share this clock.",
    "RealSense device timestamps are in the device's own domain and are NOT compared with host times "
    "(only an informational offset is reported).",
    "No hardware synchronisation exists between cameras and motors; residuals are host-receipt differences and include "
    "unknown capture/exposure latency and bus/USB latency.",
    "No interpolation: every associated value is an original sample; samples farther than the tolerance are reported unmatched.",
]

COMMAND_SEMANTICS = (
    "Recorded commands are SUBMITTED Goal_Position values (clipped follower target + raw ticks computed with the bus "
    "normalisation). They are not read back from the Goal_Position register and have no acknowledgement; whether the motor "
    "reached them is evidenced only by the MEASURED follower_state. The leader measurement is stored separately.")


def _dt_stats_ms(t_ns: list[int]) -> dict:
    if len(t_ns) < 2:
        return {"n": len(t_ns), "rate_hz": None}
    d = np.diff(np.asarray(t_ns, dtype=np.int64)) / 1e6
    span = (t_ns[-1] - t_ns[0]) / 1e9
    return {"n": len(t_ns), "rate_hz": (len(t_ns) - 1) / span if span > 0 else None, "dt_ms_p50": float(np.median(d)),
            "dt_ms_p95": float(np.percentile(d, 95)), "dt_ms_max": float(d.max()), "dt_ms_min": float(d.min())}


def _ts_health(t_ns: list[int]) -> dict:
    return {"duplicate_timestamps": sum(1 for a, b in zip(t_ns, t_ns[1:]) if b == a),
            "non_monotonic": sum(1 for a, b in zip(t_ns, t_ns[1:]) if b < a)}


def _res_stats(res: list[float]) -> dict | None:
    if not res:
        return None
    a = np.abs(np.asarray(res, dtype=float)) / 1e6
    return {"n": len(res), "mean_abs_ms": float(a.mean()), "p95_abs_ms": float(np.percentile(a, 95)), "max_abs_ms": float(a.max())}


def _camera_section(ep: Path, meta: dict) -> dict:
    out = {}
    for c in sorted(meta.get("streams", {})):
        fr = list(iter_jsonl(ep / "streams" / c / "frames.jsonl"))
        cam = read_json(ep / "streams" / c / "camera.json") if (ep / "streams" / c / "camera.json").exists() else {}
        host = [f["host_mono_ns"] for f in fr]
        idx = [f["recv_index"] for f in fr]
        fn = [f["frame_number"] for f in fr if f.get("frame_number") is not None]
        dev = [f["device_timestamp_ms"] for f in fr if f.get("device_timestamp_ms") is not None]
        sec = {"frames": len(fr), "role": meta["streams"][c].get("role"), "timing": _dt_stats_ms(host), **_ts_health(host),
               "recorder_dropped_recv_gaps": (max(idx) + 1 - len(idx)) if idx else 0,
               "device_frame_number_gaps": sum(b - a - 1 for a, b in zip(fn, fn[1:]) if b - a > 1),
               "device_frame_counter_available": bool(fn),
               "device_timestamp_available": bool(dev),
               "device_timestamp_duplicates": sum(1 for a, b in zip(dev, dev[1:]) if b == a),
               "device_timestamp_non_monotonic": sum(1 for a, b in zip(dev, dev[1:]) if b < a),
               "timestamp_domain": fr[0].get("timestamp_domain") if fr else None,
               "stream_profile": cam.get("stream_profile"), "device": cam.get("device"),
               "depth_scale": cam.get("depth_scale"), "has_depth": bool(meta["streams"][c].get("depth")),
               "has_color_intrinsics": bool(cam.get("color_intrinsics")),
               "has_depth_intrinsics": bool(cam.get("depth_intrinsics")),
               "has_depth_to_color_extrinsics": bool(cam.get("extrinsics_depth_to_color"))}
        if sec["has_depth"] and fr:
            import cv2

            df = depth_freshness(fr)
            sec["depth_freshness"] = {k: v for k, v in df.items() if k not in ("status", "age_ms")}
            sec["depth_freshness"]["note"] = ("repeated_previous_depth = the frameset carried the previous depth frame again (kept with its "
                                              "original frame number/timestamp); missing = no depth frame/file. Aligned depth inherits "
                                              "the status of the native depth frame it was derived from.")
            sec["negotiation"] = cam.get("negotiation")
            for key in ("depth_native", "depth_aligned"):
                blk = cam.get(key) or {}
                a = cv2.imread(str(ep / "streams" / c / fr[0][f"{key}_file"]), cv2.IMREAD_UNCHANGED) if fr[0].get(f"{key}_file") else None
                sec[f"{key}_described_size"] = [blk.get("width"), blk.get("height")]
                sec[f"{key}_file_size_dtype"] = None if a is None else [a.shape[1], a.shape[0], str(a.dtype)]
                sec[f"{key}_geometry"] = blk.get("geometry")
            sec["color_file_size"] = None
            if fr[0].get("color_file"):
                ci = cv2.imread(str(ep / "streams" / c / fr[0]["color_file"]), cv2.IMREAD_UNCHANGED)
                sec["color_file_size"] = None if ci is None else [ci.shape[1], ci.shape[0]]
            frac = []
            for f in fr[:: max(1, len(fr) // 8)]:
                for key in ("depth_native_file", "depth_aligned_file"):
                    if f.get(key):
                        a = cv2.imread(str(ep / "streams" / c / f[key]), cv2.IMREAD_UNCHANGED)
                        if a is not None:
                            frac.append((key, float((a == 0).mean())))
            for key in ("depth_native_file", "depth_aligned_file"):
                v = [x for k, x in frac if k == key]
                sec[f"invalid_depth_fraction_{key.removesuffix('_file')}"] = {
                    "sampled_frames": len(v), "mean": float(np.mean(v)) if v else None,
                    "note": "0 = no measurement (invalid), never a zero-distance reading"}
        out[c] = sec
    return out


def _telemetry_section(ep: Path) -> dict:
    out = {}
    for k in ("follower_state", "leader_state", "commands"):
        recs = list(iter_jsonl(ep / "telemetry" / f"{k}.jsonl"))
        t = [r["t_start_ns"] for r in recs]
        seq = [r["seq"] for r in recs]
        out[k] = {"records": len(recs), "timing": _dt_stats_ms(t), **_ts_health(t),
                  "seq_gaps": (seq[-1] - seq[0] + 1 - len(seq)) if seq else 0,
                  "read_failures": sum(1 for r in recs if r.get("read_ok") is False),
                  "has_raw": any("raw" in r for r in recs), "has_normalized": any("normalized" in r for r in recs)}
        if k == "commands" and recs:
            out[k]["units_seen"] = sorted({str(r.get("units")) for r in recs})
            out[k]["clipped_fraction"] = (sum(1 for r in recs if any((r.get("clipped") or {}).values())) / len(recs)
                                          if any("clipped" in r for r in recs) else None)
            out[k]["ack_available"] = any(r.get("ack") is not None for r in recs)
    return out


def _association_section(ep: Path) -> dict:
    rows = list(iter_jsonl(ep / "sync" / "associations.jsonl"))
    sync = read_json(ep / "sync" / "sync_report.json")
    out = {"reference_frames": len(rows), "tolerances_ms": sync.get("tolerances_ms"), "method": sync.get("method"),
           "hardware_synchronisation": sync.get("hardware_synchronisation"), "sources": {}}
    for key in ("follower_state", "leader_state", "command_prev", "command_next"):
        st = Counter(r[key]["status"] for r in rows)
        res = [r[key]["residual_ns"] for r in rows if r[key]["status"] == "matched"]
        out["sources"][key] = {"status_counts": dict(st), "unmatched_frames": len(rows) - st.get("matched", 0),
                               "residual_ms": _res_stats(res)}
    cams = {}
    for r in rows:
        for c, a in r["cameras"].items():
            cams.setdefault(c, []).append(a)
    for c, lst in cams.items():
        out["sources"][f"camera:{c}"] = {"status_counts": dict(Counter(a["status"] for a in lst)),
                                         "residual_ms": _res_stats([a["residual_ns"] for a in lst if a["status"] == "matched"])}
    out["device_clock_offset_estimates"] = sync.get("device_clock_offset_estimates")
    return out


def _unit_section(ep: Path, meta: dict) -> dict:
    out: dict = {}
    pcal = ep / "calibration" / "robot_motor_calibration.json"
    state = list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))
    ok = [r for r in state if r.get("read_ok", True) and r.get("raw")]
    if not pcal.exists() or not ok:
        return {"note": "no follower calibration or no valid follower state; joint-angle checks not possible"}
    cal = read_json(pcal)
    mp = AngleMapping(cal["joints"], cal["calibration"], cal.get("calibration_file"), cal.get("calibration_sha256"))
    out["follower_calibration"] = {"file": cal.get("calibration_file"), "sha256": cal.get("calibration_sha256")}
    per = {}
    diff_deg, diff_grip = 0.0, 0.0
    has_norm = any("normalized" in r for r in ok)
    okn = [r for r in ok if "normalized" in r]
    convn = [mp.convert(r["raw"]) for r in okn]
    rev_names = [x["name"] for x in mp.joints if x["type"] == "revolute"]
    for j in cal["joints"]:
        n = j["name"]
        raw = np.array([r["raw"][n] for r in ok])
        c = cal["calibration"][n]
        conv = [mp.convert(r["raw"]) for r in ok]
        if j["type"] == "revolute":
            deg = np.degrees([cv["joint_angles_rad"][rev_names.index(n)] for cv in conv])
            rng = {"measured_deg_min": float(deg.min()), "measured_deg_max": float(deg.max())}
            if okn:
                dn = np.degrees([cv["joint_angles_rad"][rev_names.index(n)] for cv in convn])
                diff_deg = max(diff_deg, float(np.abs(np.array([r["normalized"][n] for r in okn]) - dn).max()))
        else:
            g = np.array([cv["gripper"][n]["value"] for cv in conv])
            rng = {"measured_gripper_0_100_min": float(g.min()), "measured_gripper_0_100_max": float(g.max())}
            if okn:
                gn = np.array([cv["gripper"][n]["value"] for cv in convn])
                diff_grip = max(diff_grip, float(np.abs(np.array([r["normalized"][n] for r in okn]) - gn).max()))
        per[n] = {"raw_min": int(raw.min()), "raw_max": int(raw.max()), "calibration_range": [c["range_min"], c["range_max"]],
                  "outside_calibration_range": bool((raw < c["range_min"]).any() or (raw > c["range_max"]).any()), **rng}
    out["per_joint"] = per
    out["collector_mapping_vs_lerobot_normalized"] = (
        {"max_abs_diff_arm_deg": diff_deg, "max_abs_diff_gripper_pct": diff_grip,
         "note": "collector AngleMapping vs the values LeRobot computed in the bridge; should be ~0"} if has_norm else
        {"note": "no lerobot-normalised values published; cross-check not possible"})
    out["caveat"] = ("Zero is the midpoint of the calibration range, not a physically validated reference; signs and gripper "
                     "mapping are unvalidated (sim_before_act/docs/CALIBRATION_AUDIT.md). Encoder range != safe range.")
    return out


def _command_tracking(ep: Path) -> dict | None:
    rows = list(iter_jsonl(ep / "sync" / "associations.jsonl"))
    cmds = list(iter_jsonl(ep / "telemetry" / "commands.jsonl"))
    state = list(iter_jsonl(ep / "telemetry" / "follower_state.jsonl"))
    errs: dict[str, list[float]] = {}
    for r in rows:
        if r["command_prev"]["status"] != "matched" or r["follower_state"]["status"] != "matched":
            continue
        c, s = cmds[r["command_prev"]["index"]], state[r["follower_state"]["index"]]
        if c.get("units") != LEROBOT_NORMALIZED_UNITS or "normalized" not in s:
            return None
        for n, v in (c.get("values") or {}).items():
            if n in s["normalized"]:
                errs.setdefault(n, []).append(abs(float(v) - float(s["normalized"][n])))
    if not errs:
        return None
    return {"definition": "|latest follower target - measured follower position| at each reference frame, follower-normalised "
                          "units (degrees / gripper %); a lag/tracking statistic, not an error against ground truth",
            "per_joint": {n: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
                          for n, v in errs.items()}}


def _protocol_section(ep: Path, meta: dict) -> dict | None:
    """Per-segment motion statistics for operator-protocol episodes. Reported, never graded: no threshold is applied."""
    pr = meta.get("protocol")
    cues = (pr or {}).get("cues_issued") or []
    if not cues:
        return None
    end_ns = max([r.get("t_start_ns", 0) for r in iter_jsonl(ep / "telemetry" / "follower_state.jsonl")] or [0])
    recs = {k: [r for r in iter_jsonl(ep / "telemetry" / f"{k}.jsonl") if r.get("read_ok", True) and r.get("raw")]
            for k in ("follower_state", "leader_state")}
    segs = []
    for i, cu in enumerate(cues):
        a = cu["cue_host_mono_ns"]
        b = cues[i + 1]["cue_host_mono_ns"] if i + 1 < len(cues) else end_ns
        sec = {"label": cu["label"], "window_s": (b - a) / 1e9, "joints": {}}
        for k, rs in recs.items():
            w = [r for r in rs if a <= r["t_start_ns"] < b]
            sec[f"{k}_samples"] = len(w)
            for n in (w[0]["raw"] if w else {}):
                v = np.array([r["raw"][n] for r in w], dtype=float)
                sec["joints"].setdefault(n, {})[k] = {"raw_std_ticks": float(v.std()), "raw_peak_to_peak_ticks": float(np.ptp(v))}
        segs.append(sec)
    return {"segments": segs, "note": "raw ticks in SDK units after homing offset; follower_state is MEASURED, leader_state is the "
                                      "operator's leader arm. Statistics only: no pass/fail threshold is applied."}


def quality_report(ep: str | Path) -> dict:
    ep = Path(ep)
    meta = read_json(ep / "episode.json")
    q = meta.get("quality_flags", {})
    start = meta.get("start_wall_ns")
    end = meta.get("end_wall_ns")
    pilot = meta.get("pilot")
    rep = {
        "episode_id": meta["episode_id"], "status": meta.get("status"), "outcome": meta.get("outcome"),
        "synthetic_or_mock": bool(q.get("mock_data") or q.get("synthetic_test_fixture")),
        "data_origin": ("SYNTHETIC/MOCK - not a physical recording" if (q.get("mock_data") or q.get("synthetic_test_fixture"))
                        else "recorded by the collector with the configured real backends"),
        "duration_s": (end - start) / 1e9 if start and end else None,
        "pilot": pilot, "missing_modalities": q.get("missing_modalities", []),
        "quality_flags": q, "cameras": _camera_section(ep, meta), "telemetry": _telemetry_section(ep),
        "association": _association_section(ep), "units": _unit_section(ep, meta),
        "command_tracking": _command_tracking(ep), "clock_domain_assumptions": CLOCK_ASSUMPTIONS,
        "protocol_segments": _protocol_section(ep, meta),
        "effective_config_sha256": meta.get("effective_config_sha256"),
        "stale_calibration_in_use": meta.get("stale_calibration_in_use"),
        "command_semantics": COMMAND_SEMANTICS,
        "controller": meta.get("controller"), "eligibility_as_accepted_demonstration": eligibility(meta),
        "pose_status": {"cube_start": pose_missing((pilot or {}).get("cube_start_pose_mat_frame")),
                        "tray": pose_missing((pilot or {}).get("tray_pose_mat_frame"))} if pilot else None,
        "not_captured": ["object poses are not measured by the collector (operator-entered only)",
                         "no validated mat-to-robot-base transform", "no hardware synchronisation",
                         "no torque/current commands exist in the control path; none recorded"],
    }
    return rep


def render_markdown(rep: dict) -> str:
    L = [f"# Quality report: {rep['episode_id']}", "", f"**Data origin:** {rep['data_origin']}", "",
         f"- status: `{rep['status']}`; outcome: `{rep['outcome']}`; duration: {rep['duration_s']}",
         f"- missing modalities: {rep['missing_modalities'] or 'none'}",
         f"- eligible as accepted demonstration: **{rep['eligibility_as_accepted_demonstration']['eligible']}** "
         f"{rep['eligibility_as_accepted_demonstration']['reasons']}", "", "## Cameras", ""]
    L += ["| camera | frames | rate Hz | dt p95 ms | recorder drops | device gaps | dup ts | non-mono | domain |", "|---|---|---|---|---|---|---|---|---|"]
    for c, s in rep["cameras"].items():
        t = s["timing"]
        L.append(f"| {c} | {s['frames']} | {t.get('rate_hz') and round(t['rate_hz'], 2)} | "
                 f"{t.get('dt_ms_p95') and round(t['dt_ms_p95'], 1)} | {s['recorder_dropped_recv_gaps']} | "
                 f"{s['device_frame_number_gaps'] if s['device_frame_counter_available'] else 'n/a (no device counter)'} | "
                 f"{s['duplicate_timestamps']} | {s['non_monotonic']} | {s['timestamp_domain']} |")
    dcams = {c: s["depth_freshness"] for c, s in rep["cameras"].items() if s.get("depth_freshness")}
    if dcams:
        L += ["", "### Depth freshness (per colour frame; repeated depth keeps its original frame number/timestamp)", "",
              "| camera | fresh | repeated previous depth | missing | depth frame-number skips | colour-depth device ts offset max ms |",
              "|---|---|---|---|---|---|"]
        for c, d in dcams.items():
            k = d["counts"]
            L.append(f"| {c} | {k['fresh']} | {k['repeated_previous_depth']} | {k['missing']} | "
                     f"{d['depth_frame_number_skipped']} | {d['color_depth_device_timestamp_offset_ms_max']} |")
        L += ["", "Depth age = colour-frame host receipt minus first host receipt of that depth frame (lower bound; sensor "
              "exposure latency unknown).", "", "| camera | repeat fraction | longest repeat run | depth age max ms | depth age mean ms |",
              "|---|---|---|---|---|"]
        for c, d in dcams.items():
            a = d.get("age_ms_stats") or {}
            rf = d.get("repeat_fraction")
            L.append(f"| {c} | {'n/a' if rf is None else round(rf, 3)} | {d.get('longest_repeat_run')} | "
                     f"{a.get('max') and round(a['max'], 2)} | {a.get('mean') and round(a['mean'], 2)} |")
    L += ["", "## Telemetry", "", "| stream | records | rate Hz | dt max ms | seq gaps | failed reads | dup ts | non-mono |", "|---|---|---|---|---|---|---|---|"]
    for k, s in rep["telemetry"].items():
        t = s["timing"]
        L.append(f"| {k} | {s['records']} | {t.get('rate_hz') and round(t['rate_hz'], 2)} | "
                 f"{t.get('dt_ms_max') and round(t['dt_ms_max'], 1)} | {s['seq_gaps']} | {s['read_failures']} | "
                 f"{s['duplicate_timestamps']} | {s['non_monotonic']} |")
    L += ["", "## Association (sample time - frame receipt time)", "", f"tolerances (ms): {rep['association']['tolerances_ms']}; "
          f"hardware synchronisation: **{rep['association']['hardware_synchronisation']}**", "",
          "| source | matched/unmatched | residual mean / p95 / max |ms| |", "|---|---|---|"]
    for k, s in rep["association"]["sources"].items():
        r = s["residual_ms"]
        L.append(f"| {k} | {s['status_counts']} | " + (f"{r['mean_abs_ms']:.2f} / {r['p95_abs_ms']:.2f} / {r['max_abs_ms']:.2f}" if r else "n/a") + " |")
    u = rep["units"]
    if "per_joint" in u:
        L += ["", "## Joint ranges vs calibration", "", "| joint | raw min..max | calibration range | outside | measured |", "|---|---|---|---|---|"]
        for n, j in u["per_joint"].items():
            meas = (f"{j['measured_deg_min']:.1f}..{j['measured_deg_max']:.1f} deg" if "measured_deg_min" in j else
                    f"{j['measured_gripper_0_100_min']:.1f}..{j['measured_gripper_0_100_max']:.1f} (0-100)")
            L.append(f"| {n} | {j['raw_min']}..{j['raw_max']} | {j['calibration_range']} | {j['outside_calibration_range']} | {meas} |")
        L += ["", f"collector mapping vs LeRobot normalised: {u['collector_mapping_vs_lerobot_normalized']}", "", f"> {u['caveat']}"]
    if rep["command_tracking"]:
        L += ["", "## Command tracking (lag statistic)", "", rep["command_tracking"]["definition"], ""]
        for n, s in rep["command_tracking"]["per_joint"].items():
            L.append(f"- {n}: median {s['median']:.2f}, p95 {s['p95']:.2f}, max {s['max']:.2f}")
    L += ["", "## Stream profiles (requested vs negotiated) and depth representations", ""]
    for c, s in rep["cameras"].items():
        neg = s.get("negotiation")
        if neg:
            L.append(f"- {c}: requested {neg.get('requested')} | actual {neg.get('actual')} | matches request: "
                     f"**{neg.get('matches_request')}** (silent substitution: {neg.get('silent_substitution')})")
        elif s.get("stream_profile"):
            L.append(f"- {c}: profile {s['stream_profile']} (no negotiation record)")
        if s.get("has_depth"):
            L.append(f"  - colour file {s.get('color_file_size')}; native depth described {s.get('depth_native_described_size')} / file "
                     f"{s.get('depth_native_file_size_dtype')} ({s.get('depth_native_geometry')}); aligned depth described "
                     f"{s.get('depth_aligned_described_size')} / file {s.get('depth_aligned_file_size_dtype')} "
                     f"({s.get('depth_aligned_geometry')}); invalid(0) fraction native "
                     f"{(s.get('invalid_depth_fraction_depth_native') or {}).get('mean')}, aligned "
                     f"{(s.get('invalid_depth_fraction_depth_aligned') or {}).get('mean')}")
    L += ["", "## Frozen configuration", "", f"- effective config sha256: `{rep.get('effective_config_sha256')}`",
          f"- stale calibration in use: {rep.get('stale_calibration_in_use')}", "", "## Command semantics", "", rep["command_semantics"]]
    if rep.get("protocol_segments"):
        L += ["", "## Protocol segments (statistics only, no thresholds)", ""]
        for sg in rep["protocol_segments"]["segments"]:
            L.append(f"- **{sg['label']}** ({sg['window_s']:.1f} s; follower samples {sg['follower_state_samples']}, leader {sg['leader_state_samples']})")
            for n, j in sg["joints"].items():
                f, l = j.get("follower_state", {}), j.get("leader_state", {})
                L.append(f"  - {n}: follower std {f.get('raw_std_ticks', float('nan')):.2f} / p-p {f.get('raw_peak_to_peak_ticks', float('nan')):.0f} ticks; "
                         f"leader std {l.get('raw_std_ticks', float('nan')):.2f} / p-p {l.get('raw_peak_to_peak_ticks', float('nan')):.0f} ticks")
        L.append(f"\n> {rep['protocol_segments']['note']}")
    L += ["", "## Clock-domain assumptions", ""] + [f"- {a}" for a in rep["clock_domain_assumptions"]]
    L += ["", "## Not captured", ""] + [f"- {a}" for a in rep["not_captured"]]
    return "\n".join(L) + "\n"


def reports_dir(ep: str | Path) -> Path:
    ep = Path(ep)
    return ep.parent / (ep.name + ".reports")


def write_quality_report(ep: str | Path) -> tuple[Path, Path]:
    rep = quality_report(ep)
    d = reports_dir(ep)
    d.mkdir(exist_ok=True)
    j, md = d / "quality_report.json", d / "quality_report.md"
    write_json_atomic(j, rep)
    md.write_text(render_markdown(rep))
    return j, md


__all__ = ["quality_report", "render_markdown", "write_quality_report", "reports_dir"]
