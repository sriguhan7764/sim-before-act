"""First acceptance gate: one operator-recorded episode must be recorded, validated, replayed offline, exported into
both lightweight profiles, and checked for consistent units and frame associations.

This module is read-only with respect to the master. It creates only ``<episode>.exports/`` (via the normal export
code) and ``<episode>.reports/``. It imports no robot adapters and sends nothing to any hardware.
Thresholds are explicit constants below; a failure means "not accepted", never "fixed up".
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .exports import ExportBlocked, PROFILES, _Master, export_dir_for, export_episode, load_export
from .joints import LEROBOT_NORMALIZED_UNITS
from .preflight import freeze_config, load_registry, stale_hashes
from .quality import quality_report, render_markdown, reports_dir
from .replay import iter_frames
from .sync import depth_freshness
from .util import iter_jsonl, read_json, sha256_file, write_json_atomic
from .validate import validate

LIGHT = ("rgb_joint_angles", "rgbd_motor_positions")
MIN_STATE_MATCH_FRACTION = 0.95      # fraction of reference frames with a follower state inside tolerance
MIN_COMMAND_MATCH_FRACTION = 0.95    # fraction with a command (latest at/before the frame) inside tolerance
MAX_NORMALIZED_DIFF = 1e-3           # |collector mapping - LeRobot normalised|, degrees / gripper %
MAX_IMAGE_CHECKS = 60                # byte-comparison budget per export


CATEGORIES = ("collector_integrity", "physical_calibration", "manipulation_readiness")
STATUSES = ("PASS", "FAIL", "WARN", "NOT_CHECKED")


def _c(name: str, status: str, detail: str = "", cat: str = "collector_integrity") -> dict:
    assert status in STATUSES and cat in CATEGORIES
    return {"name": name, "status": status, "detail": detail, "category": cat}


def _frac(counts: dict, n: int) -> float:
    return counts.get("matched", 0) / n if n else 0.0


def _check_export(ep: Path, m: _Master, profile: str, out: Path) -> list[dict]:
    rgbd = profile == "rgbd_motor_positions"
    ex = load_export(out)
    s, arr, checks = ex["samples"], ex["arrays"], []
    p = f"[{profile}] "
    checks.append(_c(p + "sample count == master reference frames", "PASS" if len(s) == len(m.rows) else "FAIL",
                     f"{len(s)} vs {len(m.rows)}"))
    ref = m.sync_report.get("reference_camera")
    bad = []
    for i, (smp, row) in enumerate(zip(s, m.rows)):
        ce = smp["cameras"].get(ref, {})
        fr = m.frames[ref][row["frame_index"]]
        if not (ce.get("master_frame_index") == row["frame_index"] and ce.get("host_mono_ns") == fr["host_mono_ns"]
                and ce.get("frame_number") == fr.get("frame_number") and smp["t_host_mono_ns"] == row["t_host_mono_ns"]):
            bad.append(i)
    checks.append(_c(p + "source-frame association (index, host time, frame number) matches master", "PASS" if not bad else "FAIL",
                     f"{len(bad)} mismatching samples {bad[:5]}"))
    # images / depth byte-identical to master files
    step = max(1, len(s) // MAX_IMAGE_CHECKS)
    bad_img, n_img = [], 0
    for smp in s[::step]:
        for c, ce in smp["cameras"].items():
            for key, mkey in (("image_path", "color_file"), ("depth_native_path", "depth_native_file"),
                              ("depth_aligned_path", "depth_aligned_file")):
                if ce.get(key):
                    n_img += 1
                    src = m.frames[c][ce["master_frame_index"]][mkey]
                    if sha256_file(out / ce[key]) != sha256_file(m.ep / "streams" / c / src):
                        bad_img.append(ce[key])
                    if key.startswith("depth"):
                        a = cv2.imread(str(out / ce[key]), cv2.IMREAD_UNCHANGED)
                        if a is None or a.dtype != np.uint16:
                            bad_img.append(ce[key] + " (not uint16)")
    checks.append(_c(p + "images/depth byte-identical to master; depth uint16", "PASS" if not bad_img and n_img else "FAIL",
                     f"{n_img} files checked (every {step}th sample); bad: {bad_img[:3]}"))
    if rgbd:
        miss = [k for k in ("depth_scale", "color_intrinsics", "depth_intrinsics", "extrinsics_depth_to_color")
                if not all(ex["cameras"][c].get(k) for c in ex["cameras"] if m.cam_desc.get(c, {}).get("depth_scale") is not None)]
        checks.append(_c(p + "depth scale + camera calibration present", "PASS" if not miss else "FAIL", f"missing: {miss}"))
    valid = arr["state_valid"]
    want = np.array([r["follower_state"]["status"] == "matched" for r in m.rows])
    checks.append(_c(p + "state_valid equals master association", "PASS" if np.array_equal(valid, want) else "FAIL",
                     f"{int(valid.sum())}/{len(valid)} valid"))
    J = [j["name"] for j in m.joints]
    rev = [j["name"] for j in m.joints if j["type"] == "revolute"]
    bad_state = []
    for i, smp in enumerate(s):
        st = smp["state"]
        if st["association_status"] != "matched":
            continue
        raw = m.state[m.rows[i]["follower_state"]["index"]]["raw"]
        if st.get("master_state_index") != m.rows[i]["follower_state"]["index"]:
            bad_state.append(i)
        elif rgbd and list(arr["motor_positions_raw"][i]) != [raw[n] for n in J]:
            bad_state.append(i)
        elif m.mapping is not None and not np.allclose(arr["joint_angles_rad"][i], m.mapping.convert(raw)["joint_angles_rad"],
                                                       atol=1e-12):
            bad_state.append(i)
    checks.append(_c(p + "state values equal master raw (and recomputed angles)", "PASS" if not bad_state else "FAIL",
                     f"{len(bad_state)} mismatches {bad_state[:5]}"))
    if m.mapping is not None and not rgbd:
        g = arr["gripper_normalized_0_100"][valid]
        ok = bool(np.all((g >= 0) & (g <= 100))) if g.size else False
        checks.append(_c(p + "gripper in explicit units 0..100 (not radians)", "PASS" if ok else "FAIL",
                         f"range {g.min() if g.size else None}..{g.max() if g.size else None}"))
        sch = ex["schema"]["units"]
        checks.append(_c(p + "schema declares angle and gripper units", "PASS" if "radians" in sch["joint_angles_rad"]
                         and "NOT radians" in sch["gripper"] else "FAIL", ""))
    if rgbd:
        checks.append(_c(p + "schema declares raw motor units", "PASS" if "encoder ticks" in ex["schema"]["units"]["motor_positions_raw"]
                         else "FAIL", ""))
    # commands preserved with associations
    for w_ in ("prev", "next"):
        key = f"command_{w_}_valid"
        want = np.array([r[f"command_{w_}"]["status"] == "matched" for r in m.rows])
        if key not in arr:
            checks.append(_c(p + f"command_{w_} association preserved", "FAIL", "array missing"))
            continue
        checks.append(_c(p + f"command_{w_} association preserved", "PASS" if np.array_equal(arr[key], want) else "FAIL",
                         f"{int(want.sum())} matched"))
    if m.commands and any(c.get("units") == LEROBOT_NORMALIZED_UNITS for c in m.commands) and not rgbd:
        i = next((k for k, r in enumerate(m.rows) if r["command_prev"]["status"] == "matched"), None)
        if i is not None:
            c = m.commands[m.rows[i]["command_prev"]["index"]]
            exp = np.radians([float(c["values"][n]) for n in rev])
            checks.append(_c(p + "command targets converted deg->rad consistently", "PASS" if np.allclose(
                arr["command_prev_target_rad"][i], exp) else "FAIL", ""))
    return checks


def _camera_checks(ep: Path, meta: dict, quality_cfg: dict | None = None) -> list[dict]:
    """Requested vs negotiated profile, and separate native/aligned depth dimensions checked against the stored files."""
    out = []
    for c in sorted(meta.get("streams", {})):
        cp = ep / "streams" / c / "camera.json"
        cam = read_json(cp) if cp.exists() else {}
        neg = cam.get("negotiation")
        p = f"   camera {c}: "
        if not neg:
            out.append(_c(p + "requested vs negotiated profile", "NOT_CHECKED", "no negotiation record (episode predates this check)"))
        else:
            syn = " [SYNTHETIC negotiation]" if neg.get("synthetic") else ""
            out.append(_c(p + "negotiated profile == requested (no silent substitution)",
                          "PASS" if neg.get("matches_request") and not neg.get("silent_substitution") else "FAIL",
                          f"requested {neg.get('requested')} actual {neg.get('actual')} mismatches {neg.get('mismatches')}{syn}"))
        if not meta["streams"][c].get("depth"):
            continue
        fr = next(iter_jsonl(ep / "streams" / c / "frames.jsonl"), None)
        for kind in ("depth_native", "depth_aligned"):
            blk = cam.get(kind)
            if not blk or fr is None or not fr.get(f"{kind}_file"):
                out.append(_c(p + f"{kind} dimensions recorded separately", "FAIL", "description block or frame file missing"))
                continue
            a = cv2.imread(str(ep / "streams" / c / fr[f"{kind}_file"]), cv2.IMREAD_UNCHANGED)
            ok = a is not None and a.dtype == np.uint16 and (a.shape[1], a.shape[0]) == (blk.get("width"), blk.get("height"))
            out.append(_c(p + f"{kind} stored size == recorded description", "PASS" if ok else "FAIL",
                          f"described {blk.get('width')}x{blk.get('height')} {blk.get('format')}; file "
                          f"{None if a is None else (a.shape[1], a.shape[0])}"))
        df = depth_freshness(list(iter_jsonl(ep / "streams" / c / "frames.jsonl")))
        k = df["counts"]
        n = sum(k.values()) or 1
        st = "FAIL" if k["missing"] else "WARN" if k["repeated_previous_depth"] or df["depth_timestamp_non_monotonic"] else "PASS"
        out.append(_c(p + "depth frames fresh (repeats/missing flagged, never treated as fresh)", st,
                      f"{k}; repeated fraction {k['repeated_previous_depth'] / n:.3f}; depth frame-number skips "
                      f"{df['depth_frame_number_skipped']}; depth ts non-monotonic {df['depth_timestamp_non_monotonic']}; "
                      "repeats are reported and labelled per row; thresholds are the separate checks below"))
        out += _depth_threshold_checks(p, df, quality_cfg or {})
    return out


def _depth_threshold_checks(prefix: str, df: dict, qc: dict) -> list[dict]:
    """Configurable depth-age / repeat-rate checks. A null threshold is NOT_CHECKED (reported, never silently PASS).

    A configured threshold is an operator choice: PASS/FAIL against it says nothing about physical validity.
    """
    age = (df.get("age_ms_stats") or {}).get("max")
    measured = {"max_depth_repeat_fraction": df.get("repeat_fraction"), "max_depth_age_ms": age,
                "max_consecutive_repeated_depth": df.get("longest_repeat_run")}
    out = []
    for key, val in measured.items():
        thr = qc.get(key)
        name = f"{prefix}depth quality {key}"
        if thr is None:
            out.append(_c(name, "NOT_CHECKED", f"measured {val}; no threshold configured (quality_checks.{key}=null)"))
        elif val is None:
            out.append(_c(name, "NOT_CHECKED", f"threshold {thr} configured but nothing measurable (no depth frames)"))
        else:
            out.append(_c(name, "PASS" if val <= thr else "FAIL",
                          f"measured {val:.4g} vs configured limit {thr} (operator-chosen; not physically validated)"))
    return out


def _config_checks(ep: Path, meta: dict) -> list[dict]:
    """Frozen effective-config hash, and that the calibration files in use are the registry's ACTIVE ones."""
    out = []
    fp = ep / "config" / "effective_config.json"
    if not fp.exists():
        return [_c("   frozen effective config present", "NOT_CHECKED", "no config/effective_config.json (episode predates this check)")]
    fz = read_json(fp)
    h_snap = freeze_config(meta.get("config_snapshot") or {})[1]
    h_file = freeze_config(fz.get("config") or {})[1]
    ok = h_snap == h_file == fz.get("config_sha256") == meta.get("effective_config_sha256")
    out.append(_c("   frozen config hash: recomputed(config_snapshot) == file == episode.json", "PASS" if ok else "FAIL",
                  f"snapshot {h_snap[:12]} file {h_file[:12]} stored {str(fz.get('config_sha256'))[:12]} meta "
                  f"{str(meta.get('effective_config_sha256'))[:12]}"))
    stale = meta.get("stale_calibration_in_use") or []
    out.append(_c("   no stale calibration file in use", "PASS" if not stale else "FAIL", f"stale in use: {stale or 'none'}"))
    reg = (fz.get("config") or {}).get("calibration_registry")
    if reg and Path(reg).exists():
        sh = stale_hashes(reg)
        bad = [r for r, c in (fz.get("calibration_files") or {}).items() if c.get("sha256") in sh]
        reg_doc = load_registry(reg)
        act = {k: v.get("sha256") for k, v in (reg_doc.get("active") or {}).items()}
        wrong = [r for r, c in (fz.get("calibration_files") or {}).items() if act.get(r) and c.get("sha256") != act[r]]
        out.append(_c("   calibration files == registry ACTIVE files (by sha256)", "PASS" if not bad and not wrong else "FAIL",
                      f"stale: {bad or 'none'}; differs from active: {wrong or 'none'}"))
    else:
        out.append(_c("   calibration files == registry ACTIVE files (by sha256)", "NOT_CHECKED", "no calibration registry configured"))
    return out


def _protocol_check(meta: dict) -> dict:
    pr = meta.get("protocol")
    if not pr:
        return _c("   operator protocol cues recorded", "NOT_CHECKED", "episode was not recorded with --protocol")
    cues = pr.get("cues_issued") or []
    want = [s["label"] for s in (pr.get("nominal") or {}).get("segments", [])]
    return _c("   operator protocol cues recorded", "PASS" if cues and [c["label"] for c in cues] == want else "FAIL",
              f"{pr.get('name')}: issued {[c['label'] for c in cues]} of {want} (prompts only; they do not prove what the arm did)")


def _physical_checks(meta: dict) -> list[dict]:
    """Physical calibration. Nothing the collector records can validate these; they stay NOT_CHECKED until measured."""
    cat = "physical_calibration"
    nc = "not physically validated: needs operator-supervised reference poses/measurements (sim_before_act/docs/CALIBRATION_AUDIT.md)"
    out = [_c(f"   {n}", "NOT_CHECKED", nc, cat) for n in
           ("joint zero offsets (calibration-range midpoint != physical zero)", "joint rotation signs", "gripper mapping (0-100 vs opening)",
            "physical joint limits (encoder range != safe range)")]
    c2m = (meta.get("calibration") or {}).get("camera_to_mat")
    out.append(_c("   camera-to-mat calibration", "NOT_CHECKED" if not c2m else "PASS" if c2m.get("operator_accepted") else "FAIL",
                  "none recorded" if not c2m else f"{c2m.get('calibration_id')} operator_accepted={c2m.get('operator_accepted')}", cat))
    b2m = (meta.get("calibration") or {}).get("robot_base_to_mat")
    out.append(_c("   robot-base-to-mat alignment", "NOT_CHECKED" if not b2m else "WARN",
                  "no validated transform recorded" if not b2m else "present but independent validation is not performed here", cat))
    return out


def _manipulation_checks(meta: dict, rep: dict) -> list[dict]:
    cat = "manipulation_readiness"
    pil = meta.get("pilot")
    if not pil:
        return [_c("   pilot metadata", "NOT_CHECKED", "no pilot block", cat)]
    miss = rep["pose_status"]
    el = rep["eligibility_as_accepted_demonstration"]
    return [
        _c("   cube start pose measured", "PASS" if not miss["cube_start"] else "NOT_CHECKED", f"missing {miss['cube_start']}", cat),
        _c("   tray pose measured", "PASS" if not miss["tray"] else "NOT_CHECKED", f"missing {miss['tray']}", cat),
        _c("   outcome label with source", "PASS" if meta.get("outcome") in ("success", "failure") and pil.get("outcome_label_source")
           not in (None, "none") else "NOT_CHECKED", f"outcome={meta.get('outcome')} source={pil.get('outcome_label_source')}", cat),
        _c("   eligible as accepted demonstration", "PASS" if el["eligible"] else "NOT_CHECKED", "; ".join(el["reasons"]) or "ok", cat)]


def verdicts(checks: list[dict]) -> dict:
    out = {}
    for cat in CATEGORIES:
        cs = [c for c in checks if c["category"] == cat]
        n = {s: sum(1 for c in cs if c["status"] == s) for s in STATUSES}
        if cat == "collector_integrity":
            v = "FAIL" if n["FAIL"] else "PASS"
        elif cat == "physical_calibration":
            v = "FAIL" if n["FAIL"] else "NOT_VALIDATED" if n["NOT_CHECKED"] or n["WARN"] else "PASS"
        else:
            v = "NOT_READY" if n["FAIL"] or n["NOT_CHECKED"] or n["WARN"] else "READY"
        out[cat] = {"verdict": v, "counts": n}
    return out


def run_acceptance(ep: str | Path, force_exports: bool = False, quality_overrides: dict | None = None) -> dict:
    """quality_overrides: depth thresholds (see config ``quality_checks``) that replace the episode's frozen values."""
    ep = Path(ep)
    checks: list[dict] = []
    meta = read_json(ep / "episode.json")
    quality_cfg = {**((meta.get("config_snapshot") or {}).get("quality_checks") or {}),
                   **{k: v for k, v in (quality_overrides or {}).items() if v is not None}}
    q = meta.get("quality_flags", {})
    synthetic = bool(q.get("mock_data") or q.get("synthetic_test_fixture"))
    v = validate(ep)
    checks.append(_c("1. master integrity (hashes, monotonic timestamps, readable images)", "PASS" if v["ok"] else "FAIL",
                     "; ".join(v["errors"]) or "ok"))
    checks.append(_c("   recording status", "PASS" if meta.get("status") == "complete" else "FAIL",
                     f"status={meta.get('status')} (drops/aborts make an episode ineligible for the gate)"))
    rep = quality_report(ep)
    n = rep["association"]["reference_frames"]
    for k in ("follower_state", "command_prev"):
        f = _frac(rep["association"]["sources"][k]["status_counts"], n)
        thr = MIN_STATE_MATCH_FRACTION if k == "follower_state" else MIN_COMMAND_MATCH_FRACTION
        checks.append(_c(f"   association coverage {k}", "PASS" if f >= thr else "FAIL", f"{f:.3f} (need >= {thr})"))
    dups = {k: s["duplicate_timestamps"] for k, s in [*rep["cameras"].items(), *rep["telemetry"].items()]}
    checks.append(_c("   duplicate timestamps", "PASS" if not any(dups.values()) else "WARN", str(dups)))
    gaps = {k: s["seq_gaps"] for k, s in rep["telemetry"].items()}
    gaps.update({k: s["recorder_dropped_recv_gaps"] for k, s in rep["cameras"].items()})
    checks.append(_c("   dropped frames / sequence gaps", "PASS" if not any(gaps.values()) else "FAIL", str(gaps)))
    if rep["missing_modalities"]:
        checks.append(_c("   required signals present", "FAIL", f"missing: {rep['missing_modalities']}"))
    else:
        checks.append(_c("   required signals present", "PASS", "cameras, depth, follower, leader, commands, calibration"))
    pil = meta.get("pilot")
    checks.append(_c("   pilot metadata present (role, start configuration)", "PASS" if pil and pil.get("role")
                     and pil.get("robot_start_configuration") else "FAIL" if not synthetic else "WARN",
                     f"role={pil and pil.get('role')}; start config={'yes' if pil and pil.get('robot_start_configuration') else 'no'}"))
    if pil:
        miss = rep["pose_status"]
        checks.append(_c("   cube/tray poses measured", "PASS" if not miss["cube_start"] and not miss["tray"] else "WARN",
                         f"cube missing {miss['cube_start']}; tray missing {miss['tray']} (never auto-filled)"))
    cs = (rep.get("controller") or {}).get("published_session")
    hm = q.get("follower_calibration_hash_matches_controller")
    checks.append(_c("   calibration hash: recorder mapping == controller session", "PASS" if hm else
                     "NOT_CHECKED" if hm is None else "FAIL", f"{hm}; controller session {'recorded' if cs else 'absent'}"))
    nd = rep["units"].get("collector_mapping_vs_lerobot_normalized", {})
    if "max_abs_diff_arm_deg" in nd:
        okd = nd["max_abs_diff_arm_deg"] <= MAX_NORMALIZED_DIFF and nd["max_abs_diff_gripper_pct"] <= MAX_NORMALIZED_DIFF
        checks.append(_c("   collector joint mapping == LeRobot normalised values", "PASS" if okd else "FAIL", str(nd)))
    else:
        checks.append(_c("   collector joint mapping == LeRobot normalised values", "NOT_CHECKED", nd.get("note", "")))
    # offline replay (master)
    try:
        nr = sum(1 for _ in iter_frames(ep))
        checks.append(_c("3. offline replay of master (no hardware code imported)", "PASS" if nr == n else "FAIL", f"{nr} frames / {n} expected"))
    except Exception as e:
        checks.append(_c("3. offline replay of master", "FAIL", f"{type(e).__name__}: {e}"))
    m = _Master(ep)
    outs = {}
    for prof in LIGHT:
        try:
            d = export_dir_for(ep, prof)
            outs[prof] = d if (d.exists() and not force_exports) else export_episode(ep, prof, force=force_exports or d.exists())
            ev = validate(outs[prof])
            checks.append(_c(f"4. export {prof}: written + manifest verified", "PASS" if ev["ok"] else "FAIL",
                             "; ".join(ev["errors"]) or str(outs[prof])))
        except ExportBlocked as e:
            checks.append(_c(f"4. export {prof}", "FAIL", f"BLOCKED: {e}"))
        except Exception as e:
            checks.append(_c(f"4. export {prof}", "FAIL", f"{type(e).__name__}: {e}"))
    for prof, d in outs.items():
        try:
            nr = sum(1 for _ in iter_frames(d))
            checks.append(_c(f"3. offline replay of export {prof}", "PASS" if nr > 0 else "FAIL", f"{nr} samples"))
            checks += _check_export(ep, m, prof, d)
        except Exception as e:
            checks.append(_c(f"5. export checks {prof}", "FAIL", f"{type(e).__name__}: {e}"))
    if len(outs) == 2:
        a, b = (load_export(outs[p])["arrays"]["t_host_mono_ns"] for p in LIGHT)
        pa, pb = (load_export(outs[p])["manifest"]["parent"]["episode_id"] for p in LIGHT)
        checks.append(_c("5. both exports share episode identity and frame timeline", "PASS" if pa == pb == meta["episode_id"]
                         and np.array_equal(a, b) else "FAIL", f"parent {pa} / {pb}; timelines equal={np.array_equal(a, b)}"))
    checks += _camera_checks(ep, meta, quality_cfg) + _config_checks(ep, meta) + [_protocol_check(meta)]
    checks += _physical_checks(meta) + _manipulation_checks(meta, rep)
    vd = verdicts(checks)
    res = {"episode_id": meta["episode_id"], "overall": vd["collector_integrity"]["verdict"],
           "overall_meaning": "collector integrity only; physical calibration and manipulation readiness are reported separately",
           "verdicts": vd,
           "synthetic_or_mock": synthetic,
           "evidence_label": "SYNTHETIC/MOCK data: validates software only" if synthetic else "recorded by the collector (real backends)",
           "checks": checks, "thresholds": {"min_state_match_fraction": MIN_STATE_MATCH_FRACTION,
                                            "min_command_match_fraction": MIN_COMMAND_MATCH_FRACTION,
                                            "max_normalized_diff": MAX_NORMALIZED_DIFF},
           "not_covered": ["physical correctness of joint zero/sign/gripper mapping", "synchronisation accuracy beyond host timestamps",
                           "object poses (operator-entered)", "mat-to-robot-base alignment"]}
    d = reports_dir(ep)
    d.mkdir(exist_ok=True)
    write_json_atomic(d / "quality_report.json", rep)
    (d / "quality_report.md").write_text(render_markdown(rep))
    write_json_atomic(d / "acceptance.json", res)
    (d / "acceptance.md").write_text(render_acceptance(res))
    return res


def render_acceptance(res: dict) -> str:
    L = [f"# Acceptance gate: {res['episode_id']}", "", f"{res['evidence_label']}", "", "## Verdicts (three independent questions)", "",
         "| category | verdict | counts |", "|---|---|---|"]
    L += [f"| {k} | **{v['verdict']}** | {v['counts']} |" for k, v in res["verdicts"].items()]
    L += ["", "A collector-integrity PASS says nothing about physical calibration (signs, zero offsets, gripper mapping, base "
          "alignment) or about object poses/outcomes. NOT_CHECKED = nothing in the recording can establish it.", ""]
    for cat in res["verdicts"]:
        L += [f"## {cat}", "", "| status | check | detail |", "|---|---|---|"]
        L += [f"| {c['status']} | {c['name'].strip()} | {c['detail']} |" for c in res["checks"] if c["category"] == cat]
        L.append("")
    L += ["Not covered by this gate: " + "; ".join(res["not_covered"]), ""]
    return "\n".join(L)
