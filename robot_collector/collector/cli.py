"""Command line interface: python -m collector <command>."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import calibration as calmod
from .config import ConfigError, load_config, markers_configured
from .exports import ExportBlocked, PROFILES, export_episode
from .hardware_gate import HardwareAccessDenied
from .pilot import LABEL_SOURCES, PROTOCOLS, ROLES, PilotError, build_pilot_meta, load_layout
from .util import canonical_json, read_json, write_json_atomic


def _pilot_from_args(a) -> dict | None:
    """Validate pilot metadata BEFORE any device is opened. Returns None for legacy (role-less) recordings."""
    if not a.role:
        return None
    layout = load_layout(a.layout) if a.layout else None
    return build_pilot_meta(a.role, layout, a.cube_position_id, attempt_index=a.attempt_index, notes=a.pilot_notes or "")


def _rec(cfg_path: str, taps_dir: str | None = None):
    from .recorder import Recorder

    cfg = load_config(cfg_path)
    if taps_dir:  # teleop process owns the buses; read its taps instead of opening ports
        r = cfg["robot"]
        r["follower"] = dict(r["follower"], backend="jsonl_tap", path=f"{taps_dir}/follower_state.jsonl")
        r["leader"] = dict(r["leader"], backend="jsonl_tap", path=f"{taps_dir}/leader_state.jsonl")
        r["commands"] = {"backend": "jsonl_tap", "path": f"{taps_dir}/commands.jsonl"}
    return cfg, Recorder(cfg, config_path=str(cfg_path))


def cmd_inspect(a) -> int:
    from .inspect_env import inspect_environment

    print(json.dumps(inspect_environment(), indent=2))
    return 0


def _banner(rec) -> None:
    if rec.is_mock:
        print("*" * 60 + "\n*** MOCK BACKEND IN USE: synthetic data, NOT hardware ***\n" + "*" * 60)


def cmd_preview(a) -> int:
    cfg, rec = _rec(a.config)
    rec.connect()
    rec.start_workers()
    _banner(rec)
    for n, s in rec.status.items():
        print(f"{n}: {'connected' if s['connected'] else 'NOT connected: ' + str(s['error'])}")
    if a.headless:
        end = time.time() + a.seconds
        while time.time() < end:
            time.sleep(1)
            print(json.dumps(rec.stats()["streams"]))
        rec.shutdown()
        return 0
    from .ui import OperatorUI

    OperatorUI(rec, allow_record=False).run()
    return 0


def cmd_calibrate(a) -> int:
    cfg, rec = _rec(a.config)
    missing = markers_configured(cfg)
    if missing:
        print("Cannot calibrate; unset (never guessed):", ", ".join(missing))
        return 2
    prim = next((c for c in cfg["cameras"] if c["enabled"] and c["role"] == "primary"), None)
    if prim is None:
        print("no enabled primary camera in config")
        return 2
    rec.connect()
    cam = rec.cameras[prim["name"]]
    if not rec.status[prim["name"]]["connected"]:
        print("camera not connected:", rec.status[prim["name"]]["error"])
        return 2
    _banner(rec)
    print("SETUP CHECK: the configured size (%.1f mm) must be the OUTER BLACK square, excluding the white paper margin."
          % cfg["markers"]["size_mm"])
    if input("Confirm this is true [y/N]: ").strip().lower() != "y":
        print("Aborted: confirm the physical marker size first.")
        return 2
    det = calmod.make_detector(cfg["markers"]["dictionary"])
    res, intr = calmod.run_calibration(cfg, cam, det, a.frames)
    rec.shutdown()
    print(json.dumps({k: v for k, v in res.items() if k != "T_camera_from_mat"}, indent=2))
    fails = calmod.evaluate_acceptance(res, cfg["calibration"])
    print("automatic checks:", "PASS" if not fails else "FAIL: " + "; ".join(fails))
    if res.get("ok"):
        print("T_camera_from_mat (p_cam = T @ p_mat):")
        for r in res["T_camera_from_mat"]:
            print("  ", [round(x, 5) for x in r])
        print("camera position in mat frame (m):", [round(x, 3) for x in res["camera_position_in_mat_m"]],
              "- compare with your physical setup (approx. 0.6 m high, right of the robot).")
    acc = input("Operator: accept this calibration? [y/N]: ").strip().lower() == "y"
    if acc and fails:
        if input("Automatic checks failed. Accept anyway (will be recorded as override)? [y/N]: ").lower() != "y":
            acc = False
    note = input("Note (optional): ")
    d = calmod.save_snapshot(cfg["paths"]["calibrations_root"], cfg, prim["name"], intr, res, fails, acc, note)
    print("saved immutable snapshot:", d, "| accepted:", acc)
    if cfg["markers"]["size_excludes_white_margin_confirmed"] is not True:
        print("Remember to set markers.size_excludes_white_margin_confirmed: true in the config.")
    return 0 if acc else 1


def cmd_robot_calibrate(a) -> int:
    from .config import load_config
    from .robot_calibration import capture_ranges

    cfg = load_config(a.config)
    path = capture_ranges(cfg["robot"][a.role], a.role, a.seconds, a.out_dir, a.name or f"{a.role}_calibration_guhan_c")
    print("saved (read-only file):", path)
    return 0


def cmd_teleop(a) -> int:
    from .teleop import run_teleop

    return run_teleop(load_config(a.config), a.hz, a.max_step, a.scale, a.margin, a.invert or [], a.taps_dir,
                      a.dry_run, a.on_exit, a.seconds)


def cmd_record(a) -> int:
    try:
        pilot = _pilot_from_args(a)
    except (PilotError, OSError) as e:
        print("pilot metadata error:", e)
        return 2
    if a.outcome in ("success", "failure") and not a.label_source:
        print("--outcome success/failure needs --label-source (operator_visual | operator_measured); refusing to label silently")
        return 2
    a.pilot = pilot
    if getattr(a, "protocol", None):
        a.headless = True
    cfg, rec = _rec(a.config, a.taps_dir)
    rec.connect()
    rec.start_workers()
    _banner(rec)
    for n, s in rec.status.items():
        print(f"{n}: {'connected' if s['connected'] else 'NOT connected: ' + str(s['error'])}")
    if a.headless or a.duration:
        return _headless_record(a, rec)
    from .ui import OperatorUI

    OperatorUI(rec, True, task=a.task, instruction=a.instruction or "", pilot=pilot).run()
    return 0


def _headless_record(a, rec) -> int:
    try:
        rec.start_episode(a.task, a.instruction or "", operator=a.operator or "", pilot=a.pilot)
    except Exception as e:
        print("cannot start:", e)
        rec.shutdown()
        return 2
    print("recording", rec.stats()["episode_id"])
    cues: list[dict] = []
    try:
        if getattr(a, "protocol", None):
            segs = PROTOCOLS[a.protocol]["segments"]
            t0 = time.monotonic()
            t0_ns = time.monotonic_ns()
            for sg in segs:
                while time.monotonic() - t0 < sg["start_s"] and not rec.should_abort():
                    time.sleep(0.05)
                cues.append({**sg, "cue_host_mono_ns": time.monotonic_ns(), "cue_offset_s": time.monotonic_ns() / 1e9 - t0_ns / 1e9})
                print(f">>> {sg['label'].upper()}: {sg['instruction']}")
            while time.monotonic() - t0 < segs[-1]["end_s"] and not rec.should_abort():
                time.sleep(0.05)
        elif a.duration:
            end = time.time() + a.duration
            while time.time() < end and not rec.should_abort():
                time.sleep(0.2)
        else:
            print("commands: f=finish success, x=finish failure, u=finish unlabelled, a=abort, n=note")
            note = ""
            for line in sys.stdin:
                c = line.strip()
                if c in ("f", "x", "u"):
                    break
                if c == "a":
                    print("aborted ->", rec.abort_episode(input("reason: ")))
                    rec.shutdown()
                    return 1
                if c == "n":
                    note = input("note: ")
            a.outcome = {"f": "success", "x": "failure", "u": "unlabelled"}.get(c, a.outcome)
            if a.outcome in ("success", "failure"):
                a.label_source = a.label_source or "operator_visual"  # the operator just typed the label
            a.notes = note
    except KeyboardInterrupt:
        pass
    reason = rec.should_abort()
    if cues and rec._writer is not None:
        rec._writer.update_meta(protocol={"name": a.protocol, "nominal": PROTOCOLS[a.protocol], "cues_issued": cues,
                                          "note": "cues are operator prompts on the host monotonic clock; they do not prove what the arm did"})
    if reason:
        p = rec.abort_episode(reason)
    else:
        try:
            p = rec.finish_episode(a.outcome, None, getattr(a, "notes", "") or "", a.label_source)
        except ValueError as e:
            print("cannot finish with that label:", e, "-> finishing as unlabelled (relabel later with `annotate`)")
            p = rec.finish_episode("unlabelled", None, getattr(a, "notes", "") or "")
    print("saved:", p, "| validate with: python -m collector validate --episode", p)
    rec.shutdown()
    return 0


def cmd_validate(a) -> int:
    from .validate import validate

    r = validate(a.episode)
    print(json.dumps(r, indent=2))
    return 0 if r["ok"] else 1


def cmd_replay(a) -> int:
    from .replay import replay

    n = replay(a.episode, display=not a.no_display, max_frames=a.max_frames)
    print(f"replayed {n} frames (offline; nothing sent to any robot)")
    return 0


def cmd_export(a) -> int:
    try:
        out = export_episode(a.episode, a.profile, allow_partial=a.allow_partial, force=a.force, link=a.link)
    except ExportBlocked as e:
        print("BLOCKED:", e)
        return 3
    print("export written:", out)
    return 0


def cmd_recover(a) -> int:
    from .recover import recover

    out = recover(a.episode)
    print("recovered as (flagged interrupted, partial):", out)
    return 0


def cmd_quality_report(a) -> int:
    from .quality import write_quality_report

    j, md = write_quality_report(a.episode)
    print("quality report written (master unchanged):", j, md)
    print(md.read_text())
    return 0


def cmd_preflight(a) -> int:
    from .preflight import open_cameras_check, run_preflight

    cfg = load_config(a.config)
    reg = cfg.get("calibration_registry")
    if not reg:
        print("config has no calibration_registry; refusing to certify calibration files")
        return 2
    res = run_preflight(cfg, reg)
    print(f"effective config sha256: {res['effective_config_sha256']}")
    bad = False
    for c in res["calibration_files"]:
        print(f"  {c['status']:<4} {c['name']}: {c['detail']}")
        bad |= c["status"] != "PASS"
    for n, p in res["cameras"].items():
        if "requested" in p:
            print(f"  camera {n}: requested {p['requested']}")
            print(f"    supported alone: colour={p['color_supported_alone']} depth={p['depth_supported_alone']}; "
                  f"combination resolvable={p.get('combination_resolvable')}; resolved (not started)={p.get('resolved_without_streaming')}")
            bad |= not p["supported"]
        else:
            print(f"  camera {n}: {json.dumps(p)}")
            bad |= not p.get("supported", False)
    if a.open_cameras:
        res["opened_streams"] = open_cameras_check(cfg, a.seconds)
        for n, r in res["opened_streams"].items():
            neg = r.get("negotiation") or {}
            print(f"  STREAM {n}: opened={r['opened']} error={r.get('error')}\n    requested={neg.get('requested')}\n    actual   ={neg.get('actual')}"
                  f"\n    matches_request={neg.get('matches_request')} frames={r.get('frames')} rate={r.get('measured_rate_hz')}"
                  f"\n    timing={r.get('timing_boundaries')}\n    depth_freshness={r.get('depth_freshness')}"
                  f"\n    shapes={r.get('frame_shapes_dtypes')} invalid depth native/aligned="
                  f"{r.get('invalid_depth_fraction_native_mean')}/{r.get('invalid_depth_fraction_aligned_mean')}")
            bad |= (not r["opened"]) or not neg.get("matches_request", False)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"preflight_{time.strftime('%Y%m%dT%H%M%S')}_{res['effective_config_sha256'][:8]}.json"
    write_json_atomic(f, res)
    print("saved:", f, "| overall:", "BLOCKED" if bad else "OK (no stream was started)" if not a.open_cameras else "OK")
    return 1 if bad else 0


def cmd_acceptance(a) -> int:
    from .acceptance import run_acceptance

    res = run_acceptance(a.episode, force_exports=a.force_exports, quality_overrides={
        "max_depth_repeat_fraction": a.max_depth_repeat_fraction, "max_depth_age_ms": a.max_depth_age_ms,
        "max_consecutive_repeated_depth": a.max_consecutive_repeated_depth})
    print(f"[{res['evidence_label']}]")
    for k, v in res["verdicts"].items():
        print(f"{k.upper():<24} {v['verdict']}  {v['counts']}")
    for cat in res["verdicts"]:
        print(f"\n== {cat} ==")
        for c in res["checks"]:
            if c["category"] == cat:
                print(f"  {c['status']:<11} {c['name'].strip()}  -- {c['detail']}")
    from .quality import reports_dir

    print("\nreports:", reports_dir(a.episode))
    return 0 if res["overall"] == "PASS" else 1


def cmd_dataset(a) -> int:
    from . import dataset as D
    from .preflight import stale_hashes

    c = a.dcmd
    if c == "manifest":
        root = Path(a.episodes_root)
        dirs = [root / n for n in a.episode] if a.episode else sorted(
            p for p in root.glob("ep_*") if p.is_dir() and not p.name.endswith((".exports", ".annotations", ".inprogress")))
        man = D.build_manifest(dirs, {"require_measured_poses": not a.allow_unmeasured_poses}, validate_hashes=not a.no_hash_check, name=a.name,
                               stale_calibrations=stale_hashes(a.calibration_registry) if a.calibration_registry else None)
        write_json_atomic(a.out, man)
        print(json.dumps(man["inventory"], indent=2))
        print("manifest written:", a.out)
        return 0
    man_path = Path(a.manifest)
    if c == "validate":
        man = read_json(man_path)
        problems = [f"{e['entry_id']}: {x}" for e in man["entries"] for x in D.validate_entry(e)]
        if D.manifest_hash(man) != man["content_sha256"]:
            problems.append("manifest content hash mismatch")
        ok = not problems and all(e["quality"]["validation_ok"] is not False for e in man["entries"] if e["source_type"] == "collector_episode")
        print(json.dumps({"ok": ok, "manifest_problems": problems,
                          "episodes_failing_master_validation": [e["entry_id"] for e in man["entries"]
                                                                 if e["source_type"] == "collector_episode" and e["quality"]["validation_ok"] is False]}, indent=2))
        return 0 if ok else 1
    if c in ("inventory", "quality"):
        man = read_json(man_path)
        print(json.dumps(D.inventory(man) if c == "inventory" else D.quality_summary(man), indent=2))
        return 0
    if c == "split":
        man = read_json(man_path)
        sp = D.make_split(man, a.seed)
        try:
            D.write_split(a.out, sp)
        except D.DatasetError as e:
            print("REFUSED:", e)
            return 3
        print(sp["status"], "- actual", sp["actual"], "plan", sp["plan"], "shortfall", sp["shortfall"])
        if a.apply_to:
            write_json_atomic(a.apply_to, D.apply_split(man, sp))
            print("manifest with split applied:", a.apply_to)
        return 0
    if c == "verify-split":
        problems = D.verify_split(read_json(a.split), read_json(man_path))
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
        return 0 if not problems else 1
    rd = D.DatasetReader(man_path)
    ep = rd.episode(a.entry)
    if c == "replay":
        from .replay import replay

        print(f"replayed {replay(ep.dir, display=False, max_frames=a.max_frames)} frames (offline; nothing sent to any robot)")
        return 0
    try:
        print("export written:", export_episode(ep.dir, a.profile))
    except ExportBlocked as e:
        print("BLOCKED:", e)
        return 3
    return 0


def cmd_annotate(a) -> int:
    ep = Path(a.episode)
    if a.outcome in ("success", "failure") and not a.label_source:
        print("success/failure labels need --label-source (operator_visual | operator_measured)")
        return 2
    if a.outcome in ("unknown", "unlabelled", "aborted") and a.label_source not in (None, "none"):
        print(f"outcome '{a.outcome}' cannot carry a label source")
        return 2
    d = ep.parent / (ep.name + ".annotations")
    d.mkdir(exist_ok=True)
    v = len(list(d.glob("v*.json"))) + 1
    ann = {"version": v, "parent_episode": read_json(ep / "episode.json")["episode_id"], "outcome": a.outcome,
           "failure_category": a.failure_category, "notes": a.notes, "label_source": a.label_source or "none",
           "accepted_for_dataset": {"yes": True, "no": False}.get(a.accepted_for_dataset),
           "accepted_reason": a.accepted_reason, "wall_ns": time.time_ns()}
    write_json_atomic(d / f"v{v:03d}.json", ann)
    print("annotation written (master unchanged):", d / f"v{v:03d}.json")
    from .pilot import eligibility

    if ann["accepted_for_dataset"]:
        el = eligibility(read_json(ep / "episode.json"), ann)
        if not el["eligible"]:
            print("WARNING: accepted_for_dataset=yes but metadata does not support it yet:", el["reasons"])
    return 0


def _authorise_hardware(a) -> int | None:
    """Hardware stays locked unless the operator passes --allow-hardware AND confirms. Returns an exit code to refuse, else None."""
    from . import hardware_gate

    if not a.allow_hardware:
        return None
    who = a.hardware_confirmed_by
    if not who:
        if not sys.stdin.isatty():
            print("--allow-hardware needs operator confirmation: run from a terminal, or pass --hardware-confirmed-by NAME "
                  "(recorded in the episode as the authorisation provenance). Nothing was opened.")
            return 4
        if input(f"Type '{hardware_gate.CONFIRM_PHRASE}' to allow this process to open cameras/serial ports: ").strip() != hardware_gate.CONFIRM_PHRASE:
            print("not confirmed; nothing was opened")
            return 4
        who = "interactive terminal confirmation"
    hardware_gate.authorise(who)
    print(f"[hardware access AUTHORISED for this process by: {who}]")
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="collector")
    p.add_argument("--allow-hardware", action="store_true",
                   help="permit this process to open real cameras / serial ports / enumerate devices (default: refuse; mock and offline tools never need it)")
    p.add_argument("--hardware-confirmed-by", default=None, metavar="NAME",
                   help="non-interactive confirmation: operator name recorded as the authorisation provenance")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    s = sub.add_parser("preview")
    s.add_argument("--config", required=True)
    s.add_argument("--headless", action="store_true")
    s.add_argument("--seconds", type=float, default=5)
    s.set_defaults(fn=cmd_preview)
    s = sub.add_parser("calibrate")
    s.add_argument("--config", required=True)
    s.add_argument("--frames", type=int)
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser("robot-calibrate", help="READ-ONLY range capture; operator moves joints by hand")
    s.add_argument("--config", required=True)
    s.add_argument("--role", required=True, choices=["follower", "leader"])
    s.add_argument("--seconds", type=float, default=30.0)
    s.add_argument("--out-dir", default="data/robot_calibration")
    s.add_argument("--name")
    s.set_defaults(fn=cmd_robot_calibrate)
    s = sub.add_parser("teleop", help="leader->follower teleop (WRITES follower goals/torque); publishes taps for the recorder")
    s.add_argument("--config", required=True)
    s.add_argument("--hz", type=float, default=100.0)
    s.add_argument("--max-step", type=float, default=12.0, help="max goal change per cycle, ticks")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--margin", type=int, default=100, help="ticks beyond calibrated range allowed")
    s.add_argument("--invert", nargs="*", help="joint names whose direction is opposite on the leader")
    s.add_argument("--taps-dir", default="/tmp/collector_taps")
    s.add_argument("--dry-run", action="store_true", help="read both arms, write nothing")
    s.add_argument("--on-exit", choices=["release", "hold"], default="release")
    s.add_argument("--seconds", type=float)
    s.set_defaults(fn=cmd_teleop)
    s = sub.add_parser("record")
    s.add_argument("--config", required=True)
    s.add_argument("--taps-dir", help="read follower/leader/commands from a running teleop's taps instead of opening ports")
    s.add_argument("--task", required=True)
    s.add_argument("--instruction")
    s.add_argument("--operator")
    s.add_argument("--headless", action="store_true")
    s.add_argument("--duration", type=float, help="record for N seconds then finish (implies headless)")
    s.add_argument("--outcome", default="unlabelled", choices=["success", "failure", "unlabelled"])
    s.add_argument("--label-source", choices=[x for x in LABEL_SOURCES if x != "none"],
                   help="who decided the outcome; required with --outcome success/failure")
    s.add_argument("--role", choices=ROLES, help="setup | demonstration | evaluation (enables structured pilot metadata)")
    s.add_argument("--cube-position-id", help="e.g. P1..P5 (required for demonstrations)")
    s.add_argument("--layout", default="configs/pilot_layout.yaml", help="measured layout file (null poses stay null)")
    s.add_argument("--attempt-index", type=int)
    s.add_argument("--pilot-notes")
    s.add_argument("--protocol", choices=sorted(PROTOCOLS), help="timed operator cues (stationary / small motion / stationary); "
                   "cue times are recorded in the episode; sets the duration")
    s.set_defaults(fn=cmd_record)
    s = sub.add_parser("validate")
    s.add_argument("--episode", required=True)
    s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("replay")
    s.add_argument("--episode", required=True)
    s.add_argument("--no-display", action="store_true")
    s.add_argument("--max-frames", type=int)
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("export")
    s.add_argument("--episode", required=True)
    s.add_argument("--profile", required=True, choices=PROFILES)
    s.add_argument("--allow-partial", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--link", action="store_true", help="full_master: hard-link instead of copy")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("recover")
    s.add_argument("--episode", required=True)
    s.set_defaults(fn=cmd_recover)
    s = sub.add_parser("annotate")
    s.add_argument("--episode", required=True)
    s.add_argument("--outcome", choices=["success", "failure", "aborted", "unknown", "unlabelled"], required=True)
    s.add_argument("--label-source", choices=[x for x in LABEL_SOURCES if x != "none"])
    s.add_argument("--accepted-for-dataset", choices=["yes", "no"], help="curator decision; recorded, never inferred")
    s.add_argument("--accepted-reason", default="")
    s.add_argument("--failure-category")
    s.add_argument("--notes", default="")
    s.set_defaults(fn=cmd_annotate)
    s = sub.add_parser("quality-report", help="rates, drops, duplicate/non-monotonic timestamps, residuals, units, clock assumptions")
    s.add_argument("--episode", required=True)
    s.set_defaults(fn=cmd_quality_report)
    s = sub.add_parser("preflight", help="no-motion pre-recording checks: frozen config hash, active/stale calibration, camera profile support")
    s.add_argument("--config", required=True)
    s.add_argument("--open-cameras", action="store_true",
                   help="ALSO start the camera streams for --seconds to read the NEGOTIATED profile (camera access; no motion)")
    s.add_argument("--seconds", type=float, default=2.0)
    s.add_argument("--out-dir", default="data/preflight")
    s.set_defaults(fn=cmd_preflight)
    s = sub.add_parser("acceptance", help="first acceptance gate: validate, replay offline, export both profiles, consistency checks")
    s.add_argument("--episode", required=True)
    s.add_argument("--force-exports", action="store_true", help="regenerate existing exports (master untouched)")
    s.add_argument("--max-depth-repeat-fraction", type=float, default=None, help="operator-chosen limit; default: not configured")
    s.add_argument("--max-depth-age-ms", type=float, default=None, help="operator-chosen limit on host-receipt depth age")
    s.add_argument("--max-consecutive-repeated-depth", type=int, default=None, help="operator-chosen limit on repeat runs")
    s.set_defaults(fn=cmd_acceptance)
    s = sub.add_parser("dataset", help="dataset v0.1 tools (offline; read-only over masters): manifest, validate, inventory, split, replay, export")
    ds = s.add_subparsers(dest="dcmd", required=True)
    d = ds.add_parser("manifest", help="build a v0.1 manifest from episode directories")
    d.add_argument("--episodes-root", required=True)
    d.add_argument("--episode", action="append", help="episode directory name (repeatable); default: every finished ep_* directory")
    d.add_argument("--out", required=True)
    d.add_argument("--name", default="dataset")
    d.add_argument("--no-hash-check", action="store_true", help="skip per-file hash validation (entries then say validation was not checked)")
    d.add_argument("--allow-unmeasured-poses", action="store_true", help="policy: do not exclude demonstrations whose cube/tray pose is null")
    d.add_argument("--calibration-registry", help="configs/calibration_registry.yaml: episodes recorded under a stale calibration are excluded")
    d.set_defaults(fn=cmd_dataset)
    for nm, hp in (("validate", "structural + hash validation of a manifest and its episodes"),
                   ("inventory", "counts by role/origin/outcome/split and shortfall vs the 25/5 plan"),
                   ("quality", "per-episode quality summary")):
        d = ds.add_parser(nm, help=hp)
        d.add_argument("--manifest", required=True)
        d.set_defaults(fn=cmd_dataset)
    d = ds.add_parser("split", help="derive whole-episode train/val split (FROZEN only if the plan is met)")
    d.add_argument("--manifest", required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--apply-to", help="also write a manifest copy with the split filled in")
    d.set_defaults(fn=cmd_dataset)
    d = ds.add_parser("verify-split", help="check a split file against a manifest")
    d.add_argument("--manifest", required=True)
    d.add_argument("--split", required=True)
    d.set_defaults(fn=cmd_dataset)
    for nm in ("replay", "export"):
        d = ds.add_parser(nm, help=f"{nm} one manifest entry offline")
        d.add_argument("--manifest", required=True)
        d.add_argument("--entry", required=True)
        if nm == "replay":
            d.add_argument("--max-frames", type=int, default=None)
        else:
            d.add_argument("--profile", choices=list(PROFILES), required=True)
        d.set_defaults(fn=cmd_dataset)
    a = p.parse_args(argv)
    refused = _authorise_hardware(a)
    if refused is not None:
        return refused
    try:
        return a.fn(a)
    except ConfigError as e:
        print("config error:", e)
        return 2
    except HardwareAccessDenied as e:
        print("HARDWARE ACCESS DENIED:", e)
        return 5
