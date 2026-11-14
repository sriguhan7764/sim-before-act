"""``python -m collector.act <command>`` (run from the robot_collector directory).

Commands that need torch (train, resume, freeze, infer, evaluate, smoke) must run in the lerobot env with
``PYTHONPATH=/home/apic3/lerobot_v6_2/lerobot/src:<robot_collector>``. fixture / convert / verify-lineage / validate-segment are numpy-only.
None of these commands touches a robot, camera or serial port.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _p() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m collector.act", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fixture", help="write the SYNTHETIC fixture (masters + manifest + provisional split)")
    s.add_argument("--out", required=True)

    s = sub.add_parser("convert", help="dataset v0.1 manifest + split -> converted ACT dataset + audit")
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--config", help="yaml with a `convert:` section (see configs/act_convert.yaml)")
    s.add_argument("--root", help="directory holding the episodes (default: the manifest's directory)")
    s.add_argument("--allow-provisional-split", action="store_true", help="SMOKE ONLY; recorded in the output")

    s = sub.add_parser("verify-lineage", help="re-read the source episodes and confirm every converted row")
    s.add_argument("--converted", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--root")

    for name in ("train", "resume"):
        s = sub.add_parser(name, help=f"{name} ACT on the TRAIN split (val split selects the checkpoint)")
        s.add_argument("--config", required=True)
        s.add_argument("--converted", required=True)
        s.add_argument("--run", required=True)
        s.add_argument("--stop-after", type=int, help="stop after this many steps (resume later)")

    s = sub.add_parser("freeze", help="freeze a checkpoint + preprocessing + config with hashes")
    s.add_argument("--run", required=True)
    s.add_argument("--converted", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--which", default="best", choices=("best", "last"))

    s = sub.add_parser("infer", help="frozen checkpoint + state(+images) -> action segment JSON")
    s.add_argument("--frozen", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--converted", help="read the state/images of a converted row")
    s.add_argument("--split", default="val")
    s.add_argument("--row", type=int, default=0)
    s.add_argument("--state", type=float, nargs=6, help="6 values: degrees x5, gripper 0..100 (measured follower state)")
    s.add_argument("--image", action="append", default=[], metavar="CAM=PATH.png", help="RGB image at the training size")

    s = sub.add_parser("evaluate", help="OFFLINE val loss / prediction error vs recorded commands (not task success)")
    s.add_argument("--frozen", required=True)
    s.add_argument("--converted", required=True)
    s.add_argument("--split", default="val")
    s.add_argument("--out")

    s = sub.add_parser("smoke", help="bounded SYNTHETIC smoke test of the whole chain (CPU, lerobot env)")
    s.add_argument("--out", required=True)
    s.add_argument("--config")
    s.add_argument("--stop-after", type=int, default=3)

    s = sub.add_parser("validate-segment", help="validate an action-segment-v0.1 JSON (numpy only)")
    s.add_argument("path")
    return ap


def main(argv=None) -> int:
    a = _p().parse_args(argv)
    if a.cmd == "fixture":
        from .synthetic import build_fixture

        print(json.dumps(build_fixture(a.out), indent=1))
    elif a.cmd == "convert":
        from .convert import ConvertConfig, convert_dataset

        cfg = ConvertConfig.from_yaml(a.config) if a.config else ConvertConfig()
        conv = convert_dataset(a.manifest, a.split, a.out, cfg, allow_provisional_split=a.allow_provisional_split, root=a.root)
        print(json.dumps({"out": a.out, "conversion_sha256": conv["conversion_sha256"], "counts": conv["counts"],
                          "data_origin": conv["data_origin"], "labels": conv["labels"]}, indent=1))
    elif a.cmd == "verify-lineage":
        from .convert import verify_lineage

        r = verify_lineage(a.converted, a.manifest, root=a.root)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    elif a.cmd in ("train", "resume"):
        from .train import TrainConfig, train

        r = train(TrainConfig.from_yaml(a.config), a.converted, a.run, resume=a.cmd == "resume", stop_after_steps=a.stop_after)
        print(json.dumps(r, indent=1))
    elif a.cmd == "freeze":
        from .train import freeze

        d = freeze(a.run, a.converted, a.out, a.which)
        print(json.dumps({k: d[k] for k in ("checkpoint_id", "kind", "trained_for_task", "checkpoint_sha256", "preprocessing_sha256",
                                            "config_sha256", "label")}, indent=1))
    elif a.cmd == "infer":
        from .infer import FrozenPolicy, infer_from_converted, write_segment

        if a.converted:
            seg = infer_from_converted(a.frozen, a.converted, a.out, a.split, a.row)
        else:
            if not a.state or not a.image:
                sys.exit("infer needs --converted, or --state with --image CAM=PATH for every camera")
            import cv2

            imgs = {}
            for spec in a.image:
                cam, _, path = spec.partition("=")
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is None:
                    sys.exit(f"cannot read image {path}")
                imgs[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            seg = write_segment(FrozenPolicy(a.frozen), a.out, a.state, imgs)
        print(json.dumps({"segment_id": seg["segment_id"], "out": a.out, "policy": seg["policy"], "provenance_label": seg["provenance_label"]},
                         indent=1))
    elif a.cmd == "evaluate":
        from .infer import evaluate_frozen

        print(json.dumps(evaluate_frozen(a.frozen, a.converted, a.split, a.out), indent=1))
    elif a.cmd == "smoke":
        from .smoke import DEFAULT_CONFIG, run_smoke

        r = run_smoke(a.out, a.config or DEFAULT_CONFIG, a.stop_after)
        print(json.dumps({k: r[k] for k in ("label", "kind", "trained_for_task", "resume_matches_straight_final_weights", "segment_problems",
                                            "frozen_verify_problems", "device_used", "seconds")}, indent=1))
    elif a.cmd == "validate-segment":
        from .segment import validate_action_segment_dict

        probs = validate_action_segment_dict(json.loads(Path(a.path).read_text()))
        print(json.dumps({"valid": not probs, "problems": probs}, indent=1))
        return 0 if not probs else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
