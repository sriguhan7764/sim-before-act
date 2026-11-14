"""python -m sim_before_act.evaluation report --shadow-root DIR [--manifest M.json] [--split S.json] [--budget-rows R.json] --out DIR"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..shadow.workflow import load_trials
from .report import MIN_PER_CLASS, build_report, write_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_before_act.evaluation", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--shadow-root", help="directory of trial folders (omit when no trials exist)")
    r.add_argument("--manifest", help="dataset manifest JSON from `collector dataset manifest`")
    r.add_argument("--split", help="split manifest JSON")
    r.add_argument("--budget-rows", help="JSON list of {case_id,n_rollouts,...} rows for the rollout-budget comparison")
    r.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    load = lambda p: json.loads(Path(p).read_text()) if p else None  # noqa: E731
    try:
        trials = load_trials(a.shadow_root) if a.shadow_root else []
        rep = build_report(manifest=load(a.manifest), split=load(a.split),
                           physical_trials=[t for t in trials if t["physical"]],
                           synthetic_trials=[t for t in trials if not t["physical"]],
                           budget_rows=load(a.budget_rows), min_per_class=a.min_per_class)
    except (OSError, ValueError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    for k, v in write_report(a.out, rep, load(a.manifest), load(a.split)).items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
