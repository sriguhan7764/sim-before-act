"""python -m sim_before_act.shadow {annotate,list}   (offline; never launches or contacts hardware)."""
from __future__ import annotations

import argparse
import json
import sys

from .workflow import ShadowError, annotate_trial, load_trials


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_before_act.shadow", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate", help="append an operator annotation to a persisted prediction")
    a.add_argument("--root", required=True)
    a.add_argument("--trial", required=True)
    a.add_argument("--by", required=True, help="operator name")
    a.add_argument("--execution", required=True, choices=["executed", "not_executed", "unknown"])
    a.add_argument("--outcome", choices=["success", "failure", "aborted", "unknown"])
    a.add_argument("--horizon", choices=["full_task", "segment"], help="what the success/failure label refers to")
    a.add_argument("--label-source", choices=["operator_visual", "operator_measured"])
    a.add_argument("--abort-reason")
    a.add_argument("--executed-command-ref", help="JSON object, e.g. '{\"episode_id\": \"...\", \"action_sha256\": \"...\"}'")
    a.add_argument("--deviation", action="append", help="repeatable; pass --no-deviations to state there were none")
    a.add_argument("--no-deviations", action="store_true")
    a.add_argument("--replanning", choices=["yes", "no"], help="did replanning/temporal aggregation alter the executed commands?")
    a.add_argument("--notes", default="")
    ls = sub.add_parser("list", help="list trials with their matched annotation state")
    ls.add_argument("--root", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "annotate":
            devs = None if not (args.deviation or args.no_deviations) else list(args.deviation or [])
            ref = json.loads(args.executed_command_ref) if args.executed_command_ref else None
            repl = None if args.replanning is None else args.replanning == "yes"
            p = annotate_trial(args.root, args.trial, execution=args.execution, annotated_by=args.by, observed_outcome=args.outcome,
                               outcome_horizon=args.horizon, label_source=args.label_source, abort_reason=args.abort_reason,
                               executed_command_ref=ref, deviations_from_verified_action=devs,
                               replanning_or_temporal_aggregation=repl, notes=args.notes)
            print(p)
        else:
            for t in load_trials(args.root):
                print(f"{t['trial_id']}\t{t['data_origin']}\tpred={t['predicted_decision']}\texec={t['execution']}\t"
                      f"outcome={t['observed_outcome']}\texact={t['comparison_exact']}\t{t['excluded_reason'] or ''}")
    except (ShadowError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
