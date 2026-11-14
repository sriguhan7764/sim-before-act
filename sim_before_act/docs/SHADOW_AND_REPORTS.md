# Shadow evaluation and review reports (Milestones H, I) — offline software only

Status: **Implemented; tested with synthetic logic fixtures only.** No physical trial exists. Nothing here opens hardware.

## Shadow workflow (`sim_before_act.shadow`)
1. `record_prediction(root, ...)` persists `<root>/<trial_id>/prediction.json` **once** (exclusive, read-only). The wall-clock
   prediction timestamp is taken inside the function. The record binds: trial id, initial state (values, hash, timestamp, clock
   domain), action/segment id + hash, policy checkpoint identity, execution horizon, verification config hash and result
   (status, decision, N, seed, failure fraction, interval, outcome-category counts, threshold + calibrated flag), timings, hardware.
2. `annotate_trial` appends operator annotations `annotations/vNNN.json` (never rewrites the prediction): execution
   occurred / not / unknown; observed outcome (success, failure, aborted, unknown); abort reason; label source; **outcome horizon**
   (`full_task` or `segment`); executed-command reference; deviations; replanning/temporal-aggregation flag.
3. `match_trial` joins the prediction with the latest annotation.

Rules enforced (tests in `tests/unit/test_shadow.py`):
- Offline decision vocabulary is `WOULD_EXECUTE` / `REJECT` (offline label `WOULD_REJECT`, includes insufficient evidence); `EXECUTE` is refused.
  The verification block also carries `approval_scope` (`full` / `safety_only` / `none`), `required_criteria`, `denominators` and `coverage`.
- A **full-task outcome never labels a segment prediction**; only an operator-supplied `segment` label does.
- Replanning/temporal aggregation, deviations, a different executed-command hash, or any of these *not assessed* ⇒
  `comparison_exact = false` with reasons.
- Aborted / unknown / not-executed / unannotated trials are excluded from the comparison and listed with the reason.
- A `physical_trial` prediction requires `trained_for_task = true`; synthetic trials must carry a `SYNTHETIC…` label.
- There is no executor argument and no hardware import anywhere in the package (subprocess import check).

CLI: `python -m sim_before_act.shadow annotate|list`. Predictions are produced by the integration path (Milestone G).

## Reports (`sim_before_act.evaluation`)
`python -m sim_before_act.evaluation report --shadow-root DIR --manifest M.json [--split S.json] [--budget-rows R.json] --out OUT`
writes `review_report.json/.md` and `dataset_card.md`. Dataset counts come from the manifest (not the plan).
- Physical and synthetic trials are reported in separate sections and never pooled (enforced).
- No trials ⇒ explicit "NO physical shadow trials" (no success rate, no comparison).
- Prediction-vs-outcome rates and ROC-AUC are computed only with ≥ `MIN_PER_CLASS` (5) observed segment-labelled successes **and**
  failures with numeric scores; otherwise `withheld` states why. Only-successes / only-failures / small N are covered by tests.
- Thresholds are read from persisted predictions and reported with their calibrated flag; the module cannot tune them.
- Latency summaries use recorded `timings.inference_s` / `timings.verification_s` and say when absent.
- Rollout-budget comparison reports per-case decisions and interval widths for N ∈ {1, 5, 10}, with caveats.
- `MIN_PER_CLASS` is a software guard against unstable estimates, not a statistically derived power threshold.
