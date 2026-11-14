# Offline demo report

> **SYNTHETIC OFFLINE DEMONSTRATION - invented fixture data and a random-init smoke-test policy. Not recorded physical data, not physically evaluated, not a thesis finding. No hardware was opened.**

- Data origin: **synthetic** (fixture episodes written by `collector.act.synthetic`; invented values).
- Scene: `SYNTHETIC DEMO SCENE (invented values, CAD geometry; not a calibrated twin)`.

## A. ACT smoke integration (honest integration test)

- Policy: ACT smoke test (random-init, few CPU steps, trained_for_task=false). `trained_for_task` is **false**; this is not a trained policy and says nothing about ACT performance.
- Gate status **INSUFFICIENT_EVIDENCE**, decision **REJECT** (offline label WOULD_REJECT, approval scope `none`). Reasons: ['task_outcome_unavailable', 'status_INSUFFICIENT_EVIDENCE']. Nothing was executed or sent; threshold 0.3 and coverage requirement 0.8 are UNCALIBRATED.
- Rollouts: N=10, base seed 0; counts {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 10, 'timeout': 0, 'invalid_rollout': 0}.
- Evidence: safety evaluable 10/10 (0 violating); task evaluable 0/10 (0 success, 0 failure, 10 unknown); determined for required ['safety', 'task']: 0/10; need >= 8.
- Denominators: {'all_rollouts': 10, 'valid': 10, 'safety_evaluable': 10, 'task_evaluable': 0, 'determined': 0, 'used_for_fraction': 'none (fraction withheld)', 'excluded_from_fraction': {'invalid': 0, 'timeout': 0, 'undetermined_valid': 10}}. Risk estimate: None (None = withheld, not zero). Interval: None.
- Horizon: segment-horizon safety/constraint observations only; NOT a full-task success prediction
- A completed simulation is not a valid outcome observation. Safety was evaluable for every rollout; the task outcome was not evaluable because the rollout is shorter than the assumed task horizon. Per-rollout reasons: `04_verification/outcome_audit.json`.

## B. Deterministic SYNTHETIC verification fixtures (NOT the ACT policy)

> SYNTHETIC deterministic verification fixtures (invented actions and an invented task horizon). Not ACT output, not recorded data, not physically evaluated. A case meeting its expectation shows the GATE LOGIC, nothing else.

All expectations met: **True**

| case | scene | expected status/decision/scope | observed status/decision/scope | counts | fraction | met |
|---|---|---|---|---|---|---|
| B1_hold_pose_safety_only_scope | default (task horizon 8.0 s) | COMPLETED / WOULD_EXECUTE / safety_only | COMPLETED / WOULD_EXECUTE / safety_only | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 10, 'timeout': 0, 'invalid_rollout': 0} | 0.0 | True |
| B2_hold_pose_task_horizon_reached | synthetic variant: simulation.task_horizon_s=1.5 (invented) so the 1.83 s rollout reaches the task horizon | COMPLETED / REJECT / none | COMPLETED / REJECT / none | {'task_success': 0, 'task_failure': 10, 'safety_violation': 0, 'unknown': 0, 'timeout': 0, 'invalid_rollout': 0} | 1.0 | True |
| B3_known_constraint_violation | default | COMPLETED / REJECT / none | COMPLETED / REJECT / none | {'task_success': 0, 'task_failure': 0, 'safety_violation': 10, 'unknown': 0, 'timeout': 0, 'invalid_rollout': 0} | 1.0 | True |
| B4_insufficient_evidence | default (task horizon 8.0 s) | INSUFFICIENT_EVIDENCE / REJECT / none | INSUFFICIENT_EVIDENCE / REJECT / none | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 10, 'timeout': 0, 'invalid_rollout': 0} | None | True |
| B5_all_rollouts_invalid | default | INSUFFICIENT_VALID_ROLLOUTS / REJECT / none | INSUFFICIENT_VALID_ROLLOUTS / REJECT / none | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 0, 'timeout': 0, 'invalid_rollout': 10} | None | True |

Why each fixture is expected to behave so:

- **B1_hold_pose_safety_only_scope**: arm holds its start pose: no joint-limit or unsafe-contact violation, so safety is evaluable and clean. Task is NOT evaluated: this is an explicitly safety-only scope, not a full-action approval
- **B2_hold_pose_task_horizon_reached**: arm holds its pose; the cube is never touched, so lift/inside_tray/release_settle are evaluable and false at the task horizon: task_failure, with no safety violation
- **B3_known_constraint_violation**: invented actions drive the arm into the mat: constraint/contact violation is a known safety outcome even though the task outcome stays unknown
- **B4_insufficient_evidence**: same safe hold, full (safety+task) scope: safety is clean but the 1.83 s rollout is below the 8.0 s task horizon, so no task outcome exists for ANY rollout
- **B5_all_rollouts_invalid**: the segment's start state is outside the simulator joint limits: every rollout is invalid and refused, never clipped

## Steps

- **fixture** - OK: 4 SYNTHETIC fixture episodes; roles=['synthetic_fixture']; split data_origin=synthetic_fixture frozen=False
- **collector_validate_replay_export** - OK: validate, inventory, offline replay and both export profiles on a SYNTHETIC fixture episode
- **policy_segment** - OK: ACT smoke test (random-init, few CPU steps, trained_for_task=false)
- **A_act_smoke_verification** - OK: integration ran (policy: ACT smoke test (random-init, few CPU steps, trained_for_task=false)). Gate status=INSUFFICIENT_EVIDENCE decision=REJECT (WOULD_REJECT, scope=none); safety evaluable 10/10 (0 violating); task evaluable 0/10 (0 success, 0 failure, 10 unknown); determined for required ['safety', 'task']: 0/10; need >= 8; reasons=['task_outcome_unavailable', 'status_INSUFFICIENT_EVIDENCE']. A rejection or insufficient-evidence result here is an honest integration outcome, not a step failure
- **budget_study** - OK: N=1/5/10, same cases and base seed, seeds nested
- **B_synthetic_verification_fixtures** - OK: B1_hold_pose_safety_only_scope: COMPLETED/WOULD_EXECUTE; B2_hold_pose_task_horizon_reached: COMPLETED/REJECT; B3_known_constraint_violation: COMPLETED/REJECT; B4_insufficient_evidence: INSUFFICIENT_EVIDENCE/REJECT; B5_all_rollouts_invalid: INSUFFICIENT_VALID_ROLLOUTS/REJECT. SYNTHETIC fixtures, not ACT output
- **shadow_record** - OK: synthetic prediction persisted + 'not_executed' annotation; no outcome exists
- **report** - OK: review report + dataset card written (/home/apic3/guhan_c/sim_before_act/outputs/offline_demo_gate_fix_v1/07_report/review_report.md)

## Rollout-budget comparison (SYNTHETIC cases, default full scope)

| case | N | failure fraction | status | decision | counts |
|---|---|---|---|---|---|
| fixture_safe | 1 | None | INSUFFICIENT_EVIDENCE | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 1, 'timeout': 0, 'invalid_rollout': 0} |
| fixture_safe | 5 | None | INSUFFICIENT_EVIDENCE | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 5, 'timeout': 0, 'invalid_rollout': 0} |
| fixture_safe | 10 | None | INSUFFICIENT_EVIDENCE | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 10, 'timeout': 0, 'invalid_rollout': 0} |
| fixture_unsafe_invented | 1 | 1.0 | COMPLETED | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 1, 'unknown': 0, 'timeout': 0, 'invalid_rollout': 0} |
| fixture_unsafe_invented | 5 | 1.0 | COMPLETED | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 5, 'unknown': 0, 'timeout': 0, 'invalid_rollout': 0} |
| fixture_unsafe_invented | 10 | 1.0 | COMPLETED | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 10, 'unknown': 0, 'timeout': 0, 'invalid_rollout': 0} |
| policy_segment | 1 | None | INSUFFICIENT_EVIDENCE | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 1, 'timeout': 0, 'invalid_rollout': 0} |
| policy_segment | 5 | None | INSUFFICIENT_EVIDENCE | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 5, 'timeout': 0, 'invalid_rollout': 0} |
| policy_segment | 10 | None | INSUFFICIENT_EVIDENCE | REJECT | {'task_success': 0, 'task_failure': 0, 'safety_violation': 0, 'unknown': 10, 'timeout': 0, 'invalid_rollout': 0} |

## Not claimed

- No recorded physical demonstration was used; no physical trial exists; nothing was physically evaluated.
- The simulator is CAD geometry with invented scene values, not a physically calibrated twin; the calibration mismatch remains a blocker.
- A random-init ACT smoke checkpoint says nothing about ACT performance. ACT is a pilot policy, not a language-conditioned VLA.
- Fixture cases demonstrate gate logic only. No fixture result is ACT task success or a prediction of physical outcome.
- `unknown` is never counted as success or zero risk; WOULD_EXECUTE needs evaluable results for every required criterion.

## Outputs

- `01_fixture_episodes/` SYNTHETIC masters, manifest, split (fixture split, never frozen)
- `04_verification/` Demo A: verification.json, outcome_audit.json, per-rollout traces, plot, contact sheet when a GL backend worked
- `05_budget_rows.json`, `05b_synthetic_verification_fixtures/` (Demo B), `06_shadow/`, `07_report/`, `logs/`
