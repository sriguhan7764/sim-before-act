# Status, limitations and next steps

## Done (milestone 1, offline slice) - verified by running it

* Record schema, action contract, config validation.
* MuJoCo twin with full per-rollout reset, seeded documented perturbations, explicit outcome predicates, separate outcome categories.
* Estimator (simulated failure fraction + Wilson interval), verification statuses, fail-closed gate, approval binding + `GuardedExecutor`.
* Trace per attempt (including rejections and verification failures), exclusive read-only writes.
* CLI `verify` and `import-lerobot`; 3 synthetic records; read-only importer for the real LeRobot recording.
* 89 automated tests passed (`pytest -q`): unit, integration, fault/gate, simulation.
  Protected files (calibration JSONs, `run.sh`, `so101_mujoco/scene.xml`, `so101_new_calib.xml`) were re-hashed after the work and are unchanged.

## Milestones E and F (offline sprint) - verified by running it

| Item | Implemented | Tested (synthetic) | Tested on real recorded data | Physically validated |
|---|---|---|---|---|
| E: extended action contract (`contract_id`, units, reference, chunk duration, limits provenance; relative rejected) | yes | yes | no | no |
| E: action-segment loader/validator (hash tamper check, units/order/rep/hz, limits without clipping, provenance + untrained labels, replanning flag, horizon <= length) | yes | yes | no | no |
| E: `policy_to_sim` adapter (exact hash recorded; executed prefix) | yes | yes (invented fixtures only) | no | no |
| F: scene builder (MjSpec composition of the protected SO-101 model, mat, cube, tray), synthetic + physical modes | yes | yes | no | no |
| F: outcome predicates and separate outcome categories | yes | yes (constructed states + short rollouts) | no | no |
| F: headless rollout, perturbation recording, visuals (MuJoCo EGL contact sheet/MP4 + cv2 plot), CLI `scene-check`/`scene-rollout` | yes | yes | no | no |

**Blocked:** (1) the calibration mismatch (simulator limits/zero vs the user's follower calibration; physical mode refuses while
`calibration_blocker_resolved_by` is null); (2) measurements that do not exist yet: base-to-mat transform, cube pose/size, tray pose/dimensions with sigmas, calibration reference. The synthetic demo scene values are INVENTED, the CAD geometry is not a calibrated twin,
all outcome thresholds are assumed/uncalibrated, and no rollout of a full pick-and-place has been demonstrated (segment-horizon rollouts give `unknown`). See `SCENE.md` and `ACTION_CONTRACT.md`.
The integration test with the ACT-based producer is done (Milestone G): `integration/offline_pipeline.py`, `tests/integration/test_offline_pipeline.py`, `tests/unit/test_gate_evidence.py` (evidence/gate semantics), including a cross-environment ACT-smoke -> verification test.
Tests after the gate-semantics correction: `pytest -q -rs` in `sim_before_act` under `guhan_c` -> 181 passed, 0 skipped (see `robot_collector/docs/HANDOFF.md`).

## Milestones G, H, I, J (offline sprint) - verified by running it

| Item | Implemented | Tested (synthetic) | Tested on real recorded data | Physically validated |
|---|---|---|---|---|
| G: segment -> N seeded rollouts -> categorised counts, per-criterion evidence (safety / task), evidence-gated fail-closed offline decision (INSUFFICIENT_EVIDENCE -> WOULD_REJECT); budget N=1/5/10 | yes | yes | no | no |
| H: shadow predictions (immutable, hashed, timestamped), operator annotations, "not exact" flags, no executor | yes | yes | no | no |
| I: review report + dataset card from manifests; metrics withheld without both classes (>= 5 each) | yes | yes | no | no |
| J: one-command offline demo `python -m sim_before_act.demo --out NEW_DIR` | yes | yes (with a hardware-refusing shim) | no | no |

Everything in these rows is **synthetic**. Gate semantics (corrected after the sprint audit): a completed simulation is not a valid outcome
observation. Segment-horizon rollouts give `task_outcome=unknown`; with the default full scope (safety + task required) that is
`INSUFFICIENT_EVIDENCE` -> `REJECT` / `WOULD_REJECT`, risk withheld (never 0). Safety-only scope must be requested explicitly and is
labelled `approval_scope: safety_only`. The 0.3 threshold and 0.8 coverage are UNCALIBRATED, and offline `EXECUTE` is `WOULD_EXECUTE` only.

## Known limitations

* The MuJoCo scene is a generic SO-101 model, not validated against the physical robot, the user's calibration or controller. Sim fidelity unknown.
* Perturbation distributions and `max_state_age_s` are uncalibrated placeholders. The gate threshold 0.3 is a demo value, not selected on calibration data.
* Gripper 0..100 -> MJCF mapping is an unvalidated assumption (see `ACTION_CONTRACT.md`).
* Horizon is the action chunk + settle; no full-task success prediction. Task predicate is end-effector position only (no grasp/place predicate yet).
* The simulated failure fraction is not a calibrated probability; the Wilson interval is Monte-Carlo sampling error under the chosen rollout model only (no simulator mismatch; not used by the gate, which compares the point estimate with the threshold using a strict `<`; the rule is written into every trace).
* Contact predicates are sampled every 10 ms and may miss shorter transients.
* Velocities are not recorded in the LeRobot dataset; the simulator starts at rest.
* Offline: the rejection hold/stop behaviour of the physical controller is **not defined**; no execution adapter exists.
* Timing numbers are PRELIMINARY single informal measurements, not an E3 study (noted in each trace). No git repository, so `git_commit` is null in traces.
* The synthetic tests are engineering tests; they do not support any research hypothesis.

## Blockers for using the real recorded data

1. **Object poses**: the LeRobot recording has none; needs a documented method (e.g. ArUco/camera-to-robot transform; the mat camera calibration exists in `robot_collector` but there is no validated camera-to-robot-base transform).
2. **Outcome labels**: none recorded; do not invent them. Needs operator labelling per episode/chunk.
3. **Task target / predicate** for the chunk horizon.
4. **Model/robot agreement**: the first real frame is outside the model's joint limits (shoulder_lift about -103 deg vs +/-100 deg). The read-only audit (`CALIBRATION_AUDIT.md`) found no conversion bug: the value is consistent with the user's calibration (range +/-105.3 deg) and the MJCF limit is inherited from a CAD model. Zero offsets, signs and the physical stops still need measurement. See `DATA_READINESS.md`.
5. Gripper mapping and actuator response validation.

## Remaining plan (not started)

1. Measure calibration data: joint-state agreement, EE position/orientation error, tracking error, contact behaviour (needs authorised hardware time); derive tolerances and perturbation sigmas from it.
2. Sidecar + labelling workflow for the real recordings; collect action-state-outcome records incl. safely induced failures.
3. E1 (failure prediction metrics), E2 (budget study, paired prefixes already supported), E3 (timing with warm-up), E4 (association, episode-level CIs), baselines (constant prevalence, kinematic checks, nominal sim). Split by episode/session.
4. SmolVLA integration (check actual LeRobot 0.6.1 interfaces and action normalisation first), then shadow mode, then gated mode under an approved physical protocol.
5. Parquet trace tables, optional.
