# Hardware smoke-test procedure (NOT YET RUN - no hardware was connected)

Run in order; stop and fix on any failure. Record results in `docs/STATUS.md`.

1. **Preview without motion.** `python -m collector inspect` (find serials/ports), fill `configs/setup.yaml`
   (`mode: diagnostic`), then `python -m collector preview --config configs/setup.yaml`. Confirm RGB (+depth view),
   camera/robot status "OK", stream rates, no queue/drops. The recorder issues no motion.
2. **Markers and dimensions.** Confirm dictionary, ids, orientations, that 40 mm is the outer black square, and that the
   four ids are detected in preview. Run `python -m collector calibrate ...`; check reprojection RMS, stability, and the
   camera position in the mat frame; accept only if sensible.
3. **Stationary episode (~10 s).** `python -m collector record --config configs/setup.yaml --task smoke_stationary --duration 10`
4. **Manually teleoperated episode (~15 s)** using your normal teleop process (with tap if it owns the port,
   see COEXISTENCE.md). Use the UI to label and finish.
5. **Validate.** `python -m collector validate --episode <ep>`. Check: no recv/device frame gaps (or understood),
   monotonic timestamps, units/raw ranges 0-4095 plausible, association residuals within tolerance, unmatched counts ~0,
   depth decodes as uint16 with expected scale (e.g. value*scale ~ distance to mat).
6. **Exports.** `python -m collector export --episode <ep> --profile rgb_joint_angles` and `--profile rgbd_motor_positions`
   (set `robot.follower.calibration_file` first, otherwise the angle export is blocked).
7. **Replay each export independently.** `python -m collector replay --episode <ep>.exports/rgb_joint_angles__v1`, same for rgbd.
8. **Master match.** Pick samples: export `state_seq` -> row in `telemetry/follower_state.jsonl`; raw values must be
   identical, images byte-identical (`validate` checks export hashes; compare a few against master manually). Re-run
   export with `--force` and confirm identical `content_sha256`; confirm `validate` on the master is unchanged.
9. **Throughput.** From `quality_report.json` / `sync_report.json` and the UI: achieved camera Hz, telemetry Hz, writer
   items/s, queue peak, drops, residual mean/p95/max. Approve main collection only if drops = 0 over a 60 s episode at
   the intended settings (or adjust resolution/fps/queue/disk), and residuals are within tolerance.
