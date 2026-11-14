# Calibration guide

## Frames and naming
- **Mat frame** (metres): origin = midpoint of the mat edge nearest the robot; +X right (seen from the robot);
  +Y away from the robot; +Z up; mat spans X in [-0.2, 0.2], Y in [0, 0.4]. Marker centres (40 mm squares, flush with
  the mat corners): NL (-0.18, 0.02), NR (+0.18, 0.02), FL (-0.18, 0.38), FR (+0.18, 0.38), Z = 0.
- **`T_camera_from_mat`** (4x4) maps mat-frame points into camera coordinates: `p_cam = T @ p_mat`
  (OpenCV camera: +X right, +Y down, +Z forward). `inv(T)[:3,3]` is the camera position in the mat frame.
- **Robot-base-to-mat** is a *separate* calibration. It is stored as `null`; it is not inferred from the arm being
  "centred", and the ~600 mm camera height is never used as a pose.

## Before calibrating
1. Print markers from one known ArUco dictionary. Put its OpenCV name (e.g. `DICT_4X4_50`) in
   `markers.dictionary` - the collector will not guess it.
2. Set `markers.corners.<near_left|near_right|far_left|far_right>.id` to the printed ids.
3. Set `orientation_deg` per marker: the counter-clockwise rotation (viewed from above) of the marker's printed
   "top" edge away from +Y. 0 = top edge points to +Y (away from robot); 90 = top points to -X (left); 180 = to the
   robot; 270 = to +X. Corner order used is OpenCV's TL, TR, BR, BL of the printed marker. A wrong value shows up as a
   large reprojection error (tested).
4. **Setup check:** `size_mm` (40) must be the *outer black square*, excluding the white paper margin. `calibrate`
   asks you to confirm, and research mode requires `markers.size_excludes_white_margin_confirmed: true`.
5. Confirm all four markers are fully visible and flat on the mat; use `preview` to see detected ids (green = expected
   id, orange = unexpected).

## Run
`python -m collector calibrate --config configs/setup.yaml [--frames 30]`

It grabs N frames, detects markers, and solves one joint PnP over **all detected marker corners of all frames** (camera
and mat are static), using the camera's own colour intrinsics/distortion. It reports:
- reprojection RMS/max px (overall and per marker), marker ids used/missing;
- pose stability across frames that saw all markers: translation std (mm), rotation std (deg);
- the camera position in the mat frame - sanity-check it against the physical setup (right of robot, roughly 0.6 m up).

Thresholds: `calibration.*` in config. You then explicitly accept/reject (`y/N`; overriding failed automatic checks is
recorded). The result is an **immutable snapshot** `data/calibrations/cal_<UTC>_<hash>/calibration.json` (+ `.sha256`,
read-only). New calibrations create new snapshots; nothing is overwritten. Each episode embeds a copy and records its
id + hash. Research mode uses the latest *accepted* snapshot.

## Robot motor calibration (joint angles)
Set `robot.follower.calibration_file` to the LeRobot calibration JSON for **your** follower (candidates are listed by
`inspect`; the collector does not choose). It is only read and copied (+sha256) into episodes. Angles are valid only if
the JSON matches the homing offsets currently in the motors. The collector never recalibrates or moves the arm.

## Diagnostic vs research
`mode: diagnostic` records without an accepted calibration and flags the episode
(`camera_calibration_missing`, `diagnostic`). `mode: research` blocks `Start episode` unless `research_gates` pass.
