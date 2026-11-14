# Data-readiness checklist (planning; nothing here was collected or generated)

Principle: metadata that was never observed cannot be recovered from a recording. Anything estimated later from the
videos must be labelled as an estimate with its method, and is never ground truth.

## What the existing recording contains

`g_c_20261113_174844`: 1 episode, 598 frames at 30 fps, `observation.state` and `action` (6 joints, normalised),
front RGB, front depth (video), wrist RGB. Front frame 0 shows a red cube and a blue bowl on the mat with the four
ArUco markers, so *visual evidence* of objects exists. No object poses, task target, outcome label, raw ticks,
calibration id/hash or velocities are stored.

## Checklist

| Item | Needed for offline replay | Present now | Action |
|---|---|---|---|
| **Object-pose acquisition** | Pose of each object at the chunk start, with frame, source, uncertainty | No. Objects are visible in RGB(-D) only. | Choose and document a method: (a) operator-measured on the mat grid, (b) ArUco/colour segmentation + depth from the front camera using the existing camera-to-mat calibration. Store as sidecar with `source` and sigma. Do not infer from this recording without stating the method and its error. |
| **Camera-to-robot calibration** | Camera → mat → robot-base transform, with error | Camera-to-mat exists (`robot_collector`, ~1.55 px reprojection). **No validated mat→robot-base transform.** | Measure the robot base pose in the mat frame (see CALIBRATION_AUDIT §7), record the method and uncertainty. Until then object poses cannot be placed in the MJCF frame. |
| **Joint calibration / zero** | Validated mapping recorded joint value → simulator joint | Unresolved (see audit) | Physical measurement of zero/sign/limits; record raw ticks and calibration hash in future recordings. |
| **Task-target definition** | Explicit predicate (e.g. object in bowl region, tolerance) in sim frame | No | Define per task (e.g. "cube centre inside bowl radius R, z above floor"), with the numbers and their source. |
| **Matching actions to outcomes** | An outcome label belongs to the recorded action chunk only | No label | Segment the episode into chunks (start/end frame, state at start); attach the label to exactly those frames. Never reuse a label for a different action. |
| **Success / failure / unknown labels** | Operator-assigned, with failure type and label source | No | Add an operator sidecar per episode/chunk: `success | failure | unknown`, failure type, who/when/how; leave `unknown` if not observed. Videos may be reviewed by the operator, never auto-labelled. |
| **Robot velocities / controller metadata** | Initial velocity, controller gains, `max_relative_target` | Velocities not recorded; `max_relative_target=10` known from `run.sh` | Record controller settings in sidecar; velocities remain unavailable for this recording (sim starts at rest, flagged). |
| **Episode count / splits** | Several episodes, split by episode/session and arrangement | 1 episode | Needs more episodes before any calibration/test split. |

## Can the existing recording support offline evaluation?

* **Replaying a window through the pipeline: yes, mechanically** (it already loads and validates).
* **E1/E2/E4 (prediction vs observed outcome): no.** There are no outcome labels, no object poses, no task target
  and only one episode; there is nothing to evaluate against and no split is possible.
* **E3 (runtime feasibility): not as currently measured**; timing is preliminary.
* The recording remains useful as a *source of realistic joint trajectories* once the joint mapping is resolved.
  It is not, by itself, a replayable action-state-outcome record.

## Smallest evidence-backed next step

1. A no-motion, read-only physical measurement session (user-operated, powered-down or hand-moved arm; nothing
   commanded by software): record each joint's external-reference angle at the rest pose and at one reference pose,
   plus the raw `Present_Position` ticks read through LeRobot, to determine per-joint zero offset and sign (H1, H2, H3
   in the audit) and the gripper open/closed angles (H4). This is the minimum needed before any real action can have
   valid simulator semantics.
2. Then one operator-written sidecar for a single chunk (start state frame, object pose with source, task target,
   outcome label or `unknown`). That gives the first real record a valid replay with explicit provenance.
