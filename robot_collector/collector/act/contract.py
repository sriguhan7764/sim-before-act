"""Constants shared by the ACT conversion, training, inference and action-segment code. No heavy imports."""
from __future__ import annotations

import hashlib
import json

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
UNITS = ("degrees", "degrees", "degrees", "degrees", "degrees", "percent_0_100")
N_JOINTS = len(JOINT_NAMES)

CONVERSION_SCHEMA = "act-conversion-v0.1"
SEGMENT_SCHEMA = "action-segment-v0.1"
FRAMEWORK = "lerobot 0.6.1 ACT"

# LeRobot feature keys (lerobot.utils.constants: OBS_STATE, OBS_IMAGES + ".<cam>", ACTION)
KEY_STATE = "observation.state"
KEY_IMAGE_PREFIX = "observation.images."
KEY_ACTION = "action"
KEY_ACTION_IS_PAD = "action_is_pad"

ACTION_SIGNAL_DEFAULT = "commands.values"
ACTION_SIGNALS = {
    "commands.values": "controller's CLIPPED follower target (LeRobot normalised units); the default training action",
    "leader_measurement_normalized": ("leader arm measurement stored next to the command (leader calibration units). "
                                      "SUBSTITUTION: not what the follower was asked to reach; requires allow_signal_substitution"),
}

SMOKE_LABEL = "SYNTHETIC SMOKE TEST - software execution only, not a thesis finding"
SEGMENT_SMOKE_PROVENANCE = "SYNTHETIC SMOKE-TEST POLICY OUTPUT - not a trained policy, not a thesis finding"
PROVISIONAL_LABEL = "PROVISIONAL SPLIT - software/smoke use only; not for main training"
SYNTHETIC_DATA_LABEL = "SYNTHETIC FIXTURE DATA - fabricated by the software fixture builder; not real demonstrations"

SMOKE_KIND = "smoke_random_init_few_steps"

SEGMENT_CONTRACT = {
    "joint_names": list(JOINT_NAMES),
    "units": list(UNITS),
    "representation": "absolute_joint_position_targets",
    "interpolation": "zero_order_hold",
    # These two strings must equal sim_before_act.robot.action_contract.REFERENCE / GRIPPER_SEMANTICS exactly (the simulator side
    # is authoritative and its contract_id hashes them). Meaning: LeRobot normalised targets (degrees for 5 joints, 0..100
    # gripper), NOT MuJoCo torques, NOT confirmed physical execution; 0=closed and the simulator gripper mapping are unvalidated.
    "reference": "follower joint-position target space",
    "gripper": "normalized_0_100_0_closed",
}


def canonical(obj) -> str:
    """Deterministic compact JSON used for every hash in this package."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(canonical(obj))


def contract_id(control_hz: float) -> str:
    """sha256 of the canonical contract block (including control_hz) as it appears in an action segment."""
    return sha256_obj({**SEGMENT_CONTRACT, "control_hz": float(control_hz)})
