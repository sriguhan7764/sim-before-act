"""Structured episode metadata for the red-cube -> blue-tray pilot (setup / demonstration / evaluation).

Nothing here estimates or invents a pose. Poses come from a layout file the operator fills in after measuring
(``configs/pilot_layout.yaml``; every pose ships as null). A missing pose is recorded as missing and flagged; it is
never replaced by a default. This metadata is operator-entered provenance, not an automatic scene reconstruction.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .util import sha256_file

ROLES = ("setup", "demonstration", "evaluation")
LABEL_SOURCES = ("operator_visual", "operator_measured", "none")
POSE_SOURCES = ("manual_measurement", "known_fixture_placement", "operator_estimate")
POSE_KEYS = ("x_m", "y_m", "yaw_deg")
OUTCOMES_NEEDING_LABEL_SOURCE = ("success", "failure")

# Timed operator cues for a short setup recording. Times are nominal (seconds after the episode start).
PROTOCOLS = {
    "setup_short": {"segments": [
        {"label": "stationary_start", "start_s": 0.0, "end_s": 5.0,
         "instruction": "do not touch either arm (leader held still; follower should hold its pose)"},
        {"label": "small_supervised_motion", "start_s": 5.0, "end_s": 17.0,
         "instruction": "move the LEADER slowly and gently through a small pick-and-place-like motion well inside its range; "
                        "never push toward a mechanical stop; release the leader and stop at once if anything looks wrong"},
        {"label": "stationary_end", "start_s": 17.0, "end_s": 22.0,
         "instruction": "hold the leader still again (follower should settle)"}]},
}


class PilotError(ValueError):
    pass


def load_layout(path: str | Path) -> dict:
    p = Path(path)
    doc = yaml.safe_load(p.read_text()) or {}
    doc["_sha256"] = sha256_file(p)
    doc["_path"] = str(p)
    return doc


def pose_missing(pose: dict | None) -> list[str]:
    """Names of the pose fields that are not measured (empty list = complete)."""
    if not pose:
        return ["pose"]
    miss = [k for k in POSE_KEYS if pose.get(k) is None]
    if pose.get("source") not in POSE_SOURCES:
        miss.append("source")
    if pose.get("sigma_mm") is None:
        miss.append("sigma_mm")
    return miss


def _clean_pose(pose: dict | None) -> dict | None:
    if pose is None:
        return None
    return {k: pose.get(k) for k in (*POSE_KEYS, "source", "sigma_mm", "measured_by", "measurement_note")}


def build_pilot_meta(role: str, layout: dict | None = None, cube_position_id: str | None = None,
                     cube_pose: dict | None = None, tray_pose: dict | None = None, attempt_index: int | None = None,
                     notes: str = "") -> dict:
    """Assemble the ``pilot`` block. Pose arguments override the layout; neither is ever auto-filled."""
    if role not in ROLES:
        raise PilotError(f"role must be one of {ROLES}, got {role!r}")
    layout = layout or {}
    if role == "demonstration" and not cube_position_id:
        raise PilotError("demonstration episodes need --cube-position-id (one of the five measured start positions)")
    if cube_position_id and layout.get("cube_start_positions") is not None \
            and cube_position_id not in layout["cube_start_positions"] and cube_pose is None:
        raise PilotError(f"cube position '{cube_position_id}' is not in the layout "
                         f"({sorted(layout['cube_start_positions'])}); pass an explicit measured pose or fix the id")
    cube = cube_pose if cube_pose is not None else (layout.get("cube_start_positions") or {}).get(cube_position_id)
    tray = tray_pose if tray_pose is not None else (layout.get("tray") or {}).get("pose_mat_frame")
    return {
        "role": role,
        "cube_position_id": cube_position_id,
        "cube_start_pose_mat_frame": _clean_pose(cube),
        "tray_pose_mat_frame": _clean_pose(tray),
        "cube_dimensions_mm": (layout.get("cube") or {}).get("dimensions_mm"),
        "tray_dimensions_mm": {k: (layout.get("tray") or {}).get(k) for k in
                               ("interior_diameter_mm", "rim_height_mm", "interior_depth_mm", "rim_thickness_mm")},
        "robot_start_configuration": None,   # filled from the first measured follower state at start_episode
        "outcome_label_source": "none",
        "attempt_index": attempt_index,
        "layout": {"path": layout.get("_path"), "sha256": layout.get("_sha256"), "layout_version": layout.get("layout_version")},
        "frame_conventions": layout.get("frames"),
        "operator_notes": notes,
        "note": "operator-entered; poses are NOT estimated from images. Null/flagged where unmeasured.",
    }


def pilot_quality_flags(pilot: dict | None) -> dict:
    """Explicit flags; consumers must not treat a missing pose as zero."""
    if not pilot:
        return {"pilot_metadata_missing": True}
    cube_missing = pose_missing(pilot.get("cube_start_pose_mat_frame"))
    tray_missing = pose_missing(pilot.get("tray_pose_mat_frame"))
    return {
        "pilot_metadata_missing": False,
        "cube_pose_missing_fields": cube_missing,
        "tray_pose_missing_fields": tray_missing,
        "cube_pose_measured": not cube_missing,
        "tray_pose_measured": not tray_missing,
        "robot_start_configuration_missing": pilot.get("robot_start_configuration") is None,
    }


def check_label(outcome: str, label_source: str) -> None:
    if label_source not in LABEL_SOURCES:
        raise PilotError(f"label source must be one of {LABEL_SOURCES}")
    if outcome in OUTCOMES_NEEDING_LABEL_SOURCE and label_source == "none":
        raise PilotError(f"outcome '{outcome}' needs a label source (who/what decided it); use 'unlabelled' otherwise")
    if outcome == "unlabelled" and label_source != "none":
        raise PilotError("an 'unlabelled' outcome cannot have a label source")


def eligibility(meta: dict, annotation: dict | None = None) -> dict:
    """Why an episode may or may not count as an accepted demonstration. Never raises; returns reasons.

    The accepted-for-dataset decision itself is a curator annotation (the master is immutable); this only reports
    whether the recorded metadata could support it.
    """
    pilot = meta.get("pilot") or {}
    qf = meta.get("quality_flags", {})
    reasons = []
    if pilot.get("role") != "demonstration":
        reasons.append(f"role is {pilot.get('role')!r}, not 'demonstration'")
    outcome = (annotation or {}).get("outcome", meta.get("outcome"))
    if outcome != "success":
        reasons.append(f"outcome is {outcome!r}, not 'success'")
    if meta.get("status") != "complete":
        reasons.append(f"status is {meta.get('status')!r}, not 'complete'")
    if qf.get("mock_data"):
        reasons.append("mock/synthetic data")
    if not pilot.get("cube_position_id"):
        reasons.append("no cube position id")
    if pose_missing(pilot.get("cube_start_pose_mat_frame")):
        reasons.append("cube start pose not measured")
    if pose_missing(pilot.get("tray_pose_mat_frame")):
        reasons.append("tray pose not measured")
    for m in qf.get("missing_modalities", []):
        reasons.append(f"missing modality: {m}")
    return {"eligible": not reasons, "reasons": reasons}
