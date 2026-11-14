"""Scene configuration for the cube-to-tray MuJoCo scene: schema, validation and the physical-mode refusal logic.

Two modes, strictly separated:

* ``synthetic_demo``: every number that describes the world (base pose, cube, tray, initial configuration) is INVENTED
  and its ``source`` must say so. All outputs carry SYNTHETIC_LABEL.
* ``physical``: every world-describing input must be a MEASURED value with a ``source`` and an uncertainty (sigma), and
  the documented calibration mismatch blocker must be resolved (``calibration.calibration_blocker_resolved_by`` names the
  evidence). Anything missing -> ``SceneRefusal`` listing exactly what is missing. Nothing is defaulted or guessed.

CAD geometry is NOT a physically calibrated twin; that statement is part of every label.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from ...util import hash_obj

SCENE_SCHEMA_VERSION = "sba-scene/0.1"
MODES = ("synthetic_demo", "physical")
SYNTHETIC_LABEL = "SYNTHETIC DEMO SCENE (invented values, CAD geometry; not a calibrated twin)"
PHYSICAL_LABEL = ("PHYSICAL SCENE (measured inputs as configured; CAD geometry, NOT a physically calibrated twin; "
                  "simulator fidelity unvalidated)")
BLOCKER_TEXT = ("documented calibration mismatch is unresolved: simulator joint limits/zero (CAD-derived MJCF) vs the user's "
                "active LeRobot follower calibration (see sim_before_act/docs/STATUS.md and robot_collector/docs/CALIBRATION.md); "
                "set calibration.calibration_blocker_resolved_by to the evidence once it is resolved")
THRESHOLD_STATUS = "ASSUMED_UNCALIBRATED"

# group -> (numeric fields that must be present, uncertainty fields required in physical mode)
INPUT_GROUPS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "base_to_mat": (("x_m", "y_m", "z_m", "yaw_deg"), ("sigma_xy_m", "sigma_yaw_deg")),
    "cube": (("side_m", "mass_kg", "x_m", "y_m", "yaw_deg"), ("sigma_xy_m", "sigma_yaw_deg")),
    "tray": (("x_m", "y_m", "yaw_deg", "interior_x_m", "interior_y_m", "rim_height_m", "wall_thickness_m",
              "floor_thickness_m"), ("sigma_xy_m", "sigma_yaw_deg", "sigma_dim_m")),
}
POSITIVE = {"cube": ("side_m", "mass_kg"),
            "tray": ("interior_x_m", "interior_y_m", "rim_height_m", "wall_thickness_m", "floor_thickness_m")}
OUTCOME_KEYS = ("joint_limit_tolerance_deg", "penetration_tolerance_m", "max_abs_qvel_rad_s", "lift_height_m",
                "inside_tray_margin_m", "gripper_open_pct", "v_settle_m_s", "t_settle_s", "contact_dist_threshold_m")
SIM_KEYS = ("timestep_s", "settle_s", "task_horizon_s", "record_every_substeps", "wall_timeout_s", "init_limit_tolerance_deg")


class SceneConfigError(ValueError):
    """The scene config itself is malformed (distinct from a physical-mode refusal)."""


class SceneRefusal(RuntimeError):
    """Physical mode refused: ``missing`` lists every missing input/evidence precisely."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__("scene refused; " + "; ".join(self.missing))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass
class SceneConfig:
    raw: dict
    path: Path | None
    hash: str

    @property
    def mode(self) -> str:
        return self.raw["mode"]

    @property
    def label(self) -> str:
        return SYNTHETIC_LABEL if self.mode == "synthetic_demo" else PHYSICAL_LABEL

    def section(self, name: str) -> dict:
        return self.raw[name]

    @property
    def inputs(self) -> dict:
        return self.raw["inputs"]

    @property
    def base_dir(self) -> Path:
        return self.path.parent if self.path else Path.cwd()

    def robot_model_xml(self) -> Path:
        p = Path(self.raw["robot"]["model_xml"])
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def perturbation_cfg(self) -> dict:
        """Perturbation sampler config (same format as offline_demo.yaml); {} when the scene defines none."""
        return copy.deepcopy(self.raw.get("perturbations") or {})

    # ---- physical-mode refusal -----------------------------------------------------------------------------------
    def missing_inputs(self, init_state_available: bool = False) -> list[str]:
        """Precise list of what a physical run still needs. Empty for synthetic_demo (its values are invented by design)."""
        if self.mode != "physical":
            return []
        miss: list[str] = []
        for group, (fields, sigmas) in INPUT_GROUPS.items():
            g = self.inputs.get(group) or {}
            for f in fields:
                if not _num(g.get(f)):
                    miss.append(f"inputs.{group}.{f}: measured value missing (null)")
            for f in sigmas:
                if not _num(g.get(f)):
                    miss.append(f"inputs.{group}.{f}: uncertainty (sigma) missing (null)")
            src = g.get("source")
            if not (isinstance(src, str) and src.strip()):
                miss.append(f"inputs.{group}.source: measurement provenance missing (null)")
            elif "invented" in src.lower() or "synthetic" in src.lower():
                miss.append(f"inputs.{group}.source: says invented/synthetic ({src!r}); physical mode needs a measurement")
        ir = self.inputs.get("initial_robot") or {}
        vals = ir.get("values")
        if not init_state_available and not (isinstance(vals, list) and len(vals) == 6 and all(_num(v) for v in vals)):
            miss.append("inputs.initial_robot.values: measured initial joint state missing (or pass the segment state)")
        cal = self.raw.get("calibration") or {}
        ref = cal.get("reference") or {}
        for f in ("follower_calibration_sha256", "source"):
            if not (isinstance(ref.get(f), str) and ref[f].strip()):
                miss.append(f"calibration.reference.{f}: missing (null)")
        by = cal.get("calibration_blocker_resolved_by")
        if not (isinstance(by, str) and by.strip()):
            miss.append("calibration.calibration_blocker_resolved_by: null -> " + BLOCKER_TEXT)
        return miss

    def require_runnable(self, init_state_available: bool = False) -> None:
        miss = self.missing_inputs(init_state_available)
        if miss:
            raise SceneRefusal(miss)

    def describe(self) -> dict:
        return {"schema_version": SCENE_SCHEMA_VERSION, "mode": self.mode, "label": self.label, "config_hash": self.hash,
                "path": str(self.path) if self.path else None,
                "inputs": copy.deepcopy(self.inputs), "calibration": copy.deepcopy(self.raw.get("calibration")),
                "geometry_note": "CAD geometry (SO-101 MJCF from onshape-to-robot) + simple primitives; NOT a physically calibrated twin"}


def _validate(raw: dict) -> None:
    if not isinstance(raw, dict) or raw.get("schema_version") != SCENE_SCHEMA_VERSION:
        raise SceneConfigError(f"scene schema_version must be {SCENE_SCHEMA_VERSION}")
    if raw.get("mode") not in MODES:
        raise SceneConfigError(f"mode must be one of {MODES}")
    for sec in ("robot", "contract", "simulation", "controller", "mat", "inputs", "outcome"):
        if not isinstance(raw.get(sec), dict):
            raise SceneConfigError(f"scene section '{sec}' missing")
    if not raw["robot"].get("model_xml"):
        raise SceneConfigError("robot.model_xml missing")
    for k in SIM_KEYS:
        if not (_num(raw["simulation"].get(k)) and raw["simulation"][k] > 0):
            raise SceneConfigError(f"simulation.{k} must be a positive number")
    mat = raw["mat"]
    for k in ("size_x_m", "size_y_m", "thickness_m"):
        if not (_num(mat.get(k)) and mat[k] > 0):
            raise SceneConfigError(f"mat.{k} must be a positive number")
    out = raw["outcome"]
    for k in OUTCOME_KEYS:
        if not (_num(out.get(k)) and out[k] >= 0):
            raise SceneConfigError(f"outcome.{k} must be a non-negative number")
    if out.get("thresholds_calibrated") is not False:
        raise SceneConfigError("outcome.thresholds_calibrated must be false: no threshold has been calibrated")
    if not isinstance(out.get("unsafe_contact_categories"), list):
        raise SceneConfigError("outcome.unsafe_contact_categories must be a list")
    synthetic = raw["mode"] == "synthetic_demo"
    for group, (fields, _sig) in INPUT_GROUPS.items():
        g = raw["inputs"].get(group)
        if not isinstance(g, dict):
            raise SceneConfigError(f"inputs.{group} missing")
        for f in fields:
            v = g.get(f)
            if v is None and not synthetic:
                continue                     # physical: missing is a REFUSAL, reported precisely, not a config error
            if not _num(v):
                raise SceneConfigError(f"inputs.{group}.{f} must be a finite number"
                                       + (" (synthetic scene must be fully specified)" if synthetic else ""))
            if f in POSITIVE.get(group, ()) and v <= 0:
                raise SceneConfigError(f"inputs.{group}.{f} must be > 0")
        if synthetic and "invented" not in str(g.get("source", "")).lower():
            raise SceneConfigError(f"synthetic_demo inputs.{group}.source must state that the values are INVENTED")
    ir = raw["inputs"].get("initial_robot")
    if not isinstance(ir, dict):
        raise SceneConfigError("inputs.initial_robot missing")
    if synthetic:
        v = ir.get("values")
        if not (isinstance(v, list) and len(v) == 6 and all(_num(x) for x in v)):
            raise SceneConfigError("inputs.initial_robot.values must be 6 numbers (contract units) in synthetic_demo")
        if "invented" not in str(ir.get("source", "")).lower():
            raise SceneConfigError("synthetic_demo inputs.initial_robot.source must state INVENTED")
    elif ir.get("values") is not None and not (isinstance(ir["values"], list) and len(ir["values"]) == 6
                                                 and all(_num(x) for x in ir["values"])):
        raise SceneConfigError("inputs.initial_robot.values must be null or 6 numbers")
    if raw.get("calibration") is not None and not isinstance(raw["calibration"], dict):
        raise SceneConfigError("calibration must be a mapping")
    for name, spec in ((raw.get("perturbations") or {}).get("variables") or {}).items():
        for req in ("dist", "units", "source", "calibrated"):
            if req not in spec:
                raise SceneConfigError(f"perturbation '{name}' must declare '{req}'")
        if spec["dist"] == "normal" and not (spec.get("sigma", -1) >= 0 and spec.get("bound", -1) > 0):
            raise SceneConfigError(f"perturbation '{name}': normal needs sigma>=0 and bound>0")
        if spec["dist"] == "uniform" and not spec["low"] <= spec["high"]:
            raise SceneConfigError(f"perturbation '{name}': uniform needs low<=high")
        if spec["dist"] not in ("normal", "uniform"):
            raise SceneConfigError(f"perturbation '{name}': unsupported dist {spec['dist']!r}")


def load_scene_config(src: str | Path | dict, overrides: dict | None = None) -> SceneConfig:
    """Load + validate a scene config from a YAML path or a dict. ``overrides``: {"a.b.c": value} dotted-path edits."""
    path: Path | None = None
    if isinstance(src, dict):
        raw = copy.deepcopy(src)
    else:
        path = Path(src).resolve()
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as e:
            raise SceneConfigError(f"cannot read scene config {path}: {e}") from e
    for k, v in (overrides or {}).items():
        cur = raw
        keys = k.split(".")
        for p in keys[:-1]:
            cur = cur.setdefault(p, {})
        cur[keys[-1]] = v
    _validate(raw)
    return SceneConfig(raw=raw, path=path, hash=hash_obj(raw))
