"""Builds the MuJoCo cube-to-tray scene by COMPOSING the existing SO-101 MJCF (read-only) with a mat, a cube and a tray.

The protected robot files are only READ (``MjSpec.from_file``); every change (base pose, added bodies, geom names) is
made on the in-memory spec. Nothing is written back.

Frames: the world frame IS the mat frame (origin = midpoint of the mat edge nearest the robot, +X right seen from the robot,
+Y away from the robot across the mat, +Z up; mat top surface at z = 0). The robot's MJCF ``base`` body frame is placed at
``inputs.base_to_mat`` (x, y, z, yaw). The MJCF robot looks along its own +X, so a base yaw of +90 deg makes it look along the
mat's +Y. Which physical point of the real robot the MJCF ``base`` origin corresponds to is NOT calibrated.

Geom naming (used to categorise contacts): ``robot_arm/<body>/<k>`` (collision geoms of non-gripper links),
``robot_gripper/<body>/<k>`` (collision geoms of the ``gripper`` and ``moving_jaw`` bodies), ``cube``, ``tray_*``, ``mat``, ``table``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ...robot.action_contract import JOINT_NAMES, ActionContract
from ...util import hash_obj, sha256_file
from .config import SceneConfig, SceneConfigError

GRIPPER_BODIES = ("gripper", "moving_jaw_so101_v1")
# cube friction/softness copied from so101_mujoco/scene.xml (ASSUMED, uncalibrated)
CUBE_CONTACT = {"friction": [1.5, 0.02, 0.0005], "condim": 4, "solref": [0.02, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]}   # last two = MuJoCo defaults


def yaw_quat(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)])


def _cam_xyaxes(pos, target) -> list[float]:
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return [*x, *y]


@dataclass
class Scene:
    cfg: SceneConfig
    model: mujoco.MjModel
    contract: ActionContract
    qadr: np.ndarray                 # qpos addresses of the six contract joints
    act: np.ndarray                  # actuator ids of the six contract joints
    jnt_ids: np.ndarray
    cube_qadr: int
    cube_dofadr: int
    cube_geom: int
    ee_site: int
    geom_kind: list[str]             # per geom: 'robot_arm'|'robot_gripper'|'cube'|'tray'|'mat'|'table'|'other'
    tray_geoms: set
    asset_hash: str
    controller: dict
    geometry: dict                   # resolved numbers used by the predicates

    def new_data(self) -> mujoco.MjData:
        return mujoco.MjData(self.model)

    @property
    def label(self) -> str:
        return self.cfg.label

    def describe(self) -> dict:
        m = self.model
        return {"label": self.label, "mode": self.cfg.mode, "engine": "mujoco", "mujoco_version": mujoco.__version__,
                "scene_config_hash": self.cfg.hash, "robot_model_xml": str(self.cfg.robot_model_xml()),
                "robot_asset_hash": self.asset_hash, "timestep_s": float(m.opt.timestep), "ngeom": int(m.ngeom),
                "nbody": int(m.nbody), "nu": int(m.nu), "contract_id": self.contract.contract_id,
                "geometry": self.geometry, "controller": self.controller,
                "fidelity": "CAD geometry + simple primitives; NOT a physically calibrated twin"}

    def ee_position(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.ee_site].copy()

    def cube_pos(self, data) -> np.ndarray:
        return data.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    def place_cube(self, data, x: float, y: float, yaw_deg: float, z: float | None = None) -> None:
        """Set the cube pose in the mat frame (default z: resting on the mat). Velocities zeroed. Caller runs mj_forward."""
        g = self.geometry
        zz = g["mat_top_z_m"] + g["cube_side_m"] / 2 if z is None else z
        data.qpos[self.cube_qadr:self.cube_qadr + 3] = [x, y, zz]
        data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = yaw_quat(math.radians(yaw_deg))
        data.qvel[self.cube_dofadr:self.cube_dofadr + 6] = 0.0

    def set_robot_q(self, data, q_rad: np.ndarray) -> None:
        data.qpos[self.qadr] = q_rad
        data.ctrl[self.act] = q_rad
        data.qvel[:] = 0.0


def _read_controller(model: mujoco.MjModel, expected: dict) -> dict:
    act = [model.actuator(n).id for n in JOINT_NAMES]
    jid = [model.joint(n).id for n in JOINT_NAMES]
    dof = [model.joint(n).dofadr[0] for n in JOINT_NAMES]
    rb = {"kp": [float(model.actuator_gainprm[a, 0]) for a in act],
          "kv": [float(-model.actuator_biasprm[a, 2]) for a in act],
          "forcerange_nm": [[float(x) for x in model.actuator_forcerange[a]] for a in act],
          "ctrlrange_rad": [[float(x) for x in model.actuator_ctrlrange[a]] for a in act],
          "joint_damping": [float(model.dof_damping[d]) for d in dof],
          "joint_armature": [float(model.dof_armature[d]) for d in dof],
          "joint_frictionloss": [float(model.dof_frictionloss[d]) for d in dof]}
    for key, ex in (("kp", expected.get("kp")), ("kv", expected.get("kv")), ("joint_damping", expected.get("joint_damping")),
                    ("joint_armature", expected.get("joint_armature")), ("joint_frictionloss", expected.get("joint_frictionloss"))):
        if ex is not None and not np.allclose(rb[key], ex, rtol=1e-6):
            raise SceneConfigError(f"controller.expected.{key}={ex} does not match the robot model read-back {rb[key][0]}: "
                                   "the protected robot model changed or the config is stale")
    if expected.get("forcerange_nm") is not None and not np.allclose(np.abs(np.asarray(rb["forcerange_nm"])), expected["forcerange_nm"]):
        raise SceneConfigError("controller.expected.forcerange_nm does not match the robot model")
    return {**rb, "type": "mujoco position actuators (PD): force = kp*(target-q) - kv*qdot, clipped to forcerange",
            "source": "read back from the compiled robot MJCF (CAD-derived sts3215 class defaults); NOT identified from the real Feetech controller",
            "note": "ACT/policy outputs are absolute joint-position TARGETS; they are written to ctrl, not applied as torques"}


def build_scene(cfg: SceneConfig, init_state_available: bool = False) -> Scene:
    """Compose and compile the scene. Physical mode: raises SceneRefusal listing every missing input / the open blocker."""
    cfg.require_runnable(init_state_available)
    inp, sim, mat = cfg.inputs, cfg.section("simulation"), cfg.section("mat")
    xml = cfg.robot_model_xml()
    if not xml.is_file():
        raise FileNotFoundError(f"robot model not found: {xml}")
    pin = cfg.section("robot").get("model_sha256")
    if pin and sha256_file(xml) != pin:
        raise SceneConfigError(f"robot model sha256 differs from the pinned robot.model_sha256 ({xml})")
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.option.timestep = float(sim["timestep_s"])
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST    # as so101_mujoco/scene.xml
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0, 0, 0]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    b = inp["base_to_mat"]
    base = spec.body("base")
    base.pos = [b["x_m"], b["y_m"], b["z_m"]]
    base.quat = yaw_quat(math.radians(b["yaw_deg"]))

    # name the collision geoms of the robot (by parent body) so contacts can be categorised from geom names
    counts: dict[str, int] = {}
    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            continue
        bn = g.parent.name
        k = counts.get(bn, 0)
        counts[bn] = k + 1
        g.name = f"{'robot_gripper' if bn in GRIPPER_BODIES else 'robot_arm'}/{bn}/{k}"

    w = spec.worldbody
    t = float(mat["thickness_m"])
    sx, sy = float(mat["size_x_m"]), float(mat["size_y_m"])
    w.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.05], pos=[0, 0.2, -t],
               rgba=[0.35, 0.35, 0.38, 1])
    w.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, t / 2], pos=[0, sy / 2, -t / 2],
               rgba=[0.62, 0.66, 0.6, 1], friction=[float(mat.get("friction", 1.0)), 0.005, 0.0001])
    w.add_light(pos=[0, 0.2, 2.5], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5])

    c = inp["cube"]
    side = float(c["side_m"])
    cb = w.add_body(name="cube", pos=[c["x_m"], c["y_m"], side / 2])
    cb.add_freejoint(name="cube_free")
    cb.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[side / 2] * 3, mass=float(c["mass_kg"]),
                rgba=[0.85, 0.1, 0.1, 1], friction=CUBE_CONTACT["friction"], condim=CUBE_CONTACT["condim"],
                solref=CUBE_CONTACT["solref"], solimp=CUBE_CONTACT["solimp"])

    tr = inp["tray"]
    ix, iy, rim, wt, ft = (float(tr[k]) for k in ("interior_x_m", "interior_y_m", "rim_height_m", "wall_thickness_m", "floor_thickness_m"))
    tb = w.add_body(name="tray", pos=[tr["x_m"], tr["y_m"], 0.0], quat=yaw_quat(math.radians(tr["yaw_deg"])))
    blue = [0.1, 0.25, 0.85, 1]
    tb.add_geom(name="tray_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[ix / 2 + wt, iy / 2 + wt, ft / 2], pos=[0, 0, ft / 2], rgba=blue)
    zc = ft + rim / 2
    for nm, pos, size in (("tray_wall_px", [ix / 2 + wt / 2, 0, zc], [wt / 2, iy / 2 + wt, rim / 2]),
                          ("tray_wall_nx", [-ix / 2 - wt / 2, 0, zc], [wt / 2, iy / 2 + wt, rim / 2]),
                          ("tray_wall_py", [0, iy / 2 + wt / 2, zc], [ix / 2, wt / 2, rim / 2]),
                          ("tray_wall_ny", [0, -iy / 2 - wt / 2, zc], [ix / 2, wt / 2, rim / 2])):
        tb.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=blue)

    # fixed cameras for rendering (target = middle of the working area)
    for name, pos in (("overview", [0.62, -0.30, 0.50]), ("front", [0.0, 0.85, 0.40]), ("top", [0.0, 0.2, 0.9])):
        xy = _cam_xyaxes(pos, [0.0, 0.2, 0.04]) if name != "top" else [1, 0, 0, 0, 1, 0]
        w.add_camera(name=name, pos=pos, xyaxes=xy)

    model = spec.compile()
    contract = ActionContract.build(cfg.section("contract"),
                                    {n: tuple(np.rad2deg(model.jnt_range[model.joint(n).id])) for n in JOINT_NAMES},
                                    tuple(float(x) for x in model.jnt_range[model.joint("gripper").id]))
    kinds = []
    for gi in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        kinds.append("robot_arm" if nm.startswith("robot_arm/") else "robot_gripper" if nm.startswith("robot_gripper/")
                     else "cube" if nm == "cube" else "tray" if nm.startswith("tray_") else nm if nm in ("mat", "table") else "other")
    root = xml.parent
    assets = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
              if p.is_file() and p.suffix.lower() in (".xml", ".stl") and "__pycache__" not in p.parts and p.parent in (root, root / "assets")}
    geometry = {
        "frame": "mat frame = world: origin midpoint of the mat edge nearest the robot, +X right (seen from robot), +Y away, +Z up",
        "mat_size_m": [sx, sy, t], "mat_top_z_m": 0.0, "mat_x_range_m": [-sx / 2, sx / 2], "mat_y_range_m": [0.0, sy],
        "cube_side_m": side, "cube_mass_kg": float(c["mass_kg"]),
        "tray_pose": {"x_m": tr["x_m"], "y_m": tr["y_m"], "yaw_deg": tr["yaw_deg"]},
        "tray_interior_m": [ix, iy], "tray_rim_height_m": rim, "tray_floor_top_z_m": ft, "tray_rim_top_z_m": ft + rim,
        "tray_wall_thickness_m": wt, "base_to_mat": {k: b[k] for k in ("x_m", "y_m", "z_m", "yaw_deg")},
        "cube_contact_params": CUBE_CONTACT, "cube_contact_params_source": "copied from so101_mujoco/scene.xml; ASSUMED, uncalibrated",
        "robot_collision_geoms": {k: v for k, v in counts.items()},
        "gripper_bodies": list(GRIPPER_BODIES),
    }
    return Scene(cfg=cfg, model=model, contract=contract,
                 qadr=np.array([model.joint(n).qposadr[0] for n in JOINT_NAMES]),
                 act=np.array([model.actuator(n).id for n in JOINT_NAMES]),
                 jnt_ids=np.array([model.joint(n).id for n in JOINT_NAMES]),
                 cube_qadr=int(model.joint("cube_free").qposadr[0]), cube_dofadr=int(model.joint("cube_free").dofadr[0]),
                 cube_geom=model.geom("cube").id, ee_site=model.site("gripperframe").id, geom_kind=kinds,
                 tray_geoms={i for i, k in enumerate(kinds) if k == "tray"}, asset_hash=hash_obj(assets),
                 controller=_read_controller(model, cfg.section("controller").get("expected") or {}), geometry=geometry)
