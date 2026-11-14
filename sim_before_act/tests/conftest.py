import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "offline_demo.yaml"
FIX_DIR = ROOT / "experiments" / "synthetic_demo"


def pytest_collection_modifyitems(items):
    for it in items:
        if "simulation" in it.nodeid.split("/") or "/simulation/" in str(it.fspath):
            it.add_marker(pytest.mark.simulation)


@pytest.fixture(scope="session")
def cfg_path():
    return CFG_PATH


@pytest.fixture(scope="session")
def base_cfg():
    from sim_before_act.config import load_config
    return load_config(CFG_PATH)


@pytest.fixture(scope="session")
def twin(base_cfg):
    from sim_before_act.twin.mujoco_twin import MujocoTwin
    return MujocoTwin(base_cfg.scene_xml(), base_cfg.section("twin"), base_cfg.section("rollout"))


@pytest.fixture(scope="session")
def contract(base_cfg, twin):
    from sim_before_act.robot.action_contract import ActionContract
    return ActionContract.build(base_cfg.section("contract"), twin.joint_limits_deg, twin.gripper_range_rad)


@pytest.fixture
def make_cfg():
    from sim_before_act.config import load_config

    def _make(**overrides):
        return load_config(CFG_PATH, overrides or None)
    return _make


@pytest.fixture
def stack(make_cfg, twin):
    from sim_before_act.runtime.offline import build_stack
    from sim_before_act.twin.mujoco_twin import MujocoTwin

    def _stack(**overrides):
        cfg = make_cfg(**overrides)
        # a twin is bound to its rollout/twin config; build a fresh one when those are overridden
        own = any(k.startswith(("rollout.", "twin.")) for k in overrides)
        t = MujocoTwin(cfg.scene_xml(), cfg.section("twin"), cfg.section("rollout")) if own else twin
        return cfg, build_stack(cfg, t)
    return _stack


@pytest.fixture
def rec_dict():
    def _load(name="safe_reach"):
        return copy.deepcopy(json.loads((FIX_DIR / f"{name}.json").read_text()))
    return _load


@pytest.fixture
def rec(rec_dict):
    from sim_before_act.schemas.record import ReplayRecord

    def _mk(name="safe_reach", **mut):
        d = rec_dict(name)
        for path, val in mut.items():
            cur = d
            keys = path.split("__")
            for k in keys[:-1]:
                cur = cur[k]
            cur[keys[-1]] = val
        return ReplayRecord.from_dict(d)
    return _mk
