"""End-to-end through the Recorder using the EXPLICIT mock backend (synthetic data)."""
import time
from pathlib import Path

import pytest

from collector.config import load_config
from collector.recorder import Recorder
from collector.util import read_json
from collector.validate import validate
from conftest import wait_until_ok

ROOT = Path(__file__).resolve().parents[1]


def mock_cfg(tmp_path, **over):
    import yaml
    raw = yaml.safe_load((ROOT / "configs/mock.yaml").read_text())
    raw["paths"] = {"episodes_root": str(tmp_path / "eps"), "calibrations_root": str(tmp_path / "cal")}
    raw["robot"]["follower"]["calibration_file"] = str(ROOT / "configs/mock_motor_calibration.json")
    raw.update(over)
    return load_config(raw)


def test_mock_record_flags_mock_diagnostic_and_auto_exports(tmp_path):
    rec = Recorder(mock_cfg(tmp_path))
    rec.connect()
    rec.start_workers()
    rec.start_episode("pick_place", "pick the block")
    wait_until_ok(lambda: rec.status["mock_rgbd"]["count"] >= 5 and rec.status["follower"]["count"] >= 5, "mock samples")
    ep = rec.finish_episode("unlabelled")
    rec.shutdown()
    meta = read_json(ep / "episode.json")
    qf = meta["quality_flags"]
    assert qf["mock_data"] and qf["diagnostic"] and qf["camera_calibration_missing"]
    assert "leader_state" in qf["missing_modalities"]
    assert "MOCK" in ep.name and meta["outcome"] == "unlabelled"
    assert validate(ep)["ok"]
    assert len(list((ep.parent / (ep.name + ".exports")).iterdir())) == 2  # both lightweight exports


def test_research_mode_blocks_without_calibration_and_with_mock(tmp_path):
    rec = Recorder(mock_cfg(tmp_path, mode="research"))
    rec.connect()
    rec.start_workers()
    with pytest.raises(RuntimeError, match="readiness gates"):
        rec.start_episode("t")
    rec.shutdown()


def test_unconnected_source_is_reported_not_mocked(tmp_path):
    cfg = mock_cfg(tmp_path)
    cfg["cameras"] = [{**cfg["cameras"][0], "name": "rs", "backend": "opencv", "device": None}]
    cfg["sync"]["reference_camera"] = "rs"
    rec = Recorder(cfg)
    rec.connect()
    st = rec.status["rs"]
    assert not st["connected"] and "device" in st["error"]
    assert not rec.sources_ok["rs"]


def test_shutdown_never_calls_robot_write_paths(tmp_path):
    class Spy:
        is_mock = True
        calls = []
        def open(self): self.calls.append("open")
        def close(self): self.calls.append("close")
        def poll(self): time.sleep(0.01); return []
        def describe(self): return {}
    spy = Spy()
    rec = Recorder(mock_cfg(tmp_path), follower=spy)
    rec.connect(); rec.start_workers(); rec.shutdown()
    assert set(spy.calls) == {"open", "close"}
