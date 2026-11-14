"""Importer tests use a tiny SYNTHETIC LeRobot-v3-shaped dataset built in a temp dir (not the user's real recording)."""
import json

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from sim_before_act.data.lerobot_import import import_lerobot  # noqa: E402
from sim_before_act.runtime.offline import run_attempt  # noqa: E402
from sim_before_act.schemas.record import ReplayRecord  # noqa: E402

NAMES = [f"{n}.pos" for n in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


@pytest.fixture
def ds(tmp_path):
    root = tmp_path / "ds"; (root / "meta").mkdir(parents=True); (root / "data" / "chunk-000").mkdir(parents=True)
    feat = {"dtype": "float32", "shape": [6], "names": NAMES}
    (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30, "robot_type": "so_follower",
                                                         "features": {"action": feat, "observation.state": feat}}))
    n = 40
    rows = [[float(i)] * 5 + [50.0] for i in range(n)]
    pq.write_table(pa.table({"action": rows, "observation.state": rows, "timestamp": [i / 30 for i in range(n)],
                             "frame_index": list(range(n)), "episode_index": [0] * n}), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_import_without_sidecar_is_honestly_unverifiable(ds, make_cfg, stack, tmp_path):
    d = import_lerobot(ds, 0, 2, 10)
    assert d["objects"] == [] and d["task"]["target_xyz_m"] is None and d["outcome"] == {} and d["synthetic"] is False
    assert d["action"]["chunk"][0][0] == 2.0 and len(d["action"]["chunk"]) == 10 and d["state"]["timestamp_s"] == pytest.approx(2 / 30)
    cfg, st = stack()
    res = run_attempt(cfg, ReplayRecord.from_dict(d), tmp_path / "t", st)
    assert res.verification.status == "INVALID_INPUT" and res.decision.decision == "REJECT"
    assert any("No operator sidecar" in x for x in res.trace["limitations"])


def test_import_rejects_out_of_episode_window(ds):
    with pytest.raises(ValueError):
        import_lerobot(ds, 0, 35, 10)


def test_import_never_writes_into_dataset(ds):
    before = sorted((p, p.stat().st_mtime_ns) for p in ds.rglob("*"))
    import_lerobot(ds, 0, 0, 5)
    assert before == sorted((p, p.stat().st_mtime_ns) for p in ds.rglob("*"))
