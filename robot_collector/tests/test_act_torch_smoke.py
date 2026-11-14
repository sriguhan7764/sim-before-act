"""Runs tests/torch_checks_act.py in the lerobot env (torch + LeRobot 0.6.1, CPU) from the torch-free collector test env.

SYNTHETIC SMOKE TEST - software execution only, not a thesis finding. No hardware is touched.
Skipped with an explicit reason when the lerobot env or its source tree is missing.
"""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEROBOT_PY = Path(os.environ.get("ACT_LEROBOT_PYTHON", "/home/apic3/miniforge3/envs/guhan_c_lerobot/bin/python"))
LEROBOT_SRC = Path(os.environ.get("ACT_LEROBOT_SRC", "/home/apic3/lerobot_v6_2/lerobot/src"))


@pytest.mark.skipif(not LEROBOT_PY.exists(), reason=f"lerobot env python not found: {LEROBOT_PY}")
@pytest.mark.skipif(not LEROBOT_SRC.exists(), reason=f"LeRobot source tree not found: {LEROBOT_SRC}")
def test_torch_checks_pass_in_the_lerobot_env():
    scratch = ROOT / ".work_act" / f"torch_checks_pytest_{os.getpid()}"
    env = {**os.environ, "PYTHONPATH": f"{LEROBOT_SRC}{os.pathsep}{ROOT}", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([str(LEROBOT_PY), str(ROOT / "tests" / "torch_checks_act.py"), str(scratch)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    m = re.search(r"torch checks: (\d+) passed", r.stdout)
    assert m and int(m.group(1)) >= 60, r.stdout
    assert not scratch.exists()
