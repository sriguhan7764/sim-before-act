"""Visualise ONE rollout of a record in the MuJoCo twin (SYNTHETIC fixtures or any record).

  # video (offscreen, needs a GL backend; MUJOCO_GL=glfw works with a display)
  MUJOCO_GL=glfw python experiments/synthetic_demo/show_rollout.py --record safe_reach.json --seed-index 0 --video out.mp4
  # live window, real time
  python experiments/synthetic_demo/show_rollout.py --record floor_contact.json --live

Uses the same run_rollout() as the verifier (same reset, seeds, perturbations, predicates); the visualisation is read-only.
The scene is the generic, unvalidated SO-101 MJCF: this is a picture of the simulator, not of the physical robot.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sim_before_act.config import load_config  # noqa: E402
from sim_before_act.runtime.offline import build_stack  # noqa: E402
from sim_before_act.schemas.record import load_record  # noqa: E402
from sim_before_act.twin.perturbation import PerturbationSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "offline_demo.yaml"))
    ap.add_argument("--seed-index", type=int, default=0)
    ap.add_argument("--nominal", action="store_true", help="no perturbation")
    ap.add_argument("--video"); ap.add_argument("--live", action="store_true")
    ap.add_argument("--slow", type=float, default=1.0, help="live playback slowdown factor")
    a = ap.parse_args()
    rp = Path(a.record)
    rec = load_record(rp if rp.exists() else Path(__file__).parent / a.record)
    cfg = load_config(a.config)
    twin, contract, verifier, _ = build_stack(cfg)
    n = len(rec.action.chunk)
    pcfg = cfg.section("perturbations")
    smp = PerturbationSampler({"enabled": False} if a.nominal else pcfg, cfg.section("verification")["base_seed"], 50).sample(a.seed_index, n)
    objs = {o.name: {"position_m": o.position_m, "quat_wxyz": o.quat_wxyz} for o in rec.objects}
    args = (contract, np.array(rec.state.positions), objs, np.array(rec.action.chunk), smp, rec.action.control_hz,
            np.array(rec.task.target_xyz_m), rec.task.tolerance_m)
    print(f"{rec.record_id}  SYNTHETIC={rec.synthetic}  perturbation={'nominal' if smp.nominal else smp.summary()}")

    if a.live:
        import mujoco.viewer
        view = mujoco.viewer.launch_passive(twin.model, twin.data)
        t_start = time.monotonic()

        def cb(m, d, step):
            if step % 8 == 0:
                view.sync()
                lag = step * m.opt.timestep * a.slow - (time.monotonic() - t_start)
                if lag > 0:
                    time.sleep(lag)
        r = twin.run_rollout(*args, step_cb=cb)
        view.sync()
        print("rollout:", r.outcome, "task_failure=", r.task_failure, "safety=", r.safety_violations, "ee_err_m=", r.ee_error_m)
        print("window stays open; close it to exit")
        while view.is_running():
            time.sleep(0.1)
        return 0

    import cv2
    import mujoco
    out = a.video or str(ROOT / "traces_out" / f"{rec.record_id}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    W, H, FPS = 640, 480, 30
    rend = mujoco.Renderer(twin.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.0, 0.1]; cam.distance, cam.azimuth, cam.elevation = 0.9, 140, -25
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    state = {"label": "", "viol": set()}
    every = int(round(1.0 / (FPS * twin.dt)))

    def cb(m, d, step):
        if step % every:
            return
        rend.update_scene(d, cam)
        img = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        t = step * m.opt.timestep
        contacts = twin._contacts(0.0, True)
        state["viol"].update(contacts)
        cv2.putText(img, f"{rec.record_id}  t={t:4.2f}s  SYNTHETIC / unvalidated model", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if state["viol"]:
            cv2.putText(img, "SAFETY VIOLATION: " + ",".join(sorted(state["viol"])), (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        vw.write(img)
    r = twin.run_rollout(*args, step_cb=cb)
    for _ in range(FPS):  # hold the last frame ~1 s with the result
        rend.update_scene(twin.data, cam)
        img = cv2.cvtColor(rend.render(), cv2.COLOR_RGB2BGR)
        txt = f"outcome={r.outcome} task_failure={r.task_failure} safety={r.safety_violations}"
        cv2.putText(img, txt, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        vw.write(img)
    vw.release()
    print("rollout:", r.outcome, "task_failure=", r.task_failure, "safety=", r.safety_violations, "ee_err_m=", r.ee_error_m)
    print("video:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
