"""Offscreen MuJoCo replay worker (run as a subprocess so the GL backend can be chosen via MUJOCO_GL before mujoco is imported).

Replays the saved qpos trace (no physics, ``mj_forward`` only) and writes a PNG contact sheet (and optionally an MP4).
Every image carries the scene label. Usage: python -m sim_before_act.twin.scene.render_worker CFG.json TRACE.npz OUT.png [--video OUT.mp4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .builder import build_scene
from .config import load_scene_config

CAMERAS = ("overview", "top")


def _banner(img: np.ndarray, text: str, colour=(0, 0, 255)) -> np.ndarray:
    h = 20
    out = np.full((img.shape[0] + h, img.shape[1], 3), 255, np.uint8)
    out[h:] = img
    cv2.putText(out, text[:int(img.shape[1] / 6.0)], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg_json")
    ap.add_argument("trace_npz")
    ap.add_argument("out_png")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    a = ap.parse_args(argv)

    cfg = load_scene_config(json.loads(Path(a.cfg_json).read_text()))
    scene = build_scene(cfg, init_state_available=True)
    m = scene.model
    tr = np.load(a.trace_npz)
    qpos, t = tr["qpos"], tr["t_s"]
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=a.height, width=a.width)       # raises if no GL context can be created

    def frame(i: int, cam: str) -> np.ndarray:
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"t={t[i]:.2f}s", (4, a.height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"t={t[i]:.2f}s", (4, a.height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    idx = np.unique(np.linspace(0, len(t) - 1, max(2, a.frames)).round().astype(int))
    rows = [np.hstack([frame(int(i), cam) for i in idx]) for cam in CAMERAS]
    sheet = np.vstack(rows)
    sheet = _banner(sheet, cfg.label)
    sheet = _banner(sheet, "replayed qpos trace via mj_forward (cameras: overview, top); CAD geometry, not a calibrated twin", (0, 0, 0))
    cv2.imwrite(a.out_png, sheet)

    if a.video:
        vw = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (a.width, a.height + 20))
        if not vw.isOpened():
            print("VIDEO_UNAVAILABLE: cv2.VideoWriter could not open", file=sys.stderr)
        else:
            for i in range(0, len(t), max(1, len(t) // 120)):
                vw.write(_banner(frame(i, "overview"), cfg.label))
            vw.release()
    print(f"RENDER_OK gl={__import__('os').environ.get('MUJOCO_GL', 'default')} frames={len(idx)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
