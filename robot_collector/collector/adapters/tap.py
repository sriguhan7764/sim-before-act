"""JSONL tap: lets an existing teleoperation process share telemetry/commands without a second bus opener.

Teleop side (add two lines to your existing loop; the collector never touches the robot):

    from collector.adapters.tap import TapWriter
    tap = TapWriter("/tmp/follower_state.jsonl")
    tap.write({"raw": {"shoulder_pan": 2048, ...}})          # after reading Present_Position
    cmd_tap.write({"values": {...}, "units": "..."})        # right before send_action

Timestamps use CLOCK_MONOTONIC (time.monotonic_ns), which is shared by processes on one Linux host,
so tap timestamps are directly comparable with the collector's camera receipt times.
Each record also gets ``t_receipt_ns`` stamped by the collector when it is read from the file.
"""
from __future__ import annotations

import hashlib
import json
import os

from ..util import mono_ns, wall_ns
from .base import StateSource


class TapWriter:
    def __init__(self, path: str):
        self.f = open(path, "a", buffering=1)

    def write(self, rec: dict) -> None:
        t = mono_ns()
        rec = {"t_start_ns": t, "t_end_ns": t, "wall_ns": wall_ns(), "read_ok": True, **rec}
        self.f.write(json.dumps(rec) + "\n")


class TapSource(StateSource):
    def __init__(self, path: str, kind: str, units: str | None = None):
        self.path, self.kind, self.units = path, kind, units
        self._f, self._buf = None, b""

    def open(self):
        if not self.path:
            raise RuntimeError(f"{self.kind} tap: path unset")
        self._f = open(self.path, "rb")
        self._f.seek(0, os.SEEK_END)  # only new records

    def close(self):
        if self._f:
            self._f.close()

    def poll(self):
        import time

        chunk = self._f.read()
        if not chunk:
            time.sleep(0.002)
            return []
        self._buf += chunk
        *lines, self._buf = self._buf.split(b"\n")
        out, t = [], mono_ns()
        for ln in lines:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            r["t_receipt_ns"] = t
            out.append(r)
        return out

    def describe(self):
        d = {"backend": "jsonl_tap", "is_mock": False, "kind": self.kind, "path": self.path, "units": self.units,
             "session": None}
        sidecar = os.path.join(os.path.dirname(self.path) or ".", "session.json")
        try:  # written by the publishing process (collector.lerobot_bridge): controller config + signal semantics
            with open(sidecar) as f:
                d["session"] = json.load(f)
            d["session_sha256"] = hashlib.sha256(open(sidecar, "rb").read()).hexdigest()
        except (OSError, json.JSONDecodeError):
            pass
        return d
