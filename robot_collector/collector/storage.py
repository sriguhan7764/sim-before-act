"""Canonical master episode writer: bounded queue, dedicated writer thread, atomic finalisation.

Layout (see docs/SCHEMA.md):
  <id>.inprogress/ -> renamed to <id>/ on finalisation (atomic rename on one filesystem)
"""
from __future__ import annotations

import os
import queue
from concurrent.futures import ThreadPoolExecutor
import shutil
import stat
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .sync import build_associations
from .util import (SCHEMA_VERSION, jsonl_line, new_id, read_json, sha256_file, wall_ns, write_json_atomic)

STATUS_COMPLETE = "complete"
STATUS_COMPLETE_WITH_DROPS = "complete_with_drops"
STATUS_ABORTED = "aborted"
STATUS_ABORTED_OVERFLOW = "aborted_writer_overflow"
STATUS_ABORTED_DISK = "aborted_low_disk"
STATUS_INTERRUPTED = "interrupted_recovered"
STATUS_FAILED_WRITE = "failed_write_error"
OUTCOMES = ("success", "failure", "aborted", "unlabelled")


class _ThreadGroup:
    """Lets callers join all per-stream writer threads like a single thread."""

    def __init__(self, w):
        self._w = w

    def join(self):
        for t in list(self._w.threads.values()):
            t.join()


class EpisodeWriter:
    def __init__(self, root: str | Path, meta: dict, rec_cfg: dict, sync_cfg: dict, episode_id: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.id = episode_id or new_id("ep" + ("_MOCK" if meta.get("quality_flags", {}).get("mock_data") else ""))
        self.final_dir = self.root / self.id
        self.dir = self.root / (self.id + ".inprogress")
        if self.final_dir.exists() or self.dir.exists():
            raise FileExistsError(f"episode {self.id} already exists; refusing to overwrite")
        self.dir.mkdir()
        (self.dir / "streams").mkdir()
        (self.dir / "telemetry").mkdir()
        self.cfg, self.sync_cfg = rec_cfg, sync_cfg
        # One bounded queue + writer thread per stream keeps per-stream ordering while letting cameras and telemetry
        # drain in parallel (PNG encoding releases the GIL). Telemetry records are tiny, so they get a deeper queue.
        self.queues: dict[str, queue.Queue] = {}
        self.threads: dict[str, threading.Thread] = {}
        self._pool = ThreadPoolExecutor(max_workers=int(rec_cfg.get("png_workers", 4)), thread_name_prefix="png")
        self.meta = {"schema_version": SCHEMA_VERSION, "episode_id": self.id, "status": "in_progress",
                     "outcome": "unlabelled", "start_wall_ns": wall_ns(), **meta}
        self.meta.setdefault("quality_flags", {})
        self.meta.setdefault("streams", {})
        self.counts = {"submitted": {}, "written": {}, "dropped": {}}
        self.lock = threading.Lock()
        self.abort_reason: str | None = None
        self.write_error: str | None = None
        self.overflow_abort = False
        self._files: dict[str, object] = {}
        self._closing = threading.Event()
        self._last_flush = time.monotonic()
        self._flush_lock = threading.Lock()
        self.writer_stats = {"items_written": 0, "busy_s": 0.0, "t0": time.monotonic(), "max_queue": 0}
        self.thread = _ThreadGroup(self)
        self._events = open(self.dir / "events.jsonl", "ab", buffering=0)
        self.flush_meta()

    # ---------- submission (called from acquisition threads; never blocks) ----------
    def submit(self, stream: str, kind: str, payload) -> bool:
        """Non-blocking. Items arriving after finalisation started are refused (not counted as drops)."""
        with self.lock:
            if self._closing.is_set():
                return False
            self.counts["submitted"][stream] = self.counts["submitted"].get(stream, 0) + 1
            q = self.queues.get(stream)
            if q is None:
                depth = int(self.cfg["queue_max_items"])
                q = self.queues[stream] = queue.Queue(maxsize=depth if kind == "frame" else max(depth, 4096))
                t = self.threads[stream] = threading.Thread(target=self._run, args=(stream, q),
                                                            name=f"writer-{stream}", daemon=True)
                t.start()
            try:
                q.put_nowait((stream, kind, payload))
                self.writer_stats["max_queue"] = max(self.writer_stats["max_queue"], q.qsize())
                return True
            except queue.Full:
                self.counts["dropped"][stream] = self.counts["dropped"].get(stream, 0) + 1
        self.log_event("queue_overflow", stream=stream, kind=kind, queue_max=q.maxsize)
        if self.cfg["overflow_policy"] == "abort":
            self.overflow_abort = True
            self.abort_reason = "writer queue overflow (policy=abort)"
        return False

    def log_event(self, event: str, **kw) -> None:
        with self.lock:
            self._events.write(jsonl_line({"event": event, "wall_ns": wall_ns(), "mono_ns": time.monotonic_ns(), **kw}))

    # ---------- writer thread ----------
    def _fh(self, rel: str):
        f = self._files.get(rel)
        if f is None:
            p = self.dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            f = self._files[rel] = open(p, "ab")
        return f

    def _png(self, rel: str, arr: np.ndarray) -> None:
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(p), arr, [cv2.IMWRITE_PNG_COMPRESSION, int(self.cfg["png_compression"])])
        if not ok:
            raise IOError(f"imwrite failed: {rel}")

    def _write_item(self, stream: str, kind: str, payload) -> None:
        if kind == "frame":
            fr, rec = payload  # CameraFrame, base record
            n = rec["recv_index"]
            base = f"streams/{stream}"
            rec = dict(rec)
            jobs = []
            if fr.color_rgb is not None:
                jobs.append((f"{base}/color/{n:06d}.png", cv2.cvtColor(fr.color_rgb, cv2.COLOR_RGB2BGR)))
                rec["color_file"] = f"color/{n:06d}.png"
            for d in ("depth_native", "depth_aligned"):
                a = getattr(fr, d)
                if a is not None:
                    if a.dtype != np.uint16:
                        raise TypeError("depth must be uint16")
                    jobs.append((f"{base}/{d}/{n:06d}.png", a))
                    rec[f"{d}_file"] = f"{d}/{n:06d}.png"
            for f in [self._pool.submit(self._png, rel, arr) for rel, arr in jobs]:
                f.result()  # re-raises encode/write errors before the index line is written
            self._fh(f"{base}/frames.jsonl").write(jsonl_line(rec))  # index line only after files exist
        else:
            self._fh(f"telemetry/{stream}.jsonl").write(jsonl_line(payload))
        with self.lock:
            self.counts["written"][stream] = self.counts["written"].get(stream, 0) + 1

    def _run(self, stream: str, q: queue.Queue) -> None:
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                if self._closing.is_set():
                    return
                self._maybe_flush()
                continue
            t = time.monotonic()
            try:
                self._write_item(*item)
            except Exception as e:  # keep draining so the episode is flagged, not hung
                self.write_error = f"{type(e).__name__}: {e}"
                self.log_event("write_error", error=self.write_error)
                with self.lock:
                    self.counts["dropped"][item[0]] = self.counts["dropped"].get(item[0], 0) + 1
            with self.lock:
                self.writer_stats["items_written"] += 1
                self.writer_stats["busy_s"] += time.monotonic() - t
            self._maybe_flush()

    def _maybe_flush(self) -> None:
        with self._flush_lock:
            if time.monotonic() - self._last_flush < self.cfg["flush_interval_s"]:
                return
            self._last_flush = time.monotonic()
            for f in list(self._files.values()):
                f.flush()
                os.fsync(f.fileno())
        self.flush_meta()

    def queue_stats(self) -> dict:
        qs = list(self.queues.values())
        return {"size": sum(q.qsize() for q in qs), "max": sum(q.maxsize for q in qs),
                "peak": self.writer_stats["max_queue"],
                "per_stream": {k: {"size": q.qsize(), "max": q.maxsize} for k, q in self.queues.items()}}

    def flush_meta(self) -> None:
        with self.lock:
            m = dict(self.meta)
            m["counts"] = {k: dict(v) for k, v in self.counts.items()}
        write_json_atomic(self.dir / "episode.json", m, fsync=False)

    def writer_rate_hz(self) -> float:
        dt = max(1e-6, time.monotonic() - self.writer_stats["t0"])
        return self.writer_stats["items_written"] / dt

    # ---------- finalisation ----------
    def update_meta(self, **kw) -> None:
        with self.lock:
            self.meta.update(kw)

    def finalize(self, outcome: str = "unlabelled", failure_category: str | None = None, notes: str = "",
                 aborted_reason: str | None = None) -> Path:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        with self.lock:
            self._closing.set()
        self.thread.join()
        self._pool.shutdown(wait=True)
        for f in self._files.values():
            f.flush()
            os.fsync(f.fileno())
            f.close()
        self._events.close()
        dropped = sum(self.counts["dropped"].values())
        reason = aborted_reason or self.abort_reason
        if reason:
            status = (STATUS_ABORTED_OVERFLOW if self.overflow_abort else
                      STATUS_ABORTED_DISK if "disk" in reason else STATUS_ABORTED)
            outcome = "aborted"
        elif self.write_error:
            status = STATUS_FAILED_WRITE
        else:
            status = STATUS_COMPLETE_WITH_DROPS if dropped else STATUS_COMPLETE
        qf = self.meta["quality_flags"]
        qf["data_dropped"] = bool(dropped)
        qf["write_error"] = self.write_error
        with self.lock:
            self.meta.update(status=status, outcome=outcome, failure_category=failure_category, notes=notes,
                             aborted_reason=reason, end_wall_ns=wall_ns())
        self.flush_meta()
        return finalize_directory(self.dir, self.final_dir, self.sync_cfg, self.meta.get("reference_camera"))


def finalize_directory(inprog: Path, final: Path, sync_cfg: dict, reference_camera: str | None = None) -> Path:
    """Compute derived tables, quality report and manifest, freeze files, atomically publish."""
    (inprog / "sync").mkdir(exist_ok=True)
    rows, report = build_associations(inprog, sync_cfg, reference_camera)
    with open(inprog / "sync" / "associations.jsonl", "wb") as f:
        for r in rows:
            f.write(jsonl_line(r))
    write_json_atomic(inprog / "sync" / "sync_report.json", report)
    meta = read_json(inprog / "episode.json")
    streams = {}
    for c in sorted(meta.get("streams", {})):
        recs = [json_ for json_ in _iter(inprog / "streams" / c / "frames.jsonl")]
        idx = [r["recv_index"] for r in recs]
        gaps = (max(idx) + 1 - len(idx)) if idx else 0
        streams[c] = {"frames_written": len(recs), "missing_recv_indices": gaps,
                      "dropped_counted": meta.get("counts", {}).get("dropped", {}).get(c, 0)}
    tele = {k: sum(1 for _ in _iter(inprog / "telemetry" / f"{k}.jsonl"))
            for k in ("follower_state", "leader_state", "commands")}
    q = {"schema_version": SCHEMA_VERSION, "status": meta["status"], "streams": streams, "telemetry_records": tele,
         "counts": meta.get("counts", {}), "quality_flags": meta.get("quality_flags", {}), "sync": report}
    write_json_atomic(inprog / "quality_report.json", q)
    files = {}
    for p in sorted(inprog.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json",) and not p.name.endswith(".tmp"):
            files[p.relative_to(inprog).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    write_json_atomic(inprog / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "kind": "master_episode", "episode_id": meta["episode_id"],
        "status": meta["status"], "mock": bool(meta.get("quality_flags", {}).get("mock_data")), "files": files})
    for p in inprog.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    if final.exists():
        raise FileExistsError(final)
    os.rename(inprog, final)
    fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return final


def _iter(path):
    from .util import iter_jsonl

    return iter_jsonl(path)


def disk_free_gb(path: str | Path) -> float:
    return shutil.disk_usage(path).free / 1e9
