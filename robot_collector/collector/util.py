"""Small shared helpers: timestamps, hashing, atomic JSON, tolerant JSONL reading."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = "1.0.0"


def mono_ns() -> int:
    """Host monotonic clock (CLOCK_MONOTONIC; comparable across processes on one Linux host)."""
    return time.monotonic_ns()


def wall_ns() -> int:
    """Host wall clock, for traceability only; never used for association."""
    return time.time_ns()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_id(prefix: str) -> str:
    return f"{prefix}_{utc_stamp()}_{secrets.token_hex(3)}"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=_default) + "\n"


def _default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json_atomic(path: str | Path, obj: Any, fsync: bool = True) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(canonical_json(obj))
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def iter_jsonl(path: str | Path, tolerant: bool = True) -> Iterator[dict]:
    """Yield records; with tolerant=True a truncated/corrupt trailing line is skipped."""
    if not Path(path).exists():
        return
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not tolerant:
                    raise
                return


def read_jsonl(path: str | Path, tolerant: bool = True) -> list[dict]:
    return list(iter_jsonl(path, tolerant))


def jsonl_line(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, default=_default, separators=(",", ":")) + "\n").encode()


def repo_commit(start: str | Path | None = None) -> str | None:
    """Git commit of the collector checkout, or None when not a git repository."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(start or Path(__file__).resolve().parent),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None
