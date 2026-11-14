"""Small shared helpers: canonical JSON/hashing, JSON-safe conversion, exclusive atomic writes."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


def to_jsonable(o):
    """Convert numpy/dataclass/Path values to JSON types. Non-finite floats become None (never silently 0)."""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return to_jsonable(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        return to_jsonable(o.item())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def canonical_json(o) -> str:
    return json.dumps(to_jsonable(o), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def hash_obj(o) -> str:
    return sha256_hex(canonical_json(o))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_exclusive(path: Path, obj, readonly: bool = True) -> None:
    """Write JSON to a temp file then hard-link into place; raises FileExistsError instead of overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    if readonly:
        os.chmod(path, 0o444)
