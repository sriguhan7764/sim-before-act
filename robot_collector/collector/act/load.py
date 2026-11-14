"""Reader of a converted ACT dataset (numpy only) and the action-chunk builder.

Chunk rule (same semantics as LeRobotDataset's ``action_is_pad``, confirmed in lerobot 0.6.1 ``dataset_reader._get_query_indices``:
indices are clamped to the episode and ``is_pad`` is True beyond the episode): for the row at control-grid index ``g`` of episode
``e``, chunk position ``j`` refers to grid index ``g + j`` of THE SAME episode.

* if that grid step is a kept row of the episode -> its action, ``is_pad = False``;
* otherwise (past the episode end, OR a step dropped during conversion) -> ``is_pad = True`` and the value repeats the last valid
  action of this chunk. Padded values never reach the loss (``action_is_pad`` masks them) and are not interpolated.
Episode boundaries are never crossed, and a dropped step is never filled with a neighbour's value.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import CONVERSION_SCHEMA, KEY_IMAGE_PREFIX, sha256_obj

STEPS_FILE = "steps.npz"
CONVERSION_FILE = "conversion.json"
STATS_FILE = "normalization_stats.json"
IMAGE_INDEX_FILE = "image_index.json"
SPLIT_CODES = {"train": 0, "val": 1}


class ConvertedDatasetError(ValueError):
    pass


def arrays_sha256(arrays: dict) -> str:
    """Content hash of named arrays (independent of the npz container, which embeds timestamps)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def stats_hash(stats_doc: dict) -> str:
    return sha256_obj({k: v for k, v in stats_doc.items() if k != "sha256"})


class Converted:
    """Read-only view of ``<converted_dir>``. ``verify=True`` recomputes the array/stats hashes recorded in conversion.json."""

    def __init__(self, root: str | Path, verify: bool = True):
        self.root = Path(root)
        cj = self.root / CONVERSION_FILE
        if not cj.exists():
            raise ConvertedDatasetError(f"{self.root} is not a converted ACT dataset (no {CONVERSION_FILE})")
        self.conv = json.loads(cj.read_text())
        if self.conv.get("schema") != CONVERSION_SCHEMA:
            raise ConvertedDatasetError(f"unsupported conversion schema {self.conv.get('schema')!r}")
        with np.load(self.root / STEPS_FILE, allow_pickle=False) as z:
            self.a = {k: z[k] for k in z.files}
        self.stats_doc = json.loads((self.root / STATS_FILE).read_text())
        if verify:
            if arrays_sha256(self.a) != self.conv["files"]["steps_arrays_sha256"]:
                raise ConvertedDatasetError("steps.npz content does not match conversion.json (edited after conversion)")
            if stats_hash(self.stats_doc) != self.stats_doc["sha256"] or self.stats_doc["sha256"] != self.conv["normalization_sha256"]:
                raise ConvertedDatasetError("normalization_stats.json does not match its recorded hash")
        self.n = int(self.a["episode_idx"].shape[0])
        self.cameras = list(self.conv["cameras"])
        self._grid: dict[int, dict[int, int]] = {}
        for r in range(self.n):
            self._grid.setdefault(int(self.a["episode_idx"][r]), {})[int(self.a["grid_index"][r])] = r

    @property
    def data_origin(self) -> str:
        return self.conv["data_origin"]

    @property
    def sha256(self) -> str:
        return self.conv["conversion_sha256"]

    def episode(self, idx: int) -> dict:
        return self.conv["episodes"][idx]

    def row_indices(self, split: str, require_wrist: bool = False) -> np.ndarray:
        m = self.a["split_code"] == SPLIT_CODES[split]
        if require_wrist and "wrist" in self.cameras:
            m &= self.a["wrist_valid"]
        return np.nonzero(m)[0]

    def image_rel(self, cam: str, row: int) -> str | None:
        fi = int(self.a[f"{cam}_frame_index"][row])
        if fi < 0:
            return None
        ep = self.episode(int(self.a["episode_idx"][row]))["episode_id"]
        return f"images/{cam}/{ep}/{fi:06d}.png"

    def image_path(self, cam: str, row: int) -> Path | None:
        rel = self.image_rel(cam, row)
        return self.root / rel if rel else None

    def chunk(self, row: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
        """(actions (chunk_size, 6) float64, is_pad (chunk_size,) bool) for ``row``; see the module docstring."""
        e, g = int(self.a["episode_idx"][row]), int(self.a["grid_index"][row])
        grid = self._grid[e]
        act = np.empty((chunk_size, self.a["action"].shape[1]), dtype=np.float64)
        pad = np.ones(chunk_size, dtype=bool)
        last = None
        for j in range(chunk_size):
            r = grid.get(g + j)
            if r is not None:
                last = self.a["action"][r]
                pad[j] = False
            act[j] = last
        return act, pad

    def padding_summary(self, chunk_size: int) -> dict:
        """How many rows have padded chunk positions, split into episode-end padding and padding caused by dropped steps."""
        end = gap = full = 0
        last_grid = {e: max(m) for e, m in self._grid.items()}
        for r in range(self.n):
            _, pad = self.chunk(r, chunk_size)
            e, g = int(self.a["episode_idx"][r]), int(self.a["grid_index"][r])
            beyond = np.arange(g, g + chunk_size) > last_grid[e]
            end += int((pad & beyond).any())
            gap += int((pad & ~beyond).any())
            full += int(not pad.any())
        return {"chunk_size": chunk_size, "rows": self.n, "rows_with_episode_end_padding": end,
                "rows_with_padding_from_dropped_steps": gap, "rows_fully_valid": full}

    def feature_keys(self) -> list[str]:
        return [KEY_IMAGE_PREFIX + c for c in self.cameras]
