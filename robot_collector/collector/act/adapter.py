"""Thin torch adapter: converted ACT dataset -> batches for ``ACTPolicy.forward`` (torch is imported lazily).

Batch keys (what lerobot 0.6.1 ACT expects): ``observation.state`` (B,6) float32 degrees/percent; ``observation.images.<cam>``
(B,3,H,W) float32 in [0,1]; ``action`` (B,chunk,6) float32; ``action_is_pad`` (B,chunk) bool. Normalisation is NOT applied here: the
LeRobot MEAN_STD pre-processor is rebuilt from the frozen ``normalization_stats.json`` and applied identically to train and val.
"""
from __future__ import annotations

import numpy as np

from .contract import KEY_ACTION, KEY_ACTION_IS_PAD, KEY_IMAGE_PREFIX, KEY_STATE
from .load import Converted


def _read_png_rgb(path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read converted image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_to_tensor_array(rgb_u8: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> 3xHxW float32 in [0, 1] (same scaling as the image statistics)."""
    return np.ascontiguousarray(rgb_u8.astype(np.float32).transpose(2, 0, 1) / 255.0)


class ActStepsDataset:
    """Rows of one split of a converted dataset. ``require_wrist`` excludes rows whose wrist frame is masked-missing."""

    def __init__(self, conv: Converted, split: str, chunk_size: int, require_wrist: bool = True):
        self.conv, self.split, self.chunk_size = conv, split, int(chunk_size)
        self.rows = conv.row_indices(split, require_wrist=require_wrist)
        self._img: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(len(self.rows))

    def _image(self, cam: str, row: int) -> np.ndarray:
        p = self.conv.image_path(cam, row)
        if p is None:
            raise ValueError(f"row {row} has no {cam} image (masked-missing); exclude it with require_wrist=True")
        key = str(p)
        if key not in self._img:
            self._img[key] = image_to_tensor_array(_read_png_rgb(p))
        return self._img[key]

    def item(self, i: int) -> dict:
        r = int(self.rows[i])
        act, pad = self.conv.chunk(r, self.chunk_size)
        out = {KEY_STATE: self.conv.a["state"][r].astype(np.float32), KEY_ACTION: act.astype(np.float32), KEY_ACTION_IS_PAD: pad}
        for cam in self.conv.cameras:
            out[KEY_IMAGE_PREFIX + cam] = self._image(cam, r)
        return out

    def raw_actions(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """float64 (chunk,6) recorded commands and the pad mask for row ``i`` (for the OFFLINE prediction error)."""
        return self.conv.chunk(int(self.rows[i]), self.chunk_size)

    def batch(self, idx) -> dict:
        import torch

        items = [self.item(int(i)) for i in idx]
        return {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0]}


def index_stream(n_rows: int, batch_size: int, seed: int, step: int) -> np.ndarray:
    """Row positions of training step ``step`` (0-based). A pure function of (seed, step): epoch ``e`` is a permutation drawn from
    ``default_rng([seed, e])`` and batches are consecutive slices of the concatenated epochs, so resuming needs only the step."""
    start = step * batch_size
    out: list[int] = []
    pos = start
    while len(out) < batch_size:
        e, off = divmod(pos, n_rows)
        perm = np.random.default_rng([int(seed), int(e)]).permutation(n_rows)
        take = min(batch_size - len(out), n_rows - off)
        out.extend(perm[off: off + take].tolist())
        pos += take
    return np.asarray(out, dtype=np.int64)


def stats_tensors(stats_doc: dict) -> dict:
    """normalization_stats.json -> {key: {"mean": tensor, "std": tensor}} in the layout lerobot's processors expect."""
    import torch

    out = {}
    for k, v in stats_doc["stats"].items():
        mean = torch.tensor(v["mean"], dtype=torch.float32)
        std = torch.tensor(v["std"], dtype=torch.float32)
        if k.startswith(KEY_IMAGE_PREFIX):
            mean, std = mean.reshape(3, 1, 1), std.reshape(3, 1, 1)
        out[k] = {"mean": mean, "std": std}
    return out
