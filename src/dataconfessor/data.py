"""Turning a manifest into batches, once.

The screen makes eight passes over the data (four proxy models, each trained then used to
score every record), so decoding JPEGs eight times would dominate a run that is supposed to
finish in minutes. Every record is therefore decoded once into a memory-mapped array on
disk, and every pass after that is a slice. The cache is keyed by the manifest contents and
the preprocessing, so changing either rebuilds it and nothing silently reuses the wrong one.

Records are read here and nowhere else. Nothing in this module returns record contents to a
caller outside the training loop.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .manifest import Manifest

DEFAULT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAY_MEAN, GRAY_STD = 0.449, 0.226
"""Channel-averaged ImageNet statistics, for the grayscale case."""


def _ram_budget() -> int:
    """Bytes of the record cache that may be held in memory.

    `DATACONFESSOR_RAM_GB` overrides it. Otherwise half of free memory where psutil is
    installed, and a conservative 4 GiB where it is not -- enough for the cohorts this tool
    is aimed at, and small enough not to push a laptop into swap.
    """
    override = os.environ.get("DATACONFESSOR_RAM_GB")
    if override:
        return int(float(override) * 2**30)
    try:
        import psutil

        return int(psutil.virtual_memory().available * 0.5)
    except Exception:
        return 4 * 2**30


def _native_size(path: Path) -> int:
    """Longest side of one image, read from the header without decoding the pixels."""
    from PIL import Image

    with Image.open(path) as im:
        return max(im.size)


def _fingerprint(man: Manifest, size: int, loader: str | None) -> str:
    h = hashlib.sha256()
    h.update(f"{size}|{loader}|{man.kind}|{len(man.paths)}".encode())
    for p in man.paths:
        h.update(str(p).encode())
    return h.hexdigest()[:16]


def _load_hook(spec: str) -> Callable[[Path], np.ndarray]:
    """Import `path/to/file.py:function` as a per-record loader.

    The escape hatch for DICOM series, WFDB signals, NIfTI volumes and anything else with a
    reader of its own. The function takes a path and returns a float32 array shaped like the
    other records -- (H, W) or (H, W, C) for images, (C, T) for signals.
    """
    file, _, func = spec.partition(":")
    if not func:
        raise ValueError(f"--loader must be 'path/to/file.py:function', got {spec!r}")
    mod_spec = importlib.util.spec_from_file_location("dataconfessor_loader", file)
    if mod_spec is None or mod_spec.loader is None:
        raise ImportError(f"cannot import {file}")
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules["dataconfessor_loader"] = module
    mod_spec.loader.exec_module(module)
    return getattr(module, func)


class RecordSet:
    """Cached records plus the batching the training loop needs."""

    def __init__(
        self,
        man: Manifest,
        *,
        size: int | None = None,
        cache_dir: str | Path | None = None,
        loader: str | None = None,
        device: torch.device | str = "cpu",
        progress: Callable[[str], None] = lambda _: None,
    ):
        self.man = man
        # Never upsample: a 28 px dataset blown up to 224 costs 64x the compute and adds no
        # information, and it would also hide the small-input case from `choose_arch`.
        self.size = size if size is not None else (
            min(_native_size(man.paths[0]), DEFAULT_SIZE) if man.kind == "image"
            else DEFAULT_SIZE
        )
        self.device = torch.device(device)
        self._norm_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.is_multilabel = man.is_multilabel
        self.num_classes = man.num_classes
        self.kind = man.kind
        self.patient_codes = man.patient_codes
        self.labels = man.labels

        cdir = Path(cache_dir) if cache_dir else man.root / ".dataconfessor_cache"
        cdir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cdir / f"records_{_fingerprint(man, self.size, loader)}.npy"
        self._read = _load_hook(loader) if loader else None
        if not self.cache_path.exists():
            self._build(progress)
        self.data = np.load(self.cache_path, mmap_mode="r")
        # Training reads random batches, which on a memory-mapped file is random disk I/O --
        # it dominated a screen of 33,000 images, costing more than the GPU work it fed. If
        # the whole cache fits in the budget, read it once, sequentially, and never touch the
        # disk again. Above the budget it stays mapped and the run is slower but bounded.
        if self.data.nbytes <= _ram_budget():
            progress(f"  loading {self.data.nbytes / 2**30:.1f} GiB of records into memory")
            self.data = np.ascontiguousarray(self.data)
        self.in_channels = int(self.data.shape[-1]) if man.kind == "image" else int(
            self.data.shape[1]
        )
        self.input_size = int(self.data.shape[1]) if man.kind == "image" else None

    # -- cache construction -------------------------------------------------------------

    def _build(self, progress: Callable[[str], None]) -> None:
        first = self._read_record(self.man.paths[0])
        shape = (len(self.man.paths), *first.shape)
        dtype = np.uint8 if self.man.kind == "image" else np.float32
        tmp = self.cache_path.with_suffix(".partial.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
        step = max(1, len(self.man.paths) // 4)
        for i, p in enumerate(self.man.paths):
            arr = first if i == 0 else self._read_record(p)
            if arr.shape != first.shape:
                raise ValueError(
                    f"{p} has shape {arr.shape}, but the first record has {first.shape}. "
                    f"Records must share a shape. For signals, resample or crop them to a "
                    f"fixed length first; for images this is handled by --size."
                )
            out[i] = arr
            if i % step == 0:
                progress(f"  caching records {i + 1}/{len(self.man.paths)}")
        out.flush()
        del out
        tmp.replace(self.cache_path)

    def _read_record(self, path: Path) -> np.ndarray:
        if self._read is not None:
            arr = np.asarray(self._read(path))
            return arr.astype(np.uint8 if self.man.kind == "image" else np.float32)
        if self.man.kind == "signal":
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2:
                raise ValueError(
                    f"{path} has {arr.ndim} dimensions; a signal record must be (channels, "
                    f"time). Volumes and other shapes need --loader (see the README)."
                )
            # Per-record standardisation: ECG amplitude scales vary between recording
            # systems, and an untrained one-epoch proxy has no chance of normalising them.
            mu, sd = arr.mean(axis=1, keepdims=True), arr.std(axis=1, keepdims=True)
            return (arr - mu) / np.maximum(sd, 1e-6)

        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB" if im.mode in ("RGB", "RGBA", "P") else "L")
            im = im.resize((self.size, self.size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        return arr if arr.ndim == 3 else arr[:, :, None]

    # -- batching -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.man.paths)

    def _norm(self, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        """ImageNet statistics on the compute device, built once."""
        if self._norm_cache is None:
            if channels == 3:
                mean, std = IMAGENET_MEAN, IMAGENET_STD
            else:
                mean, std = (GRAY_MEAN,) * channels, (GRAY_STD,) * channels
            self._norm_cache = (
                torch.tensor(mean, device=self.device).view(1, channels, 1, 1),
                torch.tensor(std, device=self.device).view(1, channels, 1, 1),
            )
        return self._norm_cache

    def batch(self, idx: np.ndarray, *, normalise: bool = True):
        """(x, y) for these record positions, on the screen's device.

        The uint8 batch is moved to the device *before* it is turned into floats. Converting
        5 GB of cached pixels to float32 on the CPU made data preparation, not the GPU, the
        bottleneck of a screen; this way the host only copies bytes.
        """
        idx = np.asarray(idx)
        raw = torch.from_numpy(np.asarray(self.data[idx])).to(self.device, non_blocking=True)
        if self.kind == "image":
            x = raw.permute(0, 3, 1, 2).float().div_(255.0)
            if normalise:
                mean, std = self._norm(x.shape[1])
                x = (x - mean) / std
        else:
            x = raw.float()

        y = torch.from_numpy(self.labels[idx]).to(self.device, non_blocking=True)
        return x, y.float() if self.is_multilabel else y.long()
