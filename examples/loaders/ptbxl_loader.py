"""Read one PTB-XL record from its WFDB files as (channels, time). Needs `pip install wfdb`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb


def load(path: str | Path) -> np.ndarray:
    """(12, 1000) float32 for a 100 Hz record, (12, 5000) at 500 Hz."""
    signal, _fields = wfdb.rdsamp(str(Path(path).with_suffix("")))
    return np.asarray(signal, dtype=np.float32).T
