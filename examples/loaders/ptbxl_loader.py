"""Read one PTB-XL record straight from its WFDB files.

The manifest points at the `.dat` of each record; WFDB wants the path without a suffix and
returns `(samples, leads)`, so this flips it to the `(channels, time)` the screen expects.
Nothing else is needed: the per-record standardisation the tool applies to signals handles
the amplitude scale.

    dataconfessor screen manifest.csv --loader examples/loaders/ptbxl_loader.py:load \
        --kind signal

Needs `pip install wfdb`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb


def load(path: str | Path) -> np.ndarray:
    """(12, 1000) float32 for a 100 Hz record, (12, 5000) at 500 Hz."""
    signal, _fields = wfdb.rdsamp(str(Path(path).with_suffix("")))
    return np.asarray(signal, dtype=np.float32).T
