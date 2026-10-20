"""What a run cost, in joules.

Energy is reported because a screen exists to be the cheap alternative to a shadow-model
audit, and "cheap" is a claim that should carry a number. The boundary is the same one the
source study used, so the figures are comparable with it:

* **GPU board energy only**, read from the NVIDIA driver's own cumulative counter (NVML),
  including the board's idle draw while the job runs.
* CPU, memory, storage and cooling are **outside** the boundary, and power usage
  effectiveness is taken as 1.0 -- appropriate for a directly metered single GPU, and an
  understatement for a data centre.
* A CPU-only run is **not metered at all**. There is no portable counter for it: RAPL needs
  privileges and exists only on some platforms. The tool says "not metered" rather than
  inventing an estimate from runtime and nameplate power, which is the estimate the study
  found to be wrong.

Carbon is deliberately not computed here. It depends on the grid the machine is plugged
into, which this process cannot know; `report.py` quotes a figure only when the operator
supplies the intensity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Energy:
    """One metered interval."""

    seconds: float
    joules: float | None
    """None when no counter was available -- a CPU run, or a driver without the counter."""
    device: str

    @property
    def kwh(self) -> float | None:
        return None if self.joules is None else self.joules / 3.6e6

    @property
    def watts(self) -> float | None:
        if self.joules is None or self.seconds <= 0:
            return None
        return self.joules / self.seconds

    def text(self) -> str:
        """A short human line: minutes, and energy in a unit that does not read as zero."""
        out = f"{self.seconds / 60:.1f} minutes"
        if self.joules is None:
            return out + " (energy not metered on this device)"
        kwh = self.kwh or 0.0
        amount = f"{kwh:.4f} kWh" if kwh >= 1e-4 else f"{self.joules / 1000:.1f} kJ"
        return f"{out}, {amount}"

    def as_dict(self) -> dict:
        return {
            "seconds": round(self.seconds, 2),
            "minutes": round(self.seconds / 60, 2),
            "joules": None if self.joules is None else round(self.joules, 1),
            "kwh": None if self.kwh is None else round(self.kwh, 6),
            "mean_watts": None if self.watts is None else round(self.watts, 1),
            "device": self.device,
            "boundary": "GPU board only (NVML), PUE 1.0" if self.joules is not None
                        else "not metered: no GPU energy counter on this device",
        }


class Meter:
    """`with Meter(device) as m: ...` then read `m.result`.

    Nested or repeated use is fine: each meter reads the counter at entry and exit, so a
    screen can meter each retry separately and still report their sum.
    """

    def __init__(self, device: str = "cpu"):
        self.device = str(device)
        self.result = Energy(0.0, None, self.device)
        self._read = _reader(self.device)
        self._t0 = 0.0
        self._j0: float | None = None

    def __enter__(self) -> "Meter":
        self._j0 = self._read()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        seconds = time.perf_counter() - self._t0
        j1 = self._read()
        joules = None if (self._j0 is None or j1 is None) else max(j1 - self._j0, 0.0)
        self.result = Energy(seconds, joules, self.device)


def total(parts) -> Energy:
    """Sum metered intervals. Unmetered anywhere means unmetered overall, not zero."""
    parts = list(parts)
    if not parts:
        return Energy(0.0, None, "cpu")
    joules = (None if any(p.joules is None for p in parts)
              else float(sum(p.joules for p in parts)))
    return Energy(float(sum(p.seconds for p in parts)), joules, parts[0].device)


def _reader(device: str):
    """A function returning cumulative board joules, or None where that is unavailable."""
    if not device.startswith("cuda"):
        return lambda: None
    try:
        import pynvml
        import torch

        pynvml.nvmlInit()
        index = torch.cuda.current_device() if ":" not in device else int(device.split(":")[1])
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        def read() -> float | None:
            try:
                # millijoules since the driver loaded
                return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
            except Exception:
                return None

        read()  # fail fast here rather than halfway through a run
        return read
    except Exception:
        return lambda: None


def device_name(device: str) -> str:
    if not device.startswith("cuda"):
        return "CPU"
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return device
