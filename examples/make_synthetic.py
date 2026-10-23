"""A synthetic cohort with a known atypical tail, for trying the tool and for the tests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SIZE = 32
TAIL_FRACTION = 0.15


def make(
    out_dir: str | Path,
    *,
    n_patients: int = 400,
    tail_fraction: float = TAIL_FRACTION,
    max_records: int = 3,
    seed: int = 0,
) -> dict:
    """Write images, a manifest and the ground-truth list of atypical patients."""
    from PIL import Image

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    legibility = rng.uniform(0.0, 1.0, size=n_patients)
    tail = set(np.argsort(legibility)[: int(round(tail_fraction * n_patients))].tolist())

    rows, truth = [], []
    for p in range(n_patients):
        label = int(p % 2)
        name = f"P{p:04d}"
        if p in tail:
            truth.append(name)
        # Contrast falls with legibility, and the least legible patients get the other
        # class's blob: they are not merely noisy, they contradict the rule.
        contrast = 10.0 + 90.0 * legibility[p]
        shown = 1 - label if legibility[p] < 0.08 else label
        for r in range(1 + int(rng.integers(0, max_records))):
            img = rng.normal(96, 18, size=(SIZE, SIZE))
            y0 = 4 if shown == 0 else SIZE - 12
            img[y0:y0 + 8, 12:20] += contrast
            path = out / "images" / f"{name}_{r}.png"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
            rows.append({"path": f"images/{path.name}", "label": str(label),
                         "patient_id": name})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "patient_id"])
        w.writeheader()
        w.writerows(rows)
    (out / "atypical_truth.txt").write_text("\n".join(sorted(truth)), encoding="utf-8")

    return {"manifest": str(out / "manifest.csv"), "n_records": len(rows),
            "n_patients": n_patients, "atypical": sorted(truth)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    info = make(args.out, n_patients=args.patients, seed=args.seed)
    print(f"{info['n_records']} records / {info['n_patients']} patients -> "
          f"{info['manifest']}")
    print(f"{len(info['atypical'])} planted atypical patients (ground truth in "
          f"atypical_truth.txt)")
