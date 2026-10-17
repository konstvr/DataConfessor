"""Build a DataConfessor manifest for PTB-XL.

PTB-XL is the one dataset in the source study whose labels need a rule rather than a column:
the diagnostic superclass of a record is the set of SCP codes it carries, intersected with
the codes `scp_statements.csv` marks as diagnostic, and mapped to their five classes. A
record with no diagnostic code is labelled NORM, which is what the study did.

    python examples/ptbxl_manifest.py --root /data/ptbxl --out ptbxl_manifest.csv
    dataconfessor screen ptbxl_manifest.csv --kind signal \
        --loader examples/loaders/ptbxl_loader.py:load

The labels are multi-label, so they are written `;`-separated, which is what the manifest
format expects.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path


def build(root: str | Path, out: str | Path, *, hz: int = 100) -> dict:
    root = Path(root).expanduser().resolve()
    out = Path(out).expanduser().resolve()

    with open(root / "scp_statements.csv", newline="", encoding="utf-8") as fh:
        diagnostic = {r[""]: r["diagnostic_class"] for r in csv.DictReader(fh)
                      if r.get("diagnostic") == "1.0" and r.get("diagnostic_class")}

    column = "filename_lr" if hz == 100 else "filename_hr"
    rows, missing = [], 0
    with open(root / "ptbxl_database.csv", newline="", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            classes = sorted({diagnostic[c] for c in ast.literal_eval(rec["scp_codes"])
                              if c in diagnostic})
            path = root / f"{rec[column]}.dat"
            if not path.exists():
                missing += 1
                continue
            rows.append({
                "path": path.as_posix(),
                "label": ";".join(classes) if classes else "NORM",
                "patient_id": rec["patient_id"].split(".")[0],
            })

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "patient_id"])
        w.writeheader()
        w.writerows(rows)

    patients = {r["patient_id"] for r in rows}
    return {"manifest": str(out), "records": len(rows), "patients": len(patients),
            "missing_files": missing}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="the PTB-XL release directory")
    ap.add_argument("--out", default="ptbxl_manifest.csv")
    ap.add_argument("--hz", type=int, default=100, choices=[100, 500])
    args = ap.parse_args()
    print(build(args.root, args.out, hz=args.hz))
