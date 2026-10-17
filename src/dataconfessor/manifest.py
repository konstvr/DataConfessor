"""The dataset contract: one CSV with three columns.

    path,label,patient_id
    images/ISIC_0015719.jpg,0,IP_3075186

`path` is relative to the manifest file (or absolute). `label` is a class index, a class
name, or a `;`-separated set of positive classes for multi-label tasks. `patient_id` groups
records: every record of one person must carry the same value.

That last rule is the one nothing downstream can check. A dataset once grouped 191,000
images under two "patients" (it keyed on sex) and every per-patient number that came out of
it was meaningless while looking perfectly plausible. `audit()` below reports the grouping
statistics loudly for exactly this reason -- it cannot verify the key, only make a wrong one
visible.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REQUIRED = ("path", "label", "patient_id")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ARRAY_SUFFIXES = {".npy"}
MULTILABEL_SEPARATORS = ";|"


@dataclass
class Manifest:
    """A validated dataset description. Holds paths and labels, never record contents."""

    root: Path
    paths: list[Path]
    labels: np.ndarray
    """[n] int64 class indices, or [n, n_classes] float32 indicators when multi-label."""
    patient_codes: np.ndarray
    """[n] int64. Contiguous codes; `patient_names[code]` recovers the original id."""
    patient_names: list[str]
    class_names: list[str]
    is_multilabel: bool
    kind: str
    """'image' or 'signal' -- decided by file suffix, not by the user."""
    per_record: bool = False
    """True when the dataset has no patient identifiers, so every record is its own patient."""
    warnings: list[str] = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return len(self.paths)

    @property
    def n_patients(self) -> int:
        return len(self.patient_names)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise ValueError(
            f"{path} is missing column(s) {missing}. A manifest needs exactly "
            f"{list(REQUIRED)}; found {list(rows[0])}. Build one with `dataconfessor init`."
        )
    return rows


def load(
    manifest_path: str | Path,
    *,
    no_patient_id: bool = False,
    kind: str | None = None,
) -> Manifest:
    """Read and validate a manifest. Raises on anything that would invalidate the screen.

    `kind` ("image" or "signal") overrides the guess made from file suffixes. It is what a
    dataset read through a custom loader needs, since its files can be named anything.
    """
    mpath = Path(manifest_path).expanduser().resolve()
    root = mpath.parent
    rows = _read_rows(mpath)

    paths, raw_labels, raw_patients = [], [], []
    for i, row in enumerate(rows):
        p = (row["path"] or "").strip()
        if not p:
            raise ValueError(f"row {i + 2}: empty path")
        paths.append(Path(p) if Path(p).is_absolute() else root / p)
        raw_labels.append((row["label"] or "").strip())
        raw_patients.append((row["patient_id"] or "").strip())

    suffixes = {p.suffix.lower() for p in paths}
    if kind is not None:
        if kind not in ("image", "signal"):
            raise ValueError(f"kind must be 'image' or 'signal', got {kind!r}")
    elif suffixes <= IMAGE_SUFFIXES:
        kind = "image"
    elif suffixes <= ARRAY_SUFFIXES:
        kind = "signal"
    else:
        raise ValueError(
            f"mixed or unsupported file types {sorted(suffixes)}. Natively this tool reads "
            f"images ({sorted(IMAGE_SUFFIXES)}) or .npy arrays, not both. Anything else "
            f"needs --loader together with --kind image|signal -- see references/manifest.md."
        )

    step = max(1, len(paths) // 2000)
    missing = [p for p in paths[::step] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} sampled path(s) do not exist, first: {missing[0]}. Paths are "
            f"resolved relative to the manifest file ({root})."
        )

    labels, class_names, is_multilabel = _encode_labels(raw_labels)
    warnings: list[str] = []

    per_record = no_patient_id or not any(raw_patients)
    if per_record:
        raw_patients = [f"record_{i}" for i in range(len(paths))]
        warnings.append(
            "No patient identifiers: results are PER RECORD, not per patient. If one person "
            "contributed several records, the ranking is not a patient ranking."
        )
    elif not all(raw_patients):
        raise ValueError(
            f"{sum(1 for p in raw_patients if not p)} rows have an empty patient_id. Fill "
            f"them in, or pass --no-patient-id to screen per record."
        )

    patient_names = sorted(set(raw_patients))
    index = {name: i for i, name in enumerate(patient_names)}
    patient_codes = np.fromiter((index[p] for p in raw_patients), dtype=np.int64,
                                count=len(raw_patients))

    man = Manifest(
        root=root, paths=paths, labels=labels, patient_codes=patient_codes,
        patient_names=patient_names, class_names=class_names,
        is_multilabel=is_multilabel, kind=kind, per_record=per_record, warnings=warnings,
    )
    man.warnings.extend(audit(man))
    return man


def _encode_labels(raw: list[str]) -> tuple[np.ndarray, list[str], bool]:
    multi = any(any(s in v for s in MULTILABEL_SEPARATORS) for v in raw)
    if multi:
        sets = [{t.strip() for t in v.replace("|", ";").split(";") if t.strip()} for v in raw]
        names = sorted({t for s in sets for t in s})
        idx = {n: i for i, n in enumerate(names)}
        y = np.zeros((len(raw), len(names)), dtype=np.float32)
        for i, s in enumerate(sets):
            for t in s:
                y[i, idx[t]] = 1.0
        return y, names, True

    if any(v == "" for v in raw):
        raise ValueError("some rows have an empty label; the screen needs the training label")
    names = sorted(set(raw), key=lambda v: (not v.lstrip("-").isdigit(), _as_num(v), v))
    idx = {n: i for i, n in enumerate(names)}
    return np.fromiter((idx[v] for v in raw), dtype=np.int64, count=len(raw)), names, False


def _as_num(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return 0.0


def audit(man: Manifest) -> list[str]:
    """Aggregate sanity checks. Returns warnings; raises only on a fatal grouping error.

    Everything here is a count or a distribution -- no record is opened and no identifier is
    returned, so the output is safe to show an operator or an AI assistant.
    """
    out: list[str] = []
    counts = np.bincount(man.patient_codes)

    if man.n_patients < 2:
        raise ValueError(
            "every record belongs to one patient_id. The grouping key is wrong -- check "
            "that the column really is a person identifier."
        )
    if not man.per_record and man.n_patients == man.n_records:
        out.append(
            "every patient has exactly one record, so patient_id may be a per-record key "
            "(a filename or accession) rather than a person. Verify it."
        )
    if man.n_patients < 50:
        out.append(f"only {man.n_patients} patients: the 20% flag rate selects "
                   f"{max(1, round(0.2 * man.n_patients))} of them, and the ranking will be "
                   f"noisy. The source study's smallest cohort had 546.")
    if np.median(counts) > 50:
        out.append(
            f"median {np.median(counts):.0f} records per patient. Patient risk is the MAXIMUM "
            f"over a patient's records, so it saturates when there are hundreds and the "
            f"ranking stops discriminating. Group records at series or study level if you can."
        )

    if man.is_multilabel:
        per_class = man.labels.sum(axis=0)
    else:
        per_class = np.bincount(man.labels, minlength=man.num_classes)
    rare = [(man.class_names[i], int(c)) for i, c in enumerate(per_class) if c < 10]
    if rare:
        out.append(f"class(es) with fewer than 10 records: {rare}. A proxy model cannot learn "
                   f"them in one epoch, which inflates their atypicality.")
    if man.num_classes < 2:
        raise ValueError("only one label value: there is no task to be atypical with respect "
                         "to. Atypicality is defined relative to the clinical task.")
    return out


def summary(man: Manifest) -> dict:
    """Aggregate-only description, safe to print, log or hand to an AI assistant."""
    counts = np.bincount(man.patient_codes)
    if man.is_multilabel:
        per_class = man.labels.sum(axis=0).astype(int).tolist()
    else:
        per_class = np.bincount(man.labels, minlength=man.num_classes).tolist()
    return {
        "n_records": man.n_records,
        "n_patients": man.n_patients,
        "per_record_only": man.per_record,
        "records_per_patient": {
            "min": int(counts.min()), "median": float(np.median(counts)),
            "mean": round(float(counts.mean()), 2), "max": int(counts.max()),
        },
        "kind": man.kind,
        "task": "multi-label" if man.is_multilabel else "single-label",
        "classes": dict(zip(man.class_names, per_class)),
        "warnings": man.warnings,
    }


def build(
    out_path: str | Path,
    *,
    images: str | Path,
    labels_csv: str | Path | None = None,
    image_col: str | None = None,
    label_col: str | None = None,
    patient_col: str | None = None,
    suffix: str | None = None,
    recursive: bool = True,
) -> dict:
    """Write a manifest by joining a folder of files with an existing labels CSV.

    This exists so that pointing the tool at a new dataset needs no code. Two shapes are
    supported, which between them cover the public medical datasets:

    * a labels CSV with one row per record, joined to the files by filename stem;
    * no CSV at all, in which case the label is the name of the folder each file sits in
      (the torchvision ImageFolder convention) and there are no patient identifiers.
    """
    out_path = Path(out_path).expanduser().resolve()
    root = Path(images).expanduser().resolve()
    pattern = "**/*" if recursive else "*"
    allowed = {suffix.lower()} if suffix else IMAGE_SUFFIXES | ARRAY_SUFFIXES
    files = sorted(p for p in root.glob(pattern)
                   if p.is_file() and p.suffix.lower() in allowed)
    if not files:
        raise FileNotFoundError(f"no data files under {root}")

    rows: list[dict[str, str]] = []
    unmatched = 0
    if labels_csv:
        with open(labels_csv, newline="", encoding="utf-8-sig") as fh:
            table = list(csv.DictReader(fh))
        if not table:
            raise ValueError(f"{labels_csv} has no rows")
        cols = list(table[0])
        image_col = image_col or _guess(cols, ("image", "file", "name", "id", "path"))
        label_col = label_col or _guess(cols, ("target", "label", "class", "diagnos",
                                         "malign", "benign", "finding", "outcome",
                                         "grade", "severity"))
        patient_col = patient_col or _guess(cols, ("patient", "mrn", "subject", "person",
                                             "case", "participant"))
        for flag, name in (("--image-col", image_col), ("--label-col", label_col)):
            if name is None:
                raise ValueError(f"could not guess {flag} from columns {cols}; pass it")
        by_stem = {Path(str(r[image_col])).stem: r for r in table}
        for f in files:
            r = by_stem.get(f.stem)
            if r is None:
                unmatched += 1
                continue
            rows.append({
                "path": _relative(f, out_path.parent),
                "label": str(r[label_col]).strip(),
                "patient_id": str(r[patient_col]).strip() if patient_col else "",
            })
    else:
        for f in files:
            parent = f.parent.name if f.parent != root else ""
            if not parent:
                raise ValueError(
                    f"{root} has files directly in it and no labels CSV, so there is no label "
                    f"to use. Either pass --labels-csv or arrange files as <root>/<class>/*."
                )
            rows.append({"path": _relative(f, out_path.parent), "label": parent,
                         "patient_id": ""})

    if not rows:
        raise ValueError(
            f"none of the {len(files)} files matched a row in {labels_csv} by filename stem. "
            f"Check --image-col."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(REQUIRED))
        w.writeheader()
        w.writerows(rows)

    return {
        "manifest": str(out_path),
        "n_rows": len(rows),
        "files_found": len(files),
        "files_unmatched": unmatched,
        "columns_used": {"image": image_col, "label": label_col, "patient": patient_col},
        "patient_id_present": bool(patient_col),
    }


def _guess(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for h in hints:
        for c in columns:
            if h in c.lower():
                return c
    return None


def _relative(target: Path, start: Path) -> str:
    try:
        return target.relative_to(start).as_posix()
    except ValueError:
        return target.as_posix()
