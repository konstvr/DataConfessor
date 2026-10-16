# The manifest

One CSV. Three columns. Nothing else is configuration.

```csv
path,label,patient_id
images/ISIC_0015719.jpg,0,IP_3075186
images/ISIC_0052212.jpg,1,IP_3075186
```

- **path** — relative to the manifest file, or absolute. One record per row.
- **label** — the training label: a class index or a class name. For multi-label tasks, the
  positive classes separated by `;` (for example `cardiomegaly;effusion`); an empty string
  is not allowed, so use an explicit `none` class if a record has no positive label.
- **patient_id** — every record of one person carries the same value.

`dataconfessor init` writes this file for you by joining a folder of files with a labels
CSV on the filename stem, so you rarely write it by hand.

## The patient rule

Nothing downstream can check the grouping. If the key is wrong, every number the tool
produces will look plausible and mean nothing. A dataset in the source study once grouped
191,000 images under two "patients" because the key was the sex column.

- One person, many studies, many years → **one** `patient_id`.
- An accession number, a filename, a lesion id or a series UID is **not** a patient id.
- No identifier at all → pass `--no-patient-id`. Results are then per record, and say so.

## What counts as one record

Per-patient risk is the **maximum** over a patient's records, so the record definition
changes the answer.

| Record = | Records per patient | Consequence |
|---|---|---|
| a slice | hundreds | the maximum saturates; the ranking stops discriminating |
| a series or volume | about 3–20 | recommended |
| a study | 1–2 | per-patient and per-record become the same thing |

For MRI or CT, use one series per record, with a fixed field of view in millimetres so that
acquisition settings do not change anatomical coverage.

## File formats

Read natively:

- images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.webp` — resized to a square, native
  size capped at 224 px, never upsampled;
- arrays: `.npy` holding one record shaped `(channels, time)` for a signal.

All records must share a shape (images are resized for you; signals are not resampled).

## Custom loaders

For DICOM, NIfTI, WFDB or anything else, write a function that returns one record as an
array and pass it:

```python
# my_loader.py
import pydicom, numpy as np

def load(path):
    """One record -> (H, W) uint8 for images, or (channels, time) float32 for signals."""
    arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.ptp(), 1e-6) * 255
    return arr.astype(np.uint8)
```

```bash
dataconfessor screen manifest.csv --loader my_loader.py:load
```

Every record must come back with the same shape. A 3-D volume has to be reduced to one
2-D image or handled by a loader that returns a consistent shape — the proxies here are
2-D and 1-D only.

## Multi-record patients and memory

Records are decoded once into a memory-mapped cache next to the manifest
(`.dataconfessor_cache/`, or `--cache-dir`). A 33,000-record dermoscopy set at 224 px takes
about 5 GB there. It is loaded into memory when it fits (see `DATACONFESSOR_RAM_GB`
in `troubleshooting.md`), which is the difference between a GPU-bound run and a
disk-bound one. Delete the folder when you are done; it is rebuilt on demand.
