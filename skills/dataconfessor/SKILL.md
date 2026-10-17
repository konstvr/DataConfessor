---
name: dataconfessor
description: Run a pre-training privacy screen on a medical dataset. Use when the user says /dataconfessor, or asks to screen a dataset for privacy risk, find which patients are at risk before training, flag vulnerable or exposed patients, check membership-inference risk, find out which records a model will memorise, or plan consent review, a privacy budget or a targeted defence for a training set. Drives the `dataconfessor` command line tool, which ranks patients by atypicality using four throwaway proxy models and flags the top 20%.
version: 0.1.0
---

# DataConfessor: screen a dataset before training

The tool ranks patients by **atypicality** — how poorly they are predicted by small models
that never trained on them — and flags the top 20%. Atypicality is a property of the data,
not of the model the user is about to train, so this runs before any production model
exists. Four proxy models, one epoch each, minutes on one GPU.

## Rules

1. **Never open the per-patient CSV** (`--patients`). It ranks individuals by privacy risk.
   Do not read it, print it, summarise its rows, or send it anywhere. Report its path.
2. **Never open the data files.** You need the folder layout and the column names of any
   labels CSV, not the records. `head -1 labels.csv` is enough; never `head` the image data.
3. **Never call the output a measurement.** It is a ranking and a flagged cohort. An
   unflagged patient is not safe.
4. **Never recommend deleting the flagged patients.** Removing the most exposed layer
   exposes the next one, and the flagged patients are often the disease cases.
5. Run every command yourself and report what it printed. Do not predict results.

## Step 1 — install, if needed

```bash
dataconfessor --version
```

If that fails, install it: `pip install dataconfessor`, or `pip install -e .` from a
checkout of the repository. It needs PyTorch; a GPU makes it minutes instead of an hour.

## Step 2 — find out what the dataset looks like

Ask the user for the dataset path if they did not give one. Then look at **structure only**:

```bash
ls <data_dir>
find <data_dir> -maxdepth 2 -type d | head -20
ls <data_dir> | head -5                 # file names, to see the format
head -1 <data_dir>/<labels>.csv         # column names only, never the rows
```

You need three things. If any is missing, ask the user:

| Need | Usually |
|---|---|
| the folder holding the images or `.npy` files | given by the user |
| the training label of each record | a column in a CSV, or the parent folder name |
| the **patient identifier** of each record | a column in a CSV |

If there is no patient identifier, say so: results will be per record, not per patient, and
a patient with several records will not be scored as one person.

## Step 3 — build the manifest

```bash
dataconfessor init <data_dir> --labels-csv <labels.csv> --out manifest.csv
```

It guesses the columns and prints which it used. If a guess is wrong, re-run with
`--image-col <col> --label-col <col> --patient-col <col>`. With no labels CSV it uses each
file's parent folder as the label. For formats it cannot read (DICOM, NIfTI, WFDB) see
`references/manifest.md`.

## Step 4 — check the cohort

```bash
dataconfessor check manifest.csv
```

Read the output back to the user: patients, records, records per patient, classes. **Report
every warning.** The grouping warnings matter most — a wrong patient key cannot be detected
later and silently invalidates everything.

## Step 5 — screen

```bash
dataconfessor screen manifest.csv \
    --patients flagged_patients.csv \
    --out screen_summary.json \
    --report screen_report.html
```

Add `--device cuda` if the machine has a GPU and the tool did not pick it up.

Before you launch it, tell the user what it will cost: the record count from step 4, that
the first run decodes every record into a cache of roughly `records x size x size x channels`
bytes, and that the screen itself takes minutes on a GPU and up to an hour on a CPU. Then
run it and wait. Do not run it in the background and report before it has finished.

## Step 6 — the self-consistency gate

**This decides whether the flags may be used.** Read `self_consistency` from the printed
output (or from `screen_summary.json`, field `self_consistency_passed`).

| self-consistency | What it means | What you do |
|---|---|---|
| ≥ 0.70 | two independent halves of the proxy models rank patients the same way | report the results, go to step 7 |
| < 0.70 | the halves disagree; the ranking is noise | the flags are **invalid**. Escalate, below. |
| `nan` | fewer than four proxy models | re-run with `--models 4` |

The exit code is 0 either way, so never use the exit code as the verdict. The patient CSV
is written even on failure; it is invalid.

**Escalation, one change at a time, reporting each attempt with its value and its cost:**

1. `--epochs 3`, then `--epochs 5`. The usual cause is proxies too weak to rank anyone. On
   shoulder MRI one-epoch proxies scored 0.690 and five-epoch proxies passed.
2. `--models 8`. On ECG data this raised 0.79 to 0.90 for twice the compute.
3. Check the data: is the patient column really a person? Are records near-duplicates? Are
   there enough records per class?
4. **Stop after three failed re-runs.** Do not keep trying combinations. Tell the user the
   screen cannot rank patients reliably on this dataset and that a full shadow-model audit
   is the remaining option. Datasets with no patient structure fail by nature.

Do not lower the threshold, do not hand-pick patients, do not present failed flags.

## Step 7 — report

Give the user:

- patients screened and flagged, and the proxy budget (models × epochs) and minutes;
- the self-consistency value and the verdict;
- the path of `screen_report.html` (figures, no identifiers) and of the patient CSV
  (sensitive, which you have not opened);
- these two caveats, in your own words: it is not a leakage measurement, and an unflagged
  patient is not certified safe — at this flag rate the source study recovered 65–96% of
  the tail a full audit found, so part of it is always missed;
- what the flags are for: allocating a privacy budget, targeting consent review, targeting
  an audit, targeting a training-time defence.

## Step 8 — mitigations, if the user wants them

Ask what their training recipe is (weight decay, epochs, whether they fine-tune a
foundation model), then:

```bash
dataconfessor advise --weight-decay 1e-4 --epochs 20 --finetune full
```

Pass only what the user told you; omit the rest. Present the output as what the source
study measured on its datasets, not as a prediction for theirs.

To apply the strongest measured mitigation, self-distillation of the flagged records:

```bash
dataconfessor teachers manifest.csv --flagged flagged_patients.csv --epochs <their epochs>
```

`--epochs` must be their production recipe's epoch count, not the proxy's. It writes
`soft_labels.npz`, which they substitute for the true labels of those records in their own
training loop. It lowered flagged patients' mean risk by 0.15 for 0.035 held-out AUROC in
the source study, and it did **not** lower worst-case leakage.

## If the user would rather click than talk

`dataconfessor ui` opens a local page with the same workflow as a form: a folder picker,
column mapping, the gate verdict and the report. Offer it when the user is uncomfortable
with the command line, or wants to run the screen again later without you.

## References

- `references/manifest.md` — the manifest format, custom loaders, what counts as one record.
- `references/troubleshooting.md` — errors, gate failures, slow runs.
- `references/evidence.md` — what each number in the tool's output comes from.
