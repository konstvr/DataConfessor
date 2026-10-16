<p align="center">
  <img src="assets/icon.jpg" alt="DataConfessor" width="320">
</p>

<h1 align="center">DataConfessor</h1>

<p align="center"><b>Find the patients your model will memorise — before you train it.</b></p>

A model trained on medical records can reveal who was in its training set. That risk is
usually measured after the fact, by training hundreds of shadow models and attacking your
own model. DataConfessor measures the other half of the problem instead: *which* patients
are at risk is a property of the data, so it can be screened during pre-processing, in
minutes, on four throwaway models, before a production model exists.

```bash
pip install dataconfessor
dataconfessor init  /data/images --labels-csv /data/labels.csv --out manifest.csv
dataconfessor screen manifest.csv --patients flagged.csv --report report.html
```

```
screened 2,056 patients / 33,126 records
  4 proxy models x 1 epoch(s) of resnet18 (ImageNet init) on cuda, 7.1 minutes, 0.0074 kWh
  flagged 411 patients (20%)
  self-consistency 0.812 (floor 0.7): the ranking reproduces across independent proxy
  halves. Necessary for the flags to mean anything -- it does NOT predict recall.
  NOT a leakage measurement. Unflagged does not mean safe.
```

## What it is

The screen trains four small models for one epoch each on complementary halves of your
patients, so every record is scored by the two models that never saw it. A record the rest
of the cohort cannot predict is one a model can only fit by **memorising** — that is
atypicality, and it is what puts a patient in the exposed tail. The 20% most atypical
patients are flagged.

In the study behind this tool — more than 7,000 models over six clinical datasets and four
modalities — that flag rate recovered **65–96%** of the patients a full membership-inference
audit placed in the top-5% risk tail, 3.2–4.8 times random flagging, in 1.4–7.4 GPU-minutes
and 37–1,200 times less energy than the audit it replaces.

## What it is not

- **Not a leakage measurement.** It produces a ranking and a flagged cohort, never a
  probability that a patient is exposed.
- **Not a safety certificate.** An unflagged patient is not shown to be protected; part of
  the tail is always missed.
- **Not a reason to delete anyone.** Removing the most exposed patients exposes the next
  layer, and in one dataset the flagged patients were almost exactly the melanoma patients.
  Keep them and change how they are *learned* — see `dataconfessor teachers`.

## Try it in a minute, on data you can throw away

```bash
python examples/make_synthetic.py --out ./synth      # a cohort with a planted atypical tail
dataconfessor screen ./synth/manifest.csv --epochs 10 --patients ./synth/flagged.csv \
    --report ./synth/report.html
```

Then compare `synth/flagged.csv` with `synth/atypical_truth.txt`. Run it again with
`--epochs 1` to watch the self-consistency gate refuse to produce flags from proxies that
have not learned anything.

## The input: one CSV

```csv
path,label,patient_id
images/ISIC_0015719.jpg,0,IP_3075186
```

`dataconfessor init` writes it for you from a folder of files plus whatever labels CSV the
dataset came with, matching on filename. Images (`.png/.jpg/...`) and `.npy` arrays are read
natively; DICOM, NIfTI and WFDB go through a four-line `--loader` hook. The one rule nothing
downstream can check is that **every record of one person carries the same `patient_id`** —
see `skills/dataconfessor/references/manifest.md`.

## The gate

The only validity check available on a dataset nobody has audited: split the proxy models
into two independent halves, rank patients from each, correlate. Below **0.70** the two
halves disagree, the ranking is noise, and the flags must not be used — train longer
(`--epochs 3`, then `5`), then use more proxies (`--models 8`), then look at the data. The
tool says so in its output and in the report; it does not fail quietly and it does not fail
loudly in the exit code either, so read the value.

## Commands

| | |
|---|---|
| `dataconfessor init` | build `manifest.csv` from a folder + a labels CSV |
| `dataconfessor check` | validate the manifest, describe the cohort — free and instant |
| `dataconfessor screen` | rank patients, flag the top 20%, write the report |
| `dataconfessor teachers` | soft labels for the flagged records (self-distillation) |
| `dataconfessor advise` | the measured mitigations, ranked for your recipe |

Two outputs, deliberately separated: `--out` is aggregate-only JSON, safe to log, ticket or
hand to an AI assistant; `--patients` is the flagged cohort, which identifies individuals by
privacy risk and stays on your machine. `--report` is a single self-contained HTML file with
three diagnostic panels and no identifiers.

## With an AI assistant

The repo ships a skill, so an assistant can run the whole screen from a sentence:

```bash
cp -r skills/dataconfessor ~/.claude/skills/          # Claude Code
```

```
/dataconfessor my dataset is at /data/derm, labels are in metadata.csv
```

The skill tells the assistant how to map an unfamiliar dataset onto the manifest, how to
read the gate, how to escalate when it fails, and — importantly — not to open the flagged
cohort or the records themselves. It works with any agent that can read a Markdown
instruction file and run shell commands; see `skills/dataconfessor/SKILL.md`.

## After the screen: protecting the flagged patients

`dataconfessor advise` ranks what the study measured, filtered by your recipe. The headline
ones: raising weight decay to 1e-2 cut the exposed cohort 1.38–2.95x for free; fine-tuning
only part of a foundation model cut exposure from 55% to 22–30% *and* used 60% less energy;
mixup and smaller training sets made things worse.

`dataconfessor teachers` runs the one mitigation that needs computation. It trains two
teachers on complementary patient halves and writes the prediction of the teacher that never
saw each flagged record, so you can train that record on a soft label instead of its true
one. In the study this lowered flagged patients' mean risk from 0.85 to 0.70 for 0.035 of
held-out AUROC, without raising anyone else's risk — and it did **not** lower worst-case
leakage, so it complements differential privacy rather than replacing it.

## Citing this

> Vrettos, K., Tzanis, E., Kondylakis, H., Vassalou, E. E., Doo, F. X. & Klontzas, M. E.
> *Predicting and mitigating privacy risks in medical AI.*

See `CITATION.cff`. MIT licensed.
