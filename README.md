<p align="center">
  <img src="assets/icon.jpg" alt="DataConfessor" width="280">
</p>

<h1 align="center">DataConfessor</h1>

<p align="center"><b>Find the patients your model will memorise — before you train it.</b></p>

A model trained on medical records can reveal who was in its training set. Measuring that
usually means training hundreds of shadow models and attacking your own model, after the
fact. DataConfessor works the other way round: *which* patients are at risk is a property of
the data, so it can be screened during pre-processing. It trains four small models for one
epoch each on complementary halves of your patients, scores every record with the models
that never saw it, and flags the 20% least predictable — the patients a model can only fit
by memorising.

In the study behind the tool (more than 7,000 models, six clinical datasets, four
modalities) that flag rate contained 65–96% of the patients a full membership-inference
audit placed in the top-5% risk tail, for 1.4–7.4 GPU-minutes. It is a ranking, not a
leakage measurement, and an unflagged patient is not certified safe.

## Install

```bash
pip install git+https://github.com/konstvr/DataConfessor.git
dataconfessor --version
```

Python 3.10 or newer. From a clone, `pip install .` does the same. PyTorch and the usual
scientific stack are pulled in by pip; a GPU makes a screen minutes rather than an hour, but
is not required.

## Use it with an AI assistant

The repository is also a Claude Code plugin, so an assistant can run the whole screen from a
sentence, including working out how your dataset maps onto the manifest:

```
/plugin marketplace add konstvr/DataConfessor
/plugin install dataconfessor@dataconfessor
```

```
/dataconfessor my dataset is at /data/derm, labels are in metadata.csv
```

Other ways in:

| | |
|---|---|
| any Claude Code project | `cp -r skills/dataconfessor ~/.claude/skills/` |
| claude.ai or the API | zip `skills/dataconfessor/` and upload it as an Agent Skill |
| any other agent | point it at `skills/dataconfessor/SKILL.md` — plain Markdown |

The pip package carries its own copy of the skill, so the browser interface can hand it to a
model with no checkout present. The skill covers how to map a dataset onto the manifest, how
to read the verdict, what to do when the check fails, and that the flagged-patient list and
the records themselves are not to be opened. An open 31B model followed it end to end; see
`VALIDATION.md`.

## Or run it yourself

```bash
dataconfessor init  /data/images --labels-csv /data/labels.csv --out manifest.csv
dataconfessor screen manifest.csv --patients flagged.csv --report report.html
```

```
screened 2,056 patients / 33,126 records
  4 proxy models x 1 epoch(s) of resnet18 (ImageNet init) on cuda, 16.5 minutes, 0.0191 kWh
  flagged 411 patients (20%)
  self-consistency 0.808 (floor 0.7): the ranking reproduces across independent proxy
  halves. Necessary for the flags to mean anything -- it does NOT predict recall.
```

<sup>ISIC-2020 on one Quadro RTX 5000. All 411 flagged patients were melanoma patients,
holding 561 of the 584 melanoma images in the archive — which is why the flags are not a
deletion list.</sup>

## Or run it in a browser

```bash
dataconfessor ui
```

<p align="center"><img src="assets/screenshot.png" alt="The DataConfessor interface: three
steps on the left, the self-consistency verdict and the flagged share on the right" width="860"></p>

A local page with two ways in: a chat that drives the tool with a model of your choice, or a
three-step form with a folder picker and no commands to type. The server binds `127.0.0.1`
and runs only this tool's commands; the flagged-cohort CSV is written to disk and is not
served to the page.

The chat supports Anthropic, any OpenAI-compatible endpoint, and Ollama. A local Ollama
(Base URL `http://localhost:11434`) needs no API key and keeps the conversation on the
machine; hosted providers, Ollama Cloud included, need a key and receive what the assistant
sends them — dataset paths, the commands it runs and their aggregate output. Records and the
flagged-patient list are never part of that.

## The input: one CSV

```csv
path,label,patient_id
images/ISIC_0015719.jpg,0,IP_3075186
```

`dataconfessor init` writes it from a folder of files plus whatever labels CSV the dataset
came with, matching on filename. Images (`.png/.jpg/...`) and `.npy` arrays are read
natively; DICOM, NIfTI and WFDB go through a short `--loader` hook. The one rule nothing
downstream can check: **every record of one person must carry the same `patient_id`**. See
`skills/dataconfessor/references/manifest.md`.

## The self-consistency check

Split the proxy models into two halves, rank patients from each, correlate. Below **0.70**
the halves disagree and the ranking is not reproducible — usually because the proxies are
under-trained, so the tool retrains them at 2 then 5 epochs (`--max-epochs`) and stops at
the first budget that clears the floor. Each run ends with a verdict:

| | |
|---|---|
| `passed` | cleared the floor on the first attempt |
| `passed_after_retry` | cleared it after longer training |
| `below_floor` | never cleared it; the flags are written anyway and are weaker |

The verdict appears in the summary JSON, the report, and a column of the patient CSV. Exit
code is 0 whenever flags were written; `--strict` returns 3 on `below_floor`.

## Commands

| | |
|---|---|
| `dataconfessor init` | build `manifest.csv` from a folder + a labels CSV |
| `dataconfessor check` | validate the manifest and describe the cohort |
| `dataconfessor screen` | rank patients, flag the top 20%, write the report |
| `dataconfessor teachers` | soft labels for the flagged records (self-distillation) |
| `dataconfessor advise` | the mitigations measured in the study, ranked for your recipe |
| `dataconfessor ui` | the browser interface, served from this machine only |

Outputs are separated by sensitivity: `--out` is aggregate-only JSON, safe to log or hand to
an assistant; `--patients` is the flagged cohort, which identifies individuals by risk and
stays local; `--report` is a self-contained HTML file with no identifiers. Every run reports
its GPU energy (NVML, board only, PUE 1.0) in all three.

## After the screen

`dataconfessor advise` lists what the study measured: weight decay 1e-2 cut the exposed
cohort 1.38–2.95x at no cost; partial fine-tuning of a foundation model cut exposure from
55% to 22–30% and used 60% less energy; mixup and smaller training sets made it worse.

`dataconfessor teachers` trains two teachers on complementary patient halves and writes the
prediction of the teacher that never saw each flagged record, to train that record on a soft
label instead of its own. In the study this lowered flagged patients' mean risk from 0.85 to
0.70 for 0.035 of held-out AUROC, and did not lower worst-case leakage — it complements
differential privacy rather than replacing it.

## Citing this

> Vrettos, K., Tzanis, E., Kondylakis, H., Vassalou, E. E., Doo, F. X. & Klontzas, M. E.
> *Predicting and mitigating privacy risks in medical AI.*

See `CITATION.cff`. Licensed under Apache-2.0 (`LICENSE`, `NOTICE`).
