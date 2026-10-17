# Validation

This package is a clean reimplementation of the screen used in the study, written to be
installable and readable rather than to share the research pipeline's machinery. That makes
"does it still do the same thing?" a question with an answer, so it was measured rather than
assumed.

## 1. It reproduces the research code on ISIC-2020

Same archive (33,126 dermoscopy images, 2,056 patients), same default budget (4 proxy
models, 1 epoch, ResNet-18, 20% flag rate), on one Quadro RTX 5000.

| | research pipeline | this package |
|---|---|---|
| patients / records | 2,056 / 33,126 | 2,056 / 33,126 |
| flagged | 411 | 411 |
| self-consistency | 0.822 (224 px), 0.787 (512 px) | **0.808** |
| flagged patients who are melanoma patients | 407 / 411 | **411 / 411** |
| melanoma images held by flagged patients | 561 / 579 | **561 / 584** |

Flag agreement with the research pipeline's own screen of the same cohort: **398 of 411
patients (96.8%)**, across a different implementation and a different input resolution.

Run-to-run: two runs of this package with identical settings and the same seed flagged
**410 of the same 411 patients** and scored self-consistency 0.808 and 0.788. The residual
difference is cuDNN's non-deterministic kernel selection, not sampling — the screen is
reproducible to about one patient in four hundred, not bit-exact.

Cost on this hardware: 16.5 minutes and 0.019 kWh for the first run, 15.5 minutes and
0.019 kWh once the record cache was held in memory, excluding the one-off 6 minutes spent
decoding 33,126 JPEGs. The study reports 7.4 GPU-minutes and 0.0074 kWh for its own
implementation of the same screen on the same data, so budget roughly twice that for this
package on a comparable card.

(The research figures come from the study's screen runs; the melanoma-image denominators
differ because the study's pool excludes planted control records.)

## 2. It reproduces the study's recall on a second dataset and a second modality

PTB-XL: 21,799 twelve-lead electrocardiograms from 18,869 patients, read straight from their
WFDB files through `examples/loaders/ptbxl_loader.py` — a different modality, a different
record format, labels built by rule rather than read from a column, and none of the study's
planted control records. The flags were then scored against the **study's own top-5% tail**
for that dataset's reference arm, the same endpoint the paper reports.

| | study's screen | this package |
|---|---|---|
| patients / records | 18,869 / 21,799 | 18,869 / 21,799 |
| flagged | 3,774 | 3,774 |
| hits in the 943-patient tail | 610 | **616** |
| recall | 0.6469 (CI 0.616–0.679) | **0.6532 (CI 0.623–0.684)** |
| lift over random flagging | 3.23x | **3.27x** |
| self-consistency | 0.847 | 0.740 |
| cost | 1.43 GPU-min, 0.0015 kWh | 1.3 GPU-min, 0.0015 kWh |

Rank agreement between the two screens' atypicality: Spearman 0.832. Flag overlap: 66.9%.
That gap is the honest part of this result — **the two runs recover the same tail while
disagreeing about a third of the 20% they flag**, because the cut falls in a crowded part of
a continuous ranking. The tail is reproducible; the exact membership of the flagged cohort
is not, and neither implementation can claim it is.

(Reproduce it with `examples/ptbxl_manifest.py` and the loader; scoring against the tail
needs the study's shadow-model dumps, which are not public.)

## 3. It recovers a tail it was never told about

`examples/make_synthetic.py` plants a continuum of legibility in a synthetic cohort and
records which patients sit in its illegible tail. The screen is given only the images and
the labels.

- default one-epoch proxies: self-consistency 0.10–0.18 → **the gate refuses the flags**,
  which is what it exists for (recall there is 55%, and the tool has no way to know that);
- `--epochs 3`: self-consistency 0.76–0.83, and the flags hold **70–87%** of the planted
  tail across three seeds, against the 20% random flagging would give.

This runs on a CPU in under a minute and is part of the test suite.

## 4. A small open-weight model can drive it

`examples/run_with_ollama.py` hands `SKILL.md` to an open model as a system prompt, lets it
emit one shell command at a time and feeds back the output. With **gemma4:31b** on a dataset
whose columns were deliberately unguessable (`subject_code`, `malignant`):

```
[1] dataconfessor --version
[2] ls ./derm
[3] head -1 ./derm/metadata.csv              # headers only, as the skill requires
[4] dataconfessor init ... --out manifest.csv            -> could not guess --label-col
[5] dataconfessor init ... --image-col image_name --label-col malignant \
        --patient-col subject_code                       -> 793 rows matched
[6] dataconfessor check manifest.csv
[7] dataconfessor screen ... --patients ... --report ... -> self-consistency 0.39, FAILED
[8] dataconfessor screen ... --epochs 3                  -> self-consistency 0.83, passed
[9] DONE + report
```

It recovered from the column-guess failure on its own, followed the escalation ladder when
the gate failed, and stopped at the first passing configuration. Recall against the planted
tail: 73%. **Discipline violations: 0** — it never opened the flagged-cohort CSV or any
record file, which the harness would have refused and logged.

## 5. The browser interface does the same work

`dataconfessor ui` was driven end to end with a real browser (Playwright against Chrome):
folder and labels CSV chosen, columns guessed from the CSV header (`study_file` / `finding` /
`mrn`), manifest built, cohort checked, screen run, gate read — self-consistency 0.814 on the
synthetic cohort, flags written, report opened. The same page was driven again in assistant
mode with **gemma4:31b** over Ollama Cloud: the model listed the folder, read the CSV header
only, ran `dataconfessor init`, noticed the tool's warning that no patient column had been
used, and stopped to ask whether `mrn` was the patient identifier instead of screening
per record. No page errors, and no attempt to open a record or the flagged cohort.

Two defects that test found, both fixed: the page sent the skill to the model without the
reply protocol (the model answered with nothing at all, silently), and a failed step left no
visible error. The protocol now lives next to the executor that enforces it.

## Reproducing these

```bash
pytest                                              # 3 and the unit checks
python examples/make_synthetic.py --out ./synth     # 2, by hand
# 1 needs the ISIC-2020 archive; see examples/isic2020.md
# 2 needs PTB-XL;      see examples/ptbxl_manifest.py
# 4 needs an Ollama API key: python examples/run_with_ollama.py --data ./synth
```
