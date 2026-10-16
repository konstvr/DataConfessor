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

(The research figures come from the study's screen runs; the melanoma-image denominators
differ because the study's pool excludes planted control records.)

## 2. It recovers a tail it was never told about

`examples/make_synthetic.py` plants a continuum of legibility in a synthetic cohort and
records which patients sit in its illegible tail. The screen is given only the images and
the labels.

- default one-epoch proxies: self-consistency 0.18 → **the gate refuses the flags**, which
  is the behaviour the gate exists for;
- `--epochs 3`: self-consistency 0.76–0.83, and the flags contain **73–87%** of the planted
  tail (random flagging would give 20%).

This runs on a CPU in under a minute and is part of the test suite.

## 3. A small open-weight model can drive it

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

## Reproducing these

```bash
pytest                                              # 2 and the unit checks
python examples/make_synthetic.py --out ./synth     # 2, by hand
# 1 needs the ISIC-2020 archive; see examples/isic2020.md
# 3 needs an Ollama API key: python examples/run_with_ollama.py --data ./synth
```
