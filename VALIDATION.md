# Validation

A clean reimplementation of the screen used in the study, checked against the research
pipeline it replaces.

## It reproduces the research code on ISIC-2020

Same archive (33,126 dermoscopy images, 2,056 patients), same default budget (4 proxy
models, 1 epoch, ResNet-18, 20% flag rate), on one Quadro RTX 5000.

| | research pipeline | this package |
|---|---|---|
| patients / records | 2,056 / 33,126 | 2,056 / 33,126 |
| flagged | 411 | 411 |
| self-consistency | 0.822 | **0.808** |
| flagged patients who are melanoma patients | 407 / 411 | **411 / 411** |

Flag agreement with the research pipeline's screen of the same cohort: **398 of 411
patients (96.8%)**. Two runs of this package with the same seed flagged 410 of the same 411
patients; the residual difference is cuDNN's kernel selection, not sampling.

Cost: 16.5 minutes and 0.019 kWh, excluding a one-off 6 minutes spent decoding the JPEGs.

## It recovers a tail it was never told about

`examples/make_synthetic.py` plants a known illegible tail in a synthetic cohort; the screen
is given only the images and the labels. One-epoch proxies fail the self-consistency check,
the fail-safe retrains at two epochs without being asked, and the flags then recover the
planted tail well above what random flagging would give. This runs on a CPU in under a
minute and is part of the test suite.

## A small open-weight model can drive it

`examples/run_with_ollama.py` hands `SKILL.md` to an open model as a system prompt and lets
it emit one shell command at a time. With **gemma4:31b**, on a dataset whose columns were
deliberately unguessable, it screened the cohort in six commands, read the
`passed_after_retry` verdict correctly, gave the caveats in its own words, and asked for the
training recipe before offering mitigations. **Discipline violations: 0** — it never opened
the flagged-cohort CSV or a record file.

## The browser interface does the same work

`dataconfessor ui` was driven end to end with a real browser: folder and labels CSV chosen,
columns guessed from the header, manifest built, cohort checked, screen run, the gate failed
at one epoch and passed on the retry, flags written, report opened, no page errors. The same
page in assistant mode over Ollama Cloud: the model read the CSV header only, noticed the
tool's warning that no patient column had been used, and stopped to ask rather than
screening per record.
