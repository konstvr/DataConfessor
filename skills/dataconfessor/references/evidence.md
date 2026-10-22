# Where each number comes from

Every figure the tool prints is a measurement from the source study — Vrettos et al.,
*Predicting and mitigating privacy risks in medical AI* — on its datasets, not a prediction
for yours. This file says what was measured so that nothing is quoted beyond its warrant.

## The study

More than 7,000 models, six clinical datasets, four modalities: CheXpert and NIH-CXR14
(chest radiographs), ISIC-2020 (dermoscopy), PTB-XL (12-lead ECG), SCOPE-MRI (shoulder MRI)
and PneumoniaMNIST (paediatric chest radiographs, no patient identifiers). For each training
recipe, 32–128 shadow models were each trained on a random half of the patients; a record's
attack AUC is the probability that a model that trained on it is more confident about it
than one that did not, and a patient's risk is the maximum over their records.

## The claims the screen rests on

| Claim | Measurement |
|---|---|
| Training sets *how many* are exposed | weight decay 1e-4 → 1e-2 cut the exposed cohort 1.38–2.95x; reseeding the same recipe changed it 1.07x |
| The data set *which* are exposed | normalised stability of the patient ranking 0.75–1.00 across weight decay, architecture, initialisation, normalisation, epoch count and foundation-model fine-tuning |
| Atypicality identifies them | Cohen's d 1.78–4.18 separating the top-5% most exposed patients from the rest, on all six datasets, all q < 1e-12 |
| Four models are enough | 4 models recover 95–97% of the 128-model effect; one epoch recovers 89–98% |
| The proxy need not match | a ~25k-parameter SmallCNN predicted a ResNet-50's tail at 81% of the ResNet-50's own effect |

## Recall, and what "73–100%" means

One screen per dataset was run end to end and scored against the top-5% most exposed
patients of every audited recipe that passed the canary control, with the planted control
records themselves excluded from the tail. At a 20% flag rate, recall of that tail against
each dataset's reference recipe was:

| Dataset | Recall | Lift over random |
|---|---|---|
| ISIC-2020 | 100.0% | 5.0x |
| NIH-CXR14 | 91.5% | 4.6x |
| CheXpert | 75.2% | 3.8x |
| PTB-XL | 72.7% | 3.6x |
| SCOPE-MRI (5-epoch proxies) | 100% (27/27, 95% CI 87–100) | 5.0x |

A reseeded run of the same recipe recovered 73–100%. Flags transferred to recipes the screen
had never seen — different weight decay, normalisation or architecture — at **65–100%**, the
low end being group normalisation on PTB-XL. Screening cost 1.4–7.4 GPU-minutes and
0.0015–0.085 kWh, 37–1,200 times less energy than a full audit of the same model.

**These are other datasets' numbers.** A screen on a new dataset has no ground truth, so its
own recall is unknown. Quote the range as prior experience, never as this run's performance.

## The self-consistency floor

Split-half rank agreement measured on the study's datasets: CheXpert 0.905, NIH-CXR14
0.892, PTB-XL 0.827, ISIC-2020 0.822, PneumoniaMNIST 0.496–0.635, SCOPE-MRI 0.690 at one
epoch. The four that passed all delivered usable recall; their consistency did **not** rank
their recall (Spearman −0.40 at n = 4, i.e. noise). So it is a necessary condition, not a
performance predictor. The 0.70 threshold sits between the two observed clusters: a
judgement call on five datasets, not a derived constant.

## Mitigations

| Intervention | Effect | Cost |
|---|---|---|
| weight decay 1e-2 | exposed cohort 1.38–2.95x smaller | 0.2–5.5% energy, −0.021 to +0.038 AUROC |
| partial fine-tuning of a foundation model | exposure 54.6% → 30.4% (last block) or 22.5% (head only) | 59–61% less energy, AUROC within 0.01 |
| fewer epochs | exposure 54.6% at 20 epochs → 10.9% at 2 | less energy, less utility |
| fixing an under-fitted model | exposure 70.8% → 45.2%, AUROC +0.174 | 8.5x energy in that case |
| self-distillation of flagged records | flagged patients' mean risk 0.85 → 0.70; unflagged unchanged | −0.035 AUROC, −0.044 sensitivity at 90% specificity |
| fixed softened labels (0.75 on the true class) | +0.003 — no protection | — |
| class-wise histogram matching | −0.20 risk | AUROC collapses to 0.51 |
| mixup | 38% **more** exposed | — |
| quarter-size training set | 33% **more** exposed | — |

Self-distillation lowered *average* risk only: the attack's true-positive rate at a 0.1%
false-positive rate did not fall. It complements formal guarantees such as differential
privacy; it does not replace them.

## Who is over-represented

Patients aged 80 or older were over-represented among exposed patients on all four datasets
analysed (odds ratios 1.83–6.45 across 23 training variations). Men were over-represented on
two datasets, patients under 40 on shoulder MRI. No racial or ethnic group was significantly
over-represented. Exposure is not evenly distributed, so a screen is also an equity question.
