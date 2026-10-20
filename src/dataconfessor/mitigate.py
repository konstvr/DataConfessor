"""What to do about the flagged patients.

Two halves. `advise()` ranks the interventions the source study measured, filtered by the
training recipe the user describes. `teachers()` runs the one intervention that needs
computation rather than a config change: self-distillation, where each flagged record is
trained on the prediction of a teacher that never saw that patient instead of on its label.

The first recommendation is the one people reach for and it is a mistake: deleting the
flagged patients. Removing the most exposed layer of a dataset exposes the next one (the
privacy onion), and in the source study's dermoscopy cohort the flagged patients were almost
exactly the melanoma patients -- 407 of 411 flagged, holding 561 of 579 melanoma images. You
would delete the disease.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class Recommendation:
    title: str
    evidence: str
    action: str
    cost: str
    applies: bool = True


def advise(
    *,
    weight_decay: float | None = None,
    epochs: int | None = None,
    finetune: str | None = None,
    underfitted: bool | None = None,
) -> list[Recommendation]:
    """Rank the measured interventions for a described recipe.

    Every `evidence` line is a measurement from the source study, not a general principle.
    Where a number is dataset-specific it says so, because the effect sizes varied by up to
    a factor of two across datasets while their direction did not.
    """
    recs: list[Recommendation] = []

    recs.append(Recommendation(
        title="Raise weight decay to 1e-2",
        evidence="Reduced the exposed cohort 1.38-2.95x across four datasets, against 1.07x "
                 "between seed replicates of the same recipe, at 0.2-5.5% change in energy "
                 "per epoch and -0.021 to +0.038 held-out AUROC. It did not change WHICH "
                 "patients ranked highest (normalised stability 0.75-0.98).",
        action="Set weight_decay=1e-2 in your optimiser and re-check held-out performance. "
               "With AdamW this is decoupled decay; with Adam it is not, and on pretrained "
               "backbones the coupled version cost utility.",
        cost="free",
        applies=weight_decay is None or weight_decay < 1e-2,
    ))

    recs.append(Recommendation(
        title="Fine-tune less of a foundation model",
        evidence="On an ECG foundation model, training only the head exposed 22.5% of "
                 "patients, the last block plus head 30.4%, and full fine-tuning 54.6% -- "
                 "while held-out AUROC stayed within 0.01 of its best and training energy "
                 "fell 59-61%.",
        action="Freeze the backbone and train the last block plus the classifier. Check "
               "utility before and after; this is the one lever that cut exposure and energy "
               "together.",
        cost="saves energy",
        applies=finetune in (None, "full", "partial"),
    ))

    recs.append(Recommendation(
        title="Train for fewer epochs, or early-stop on held-out performance",
        evidence="Exposure rose with training duration: 10.9% of patients after two epochs "
                 "to 54.6% after twenty, on the same recipe and data.",
        action="Pick the epoch at which held-out performance first plateaus rather than the "
               "last epoch of the schedule, and report the exposure at that checkpoint.",
        cost="saves energy",
        applies=epochs is None or epochs > 5,
    ))

    recs.append(Recommendation(
        title="Fix an under-fitted model before treating it as a privacy problem",
        evidence="Patient exposure correlated with the train-minus-held-out gap "
                 "(Spearman 0.79). Retraining an under-fitted dermoscopy model at the "
                 "published resolution raised AUROC by 0.174 and cut the exposed fraction "
                 "from 70.8% to 45.2% -- but cost 8.5x the training energy.",
        action="Compare your held-out performance with the published figure for this task. A "
               "model well below it leaks more AND performs worse.",
        cost="8.5x energy in the measured case",
        applies=underfitted is not False,
    ))

    recs.append(Recommendation(
        title="Self-distil the flagged records (dataconfessor teachers)",
        evidence="Training flagged records on a teacher's prediction instead of their label "
                 "lowered flagged patients' mean risk from 0.85 to 0.70 (-0.15, 95% CI -0.16 "
                 "to -0.14; seed replicates moved -0.004) without raising unflagged "
                 "patients' risk, at -0.035 held-out AUROC and -0.044 sensitivity at 90% "
                 "specificity. It lowered AVERAGE risk only: worst-case leakage, the attack's "
                 "true-positive rate at 0.1% false-positive rate, did not fall.",
        action="Run `dataconfessor teachers` to produce soft labels for the flagged records, "
               "then train on those labels instead of the true ones for those records only.",
        cost="two extra training runs",
    ))

    recs.append(Recommendation(
        title="Do NOT delete the flagged patients",
        evidence="Removing the most exposed records exposes the next layer (the privacy "
                 "onion effect). In the dermoscopy cohort the flagged patients were almost "
                 "exactly the melanoma patients, so deleting them would remove the disease "
                 "the model exists to detect.",
        action="Keep every patient in the training data and change how the flagged records "
               "are learned.",
        cost="-",
    ))

    recs.append(Recommendation(
        title="Spend a differential-privacy budget where it is needed",
        evidence="Screening does not provide a guarantee. What it provides is a target: DP "
                 "and consent review are expensive, and the flags say where to spend them.",
        action="If you need a formal guarantee, use DP-SGD. Use the flags to decide the "
               "budget and to prioritise consent review, not as a substitute.",
        cost="utility, usually substantial",
    ))

    recs.append(Recommendation(
        title="Avoid mixup and small training sets if privacy matters",
        evidence="Not every regulariser helps: mixup exposed 38% more examples, and a "
                 "training set a quarter of the size exposed 33% more.",
        action="Treat 'it regularises, so it must help privacy' as false until measured.",
        cost="-",
    ))

    return recs


def advice_markdown(recs: list[Recommendation], *, include_all: bool = False) -> str:
    lines = ["## Recommended mitigations", ""]
    for i, r in enumerate([r for r in recs if r.applies or include_all], start=1):
        lines += [f"### {i}. {r.title}", "",
                  f"- **What the study measured:** {r.evidence}",
                  f"- **Do:** {r.action}",
                  f"- **Cost:** {r.cost}", ""]
    lines += [
        "Every figure above is from the source study's datasets. Directions held across "
        "datasets; magnitudes did not, so measure the effect on your own model rather than "
        "quoting these numbers as your result.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
# Self-distillation: the one mitigation that needs computation rather than a config change
# ------------------------------------------------------------------------------------------


def read_flagged(patients_csv: str | Path) -> set[str]:
    """The flagged patient ids, read from the screen's sensitive per-patient CSV.

    The tool reads this file; an AI assistant driving the tool must not. It is the one
    artefact that identifies individuals by privacy risk.
    """
    import csv

    with open(patients_csv, newline="", encoding="utf-8") as fh:
        return {row["patient_id"] for row in csv.DictReader(fh) if row["flagged"] == "1"}


def teachers(
    man,
    flagged: set[str],
    *,
    epochs: int,
    arch: str | None = None,
    pretrained: bool | None = None,
    size: int | None = None,
    lr: float | None = None,
    device: str | None = None,
    cache_dir: str | None = None,
    loader: str | None = None,
    seed: int = 7_000_000,
    out_path: str | Path = "soft_labels.npz",
    progress=None,
) -> dict[str, Any]:
    """Train two teachers on complementary patient halves and emit soft labels.

    Each flagged record gets the predicted distribution of the teacher that did **not** train
    on its patient. That is what makes this a privacy intervention rather than ordinary label
    smoothing: the soft label carries what the rest of the cohort says about this record, and
    nothing about the record itself. Fixed softened labels (0.75 on the true class) were
    measured in the same study and gave no protection at all (+0.003 risk).

    The teachers must be real models, not the one-epoch proxies the screen trains -- pass the
    epoch count of your production recipe.
    """
    from . import models as _models
    from .data import RecordSet
    from .energy import Meter, device_name
    from .screen import _say, fit

    progress = progress or _say

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = RecordSet(man, size=size, cache_dir=cache_dir, loader=loader, device=dev,
                        progress=progress)
    if arch is None:
        arch = _models.choose_arch(man.kind, records.input_size)
    if pretrained is None:
        pretrained = arch == "resnet18"
    if lr is None:
        lr = 1e-4 if pretrained else 1e-3

    codes = np.asarray(man.patient_codes)
    rng = np.random.default_rng(seed)
    uniq = np.unique(codes)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    side_a = np.isin(codes, shuffled[: len(shuffled) // 2])

    unknown = flagged - set(man.patient_names)
    if unknown:
        raise ValueError(
            f"{len(unknown)} flagged patient id(s) are not in this manifest; the flags and "
            f"the manifest do not belong to the same dataset."
        )
    flagged_codes = np.array([i for i, n in enumerate(man.patient_names) if n in flagged],
                             dtype=np.int64)
    if not len(flagged_codes):
        raise ValueError("no flagged patients found in the CSV")
    is_flagged_record = np.isin(codes, flagged_codes)

    probs = np.zeros((len(records), man.num_classes), dtype=np.float32)
    with Meter(str(dev)) as meter:
        for t, in_mask in enumerate((side_a, ~side_a)):
            progress(f"  teacher {t + 1}/2 ({int(in_mask.sum()):,} records, {epochs} epochs)")
            model = fit(records, np.flatnonzero(in_mask), arch=arch,
                        pretrained=bool(pretrained), epochs=epochs, lr=lr, seed=seed + t,
                        device=dev)
            # Score only the records this teacher did NOT train on and which are flagged.
            targets = np.flatnonzero(~in_mask & is_flagged_record)
            with torch.no_grad():
                for start in range(0, len(targets), 128):
                    idx = targets[start:start + 128]
                    x, _ = records.batch(idx)
                    logits = model(x).float()
                    p = (torch.sigmoid(logits) if man.is_multilabel
                         else torch.softmax(logits, dim=1))
                    probs[idx] = p.cpu().numpy()
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    progress(f"  two teachers trained: {meter.result.text()}")

    out_path = Path(out_path)
    rows = np.flatnonzero(is_flagged_record)
    np.savez_compressed(
        out_path,
        path=np.array([str(man.paths[i]) for i in rows]),
        soft_label=probs[rows],
        class_names=np.array(man.class_names),
        multilabel=man.is_multilabel,
    )
    return {
        "soft_labels_written_to": str(out_path),
        "n_flagged_records": int(len(rows)),
        "n_flagged_patients": int(len(flagged_codes)),
        "teacher_epochs": epochs,
        "teacher_arch": arch,
        "device": device_name(str(dev)),
        "energy": meter.result.as_dict(),
        "sensitive": "This file lists the flagged records by path. Keep it local.",
        "how_to_use": "In your training loop, for a record whose path is in this file, "
                      "replace the one-hot target with its soft_label row and keep every "
                      "other record on its true label. Train the rest of the recipe "
                      "unchanged.",
    }
