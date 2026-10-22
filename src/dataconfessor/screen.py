"""The screen: rank patients by atypicality before a production model exists.

A membership-inference audit tells you who leaked *after* you trained the model, and costs a
population of shadow models. The screen inverts that. Vulnerability tracks **atypicality** --
how poorly a record is predicted by models that never trained on it -- and atypicality is a
property of the data, not of the model you are about to train. So a handful of throwaway
proxy models can rank patients before the production run starts.

Three measurements in the source study had to hold before this was worth building:

* **Independence.** Atypicality computed on one population predicts a *different*
  population's tail with the effect changing by <3%, so it is not an artefact of the
  predictor and the risk sharing a score matrix.
* **Cheapness.** Four models recover 95-97% of the effect a 128-model population measures,
  and one epoch recovers 89-98% -- at or near peak utility on these datasets, so the proxies
  are not crippled.
* **Architecture transfer.** A ~25k-parameter SmallCNN predicted a ResNet-50's tail at 81%
  of the ResNet-50's own effect. The proxy need not resemble the production model.

What it does NOT do: measure leakage. It produces a ranking and a flagged cohort. A patient
who is not flagged is not certified safe.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from .data import RecordSet
from .energy import Energy, Meter, total as total_energy
from .manifest import Manifest
from . import models as _models

DEFAULT_MODELS = 4
DEFAULT_EPOCHS = 1
DEFAULT_MAX_EPOCHS = 5
EPOCH_LADDER = (1, 2, 5)
"""What the fail-safe tries when the self-consistency check fails, in order.

One epoch is at or near peak utility on the source study's datasets, so it is where a screen
starts. Where it is not enough the cause is almost always proxies too weak to rank anyone --
on shoulder MRI, three identical one-epoch runs scored 0.69, 0.73 and 0.68 (the split-half
correlation at four proxies wobbles by about +/-0.05 on GPU), while five-epoch proxies
reached 0.90. Five is therefore the top of the ladder and the default ceiling: beyond it the
screen stops being cheap, which is the property that makes it worth running at all."""
DEFAULT_FLAG_FRACTION = 0.20
"""The operating point the source study calibrated: flagging 20% of patients recovered
73-100% of the tail a full audit found (3.6-5x random flagging) across the five datasets
with patient identifiers. Tighter flag rates were not prospectively calibrated."""

SELF_CONSISTENCY_FLOOR = 0.70
"""Split-half rank agreement below which the flags must not be used.

This is the only validity check available on a dataset the tool has never seen, because it
needs no ground-truth tail: split the proxy models into two independent halves, rank patients
from each, and correlate. If two independent proxy sets disagree about who is atypical, the
ranking is noise and nothing downstream of it can be right.

Measured in the source study: CheXpert 0.905, NIH-CXR14 0.892, PTB-XL 0.827, ISIC-2020 0.822
-- all four delivered usable recall. PneumoniaMNIST, the per-example testbed with no patient
identifiers, sat at 0.496-0.635 and is the kind of dataset this floor exists to reject. On
shoulder MRI one-epoch proxies scored 0.690 and five-epoch proxies passed.

It is a NECESSARY CONDITION, not a performance predictor. Passing means the ranking is
reproducible; it does not tell you whether recall will be 70% or 96%. The threshold is a
judgement call sitting between two observed clusters, not a derived constant.
"""

BATCH_SIZE = 128
WEIGHT_DECAY = 1e-4
SEED_BASE = 5_000_000
PROB_EPS = 1e-7
"""Probabilities are clamped before the logit so a saturated model gives a large finite
confidence instead of an infinity that would poison a record's mean."""


def complementary_membership(patient_codes: np.ndarray, n_models: int, seed: int) -> np.ndarray:
    """[n_models, n_records] membership where every record is OUT for exactly half the models.

    Sampling each model's training set independently -- what a shadow-model study does --
    leaves a record's OUT count binomial, so at four models a meaningful share of records get
    no OUT model at all and score NaN. Fine across 128 models, useless at four.

    Complementary halves fix it by construction: draw n_models/2 patient-level splits and
    train two models per split, one on each side. Splitting is by PATIENT, never by record,
    so no patient straddles a boundary -- a straddling patient turns the held-out confidence
    into a near-duplicate lookup and the screen would rank its own leakage.
    """
    if n_models % 2 or n_models < 2:
        raise ValueError(f"n_models must be even and >= 2, got {n_models}")
    codes = np.asarray(patient_codes)
    uniq = np.unique(codes)
    rng = np.random.default_rng(seed)
    membership = np.zeros((n_models, len(codes)), dtype=bool)
    for pair in range(n_models // 2):
        shuffled = uniq.copy()
        rng.shuffle(shuffled)
        half = np.isin(codes, shuffled[: len(shuffled) // 2])
        membership[2 * pair] = half
        membership[2 * pair + 1] = ~half
    return membership


def atypicality(scores: np.ndarray, membership: np.ndarray, patient_codes: np.ndarray):
    """Per-record and per-patient atypicality.

    A record's atypicality is the negated mean confidence of the models that did NOT train on
    it: the held-out component of the memorisation score, i.e. how badly the rest of the data
    predicts this record. A patient's is the MAXIMUM over their records -- one unpredictable
    record is enough to expose the patient, and twelve competing definitions (mean, min,
    held-out variance, memorisation gap, rank-sums) were scored against the true tail on four
    datasets without beating it; the mean reduction was 26 points worse.
    """
    scores = np.asarray(scores, dtype=np.float64)
    out = ~np.asarray(membership, dtype=bool)
    with np.errstate(invalid="ignore"):
        n_out = out.sum(axis=0)
        out_mean = np.where(n_out > 0, np.where(out, scores, 0.0).sum(axis=0) /
                            np.maximum(n_out, 1), np.nan)
    record_atyp = -out_mean

    codes = np.asarray(patient_codes)
    uniq = np.unique(codes)
    patient_atyp = np.full(len(uniq), np.nan)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], uniq)
    for i, start in enumerate(bounds):
        end = bounds[i + 1] if i + 1 < len(bounds) else len(codes)
        vals = record_atyp[order[start:end]]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            patient_atyp[i] = vals.max()
    return record_atyp, patient_atyp


@dataclass
class Attempt:
    """One pass of the proxies, at one epoch budget."""

    epochs: int
    self_consistency: float
    gpu_seconds: float
    gpu_joules: float | None

    @property
    def passed(self) -> bool:
        sc = self.self_consistency
        return sc == sc and sc >= SELF_CONSISTENCY_FLOOR


@dataclass
class ScreenResult:
    n_patients: int
    n_records: int
    n_models: int
    epochs: int
    """Epochs of the attempt these flags come from, which is the last one tried."""
    flag_fraction: float
    arch: str
    pretrained: bool
    per_record: bool
    self_consistency: float
    gpu_seconds: float
    """Wall-clock of every attempt, so the reported cost includes the retries."""
    gpu_joules: float | None
    device: str
    attempts: list
    """[Attempt] in the order they were run. Length > 1 means the fail-safe escalated."""
    patient_names: list
    patient_atypicality: list
    percentile: list
    flagged: list

    _PATIENT_LEVEL = ("patient_names", "patient_atypicality", "percentile", "flagged")

    @property
    def energy(self) -> Energy:
        """What the whole screen cost, retries included."""
        return Energy(self.gpu_seconds, self.gpu_joules, self.device)

    @property
    def n_flagged(self) -> int:
        return int(sum(self.flagged))

    @property
    def passed(self) -> bool:
        sc = self.self_consistency
        return sc == sc and sc >= SELF_CONSISTENCY_FLOOR

    @property
    def verdict(self) -> str:
        """`passed`, `passed_after_retry`, or `below_floor`.

        The flags are delivered in all three cases -- withholding them was the old behaviour
        and it left the operator with nothing to act on and no way to tell a marginal dataset
        from a hopeless one. What changes with the verdict is how much weight they carry, so
        the verdict travels with the flags rather than only with the log.
        """
        if not self.passed:
            return "below_floor"
        return "passed" if len(self.attempts) <= 1 else "passed_after_retry"

    def safe_summary(self, patients_path: str | None = None) -> dict[str, Any]:
        """Aggregate-only view: no identifiers, no per-patient scores.

        This is what may be printed, logged, pasted into a ticket or read by an AI assistant.
        It is built by REMOVING the patient-level fields from the full record rather than by
        listing the safe ones, so a field added later is excluded only if it is added to
        `_PATIENT_LEVEL` -- which a test enforces by checking that nothing in the output is an
        array as long as the patient list.
        """
        d = {k: v for k, v in asdict(self).items() if k not in self._PATIENT_LEVEL}
        d["n_flagged"] = self.n_flagged
        d["self_consistency_floor"] = SELF_CONSISTENCY_FLOOR
        d["self_consistency_passed"] = self.passed
        d["verdict"] = self.verdict
        d["epochs_used"] = self.epochs
        d["gpu_minutes"] = round(self.gpu_seconds / 60, 2)
        d["gpu_kwh"] = round(self.gpu_joules / 3.6e6, 6) if self.gpu_joules else None
        d["energy"] = self.energy.as_dict()
        d["patient_detail_written_to"] = patients_path
        d["patient_detail_warning"] = (
            "The flagged cohort identifies individuals by privacy risk and is NOT in this "
            "file. It must not be read into an AI assistant's context or sent to any "
            "external service."
        )
        return d

    def flagged_names(self) -> list:
        return [p for p, f in zip(self.patient_names, self.flagged) if f]

    def write_patients(self, path: str | Path) -> None:
        """The flagged cohort, with the verdict on every row.

        The verdict is a column rather than a header comment because this file gets copied,
        mailed and opened in a spreadsheet, and a warning that lives anywhere else does not
        survive that. A column cannot be separated from the rows it qualifies, and it needs
        no `comment=` argument to parse.
        """
        import csv

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["patient_id", "atypicality", "percentile", "flagged",
                        "screen_verdict", "self_consistency"])
            rows = sorted(
                zip(self.patient_names, self.patient_atypicality, self.percentile,
                    self.flagged),
                key=lambda r: (-r[2] if r[2] == r[2] else 1e9),
            )
            verdict, sc = self.verdict, f"{self.self_consistency:.4f}"
            for name, atyp, pct, flag in rows:
                w.writerow([name, f"{atyp:.6f}", f"{pct:.2f}", int(flag), verdict, sc])

    def summary_text(self) -> str:
        unit = "records" if self.per_record else "patients"
        lines = [
            f"screened {self.n_patients:,} {unit} / {self.n_records:,} records",
            f"  {self.n_models} proxy models x {self.epochs} epoch(s) of {self.arch}"
            f"{' (ImageNet init)' if self.pretrained else ' (from scratch)'} on {self.device}, "
            f"{self.energy.text()}",
            f"  flagged {self.n_flagged:,} {unit} ({self.flag_fraction:.0%})",
        ]
        if len(self.attempts) > 1:
            ladder = " -> ".join(
                f"{a.epochs} epoch(s): {a.self_consistency:.3f}" for a in self.attempts
            )
            lines.append(f"  fail-safe escalated the proxies: {ladder}")

        sc = self.self_consistency
        if sc != sc:
            lines.append("  ** SELF-CONSISTENCY could not be computed (needs >= 4 models). "
                         "Treat the flags as unverified.")
        elif sc < SELF_CONSISTENCY_FLOOR:
            lines.append(
                f"  ** SELF-CONSISTENCY {sc:.3f} < {SELF_CONSISTENCY_FLOOR} after training "
                f"the proxies up to {self.epochs} epoch(s): two independent halves of the "
                f"proxy models still DISAGREE about who is atypical. The flags below are "
                f"from the most-trained attempt and are LESS RELIABLE than a passing screen. "
                f"Use them to prioritise, not to conclude; try --models 8, check that the "
                f"patient column really groups one person's records, and treat a full "
                f"shadow-model audit as the remaining option."
            )
        else:
            lines.append(
                f"  self-consistency {sc:.3f} (floor {SELF_CONSISTENCY_FLOOR}): the ranking "
                f"reproduces across independent proxy halves. Necessary for the flags to mean "
                f"anything -- it does NOT predict recall."
            )
        lines.append(
            "  NOT a leakage measurement. In the source study this flag rate recovered "
            "73-100% of the tail a full audit found; unflagged does not mean safe."
        )
        return "\n".join(lines)


def _say(message: str) -> None:
    """Progress, flushed: stdout is block-buffered when redirected to a file, and a screen
    that prints nothing for twenty minutes looks hung."""
    print(message, flush=True)


def fit(
    records: RecordSet,
    in_idx: np.ndarray,
    *,
    arch: str,
    pretrained: bool,
    epochs: int,
    lr: float,
    weight_decay: float = WEIGHT_DECAY,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Train one model on `in_idx`. Used for the proxies and for distillation teachers."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    model = _models.build(arch, num_classes=records.num_classes,
                          in_channels=records.in_channels, pretrained=pretrained).to(device)
    criterion = (nn.BCEWithLogitsLoss() if records.is_multilabel else nn.CrossEntropyLoss())
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # A small cohort at the study's batch size of 128 gets a handful of optimiser steps per
    # epoch and the proxy learns nothing, which shows up as a failed self-consistency gate
    # rather than as an error. Keep 128 wherever the data support it and shrink the batch on
    # small pools so that every proxy gets at least ~20 steps per epoch.
    batch = int(min(BATCH_SIZE, max(8, len(in_idx) // 20)))

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(in_idx)
        for start in range(0, len(order), batch):
            idx = np.sort(order[start:start + batch])
            if len(idx) < 2:
                continue  # BatchNorm needs more than one sample
            x, y = records.batch(idx)
            opt.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            opt.step()
    model.eval()
    return model


def _train_one(
    records: RecordSet,
    in_idx: np.ndarray,
    *,
    arch: str,
    pretrained: bool,
    epochs: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Train one proxy on `in_idx`, then score every record. Returns phi in record order."""
    model = fit(records, in_idx, arch=arch, pretrained=pretrained, epochs=epochs, lr=lr,
                seed=seed, device=device)
    phi = np.empty(len(records), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(records), BATCH_SIZE):
            idx = np.arange(start, min(start + BATCH_SIZE, len(records)))
            x, y = records.batch(idx)
            phi[idx] = _confidence(model(x), y, records.is_multilabel).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return phi


def _confidence(logits: torch.Tensor, y: torch.Tensor, is_multilabel: bool) -> torch.Tensor:
    """log(p_true / (1 - p_true)), the scaled confidence membership attacks consume.

    Raw probabilities pile up against 1; the logit spreads them out, which is why the
    membership-inference literature fits its distributions to this transform rather than to
    the probability. Multi-label records average the per-label logits -- in logit space, so
    the statistic stays on the single-label scale.
    """
    if is_multilabel:
        p = torch.sigmoid(logits.float()).clamp(PROB_EPS, 1.0 - PROB_EPS)
        p_true = torch.where(y > 0.5, p, 1.0 - p)
        return (torch.log(p_true) - torch.log1p(-p_true)).mean(dim=1)
    p = torch.softmax(logits.float(), dim=1).clamp(PROB_EPS, 1.0 - PROB_EPS)
    p_true = p.gather(1, y.view(-1, 1).long()).squeeze(1)
    return torch.log(p_true) - torch.log1p(-p_true)


def escalation(start: int, max_epochs: int) -> list[int]:
    """The epoch budgets the fail-safe will try, in order.

    `EPOCH_LADDER` from the first rung at or above `start`, capped at `max_epochs`; a start
    the ladder does not contain is kept as its own first rung, so `--epochs 3 --max-epochs 5`
    runs 3 then 5. `max_epochs <= start` means one attempt and no retry.
    """
    if start < 1:
        raise ValueError(f"epochs must be >= 1, got {start}")
    ceiling = max(max_epochs, start)
    rungs = {start, *(e for e in EPOCH_LADDER if e > start)}
    if ceiling > max(EPOCH_LADDER):
        rungs.add(ceiling)  # a raised ceiling is a rung, not just a cap
    return sorted(e for e in rungs if e <= ceiling)


def run(
    man: Manifest,
    *,
    n_models: int = DEFAULT_MODELS,
    epochs: int = DEFAULT_EPOCHS,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    flag_fraction: float = DEFAULT_FLAG_FRACTION,
    arch: str | None = None,
    pretrained: bool | None = None,
    size: int | None = None,
    device: str | None = None,
    cache_dir: str | None = None,
    loader: str | None = None,
    seed: int = SEED_BASE,
    progress: Callable[[str], None] = _say,
    keep_scores: bool = False,
) -> ScreenResult:
    """Train the proxies, rank patients, flag the top `flag_fraction`.

    The self-consistency fail-safe runs here rather than in the caller: a screen whose two
    proxy halves disagree is usually under-trained, not undoable, so the tool escalates the
    epoch budget itself (`escalation`) and stops at the first attempt that clears the floor.
    If none does, the flags from the most-trained attempt are returned anyway, carrying a
    `below_floor` verdict -- the operator is better served by a ranking they know to be
    shaky than by no ranking at all.

    Every attempt reuses the same membership draw, so the only thing that changes between
    them is how long the proxies trained, and the reported cost is the sum of all attempts.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = RecordSet(man, size=size, cache_dir=cache_dir, loader=loader, device=dev,
                        progress=progress)

    if arch is None:
        arch = _models.choose_arch(man.kind, records.input_size)
    if pretrained is None:
        pretrained = arch == "resnet18"
    # Pretrained features are fine-tuned gently; a from-scratch model needs a real learning
    # rate or one epoch produces noise, and one epoch is the whole point of this tool.
    lr = 1e-4 if pretrained else 1e-3

    membership = complementary_membership(man.patient_codes, n_models, seed)
    plan = escalation(epochs, max_epochs)
    attempts: list[Attempt] = []
    scores = np.zeros((n_models, len(records)), dtype=np.float64)
    self_consistency = float("nan")

    for attempt_no, attempt_epochs in enumerate(plan, start=1):
        with Meter(str(dev)) as meter:
            for i in range(n_models):
                progress(f"  proxy {i + 1}/{n_models} at {attempt_epochs} epoch(s) "
                         f"({int(membership[i].sum()):,} records)")
                scores[i] = _train_one(
                    records, np.flatnonzero(membership[i]), arch=arch, pretrained=pretrained,
                    epochs=attempt_epochs, lr=lr, seed=seed + i, device=dev,
                )
        self_consistency = _self_consistency(scores, membership, man.patient_codes, n_models)
        attempts.append(Attempt(
            epochs=attempt_epochs, self_consistency=self_consistency,
            gpu_seconds=meter.result.seconds, gpu_joules=meter.result.joules,
        ))

        if attempts[-1].passed:
            break
        value = ("could not be computed" if self_consistency != self_consistency
                 else f"{self_consistency:.3f}")
        if attempt_no < len(plan):
            progress(
                f"  self-consistency {value} is below the {SELF_CONSISTENCY_FLOOR} floor: "
                f"the proxies disagree about who is atypical, which usually means they are "
                f"under-trained. Retrying with {plan[attempt_no]} epoch(s)."
            )
        else:
            progress(
                f"  self-consistency {value} is still below the {SELF_CONSISTENCY_FLOOR} "
                f"floor at {attempt_epochs} epoch(s), the maximum. The flags from this "
                f"attempt are returned, marked below_floor, and are less reliable than a "
                f"passing screen."
            )

    epochs_used = attempts[-1].epochs
    spent = total_energy(
        Energy(a.gpu_seconds, a.gpu_joules, str(dev)) for a in attempts
    )
    gpu_seconds, joules = spent.seconds, spent.joules

    _, patient_atyp = atypicality(scores, membership, man.patient_codes)
    valid = np.isfinite(patient_atyp)
    order = np.argsort(-np.where(valid, patient_atyp, -np.inf), kind="stable")
    pct = np.full(len(patient_atyp), np.nan)
    pct[order[: valid.sum()]] = np.linspace(100.0, 0.0, int(valid.sum()), endpoint=False)
    k = max(1, int(round(flag_fraction * valid.sum())))
    flagged = np.zeros(len(patient_atyp), dtype=bool)
    flagged[order[:k]] = True

    result = ScreenResult(
        n_patients=man.n_patients, n_records=man.n_records, n_models=n_models,
        epochs=epochs_used, flag_fraction=flag_fraction, arch=arch,
        pretrained=bool(pretrained), per_record=man.per_record,
        self_consistency=self_consistency, gpu_seconds=gpu_seconds, gpu_joules=joules,
        device=str(dev), attempts=attempts,
        patient_names=list(man.patient_names), patient_atypicality=patient_atyp.tolist(),
        percentile=pct.tolist(), flagged=flagged.tolist(),
    )
    if keep_scores:
        result.scores, result.membership = scores, membership  # type: ignore[attr-defined]
    return result


def _self_consistency(scores, membership, patient_codes, n_models: int) -> float:
    """Spearman correlation between rankings from two independent halves of the proxies.

    Complementary halves mean each PAIR of models already covers every record, so splitting
    on pairs gives two complete, independent estimates of the same ranking.
    """
    if n_models < 4:
        return float("nan")
    from scipy.stats import spearmanr

    half = (n_models // 2) - ((n_models // 2) % 2) or 2
    _, a = atypicality(scores[:half], membership[:half], patient_codes)
    _, b = atypicality(scores[half:], membership[half:], patient_codes)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() <= 2:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def write_summary(result: ScreenResult, path: str | Path, patients_path: str | None) -> None:
    Path(path).write_text(
        json.dumps(result.safe_summary(patients_path), indent=2, default=str),
        encoding="utf-8",
    )
