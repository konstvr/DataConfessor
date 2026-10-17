"""One runnable check per piece of logic that could break silently.

The end-to-end test is the important one: it plants a known atypical tail in a synthetic
cohort and asserts the screen recovers it. Everything the tool claims rests on that, and on
the aggregate/sensitive split that keeps the flagged cohort out of an assistant's context.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from make_synthetic import make  # noqa: E402

from dataconfessor import load_manifest, screen  # noqa: E402
from dataconfessor.screen import complementary_membership  # noqa: E402


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    info = make(out, n_patients=400, seed=1)
    return out, info


def test_complementary_membership_holds_every_record_out_exactly_half_the_time():
    codes = np.repeat(np.arange(50), 3)
    m = complementary_membership(codes, 4, seed=0)
    assert (m.sum(axis=0) == 2).all()
    # No patient may straddle a split: a straddling patient turns the held-out confidence
    # into a near-duplicate lookup and the screen would rank its own leakage.
    for model in m:
        for code in np.unique(codes):
            assert len(set(model[codes == code].tolist())) == 1


def test_odd_model_count_is_refused():
    with pytest.raises(ValueError):
        complementary_membership(np.arange(10), 3, seed=0)


def test_manifest_rejects_a_single_patient_grouping(tmp_path):
    (tmp_path / "a.png").write_bytes(b"")
    csv = tmp_path / "m.csv"
    csv.write_text("path,label,patient_id\na.png,0,P1\na.png,1,P1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one patient_id"):
        load_manifest(csv)


def test_manifest_reports_grouping_and_classes(cohort):
    out, info = cohort
    man = load_manifest(out / "manifest.csv")
    from dataconfessor import describe

    s = describe(man)
    assert s["n_patients"] == info["n_patients"]
    assert s["n_records"] == info["n_records"]
    assert set(s["classes"]) == {"0", "1"}


def test_screen_recovers_a_planted_atypical_tail(cohort):
    out, info = cohort
    man = load_manifest(out / "manifest.csv")
    result = screen(man, epochs=3, device="cpu", progress=lambda _: None, keep_scores=True)

    assert result.self_consistency >= 0.70, "planted tail should be reproducible"
    flagged = set(result.flagged_names())
    truth = set(info["atypical"])
    recall = len(flagged & truth) / len(truth)
    assert recall > 0.5, f"recall {recall:.2f} barely beats random flagging (0.20)"


def test_under_trained_proxies_fail_the_self_consistency_gate(cohort):
    """The gate's whole job: proxies too weak to rank patients must not produce flags.

    One epoch on this cohort is not enough to learn the rule, so the two halves of the
    proxy set disagree and `passed` is False -- which is what sends a user up the
    escalation ladder instead of acting on noise.
    """
    out, _ = cohort
    man = load_manifest(out / "manifest.csv")
    result = screen(man, epochs=1, device="cpu", progress=lambda _: None)
    assert not result.passed
    assert "Do not act on them" in result.summary_text()


def test_safe_summary_contains_no_patient_level_data(cohort):
    out, _ = cohort
    man = load_manifest(out / "manifest.csv")
    result = screen(man, epochs=1, device="cpu", progress=lambda _: None)
    safe = result.safe_summary("flagged.csv")

    # The guard is structural: nothing in the aggregate view may be as long as the patient
    # list, so a per-patient field added later cannot leak through unnoticed.
    for key, value in safe.items():
        if isinstance(value, (list, tuple)):
            assert len(value) < result.n_patients, f"{key} looks patient-level"
    assert "patient_detail_warning" in safe


def test_cli_screen_prints_no_identifiers(cohort, tmp_path):
    out, info = cohort
    proc = subprocess.run(
        [sys.executable, "-m", "dataconfessor.cli", "screen", str(out / "manifest.csv"),
         "--epochs", "1", "--device", "cpu", "--patients", str(tmp_path / "p.csv"),
         "--out", str(tmp_path / "s.json")],
        capture_output=True, text=True, check=True,
    )
    for name in info["atypical"][:20]:
        assert name not in proc.stdout, "a patient id reached stdout"
    assert (tmp_path / "p.csv").exists()


def test_multilabel_manifest_runs_end_to_end(cohort, tmp_path):
    """Multi-label records carry `;`-separated classes and score through the BCE path."""
    import csv

    out, _ = cohort
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    multi = tmp_path / "multi.csv"
    with open(multi, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "patient_id"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"path": str((out / r["path"]).resolve()),
                        "label": "lesion;rare" if i % 3 == 0 else "lesion",
                        "patient_id": r["patient_id"]})

    man = load_manifest(multi)
    assert man.is_multilabel and man.class_names == ["lesion", "rare"]
    result = screen(man, epochs=1, device="cpu", progress=lambda _: None)
    assert np.isfinite(result.patient_atypicality).all()
    assert result.n_flagged == round(0.2 * result.n_patients)


def test_packaged_skill_matches_the_repository_skill():
    """The wheel carries its own copy of SKILL.md, because `pip install` has no repo to
    read it from and the assistant mode is only as good as the instructions it sends.
    Two copies drift, so the drift is a test failure rather than a surprise."""
    root = Path(__file__).resolve().parents[1]
    packaged = root / "src" / "dataconfessor" / "web" / "SKILL.md"
    source = root / "skills" / "dataconfessor" / "SKILL.md"
    assert packaged.read_text(encoding="utf-8") == source.read_text(encoding="utf-8"), (
        "src/dataconfessor/web/SKILL.md is out of date: copy skills/dataconfessor/SKILL.md "
        "over it."
    )
