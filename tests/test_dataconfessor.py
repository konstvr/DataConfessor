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
    """The gate's whole job: proxies too weak to rank patients must say so.

    One epoch is not enough to learn the rule on this cohort. With the fail-safe disabled
    (`max_epochs` equal to the starting epochs) the check fails, the flags are still
    delivered, and both the verdict and the printed summary say they are the weaker kind.
    """
    out, _ = cohort
    man = load_manifest(out / "manifest.csv")
    result = screen(man, epochs=1, max_epochs=1, device="cpu", progress=lambda _: None)

    assert not result.passed and result.verdict == "below_floor"
    assert len(result.attempts) == 1
    assert result.n_flagged > 0
    assert "LESS RELIABLE" in result.summary_text()


def test_failsafe_rescues_the_synthetic_cohort(cohort):
    """End to end, with real proxies: one epoch fails here and the fail-safe fixes it."""
    out, info = cohort
    man = load_manifest(out / "manifest.csv")
    result = screen(man, device="cpu", progress=lambda _: None)

    assert result.passed, "the ladder should reach a passing budget on this cohort"
    assert result.verdict == "passed_after_retry" and result.epochs > 1
    recall = len(set(result.flagged_names()) & set(info["atypical"])) / len(info["atypical"])
    assert recall > 0.5, f"recall {recall:.2f} barely beats random flagging (0.20)"


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


# --- the self-consistency fail-safe -------------------------------------------------------
#
# These patch the two expensive parts (training a proxy, and the split-half correlation it
# feeds) so the test is about the escalation logic and nothing else: which epoch budgets get
# tried, which attempt's flags come back, and what the verdict says.


@pytest.fixture
def fake_proxies(monkeypatch):
    """Make `run()` cheap, and make its self-consistency follow a script."""
    # `from dataconfessor import screen` would give the package-level alias for screen.run,
    # so the module is taken from the import system instead.
    from importlib import import_module

    sc = import_module("dataconfessor.screen")

    def fake_train(records, in_idx, *, epochs, seed, **_):
        rng = np.random.default_rng(seed + epochs)
        return rng.normal(size=len(records))

    def install(sequence):
        values = list(sequence)
        seen = []

        def fake_consistency(scores, membership, codes, n_models):
            seen.append(values[len(seen)] if len(seen) < len(values) else values[-1])
            return seen[-1]

        monkeypatch.setattr(sc, "_train_one", fake_train)
        monkeypatch.setattr(sc, "_self_consistency", fake_consistency)
        return seen

    return install


def test_escalation_schedule():
    from dataconfessor.screen import escalation

    assert escalation(1, 5) == [1, 2, 5]
    assert escalation(3, 5) == [3, 5]
    assert escalation(1, 1) == [1]          # retrying turned off
    assert escalation(2, 12) == [2, 5, 12]  # a raised ceiling is a rung, not only a cap
    assert escalation(7, 5) == [7]          # a start above the ceiling still runs once


def test_failsafe_retries_until_the_check_passes(cohort, fake_proxies):
    out, _ = cohort
    man = load_manifest(out / "manifest.csv")
    seen = fake_proxies([0.65, 0.68, 0.85])

    result = screen(man, device="cpu", progress=lambda _: None)

    assert seen == [0.65, 0.68, 0.85], "it should stop at the first passing attempt"
    assert [a.epochs for a in result.attempts] == [1, 2, 5]
    assert result.epochs == 5, "the flags must come from the attempt that passed"
    assert result.self_consistency == 0.85
    assert result.passed and result.verdict == "passed_after_retry"
    assert result.gpu_seconds == pytest.approx(sum(a.gpu_seconds for a in result.attempts))


def test_failsafe_delivers_marked_flags_when_it_never_passes(cohort, fake_proxies, tmp_path):
    out, _ = cohort
    man = load_manifest(out / "manifest.csv")
    fake_proxies([0.41, 0.52, 0.60])

    result = screen(man, device="cpu", progress=lambda _: None)

    assert not result.passed
    assert result.verdict == "below_floor"
    assert result.epochs == 5, "the flags come from the most-trained attempt"
    assert result.n_flagged == round(0.2 * result.n_patients), "flags are still delivered"

    summary = result.safe_summary("flagged.csv")
    assert summary["verdict"] == "below_floor"
    assert summary["self_consistency_floor"] == 0.70
    assert summary["epochs_used"] == 5
    assert [a["self_consistency"] for a in summary["attempts"]] == [0.41, 0.52, 0.60]

    # The warning travels with the rows, so the file cannot be separated from its verdict.
    import csv

    path = tmp_path / "flagged.csv"
    result.write_patients(path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert {r["screen_verdict"] for r in rows} == {"below_floor"}
    assert len(rows) == result.n_patients


def test_one_attempt_when_the_first_check_passes(cohort, fake_proxies):
    out, _ = cohort
    man = load_manifest(out / "manifest.csv")
    seen = fake_proxies([0.91])

    result = screen(man, device="cpu", progress=lambda _: None)

    assert len(seen) == 1 and len(result.attempts) == 1
    assert result.verdict == "passed" and result.epochs == 1


def test_loader_spec_accepts_a_windows_path(tmp_path):
    """`D:/data/loader.py:load` must split on the last colon, not the drive letter.

    The loader under test is written here rather than taken from `examples/`, so this
    exercises the spec parsing and nothing else. The earlier version imported the PTB-XL
    loader, which pulls in wfdb: that made a string-parsing test depend on a third-party
    import running cleanly at the end of a long session, and it failed in full runs while
    passing alone.
    """
    from importlib import import_module

    data = import_module("dataconfessor.data")
    loader = tmp_path / "my_loader.py"
    loader.write_text("def load(path):\n    return path\n", encoding="utf-8")

    hook = data._load_hook(f"{loader}:load")          # native path, drive letter and all
    assert callable(hook) and hook("x") == "x"
    assert callable(data._load_hook(f"{loader.as_posix()}:load"))

    for bad in ("loader.py", "loader:load", ":load", str(loader)):
        with pytest.raises(ValueError, match="path/to/file.py:function"):
            data._load_hook(bad)
