"""Command line interface.

Five commands, in the order a run uses them:

    dataconfessor init      build a manifest.csv from a folder of files (+ a labels CSV)
    dataconfessor check     validate the manifest and describe the cohort -- free, instant
    dataconfessor screen    train the proxies, rank patients, write the flags and the report
    dataconfessor teachers  soft labels for the flagged records (self-distillation)
    dataconfessor advise    the measured mitigations, ranked for your recipe

Everything printed to stdout is aggregate: counts, statistics, verdicts and file paths. The
one output that identifies individuals -- the per-patient CSV -- is written to disk and
never printed, because the usual operator of this tool is an AI assistant whose context is
not the right place for a list of patients ranked by privacy risk.

Each command ends with a NEXT line naming the command to run next, so an assistant driving
the tool does not have to infer the workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("manifest", help="path to manifest.csv")
    p.add_argument("--size", type=int, default=None,
                   help="resize images to SIZE x SIZE (default: their native size, capped "
                        "at 224; never upsampled)")
    p.add_argument("--device", default=None, help="cuda, cpu, cuda:1 ... (default: auto)")
    p.add_argument("--cache-dir", default=None,
                   help="where decoded records are cached (default: beside the manifest)")
    p.add_argument("--loader", default=None,
                   help="path/to/file.py:function returning one record as an array, for "
                        "formats this tool does not read natively (DICOM, NIfTI, WFDB)")
    p.add_argument("--kind", choices=["image", "signal"], default=None,
                   help="what a record is, when --loader reads files this tool does not "
                        "recognise by suffix (DICOM, WFDB, NIfTI)")
    p.add_argument("--no-patient-id", action="store_true",
                   help="treat every record as its own patient; results become per-record")


def cmd_init(args: argparse.Namespace) -> int:
    from . import manifest as mf

    info = mf.build(
        args.out, images=args.images, labels_csv=args.labels_csv,
        image_col=args.image_col, label_col=args.label_col, patient_col=args.patient_col,
        suffix=args.suffix, recursive=not args.no_recursive,
    )
    print(json.dumps(info, indent=2))
    if not info["patient_id_present"]:
        print("\nWARNING: no patient column was used, so every record is its own patient. "
              "If one person contributed several records, fill in patient_id before "
              "screening or the ranking is not a patient ranking.")
    if info["files_unmatched"]:
        print(f"\nWARNING: {info['files_unmatched']} file(s) had no row in the labels CSV "
              f"and were left out.")
    print(f"\nNEXT: dataconfessor check {info['manifest']}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from . import manifest as mf

    man = mf.load(args.manifest, no_patient_id=args.no_patient_id, kind=args.kind)
    print(json.dumps(mf.summary(man), indent=2))
    print(f"\nNEXT: dataconfessor screen {args.manifest} --patients flagged_patients.csv "
          f"--report screen_report.html --out screen_summary.json")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    # `from . import screen` would pick up the package-level alias for
    # screen.run, so the submodule is imported by path.
    from . import manifest as mf, mitigate, report as rep
    from .screen import run as run_screen, write_summary

    man = mf.load(args.manifest, no_patient_id=args.no_patient_id, kind=args.kind)
    for w in man.warnings:
        print(f"WARNING: {w}")

    result = run_screen(
        man, n_models=args.models, epochs=args.epochs, max_epochs=args.max_epochs,
        flag_fraction=args.flag, arch=args.arch, size=args.size, device=args.device,
        cache_dir=args.cache_dir, loader=args.loader, keep_scores=bool(args.report),
    )
    print()
    print(result.summary_text())

    if args.patients:
        result.write_patients(args.patients)
        print(f"\nFLAGGED COHORT (sensitive) -> {args.patients}")
        print(f"  verdict: {result.verdict} -- also a column in the file, so it cannot be "
              f"separated from the rows it qualifies")
        print("  This file ranks individuals by privacy risk. Keep it local: do not read it "
              "into an AI assistant's context, print it, or send it anywhere.")
    else:
        print("\nNo --patients given, so the flagged cohort was not written. Re-run with "
              "--patients <path.csv> to get the actionable list.")

    if args.out:
        write_summary(result, args.out, args.patients)
        print(f"\nSummary (aggregate only, safe to share) -> {args.out}")

    if args.report:
        html = rep.build(
            result, man, recommendations=mitigate.advise(),
            patients_path=args.patients, summary_path=args.out,
            scores=getattr(result, "scores", None),
            membership=getattr(result, "membership", None),
        )
        rep.write(args.report, html)
        print(f"Report (no identifiers, safe to share) -> {args.report}")

    if not result.passed:
        print(f"\nNEXT: the fail-safe already retried up to {result.epochs} epoch(s) and the "
              f"check is still below the floor, so these flags are marked below_floor and "
              f"are less reliable than a passing screen.")
        print("  Use them to prioritise, not to conclude. Worth trying, in this order: "
              "--models 8; confirming that the patient column really groups one person's "
              "records; a higher ceiling with --max-epochs. If none of that clears the "
              "floor, this dataset needs a full shadow-model audit. Do not lower the "
              "threshold.")
    else:
        print(f"\nNEXT: dataconfessor advise  (mitigations), or dataconfessor teachers "
              f"{args.manifest} --flagged {args.patients or '<patients.csv>'} --epochs "
              f"<your production epochs>  (soft labels for self-distillation)")
    # The flags are always delivered, so the exit code stays 0 unless the operator asked a
    # pipeline to stop on a below-floor verdict.
    return 3 if (args.strict and not result.passed) else 0


def cmd_teachers(args: argparse.Namespace) -> int:
    from . import manifest as mf, mitigate

    man = mf.load(args.manifest, no_patient_id=args.no_patient_id, kind=args.kind)
    flagged = mitigate.read_flagged(args.flagged)
    info = mitigate.teachers(
        man, flagged, epochs=args.epochs, arch=args.arch, size=args.size,
        device=args.device, cache_dir=args.cache_dir, loader=args.loader, out_path=args.out,
    )
    print(json.dumps(info, indent=2))
    print("\nNEXT: train your production model with these soft labels substituted for the "
          "true labels of the listed records, then re-measure utility. The source study saw "
          "-0.035 held-out AUROC for -0.15 mean risk on the flagged patients.")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import serve

    return serve(workdir=args.workdir, port=args.port, open_browser=not args.no_browser)


def cmd_advise(args: argparse.Namespace) -> int:
    from . import mitigate

    recs = mitigate.advise(weight_decay=args.weight_decay, epochs=args.epochs,
                           finetune=args.finetune, underfitted=args.underfitted)
    print(mitigate.advice_markdown(recs, include_all=args.all))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dataconfessor",
        description="Flag the patients a training run is most likely to expose, before you "
                    "train it.",
    )
    ap.add_argument("--version", action="version", version=f"dataconfessor {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="build a manifest.csv from files + a labels CSV")
    p.add_argument("images", help="folder containing the record files")
    p.add_argument("--labels-csv", default=None,
                   help="CSV with one row per record; joined to the files by filename stem. "
                        "Without it, the label is each file's parent folder name.")
    p.add_argument("--image-col", default=None, help="column holding the filename")
    p.add_argument("--label-col", default=None, help="column holding the training label")
    p.add_argument("--patient-col", default=None, help="column holding the patient id")
    p.add_argument("--suffix", default=None, help="only files with this suffix, e.g. .npy")
    p.add_argument("--no-recursive", action="store_true")
    p.add_argument("--out", default="manifest.csv")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("check", help="validate a manifest and describe the cohort")
    _add_data_args(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("screen", help="rank patients by atypicality and flag the top ones")
    _add_data_args(p)
    p.add_argument("--models", type=int, default=4, help="proxy models, even number "
                                                         "(default 4)")
    p.add_argument("--epochs", type=int, default=1,
                   help="epochs per proxy to start from (default 1)")
    p.add_argument("--max-epochs", type=int, default=5,
                   help="ceiling for the fail-safe's automatic retries (default 5). While "
                        "the self-consistency check is below the floor the screen escalates "
                        "1 -> 2 -> 5 epochs by itself; set this equal to --epochs to turn "
                        "retrying off.")
    p.add_argument("--strict", action="store_true",
                   help="exit 3 rather than 0 when the check is still below the floor, for "
                        "pipelines that must stop instead of acting on weaker flags")
    p.add_argument("--flag", type=float, default=0.20,
                   help="fraction of patients to flag (default 0.20, the calibrated point)")
    p.add_argument("--arch", default=None,
                   help="resnet18, smallcnn, resnet1d18 (default: chosen from the data)")
    p.add_argument("--patients", default=None,
                   help="write the per-patient CSV here. SENSITIVE: ranks individuals by "
                        "risk, keep local.")
    p.add_argument("--out", default=None, help="aggregate summary JSON (no identifiers)")
    p.add_argument("--report", default=None, help="HTML report (no identifiers)")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("teachers", help="soft labels for the flagged records")
    _add_data_args(p)
    p.add_argument("--flagged", required=True, help="the per-patient CSV written by screen")
    p.add_argument("--epochs", type=int, required=True,
                   help="epochs of YOUR production recipe; teachers must be real models, "
                        "not one-epoch proxies")
    p.add_argument("--arch", default=None)
    p.add_argument("--out", default="soft_labels.npz")
    p.set_defaults(func=cmd_teachers)

    p = sub.add_parser("ui", help="open the browser interface (local only)")
    p.add_argument("--workdir", default=None,
                   help="where runs write their manifest, flags and report "
                        "(default: the current directory)")
    p.add_argument("--port", type=int, default=0, help="0 picks a free port")
    p.add_argument("--no-browser", action="store_true", help="print the URL, do not open it")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("advise", help="the measured mitigations, ranked")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="the weight decay your recipe uses")
    p.add_argument("--epochs", type=int, default=None, help="epochs your recipe trains for")
    p.add_argument("--finetune", choices=["none", "head", "partial", "full"], default=None,
                   help="how much of a pretrained model your recipe trains")
    p.add_argument("--underfitted", action="store_true", default=None,
                   help="held-out performance is below the published figure for this task")
    p.add_argument("--all", action="store_true", help="include recommendations that do not "
                                                      "apply to the described recipe")
    p.set_defaults(func=cmd_advise)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
