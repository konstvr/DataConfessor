"""The human-readable half of a run: three diagnostic panels in one HTML file.

The panels are chosen to make the two failure modes of this tool visible rather than to
decorate it:

* the ranking can be *flat*, in which case flagging the top 20% cuts an arbitrary line
  through a continuum -- panel a shows the shape of the ranking and where the cut falls;
* the ranking can be *noise*, in which case two independent halves of the proxy models
  disagree -- panel c is that disagreement, plotted, with the 0.70 floor named.

No patient identifier appears in this file. It is safe to attach to a ticket.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np

FLAG = "#D55E00"
REST = "#0072B2"
INK, MUTED = "#1a1a19", "#5c5c58"
"""Okabe-Ito vermillion and blue: validated for deuteranopia, protanopia and tritanopia
(worst adjacent pair dE 21.9) against a near-white surface. Grey is ink, never a series."""


def _energy(joules: float) -> str:
    kwh = joules / 3.6e6
    return f", {kwh:.4f} kWh" if kwh >= 1e-4 else f", {joules / 1000:.1f} kJ"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, width=0.8)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontsize(9)
    ax.title.set_color(INK)
    ax.title.set_fontsize(10)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def panels(result, scores=None, membership=None, patient_codes=None) -> list[tuple[str, str, str]]:
    """[(id, caption, base64 png)] -- as many as the available data allow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    atyp = np.asarray(result.patient_atypicality, dtype=float)
    flagged = np.asarray(result.flagged, dtype=bool)
    ok = np.isfinite(atyp)
    unit = "record" if result.per_record else "patient"
    out: list[tuple[str, str, str]] = []

    # (a) the ranking, and where the line is drawn
    order = np.argsort(-np.where(ok, atyp, -np.inf))[: int(ok.sum())]
    ranked, ranked_flag = atyp[order], flagged[order]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(len(ranked))
    ax.fill_between(x, ranked.min(), ranked, where=ranked_flag, color=FLAG, linewidth=0,
                    label=f"flagged ({int(flagged.sum()):,})")
    ax.fill_between(x, ranked.min(), ranked, where=~ranked_flag, color=REST, linewidth=0,
                    label=f"not flagged ({int((~flagged).sum()):,})")
    cut = int(ranked_flag.sum())
    ax.axvline(cut, color=INK, linewidth=1.0, linestyle="--")
    ax.annotate(f"{result.flag_fraction:.0%} cut", (cut, ranked.max()),
                xytext=(6, -4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel(f"{unit}s, ranked by atypicality")
    ax.set_ylabel("atypicality")
    ax.set_title("a  Where the flag line falls")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")
    _style(ax)
    out.append(("a", f"Atypicality of every {unit}, highest first. A ranking that drops "
                     f"steeply has a distinct tail; a flat one means the cut is arbitrary "
                     f"and the flags are closer to a budget than to a finding.", _png(fig)))

    # (b) the distribution either side of the cut
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bins = np.linspace(np.nanmin(atyp), np.nanmax(atyp), 40)
    ax.hist([atyp[ok & ~flagged], atyp[ok & flagged]], bins=bins, stacked=True,
            color=[REST, FLAG], label=["not flagged", "flagged"], linewidth=0)
    ax.set_xlabel("atypicality")
    ax.set_ylabel(f"{unit}s")
    ax.set_title("b  Distribution of atypicality")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style(ax)
    out.append(("b", f"How atypicality is distributed across the cohort. The flagged "
                     f"{unit}s are the right-hand tail by construction; what matters is "
                     f"whether that tail is separated or continuous.", _png(fig)))

    # (c) the validity check, plotted
    if scores is not None and membership is not None and patient_codes is not None:
        from .screen import SELF_CONSISTENCY_FLOOR, atypicality as _atyp

        n = len(scores)
        half = (n // 2) - ((n // 2) % 2) or 2
        _, a = _atyp(scores[:half], membership[:half], patient_codes)
        _, b = _atyp(scores[half:], membership[half:], patient_codes)
        good = np.isfinite(a) & np.isfinite(b)
        ra = np.argsort(np.argsort(a[good])) / max(good.sum() - 1, 1) * 100
        rb = np.argsort(np.argsort(b[good])) / max(good.sum() - 1, 1) * 100
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        colour = FLAG if result.passed else "#8a8a86"
        ax.scatter(ra, rb, s=5, alpha=0.45, color=colour, linewidths=0)
        ax.plot([0, 100], [0, 100], color=INK, linewidth=0.8, linestyle="--")
        ax.set_xlabel("rank from proxy half A (%)")
        ax.set_ylabel("rank from half B (%)")
        verdict = "passes" if result.passed else "FAILS"
        ax.set_title(f"c  Self-consistency {result.self_consistency:.3f} ({verdict})")
        _style(ax)
        out.append(("c", f"Each {unit} ranked twice, by two independent halves of the proxy "
                         f"models. Agreement (Spearman) is {result.self_consistency:.3f} "
                         f"against a floor of {SELF_CONSISTENCY_FLOOR}. This is a necessary "
                         f"condition for the flags to mean anything; it does not predict how "
                         f"much of the true tail they contain.", _png(fig)))
    return out


_CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a19;
 background:#fcfcfb;margin:0;padding:32px 24px;}
main{max-width:860px;margin:0 auto;}
h1{font-size:24px;margin:0 0 4px;} h2{font-size:17px;margin:32px 0 8px;}
h3{font-size:15px;margin:18px 0 4px;}
.sub{color:#5c5c58;margin:0 0 24px;}
table{border-collapse:collapse;width:100%;font-size:14px;}
td,th{text-align:left;padding:6px 10px;border-bottom:1px solid #e6e6e3;}
th{color:#5c5c58;font-weight:500;width:42%;}
figure{margin:20px 0;} figcaption{color:#5c5c58;font-size:13px;margin-top:6px;}
img{max-width:100%;}
.banner{padding:12px 14px;border-radius:6px;font-size:14px;margin:16px 0;}
.pass{background:#eef5fb;border-left:4px solid #0072B2;}
.fail{background:#fdf0e9;border-left:4px solid #D55E00;}
.warn{background:#fdf8e9;border-left:4px solid #b8860b;}
code{background:#f0f0ed;padding:1px 5px;border-radius:3px;font-size:13px;}
li{margin:4px 0;}
"""


def build(
    result,
    man,
    *,
    recommendations,
    patients_path: str | None,
    summary_path: str | None,
    scores=None,
    membership=None,
) -> str:
    """The whole report as one self-contained HTML string."""
    from .manifest import summary as man_summary
    from .screen import SELF_CONSISTENCY_FLOOR

    e = html.escape
    unit = "record" if result.per_record else "patient"
    ms = man_summary(man)
    figs = panels(result, scores, membership, man.patient_codes)

    gate = (
        f'<div class="banner pass"><b>Self-consistency {result.self_consistency:.3f} &ge; '
        f'{SELF_CONSISTENCY_FLOOR}.</b> Two independent halves of the proxy models rank '
        f'{unit}s the same way, so the ranking is reproducible. This is necessary for the '
        f'flags to mean anything &mdash; it does not predict how much of the true tail they '
        f'contain.</div>'
        if result.passed else
        f'<div class="banner fail"><b>Self-consistency {result.self_consistency:.3f} &lt; '
        f'{SELF_CONSISTENCY_FLOOR}: do not act on these flags.</b> Two independent halves of '
        f'the proxy models disagree about who is atypical, so the ranking is not reproducible '
        f'on this data. Re-run with more epochs (<code>--epochs 3</code>, then '
        f'<code>--epochs 5</code>), then more proxies (<code>--models 8</code>), then check '
        f'the patient grouping. The flag column in the CSV is invalid until this passes.</div>'
    )

    rows = [
        (f"{unit.capitalize()}s screened", f"{result.n_patients:,}"),
        ("Records", f"{result.n_records:,}"),
        (f"{unit.capitalize()}s flagged", f"{result.n_flagged:,} ({result.flag_fraction:.0%})"),
        ("Proxy budget", f"{result.n_models} models &times; {result.epochs} epoch(s), "
                         f"{e(result.arch)}{' (ImageNet init)' if result.pretrained else ''}"),
        ("Compute", f"{result.gpu_seconds / 60:.1f} minutes on {e(result.device)}"
                    + (_energy(result.gpu_joules) + " (GPU board, NVML)"
                       if result.gpu_joules else "")),
        ("Records per " + unit, f"min {ms['records_per_patient']['min']}, "
                                f"median {ms['records_per_patient']['median']:.0f}, "
                                f"max {ms['records_per_patient']['max']}"),
        ("Task", f"{ms['task']}, {len(ms['classes'])} classes"),
    ]
    if patients_path:
        rows.append(("Flagged cohort (sensitive)", f"<code>{e(patients_path)}</code>"))
    if summary_path:
        rows.append(("Machine-readable summary", f"<code>{e(summary_path)}</code>"))

    warn = ""
    if man.warnings:
        warn = ('<div class="banner warn"><b>Data warnings</b><ul>'
                + "".join(f"<li>{e(w)}</li>" for w in man.warnings) + "</ul></div>")

    recs = "".join(
        f"<h3>{i}. {e(r.title)}</h3><ul>"
        f"<li><b>Measured:</b> {e(r.evidence)}</li>"
        f"<li><b>Do:</b> {e(r.action)}</li>"
        f"<li><b>Cost:</b> {e(r.cost)}</li></ul>"
        for i, r in enumerate([r for r in recommendations if r.applies], start=1)
    )

    figs_html = "".join(
        f'<figure><img alt="panel {k}" src="data:image/png;base64,{png}">'
        f'<figcaption><b>{k}</b> &nbsp;{e(cap)}</figcaption></figure>'
        for k, cap, png in figs
    )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DataConfessor screen</title><style>{_CSS}</style></head><body><main>
<h1>Privacy screen</h1>
<p class="sub">{result.n_flagged:,} of {result.n_patients:,} {unit}s flagged as most likely
to be exposed by membership inference, before any production model is trained.</p>
{gate}
{warn}
<h2>Run</h2>
<table>{"".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)}</table>
<h2>What the screen found</h2>
{figs_html}
<h2>What this is, and is not</h2>
<ul>
<li>It is a <b>ranking and a flagged cohort</b>, not a leakage measurement. Never report it
as a privacy measurement or as a probability of exposure.</li>
<li>An unflagged {unit} is <b>not certified safe</b>. At this flag rate the screen recovered
65&ndash;96% of the tail a full audit found across six dataset&ndash;recipe combinations in
the source study, so part of the tail is always missed.</li>
<li>The recall figures above come from other datasets. This run has no ground truth, so its
own recall is unknown.</li>
<li>Intended uses: allocating a privacy budget, targeting consent review, targeting an audit,
targeting a training-time defence.</li>
</ul>
{"<h2>Recommended mitigations</h2>" + recs if recs else ""}
<p class="sub">Generated by DataConfessor. Method: Vrettos et al., <i>Predicting and
mitigating privacy risks in medical AI</i>.</p>
</main></body></html>"""


def write(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")
