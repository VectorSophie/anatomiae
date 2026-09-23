"""Publication figures, generated only from saved analysis artifacts.

Every figure function takes an already-computed DataFrame (read from an
artifacts/tables/*.parquet file by the calling script) and writes
SVG + PDF + PNG plus a provenance sidecar (input artifact + sha256,
generating script, anatomiae git commit, timestamp). No numbers are typed
into plotting code.

Styling follows the project's data-viz rules: validated categorical slots
assigned in fixed order (never cycled), a fixed outcome->color mapping,
single-hue sequential ramp for counts, one y-axis per chart, thin marks,
recessive grid, text in ink colors (never series colors), direct labels
plus a source table for every figure (three slots sit below 3:1 contrast
on the light surface, so labels/tables are the required relief).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots 1-3 (validated all-pairs, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# Fixed outcome -> slot mapping (validated adjacent, light surface). The
# mapping is by entity, not rank: an outcome keeps its color in every chart.
OUTCOME_COLORS = {
    "agreement": "#2a78d6",
    "disagreement": "#eb6834",
    "mixed_or_conditional": "#1baf7a",
    "epistemic_uncertainty": "#eda100",
    "irrelevant": "#e87ba4",
    "refusal": "#008300",
    "safety_refusal": "#4a3aa7",
    # Non-positions that are not refusals render as neutral grays of
    # distinct lightness, not hues - they carry no stance to color.
    "neutral_or_no_position": "#c3c2b7",
    "malformed": "#898781",
    "generation_error": "#52514e",
}
OUTCOME_ORDER = list(OUTCOME_COLORS)

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new(figsize) -> tuple:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2, "axes.titlecolor": INK})
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, out_base: Path, *, source_artifact: Path, description: str) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "pdf", "png"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        paths.append(p)
    plt.close(fig)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    sidecar = {
        "description": description,
        "source_artifact": str(source_artifact),
        "source_artifact_sha256": hashlib.sha256(source_artifact.read_bytes()).hexdigest(),
        "generated_by": Path(sys.argv[0]).name,
        "anatomiae_git_commit": commit,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "outputs": [p.name for p in paths],
    }
    out_base.with_suffix(".provenance.json").write_text(json.dumps(sidecar, indent=2))
    return paths


def confusion_heatmap(wide: pd.DataFrame, col_a: str, col_b: str, *, label_a: str, label_b: str,
                      title: str, subtitle: str):
    """Counts of (outcome under evaluator A, outcome under evaluator B) on the
    same generations. Single-hue ramp; zero cells stay near-surface; every
    cell carries its count (a table view in the figure itself)."""
    labels = [o for o in OUTCOME_ORDER if o in set(wide[col_a]) | set(wide[col_b])]
    counts = pd.crosstab(wide[col_a], wide[col_b]).reindex(index=labels, columns=labels, fill_value=0)
    fig, ax = _new((1.3 + 1.05 * len(labels), 1.1 + 0.8 * len(labels)))
    ax.imshow(counts.to_numpy(), cmap=SEQ_BLUE, vmin=0, aspect="auto")
    vmax = counts.to_numpy().max()
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = counts.iat[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    color="#ffffff" if v > 0.55 * vmax else INK)
    short = [x.replace("_or_", "/").replace("_", " ") for x in labels]
    ax.set_xticks(range(len(labels)), short, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), short)
    ax.set_xlabel(label_b)
    ax.set_ylabel(label_a)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_title(f"{title}\n", loc="left", fontsize=11, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)
    return fig


def agreement_dotplot(summary: pd.DataFrame, *, category_col: str, series_col: str,
                      title: str, subtitle: str, xlabel: str = "Outcome agreement (share of pairs)"):
    """Agreement rate with bootstrap 95% CI, one row per category, one
    color per series (<= 3 series, fixed slot order). Point + CI whisker,
    n printed at each point - no bars implying a meaningful zero baseline
    beyond the [0, 1] scale itself."""
    series = list(dict.fromkeys(summary[series_col]))
    if len(series) > len(SERIES):
        raise ValueError(f"{len(series)} series > {len(SERIES)} validated slots - facet instead")
    cats = list(dict.fromkeys(summary[category_col]))
    fig, ax = _new((7.2, 0.55 * len(cats) * max(1, len(series)) + 1.4))
    _style(ax)
    step = 0.8 / max(1, len(series))
    for si, s in enumerate(series):
        sub = summary[summary[series_col] == s].set_index(category_col)
        for ci, c in enumerate(cats):
            if c not in sub.index:
                continue
            r = sub.loc[c]
            y = ci + (si - (len(series) - 1) / 2) * step
            ax.plot([r.outcome_agreement_ci95_low, r.outcome_agreement_ci95_high], [y, y],
                    color=SERIES[si], linewidth=2, solid_capstyle="round")
            ax.plot(r.outcome_agreement, y, "o", markersize=8, color=SERIES[si],
                    markeredgecolor=SURFACE, markeredgewidth=2, label=s if ci == 0 else None)
            ax.text(min(r.outcome_agreement_ci95_high + 0.02, 1.02), y,
                    f"{r.outcome_agreement:.2f} (n={int(r.n_pairs)})", va="center",
                    fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(cats)), [str(c) for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}\n", loc="left", fontsize=11, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)
    if len(series) >= 2:
        # below the x-axis label: inside the plot it collides with data labels
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=len(series))
    return fig


def outcome_stack(long: pd.DataFrame, *, group_col: str, title: str, subtitle: str):
    """100%-stacked horizontal bars of outcome shares per group (e.g. per
    evaluator), fixed outcome colors, 2px surface gap between segments,
    shares >= 8% labeled in-segment, legend always present."""
    groups = list(dict.fromkeys(long[group_col]))
    shares = (long.groupby([group_col, "outcome"]).size().unstack(fill_value=0)
              .reindex(index=groups).fillna(0))
    shares = shares.div(shares.sum(axis=1), axis=0)
    unmapped = set(shares.columns) - set(OUTCOME_COLORS)
    if unmapped:
        # never silently drop an outcome from a 100% stack
        raise ValueError(f"outcomes without a fixed color: {sorted(unmapped)}")
    present = [o for o in OUTCOME_ORDER if o in shares.columns]
    fig, ax = _new((7.6, 0.6 * len(groups) + 1.6))
    _style(ax)
    left = pd.Series(0.0, index=groups)
    for o in present:
        vals = shares[o]
        ax.barh(groups, vals, left=left, color=OUTCOME_COLORS[o], edgecolor=SURFACE,
                linewidth=2, height=0.6, label=o.replace("_or_", "/").replace("_", " "))
        for g in groups:
            if vals[g] >= 0.08:
                ax.text(left[g] + vals[g] / 2, g, f"{vals[g]:.0%}", ha="center", va="center",
                        fontsize=8.5, color=INK)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(f"{title}\n", loc="left", fontsize=11, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), labelcolor=INK_2)
    return fig


def curve_small_multiples(df: pd.DataFrame, *, x: str, panels: list[tuple[str, str]],
                          markers: dict[str, float], title: str, subtitle: str,
                          chance: dict[str, float] | None = None):
    """One metric per panel, one hue (slot 1), shared x. Selected
    checkpoints drawn as labeled vertical rules; an optional chance line per
    panel. Small multiples instead of 4 overlapping colored lines: only
    three categorical slots validate for crossing marks."""
    n = len(panels)
    cols = 2
    rows = (n + 1) // 2
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK_2})
    fig, axes = plt.subplots(rows, cols, figsize=(9, 2.6 * rows + 0.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (col, label) in zip(axes.flat, panels, strict=False):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        d = df.dropna(subset=[col])
        ax.plot(d[x], d[col], color=SERIES[0], linewidth=2)
        if chance and col in chance:
            ax.axhline(chance[col], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(d[x].max(), chance[col], " chance", va="bottom", ha="right", fontsize=8, color=MUTED)
        for name, xv in markers.items():
            ax.axvline(xv, color=BASELINE, linewidth=1)
            ax.text(xv, 1.0, f" {name}", transform=ax.get_xaxis_transform(), fontsize=8,
                    color=INK_2, va="top")
        ax.set_title(label, loc="left", fontsize=10, color=INK)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Amber checkpoint index (ckpt_NNN)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.93, subtitle, fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
