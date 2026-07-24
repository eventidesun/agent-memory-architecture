"""Figures from raw records. Reads results/raw/*.jsonl and writes to
results/figures/. Computes nothing that analysis/metrics.py does not.

Three figures:
  fig2_forgetting_curve   retention against gold loss across tau
  fig3_ablations          recall@3 by category, six ablations
  fig4_sensitivity        the repetition dissociation across parameter ranges

Figure 1 (the P04/P05 rank inversion) is a table rather than a plot and is
handled separately.

Each figure writes both PDF (for the paper) and PNG (for looking at).
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.metrics import load, recall_at_k, by_category, by_condition

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(_ROOT, "results", "raw")
OUT = os.path.join(_ROOT, "results", "figures")

K = 3

# Muted palette; readable in greyscale, which conference print often is.
INK = "#222222"
ACCENT = "#B4433A"
MUTED = "#7A8B99"
GRID = "#DDDDDD"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 2 — the forgetting curve
# ---------------------------------------------------------------------------

def fig_forgetting_curve(data=None):
    """Retention against gold loss across the threshold range.

    data: list of (tau, mean_returned, gold_lost, n_probes). Passed in rather
    than recomputed, because producing it requires running the retriever and
    figures should not run experiments.
    """
    if data is None:
        data = [
            (None,  49.0,  0, 64),
            (1.817, 26.5,  1, 64),
            (3.0,   16.6,  4, 64),
            (4.5,    7.5, 16, 64),
            (6.0,    2.6, 33, 64),
        ]

    n_store = 49
    retention = [100 * d[1] / n_store for d in data]
    loss = [100 * d[2] / d[3] for d in data]
    labels = ["no threshold" if d[0] is None else f"\u03c4 = {d[0]}" for d in data]

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.plot(retention, loss, "-", color=MUTED, linewidth=1.2, zorder=1)
    ax.scatter(retention, loss, s=34, color=INK, zorder=3)

    # highlight the derived value
    for i, d in enumerate(data):
        if d[0] == 1.817:
            ax.scatter([retention[i]], [loss[i]], s=110, facecolors="none",
                       edgecolors=ACCENT, linewidths=1.6, zorder=4)

    offsets = [(-2, 5), (4, 5), (4, 5), (4, 4), (4, -2)]
    for i, lab in enumerate(labels):
        dx, dy = offsets[i]
        ax.annotate(lab, (retention[i], loss[i]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=INK)

    ax.annotate("derived", (retention[1], loss[1]),
                textcoords="offset points", xytext=(6, -12),
                fontsize=7.5, color=ACCENT, style="italic")

    ax.set_xlabel("Store retained (%)")
    ax.set_ylabel("Probes where the gold memory\nfell below threshold (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(-3, 58)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig2_forgetting_curve")


# ---------------------------------------------------------------------------
# Figure 3 — ablations
# ---------------------------------------------------------------------------

ABLATION_ORDER = [
    ("E_full", "full model"),
    ("E_no_person_conditioning", "\u2212 person conditioning"),
    ("E_no_spreading", "\u2212 spreading activation"),
]

NULL_ABLATIONS = ["E_no_decay", "E_no_salience", "E_no_reinforcement"]

CAT_LABEL = {"LEX": "Lexical", "PAR": "Paraphrase", "PC_": "Person",
             "REC": "Recency", "REP": "Repetition", "SAL": "Salience"}


def fig_ablations():
    recs = load(os.path.join(RAW, "ablations.jsonl"))
    conds = by_condition(recs)
    cats = ["LEX", "PAR", "PC_", "REC", "REP", "SAL"]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    n = len(ABLATION_ORDER)
    width = 0.8 / n
    xs = range(len(cats))

    for i, (name, label) in enumerate(ABLATION_ORDER):
        if name not in conds:
            continue
        vals = []
        for cat in cats:
            sub = [r for r in conds[name] if r["probe_id"][:3] == cat]
            v = recall_at_k(sub, K)
            vals.append(100 * v if v is not None else 0)
        offset = (i - (n - 1) / 2) * width
        colour = INK if name == "E_full" else MUTED
        alpha = 1.0 if name == "E_full" else 0.35 + 0.11 * i
        ax.bar([x + offset for x in xs], vals, width * 0.92,
               label=label, color=colour, alpha=alpha, zorder=3)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([CAT_LABEL[c] for c in cats])
    ax.set_ylabel(f"Recall@{K} (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    save(fig, "fig3_ablations")


# ---------------------------------------------------------------------------
# Figure 4 — the repetition dissociation across the sensitivity range
# ---------------------------------------------------------------------------

PARAM_LABEL = {
    "lambda": "\u03bb  (salience \u2192 decay)",
    "noise_sigma": "\u03c3  (retrieval noise)",
    "d_base": "d  (base decay rate)",
    "w_total": "W  (total source activation)",
    "tau": "\u03c4  (retrieval threshold)",
}
FROZEN = {"lambda": 0.5, "noise_sigma": 1.2, "d_base": 0.5,
          "w_total": 11.0, "tau": 1.817}


def fig_sensitivity():
    recs = load(os.path.join(RAW, "sensitivity.jsonl"))
    l1 = load(os.path.join(RAW, "level1.jsonl"))

    c_rep = [r for r in l1
             if r["condition"] == "C_dense" and r["probe_id"][:3] == "REP"]
    c_baseline = 100 * recall_at_k(c_rep, K)

    grouped = defaultdict(lambda: defaultdict(list))
    for r in recs:
        grouped[r["sweep_param"]][r["sweep_value"]].append(r)

    params = [p for p in PARAM_LABEL if p in grouped]
    fig, axes = plt.subplots(1, len(params), figsize=(11, 2.9), sharey=True)
    if len(params) == 1:
        axes = [axes]

    for ax, param in zip(axes, params):
        values = list(grouped[param].keys())
        xs, ys, frozen_i = [], [], None
        for i, v in enumerate(values):
            sub = [r for r in grouped[param][v] if r["probe_id"][:3] == "REP"]
            xs.append(i)
            ys.append(100 * recall_at_k(sub, K))
            if v == FROZEN.get(param):
                frozen_i = i

        ax.axhline(c_baseline, color=ACCENT, linewidth=1.1,
                   linestyle="--", zorder=2)
        ax.plot(xs, ys, "-", color=MUTED, linewidth=1.2, zorder=3)
        ax.scatter(xs, ys, s=26, color=INK, zorder=4)
        if frozen_i is not None:
            ax.scatter([xs[frozen_i]], [ys[frozen_i]], s=95, facecolors="none",
                       edgecolors=ACCENT, linewidths=1.5, zorder=5)

        ax.set_xticks(xs)
        ax.set_xticklabels(
            ["none" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
             for v in values],
            fontsize=7.5)
        ax.set_xlabel(PARAM_LABEL[param], fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(f"Repetition recall@{K} (%)")
    axes[-1].annotate("dense retrieval", (len(xs) - 1, c_baseline),
                      textcoords="offset points", xytext=(-4, 5),
                      fontsize=7.5, color=ACCENT, ha="right")
    fig.suptitle("Circled point marks the frozen value", fontsize=8,
                 color=MUTED, y=1.03)
    save(fig, "fig4_sensitivity")


if __name__ == "__main__":
    fig_forgetting_curve()
    fig_ablations()
    fig_sensitivity()