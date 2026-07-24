"""Sensitivity results. Reads records; computes nothing new.

Reports recall@3 by category for each parameter value, and the E-minus-C
repetition gap, which is the dissociation the paper claims."""

import json
import os
from collections import defaultdict

from analysis.metrics import load, recall_at_k, by_category

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = 3

FROZEN = {"lambda": 0.5, "noise_sigma": 1.2, "d_base": 0.5,
          "w_total": 11.0, "tau": 1.817}


def pct(x):
    return "  -- " if x is None else f"{100*x:4.1f}%"


def run():
    recs = load(os.path.join(_ROOT, "results", "raw", "sensitivity.jsonl"))

    # C's per-category recall, for the dissociation comparison
    l1 = load(os.path.join(_ROOT, "results", "raw", "level1.jsonl"))
    c_recs = [r for r in l1 if r["condition"] == "C_dense"]
    c_by_cat = {cat: recall_at_k(rs, K)
                for cat, rs in by_category(c_recs).items()}

    grouped = defaultdict(lambda: defaultdict(list))
    for r in recs:
        grouped[r["sweep_param"]][r["sweep_value"]].append(r)

    for param in grouped:
        print("=" * 78)
        print(f"{param}   (frozen value: {FROZEN.get(param)})")
        print("=" * 78)
        cats = sorted(by_category(recs))
        print(f"{'value':>10}  {'overall':>8}  " +
              "".join(f"{c:>8}" for c in cats) + f"{'REP-C':>9}")
        for value in grouped[param]:
            rs = grouped[param][value]
            mark = " *" if value == FROZEN.get(param) else "  "
            row = f"{str(value):>10}{mark}{pct(recall_at_k(rs, K)):>8}  "
            cat_recall = {c: recall_at_k(sub, K)
                          for c, sub in by_category(rs).items()}
            for c in cats:
                row += f"{pct(cat_recall.get(c)):>8}"
            rep = cat_recall.get("REP")
            gap = (f"{100*(rep - c_by_cat.get('REP', 0)):+6.1f}"
                   if rep is not None else "    --")
            print(row + f"{gap:>9}")
        print()

    print("* marks the frozen value.")
    print("REP-C is E's repetition recall minus C's, in points.")
    print("The claim holds if that column stays positive across the range.")


if __name__ == "__main__":
    run()