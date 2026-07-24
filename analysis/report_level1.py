"""Print the Level 1 summary. Reads records, computes nothing new."""

from analysis.metrics import (
    load, summarise, summarise_by_category, prediction_agreement,
    RANKED_CONDITIONS,
)

K = 3


def pct(x):
    return "  --  " if x is None else f"{100*x:5.1f}%"


def run():
    recs = load()

    print("=" * 78)
    print(f"LEVEL 1 SUMMARY   (n = {len(recs)} records)")
    print("=" * 78)
    print(f"{'condition':16} {'n':>4} {'found':>6} {'rec@1':>7} {f'rec@{K}':>7} "
          f"{'MRR':>6} {'miss':>7} {'d@1':>7} {'med rank':>9}")
    for cond, s in sorted(summarise(recs, K).items()):
        med = "  --" if s["median_rank"] is None else f"{s['median_rank']:.1f}"
        print(f"{cond:16} {s['n_total']:>4} {s['n_found']:>6} "
              f"{pct(s['recall@1'])} {pct(s[f'recall@{K}'])} "
              f"{s['mrr']:>6.3f} {pct(s['total_miss'])} "
              f"{pct(s['distractor_at_1'])} {med:>9}")
    print("\nB returns everything unranked; its ranks are storage position.")
    print("Rank statistics are over found probes only — read them with 'found'.")

    print()
    print("=" * 78)
    print(f"BY CATEGORY   (recall@{K} / total miss)")
    print("=" * 78)
    grid = summarise_by_category(recs, K)
    print(f"{'category':10} {'n':>4}   " +
          "".join(f"{c:>22}" for c in RANKED_CONDITIONS))
    for cat in sorted(grid):
        n = next(iter(grid[cat].values()))["n"]
        cells = ""
        for c in RANKED_CONDITIONS:
            s = grid[cat].get(c)
            cells += (f"{pct(s[f'recall@{K}'])} /{pct(s['total_miss'])}".rjust(22)
                      if s else " " * 22)
        print(f"{cat:10} {n:>4}   {cells}")

    print()
    print("=" * 78)
    print("AGREEMENT WITH PREDICTION TABLE   (gold at rank 1)")
    print("=" * 78)
    for cond, a in sorted(prediction_agreement(recs).items()):
        scored = a["correct"] + a["wrong"]
        rate = f"{100*a['correct']/scored:5.1f}%" if scored else "  --  "
        print(f"{cond:16} correct {a['correct']:>3}  wrong {a['wrong']:>3}  "
              f"uncertain {a['uncertain']:>3}   {rate}")

    print("\nMissed predictions:")
    for cond, a in sorted(prediction_agreement(recs).items()):
        for probe_id, note in a["misses"]:
            print(f"  {cond:16} {probe_id:10} {note}")


if __name__ == "__main__":
    run()