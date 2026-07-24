"""Ablation results by category. Reads records; computes nothing new."""

from analysis.metrics import load, by_condition, by_category, recall_at_k, mrr

K = 3
ORDER = ["E_full", "E_no_decay", "E_no_salience",
         "E_no_person_conditioning", "E_no_spreading", "E_no_reinforcement"]


def pct(x):
    return "  --  " if x is None else f"{100*x:5.1f}%"


def run():
    recs = load("results/raw/ablations.jsonl")

    print("=" * 82)
    print(f"ABLATIONS — recall@{K} overall")
    print("=" * 82)
    conds = by_condition(recs)
    baseline = recall_at_k(conds.get("E_full", []), K)
    for name in ORDER:
        if name not in conds:
            continue
        r = recall_at_k(conds[name], K)
        delta = "" if name == "E_full" else f"  ({100*(r-baseline):+5.1f} pts)"
        print(f"{name:28} {pct(r)}  MRR {mrr(conds[name]):.3f}{delta}")

    print()
    print("=" * 82)
    print(f"BY CATEGORY — recall@{K}")
    print("=" * 82)
    cats = sorted(by_category(recs))
    print(f"{'ablation':28}" + "".join(f"{c:>9}" for c in cats))
    for name in ORDER:
        if name not in conds:
            continue
        row = f"{name:28}"
        for cat in cats:
            sub = [r for r in conds[name] if r["probe_id"][:3] == cat]
            row += f"{pct(recall_at_k(sub, K)):>9}"
        print(row)

    print("\nA component is doing work where removing it moves the number.")
    print("Watch REP under no_reinforcement and PC_ under no_person_conditioning.")


if __name__ == "__main__":
    run()