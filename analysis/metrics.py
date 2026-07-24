import json
import statistics
from collections import defaultdict

RANKED_CONDITIONS = ("C_dense", "D_typed", "E_activation")


def load(path="results/raw/level1.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(records, k):
    """Fraction of probes where the gold appeared at rank <= k."""
    if not records:
        return None
    hits = sum(1 for r in records
               if r["gold_rank"] is not None and r["gold_rank"] <= k)
    return hits / len(records)


def mrr(records):
    """Mean reciprocal rank. A miss contributes 0."""
    if not records:
        return None
    return statistics.mean(
        0.0 if r["gold_rank"] is None else 1.0 / r["gold_rank"]
        for r in records
    )


def rank_stats(records):
    """Mean and median gold rank over probes where the gold was found,
    with the count it was computed over. Never report these without n_found."""
    found = [r["gold_rank"] for r in records if r["gold_rank"] is not None]
    return {
        "n_total": len(records),
        "n_found": len(found),
        "mean_rank": statistics.mean(found) if found else None,
        "median_rank": statistics.median(found) if found else None,
    }


def distractor_at_1_rate(records):
    if not records:
        return None
    return sum(1 for r in records if r["distractor_at_1"]) / len(records)


def total_miss_rate(records):
    """Fraction of probes returning no gold at any rank."""
    if not records:
        return None
    return sum(1 for r in records if r["gold_rank"] is None) / len(records)


def by_condition(records, conditions=None):
    out = defaultdict(list)
    for r in records:
        if conditions is None or r["condition"] in conditions:
            out[r["condition"]].append(r)
    return dict(out)


def by_tag(records, tag):
    """Records carrying a given property tag."""
    return [r for r in records if tag in r.get("property_tags", [])]


def by_category(records):
    """Group by probe_id prefix: PAR, PC_, LEX, SAL, REP, REC."""
    out = defaultdict(list)
    for r in records:
        out[r["probe_id"][:3]].append(r)
    return dict(out)


def summarise(records, k=3):
    """One row per condition."""
    rows = {}
    for cond, recs in by_condition(records).items():
        stats = rank_stats(recs)
        rows[cond] = {
            **stats,
            f"recall@{k}": recall_at_k(recs, k),
            "recall@1": recall_at_k(recs, 1),
            "mrr": mrr(recs),
            "distractor_at_1": distractor_at_1_rate(recs),
            "total_miss": total_miss_rate(recs),
        }
    return rows


def summarise_by_category(records, k=3, conditions=RANKED_CONDITIONS):
    """condition x category grid. This is the dissociation table."""
    grid = {}
    for cat, cat_recs in by_category(records).items():
        grid[cat] = {}
        for cond, recs in by_condition(cat_recs, conditions).items():
            grid[cat][cond] = {
                "n": len(recs),
                f"recall@{k}": recall_at_k(recs, k),
                "recall@1": recall_at_k(recs, 1),
                "total_miss": total_miss_rate(recs),
                "distractor_at_1": distractor_at_1_rate(recs),
            }
    return grid


def prediction_agreement(records, probes_path="benchmark/probes.json", k=1):
    """Compare outcomes against the committed prediction table.

    Predicted 1 = gold at rank 1. 2 = fails. 3 = uncertain, excluded from
    the agreement rate and counted separately.
    """
    with open(probes_path) as f:
        probes = {p["probe_id"]: p for p in json.load(f)}

    letter = {"A_stateless": "A", "B_full_history": "B", "C_dense": "C",
              "D_typed": "D", "E_activation": "E"}

    out = defaultdict(lambda: {"correct": 0, "wrong": 0, "uncertain": 0,
                               "misses": []})
    for r in records:
        cond = letter.get(r["condition"])
        pred = probes[r["probe_id"]]["predicted"].get(cond)
        if pred is None:
            continue
        if pred == 3:
            out[r["condition"]]["uncertain"] += 1
            continue
        actual = 1 if (r["gold_rank"] is not None and r["gold_rank"] <= k) else 2
        if actual == pred:
            out[r["condition"]]["correct"] += 1
        else:
            out[r["condition"]]["wrong"] += 1
            out[r["condition"]]["misses"].append(
                (r["probe_id"], f"predicted {pred}, got {actual}")
            )
    return dict(out)