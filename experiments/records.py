"""One raw record per probe x condition. Metrics are computed downstream
from these records, never inline (spec invariant 4)."""

import json
import subprocess


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def params_snapshot() -> dict:
    from activation.params import (
        D_BASE, NOISE_SIGMA, W, TAU, LAMBDA, MIN_ELAPSED, SIMILARITY,
    )
    return {
        "d_base": D_BASE, "noise_sigma": NOISE_SIGMA, "w": W,
        "tau": TAU, "lambda": LAMBDA, "min_elapsed": MIN_ELAPSED,
        "similarity": SIMILARITY,
    }


def build_record(probe: dict, condition: str, seed: int,
                 ranked: list, extras: dict = None) -> dict:
    """ranked: list[ScoredMemory] from a retriever.
    extras: condition-specific fields, e.g. {"inferred_type": ...} for D."""
    ranked_ids = [m.memory_id for m in ranked]

    gold_ranks = [
        ranked_ids.index(g) + 1 for g in probe["gold_ids"] if g in ranked_ids
    ]
    gold_rank = min(gold_ranks) if gold_ranks else None

    record = {
        "probe_id": probe["probe_id"],
        "condition": condition,
        "seed": seed,
        "embedder": "all-MiniLM-L6-v2",
        "metric_space": {"C_dense": "l2", "D_typed": "l2",
                         "E_activation": "activation"}.get(condition, "none"),
        "params": params_snapshot(),
        "git_sha": git_sha(),
        "ranked": [
            {"memory_id": m.memory_id, "score": m.score, "rank": m.rank}
            for m in ranked
        ],
        "gold_ids": probe["gold_ids"],
        "gold_rank": gold_rank,
        "distractor_at_1": (
            bool(ranked) and ranked[0].memory_id in probe["distractor_ids"]
        ),
        "property_tags": probe.get("property_tags", []),
        "pair_id": probe.get("pair_id"),
    }
    if extras:
        record.update(extras)
    return record


def append_record(record: dict, path: str) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")