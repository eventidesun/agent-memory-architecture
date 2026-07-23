"""Level 1: mechanism evaluation. C, D, E -> ranked IDs on all probes.
A and B run for documentation but are excluded from top-k comparison
(DESIGN 7.1). E runs reinforce=False: evaluation must not mutate the
store. Raw records only; metrics downstream."""

import json
import os

from retrievers.base import Context
from retrievers.stateless import StatelessRetriever
from retrievers.full_history import FullHistoryRetriever
from retrievers.dense import DenseRetriever
from retrievers.typed import TypedRetriever
from retrievers.activation import ActivationRetriever
from experiments.records import build_record, append_record

SEED = 0
K = 3
OUT = "results/raw/level1.jsonl"


def run(probes_path: str, participant_id: str, current_time: float,
        session_id: int) -> None:
    with open(probes_path) as f:
        probes = json.load(f)

    retrievers = [
        StatelessRetriever(),
        FullHistoryRetriever(),
        DenseRetriever(),
        TypedRetriever(),
        ActivationRetriever(seed=SEED, reinforce=False),
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    for probe in probes:
        ctx = Context(
            speaker=probe.get("speaker", participant_id),
            current_time=probe.get("current_time", current_time),
            session_id=probe.get("session_id", session_id),
        )
        for r in retrievers:
            ranked = r.retrieve(probe["query"], ctx, K)

            extras = {}
            if r.name == "D_typed":
                extras["inferred_type"] = r.last_inferred_type

            record = build_record(probe, r.name, SEED, ranked, extras)
            append_record(record, OUT)
            print(f"{probe['probe_id']}  {r.name:15s}  "
                  f"gold_rank={record['gold_rank']}  "
                  f"d@1={record['distractor_at_1']}")


if __name__ == "__main__":
    run("benchmark/probes.json", participant_id="PILOT",
        current_time=5.0, session_id=5)