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
K = 49
OUT = "results/raw/level1.jsonl"


def run(
    probes_path: str,
    participant_id: str,
    current_time: float,
    session_id: int,
) -> None:
    with open(probes_path, encoding="utf-8") as f:
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
            corpus=probe.get("corpus", "primary"),
        )

        for retriever in retrievers:
            ranked = retriever.retrieve(probe["query"], ctx, K)

            extras = {}
            if retriever.name == "D_typed":
                extras["inferred_type"] = retriever.last_inferred_type

            record = build_record(
                probe,
                retriever.name,
                SEED,
                ranked,
                extras,
            )
            append_record(record, OUT)

            print(
                f"{probe['probe_id']}  "
                f"{retriever.name:15s}  "
                f"gold_rank={record['gold_rank']}  "
                f"d@1={record['distractor_at_1']}"
            )


if __name__ == "__main__":
    run(
        "benchmark/probes.json",
        participant_id="P01",
        current_time=6.0,
        session_id=6,
    )