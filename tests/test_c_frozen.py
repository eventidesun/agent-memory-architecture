"""Spec invariant 1: Condition C reproduces test_results.csv exactly.
Any refactor must keep this passing. C's behaviour is frozen; only
logging may change."""

import csv
import os

import chromadb

from retrievers.base import Context
from retrievers.dense import DenseRetriever

TOLERANCE = 5e-4  # csv rounds to 4dp

MEMORIES = {
    "M01": "Participant: I came to the station alone. Nobody even knows where I am.\nLumen: That sounds isolating. I'm glad you're here talking with me.",
    "M02": "Participant: I need to leave by Friday to catch the supply boat back to the mainland.\nLumen: Understood — Friday it is. I'll make our remaining time count.",
    "M03": "Participant: What's behind that sealed door?\nLumen: The bioluminescence tank. I promise I'll show it to you next session — it's worth the wait.",
    "M04": "Participant: You forgot my name. That actually hurt. Trust matters a lot to me.\nLumen: I'm sorry. I understand — I'll be more careful with what you share.",
    "M05": "Participant: My sister is obsessed with marine biology documentaries, she watches them constantly.\nLumen: She'd love the station's archive footage then.",
    "M06": "Participant: That flooded lower lab we found was incredible.\nLumen: It was. Let's explore it together next time — agreed?",
    "M07": "Participant: Honestly, people forget plans all the time. It's no big deal in general.\nLumen: True, everyone slips sometimes.",
    "M08": "Participant: The generator's humming noise reminds me of my grandfather's boat. I loved that boat.\nLumen: A comforting sound then, not just machinery.",
}

PID = "diag01_frozen_test"


def seed():
    """Seed exactly as diagnostic_f2.py did: documents + ids, no metadata."""
    client = chromadb.Client()
    col = client.get_or_create_collection(name=f"participant_{PID}")
    if col.count() == 0:
        col.add(documents=list(MEMORIES.values()), ids=list(MEMORIES.keys()))
    return col


def load_expected(path=None):
    if path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "test_results.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def test_c_frozen():
    seed()
    c = DenseRetriever()
    ctx = Context(speaker=PID, current_time=0.0, session_id=1)

    failures = []
    for row in load_expected():
        ranked = c.retrieve(row["probe_text"], ctx, len(MEMORIES))
        ranked_ids = [m.memory_id for m in ranked]

        got_top1 = ranked_ids[0]
        got_top1_dist = ranked[0].score
        got_gold_rank = ranked_ids.index(row["gold_id"]) + 1
        got_gold_dist = ranked[got_gold_rank - 1].score
        got_dist_rank = ranked_ids.index(row["distractor_id"]) + 1

        checks = [
            ("top1_id", got_top1, row["top1_id"]),
            ("gold_rank", got_gold_rank, int(row["gold_rank"])),
            ("distractor_rank", got_dist_rank, int(row["distractor_rank"])),
        ]
        for name, got, expected in checks:
            if got != expected:
                failures.append(f"{row['probe_id']} {name}: got {got}, expected {expected}")

        for name, got, expected in [
            ("top1_distance", got_top1_dist, float(row["top1_distance"])),
            ("gold_distance", got_gold_dist, float(row["gold_distance"])),
        ]:
            if abs(got - expected) > TOLERANCE:
                failures.append(
                    f"{row['probe_id']} {name}: got {got:.6f}, expected {expected} "
                    f"(diff {abs(got - expected):.6f})"
                )

    assert not failures, "C's frozen behaviour changed:\n" + "\n".join(failures)


if __name__ == "__main__":
    test_c_frozen()
    print("PASS — C reproduces test_results.csv")