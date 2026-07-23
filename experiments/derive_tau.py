"""
experiments/derive_tau.py

Derives the retrieval threshold (tau) from the corpus's activation
distribution. Runs E with tau=None so nothing is filtered, and collects
every activation across all query-memory pairs.

Queries are the corpus's own user messages — the same utterance
distribution used to derive the cue weights. Derived strictly from
corpus properties, never from probe outcomes.
"""

import json
import os
import statistics

from retrievers.base import Context
from retrievers.activation import ActivationRetriever
from memory import get_or_create_collection

SEED = 0
CURRENT_TIME = 6.0  # evaluated one session after the 5-session corpus ends
OUT = "results/raw/tau_derivation.jsonl"
CORPUS = "primary"


def user_message(document):
    first_line = document.split("\n")[0]
    return first_line.split(": ", 1)[1] if ": " in first_line else first_line


def run():
    os.makedirs("results/raw", exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)

    col = get_or_create_collection(CORPUS)
    stored = col.get(include=["documents", "metadatas"])
    n = col.count()

    meta_by_id = dict(zip(stored["ids"], stored["metadatas"]))
    queries = [(m["speaker"], user_message(d))
               for d, m in zip(stored["documents"], stored["metadatas"])]

    e = ActivationRetriever(seed=SEED, tau=None, reinforce=False)
    all_activations = []

    with open(OUT, "a") as f:
        for speaker, q in queries:
            ctx = Context(speaker=speaker, current_time=CURRENT_TIME,
                          session_id=6, corpus=CORPUS)
            for m in e.retrieve(q, ctx, n):
                meta = meta_by_id[m.memory_id]
                rec = {
                    "query_speaker": speaker,
                    "query": q[:60],
                    "memory_id": m.memory_id,
                    "activation": m.score,
                    "rank": m.rank,
                    "session_id": meta["session_id"],
                    "salience": meta["salience"],
                    "n_presentations": len(json.loads(meta["presentation_log"])),
                }
                all_activations.append(rec)
                f.write(json.dumps(rec) + "\n")

    vals = sorted(r["activation"] for r in all_activations)
    print("=" * 50)
    print("ACTIVATION DISTRIBUTION SUMMARY (tau = None)")
    print("=" * 50)
    print(f"n      = {len(vals)}")
    print(f"min    = {vals[0]:.3f}")
    print(f"p10    = {vals[int(0.10 * len(vals))]:.3f}")
    print(f"p25    = {vals[int(0.25 * len(vals))]:.3f}")
    print(f"median = {statistics.median(vals):.3f}")
    print(f"p75    = {vals[int(0.75 * len(vals))]:.3f}")
    print(f"max    = {vals[-1]:.3f}\n")

    print("BY MEMORY PROFILE (MEAN ACTIVATION):")
    old_trivial = [r["activation"] for r in all_activations
                   if r["session_id"] <= 2 and r["salience"] <= 0.3
                   and r["n_presentations"] == 1]
    recent_salient = [r["activation"] for r in all_activations
                      if r["session_id"] >= 4 and r["salience"] >= 0.6]
    repeated = [r["activation"] for r in all_activations
                if r["n_presentations"] >= 3]

    for name, group in [("old/trivial/once", old_trivial),
                        ("recent/salient", recent_salient),
                        ("repeated 3x", repeated)]:
        if group:
            sd = statistics.stdev(group) if len(group) > 1 else 0.0
            print(f"  {name:20} n={len(group):5}  "
                  f"mean={statistics.mean(group):.3f}  std={sd:.3f}")
        else:
            print(f"  {name:20} n=0")

    if old_trivial:
        print("=" * 50)
        print(f"DERIVED TAU ANCHOR VALUE: {statistics.mean(old_trivial):.3f}")
        print("=" * 50)


if __name__ == "__main__":
    run()