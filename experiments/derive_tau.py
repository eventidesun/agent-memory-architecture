"""
scripts/derive_tau.py
Derives the retrieval threshold (tau) from the corpus's activation distribution.
Run with tau=None so nothing is filtered; collects every activation Model E 
computes across generic queries. Derived strictly from corpus activation 
properties, never from probe outcomes.
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
QUERIES = [
    "what have we talked about?",
    "how are things going?",
    "what should I do tonight?",
    "what am I working on?",
    "tell me something about myself",
    "what did I say I would do?",
    "how have I been feeling?",
    "what do I like?",
]
def run(participants=("P01", "P02", "P03")):
    os.makedirs("results/raw", exist_ok=True)
    # Clear prior run output if present to prevent appending duplicates
    if os.path.exists(OUT):
        os.remove(OUT)
    e = ActivationRetriever(seed=SEED, tau=None, reinforce=False)
    all_activations = []
    with open(OUT, "a") as f:
        for pid in participants:
            col = get_or_create_collection("primary")
            n = col.count()
            ctx = Context(speaker=pid, current_time=CURRENT_TIME, session_id=6, corpus="primary")
            for q in QUERIES:
                for m in e.retrieve(q, ctx, n):
                    meta = col.get(ids=[m.memory_id], include=["metadatas"])["metadatas"][0]
                    rec = {
                        "participant": pid,
                        "query": q,
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
    print("==================================================")
    print("ACTIVATION DISTRIBUTION SUMMARY (tau = None)")
    print("==================================================")
    print(f"n      = {len(vals)}")
    print(f"min    = {vals[0]:.3f}")
    print(f"p10    = {vals[int(0.10 * len(vals))]:.3f}")
    print(f"p25    = {vals[int(0.25 * len(vals))]:.3f}")
    print(f"median = {statistics.median(vals):.3f}")
    print(f"p75    = {vals[int(0.75 * len(vals))]:.3f}")
    print(f"max    = {vals[-1]:.3f}\n")
    print("BY MEMORY PROFILE (MEAN ACTIVATION):")
    old_trivial = [
        r["activation"]
        for r in all_activations
        if r["session_id"] <= 2 and r["salience"] <= 0.3 and r["n_presentations"] == 1
    ]
    recent_salient = [
        r["activation"]
        for r in all_activations
        if r["session_id"] >= 4 and r["salience"] >= 0.6
    ]
    repeated = [
        r["activation"] for r in all_activations if r["n_presentations"] >= 3
    ]
    for name, group in [
        ("old/trivial/once", old_trivial),
        ("recent/salient", recent_salient),
        ("repeated 3x", repeated),
    ]:
        if group:
            mean_val = statistics.mean(group)
            std_val = statistics.stdev(group) if len(group) > 1 else 0.0
            print(f"  {name:20} n={len(group):4}  mean={mean_val:.3f}  std={std_val:.3f}")
        else:
            print(f"  {name:20} n=0")
    if old_trivial:
        tau_derived = statistics.mean(old_trivial)
        print("==================================================")
        print(f"DERIVED TAU ANCHOR VALUE: {tau_derived:.3f}")
        print("==================================================")
if __name__ == "__main__":
    run()