"""Ablations on Condition E. Each knocks out one component and reruns the
full benchmark, so a behaviour can be attributed to a mechanism rather than
observed in aggregate.

All ablations are argument or wrapper changes. Nothing in the activation
package is edited, and the store is never modified — the reinforcement
ablation truncates presentation logs at read time.

Records use the same schema as Level 1 with an added `ablation` field, so
analysis/metrics.py reads them unchanged."""

import json
import os

from retrievers.base import Context
from retrievers.activation import ActivationRetriever
from activation.params import W, LAMBDA, D_BASE
from experiments.records import build_record, append_record

SEED = 0
K = 49
CURRENT_TIME = 6.0
SESSION_ID = 6
CORPUS = "primary"
OUT = "results/raw/ablations.jsonl"


class TruncatedLogRetriever(ActivationRetriever):
    """E with every presentation log truncated to its first entry.

    Removes reinforcement without touching the store: the retriever reads the
    log, keeps only the write, and computes B_i from that. All other memories
    and all other conditions see the corpus unchanged.
    """
    name = "E_no_reinforcement"

    def retrieve(self, query, ctx, k):
        import json as _json
        from memory import get_or_create_collection
        from activation.base_level import base_level
        from activation.spreading import spreading
        from retrievers.base import ScoredMemory

        collection = get_or_create_collection(ctx.corpus)
        if collection.count() == 0:
            return []

        stored = collection.get(include=["documents", "metadatas", "embeddings"])
        query_embedding = collection._embedding_function([query])[0]

        scored = []
        for mem_id, text, meta, emb in zip(
            stored["ids"], stored["documents"],
            stored["metadatas"], stored["embeddings"],
        ):
            meta = meta or {}
            log = _json.loads(meta.get("presentation_log", "[]"))
            log = log[:1]                      # <-- the ablation
            salience = float(meta.get("salience", 0.0))

            b = base_level(log, ctx.current_time, salience, self.lam)
            s = spreading(query_embedding, list(emb),
                          meta.get("speaker", ""), ctx.speaker, self.w)
            a = b + s + self._rng.gauss(0.0, self.noise_sigma)
            scored.append((a, mem_id, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        if self.tau is not None:
            scored = [t for t in scored if t[0] > self.tau]

        return [
            ScoredMemory(memory_id=mid, score=a, rank=rank, text=text)
            for rank, (a, mid, text) in enumerate(scored[:k], start=1)
        ]


def make_retrievers():
    """Full E plus five knockouts. Each returns a fresh retriever with its own
    seeded RNG, so the noise sequence is identical across ablations and any
    difference is attributable to the component, not to sampling."""

    full = ActivationRetriever(seed=SEED, reinforce=False)
    full.name = "E_full"

    no_decay = ActivationRetriever(seed=SEED, reinforce=False, d_base=0.0)
    no_decay.name = "E_no_decay"

    no_salience = ActivationRetriever(seed=SEED, lam=0.0, reinforce=False)
    no_salience.name = "E_no_salience"

    no_speaker = ActivationRetriever(
        seed=SEED, reinforce=False,
        w={"utterance": W["utterance"], "speaker": 0.0})
    no_speaker.name = "E_no_person_conditioning"

    no_spreading = ActivationRetriever(
        seed=SEED, reinforce=False,
        w={"utterance": 0.0, "speaker": 0.0})
    no_spreading.name = "E_no_spreading"

    no_reinforcement = TruncatedLogRetriever(seed=SEED, reinforce=False)

    return [full, no_decay, no_salience, no_speaker,
            no_spreading, no_reinforcement]


def run(probes_path="benchmark/probes.json"):
    with open(probes_path) as f:
        probes = json.load(f)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)

    for r in make_retrievers():
        for probe in probes:
            ctx = Context(
                speaker=probe["speaker"],
                current_time=probe.get("current_time", CURRENT_TIME),
                session_id=probe.get("session_id", SESSION_ID),
                corpus=probe.get("corpus", CORPUS),
            )
            ranked = r.retrieve(probe["query"], ctx, K)
            record = build_record(probe, r.name, SEED, ranked,
                                  {"ablation": r.name})
            append_record(record, OUT)
        print(f"{r.name:28} done")


if __name__ == "__main__":
    run()