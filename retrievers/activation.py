"""Condition E — ACT-R-inspired relational activation retrieval.

A_i = B_i + S_i + noise. Selection under a bounded retrieval budget:
memories below TAU are inaccessible this turn, not deleted.
Retrieval is a presentation (DESIGN 5.3 decision 2): returned
memories have current_time appended to their presentation_log."""

import json
import random

from activation.base_level import base_level
from activation.spreading import spreading
from activation.params import W, TAU, LAMBDA, NOISE_SIGMA
from retrievers.base import Context, ScoredMemory
from memory import get_or_create_collection


class ActivationRetriever:
    name = "E_activation"

    def __init__(self, seed: int, w=None, tau=TAU, lam=LAMBDA,
                 noise_sigma=NOISE_SIGMA, reinforce=True):
        self._rng = random.Random(seed)
        self.seed = seed
        self.w = w if w is not None else dict(W)
        self.tau = tau
        self.lam = lam
        self.noise_sigma = noise_sigma
        self.reinforce = reinforce

    def retrieve(self, query: str, ctx: Context, k: int) -> list[ScoredMemory]:
        collection = get_or_create_collection(ctx.speaker)
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
            log = json.loads(meta.get("presentation_log", "[]"))
            salience = float(meta.get("salience", 0.0))

            b = base_level(log, ctx.current_time, salience, self.lam)
            s = spreading(query_embedding, list(emb),
                          meta.get("speaker", ""), ctx.speaker, self.w)
            noise = self._rng.gauss(0.0, self.noise_sigma)
            a = b + s + noise

            scored.append((a, mem_id, text))

        scored.sort(key=lambda x: x[0], reverse=True)

        if self.tau is not None:
            scored = [t for t in scored if t[0] > self.tau]

        top = scored[:k]

        if self.reinforce:
            self._record_presentations(collection, [m for _, m, _ in top],
                                       stored, ctx.current_time)

        return [
            ScoredMemory(memory_id=mem_id, score=a, rank=rank, text=text)
            for rank, (a, mem_id, text) in enumerate(top, start=1)
        ]

    def _record_presentations(self, collection, memory_ids, stored, now):
        for mem_id in memory_ids:
            idx = stored["ids"].index(mem_id)
            meta = dict(stored["metadatas"][idx] or {})
            log = json.loads(meta.get("presentation_log", "[]"))
            log.append(now)
            meta["presentation_log"] = json.dumps(log)
            collection.update(ids=[mem_id], metadatas=[meta])