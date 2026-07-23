"""
Condition C — dense semantic retrieval

This is the field-standard baseline and the condition that produced the P04/P05 result

The class is a thin wrapper around `memory.retrieve_scored()`
It uses the same embedder, L2 distance calculation, top-k value, and result
ordering as the code used to generate `test_results.csv`

Its only role is to convert `memory.ScoredMemory`, which stores a `distance`,
into `base.ScoredMemory`, which stores the value as `score`.

In this condition, `score` is an L2 distance, so lower values are better
This is different from Condition E, where activation scores are better when they
are higher. Because the score directions and scales differ,
conditions should only be compared by rank.
"""


from retrievers.base import Context, ScoredMemory
from memory import retrieve_scored


class DenseRetriever:
    name = "C_dense"

    def retrieve(self, query: str, ctx: Context, k: int) -> list[ScoredMemory]:
        raw = retrieve_scored(ctx.speaker, query, n_results=k)

        return [
            ScoredMemory(
                memory_id=m.memory_id,
                score=m.distance, # L2 distance, the lower, the better
                rank=m.rank, # rank is already set and not recomputed
                text=m.text,
            )
            for m in raw
        ]