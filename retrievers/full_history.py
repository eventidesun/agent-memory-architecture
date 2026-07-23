"""
Condition B — full history (availability baseline).

Returns the entire store, UNRANKED. B has no retrieval mechanism: it is
"put the whole transcript in the prompt." It therefore has no opinion
about which memory is most relevant, and we do not invent one for it

`rank` here is position-in-storage-order, not relevance order. B is
excluded from Level 1 top-k comparisons and handled
separately; its measurements are tokens, latency, cost, and the
availability-utilization gap at Level 2

`k` is accepted to satisfy the interface and deliberately ignored and
B's defining property is that it does not select
"""

from retrievers.base import Context, ScoredMemory
from memory import get_or_create_collection

class FullHistoryRetriever:
    name = "B_full_history"

    def retrieve(self, query: str, ctx: Context, k: int) -> list[ScoredMemory]:
        collection = get_or_create_collection(ctx.speaker)
        if collection.count() == 0:
            return []

        results = collection.get()  # no query — everything, in storage order

        return [
            ScoredMemory(memory_id=mem_id, score=0.0, rank=i, text=text)
            for i, (mem_id, text) in enumerate(
                zip(results["ids"], results["documents"]), start=1
            )
        ]