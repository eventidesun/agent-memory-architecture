# Condition A, Stateless

from retrievers.base import Context, ScoredMemory


class StatelessRetriever:
    name = "A_stateless"

    def retrieve(self, query: str, ctx: Context, k: int) -> list[ScoredMemory]:
        return []