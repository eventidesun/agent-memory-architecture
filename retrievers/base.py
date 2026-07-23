"""
retrievers/base.py

For all the five conditions to implement - the one interface all 
five memory conditions (A, B, C, D, E) have to follow so the experiment 
runner can treat them interchangeably.

So the experiment runner interacts with every condition ONLY through this interface, 
that is the if a condition needs information the interface doesn't carry, the interface
is wrong
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

@dataclass(frozen=True)
class Context:
    speaker: str
    current_time: float
    session_id: int
    corpus: str

@dataclass(frozen=True)
class ScoredMemory:
    memory_id: str
    score: float
    rank: int # 1-indexed
    text: str


@runtime_checkable
class Retriever(Protocol):
    # one method and identical signature for all 5 conditions
    name: str

    def retrieve(self, query: str, ctx: Context, k: int) -> list[ScoredMemory]:
        pass