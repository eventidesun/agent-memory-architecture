"""Condition D — schema/typed retrieval.

The query is classified into one memory type at retrieval time. The
retriever then searches only memories of that type and ranks them by
semantic similarity.

The classifier does not have access to the probe labels. If no stored
memories match the inferred type, the retriever returns no results
rather than falling back to dense retrieval.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from retrievers.base import Context, ScoredMemory
from schema import MEMORY_TYPES
from memory import get_or_create_collection

load_dotenv()

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

INFERENCE_MODEL = "openai/gpt-4o"
INFERENCE_TEMPERATURE = 0.0

_PROMPT = """Classify what kind of memory this question is asking about.

Return exactly one word from this list, nothing else:
{types}

  fact                 - stable information about the person or world
  preference           - something the person likes, dislikes, or wants
  promise              - a commitment made by either party
  conflict             - disagreement, friction, or a stated grievance
  emotional_state      - how someone felt at a moment in time
  relationship_update  - a change in how the two parties relate
  narrative_event      - something that happened in the shared story"""


def infer_type(query: str) -> str:
    response = _client.chat.completions.create(
        model=INFERENCE_MODEL,
        temperature=INFERENCE_TEMPERATURE,
        messages=[
            {"role": "system", "content": _PROMPT.format(types=list(MEMORY_TYPES))},
            {"role": "user", "content": query},
        ],
        max_tokens=20,
    )
    inferred = response.choices[0].message.content.strip().lower()

    if inferred not in MEMORY_TYPES:
        raise ValueError(f"Type inference returned unknown type {inferred!r}")

    return inferred


class TypedRetriever:
    name = "D_typed"

    def retrieve(self, query: str, ctx: Context, k: int) -> list[ScoredMemory]:
        collection = get_or_create_collection(ctx.corpus)
        if collection.count() == 0:
            return []

        inferred = infer_type(query)
        self.last_inferred_type = inferred

        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            where={"type": inferred},
            include=["documents", "distances"],
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]

        return [
            ScoredMemory(memory_id=mem_id, score=dist, rank=rank, text=text)
            for rank, (mem_id, text, dist) in enumerate(
                zip(ids, documents, distances), start=1
            )
        ]