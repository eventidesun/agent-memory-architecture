"""
labelling.py

Assigns metadata to each memory when it is stored. An LLM identifies the
memory type, speaker, salience, and importance.

This module is intentionally isolated from the benchmark and retrieval
code. It imports only schema, os, json, dotenv, and the OpenAI SDK. It
does not access benchmark probes or import chat.py or memory.py. This
separation is enforced by the import structure and checked by
tests/test_label_blindness.py.

The labelling prompt is corpus-independent. It describes only the
labelling task and does not refer to Lumen, the station, or any specific
setting. The same pipeline can therefore be used with the second corpus,
MSC/LoCoMo, without modification.
"""


import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from schema import MEMORY_TYPES

load_dotenv()

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Logged into every results file (build spec invariant 5).
LABELER_MODEL = "openai/gpt-4o"
LABELER_TEMPERATURE = 0.0

_PROMPT = """You label memories from a conversation between a person and an AI agent.

Given one exchange, return JSON with exactly these fields:

  "type": one of {types}
  "speaker": who produced the content this memory is about — use the
             speaker label given below, verbatim
  "salience": 0.0-1.0, how emotionally charged this exchange is for the
              person. 0 = neutral/administrative, 1 = highly emotional
              (distress, joy, conflict, vulnerability, strong affection)
  "importance": 0.0-1.0, how consequential this is for the ongoing
                relationship, independent of emotional charge. A calmly
                stated commitment is high importance, low salience.

Type definitions:
  fact                 - a stable piece of information about the person or world
  preference           - something the person likes, dislikes, or wants
  promise              - a commitment made by either party
  conflict             - disagreement, friction, or a stated grievance
  emotional_state      - how someone felt at a moment in time
  relationship_update  - a change in how the two parties relate
  narrative_event      - something that happened in the shared story

Return ONLY the JSON object. No preamble, no markdown fences."""


def label_memory(text: str, speaker: str) -> dict:
    """Label one memory. Returns dict with type, speaker, salience, importance.

    `text` is the raw exchange. `speaker` is passed in rather than inferred
    because the caller knows it and guessing would add avoidable noise.
    """
    response = _client.chat.completions.create(
        model=LABELER_MODEL,
        temperature=LABELER_TEMPERATURE,
        messages=[
            {"role": "system", "content": _PROMPT.format(types=list(MEMORY_TYPES))},
            {"role": "user", "content": f"Speaker: {speaker}\n\nExchange:\n{text}"},
        ],
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    parsed = json.loads(raw)

    if parsed["type"] not in MEMORY_TYPES:
        raise ValueError(f"Labeller returned unknown type {parsed['type']!r}")

    return {
        "type": parsed["type"],
        "speaker": speaker,
        "salience": float(parsed["salience"]),
        "importance": float(parsed["importance"]),
    }