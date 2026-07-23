"""Spreading activation: S_i = sum_k W_k * S_ki over context cues.
Cues (DESIGN 5.3): current utterance embedding + speaker identity.

Speaker is a CUE, not a filter: a non-matching memory loses the
speaker term but remains in the competition and can still surface
on base-level + utterance similarity. This is the distinction from
per-user partitioning (AFA) — shared associative memory, graded
accessibility."""

import math


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def spreading(query_embedding: list[float],
              memory_embedding: list[float],
              memory_speaker: str,
              current_speaker: str,
              w: dict) -> float:
    s_utterance = cosine(query_embedding, memory_embedding)
    s_speaker = 1.0 if memory_speaker == current_speaker else 0.0
    return w["utterance"] * s_utterance + w["speaker"] * s_speaker