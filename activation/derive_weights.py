"""Derive cue weights for spreading activation from corpus properties.

Principle: each cue contributes equal variance to S_i. Cues differ in
natural scale — cosine similarity is continuous over a narrow band,
speaker identity is a binary indicator — and no prior justifies treating
one scale as intrinsically more informative.

W_TOTAL is Honda et al.'s reported weight on semantic spreading
activation (HAI '25). Their model has one cue, so this is an extension
of their setting, not a value they report for multiple cues.

Variance is computed over all query-memory pairs, matching what E sees
at runtime: every memory is scored on every retrieval, so the candidate
set is the whole store. Queries are the corpus's own user messages —
the utterance distribution E is actually queried with — so no external
query set needs justifying.

Reads only the corpus collection. No path to benchmark/.

Run: PYTHONPATH=. ./venv/bin/python activation/derive_weights.py
Paste the output into params.py.
"""

import statistics

from activation.spreading import cosine
from activation.params import W_TOTAL
from memory import get_or_create_collection


def user_message(document):
    """Recover the user side of a fused 'Participant: ...\\nLumen: ...' doc."""
    first_line = document.split("\n")[0]
    return first_line.split(": ", 1)[1] if ": " in first_line else first_line


def derive(corpus_name="primary"):
    col = get_or_create_collection(corpus_name)
    stored = col.get(include=["documents", "metadatas", "embeddings"])

    mem_embeddings = [list(e) for e in stored["embeddings"]]
    speakers = [m["speaker"] for m in stored["metadatas"]]

    queries = [user_message(d) for d in stored["documents"]]
    query_embeddings = [list(e) for e in col._embedding_function(queries)]

    s_utterance = []
    s_speaker = []

    for q_emb, q_speaker in zip(query_embeddings, speakers):
        for m_emb, m_speaker in zip(mem_embeddings, speakers):
            s_utterance.append(cosine(q_emb, m_emb))
            s_speaker.append(1.0 if m_speaker == q_speaker else 0.0)

    sigma_u = statistics.stdev(s_utterance)
    sigma_s = statistics.stdev(s_speaker)

    raw_u = 1.0 / sigma_u
    raw_s = 1.0 / sigma_s
    total = raw_u + raw_s

    w_u = W_TOTAL * raw_u / total
    w_s = W_TOTAL * raw_s / total

    print(f"corpus:            {corpus_name}")
    print(f"pairs:             {len(s_utterance)}")
    print(f"sigma utterance:   {sigma_u:.4f}")
    print(f"sigma speaker:     {sigma_s:.4f}")
    print(f"W_TOTAL:           {W_TOTAL}")
    print()
    print("Paste into params.py:")
    print(f'W = {{"utterance": {w_u:.4f}, "speaker": {w_s:.4f}}}')
    print(f'# derived {corpus_name}: sigma_u={sigma_u:.4f}, sigma_s={sigma_s:.4f}')

    return {"utterance": w_u, "speaker": w_s}


if __name__ == "__main__":
    derive()