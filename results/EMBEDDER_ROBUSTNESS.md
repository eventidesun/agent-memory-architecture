# Embedder Robustness

Run: 2026-07-23, script: `experiments/run_embedder_robustness.py`
Raw: `results/raw/embedder_robustness.jsonl`
Predictions committed before running (PREDICTIONS.md).

## Question

Does the P04 retrieval outcome depend on the specific embedding model or
distance metric used, or does it hold across dense retrievers generally?

## Setup

Eight memories (M01–M08) and eight probes (P01–P08), identical to the
frozen baseline in `test_results.csv`. Four configurations:
{all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5} × {L2, cosine}. Self-contained
in-memory collections; the frozen retrieval path was not modified.

## Results

| probe | model | space | gold rank | distractor rank | inverted |
|---|---|---|---|---|---|
| P04 | all-MiniLM-L6-v2 | l2 | 4 | 1 | yes |
| P04 | all-MiniLM-L6-v2 | cosine | 4 | 1 | yes |
| P04 | bge-small-en-v1.5 | l2 | 2 | 1 | yes |
| P04 | bge-small-en-v1.5 | cosine | 2 | 1 | yes |
| P05 | all-MiniLM-L6-v2 | l2 | 1 | 2 | no |
| P05 | all-MiniLM-L6-v2 | cosine | 1 | 2 | no |
| P05 | bge-small-en-v1.5 | l2 | 1 | 2 | no |
| P05 | bge-small-en-v1.5 | cosine | 1 | 2 | no |

All three preregistered predictions confirmed.

1. MiniLM/L2 reproduces the frozen baseline (P04 gold rank 4, distractor
   rank 1). ✓
2. L2 and cosine produce identical rankings in all eight rows — both
   models normalize their embeddings. ✓
3. BGE-small inverts P04 as well, with a narrower margin (gold rank
   4 → 2). ✓

## Interpretation

The P04 outcome is not a property of all-MiniLM-L6-v2 or of the distance
metric. It replicates across two embedding families and both metrics.
P05 is stable at gold rank 1 in all four configurations, so the pair
behaves consistently as a controlled comparison.

BGE -- trained specifically for retrieval rather than general sentence
similarity: moves the gold memory from rank 4 to rank 2 without
displacing the distractor from rank 1. The effect is graded rather than
binary: stronger retrieval-tuned embedders mitigate it but do not
eliminate it. This is the measured answer to "wouldn't a better
embedding model fix this?"

ChromaDB's L2 distance was a library default rather than a deliberate
choice. The identical rankings under cosine rule out the metric as an
explanation for the finding.

## Scope

Two models, one corpus, and eight probes. Here, the result supports generalizing
across embedding models and distance metrics. Generalizing across
corpora requires a second corpus and remains open.