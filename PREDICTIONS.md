# PREDICTION TABLE — committed before implementing Condition E

## Legend

* 1 = predicted to retrieve the gold memory at rank 1
* 2 = predicted to fail (surface distractor, or miss the gold)
* 3 = uncertain

## Conditions

* A = stateless (no memory)
* B = full history (all memories in context)
* C = dense semantic retrieval (word/vector similarity) — frozen baseline
* D = schema / typed retrieval
* E = ACT-R-inspired activation retrieval (proposed)

| Category                |  A  |  B  |  C  |  D  |  E  |
| ----------------------- | :-: | :-: | :-: | :-: | :-: |
| Lexical match           |  1  |  1  |  1  |  1  |  1  |
| Paraphrase              |  2  |  1  |  2  |  3  |  1  |
| Recency                 |  2  |  1  |  2  |  3  |  1  |
| Repetition              |  2  |  1  |  2  |  3  |  1  |
| Emotional salience      |  2  |  3  |  2  |  3  |  1  |
| Person conditioning     |  2  |  3  |  2  |  1  |  1  |
| Old + trivial + one-off |  2  |  1  |  1  |  1  |  2  |

## Note on "Old + trivial + one-off"

* Something unimportant, said once, a long time ago.
* The kind of thing a human would normally forget.
* Model E is designed to forget this kind of memory, so E is predicted to fail (2) here ON PURPOSE — this is the honesty case that shows the benchmark is not rigged in E's favour.
* This is NOT one of the skill categories; it is an additional probe type testing correct forgetting.

## Probe-design dependencies (predictions that require careful probe construction)

* C, Recency (=2): assumes the old and recent memories are worded similarly, so C cannot win by word-overlap instead of recency.
* C, Repetition (=2): assumes the repeated and one-off memories are worded similarly, so C cannot win by word-overlap.
* C, Person conditioning (=2): assumes the two speakers' memories are worded similarly, so C cannot separate them by words alone.

We design each probe so that a system can only succeed for the RIGHT reason.

### Embedder robustness — predictions
Written before running run_embedder_robustness.py.

1. all-MiniLM-L6-v2 under L2 reproduces test_results.csv exactly,
   including P04 (gold M04 at rank 4, distractor M07 at rank 1).
2. L2 and cosine produce identical rankings for both models, since
   both normalize embeddings.
3. BGE-small reproduces the P04 inversion — the surface alignment
   between the casual probe and the dismissive memory is a property
   of the content, not of MiniLM's geometry. Possible smaller margin,
   since BGE is retrieval-tuned.

If 3 fails: the Figure 1 claim narrows from "dense retrieval as a
class" to a single-model finding, and the paper states this.

## Scaling predictions (preregistered)

Written before `experiments/run_scaling.py` exists and before any synthetic
memories are generated. The scaling experiment grows the shared store with
synthetic memories from additional speakers, holds the 64 probes and their
gold memories fixed, and sweeps store size across approximately an order of
magnitude (49 → ~530).

Memory count and speaker count increase together. This models the deployment
scenario the paper's motivation describes — an agent accumulating both more
memories and more users — rather than isolating speaker count as an
independent variable. Interpretations that depend on speaker count must be
stated as consistent-with rather than caused-by.

---

**S1 — Dense retrieval degrades with store size.**
Recall@3 for Condition C is expected to decrease as the memory store grows,
because additional semantically similar memories increase competition during
nearest-neighbour retrieval. Falsified if C remains approximately constant
across the evaluated store sizes.

**S2 — Full history becomes operationally infeasible.**
Condition B is expected to maintain complete memory availability regardless of
store size, but prompt size, latency, and inference cost are expected to
increase approximately linearly with the number of stored memories. The
purpose of measuring B is not to test whether these quantities increase — that
is close to arithmetic — but to quantify the rate of increase and identify the
point at which full-history prompting becomes impractical for deployment.

**S3 — Activation retrieval degrades more gracefully.**
Condition E is expected to lose less recall than Condition C as store size
increases, because retrieval depends on presentation history, salience, and
speaker identity in addition to semantic similarity.

**S4 — The C–E gap widens under scale.**
The performance difference between Conditions C and E is expected to increase
as the shared memory store grows. This follows from E's use of non-semantic
retrieval cues that continue to discriminate between competing memories even
when semantic competition increases. Falsified if the difference between C and
E remains approximately constant or decreases across the evaluated store sizes.

**S5 — Category-specific degradation.**
Degradation is expected to differ across benchmark categories:

| Category | Expected trend |
|---|---|
| Lexical match | Minimal degradation |
| Paraphrase | Moderate degradation |
| Emotional salience | Smaller degradation for E than C |
| Repetition | Larger advantage for E than C |
| Person-conditioning | Largest increase in the C–E gap, although absolute recall for E is not necessarily expected to be high |

The person-conditioning row predicts relative separation, not high absolute
performance. E's person-conditioning recall at the corpus scale was 41.7%; a
result that remains low while the gap over C widens confirms S5 rather than
contradicting it.

**S6 — Scaling preserves the qualitative ordering.**
The qualitative differences observed between C and E on the original corpus are
expected to persist across the evaluated store sizes. Absolute recall may
decline for both architectures, but increasing distractor pressure is not
expected to reverse their ordering on repetition-tagged probes.

---

**Scope.** These predictions concern the evaluated range only, approximately
49 to 530 memories. No claim is made about behaviour beyond it. Any result
stated in the paper should be phrased as "across the evaluated range" rather
than as an unbounded trend.

**Instrumentation.** The scaling runner writes one record per condition per
store size, carrying both retrieval metrics (recall@k, MRR, gold rank,
distractor-at-1) and operational metrics (prompt tokens, retrieval latency,
estimated cost). One schema for all five conditions: for A, C, D, and E,
prompt tokens counts the retrieved memories only; for B it counts the whole
store. Analysis reads these records and recomputes nothing.