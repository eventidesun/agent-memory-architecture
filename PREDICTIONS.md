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
