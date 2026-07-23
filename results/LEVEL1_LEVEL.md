# Level 1 — first full benchmark run

Run: 2026-07-23 · script: `experiments/run_level1.py`
Raw: `results/raw/level1.jsonl` (320 records: 64 probes × 5 conditions)
Corpus: primary (49 memories, 3 speakers) · seed 0 · k = 49 (full ranking)
current_time = 6.0 · τ = 1.817 · W = {utterance: 8.39, speaker: 2.61}
Predictions committed before the run (benchmark/probes.json).

## Observations

**D's type inference matches the gold's stored type on 27/64 probes (42%).**
When they disagree, the gold is filtered out entirely: 37/64 probes return no
gold at any rank. The failure is not ranking — it is that write-time and
query-time classification of the same content diverge, and hard categorical
routing turns that divergence into a total miss.

The asymmetry is concentrated on one type. Twenty-two golds are stored as
emotional_state; query-time inference selected emotional_state once. The rest
were inferred as preference (9), fact (4), conflict (3), narrative_event (2),
promise (2). A query about what would help someone settle down reads as a
preference question even when the memory answering it was stored as an
emotional state.

Open question: whether this is architectural or an artifact of the two prompts
describing the type set differently. The labelling prompt and D's inference
prompt were written separately. This needs checking before the finding is
stated at architecture level.

**τ = 1.817 rarely binds.** E returned a ranked gold on 63/64 probes, with
gold ranks as deep as 23. The threshold is live but few memories fall below it
at these parameter values. Whether the model forgets at all is therefore a
question for the sensitivity sweep rather than something this run settles.

**B's gold ranks are storage position, not relevance**, and are excluded from
top-k comparison. They appear in the records for documentation only.

## Not yet analysed

Per-condition, per-category metrics; agreement with the prediction table;
distractor-at-1 rates. These require `analysis/metrics.py`, which does not
exist yet. No claims are made from the console output.