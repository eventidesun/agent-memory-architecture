### Decisions 

1. Time unit: session. Social continuity is session-scaled.
   Turn-level reported as a robustness check.
2. Presentation: initial write + every retrieval into context.
   Justified by the testing effect (Roediger & Karpicke, 2006).
   Semantic restatement: only exact re-retrieval of the same memory ID counts as a presentation; semantic restatement creates a new memory.
   - Rationale: detecting semantic restatement requires a similarity threshold, which would be a free parameter tuned outside the frozen set. Treating restatements as new memories avoids this at the cost of under-counting reinforcement for paraphrased repetition, which is stated as a limitation.
3. Cues k: current utterance embedding + speaker identity.
   Speaker identity is what makes person-conditioning canonical
   spreading activation rather than a bolt-on.
4. S_ki: embedding similarity as the semantic component, plus a
   person-conditioning term. Declared as an approximation of
   ACT-R's fan-based S_ki, not a reimplementation.
5. Salience: modulates B_i (higher initial encoding and/or reduced
   decay rate d). It is never an additive term. (McGaugh, 2004.)

### Condition D — type inference
D infers probe type via an LLM call at query time. D does NOT read the
probe's property tags. Rationale: reading the tag hands D an oracle
signal that C does not receive, making the C-vs-D comparison unfair.
D's type-inference accuracy is reported as a result.

### Retrieval threshold τ — derivation rule
Written before computing τ.

τ is the mean activation of memories with session_id ≤ 2, salience ≤ 0.3,
and exactly one presentation, computed across a fixed set of eight generic
queries at current_time = 6.0 (one session after the corpus ends).

Rationale: the threshold is anchored to the memory profile the model is
designed to shed — old, low-salience, unrepeated. Derived from corpus
properties only; no probe outcomes are consulted.