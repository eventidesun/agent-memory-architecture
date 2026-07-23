### Decisions — July 20th, 2026

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

### Condition D — type inference [July 23rd, 2026]
D infers probe type via an LLM call at query time. D does NOT read the
probe's property tags. Rationale: reading the tag hands D an oracle
signal that C does not receive, making the C-vs-D comparison unfair.
D's type-inference accuracy is reported as a result.