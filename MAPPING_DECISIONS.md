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
designed to shed, old, low-salience, unrepeated. Derived from corpus
properties only; no probe outcomes are consulted.

### Presentation logs — authoring rule

A memory receives an additional presentation in each later session where its
subject is raised again. Ten of 46 memories qualify; the remaining 36 have a
single presentation at their write session.

Logs are authored rather than accumulated from E's own retrievals, so that all
five conditions see identical presentation histories and no condition is
evaluated against a store shaped by another condition's behaviour.

### Cue weights — equal-variance derivation 

Principle (stated before computing): each cue contributes equal variance to
S_i. The cues differ in natural scale, cosine similarity is continuous over a
narrow band, speaker identity is a binary indicator, and no prior justifies
treating one scale as intrinsically more informative.

W_TOTAL = 11.0 is Honda et al.'s reported weight on semantic spreading
activation (HAI '25). Their model has one cue, so adopting it as a total for a
two-cue model is an extension of their setting, not a value they report.

Weights are set inversely proportional to each cue's standard deviation across
all query-memory pairs in the corpus, normalised to sum to W_TOTAL. Queries are
the corpus's own user messages, the utterance distribution E is queried with at
runtime. Variance is computed over all pairs without filtering, matching E's
candidate set, which is the whole store.

Computed by activation/derive_weights.py:
  sigma_utterance = 0.1469
  sigma_speaker   = 0.4716
  W = {"utterance": 8.3877, "speaker": 2.6123}

The resulting numbers are corpus-dependent by design, in the same way a
z-score's mean and SD are dataset-dependent without the standardisation
procedure itself being tuned. The principle is what is frozen; the numbers
follow from it.

### Retrieval threshold τ — derivation rule 
Written before computing τ.

τ is the mean activation of memories with session_id ≤ 2, salience ≤ 0.3, and
exactly one presentation, computed across the corpus's own user messages as
queries at current_time = 6.0 (one session after the corpus ends).

Rationale: the threshold is anchored to the memory profile the model is designed
to shed, old, low-salience, unrepeated. Derived from corpus properties only; no
probe outcomes are consulted.

Computed by experiments/derive_tau.py: τ = 1.817

### Benchmark construction — person-conditioning category 

A probe belongs to the person-conditioning category only when every distractor
is a valid answer to the same neutral query for some participant, and speaker
identity is the information needed to select among them.

Consequence: queries must be speaker-neutral. A query containing vocabulary
distinctive to one participant's memories (e.g. naming a specific pet or
instrument) is a lexical-match probe, not a person-conditioning probe.

Under this rule, work and career was excluded as an overlap topic: all three
participants discuss work, but the correct answer to a work question is a fact
about that participant rather than a relational stance that a neutral query
leaves ambiguous. Four topics qualify: background sound, dogs, darkness and
nighttime, and corrections the agent has made.

### Presentation logs — authoring rule
Clarified 2026-07-23 following a systematic audit. The original wording
("a memory receives an additional presentation in each later session where its
subject is raised again") proved ambiguous: applied broadly it made most of the
corpus repeated, applied narrowly it under-counted. The audit found the initial
ten logs inconsistent with either reading. The rule below is the operational
version, applied uniformly. Clarified before any probe existed and before any
retrieval result depended on it.

A memory receives one additional presentation for a later session only when that
session returns to the same event, commitment, decision, or specific personal
situation. Recurrence of a broad topic or domain alone does not count.

Counts:
  A reviewer requests an ablation in session 2, and session 3 reports that same
  ablation running.

Does not count:
  A participant discusses one migraine episode in session 2 and later spends an
  unrelated evening in a dark room in session 4.

Logs are authored rather than accumulated from E's own retrievals, so that all
five conditions see identical presentation histories and no condition is
evaluated against a store shaped by another condition's behaviour.