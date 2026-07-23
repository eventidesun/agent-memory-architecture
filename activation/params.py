"""activation/params.py

Never tuned on the evaluation probes

Three tiers:
IMPORTED — taken from literature, cited
DERIVED — computed from a cited value, with stated rationale
EXTENSION — no literature value exists; swept in sensitivity analysis
"""

# IMPORTED
# ACT-R convention (Anderson & Lebiere 1998); adopted by Honda et al.
# (HAI '25, §4.3.2) for LLM dialogue with the same embedder.
D_BASE = 0.5

# Gaussian noise SD. Honda et al. (HAI '25, §4.3.2) hold sigma = 1.2
# across their parameter sweep.
NOISE_SIGMA = 1.2

# DERIVED
# Honda et al. sweep a single uniform context weight w over 0-25 and
# report w = 11.0 as optimal (§5.3.3). Their model has ONE cue
# (semantic similarity), so w is total source activation. This model
# has two cues, so W_TOTAL is split across them. The split is ours;
# the total is theirs. Swept in sensitivity analysis.
W_TOTAL = 11.0
W = {
    "utterance": 0.5 * W_TOTAL,
    "speaker":   0.5 * W_TOTAL,
}

# EXTENSION
# Retrieval threshold. No transferable value exists: ACT-R fits tau
# per task, and Honda et al. name retrieval_threshold as a control
# parameter without reporting a value. Derived from observed
# activation distributions on the corpus, NOT from the probes, and
# swept in sensitivity analysis.
TAU = None  # set before E runs; see DESIGN 5.3

# Floor on elapsed time so same-session presentations (elapsed = 0)
# don't divide by zero. Implementation floor, not a cognitive claim.
MIN_ELAPSED = 0.1

# Salience -> decay modulation strength: d_i = D_BASE * (1 - LAMBDA * salience_i).
# No literature value. McGaugh (2004) motivates the direction (arousal
# slows forgetting) but does not parameterize it. Swept.
LAMBDA = 0.5

# Similarity metric for S_i. NOTE: E uses cosine; Condition C uses L2.
# Honda et al. use cosine. Stated explicitly in the paper.
SIMILARITY = "cosine"