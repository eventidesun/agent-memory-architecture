"""Parameters for Condition E.

Values are imported from prior work, derived from cited values, or
treated as extensions and included in sensitivity analysis.
"""

# Imported from ACT-R and Honda et al. (HAI '25).
D_BASE = 0.5
NOISE_SIGMA = 1.2

# Total context weight adapted from Honda et al. (HAI '25).
W_TOTAL = 11.0

# Placeholder split; replace with values from derive_weights.py.
W = {"utterance": 8.3877, "speaker": 2.6123}
# derived primary: sigma_u=0.1469, sigma_s=0.4716

# Set from corpus activation distributions before evaluation.
TAU = 1.817

# Prevents division by zero for same-session presentations.
MIN_ELAPSED = 0.1

# Salience-based decay modulation; swept in sensitivity analysis.
LAMBDA = 0.5

# Condition E uses cosine similarity.
SIMILARITY = "cosine"