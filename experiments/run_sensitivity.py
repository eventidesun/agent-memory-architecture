"""Parameter sensitivity sweep for Condition E.

The question is not whether numbers change across the parameter range — they
will — but whether the SIGN of each dissociation holds. Specifically:

  - Does E's repetition advantage over C survive the range, or exist only at
    the frozen setting?
  - Does salience modulation produce any measurable effect at any defensible
    lambda, or is it inert regardless?
  - Does lowering noise convert E's rank-2 misses into rank-1 hits?

Parameters are NOT selected from this sweep. They are frozen by their
derivation rules; the sweep characterises stability only. Adopting whichever
setting looks best would be the tuning the methodology exists to prevent.

One-at-a-time: each parameter is varied while the others hold at their frozen
values. A full grid would be 400+ runs and would not answer a question this
one does not."""

import json
import os

from retrievers.base import Context
from retrievers.activation import ActivationRetriever
from activation.params import (
    W, W_TOTAL, TAU, LAMBDA, NOISE_SIGMA, D_BASE,
)
from experiments.records import build_record, append_record

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(_ROOT, "benchmark", "probes.json")
OUT = os.path.join(_ROOT, "results", "raw", "sensitivity.jsonl")

SEED = 0
K = 49
CURRENT_TIME = 6.0
SESSION_ID = 6
CORPUS = "primary"

# Ranges. Each includes the frozen value so the sweep contains the baseline.
SWEEPS = {
    "lambda":      [0.0, 0.25, 0.5, 0.75, 1.0],
    "noise_sigma": [0.0, 0.3, 0.6, 1.2, 2.0],
    "d_base":      [0.0, 0.25, 0.5, 0.75, 1.0],
    "w_total":     [1.0, 3.0, 5.0, 11.0, 20.0],
    "tau":         [None, 0.0, 1.817, 3.0, 4.5],
}


def split_w(total):
    """Preserve the equal-variance ratio while scaling the total.

    The ratio is a property of the corpus, derived once. Sweeping the total
    while holding the ratio tests the magnitude of spreading activation
    without re-deriving the split at every point."""
    ratio = W["speaker"] / W_TOTAL
    return {"utterance": total * (1 - ratio), "speaker": total * ratio}


def make_retriever(param, value):
    kwargs = dict(seed=SEED, reinforce=False)
    if param == "lambda":
        kwargs["lam"] = value
    elif param == "noise_sigma":
        kwargs["noise_sigma"] = value
    elif param == "d_base":
        kwargs["d_base"] = value
    elif param == "w_total":
        kwargs["w"] = split_w(value)
    elif param == "tau":
        kwargs["tau"] = value
    else:
        raise ValueError(f"unknown parameter {param}")
    r = ActivationRetriever(**kwargs)
    r.name = f"E_{param}={value}"
    return r


def run():
    with open(PROBES) as f:
        probes = json.load(f)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)

    for param, values in SWEEPS.items():
        for value in values:
            r = make_retriever(param, value)
            for probe in probes:
                ctx = Context(
                    speaker=probe["speaker"],
                    current_time=probe.get("current_time", CURRENT_TIME),
                    session_id=probe.get("session_id", SESSION_ID),
                    corpus=probe.get("corpus", CORPUS),
                )
                ranked = r.retrieve(probe["query"], ctx, K)
                record = build_record(
                    probe, r.name, SEED, ranked,
                    {"sweep_param": param, "sweep_value": value},
                )
                append_record(record, OUT)
            print(f"{param:14} = {str(value):8}  done")


if __name__ == "__main__":
    run()