"""Base-level activation: B_i = ln( sum_j (t - t_j)^(-d_i) ),
with salience-modulated decay d_i = D_BASE * (1 - LAMBDA * salience).
DESIGN 5.3: time unit = session; presentations = write + retrievals."""

import math

from activation.params import D_BASE, MIN_ELAPSED


def decay_rate(salience: float, lam: float, d_base: float = D_BASE) -> float:
    return d_base * (1.0 - lam * salience)


def base_level(presentation_log: list[float], current_time: float,
               salience: float, lam: float, d_base: float = D_BASE) -> float:
    if not presentation_log:
        return float("-inf")

    d = decay_rate(salience, lam, d_base)
    total = sum(
        max(current_time - t_j, MIN_ELAPSED) ** (-d)
        for t_j in presentation_log
    )
    return math.log(total)