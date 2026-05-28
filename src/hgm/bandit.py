"""Beta-Bernoulli Thompson Sampling over candidate nodes.

Each candidate is summarised by a list of 0/1 outcomes. We place a ``Beta(1+wins,
1+losses)`` posterior on each candidate's success probability, draw one sample from each
posterior, and pick the argmax. Exploration is automatic: candidates with few
observations have high-variance posteriors and so occasionally win the draw.

Reference: ``hgm.py:TS_sample``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def thompson_select(
    eval_lists: Sequence[Sequence[float]],
    rng: np.random.Generator,
    *,
    cool_down: bool = False,
    beta_exponent: float = 1.0,
    budget: int | None = None,
    spent: int | None = None,
) -> int:
    """Return the index of the candidate chosen by one Thompson draw.

    Args:
        eval_lists: one list of 0/1 (or pooled mean) outcomes per candidate.
        rng: numpy Generator (inject for deterministic tests).
        cool_down: if True, sharpen posteriors as the budget is spent (late-run
            exploitation). Scales both Beta params by
            ``budget**b / (budget - spent)**b``.
        beta_exponent: the ``b`` above.
        budget, spent: required when ``cool_down`` is True.
    """
    alphas = np.array([1.0 + np.sum(e) for e in eval_lists], dtype=float)
    betas = np.array([1.0 + len(e) - np.sum(e) for e in eval_lists], dtype=float)

    if cool_down:
        if budget is None or spent is None:
            raise ValueError("cool_down requires budget and spent")
        if budget == spent:
            scale = 10000.0
        else:
            scale = budget**beta_exponent / (budget - spent) ** beta_exponent
        alphas = alphas * scale
        betas = betas * scale

    thetas = rng.beta(alphas, betas)
    return int(np.argmax(thetas))
