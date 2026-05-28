"""Thompson Sampling sanity — no CMP dependency, runs today."""

import numpy as np

from hgm.bandit import thompson_select


def test_prefers_higher_success_rate():
    rng = np.random.default_rng(0)
    # candidate 0: mostly failures; candidate 1: mostly successes
    a = [0] * 18 + [1] * 2
    b = [1] * 18 + [0] * 2
    wins = sum(thompson_select([a, b], rng) == 1 for _ in range(1000))
    assert wins > 900  # the strong candidate should almost always win the draw


def test_explores_uncertain_candidate():
    rng = np.random.default_rng(0)
    # candidate 0: a single success (high uncertainty); candidate 1: 5/10 (low uncertainty)
    sparse = [1]
    settled = [1] * 5 + [0] * 5
    picks = [thompson_select([sparse, settled], rng) for _ in range(1000)]
    # the uncertain candidate must get explored a non-trivial fraction of the time
    assert 0.05 < np.mean(picks) < 0.95


def test_cool_down_sharpens_late():
    rng = np.random.default_rng(0)
    a = [1] * 6 + [0] * 4  # 0.6
    b = [1] * 5 + [0] * 5  # 0.5
    early = [
        thompson_select([a, b], rng, cool_down=True, budget=100, spent=1)
        for _ in range(2000)
    ]
    late = [
        thompson_select([a, b], rng, cool_down=True, budget=100, spent=99)
        for _ in range(2000)
    ]
    # picks records the chosen index; arm 0 is the better arm, so a LOWER mean means
    # the worse arm (index 1) is chosen less, i.e. more decisive exploitation.
    assert np.mean(late) < np.mean(early)
