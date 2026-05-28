"""The headline experiment: under equal eval budget, CMP-guided search should discover
higher-skill agents than the greedy/DGM baseline, averaged over seeds.

Fails until Node.descendant_evals (CMP) is implemented.
"""

import numpy as np

from hgm.search import HGMSearch, SearchConfig
from hgm.simulated import LandscapeConfig, SimulatedWorld


def _best_true_skill(use_cmp: bool, seed: int) -> float:
    world = SimulatedWorld(LandscapeConfig(seed=seed))
    cfg = SearchConfig(max_task_evals=150, seed=seed, use_cmp=use_cmp)
    result = HGMSearch(world, world, tasks=list(range(60)), config=cfg).run()
    # judge by the TRUE hidden skill of the variant the search would ship
    best_node = result.best_by_mean()
    return world.true_skill(best_node.commit_id)


def test_cmp_beats_greedy_on_average():
    seeds = range(12)
    cmp_scores = [_best_true_skill(use_cmp=True, seed=s) for s in seeds]
    greedy_scores = [_best_true_skill(use_cmp=False, seed=s) for s in seeds]
    # CMP should find meaningfully more skilled agents on average
    assert np.mean(cmp_scores) > np.mean(greedy_scores) + 0.02
