"""End-to-end harness check via the GREEDY path (use_cmp=False).

This exercises the whole search loop — expand/measure scheduling, budget accounting,
node bookkeeping, simulated evaluation and self-improvement — WITHOUT touching the CMP
code path. So it passes today and proves the plumbing is correct before CMP is written.
"""

from hgm.search import HGMSearch, SearchConfig
from hgm.simulated import LandscapeConfig, SimulatedWorld


def test_greedy_search_runs_and_spends_budget():
    world = SimulatedWorld(LandscapeConfig(seed=1))
    cfg = SearchConfig(max_task_evals=60, seed=1, use_cmp=False)
    result = HGMSearch(world, world, tasks=list(range(50)), config=cfg).run()

    assert result.n_task_evals >= cfg.max_task_evals
    assert len(result.nodes) >= 2  # root + at least one self-improved child
    assert any(n.num_evals > 0 for n in result.nodes)
    assert result.best_by_mean().num_evals > 0


def test_search_improves_over_initial():
    """Even greedy should usually end with a variant better than the initial agent,
    because the simulated landscape drifts upward on average."""
    world = SimulatedWorld(LandscapeConfig(seed=2))
    cfg = SearchConfig(max_task_evals=120, seed=2, use_cmp=False)
    result = HGMSearch(world, world, tasks=list(range(80)), config=cfg).run()

    best = result.best_by_mean()
    assert world.true_skill(best.commit_id) >= world.true_skill("initial")
