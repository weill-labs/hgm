"""Sanity check: CMP's advantage over greedy should GROW with lineage-productivity
variance (drift_step_sigma). If lineages differ more in how fast they improve, there is
more for clade-based credit assignment to discover that greedy misses.

    uv run python experiments/sweep_drift.py
"""

from __future__ import annotations

import numpy as np

from hgm.search import HGMSearch, SearchConfig
from hgm.simulated import LandscapeConfig, SimulatedWorld

N_SEEDS = 40
MAX_TASK_EVALS = 150
N_TASKS = 60


def best_skill(use_cmp: bool, seed: int, drift_sigma: float) -> float:
    world = SimulatedWorld(LandscapeConfig(seed=seed, drift_step_sigma=drift_sigma))
    cfg = SearchConfig(max_task_evals=MAX_TASK_EVALS, seed=seed, use_cmp=use_cmp)
    result = HGMSearch(world, world, tasks=list(range(N_TASKS)), config=cfg).run()
    return world.true_skill(result.best_by_mean().commit_id)


if __name__ == "__main__":
    print(f"{N_SEEDS} seeds, budget {MAX_TASK_EVALS}, {N_TASKS} tasks.\n")
    print(f"{'drift_sigma':>11} | {'CMP':>6} | {'greedy':>6} | {'advantage':>9} | wins")
    print("-" * 52)
    for drift_sigma in [0.0, 0.02, 0.04, 0.08, 0.12, 0.20]:
        cmp = np.array([best_skill(True, s, drift_sigma) for s in range(N_SEEDS)])
        grd = np.array([best_skill(False, s, drift_sigma) for s in range(N_SEEDS)])
        wins = int(np.sum(cmp > grd))
        print(
            f"{drift_sigma:>11.2f} | {cmp.mean():>6.3f} | {grd.mean():>6.3f} | "
            f"{cmp.mean() - grd.mean():>+9.3f} | {wins}/{N_SEEDS}"
        )
