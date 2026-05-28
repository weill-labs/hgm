"""Headline simulated experiment: CMP-guided search vs. the greedy/DGM baseline.

Runs both policies under an identical evaluation budget across many seeds and reports the
TRUE hidden skill of the agent each policy would ship. $0 — no API, no Docker.

    uv run python experiments/run_simulated.py
"""

from __future__ import annotations

import numpy as np

from hgm.search import HGMSearch, SearchConfig
from hgm.simulated import LandscapeConfig, SimulatedWorld

N_SEEDS = 40
MAX_TASK_EVALS = 150
N_TASKS = 60


def best_shipped_skill(use_cmp: bool, seed: int) -> tuple[float, int]:
    world = SimulatedWorld(LandscapeConfig(seed=seed))
    cfg = SearchConfig(max_task_evals=MAX_TASK_EVALS, seed=seed, use_cmp=use_cmp)
    result = HGMSearch(world, world, tasks=list(range(N_TASKS)), config=cfg).run()
    best = result.best_by_mean()
    return world.true_skill(best.commit_id), len(result.nodes) - 1  # skill, #variants


def summarize(label: str, use_cmp: bool) -> np.ndarray:
    skills, variants = [], []
    for s in range(N_SEEDS):
        sk, nv = best_shipped_skill(use_cmp, s)
        skills.append(sk)
        variants.append(nv)
    skills = np.array(skills)
    print(
        f"{label:<8} mean shipped skill = {skills.mean():.3f} "
        f"(std {skills.std():.3f}, min {skills.min():.3f}, max {skills.max():.3f}); "
        f"avg variants explored = {np.mean(variants):.1f}"
    )
    return skills


if __name__ == "__main__":
    print(
        f"Simulated HGM — {N_SEEDS} seeds, budget {MAX_TASK_EVALS} task-evals, "
        f"{N_TASKS} tasks. Initial agent skill = {LandscapeConfig().initial_skill}\n"
    )
    cmp = summarize("HGM/CMP", use_cmp=True)
    greedy = summarize("greedy", use_cmp=False)
    wins = int(np.sum(cmp > greedy))
    print(
        f"\nCMP advantage: +{(cmp.mean() - greedy.mean()):.3f} mean skill; "
        f"CMP ships a better agent in {wins}/{N_SEEDS} seeds."
    )
