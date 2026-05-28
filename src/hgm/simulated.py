"""A $0 simulated world for validating the search algorithm.

Every agent variant has a hidden ``skill`` in [0, 1] = its probability of resolving a
random task. Evaluating a variant on a task is a Bernoulli draw on its skill. Self-
improving a variant produces a child whose skill is perturbed.

Why the landscape is shaped the way it is
------------------------------------------
The whole point of CMP is to reward a node for the success of its *descendants*, not just
itself. For that to matter, lineages must differ in their *productivity* — how fast they
keep improving — and that productivity must NOT be readable from a node's current skill
alone. So each variant also carries a hidden, inherited ``drift``: the expected per-step
skill gain of its lineage. ``drift`` does a random walk down each lineage, so some
branches keep climbing while others plateau.

This creates the greedy trap: a variant can have high *current* skill but a dead lineage
(low drift), while a mediocre variant sits atop a lineage that will climb far. A greedy
"expand the current best" policy chases the former; CMP, seeing productive descendants,
invests in the latter. If our CMP + Thompson search can't beat greedy here, it's wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import FAILED


@dataclass
class Variant:
    skill: float  # P(resolve a random task)
    drift: float  # expected per-step skill gain of this lineage (hidden, inherited)


@dataclass
class LandscapeConfig:
    initial_skill: float = 0.10
    initial_drift: float = 0.00
    drift_step_sigma: float = 0.04  # how fast lineage productivity itself wanders
    skill_noise_sigma: float = 0.03  # per-mutation skill jitter on top of drift
    fail_prob: float = 0.05  # chance a self-improvement yields nothing usable
    seed: int = 0


class SimulatedWorld:
    """Implements both the Evaluator and SelfImprover protocols."""

    def __init__(self, config: LandscapeConfig | None = None) -> None:
        self.cfg = config or LandscapeConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._counter = 0
        self.variants: dict[str, Variant] = {
            "initial": Variant(
                skill=self.cfg.initial_skill, drift=self.cfg.initial_drift
            )
        }

    # --- Evaluator ---------------------------------------------------------------
    def evaluate(self, commit_id: str, task) -> int:  # noqa: ANN001 (task unused)
        skill = self.variants[commit_id].skill
        return int(self.rng.random() < skill)

    # --- SelfImprover ------------------------------------------------------------
    def improve(self, parent_commit_id: str) -> str:
        if self.rng.random() < self.cfg.fail_prob:
            return FAILED
        parent = self.variants[parent_commit_id]
        child = self._mutate(parent)
        self._counter += 1
        commit_id = f"v{self._counter}"
        self.variants[commit_id] = child
        return commit_id

    def _mutate(self, parent: Variant) -> Variant:
        """Produce a child variant. Drift random-walks; skill follows drift plus noise."""
        new_drift = parent.drift + self.rng.normal(0.0, self.cfg.drift_step_sigma)
        new_skill = (
            parent.skill + new_drift + self.rng.normal(0.0, self.cfg.skill_noise_sigma)
        )
        return Variant(
            skill=float(np.clip(new_skill, 0.0, 1.0)), drift=float(new_drift)
        )

    # --- introspection (for tests / plotting) -----------------------------------
    def true_skill(self, commit_id: str) -> float:
        return self.variants[commit_id].skill
