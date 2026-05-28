"""The HGM outer loop: a tree search over self-improving agent variants.

Single-threaded and seeded (the reference uses a thread pool; concurrency is incidental
to the algorithm and only obscures it). Each step either:

  * EXPANDS  — picks a promising node by CMP (clade) Thompson Sampling and self-improves
               it into a new child variant, or
  * MEASURES — picks a node by its OWN Thompson Sampling and spends one task evaluation
               on it, appending a fresh 0/1 outcome.

The expand/measure balance is governed by ``alpha``: expansion is allowed only while
``n_task_evals ** alpha >= (#variants so far)``. With ``alpha < 1`` the variant count
grows sublinearly in the eval budget, so most of the budget refines existing variants.

Reference: ``hgm.py`` (TS_sample / expand / sample).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from .bandit import thompson_select
from .interfaces import FAILED, Evaluator, SelfImprover
from .tree import Node


@dataclass
class SearchConfig:
    max_task_evals: int = 200
    alpha: float = 0.6  # expansion-rate exponent
    n_pseudo_descendant_evals: int = 10
    eval_random_level: float = 1.0  # P(pick a random task) vs. first available
    cool_down: bool = False
    beta_exponent: float = 1.0
    init_measurements: int = 5  # seed the root with this many evals before searching
    seed: int = 0
    use_cmp: bool = True  # True = HGM (clade); False = greedy/DGM (own evals)


@dataclass
class SearchResult:
    nodes: list[Node]
    n_task_evals: int
    history: list[dict] = field(default_factory=list)  # per-step log for inspection

    def best_by_mean(self) -> Node:
        """The evaluated node with the highest own success rate (ties: most evals)."""
        evaluated = [n for n in self.nodes if n.num_evals > 0]
        return max(evaluated, key=lambda n: (n.mean_utility, n.num_evals))


class HGMSearch:
    def __init__(
        self,
        evaluator: Evaluator,
        improver: SelfImprover,
        tasks: list[Hashable],
        config: SearchConfig | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.improver = improver
        self.tasks = list(tasks)
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.nodes: list[Node] = []
        self.root = Node(self.nodes, commit_id="initial")
        # tasks already spent on each node id (avoid re-evaluating the same task twice)
        self.submitted: dict[int, set] = {self.root.id: set()}
        self.n_task_evals = 0
        self.history: list[dict] = []

    # --- the two primitive actions (return True iff they made progress) ----------
    def _expand(self) -> bool:
        """Self-improve a promising node (chosen by CMP Thompson Sampling)."""
        candidates = [
            n for n in self.nodes if np.isfinite(n.mean_utility) and n.mean_utility > 0
        ]
        if not candidates:
            return False
        if self.config.use_cmp:
            # HGM: judge a node by its whole clade (itself + descendants).
            expand_evals = [
                n.descendant_evals(num_pseudo=self.config.n_pseudo_descendant_evals)
                for n in candidates
            ]
        else:
            # Greedy / DGM baseline: judge a node by its own outcomes only.
            expand_evals = [n.utility_measures for n in candidates]
        idx = thompson_select(
            expand_evals,
            self.rng,
            cool_down=self.config.cool_down,
            beta_exponent=self.config.beta_exponent,
            budget=self.config.max_task_evals,
            spent=self.n_task_evals,
        )
        parent = candidates[idx]
        child_commit = self.improver.improve(parent.commit_id)
        if child_commit == FAILED:
            # An attempt was made (and consumed) but yielded no usable variant.
            self.history.append(
                {"action": "expand", "parent": parent.id, "child": None}
            )
            return True
        child = Node(self.nodes, commit_id=child_commit, parent=parent)
        self.submitted[child.id] = set()
        self.history.append(
            {"action": "expand", "parent": parent.id, "child": child.id}
        )
        return True

    def _measure(self) -> bool:
        """Spend one task evaluation on a node (chosen by its OWN Thompson Sampling)."""
        candidates = [
            n for n in self.nodes if len(self.submitted[n.id]) < len(self.tasks)
        ]
        if not candidates:
            return False
        own_evals = [n.utility_measures for n in candidates]
        idx = thompson_select(
            own_evals,
            self.rng,
            cool_down=self.config.cool_down,
            beta_exponent=self.config.beta_exponent,
            budget=self.config.max_task_evals,
            spent=self.n_task_evals,
        )
        node = candidates[idx]
        available = [t for t in self.tasks if t not in self.submitted[node.id]]
        if not available:
            return False
        if self.rng.random() < self.config.eval_random_level:
            task = available[int(self.rng.integers(len(available)))]
        else:
            task = available[0]
        self.submitted[node.id].add(task)

        outcome = self.evaluator.evaluate(node.commit_id, task)
        node.utility_measures.append(int(outcome))
        self.n_task_evals += 1
        self.history.append(
            {
                "action": "measure",
                "node": node.id,
                "task": task,
                "outcome": int(outcome),
            }
        )
        return True

    # --- the loop ----------------------------------------------------------------
    def _should_expand(self) -> bool:
        # Number of variants discovered so far excluding the root ("initial").
        n_variants = len(self.nodes) - 1
        return self.n_task_evals**self.config.alpha >= n_variants

    def run(self) -> SearchResult:
        # Seed the root: expansion needs a finite, positive mean utility, so the root
        # must be evaluated enough to (very likely) register at least one success first.
        for _ in range(min(self.config.init_measurements, len(self.tasks))):
            if self.n_task_evals >= self.config.max_task_evals:
                break
            self._measure()

        # Main loop. Each iteration tries its preferred action; if that action can't make
        # progress it falls back to the other. If neither can, the search is exhausted.
        while self.n_task_evals < self.config.max_task_evals:
            prefer_expand = self._should_expand()
            first = self._expand if prefer_expand else self._measure
            second = self._measure if prefer_expand else self._expand
            if not first() and not second():
                break  # nothing left to expand or measure

        return SearchResult(self.nodes, self.n_task_evals, self.history)
