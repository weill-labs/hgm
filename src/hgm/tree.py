"""The agent-variant tree and the CMP (Clade-based Metaproductivity) metric.

Each ``Node`` is one variant of the coding agent. Its ``utility_measures`` is a list
of per-task binary outcomes (1 = task resolved, 0 = not). The tree records the
self-improvement lineage: a node's children are variants produced by mutating it.

The paper's central idea lives in ``descendant_evals`` below: a node is judged not by
its own success rate but by the *pooled* success of its entire clade (itself + all
descendants). See findings.md for the spec read from the reference implementation.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class Node:
    def __init__(
        self,
        registry: "list[Node]",
        commit_id: str,
        parent: Optional["Node"] = None,
    ) -> None:
        # ``registry`` is the flat list of all nodes in this run; appending here gives
        # every node a stable integer id and lets the search enumerate the whole tree.
        self.registry = registry
        self.commit_id = commit_id
        self.parent = parent
        self.children: list[Node] = []
        self.utility_measures: list[int] = []  # 0/1 per evaluated task
        self.id = len(registry)
        registry.append(self)
        if parent is not None:
            parent.children.append(self)

    # --- own-fitness views -------------------------------------------------------
    @property
    def num_evals(self) -> int:
        return len(self.utility_measures)

    @property
    def mean_utility(self) -> float:
        """Own success rate. ``inf`` when never evaluated, so an unevaluated node is
        treated as maximally promising until measured (matches reference)."""
        if self.num_evals == 0:
            return float("inf")
        return float(np.sum(self.utility_measures) / self.num_evals)

    # --- clade traversal ---------------------------------------------------------
    def subtree(self, fn: Callable[["Node"], object] = lambda n: n) -> list:
        """DFS over this node and all descendants (self first)."""
        out = [fn(self)]
        for child in self.children:
            out.extend(child.subtree(fn))
        return out

    # --- CMP: the heart of HGM ---------------------------------------------------
    def pseudo_self_evals(self, num_pseudo: int) -> list[int]:
        """Pseudo-count smoothing for THIS node's own outcomes.

        If the node has been evaluated fewer than ``num_pseudo`` times, return its raw
        outcomes. Otherwise, collapse them into ``num_pseudo`` copies of its mean. This
        caps how much a single heavily-evaluated node can dominate the clade pool, so
        that descendant evidence still counts. (Reference: Node.get_pseudo_decendant_evals)
        """
        if self.num_evals < num_pseudo:
            return list(self.utility_measures)
        return [self.mean_utility] * num_pseudo

    def descendant_evals(self, num_pseudo: int = 10) -> list[float]:
        """CMP estimate: the clade's pooled 0/1 outcomes.

        TODO(user): implement the Clade-based Metaproductivity pooling.

        Return a single flat list combining:
          1. this node's *smoothed* own outcomes  -> self.pseudo_self_evals(num_pseudo)
          2. the *raw* utility_measures of every strict descendant (children,
             grandchildren, ...) — NOT smoothed.

        Hints:
          - ``self.subtree()`` returns ``[self, d1, d2, ...]`` (self is index 0).
          - Each descendant ``d`` contributes ``d.utility_measures`` verbatim.
          - The result is later fed to a Beta-Bernoulli Thompson sampler, so its
            length and sum both matter (they become the Beta pseudo-counts).

        This is the one function that distinguishes HGM from greedy DGM: greedy would
        use only ``self.utility_measures``; HGM credits a parent for its whole lineage.
        """
        pooled = self.pseudo_self_evals(num_pseudo)  # smoothed OWN outcomes
        for descendant in self.subtree()[1:]:  # [1:] skips self
            pooled = pooled + list(
                descendant.utility_measures
            )  # raw descendant outcomes
        return pooled

    def to_dict(self) -> dict:
        return {
            "commit_id": self.commit_id,
            "id": self.id,
            "parent_id": self.parent.id if self.parent else None,
            "mean_utility": self.mean_utility,
            "num_evals": self.num_evals,
        }
