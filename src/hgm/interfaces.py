"""The two seams that decouple the HGM search from the world it searches.

The search loop only ever asks two things of the outside world:
  - "evaluate this agent variant on this task"  -> Evaluator
  - "mutate this agent variant into a new one"  -> SelfImprover

The simulated milestone implements these against a hidden skill landscape ($0). Phase 2/3
implement them against mini-swe-agent + SWE-bench. The search code never changes.
"""

from __future__ import annotations

from typing import Hashable, Protocol, runtime_checkable

FAILED = (
    "failed"  # sentinel commit_id for a self-improvement attempt that produced nothing
)


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, commit_id: str, task: Hashable) -> int:
        """Run the variant identified by ``commit_id`` on ``task``.

        Returns 1 if the task is resolved, else 0.
        """
        ...


@runtime_checkable
class SelfImprover(Protocol):
    def improve(self, parent_commit_id: str) -> str:
        """Produce a child variant by self-improving the parent.

        Must register the new variant under the returned ``commit_id`` so a later
        ``Evaluator.evaluate`` call can find it. Return ``FAILED`` if no usable variant
        was produced (e.g. the patch didn't apply / didn't compile).
        """
        ...
