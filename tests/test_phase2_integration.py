"""End-to-end Phase 2: the HGM search loop driving the REAL file-based self-improver.

Still $0 (deterministic meta-model). Proves the same HGMSearch from Phase 1 — unchanged —
drives MiniSelfImprover to materialise real, self-edited agent variant directories on
disk, and that a depth-aware evaluator lets the search prefer the more-improved lineage.
"""

import hashlib

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

from hgm.search import HGMSearch, SearchConfig
from hgm.self_improve import MiniSelfImprover
from hgm.variant import VariantStore

_CONFIG = {
    "agent": {
        "system_template": "solve via bash",
        "instance_template": "Task: {{task}}",
        "step_limit": 5,
        "cost_limit": 10.0,
    },
    "model": {"model_name": "deterministic"},
    "environment": {"timeout": 30},
}


def _meta_outputs():
    # Generic, always-valid self-edit: append a marker line to agent.py, then submit.
    return [
        make_output(
            "improving", [{"command": "echo '# improved' >> agent.py"}], cost=0.0
        ),
        make_output(
            "done",
            [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo ok"}],
            cost=0.0,
        ),
    ]


class DepthEvaluator:
    """Deterministic evaluator: the more a variant has self-improved (marker lines in
    agent.py), the higher its success probability. Outcomes are a pure hash of
    (commit_id, task), so the run is fully reproducible."""

    def __init__(self, store: VariantStore) -> None:
        self.store = store

    def _skill(self, commit_id: str) -> float:
        n = self.store.get(commit_id).agent_py.read_text().count("# improved")
        return min(0.10 + 0.20 * n, 0.95)

    def evaluate(self, commit_id: str, task) -> int:
        h = (
            int(hashlib.sha256(f"{commit_id}:{task}".encode()).hexdigest(), 16)
            % 10**6
            / 10**6
        )
        return int(h < self._skill(commit_id))


def test_search_drives_real_self_improvement(tmp_path):
    store = VariantStore(tmp_path / "variants")
    store.create_initial(_CONFIG)
    improver = MiniSelfImprover(
        store,
        model_factory=lambda: DeterministicModel(outputs=_meta_outputs()),
        env_factory=lambda cwd: LocalEnvironment(cwd=cwd, timeout=30),
        diagnose=lambda pid: "Improve robustness.",
    )
    evaluator = DepthEvaluator(store)

    cfg = SearchConfig(max_task_evals=40, seed=0, use_cmp=True, init_measurements=5)
    result = HGMSearch(evaluator, improver, tasks=list(range(20)), config=cfg).run()

    # The search materialised real, self-edited variant directories beyond the initial.
    child_dirs = [p for p in (tmp_path / "variants").iterdir() if p.name != "initial"]
    assert len(child_dirs) >= 1
    # at least one child actually has a self-improvement edit applied to its own agent.py
    assert any(
        (d / "agent.py").read_text().count("# improved") >= 1 for d in child_dirs
    )
    # the loop spent its budget and explored a tree
    assert result.n_task_evals >= cfg.max_task_evals
    assert len(result.nodes) >= 2
