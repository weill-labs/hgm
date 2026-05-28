"""$0 validation of the self-improvement step.

A deterministic meta-agent edits the child variant's own files, then submits. No LLM, no
Docker — proves the fork -> edit-self -> validate -> child pipeline works end to end.
"""

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

from hgm.interfaces import FAILED
from hgm.self_improve import MiniSelfImprover
from hgm.variant import VariantStore

_CONFIG = {
    "agent": {
        "system_template": "solve tasks via bash",
        "instance_template": "Task: {{task}}",
        "step_limit": 5,
        "cost_limit": 10.0,
    },
    "model": {"model_name": "deterministic"},
    "environment": {"timeout": 30},
}


def _improver(tmp_path, meta_outputs):
    store = VariantStore(tmp_path / "variants")
    store.create_initial(_CONFIG)
    improver = MiniSelfImprover(
        store,
        model_factory=lambda: DeterministicModel(outputs=meta_outputs),
        env_factory=lambda cwd: LocalEnvironment(cwd=cwd, timeout=30),
        diagnose=lambda pid: "Agent gives up too early on hard tasks.",
    )
    return store, improver


def test_self_improvement_edits_child_config(tmp_path):
    # meta-agent bumps step_limit in the child's config.yaml, then submits.
    edit = "sed -i 's/step_limit: 5/step_limit: 99/' config.yaml"
    store, improver = _improver(
        tmp_path,
        [
            make_output("Bumping the step limit.", [{"command": edit}], cost=0.0),
            make_output(
                "done",
                [
                    {
                        "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo improved"
                    }
                ],
                cost=0.0,
            ),
        ],
    )

    child_id = improver.improve("initial")

    assert child_id != FAILED
    child = store.get(child_id)
    assert child.parent_id == "initial"
    assert child.load_config()["agent"]["step_limit"] == 99  # child changed
    assert (
        store.get("initial").load_config()["agent"]["step_limit"] == 5
    )  # parent intact


def test_self_improvement_rejects_broken_child(tmp_path):
    # meta-agent corrupts agent.py so it no longer imports -> validation rejects it.
    break_it = "echo 'def (' > agent.py"
    store, improver = _improver(
        tmp_path,
        [
            make_output("oops", [{"command": break_it}], cost=0.0),
            make_output(
                "done",
                [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo x"}],
                cost=0.0,
            ),
        ],
    )

    child_id = improver.improve("initial")

    assert child_id == FAILED  # broken agent.py must not become a usable variant
