"""$0 validation of the file-based mini-swe-agent integration.

Uses DeterministicModel (scripted actions, fake cost) + LocalEnvironment in a temp dir,
so the full agent loop — prompt rendering, bash execution, the
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT submission protocol — runs without any API or Docker.
"""

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

from hgm.mini_runner import run_variant
from hgm.variant import VariantStore, initial_swebench_config

_FAKE_PATCH = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"

_MINIMAL_CONFIG = {
    "agent": {
        "system_template": "You solve tasks by issuing bash commands.",
        "instance_template": "Task: {{task}}",
        "step_limit": 5,
        "cost_limit": 10.0,
    },
    "model": {"model_name": "deterministic"},
    "environment": {"timeout": 30},
}


def _store(tmp_path):
    store = VariantStore(tmp_path / "variants")
    initial = store.create_initial(_MINIMAL_CONFIG)
    return store, initial


def test_variant_runs_and_returns_submission(tmp_path):
    store, initial = _store(tmp_path)
    submit_cmd = f"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n{_FAKE_PATCH}'"
    model = DeterministicModel(
        outputs=[
            make_output("I'll submit the fix.", [{"command": submit_cmd}], cost=0.0)
        ]
    )
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)

    result = run_variant(initial, task="fix the bug", model=model, env=env)

    assert result.exit_status == "Submitted"
    assert result.produced_patch
    assert "+x = 2" in result.submission
    assert result.cost == 0.0


def test_variant_hits_step_limit_without_submission(tmp_path):
    store, initial = _store(tmp_path)
    model = DeterministicModel(
        outputs=[
            make_output("looking...", [{"command": "echo still working"}], cost=0.0)
        ]
        * 10
    )
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)

    result = run_variant(initial, task="fix the bug", model=model, env=env)

    assert result.exit_status == "LimitsExceeded"
    assert not result.produced_patch


def test_initial_swebench_config_loads():
    cfg = initial_swebench_config()
    assert "system_template" in cfg["agent"]
    assert cfg["agent"].get("step_limit", 0) > 0
    assert "{{task}}" in cfg["agent"]["instance_template"]


def test_fork_copies_files_and_loads_agent_class(tmp_path):
    store, initial = _store(tmp_path)
    child = store.fork(initial.commit_id)

    assert child.parent_id == "initial"
    assert child.agent_py.exists() and child.config_yaml.exists()
    # the forked agent.py still defines a usable Agent class
    AgentClass = child.load_agent_class()
    assert AgentClass.__name__ == "Agent"


def test_code_patch_to_agent_py_takes_effect(tmp_path):
    """The CODE surface works: editing a child's agent.py changes runtime behaviour.

    We patch the child's Agent to raise on run(), then confirm run_variant surfaces it —
    proving the dynamically imported, patched class is the one actually executed.
    """
    store, initial = _store(tmp_path)
    child = store.fork(initial.commit_id)
    child.agent_py.write_text(
        "from minisweagent.agents.default import DefaultAgent\n"
        "class Agent(DefaultAgent):\n"
        "    def run(self, task='', **kw):\n"
        "        raise RuntimeError('patched-agent-ran')\n"
    )
    model = DeterministicModel(
        outputs=[make_output("x", [{"command": "echo hi"}], cost=0.0)]
    )
    env = LocalEnvironment(cwd=str(tmp_path), timeout=30)

    try:
        run_variant(child, task="t", model=model, env=env)
        raise AssertionError("expected the patched agent to raise")
    except RuntimeError as e:
        assert "patched-agent-ran" in str(e)
