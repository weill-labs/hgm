"""Run a file-based mini-swe-agent variant on a task and capture its result.

Loads the variant's (possibly code-patched) ``Agent`` class and its config from the
variant directory, instantiates the agent, and runs it. Does NOT score the result
against SWE-bench tests — that's Phase 3. Here we only run the agent and return the patch
it submitted plus the cost it incurred, so the wiring can be validated for $0.
"""

from __future__ import annotations

from dataclasses import dataclass

from .variant import AgentVariant

# Keys in the ``agent`` config block that DefaultAgent's AgentConfig accepts.
_AGENT_CONFIG_KEYS = {
    "system_template",
    "instance_template",
    "step_limit",
    "cost_limit",
    "wall_time_limit_seconds",
    "output_path",
}


@dataclass
class RunResult:
    submission: str  # the git patch the agent produced (may be empty)
    exit_status: str  # "Submitted", "LimitsExceeded", "TimeExceeded", ...
    cost: float  # dollars (or fake units under a deterministic model)
    n_calls: int  # number of model queries
    produced_patch: bool  # convenience: a non-empty submission was returned


def _agent_kwargs(config: dict) -> dict:
    """Filter the config's agent block to the keys AgentConfig understands."""
    agent_block = config.get("agent", {})
    return {k: v for k, v in agent_block.items() if k in _AGENT_CONFIG_KEYS}


def run_variant(variant: AgentVariant, task: str, *, model, env) -> RunResult:
    """Instantiate the variant's agent and run it on ``task``.

    ``model`` and ``env`` are injected (DeterministicModel + LocalEnvironment for $0
    tests; LitellmModel + Docker for real runs). Behaviour is governed by the variant's
    config (prompts + limits) and its agent.py (code).
    """
    config = variant.load_config()
    agent_class = variant.load_agent_class()
    agent = agent_class(model, env, **_agent_kwargs(config))
    result = agent.run(task)
    submission = result.get("submission", "") or ""
    return RunResult(
        submission=submission,
        exit_status=result.get("exit_status", ""),
        cost=float(getattr(agent, "cost", 0.0)),
        n_calls=int(getattr(agent, "n_calls", 0)),
        produced_patch=bool(submission.strip()),
    )
