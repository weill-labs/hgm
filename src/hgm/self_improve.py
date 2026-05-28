"""Self-improvement step: an agent that edits its own source.

To improve a parent variant we (1) fork its directory, (2) run a *meta* mini-swe-agent
whose working directory IS the child's files, tasked with editing ``agent.py`` and/or
``config.yaml`` to fix observed weaknesses, then (3) validate the child still loads. The
child variant is the edited directory. This mirrors the reference HGM running
``coding_agent.py --self_improve`` on its own code.

SAFETY: the meta-agent executes shell commands and the resulting child code is imported
and run. Under a real LLM this is self-modifying code and MUST run in a sandbox (Docker,
Phase 3). The $0 tests use a deterministic meta-model in a throwaway temp dir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from minisweagent.agents.default import DefaultAgent

from .interfaces import FAILED
from .variant import AgentVariant, VariantStore

# Default prompts for the meta-agent (the one doing the improving).
META_SYSTEM_TEMPLATE = (
    "You are improving a coding agent by editing its own source files. "
    "You interact with a shell by issuing bash commands."
)
META_INSTANCE_TEMPLATE = """\
You are looking at the source of a coding agent in the current directory:
  - agent.py     : a DefaultAgent subclass named `Agent` (the agent's control loop)
  - config.yaml  : its prompts (agent.system_template, agent.instance_template) and
                   limits (agent.step_limit, agent.cost_limit) and model/environment

This agent has shown the following weakness(es):
{{task}}

Improve the agent by editing agent.py and/or config.yaml in the current directory so it
performs better on such cases, while keeping `Agent` importable and config.yaml valid.
When you are done, submit a one-line summary using the required submission protocol.
"""

_META_AGENT_KEYS = {
    "system_template",
    "instance_template",
    "step_limit",
    "cost_limit",
    "wall_time_limit_seconds",
}


@dataclass
class MetaConfig:
    system_template: str = META_SYSTEM_TEMPLATE
    instance_template: str = META_INSTANCE_TEMPLATE
    step_limit: int = 40
    cost_limit: float = 1.0


class MiniSelfImprover:
    """Implements the SelfImprover protocol using a meta mini-swe-agent.

    Args:
        store: the VariantStore (forks live under its root).
        model_factory: () -> a mini-swe-agent Model for the meta-agent (fresh per call).
        env_factory: (cwd: str) -> an Environment rooted at the child's directory.
        diagnose: (parent_commit_id) -> weakness description fed to the meta-agent.
        meta: meta-agent prompts/limits.
        validate: if True, a child that fails to import / parse is rejected (FAILED).
    """

    def __init__(
        self,
        store: VariantStore,
        *,
        model_factory: Callable[[], object],
        env_factory: Callable[[str], object],
        diagnose: Optional[Callable[[str], str]] = None,
        meta: MetaConfig | None = None,
        validate: bool = True,
        spend_guard=None,
    ) -> None:
        self.store = store
        self.model_factory = model_factory
        self.env_factory = env_factory
        self.diagnose = diagnose
        self.meta = meta or MetaConfig()
        self.validate = validate
        self.spend_guard = spend_guard  # optional: records meta-agent cost

    def improve(self, parent_commit_id: str) -> str:
        child = self.store.fork(parent_commit_id)
        weakness = (
            self.diagnose(parent_commit_id)
            if self.diagnose
            else "No specific diagnosis; improve general robustness."
        )
        task = weakness

        model = self.model_factory()
        env = self.env_factory(
            str(child.path)
        )  # meta-agent operates inside child's dir
        meta_kwargs = {
            k: v for k, v in vars(self.meta).items() if k in _META_AGENT_KEYS
        }
        meta_agent = DefaultAgent(model, env, **meta_kwargs)
        try:
            meta_agent.run(task)
        except Exception:
            return FAILED
        finally:
            if self.spend_guard is not None:
                self.spend_guard.add(float(getattr(meta_agent, "cost", 0.0)))

        if self.validate and not self._validates(child):
            return FAILED
        return child.commit_id

    @staticmethod
    def _validates(child: AgentVariant) -> bool:
        """A usable child must parse its config and import an `Agent` class."""
        try:
            child.load_config()
            child.load_agent_class()
            return True
        except Exception:
            return False
