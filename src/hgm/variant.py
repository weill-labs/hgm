"""File-based agent variants for the real (mini-swe-agent) backend.

A variant is a *directory snapshot* of the coding agent's own source, mirroring how the
reference HGM treats each variant as a git commit of ``coding_agent.py``. Each variant
directory contains:

    <variant>/
        agent.py      # defines `Agent` — a DefaultAgent subclass (the CODE surface)
        config.yaml   # mini-swe-agent config: prompts + limits + model/env (PROMPT+KNOBS)

The self-improvement step forks a parent directory and lets a mini-swe-agent edit these
files (prompts, knobs, and/or the agent loop in agent.py). Running a variant dynamically
imports its (possibly patched) ``Agent`` class and loads its config.

Because we import and run model-generated ``agent.py``, real self-improvement must happen
in a sandbox (Docker, Phase 3). The $0 tests drive deterministic edits in a temp dir.
"""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from minisweagent import package_dir

# The initial agent.py: a trivial subclass. This is the code surface the self-improver
# may later override (e.g. a custom step()/query()).
INITIAL_AGENT_PY = '''\
"""Agent variant code surface. Self-improvement may override methods here."""

from minisweagent.agents.default import DefaultAgent


class Agent(DefaultAgent):
    """Identical to DefaultAgent until self-improvement changes it."""
'''


@dataclass
class AgentVariant:
    commit_id: str
    path: Path  # directory containing agent.py + config.yaml
    parent_id: str | None = None

    @property
    def agent_py(self) -> Path:
        return self.path / "agent.py"

    @property
    def config_yaml(self) -> Path:
        return self.path / "config.yaml"

    def load_config(self) -> dict:
        with open(self.config_yaml) as f:
            return yaml.safe_load(f) or {}

    def load_agent_class(self) -> type:
        """Dynamically import this variant's agent.py and return its `Agent` class.

        Each variant gets a unique module name so concurrent/distinct variants don't
        collide in sys.modules.
        """
        mod_name = f"hgm_variant_{self.commit_id}_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(mod_name, self.agent_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {self.agent_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(
            module
        )  # executes the variant's (possibly patched) code
        if not hasattr(module, "Agent"):
            raise AttributeError(f"{self.agent_py} does not define `Agent`")
        return module.Agent


class VariantStore:
    """Maps commit_id -> AgentVariant, materialising each as a directory under root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.variants: dict[str, AgentVariant] = {}
        self._counter = 0

    def get(self, commit_id: str) -> AgentVariant:
        return self.variants[commit_id]

    def _register(self, variant: AgentVariant) -> AgentVariant:
        self.variants[variant.commit_id] = variant
        return variant

    def create_initial(self, config: dict, commit_id: str = "initial") -> AgentVariant:
        path = self.root / commit_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "agent.py").write_text(INITIAL_AGENT_PY)
        with open(path / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return self._register(AgentVariant(commit_id=commit_id, path=path))

    def fork(self, parent_id: str) -> AgentVariant:
        """Copy a parent variant's directory into a fresh child variant (pre-edit)."""
        self._counter += 1
        commit_id = f"v{self._counter}"
        parent = self.get(parent_id)
        path = self.root / commit_id
        shutil.copytree(parent.path, path)
        return self._register(
            AgentVariant(commit_id=commit_id, path=path, parent_id=parent_id)
        )


def initial_swebench_config() -> dict:
    """mini-swe-agent's bundled SWE-bench config (prompts + limits + model + env)."""
    with open(package_dir / "config" / "benchmarks" / "swebench.yaml") as f:
        return yaml.safe_load(f)
