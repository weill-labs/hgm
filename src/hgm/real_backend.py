"""Real backend factories: a litellm model + per-instance Docker environment.

Architecture D — the evolved agent is mini-swe-agent powered by an OpenAI model via
litellm, run inside each SWE-bench instance's pinned Docker image (so the solver's shell
commands are sandboxed in the container, not on the host).
"""

from __future__ import annotations

import os

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel

# Most capable model that still fits a small (<$20) smoke. Bump to "gpt-5.4-pro" for a
# bigger-budget run (much pricier / slower).
SOLVER_MODEL = "gpt-5.4"

# The deliberate handicap that gives self-improvement headroom. The baseline and the HGM
# loop MUST start from this same config, so it lives here (one source of truth).
HANDICAP_SYSTEM = "You are an assistant. Use bash to fix the bug, then submit a patch."
HANDICAP_STEP_LIMIT = 12


def handicapped_config() -> dict:
    """Bundled SWE-bench config, weakened: terse system prompt + low step limit.

    This is the base agent both the baseline measurement and the HGM loop start from.
    """
    from hgm.variant import initial_swebench_config

    cfg = initial_swebench_config()
    cfg["agent"]["system_template"] = HANDICAP_SYSTEM
    cfg["agent"]["step_limit"] = HANDICAP_STEP_LIMIT
    return cfg


def make_litellm_model(
    model_name: str = SOLVER_MODEL, model_kwargs: dict | None = None
):
    """A mini-swe-agent Model backed by litellm/OpenAI. Cost is tracked per call and
    surfaced as message.extra.cost, which the agent's cost_limit and our SpendGuard read.

    ``drop_params=True`` lets litellm drop per-model-unsupported params: gpt-5-family
    models (mini/nano/5.0) reject ``temperature != 1`` and would otherwise raise
    UnsupportedParamsError; stronger models keep the temperature. Matches the bundled
    swebench.yaml config."""
    return LitellmModel(
        model_name=model_name,
        model_kwargs=model_kwargs or {"temperature": 0.0, "drop_params": True},
    )


def make_docker_env(
    image: str, cwd: str = "/testbed", *, timeout: int = 120, pull_timeout: int = 1800
):
    """A Docker environment for one SWE-bench instance image.

    forward_env passes the API key into the container only if a step needs it (the solve
    itself calls the model from the host process, so this is belt-and-suspenders).
    pull_timeout is generous because instance images are multi-GB on first pull.
    """
    return DockerEnvironment(
        image=image,
        cwd=cwd,
        timeout=timeout,
        pull_timeout=pull_timeout,
        forward_env=["OPENAI_API_KEY"],
    )


def make_selfimprove_sandbox_env(
    host_dir: str,
    *,
    image: str = "python:3.12-slim",
    timeout: int = 120,
    pull_timeout: int = 600,
):
    """Sandboxed environment for the self-improve meta-agent.

    Mounts the variant directory at /work in a generic container. The meta-agent's shell
    commands run INSIDE the container (so model-written code can't touch the host), but
    edits to /work persist back to the host variant dir. This is the sandbox required for
    live, self-modifying self-improvement.
    """
    return DockerEnvironment(
        image=image,
        cwd="/work",
        timeout=timeout,
        pull_timeout=pull_timeout,
        run_args=["--rm", "-v", f"{os.path.abspath(host_dir)}:/work"],
    )
