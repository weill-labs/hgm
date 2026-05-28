"""Stage B (costs money): ONE live, sandboxed self-improvement step.

A real LLM (gpt-5.4) acting as the meta-agent edits the initial variant's OWN files
(agent.py / config.yaml) inside a Docker container (commands sandboxed; edits persist to
the host variant dir). We then show the diff it made to itself and validate the child.

    uv run python experiments/live_selfimprove.py
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

SELF_IMPROVE_CAP_USD = 20.0
META_COST_LIMIT_USD = 2.0


def _load_openai_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    for envf in (Path(".env"), Path.home() / ".env"):
        if not envf.exists():
            continue
        for line in envf.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() in ("OPENAI_API_KEY", "OAI_KEY") and v:
                os.environ["OPENAI_API_KEY"] = v
                return


def _diff(before: str, after: str, name: str) -> str:
    d = difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), f"a/{name}", f"b/{name}"
    )
    return "".join(d) or f"(no change to {name})\n"


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found (looked in OPENAI_API_KEY / OAI_KEY in .env).")
        return 2

    from hgm.interfaces import FAILED
    from hgm.real_backend import (
        SOLVER_MODEL,
        make_litellm_model,
        make_selfimprove_sandbox_env,
    )
    from hgm.self_improve import MetaConfig, MiniSelfImprover
    from hgm.spend import SpendCapExceeded, SpendGuard
    from hgm.variant import VariantStore, initial_swebench_config

    store = VariantStore("output_hgm/selfimprove/variants")
    initial = store.create_initial(initial_swebench_config())
    before_agent = initial.agent_py.read_text()
    before_config = initial.config_yaml.read_text()

    guard = SpendGuard(cap_usd=SELF_IMPROVE_CAP_USD)
    diagnosis = (
        "The agent sometimes gives up or submits empty patches on harder issues, and its "
        "prompt does not strongly encourage reproducing the bug before fixing. Improve its "
        "system/instance prompts and limits to be more reliable at producing correct patches."
    )
    improver = MiniSelfImprover(
        store,
        model_factory=lambda: make_litellm_model(SOLVER_MODEL),
        env_factory=make_selfimprove_sandbox_env,
        diagnose=lambda pid: diagnosis,
        meta=MetaConfig(step_limit=30, cost_limit=META_COST_LIMIT_USD),
        spend_guard=guard,
    )

    print(
        f"Meta-model : {SOLVER_MODEL}  (sandboxed in Docker, cost_limit ${META_COST_LIMIT_USD})"
    )
    print("Running ONE live self-improvement step on the initial agent...\n")

    try:
        child_id = improver.improve("initial")
    except SpendCapExceeded as e:
        print(f"ABORTED by spend cap: {e}")
        return 1

    print(f"spent      = ${guard.spent:.4f}")
    if child_id == FAILED:
        print("Self-improvement FAILED (child didn't validate or meta-agent errored).")
        return 1

    child = store.get(child_id)
    print(f"child      = {child_id} (parent={child.parent_id})\n")
    print("=== how the agent edited config.yaml ===")
    print(_diff(before_config, child.config_yaml.read_text(), "config.yaml"))
    print("=== how the agent edited agent.py ===")
    print(_diff(before_agent, child.agent_py.read_text(), "agent.py"))
    print("Child validated (imports + config parse). Live self-improvement works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
