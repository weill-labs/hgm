"""A small HANDICAPPED live HGM loop (costs money, bounded by SpendGuard).

We deliberately weaken the base agent (terse prompt, low step_limit) so there is room for
self-improvement to help — the intermediate-difficulty regime where our simulation showed
CMP's biggest advantage. The unchanged HGMSearch then runs over a few SWE-bench Lite
instances, issuing real self-improve (expand) and real solve+score (measure) steps.

    uv run python experiments/live_loop.py

Tunables below. Hard stops: global SpendGuard cap + per-run cost ceiling + wall timeout.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- tunables ---------------------------------------------------------------
N_INSTANCES = 3  # distinct Lite instances (each needs a multi-GB image pull)
MAX_TASK_EVALS = 6  # total solve+score steps across the whole tree
INIT_MEASUREMENTS = 2  # seed the root before expanding
SPEND_CAP_USD = 8.0  # global hard cap
MAX_RUN_COST_USD = 1.0  # per-solve ceiling
SOLVER_STEP_LIMIT = 12  # the handicap: few steps to fix a bug
HANDICAP_SYSTEM = "You are an assistant. Use bash to fix the bug, then submit a patch."


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


def handicapped_config() -> dict:
    """Bundled SWE-bench config, weakened: terse system prompt + low step limit."""
    from hgm.variant import initial_swebench_config

    cfg = initial_swebench_config()
    cfg["agent"]["system_template"] = HANDICAP_SYSTEM
    cfg["agent"]["step_limit"] = SOLVER_STEP_LIMIT
    return cfg


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found (OPENAI_API_KEY / OAI_KEY in .env).")
        return 2

    from hgm.real_backend import (
        SOLVER_MODEL,
        make_docker_env,
        make_litellm_model,
        make_selfimprove_sandbox_env,
    )
    from hgm.real_eval import SweBenchEvaluator, load_lite_instances
    from hgm.search import HGMSearch, SearchConfig
    from hgm.self_improve import MetaConfig, MiniSelfImprover
    from hgm.spend import SpendCapExceeded, SpendGuard
    from hgm.variant import VariantStore

    all_instances = load_lite_instances(split="test")
    instance_ids = sorted(all_instances)[:N_INSTANCES]
    instances = {i: all_instances[i] for i in instance_ids}

    store = VariantStore("output_hgm/loop/variants")
    store.create_initial(handicapped_config())
    guard = SpendGuard(cap_usd=SPEND_CAP_USD)

    evaluator = SweBenchEvaluator(
        store,
        instances,
        model_factory=lambda cost_limit: make_litellm_model(SOLVER_MODEL),
        env_factory=make_docker_env,
        spend_guard=guard,
        split="test",
        max_run_cost=MAX_RUN_COST_USD,
    )
    improver = MiniSelfImprover(
        store,
        model_factory=lambda: make_litellm_model(SOLVER_MODEL),
        env_factory=make_selfimprove_sandbox_env,
        diagnose=lambda pid: (
            "The agent has a very terse prompt and a low step budget, and sometimes fails "
            "or submits empty patches. Improve its prompts/limits to fix bugs more reliably."
        ),
        meta=MetaConfig(step_limit=30, cost_limit=2.0),
        spend_guard=guard,
    )

    cfg = SearchConfig(
        max_task_evals=MAX_TASK_EVALS,
        init_measurements=INIT_MEASUREMENTS,
        alpha=0.6,
        use_cmp=True,
        eval_random_level=1.0,
        seed=0,
    )

    print(f"Model {SOLVER_MODEL} | handicap step_limit={SOLVER_STEP_LIMIT}")
    print(f"Instances ({N_INSTANCES}): {instance_ids}")
    print(
        f"Budget: max_task_evals={MAX_TASK_EVALS}, cap=${SPEND_CAP_USD}, per-run=${MAX_RUN_COST_USD}\n"
    )

    search = HGMSearch(evaluator, improver, tasks=instance_ids, config=cfg)
    try:
        result = search.run()
    except SpendCapExceeded as e:
        print(f"\nABORTED by spend cap: {e}")
        result = None

    print(f"\n==== RESULT ====  spent ${guard.spent:.4f}")
    print(f"action history: {search.history}")
    print(f"{'node':<8}{'commit':<10}{'parent':<10}{'evals':<7}{'mean':<6}")
    for node in search.nodes:
        p = node.parent.commit_id if node.parent else "-"
        print(
            f"{node.id:<8}{node.commit_id:<10}{p:<10}{node.num_evals:<7}{node.mean_utility:<6.2f}"
        )

    if result is not None and len(search.nodes) > 1:
        best = result.best_by_mean()
        print(
            f"\nbest-by-mean: {best.commit_id} (mean {best.mean_utility:.2f}, n={best.num_evals})"
        )
        print(f"initial mean: {search.root.mean_utility:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
