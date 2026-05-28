"""A HARDER live HGM loop, designed to make self-improvement *visible*.

Two changes from live_loop.py, motivated by the baseline (handicap was too mild — gpt-5.4
still hit 0.567):
  1. Harder handicap: step_limit=5 (near the floor). The base agent's main lever to
     improve is to raise its OWN step budget / enrich its prompt — exactly the edit we
     observed children make — so improvement should show as winning instances the parent
     loses.
  2. Failing-repo subset: only instances from the repos the base agent struggled with
     (sphinx, pytest, matplotlib, sympy), where there is real headroom.

Single-threaded search (faithful, deterministic) — expect a long wall-clock at
max_task_evals=50 (each measure is a Dockerized solve+score). Bounded by SpendGuard +
per-run cost_limit + the outer `timeout`.

    uv run python experiments/hard_loop.py [max_task_evals] [n_instances]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

MAX_TASK_EVALS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_INSTANCES = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SEED = 42
HARD_STEP_LIMIT = 5
HARD_REPO_PREFIXES = ("sphinx-doc__", "pytest-dev__", "matplotlib__", "sympy__")
SPEND_CAP_USD = 12.0
MAX_RUN_COST_USD = 1.0
OUT = Path("output_hgm/hard_loop")


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


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found (OPENAI_API_KEY / OAI_KEY in .env).")
        return 2

    import random

    from hgm.real_backend import (
        SOLVER_MODEL,
        handicapped_config,
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
    hard = sorted(i for i in all_instances if i.startswith(HARD_REPO_PREFIXES))
    rng = random.Random(SEED)
    instance_ids = sorted(rng.sample(hard, min(N_INSTANCES, len(hard))))
    instances = {i: all_instances[i] for i in instance_ids}

    OUT.mkdir(parents=True, exist_ok=True)
    base_cfg = handicapped_config()
    base_cfg["agent"]["step_limit"] = HARD_STEP_LIMIT  # harder than the baseline's 12
    store = VariantStore(OUT / "variants")
    store.create_initial(base_cfg)
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
            "The agent has a one-line prompt and only 5 steps, so it often runs out of "
            "budget or submits an empty patch on real bugs. Improve its prompts and limits "
            "(e.g. raise step_limit, add a reproduce-then-fix-then-verify workflow) so it "
            "reliably produces correct patches."
        ),
        meta=MetaConfig(step_limit=30, cost_limit=2.0),
        spend_guard=guard,
    )

    # --- lightweight progress logging (wrap the two paid primitives) ----------
    t0 = time.time()
    orig_eval, orig_improve = evaluator.evaluate, improver.improve

    def logged_eval(commit_id, task):
        out = orig_eval(commit_id, task)
        print(
            f"  [{time.time() - t0:6.0f}s ${guard.spent:5.2f}] measure {commit_id} on {task}: "
            f"{'RESOLVED' if out else 'failed'}",
            flush=True,
        )
        return out

    def logged_improve(parent):
        child = orig_improve(parent)
        print(
            f"  [{time.time() - t0:6.0f}s ${guard.spent:5.2f}] expand {parent} -> {child}",
            flush=True,
        )
        return child

    evaluator.evaluate, improver.improve = logged_eval, logged_improve

    cfg = SearchConfig(
        max_task_evals=MAX_TASK_EVALS,
        init_measurements=4,
        alpha=0.6,
        use_cmp=True,
        eval_random_level=1.0,
        seed=0,
    )

    print(f"HARD LOOP | model={SOLVER_MODEL} step_limit={HARD_STEP_LIMIT}")
    print(f"instances ({len(instance_ids)} from {HARD_REPO_PREFIXES}): {instance_ids}")
    print(f"max_task_evals={MAX_TASK_EVALS}, cap=${SPEND_CAP_USD}\n")

    search = HGMSearch(evaluator, improver, tasks=instance_ids, config=cfg)
    try:
        search.run()
    except SpendCapExceeded as e:
        print(f"\nABORTED by spend cap: {e}")

    print(f"\n==== RESULT ====  spent ${guard.spent:.4f}  ({time.time() - t0:.0f}s)")
    print(f"{'node':<6}{'commit':<10}{'parent':<10}{'evals':<7}{'mean':<6}")
    for node in search.nodes:
        p = node.parent.commit_id if node.parent else "-"
        m = node.mean_utility if node.num_evals else float("nan")
        print(f"{node.id:<6}{node.commit_id:<10}{p:<10}{node.num_evals:<7}{m:<6.2f}")

    evaluated = [n for n in search.nodes if n.num_evals > 0]
    children = [n for n in evaluated if n.parent is not None]
    if children:
        best_child = max(children, key=lambda n: (n.mean_utility, n.num_evals))
        print(
            f"\ninitial   : {search.root.mean_utility:.2f} (n={search.root.num_evals})"
        )
        print(
            f"best child: {best_child.commit_id} {best_child.mean_utility:.2f} (n={best_child.num_evals})"
        )
        lift = best_child.mean_utility - search.root.mean_utility
        print(f"lift vs initial on these hard instances: {lift:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
