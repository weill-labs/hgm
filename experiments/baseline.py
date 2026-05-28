"""Establish a proper baseline: pass@1 of the HANDICAPPED base agent on a fixed,
seeded sample of SWE-bench Lite, with NO self-improvement.

This is the line every self-improved variant must beat. Uses the exact same base config
the HGM loop starts from (hgm.real_backend.handicapped_config), a fixed seed for a
reproducible instance sample, parallel solves, and a hard spend cap.

    uv run python experiments/baseline.py [n_instances] [seed] [workers]

Writes per-instance results + summary to output_hgm/baseline/baseline.json.

NOTE: each instance pulls a multi-GB image and runs its test suite. 30 instances is
~tens of GB and ~30-60 min even parallelized.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

N_INSTANCES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
SPEND_CAP_USD = 12.0
MAX_RUN_COST_USD = 1.0
OUT = Path("output_hgm/baseline")


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


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (robust at small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found (OPENAI_API_KEY / OAI_KEY in .env).")
        return 2

    from hgm.real_backend import (
        HANDICAP_STEP_LIMIT,
        SOLVER_MODEL,
        handicapped_config,
        make_docker_env,
        make_litellm_model,
    )
    from hgm.real_eval import SweBenchEvaluator, load_lite_instances
    from hgm.spend import SpendCapExceeded, SpendGuard
    from hgm.variant import VariantStore

    all_instances = load_lite_instances(split="test")
    rng = random.Random(SEED)
    instance_ids = sorted(
        rng.sample(sorted(all_instances), min(N_INSTANCES, len(all_instances)))
    )
    instances = {i: all_instances[i] for i in instance_ids}

    OUT.mkdir(parents=True, exist_ok=True)
    store = VariantStore(OUT / "variants")
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

    print(f"BASELINE | model={SOLVER_MODEL} handicap step_limit={HANDICAP_STEP_LIMIT}")
    print(
        f"sample: {N_INSTANCES} Lite instances (seed={SEED}), {WORKERS} workers, cap ${SPEND_CAP_USD}\n"
    )

    results: dict[str, int] = {}
    print_lock = threading.Lock()

    def run_one(iid: str):
        try:
            outcome = evaluator.evaluate("initial", iid)
        except SpendCapExceeded:
            return iid, None
        except (
            Exception
        ) as e:  # an instance harness/Docker failure shouldn't kill the run
            with print_lock:
                print(f"  [error] {iid}: {type(e).__name__}: {str(e)[:80]}")
            return iid, None
        with print_lock:
            done = len(results) + 1
            print(
                f"  [{done}/{N_INSTANCES}] {iid}: {'RESOLVED' if outcome else 'failed'}  (${guard.spent:.2f})"
            )
        return iid, outcome

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(run_one, iid) for iid in instance_ids]
        for fut in as_completed(futures):
            iid, outcome = fut.result()
            if outcome is not None:
                results[iid] = int(outcome)

    n = len(results)
    k = sum(results.values())
    rate = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)

    summary = {
        "model": SOLVER_MODEL,
        "handicap_step_limit": HANDICAP_STEP_LIMIT,
        "seed": SEED,
        "n_requested": N_INSTANCES,
        "n_scored": n,
        "resolved": k,
        "pass_at_1": rate,
        "wilson_95ci": [lo, hi],
        "spent_usd": guard.spent,
        "per_instance": results,
    }
    (OUT / "baseline.json").write_text(json.dumps(summary, indent=2))

    print(f"\n==== BASELINE ====")
    print(f"pass@1 = {k}/{n} = {rate:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"spent  = ${guard.spent:.4f}")
    print(f"written: {OUT / 'baseline.json'}")
    if n < N_INSTANCES:
        print(
            f"NOTE: only {n}/{N_INSTANCES} scored (spend cap or errors) — rate is on scored only."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
