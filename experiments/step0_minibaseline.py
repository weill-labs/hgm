"""Ablation Step 0: is a mid-tier model in the right regime for the CMP ablation?

Measures pass@1 of an UN-handicapped mini model (real bundled SWE-bench config: full
step_limit + full prompt) on hard-repo held-out instances. We only proceed to the full
4-arm ablation if this lands mid-range (~0.15-0.40): not ~0 (rescue regime, like
step_limit=5) and not ~0.5+ (near-ceiling, like gpt-5.4 — where edits are lateral).

    uv run python experiments/step0_minibaseline.py [model] [n] [workers]

Uses the same hard-repo / seed-999 partition as significance.py, so these instances are a
subset of the ablation's held-out set.
"""

from __future__ import annotations

import math
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-5-mini"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
HARD_REPO_PREFIXES = ("sphinx-doc__", "pytest-dev__", "matplotlib__", "sympy__")
SPEND_CAP_USD = 6.0
MAX_RUN_COST_USD = 0.5
OUT = Path("output_hgm/step0")


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


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found (OPENAI_API_KEY / OAI_KEY in .env).")
        return 2

    from hgm.real_backend import make_docker_env, make_litellm_model
    from hgm.real_eval import SweBenchEvaluator, load_lite_instances
    from hgm.spend import SpendCapExceeded, SpendGuard
    from hgm.variant import VariantStore, initial_swebench_config

    all_instances = load_lite_instances(split="test")
    hard = sorted(i for i in all_instances if i.startswith(HARD_REPO_PREFIXES))
    rng = random.Random(999)
    shuffled = hard[:]
    rng.shuffle(shuffled)
    held_out = sorted(
        shuffled[: min(50, len(hard) // 2)]
    )  # same region as significance.py
    instance_ids = held_out[:N]

    OUT.mkdir(parents=True, exist_ok=True)
    store = VariantStore(OUT / "variants")
    store.create_initial(
        initial_swebench_config()
    )  # UN-handicapped: full step_limit + prompt
    guard = SpendGuard(cap_usd=SPEND_CAP_USD)
    evaluator = SweBenchEvaluator(
        store,
        {i: all_instances[i] for i in instance_ids},
        model_factory=lambda cl: make_litellm_model(MODEL),
        env_factory=make_docker_env,
        spend_guard=guard,
        split="test",
        max_run_cost=MAX_RUN_COST_USD,
    )

    sl = initial_swebench_config()["agent"].get("step_limit")
    print(
        f"STEP 0 | model={MODEL} (UN-handicapped, step_limit={sl}) | {N} hard held-out instances\n"
    )

    results: dict[str, int] = {}
    lock = threading.Lock()

    def run_one(iid):
        try:
            out = evaluator.evaluate("initial", iid)
        except SpendCapExceeded:
            return iid, None
        except Exception as e:
            with lock:
                print(f"  [error] {iid}: {type(e).__name__}: {str(e)[:70]}", flush=True)
            return iid, None
        with lock:
            print(
                f"  [{len(results) + 1}/{N}] {iid}: {'RESOLVED' if out else 'failed'} (${guard.spent:.2f})",
                flush=True,
            )
        return iid, out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(run_one, i) for i in instance_ids]
        for f in as_completed(futs):
            iid, out = f.result()
            if out is not None:
                results[iid] = int(out)

    n = len(results)
    k = sum(results.values())
    p = k / n if n else 0.0
    lo, hi = wilson_ci(k, n)
    regime = (
        "MID (good for ablation)"
        if 0.12 <= p <= 0.45
        else ("FLOOR (rescue regime)" if p < 0.12 else "CEILING (lateral edits)")
    )
    print(f"\n==== STEP 0 ====  spent ${guard.spent:.4f}")
    print(f"pass@1 = {k}/{n} = {p:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"regime: {regime}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
