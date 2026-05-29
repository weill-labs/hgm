"""Significance test: is the HGM-improved agent reliably better than the base?

Addresses the two caveats of the suggestive hard-loop result:
  - UNMATCHED subsets -> we evaluate base and improved agents on the SAME held-out
    instances (disjoint from every search set), so the comparison is paired.
  - SINGLE seed -> we run the search across 3 seeds and test each run's best variant.

Design:
  1. Partition hard-repo Lite instances into a held-out set (matched eval) and a disjoint
     search pool.
  2. Run 3 independent HGM searches (concurrently) on the pool; take each run's best variant.
  3. Evaluate the base agent + the 3 discovered agents on ALL held-out instances (parallel).
  4. Stats: per-agent pass@1 + Wilson 95% CI, and exact McNemar paired test (base vs each
     discovered agent) on the matched outcomes.

    uv run python experiments/significance.py

Bounded by a global SpendGuard. Writes output_hgm/sig/significance.json.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HARD_REPO_PREFIXES = ("sphinx-doc__", "pytest-dev__", "matplotlib__", "sympy__")
N_HELDOUT = 20
SEARCH_SEEDS = [0, 1, 2]
SEARCH_INSTANCES = 8
SEARCH_EVALS = 14
HARD_STEP_LIMIT = 5
EVAL_WORKERS = 8
SPEND_CAP_USD = 25.0
MAX_RUN_COST_USD = 1.0
OUT = Path("output_hgm/sig")


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
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) * (0.5**n)
    return min(1.0, 2 * tail)


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found (OPENAI_API_KEY / OAI_KEY in .env).")
        return 2

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
    part_rng = random.Random(999)
    shuffled = hard[:]
    part_rng.shuffle(shuffled)
    n_heldout = min(N_HELDOUT, len(hard) // 2)
    held_out = sorted(shuffled[:n_heldout])
    search_pool = sorted(shuffled[n_heldout:])
    n_search = min(SEARCH_INSTANCES, len(search_pool))

    OUT.mkdir(parents=True, exist_ok=True)
    guard = SpendGuard(cap_usd=SPEND_CAP_USD)
    t0 = time.time()
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            print(f"[{time.time() - t0:6.0f}s ${guard.spent:5.2f}] {msg}", flush=True)

    def base_config() -> dict:
        cfg = handicapped_config()
        cfg["agent"]["step_limit"] = HARD_STEP_LIMIT
        return cfg

    print(
        f"model={SOLVER_MODEL} | hard instances={len(hard)} | held_out={len(held_out)} "
        f"| search_pool={len(search_pool)} | per-search={n_search}x{SEARCH_EVALS}"
    )
    print(f"held_out: {held_out}\n")

    # ---- Stage 1: 3 independent searches (concurrent) -> best variant config each -------
    def run_search_capture(seed: int):
        rng = random.Random(1000 + seed)
        insts = sorted(rng.sample(search_pool, n_search))
        store = VariantStore(OUT / f"seed{seed}" / "variants")
        store.create_initial(base_config())
        evaluator = SweBenchEvaluator(
            store,
            {i: all_instances[i] for i in insts},
            model_factory=lambda cl: make_litellm_model(SOLVER_MODEL),
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
                "Only 5 steps and a one-line prompt; it runs out of budget "
                "or submits empty patches. Raise step_limit and add a "
                "reproduce-fix-verify workflow to fix bugs reliably."
            ),
            meta=MetaConfig(step_limit=30, cost_limit=2.0),
            spend_guard=guard,
        )
        cfg = SearchConfig(
            max_task_evals=SEARCH_EVALS, init_measurements=3, use_cmp=True, seed=0
        )
        search = HGMSearch(evaluator, improver, tasks=insts, config=cfg)
        try:
            search.run()
        except SpendCapExceeded:
            pass
        evaluated = [n for n in search.nodes if n.num_evals > 0]
        children = [n for n in evaluated if n.parent is not None]
        pool = children or evaluated
        log(f"seed{seed}: {len(search.nodes)} variants explored")
        if not pool:
            return base_config()
        best = max(pool, key=lambda n: (n.mean_utility, n.num_evals))
        return store.get(best.commit_id).load_config()

    log("Stage 1: running 3 searches concurrently...")
    best_configs: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(SEARCH_SEEDS)) as ex:
        futs = {ex.submit(run_search_capture, s): s for s in SEARCH_SEEDS}
        for f in as_completed(futs):
            s = futs[f]
            try:
                best_configs[f"seed{s}_best"] = f.result()
            except Exception as e:
                log(f"seed{s} FAILED: {type(e).__name__}: {str(e)[:80]}")

    # ---- Stage 2: matched held-out evaluation (base + discovered agents) ----------------
    agents = {"base": base_config(), **best_configs}
    mstore = VariantStore(OUT / "matched" / "variants")
    for label, cfg in agents.items():
        mstore.create_initial(cfg, commit_id=label)
    m_eval = SweBenchEvaluator(
        mstore,
        {i: all_instances[i] for i in held_out},
        model_factory=lambda cl: make_litellm_model(SOLVER_MODEL),
        env_factory=make_docker_env,
        spend_guard=guard,
        split="test",
        max_run_cost=MAX_RUN_COST_USD,
    )
    log(
        f"Stage 2: matched eval of {list(agents)} on {len(held_out)} held-out instances..."
    )
    matched: dict[str, dict[str, int]] = {label: {} for label in agents}
    pairs = [(label, iid) for label in agents for iid in held_out]

    def eval_one(label: str, iid: str):
        try:
            return label, iid, m_eval.evaluate(label, iid)
        except SpendCapExceeded:
            return label, iid, None
        except Exception as e:
            log(f"  [error] {label}/{iid}: {type(e).__name__}: {str(e)[:60]}")
            return label, iid, None

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        futs = [ex.submit(eval_one, label, iid) for label, iid in pairs]
        done = 0
        for f in as_completed(futs):
            label, iid, out = f.result()
            done += 1
            if out is not None:
                matched[label][iid] = out
            if done % 10 == 0:
                log(f"  matched eval {done}/{len(pairs)}")

    # ---- Stage 3: stats -----------------------------------------------------------------
    def rate(label):
        r = matched[label]
        n = len(r)
        k = sum(r.values())
        return k, n, (k / n if n else 0.0)

    report = {
        "held_out": held_out,
        "agents": {},
        "mcnemar_vs_base": {},
        "spent_usd": guard.spent,
    }
    for label in agents:
        k, n, p = rate(label)
        lo, hi = wilson_ci(k, n)
        report["agents"][label] = {
            "resolved": k,
            "n": n,
            "pass_at_1": p,
            "wilson_95ci": [lo, hi],
        }

    base_r = matched["base"]
    for label in best_configs:
        xr = matched[label]
        common = [i for i in held_out if i in base_r and i in xr]
        b = sum(1 for i in common if base_r[i] == 1 and xr[i] == 0)  # base wins
        c = sum(1 for i in common if base_r[i] == 0 and xr[i] == 1)  # improved wins
        report["mcnemar_vs_base"][label] = {
            "n_matched": len(common),
            "base_only": b,
            "improved_only": c,
            "p_value": mcnemar_exact(b, c),
        }

    (OUT / "significance.json").write_text(json.dumps(report, indent=2))

    print(
        f"\n==== SIGNIFICANCE ====  spent ${guard.spent:.4f}  ({time.time() - t0:.0f}s)"
    )
    for label in agents:
        a = report["agents"][label]
        print(
            f"  {label:<14} {a['resolved']}/{a['n']} = {a['pass_at_1']:.3f}  "
            f"95% CI [{a['wilson_95ci'][0]:.2f}, {a['wilson_95ci'][1]:.2f}]"
        )
    print("  McNemar (base vs improved), exact two-sided:")
    for label, m in report["mcnemar_vs_base"].items():
        sig = "SIGNIFICANT" if m["p_value"] < 0.05 else "n.s."
        print(
            f"    {label:<14} improved_only={m['improved_only']} base_only={m['base_only']} "
            f"(n={m['n_matched']})  p={m['p_value']:.3f}  {sig}"
        )
    print(f"  written: {OUT / 'significance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
