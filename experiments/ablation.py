"""The CMP ablation: does HGM's tree+CMP beat compute-matched controls?

See docs/ABLATION.md. Four conditions, identical compute (same self-improve steps S and
eval calls V), constant meta model, only the SELECTION differs:

  A. HGM (CMP)        tree, expand by clade Thompson sampling
  B. Greedy (DGM)     tree, expand by own-mean Thompson sampling
  C. Flat best-of-N   no tree: S depth-1 self-improvements of base; keep the best on a
                      val split  <-- key control for "is it just sample-N-edits-and-pick-best"
  D. Base             no improvement

Solver = an un-handicapped mid model (gpt-5-mini); meta (self-improve) model constant
across arms. Each (arm, seed) produces a best agent; ALL agents + base are then scored on
the SAME held-out matched set (n>=50, disjoint from search). Stats: per-arm pass@1 +
Wilson CI; exact McNemar (A vs base, A vs C, A vs B) on paired outcomes.

    uv run python experiments/ablation.py [n_seeds] [n_heldout] [spend_cap]
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_HELDOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 50
SPEND_CAP_USD = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
SOLVER_MODEL = "gpt-5-mini"  # un-handicapped mid solver (Step 0: 0.42, MID)
META_MODEL = "gpt-5.4"  # capable edit-generator, CONSTANT across arms
S_EXPANSIONS = 6  # self-improve steps per arm per seed
V_EVALS = 18  # in-search eval calls per arm per seed
HARD_REPO_PREFIXES = ("sphinx-doc__", "pytest-dev__", "matplotlib__", "sympy__")
EVAL_WORKERS = 8
MAX_RUN_COST_USD = 0.5
OUT = Path("output_hgm/ablation")


def _load_openai_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    for envf in (Path(".env"), Path.home() / ".env"):
        if not envf.exists():
            continue
        for line in envf.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() in ("OPENAI_API_KEY", "OAI_KEY") and v.strip().strip(
                    "\"'"
                ):
                    os.environ["OPENAI_API_KEY"] = v.strip().strip("\"'")
                    return


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) * 0.5**n)


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found.")
        return 2

    from hgm.interfaces import FAILED
    from hgm.real_backend import (
        make_docker_env,
        make_litellm_model,
        make_selfimprove_sandbox_env,
    )
    from hgm.real_eval import SweBenchEvaluator, load_lite_instances
    from hgm.search import HGMSearch, SearchConfig
    from hgm.self_improve import MetaConfig, MiniSelfImprover
    from hgm.spend import SpendCapExceeded, SpendGuard
    from hgm.variant import VariantStore, initial_swebench_config

    all_instances = load_lite_instances(split="test")
    hard = sorted(i for i in all_instances if i.startswith(HARD_REPO_PREFIXES))
    part_rng = random.Random(999)
    shuffled = hard[:]
    part_rng.shuffle(shuffled)
    n_heldout = min(N_HELDOUT, len(hard) // 2)
    held_out = sorted(shuffled[:n_heldout])
    search_pool = sorted(shuffled[n_heldout:])

    OUT.mkdir(parents=True, exist_ok=True)
    guard = SpendGuard(cap_usd=SPEND_CAP_USD)
    t0 = time.time()
    lock = threading.Lock()

    def log(m):
        with lock:
            print(f"[{time.time() - t0:6.0f}s ${guard.spent:6.2f}] {m}", flush=True)

    def base_config():
        return initial_swebench_config()  # UN-handicapped

    def mk_eval(store, insts):
        return SweBenchEvaluator(
            store,
            {i: all_instances[i] for i in insts},
            model_factory=lambda cl: make_litellm_model(SOLVER_MODEL),
            env_factory=make_docker_env,
            spend_guard=guard,
            split="test",
            max_run_cost=MAX_RUN_COST_USD,
        )

    def mk_improver(store):
        return MiniSelfImprover(
            store,
            model_factory=lambda: make_litellm_model(META_MODEL),
            env_factory=make_selfimprove_sandbox_env,
            diagnose=lambda pid: (
                "Improve this coding agent's prompts/limits/code so it "
                "fixes real bugs more reliably (reproduce, fix, verify)."
            ),
            meta=MetaConfig(step_limit=30, cost_limit=2.0),
            spend_guard=guard,
        )

    print(
        f"solver={SOLVER_MODEL} meta={META_MODEL} | seeds={N_SEEDS} | held_out={len(held_out)} "
        f"| pool={len(search_pool)} | S={S_EXPANSIONS} V={V_EVALS} | cap=${SPEND_CAP_USD}\n"
    )

    # ---- Stage 1: produce a best agent for each (arm, seed) -------------------------------
    def search_arm(arm: str, seed: int):
        rng = random.Random(7000 + seed)
        insts = sorted(rng.sample(search_pool, min(8, len(search_pool))))
        # fresh dir per (re)attempt — fork() uses copytree which fails on a stale dir
        shutil.rmtree(OUT / f"{arm}_s{seed}", ignore_errors=True)
        store = VariantStore(OUT / f"{arm}_s{seed}" / "variants")
        store.create_initial(base_config())
        ev, imp = mk_eval(store, insts), mk_improver(store)
        if arm in ("A", "B"):
            cfg = SearchConfig(
                max_task_evals=V_EVALS,
                init_measurements=3,
                use_cmp=(arm == "A"),
                max_expansions=S_EXPANSIONS,
                seed=0,
            )
            search = HGMSearch(ev, imp, tasks=insts, config=cfg)
            try:
                search.run()
            except SpendCapExceeded:
                pass
            pool = [
                n for n in search.nodes if n.num_evals > 0 and n.parent is not None
            ] or [n for n in search.nodes if n.num_evals > 0]
            log(
                f"{arm} s{seed}: {len(search.nodes)} variants, {search.n_expansions} expands"
            )
            best = (
                max(pool, key=lambda n: (n.mean_utility, n.num_evals))
                if pool
                else search.root
            )
            return store.get(best.commit_id).load_config()
        else:  # C: flat best-of-N
            children = []
            for _ in range(S_EXPANSIONS):
                try:
                    c = imp.improve("initial")
                except SpendCapExceeded:
                    break
                if c != FAILED:
                    children.append(c)
            cands = ["initial"] + children
            per = max(1, V_EVALS // len(cands))
            scores = {}
            for cid in cands:
                vinsts = rng.sample(insts, min(per, len(insts)))
                outs = []
                for t in vinsts:
                    try:
                        outs.append(ev.evaluate(cid, t))
                    except SpendCapExceeded:
                        break
                scores[cid] = (sum(outs) / len(outs)) if outs else 0.0
            log(
                f"C s{seed}: {len(children)} children, best score {max(scores.values()):.2f}"
            )
            best = max(cands, key=lambda c: scores[c])
            return store.get(best).load_config()

    # checkpoint dir: each completed arm-seed best config is persisted so a re-launch
    # skips searches it already finished.
    cfg_dir = OUT / "agentcfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    agents = {"base": base_config()}
    for f in cfg_dir.glob("*.json"):
        agents[f.stem] = json.loads(f.read_text())
    jobs = [
        (arm, s)
        for s in range(N_SEEDS)
        for arm in ("A", "B", "C")
        if f"{arm}_s{s}" not in agents
    ]
    log(
        f"Stage 1: {len(agents) - 1} arm-seeds cached; running {len(jobs)} concurrently..."
    )
    if jobs:
        with ThreadPoolExecutor(max_workers=min(9, len(jobs))) as ex:
            futs = {ex.submit(search_arm, arm, s): (arm, s) for arm, s in jobs}
            for f in as_completed(futs):
                arm, s = futs[f]
                try:
                    cfg = f.result()
                    agents[f"{arm}_s{s}"] = cfg
                    (cfg_dir / f"{arm}_s{s}.json").write_text(
                        json.dumps(cfg)
                    )  # checkpoint
                except Exception as e:
                    log(f"{arm} s{s} FAILED: {type(e).__name__}: {str(e)[:70]}")

    # ---- Stage 2: matched held-out eval of every agent ------------------------------------
    mstore = VariantStore(OUT / "matched" / "variants")
    for label, cfg in agents.items():
        mstore.create_initial(cfg, commit_id=label)
    m_eval = mk_eval(mstore, held_out)
    # resume: load already-scored (label, iid) outcomes from the checkpoint log.
    matched = {label: {} for label in agents}
    matched_path = OUT / "matched.jsonl"
    if matched_path.exists():
        for line in matched_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["label"] in matched:
                    matched[r["label"]][r["iid"]] = r["out"]
    write_lock = threading.Lock()
    pairs = [
        (label, iid)
        for label in agents
        for iid in held_out
        if iid not in matched[label]
    ]
    cached = sum(len(v) for v in matched.values())
    log(f"Stage 2: matched eval, {cached} cached, {len(pairs)} pairs to run...")

    def eval_one(label, iid):
        try:
            out = m_eval.evaluate(label, iid)
        except SpendCapExceeded:
            return label, iid, None
        except Exception:
            return label, iid, None
        if out is not None:
            with write_lock:  # checkpoint each result as it lands
                with open(matched_path, "a") as fh:
                    fh.write(
                        json.dumps({"label": label, "iid": iid, "out": int(out)}) + "\n"
                    )
        return label, iid, out

    if pairs:
        with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
            futs = [ex.submit(eval_one, label, iid) for label, iid in pairs]
            done = 0
            for f in as_completed(futs):
                label, iid, out = f.result()
                done += 1
                if out is not None:
                    matched[label][iid] = out
                if done % 25 == 0:
                    log(f"  matched {done}/{len(pairs)}")

    # ---- Stage 3: stats -------------------------------------------------------------------
    def arm_rate(arm):  # pooled over that arm's seed-agents
        ks = ns = 0
        for s in range(N_SEEDS):
            r = matched.get(f"{arm}_s{s}", {})
            ks += sum(r.values())
            ns += len(r)
        return ks, ns

    base_r = matched["base"]
    bk, bn = sum(base_r.values()), len(base_r)
    report = {
        "config": {
            "solver": SOLVER_MODEL,
            "meta": META_MODEL,
            "seeds": N_SEEDS,
            "held_out": len(held_out),
            "S": S_EXPANSIONS,
            "V": V_EVALS,
        },
        "base": {"resolved": bk, "n": bn, "pass_at_1": bk / bn if bn else 0},
        "arms": {},
        "mcnemar": {},
        "spent_usd": guard.spent,
    }

    def pooled_mcnemar(arm, other_r_fn):
        b = c = 0
        for s in range(N_SEEDS):
            ar = matched.get(f"{arm}_s{s}", {})
            for iid, av in ar.items():
                ov = other_r_fn(s, iid)
                if ov is None:
                    continue
                b += int(ov == 1 and av == 0)  # other wins
                c += int(ov == 0 and av == 1)  # this arm wins
        return b, c

    for arm in ("A", "B", "C"):
        k, n = arm_rate(arm)
        lo, hi = wilson_ci(k, n)
        report["arms"][arm] = {
            "resolved": k,
            "n": n,
            "pass_at_1": k / n if n else 0,
            "wilson_95ci": [lo, hi],
        }
        b, c = pooled_mcnemar(arm, lambda s, iid: base_r.get(iid))
        report["mcnemar"][f"{arm}_vs_base"] = {
            "other_wins": b,
            "arm_wins": c,
            "p": mcnemar_exact(b, c),
        }
    # A vs C and A vs B (paired per seed-instance)
    for other in ("C", "B"):
        b, c = pooled_mcnemar(
            "A", lambda s, iid: matched.get(f"{other}_s{s}", {}).get(iid)
        )
        report["mcnemar"][f"A_vs_{other}"] = {
            f"{other}_wins": b,
            "A_wins": c,
            "p": mcnemar_exact(b, c),
        }

    (OUT / "ablation.json").write_text(json.dumps(report, indent=2))

    print(f"\n==== ABLATION ====  spent ${guard.spent:.2f}  ({time.time() - t0:.0f}s)")
    print(f"  base   {bk}/{bn} = {report['base']['pass_at_1']:.3f}")
    for arm in ("A", "B", "C"):
        a = report["arms"][arm]
        nm = {"A": "HGM/CMP", "B": "greedy", "C": "flat best-of-N"}[arm]
        print(
            f"  {arm} {nm:<15} {a['resolved']}/{a['n']} = {a['pass_at_1']:.3f}  CI[{a['wilson_95ci'][0]:.2f},{a['wilson_95ci'][1]:.2f}]"
        )
    print("  McNemar:")
    for key, m in report["mcnemar"].items():
        sig = "SIG" if m["p"] < 0.05 else "n.s."
        print(f"    {key:<12} {m}  {sig}")
    print(f"  written: {OUT / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
