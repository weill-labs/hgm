# hgm — a clean-room Huxley-Gödel Machine

A from-scratch, tested reimplementation of the **Huxley-Gödel Machine**
([arXiv:2510.21614](https://arxiv.org/abs/2510.21614)) — a *self-improving coding agent*
that searches a tree of agent variants guided by **Clade-based Metaproductivity (CMP)** and
**Thompson Sampling**, rewriting its own source as it goes.

Built bottom-up and validated cheaply first: the core algorithm is proven for **$0** in a
simulator, then the *same* search loop is run live against real [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
agents scored by the official [SWE-bench](https://github.com/SWE-bench/SWE-bench) harness in
Docker — all under hard spend caps. Total real LLM spend for every live experiment: **~$26.8**.

## Headline results

- **The core claim holds (simulation, $0).** Under an equal eval budget, CMP-guided search
  finds higher-skill agents than the greedy/DGM baseline (+0.055 mean skill over 40 seeds).
  The advantage is an **inverted-U** in lineage-productivity variance — biggest at
  intermediate difficulty.
- **Live self-improvement is real and can be statistically significant.** Starting from a
  deliberately crippled agent (0.00 pass@1 on held-out hard SWE-bench instances), HGM
  discovered a variant scoring **0.35 — McNemar paired test p=0.016, clean 7–0 dominance**
  (it solved 7 instances the base couldn't and regressed on none), on *matched, held-out*
  instances.
- **The agent rewrites its own scaffold sensibly.** Across generations it raised its own
  step budget (5→12→20) and enriched its prompts — and the best variant descended from a
  *mediocre* parent, exactly the clade-credit case CMP is built for.
- **A quantified base-difficulty trade-off.** Too-weak a base starves the search of
  traction (2/3 seeds never expand); too-strong removes the headroom (lifts shrink below
  significance at n=20). The big significant jump and robust multi-seed gains live in
  different regimes — the inverted-U, now drawn in live SWE-bench data.

Full write-up with tables and honest caveats: **[`RESULTS.md`](RESULTS.md)**.

## How it works

```
        ┌──────────────────────────────────────────────┐
        │ HGMSearch  (the paper's contribution)          │
        │  tree of agent variants; each step either:     │
        │   EXPAND  pick a node by CMP (clade) Thompson  │
        │           sampling → self-improve it           │
        │   MEASURE pick a node by its own outcomes →    │
        │           evaluate it on one more task         │
        └───────────────┬───────────────┬───────────────┘
            Evaluator ───┘               └─── SelfImprover     ← pluggable protocols
        ┌───────────────────────┐   ┌──────────────────────────┐
        │ Simulated ($0)         │   │ Real (Architecture D)    │
        │ hidden-skill landscape │   │ mini-swe-agent + gpt-5.4 │
        └───────────────────────┘   │ SWE-bench scoring (Docker)│
                                     └──────────────────────────┘
```

The novel part — `tree.py` (CMP), `bandit.py` (Thompson Sampling), `search.py`
(expand/measure loop) — is ~300 lines and never changes between the simulated and live
backends. "An agent variant" is a directory snapshot of the agent's own files
(`agent.py` + `config.yaml`); self-improvement runs a meta-agent that edits those files in
a sandboxed container.

## Quickstart ($0, no API key, no Docker)

```bash
uv sync                                   # numpy + pytest + mini-swe-agent
uv run pytest -q                          # 21 tests
uv run python experiments/run_simulated.py    # CMP vs greedy, 40 seeds
uv run python experiments/sweep_drift.py      # the inverted-U
```

## Live experiments (cost money + need Docker)

Require Docker and an OpenAI key (`OPENAI_API_KEY`, or `OAI_KEY` in a gitignored `.env`).
Every run is bounded by a global `SpendGuard` plus a per-run `cost_limit`.

```bash
uv run python experiments/validate_scoring.py   # $0: gold patch -> resolved (Docker only)
uv run python experiments/baseline.py 30 42 6   # base agent pass@1 on 30 Lite instances
uv run python experiments/hard_loop.py 50 12     # full HGM loop on hard repos
uv run python experiments/significance.py 8      # 3-seed matched held-out + McNemar
```

## Layout

```
src/hgm/
  tree.py         Node + CMP (clade-pooled descendant evals)
  bandit.py       Beta-Bernoulli Thompson Sampling
  search.py       expand/measure outer loop (use_cmp flag toggles HGM vs greedy)
  interfaces.py   Evaluator / SelfImprover protocols
  simulated.py    hidden-skill landscape ($0 backend)
  variant.py      file-based agent variants + dynamic agent-class loader
  mini_runner.py  run a variant on a task
  self_improve.py meta-agent that edits a variant's own files
  real_eval.py    SWE-bench scoring via the official harness in Docker
  real_backend.py litellm model + Docker env factories + the base config
  spend.py        hard global spend cap
experiments/      simulation + all live runs
tests/            21 tests ($0)
```

Design notes and the algorithm spec read from the reference implementation are in
[`findings.md`](findings.md); plan and status in `task_plan.md` / `progress.md`.

## Scope & honesty

This reproduces the paper's *mechanisms* and the core CMP-vs-greedy claim, and shows a
statistically significant self-improvement at small scale. It does **not** reproduce the
paper's headline benchmark numbers — that needs its scale (hundreds of evals, more seeds, a
tuned base, ~40–60+ held-out instances for robust significance). What's here is a correct,
tested implementation you can run end-to-end for cents (sim) to tens of dollars (live).

## Credits

The paper: *Huxley-Gödel Machine* (Wang et al., 2025). Coding agent:
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent). Evaluation:
[SWE-bench](https://github.com/SWE-bench/SWE-bench). The search-loop structure is adapted in
spirit from the authors' reference code and [DGM](https://github.com/jennyzzt/dgm); this
implementation is clean-room (their code was used only as a spec and is not included here).
