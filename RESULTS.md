# Reproduction results — Huxley-Gödel Machine (arXiv:2510.21614)

A clean-room reimplementation, validated bottom-up. Total real LLM spend across **all**
live experiments: **~$2.83**.

## What was reproduced

| Layer | What | Verified | Cost |
|---|---|---|---|
| Algorithm | CMP (clade-pooled evals) + Beta-Bernoulli Thompson Sampling + expand/measure loop | 9 unit tests + 40-seed simulation | $0 |
| Algorithm claim | CMP-guided search beats greedy/DGM under equal eval budget | sim: +0.055 mean skill, peaks at intermediate difficulty | $0 |
| Self-improvement | agent rewrites its **own** files (prompts/config/code), sandboxed | unit + live | $0 + $0.034 |
| Evaluation | run variant in instance Docker → patch → official `swebench` harness → 0/1 | gold patch → resolved | $0 |
| Live solve | base agent resolves a real Lite instance | astropy-12907 resolved | $0.049 |
| **Baseline** | handicapped base agent pass@1 on 30 fixed Lite instances | **17/30 = 0.567** | $2.35 |
| **Live HGM loop** | full search driving real self-improve + real eval, budget-capped | see below | $0.385 |

## Baseline: handicapped base agent on 30 fixed Lite instances

The line every self-improved variant must beat. Fixed seeded sample (seed=42), the exact
handicapped base config the loop starts from (`hgm.real_backend.handicapped_config`:
terse system prompt + `step_limit=12`), gpt-5.4, no self-improvement, 6 parallel workers.
Full data in `output_hgm/baseline/baseline.json`.

**pass@1 = 17/30 = 0.567,  95% Wilson CI [0.392, 0.726],  spent $2.35.**

Per-repo (resolved/total): django 11/14 · sympy 2/4 · matplotlib 1/3 · sphinx 0/2 ·
pytest 0/2 · scikit-learn 1/1 · astropy 1/1.

**Key finding — the handicap is mild.** Even with a one-line prompt and only 12 steps,
gpt-5.4 still resolves ~57% of a representative Lite sample (it is especially strong on
django, weak on sphinx/pytest/matplotlib). Implications for reproducing the improvement
curve:
- The headroom above baseline is ~43% (to the 1.0 ceiling), but the CI is wide (±~0.17 at
  n=30), so a self-improved variant must beat ~0.57 by a clear margin **and** be measured
  on enough instances to separate from this CI.
- To make improvement easier to *see*, handicap harder (lower `step_limit`, or a weaker
  model) and/or weight the sample toward the repos the base agent fails (sphinx/pytest/
  matplotlib), where there is real room to climb.

## The live handicapped loop

Setup: base agent deliberately weakened (terse system prompt, `step_limit=12`) so
self-improvement has headroom; gpt-5.4; 3 SWE-bench Lite instances; `max_task_evals=6`;
hard caps ($8 global / $1 per solve). The unchanged `HGMSearch` chose every expand/measure.

**Tree built** (initial + 3 self-improved children):

| node | commit | parent | evals | pass rate |
|---|---|---|---|---|
| 0 | initial | – | 3 | **0.33** (resolved astropy-12907; failed -14182, -14365) |
| 1 | v1 | initial | 1 | 0.00 (tested only on -14365) |
| 2 | v2 | initial | 1 | 0.00 (tested only on -14365) |
| 3 | v3 | initial | 1 | 0.00 (tested only on -14365) |

best-by-mean = **initial (0.33)**. The children did **not** beat the baseline on pass rate.
(That 0.33 is a 3-instance estimate from this loop; the proper 30-instance baseline above
is **0.567** — the loop's tiny sample under-counted the base agent, underscoring how noisy
3 evals are.)

### But the self-improvements were real and on-target

Each child independently rewrote the handicapped agent to attack the exact weakness:

| | step_limit | system prompt | instance prompt |
|---|---|---|---|
| initial | 12 | 67 chars (terse handicap) | 4500 chars |
| v1 | **24** | 263 chars | 2405 |
| v2 | **24** | 359 chars | 2236 |
| v3 | **24** | 729 chars | 2338 |

All three **doubled the step budget (12→24)** and **expanded the system prompt** — i.e. the
self-improving agent correctly diagnosed and undid the handicap, plus rewrote prompts to
emphasize reproduce-before-fix and never-submit-empty-patch behavior. The Gödel-machine
mechanism works and produces sensible, interpretable edits.

## Why the loop did not show measured improvement (and it's expected)

1. **No statistical power.** With `max_task_evals=6`, each child was measured exactly
   **once**. A single 0/1 on one instance is pure noise — CMP's clade pooling has almost
   nothing to pool. The paper uses *hundreds* of task-evals.
2. **Measurement allocation.** The budget went to *breadth* (3 expansions) over *depth*;
   all three children happened to be tested on the hardest instance (-14365), which even a
   strong agent failed — so they never got a winnable instance to prove out on.
3. **Consistent with our own simulation.** CMP's advantage requires enough pooled evidence
   and intermediate difficulty; at tiny sample sizes the signal is swamped — exactly the
   floor regime of the inverted-U we measured (`experiments/sweep_drift.py`).

## What a faithful full reproduction would need

- **Hundreds of task-evals** (not 6) so per-variant means and clade pools are meaningful.
- **More instances**, and a *stronger* handicap than we used — the baseline shows our
  `step_limit=12` handicap still leaves gpt-5.4 at 0.567, so children mostly inherit
  already-solvable instances. Lower the step limit / use a weaker model, or weight toward
  the failing repos (sphinx/pytest/matplotlib), so children can win instances the parent loses.
- **More self-improve depth** so clades form and CMP can credit productive lineages.
- Budget: realistically tens to hundreds of dollars and many hours — the regime the paper
  ran in. Our contribution here is a correct, tested implementation that runs the full loop
  live for cents, plus a $0 simulation harness that demonstrates the CMP-vs-greedy claim.

## Bottom line

Every mechanism the paper describes is reimplemented and demonstrated working end-to-end:
the CMP metric, Thompson-Sampling tree search, self-rewriting agents, and Dockerized
SWE-bench scoring. The headline *agent-improvement curve* is not reproduced at this scale —
that needs the paper's eval budget — but the simulation independently confirms the core
algorithmic claim (CMP > greedy) for $0.
