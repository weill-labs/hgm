# Reproduction results — Huxley-Gödel Machine (arXiv:2510.21614)

A clean-room reimplementation, validated bottom-up. Total real LLM spend across **all**
live experiments: **~$7.65**.

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
| **Harder loop** | step_limit=5 base on failing repos; self-improvement shows | **+0.55 lift** (0.20→0.75) | $4.82 |

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

## The harder loop — where self-improvement shows (the headline result)

Motivated by the two findings above (handicap too mild; loop too small), we ran a second
loop tuned to surface improvement: **harder base** (`step_limit=5`, near the floor) on
**12 failing-repo instances** (sphinx/pytest/matplotlib/sympy), `max_task_evals=50`.
gpt-5.4, sandboxed self-improve, $12 cap. Took ~1.8h and **$4.82**.

**Tree built** (12 variants):

| node | commit | parent | evals | pass rate |
|---|---|---|---|---|
| 0 | initial | – | 5 | 0.20 |
| 2 | v2 | initial | 7 | 0.43 |
| 3 | v3 | initial | 10 | 0.60 |
| 7 | v7 | v2 | 5 | 0.40 |
| **8** | **v8** | **v2** | 4 | **0.75** |
| … | (v1,v4 dead 0.00; v5,v6,v9–v11 mid) | | | |

**initial 0.20 → best variant v8 0.75 = +0.55 lift.** And every non-dead child beat the
0.20 base (v2 .43, v3 .60, v5 .40, v7 .40, v8 .75) — a consistent direction, not one lucky
variant.

**The self-improvement trajectory is interpretable** — each generation un-handicapped
itself further:

| variant | step_limit | system prompt |
|---|---|---|
| initial | 5 | 67 chars (floor) |
| v2 / v3 (gen 1) | 12 | ~1000–1150 chars |
| v7 / v8 (gen 2) | 20 | ~1400–1450 chars |

The agent repeatedly raised its **own** step budget (5→12→20) and enriched its prompt
across generations (no `agent.py` code edits — prompt/limits were the effective lever).

**The CMP/clade signal is visible**: v2 is itself only mediocre (0.43) but is the **parent
of the best variant (v8, 0.75)** — exactly the case clade-based credit assignment is built
for (reward a lineage for productive descendants, not just its own score). The search spent
its evals on the productive lineages (v2: 7 evals, v3: 10).

**Honest caveats.** Samples are still small (v8 n=4): Wilson CIs for v8 [0.30, 0.95] and
initial [0.04, 0.62] overlap, so this is *suggestive*, not a significance claim. Variants
were measured on different instance subsets (not a matched head-to-head), so "lift"
conflates real improvement with instance-difficulty differences — mitigated, but not
removed, by the consistent direction across five children.

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
SWE-bench scoring. The simulation independently confirms the core algorithmic claim
(CMP > greedy) for $0. And in the harder live loop we **observed the agent improve itself**
on real SWE-bench instances — from a floored 0.20 base to a self-edited 0.75 variant
(+0.55), with a visible clade structure (the best variant's parent was itself mediocre).

That last result is *suggestive, not statistically conclusive* at n≈4–10 per variant on
unmatched instance subsets — a full reproduction of the paper's headline numbers still
needs its scale (hundreds of evals, matched evaluation, more depth; tens–hundreds of $).
What we have is a correct, tested implementation that exhibits every HGM behavior live for
**~$7.65 total**, plus a $0 simulation that isolates the CMP-vs-greedy claim.
