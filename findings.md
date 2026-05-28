# Findings — Reproducing the Huxley-Gödel Machine (HGM)

Paper: https://arxiv.org/abs/2510.21614
Official code: https://github.com/metauto-ai/HGM

## What the paper is
- **Huxley-Gödel Machine (HGM)**: a self-improving *coding agent*. Maintains a tree of
  agent variants; each node is a version of the agent's own source code. The agent
  rewrites its own code (self-improve step), candidates are evaluated on SWE-bench, and
  the best lineages are explored further.
- **Core contribution — CMP (Clade-based Metaproductivity)**: instead of scoring an
  agent by its *own* benchmark success, score it by the success of its *descendants*.
  This measures improvement *potential*, avoiding greedy local optima.
- **Search**: Thompson Sampling over the tree to pick which node to expand/evaluate.
- **Not model training**: no GPUs, no gradients. Cost = OpenAI API calls + Docker
  container-hours for evaluations.

## Official repo facts (verified via raw files on main)
- Python 3.10; `pip install -r requirements.txt`. Docker required (1 container/eval).
- Needs `OPENAI_API_KEY`.
- Default model (`config.py` LLMConfig): `gpt-5-mini` for self_improve / downstream / diagnose.
- Entry point: `python hgm.py` (driven by `config.yaml` + CLI overrides). Launched via `./run.sh`.
- `run.sh` does: `docker system prune -f` → activate env → `python hgm.py`.

### Key config defaults (config.py)
- OptimizationConfig: alpha=0.6, beta=1.0, cool_down=False, eval_random_level=1.0,
  n_pseudo_descendant_evals=10000
- ExecutionConfig: max_workers=16, self_improve_timeout=3600s, evaluation_timeout=3600s,
  **max_task_evals=800**  ← total evaluation budget (the expensive knob)
- EvaluationConfig: full_eval=False, polyglot=False
- PathConfig: output_dir=None, continue_from=None, initial_agent_name=""

### CLI args accepted by hgm.py
`--config --max_task_evals --max_workers --continue_from --output_dir --polyglot`
`--self_improve_llm --downstream_llm --diagnose_llm --alpha --beta --cool_down --initial_agent_name`

### SWE-bench setup (from README, VERIFY exact hash when cloning)
Inside `swe_bench/`:
```
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e36   # NOTE: README-quoted hash is 32 chars, not a full 40-char SHA — re-read README literally before trusting
pip install -e .
```

## Local environment scan (2026-05-28)
- Docker v29.1.3, daemon OK ✅
- 504 GB free disk, 32 cores, 122 GB RAM ✅
- git 2.53.0 ✅
- Python: only system 3.14.5 (need 3.10) → provision via `uv`
- conda: NOT installed → using `uv` instead (per user tooling preference)
- OPENAI_API_KEY: NOT set → **blocker, user must provide**

## Cost reality
- A full run (max_task_evals=800, full SWE-bench) ≈ hundreds of Dockerized evals × many
  gpt-5-mini calls each → plausibly hundreds of $ and days of wall-clock. Stochastic
  (Thompson Sampling + LLM nondeterminism) so exact paper numbers won't reproduce.
- **Chosen scope: smoke test, SWE-bench Lite, budget < $20.**

## ALGORITHM SPEC (from reading reference source — ground truth)

### Data structure (`tree.py`)
- `Node`: `commit_id`, `children[]`, `utility_measures[]` (list of 0/1 per-task results),
  `parent_id`, `id`.
- `num_evals = len(utility_measures)`; `mean_utility = mean(utility_measures)` (inf if empty).
- `get_sub_tree()` = node + all descendants (DFS).
- `get_pseudo_decendant_evals(num_pseudo)`: **pseudo-count smoothing** — if node has
  `< num_pseudo` evals return its own `utility_measures`, else return `[mean]*num_pseudo`.
- `get_decendant_evals(num_pseudo)` = pseudo-evals(self) ++ every descendant's
  `utility_measures`. **THIS is the CMP estimate**: the clade's pooled binary outcomes.

### CMP (Clade-based Metaproductivity)
A node's metaproductivity ≈ pooled task-resolution outcomes of itself + ALL descendants.
Expansion (choosing whom to mutate) uses **descendant-pooled** evals (CMP).
Measurement (choosing whom to evaluate more) uses the node's **own** evals.
This is the core departure from greedy "pick current best" (DGM).

### Thompson Sampling (`hgm.py: TS_sample`)
Beta-Bernoulli bandit. For each candidate eval-list `de`:
`alpha = 1 + sum(de)`, `beta = 1 + len(de) - sum(de)`; sample `theta ~ Beta(alpha,beta)`;
pick `argmax(theta)`. Optional `cool_down` scales alphas/betas by
`max_evals^beta / (max_evals - n_task_evals)^beta` to sharpen late.

### Main loop (`hgm.py`)
Thread pool submits `max_task_evals*100` `sample()` jobs; each respects global budget
`n_task_evals < max_task_evals`.
- `sample()`: **expand-vs-measure** decision via `n_task_evals^alpha >= num_nodes-1+pending`.
  - expand → among nodes with finite & positive mean_utility, TS over CMP (descendant) evals
    → `sample_child` → new node.
  - measure → among not-fully-evaluated nodes, TS over OWN evals → pick 1 task (random w.p.
    `eval_random_level` else first) → `eval_agent` on that task → append 0/1.
- Warm start: a few `expand()` calls before the sampling loop.

### Self-improvement step (`hgm_utils.sample_child` + `self_improve_step.py`)
1. `choose_entry(parent)`: pick a weakness category from parent's metadata —
   `solve_empty_patches` / `solve_stochasticity` / `solve_contextlength` (each w.p. 0.25),
   else a random unresolved/resolved instance id.
2. `diagnose_problem`: LLM turns that into a natural-language "problem statement" describing
   what to improve in the agent.
3. Run the coding agent on ITS OWN source (`coding_agent.py --self_improve`) in Docker →
   produces `model_patch.diff` → git-commit → new variant (commit_id = run_id).

### Evaluation (`hgm_utils.eval_agent` → `swe_bench/harness.py`)
Runs the agent at a commit on SWE-bench task(s) in Docker, applies patch, runs tests,
returns 0/1 per task. Default task set (non-full): `swe_bench/subsets/small.json` +
`medium.json`, shuffled with `seed=42`. Initial agent dataset hardcodes
`princeton-nlp/SWE-bench_Verified`.

### Component sizes (reuse-vs-rewrite)
- Novel/core (rewrite worth it): `tree.py` (76), search loop in `hgm.py` (~150 relevant),
  CMP in `tree.py`, TS (~18 lines). ~300 lines total = the paper's contribution.
- Generic plumbing (large, reusable): coding agent (`coding_agent.py` 273, `llm_withtools.py`
  445, `tools/`), SWE-bench harness (`swe_bench/`), Docker utils, prompts.

## Open decisions / blockers
1. OPENAI_API_KEY must be exported (env var, not CLI arg).
2. Env via `uv` (python 3.10) instead of conda — proceeding unless user objects.
3. Need to inspect actual `config.yaml` + how to restrict to a few SWE-bench Lite
   instances (subset selection) once repo is cloned — not yet confirmed from source.
