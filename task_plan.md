# Task Plan — Reimplement HGM from scratch

Goal: a clean-room reimplementation of the Huxley-Gödel Machine, using the official repo
(./reference) only as a spec. Coding agent = mini-swe-agent (not rebuilt). Validate the
SEARCH ALGORITHM against a simulated (mock) evaluator first ($0), then layer real evals.

## Architecture (our package: src/hgm/)
- tree.py    — Node + CMP (clade-pooled descendant evals)
- bandit.py  — Beta-Bernoulli Thompson Sampling
- search.py  — expand/measure outer loop (the HGM driver)
- interfaces.py — Evaluator + SelfImprover protocols (pluggable)
- simulated.py — mock Evaluator + SelfImprover with a hidden skill landscape
- (later) real_eval.py — mini-swe-agent runner + SWE-bench scoring

## Phase 1 — Core search vs simulated evaluator ($0)  ← CURRENT MILESTONE
- [ ] uv project (Python 3.13), numpy + pytest.
- [ ] Node + CMP (get_descendant_evals): pseudo-count smoothing + descendant pooling.
- [ ] Thompson Sampling (TS_sample).
- [ ] expand/measure loop with budget + alpha expansion trigger.
- [ ] Simulated evaluator + self-improver (skill landscape; USER designs mutation model).
- [ ] Tests: tree/CMP correctness, TS sanity, and the headline experiment —
      CMP-guided search finds higher-skill agents than greedy under equal eval budget.

## Phase 2 — Real coding agent (mini-swe-agent)
- [ ] Add mini-swe-agent dependency; wrap DefaultAgent/LitellmModel/LocalEnvironment.
- [ ] Define "agent variant" = config.yaml + system prompt (+ optional code patch).
- [ ] Self-improve step: diagnose weakness → agent edits its own variant → child variant.

## Phase 3 — Real evaluation (SWE-bench Lite)
- [ ] Run a variant on 1 SWE-bench instance in a sandbox → resolved? 0/1.
- [ ] Hard spend cap (<$20 for first real run). Record actual cost.

## Phase 4 — Document deltas vs paper.

## Decision log
- 2026-05-28: Scope = reimplement from scratch; reuse mini-swe-agent as the coding agent.
- 2026-05-28: Validate search against simulated evaluator first ($0), then go real.
- 2026-05-28: Python + uv. Using Python 3.13 (locally installed, no download).
- 2026-05-28: "Agent variant" surface = mini-swe-agent YAML config + system prompt (small,
  clean, prompt-driven) — the self-improvement step rewrites this.
