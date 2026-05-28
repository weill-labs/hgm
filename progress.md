# Progress — HGM reimplementation

## Status: Phase 1 ✓ and Phase 2 ✓ COMPLETE. 17/17 tests green ($0). Next: Phase 3 (real SWE-bench Lite eval).
## Issue tracking: beads (bd). hgm-tji ✓, hgm-6ne ✓, hgm-3vh (Phase 3) ready.
## Phase 1 result: CMP +0.055 mean skill vs greedy (40 seeds, wins 21/40); advantage is an
##   inverted-U in drift variance (peaks ~0.04) — floor/ceiling effects squeeze both ends.
## Phase 2: variant = directory snapshot (agent.py + config.yaml; surfaces = prompts+knobs+code).
##   Dynamic agent-class loader, run_variant bridge, MiniSelfImprover (agent edits its OWN files).
##   Integration test: unchanged HGMSearch drives real self-improvement -> on-disk variants.
##   SAFETY NOTE: live self-modification (real LLM + importing model-written agent.py) must run
##   in Docker in Phase 3 — never unsandboxed on host.

## Decisions (Phase 2)
- Mutation surface: prompts + config + CODE (Godel-machine spirit).
- Generation: agent edits its own config/source files (self-referential), via meta mini-swe-agent.

## Phase 3 (in progress) — Architecture D: all-API (mini-swe-agent + gpt-5-mini), codex dropped.
- Rationale: codex is a full agent (not a chat API), so it can't be a drop-in LLM for
  mini-swe-agent; D reuses ALL Phase 2 code. Needs OPENAI_API_KEY (live solve only).
- Built: spend.py (SpendGuard hard <$20 cap + per-run cost_limit), real_eval.py
  (SweBenchEvaluator: run variant agent in instance Docker -> score patch via OFFICIAL
  swebench harness). swebench 4.1.0 installed. 21/21 tests green ($0).
- $0 scoring validation ready: experiments/validate_scoring.py (gold patch -> resolved).
- Key: in project .env as OAI_KEY (mapped -> OPENAI_API_KEY); .env gitignored. Model = gpt-5.4
  (most capable that fits <$20; bump to gpt-5.4-pro for bigger runs). litellm verified ($0.0001).
- Scoring path VALIDATED $0: gold patch -> resolved=True via official swebench harness in Docker.
- Built real_backend.py (LitellmModel + Docker env factories), live_smoke.py (Stage A eval-only,
  $3/run + $20 global cap). 21/21 unit tests green.
- LIVE SMOKE PASSED: initial agent (gpt-5.4) RESOLVED astropy__astropy-12907 on SWE-bench Lite.
  resolved=True, spent=$0.0492. Full live path (LLM->patch->Docker tests->harness) works.
- Headroom now concrete: strong model solved instance-0 first try -> likely need to handicap the
  base scaffold so HGM has visible improvement room.
- Remaining: Stage B = sandbox the live self-improve meta-agent (Docker), then run the real HGM
  search loop on a handful of Lite instances under the $20 cap.

## Done
- Researched paper (arxiv 2510.21614) + official repo (./reference); wrote algorithm spec
  to findings.md.
- Decided: reimplement from scratch; coding agent = mini-swe-agent; validate vs simulated
  evaluator first; Python + uv (3.13).
- Built src/hgm/: tree.py (Node + CMP stub), bandit.py (Thompson Sampling), interfaces.py
  (Evaluator/SelfImprover protocols), search.py (expand/measure loop, use_cmp flag),
  simulated.py (hidden-skill landscape with inherited drift).
- Fixed a liveness bug: loop could spin forever when the root's early evals were all
  failures (mean_utility=0 -> never expandable, but always-expand chosen). Fix: seed root
  with init_measurements evals + each action returns progress flag + fallback/terminate.
- Tests: 5 passing (bandit x3, greedy-path harness x2). 4 failing ONLY on the CMP TODO.

## Next (the one user contribution)
- Implement `Node.descendant_evals` in src/hgm/tree.py (the CMP pooling). ~5 lines.
  Then: `uv run pytest` should go green, including test_cmp_beats_greedy (CMP > greedy).

## Then (Phase 2/3)
- Wrap mini-swe-agent as a real Evaluator + SelfImprover.
- Real SWE-bench Lite eval with a hard <$20 spend cap.

## Log
- 2026-05-28: Scaffold built & verified via greedy path; CMP left as guided TODO.
