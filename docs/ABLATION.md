# Proposed ablation: does CMP add value beyond compute-matched prompt optimization?

## The question this answers

Our live results showed HGM **rescues a floored agent** but, at n=50, did **not improve a
competent one** — and every gain it found was "raise the step limit, write a longer prompt."
That raises the real worry: **is HGM doing anything beyond spending compute to let a strong
LLM tune its own scaffold?** This experiment is designed to isolate that.

It has one job: hold *compute* and *model* constant across conditions, vary only the
**selection strategy**, and see whether HGM's tree + CMP beats the dumb controls on
**held-out, matched, properly-powered** evaluation.

## Arms (identical budget; differ ONLY in how self-improvements are selected)

| Arm | Self-improve selection | Isolates |
|---|---|---|
| **A. HGM (CMP)** | tree search, expand by *clade* Thompson sampling (`use_cmp=True`) | the full method |
| **B. Greedy / DGM** | tree search, expand by *own-mean* Thompson sampling (`use_cmp=False`) | **CMP vs own-fitness** credit |
| **C. Flat best-of-N** | no tree: N independent depth-1 self-improvements of the base; keep the best on the search set | **search/lineage vs "sample N edits and pick best"** |
| **D. Base (no improvement)** | none | the floor everyone must beat |

Arm **C is the key control.** If A ≈ C, HGM is just expensive prompt optimization — the
tree and CMP add nothing over "generate N scaffold edits and keep the best." If A > C, the
*search structure* genuinely matters. If A ≈ B, CMP specifically adds nothing (the tree
might still help, but clade-credit doesn't).

## Compute matching (the crux — without this the comparison is meaningless)

Every arm gets the **same** total self-improve steps `S`, the same in-search eval calls
`E`, and the same `K` seeds. The **meta (self-improve) model is held constant** across all
arms — only the *selection* differs. Arm C spends its `S` edits all at depth 1 and uses `E`
evals to rank them; A and B spend `S`/`E` via the search loop. Log actual LLM-call and
token counts per arm and assert they match within ~10%.

## Evaluation (fix the mistakes we already made)

- **Held-out + matched**: all arms' best agents *and* the base scored on the **same**
  disjoint instance set. (We learned this the hard way.)
- **Powered**: n ≥ 50 held-out instances. (n=20 gave a false positive that vanished at 50.)
- **Multi-seed**: K ≥ 3 (ideally 5) search seeds; report per-seed and pooled.
- **Stats**: per-arm pass@1 + Wilson CI; exact McNemar for A-vs-base, A-vs-C, A-vs-B.

## Pre-registered decision rules

- **A > base AND A > C (significant)** → the search adds real value beyond sampling. HGM works.
- **A ≈ C** → HGM is compute-matched prompt optimization in a trench coat.
- **A ≈ B** → CMP (clade credit) adds nothing over greedy self-fitness.
- **No arm > base** → at this scale/model, self-improvement doesn't generalize.

State the outcome whichever way it lands; a null for A-vs-C is the most important thing this
experiment can produce.

## Use a weaker (mini) model — yes, and it makes the test *better*

Switching the **solver** from gpt-5.4 to a mid-tier mini (e.g. `gpt-5-mini`) is not just a
cost cut; it removes confounds:

1. **It dissolves the "we crippled it then un-crippled it" critique.** Run the mini with its
   *real, un-handicapped* config (full step_limit, normal prompt). If it sits at a genuine
   mid pass-rate (~15–35% on the hard repos), self-improvement has honest headroom that
   *isn't* just "give yourself more steps back."
2. **It puts us in the regime where self-improvement should matter.** gpt-5.4 was so strong
   that scaffold edits were lateral (the n=50 null). A weaker model is not near-ceiling, so
   a real method has room to show a generalizing gain — and if it *still* can't, that's a
   strong result.
3. **It makes the full matrix affordable.** ~10–30× cheaper per solve, so 4 arms × ~5 seeds
   × (search + 50 held-out matched evals) lands around **$10–25** instead of $100+.
4. **It separates model strength from method.** "HGM helps a weak model but not a strong
   one" would itself be a clean, publishable finding.

Keep the **meta/self-improve model capable** (e.g. gpt-5.4 or gpt-5-mini) and *constant
across arms* — give HGM its best shot at writing good edits; the controls get the same
advantage, so it stays fair.

### Step 0 (cheap, do first)

Baseline the chosen mini's **un-handicapped** pass@1 on the hard-repo held-out set. Proceed
only if it's mid-range (not ~0 → back to the rescue regime; not ~0.5 → near-ceiling like
gpt-5.4). If it floors/ceilings, pick a different mini or instance mix before spending on
the full matrix.

## Rough budget / time

`gpt-5-mini` solver, K=5 seeds, n=50 held-out, S≈15, E≈30: ~$10–25, a few hours
(parallelized). The expensive piece is the matched eval ((1 base + 3 arms) × 50 × 5 seeds),
which is embarrassingly parallel.

## What this does NOT settle

Even a clean A>C>base result here is *our* small-scale stack (mini model, ≤15 evals/seed,
mini-swe-agent), not the paper's. It tells us whether CMP-guided search beats dumb controls
**in a fair, powered fight on a non-crippled mid model** — which is the honest question we
can actually afford to answer. The paper's human-level claim still needs the paper's scale.
