# hgm — a clean-room Huxley-Gödel Machine

A from-scratch reimplementation of the **Huxley-Gödel Machine** (HGM,
[arXiv:2510.21614](https://arxiv.org/abs/2510.21614)), a self-improving coding agent that
searches a tree of agent variants guided by **Clade-based Metaproductivity (CMP)** and
**Thompson Sampling**.

- The official authors' code lives in `./reference/` and is used only as a spec.
- The coding agent is [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), not rebuilt.
- We validate the search algorithm against a **simulated evaluator** first (\$0), then layer
  real SWE-bench evaluation.

See `findings.md` (algorithm spec), `task_plan.md` (plan), and `progress.md` (status).

## Develop

```bash
uv sync            # install deps (numpy + pytest)
uv run pytest      # run the test suite
```
