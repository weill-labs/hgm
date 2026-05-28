"""Tiny LIVE smoke (costs money): evaluate the INITIAL agent on one SWE-bench Lite
instance with a real model, real Docker, real scoring, and a hard spend cap.

This is Stage A — eval only, no self-improvement — so all model-driven shell commands run
inside the instance's container (sandboxed), not on the host. It proves the most
expensive/uncertain path end to end: LLM -> patch -> Docker tests -> resolved 0/1.

    uv run python experiments/live_smoke.py [instance_id]

Defaults to the first Lite instance (already image-cached by validate_scoring.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SMOKE_CAP_USD = 20.0
MAX_RUN_COST_USD = 3.0  # keep one solve cheap for the smoke


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


def main() -> int:
    _load_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found (looked in OPENAI_API_KEY / OAI_KEY in .env).")
        return 2

    from hgm.real_backend import SOLVER_MODEL, make_docker_env, make_litellm_model
    from hgm.real_eval import SweBenchEvaluator, load_lite_instances
    from hgm.spend import SpendCapExceeded, SpendGuard
    from hgm.variant import VariantStore, initial_swebench_config

    instances = load_lite_instances(split="test")
    instance_id = sys.argv[1] if len(sys.argv) > 1 else sorted(instances)[0]

    store = VariantStore("output_hgm/smoke/variants")
    store.create_initial(initial_swebench_config())
    guard = SpendGuard(cap_usd=SMOKE_CAP_USD)

    evaluator = SweBenchEvaluator(
        store,
        instances,
        model_factory=lambda cost_limit: make_litellm_model(SOLVER_MODEL),
        env_factory=make_docker_env,
        spend_guard=guard,
        split="test",
        max_run_cost=MAX_RUN_COST_USD,
    )

    print(f"Model      : {SOLVER_MODEL}")
    print(f"Instance   : {instance_id}")
    print(
        f"Spend cap  : ${SMOKE_CAP_USD:.2f} (per-run ceiling ${MAX_RUN_COST_USD:.2f})"
    )
    print("Running the initial agent on the instance (live; this spends money)...\n")

    try:
        outcome = evaluator.evaluate("initial", instance_id)
    except SpendCapExceeded as e:
        print(f"ABORTED by spend cap: {e}")
        return 1

    print(f"\nresolved   = {bool(outcome)}  (outcome={outcome})")
    print(f"spent      = ${guard.spent:.4f}")
    print(
        "Live eval path works end to end."
        if guard.spent > 0
        else "No spend recorded (check model/cost tracking)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
