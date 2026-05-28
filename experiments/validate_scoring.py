"""$0 validation of the SWE-bench scoring path (no LLM).

Takes a real SWE-bench Lite instance's GOLD patch and runs it through our score_patch(),
which invokes the official swebench harness in Docker. A correct gold patch must score
"resolved" — proving our apply-patch + run-tests + parse-report pipeline works before we
spend a cent on the LLM solve step.

    uv run python experiments/validate_scoring.py            # first Lite instance
    uv run python experiments/validate_scoring.py <id>       # a specific instance

NOTE: pulls the instance's Docker image (multi-GB) and runs its test suite (minutes).
"""

from __future__ import annotations

import sys

from hgm.real_eval import load_lite_instances, score_patch, swebench_image

SPLIT = "test"


def main() -> int:
    instances = load_lite_instances(split=SPLIT)
    instance_id = sys.argv[1] if len(sys.argv) > 1 else sorted(instances)[0]
    instance = instances[instance_id]
    gold = instance["patch"]  # the reference solution

    print(f"Instance : {instance_id}")
    print(f"Image    : {swebench_image(instance)}")
    print(f"Gold patch: {len(gold)} chars, {gold.count(chr(10))} lines")
    print("Scoring gold patch via official harness (Docker; this is slow)...\n")

    resolved = score_patch(instance, gold, run_id="validate_gold", split=SPLIT)

    print(f"\nresolved = {resolved}")
    if resolved:
        print("PASS: gold patch scored RESOLVED — scoring pipeline is correct.")
        return 0
    print("FAIL: gold patch did NOT score resolved — investigate harness wiring/image.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
