"""Real SWE-bench Lite evaluation (Architecture D: mini-swe-agent + gpt-5-mini).

Evaluating a variant on an instance:
  1. Build the instance's pinned Docker environment (swebench/sweb.eval.x86_64.<id>).
  2. Run the variant's (possibly self-edited) agent in /testbed via a LitellmModel.
  3. Take the submitted patch and score it with the OFFICIAL swebench harness
     (apply patch -> run FAIL_TO_PASS/PASS_TO_PASS tests in Docker) -> resolved 0/1.

Scoring (step 3) needs no LLM, so it is validated for $0 with a gold patch. Only step 2
spends money, bounded by the SpendGuard (global cap) plus each run's native cost_limit.
"""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
from pathlib import Path

from .mini_runner import _agent_kwargs
from .spend import SpendGuard
from .variant import AgentVariant, VariantStore

DATASET_NAME = "princeton-nlp/SWE-bench_Lite"


def load_lite_instances(split: str = "test") -> dict[str, dict]:
    from datasets import load_dataset

    return {
        inst["instance_id"]: dict(inst)
        for inst in load_dataset(DATASET_NAME, split=split)
    }


def swebench_image(instance: dict) -> str:
    iid = instance["instance_id"].replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest".lower()


def score_patch(
    instance: dict,
    patch: str,
    *,
    run_id: str,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
    timeout: int = 1800,
) -> bool:
    """Score a candidate patch with the official swebench harness. Returns resolved?.

    Runs the harness in a subprocess (robust to harness-version API drift) over a single
    instance, then reads the generated report JSON. No LLM involved.
    """
    instance_id = instance["instance_id"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        preds_path = tmp_path / "preds.json"
        model_name = f"hgm__{run_id}"
        preds_path.write_text(
            json.dumps(
                {
                    instance_id: {
                        "instance_id": instance_id,
                        "model_name_or_path": model_name,
                        "model_patch": patch,
                    }
                }
            )
        )
        cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            split,
            "--predictions_path",
            str(preds_path),
            "--max_workers",
            "1",
            "--run_id",
            run_id,
            "--instance_ids",
            instance_id,
            "--cache_level",
            "env",
        ]
        subprocess.run(cmd, cwd=tmp_path, timeout=timeout, check=False)
        # Harness writes <model_name>.<run_id>.json into the working dir.
        report_path = tmp_path / f"{model_name}.{run_id}.json"
        if not report_path.exists():
            return False
        report = json.loads(report_path.read_text())
        return instance_id in set(report.get("resolved_ids", []))


class SweBenchEvaluator:
    """Implements the Evaluator protocol against real SWE-bench Lite instances.

    Args:
        store: VariantStore holding the file-based variants.
        instances: instance_id -> instance dict (from load_lite_instances).
        model_factory: (cost_limit: float) -> a mini-swe-agent Model (e.g. LitellmModel
            with model_name=gpt-5-mini). cost_limit is injected so the model/agent honours
            the per-run budget derived from the SpendGuard.
        spend_guard: global spend cap.
        split: dataset split for scoring.
        env_factory: (image: str, cwd: str) -> a mini-swe-agent Docker Environment.
    """

    def __init__(
        self,
        store: VariantStore,
        instances: dict[str, dict],
        *,
        model_factory,
        env_factory,
        spend_guard: SpendGuard,
        split: str = "test",
        max_run_cost: float | None = None,
    ) -> None:
        self.store = store
        self.instances = instances
        self.model_factory = model_factory
        self.env_factory = env_factory
        self.spend_guard = spend_guard
        self.split = split
        self.max_run_cost = max_run_cost  # tighter per-run ceiling (e.g. for a smoke)

    def evaluate(self, commit_id: str, instance_id) -> int:
        self.spend_guard.check()  # abort before spending if cap already hit
        instance = self.instances[instance_id]
        variant: AgentVariant = self.store.get(commit_id)
        config = variant.load_config()

        agent_kwargs = _agent_kwargs(config)
        # Per-run cost ceiling: never let one solve blow the remaining budget.
        cost_limit = self.spend_guard.per_run_cost_limit()
        if self.max_run_cost is not None:
            cost_limit = min(cost_limit, self.max_run_cost)
        agent_kwargs["cost_limit"] = cost_limit

        model = self.model_factory(agent_kwargs["cost_limit"])
        env = self.env_factory(swebench_image(instance), "/testbed")
        agent = variant.load_agent_class()(model, env, **agent_kwargs)

        try:
            result = agent.run(instance["problem_statement"])
        finally:
            self.spend_guard.add(float(getattr(agent, "cost", 0.0)))

        patch = (result.get("submission") or "").strip()
        if not patch:
            return 0
        run_id = f"{commit_id}_{instance_id}"
        return int(score_patch(instance, patch, run_id=run_id, split=self.split))
