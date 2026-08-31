from __future__ import annotations

from collections import defaultdict
from typing import Any

import yaml

from capability_gate.artifacts import read_jsonl, write_json
from capability_gate.paths import ARTIFACTS, CONFIGS
from capability_gate.statistics.metrics import atomic_task_metrics, cohens_kappa

CARDINAL = ("north", "south", "east", "west")
FAIL_LABELS = {
    "direct_visual_relation": "ATOMIC_VISUAL_FAIL",
    "direct_text_relation": "ATOMIC_TEXT_FAIL",
    "direction_reversal": "ATOMIC_DIRECTION_FAIL",
    "cross_modal_bridge_binding": "ATOMIC_BINDING_FAIL",
}


def _load_scoring() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / "scoring.yaml").read_text(encoding="utf-8"))


def adjudicate_atomic() -> dict[str, Any]:
    scoring = _load_scoring()
    prediction_path = ARTIFACTS / "atomic" / "predictions.jsonl"
    if not prediction_path.exists():
        blocked = ARTIFACTS / "manifests" / "engineering_smoke.json"
        state = {
            "decision": "BLOCKED_BY_MODEL_ADAPTER",
            "qualified_count": 0,
            "models": {},
            "reason": "atomic predictions absent",
        }
        if blocked.exists():
            import json

            smoke = json.loads(blocked.read_text(encoding="utf-8"))
            state["decision"] = smoke.get("blocking_decision", state["decision"])
            state["reason"] = smoke.get("reason", state["reason"])
        write_json(ARTIFACTS / "atomic" / "adjudication.json", state)
        return state

    rows = read_jsonl(prediction_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_key"]].append(row)
    models = {}
    cll_qualified_count = 0
    generation_qualified_count = 0
    for model_key, model_rows in grouped.items():
        tasks = {}
        failed_tasks = []
        cll_all_pass = True
        generation_all_pass = True
        for task, gate in scoring["atomic_gates"].items():
            task_rows = [row for row in model_rows if row["task"] == task]
            cll = atomic_task_metrics(task_rows, CARDINAL, "top_answer")
            generated = atomic_task_metrics(task_rows, CARDINAL, "constrained_generation_answer")
            cll_pass = (
                cll["accuracy"] >= gate["accuracy_min"]
                and cll["one_sided_95_exact_lower"] >= gate["lower_bound_min"]
            )
            generation_pass = (
                generated["accuracy"] >= gate["accuracy_min"]
                and generated["one_sided_95_exact_lower"] >= gate["lower_bound_min"]
            )
            cll_all_pass &= cll_pass
            generation_all_pass &= generation_pass
            if not cll_pass:
                failed_tasks.append(task)
            task_cll = [row["top_answer"] for row in task_rows]
            task_generated = [row["constrained_generation_answer"] for row in task_rows]
            task_kappa = cohens_kappa(task_cll, task_generated, CARDINAL)
            tasks[task] = {
                "cll": cll,
                "constrained_generation": generated,
                "cll_gate": cll_pass,
                "generation_gate": generation_pass,
                "agreement": {
                    "exact": sum(a == b for a, b in zip(task_cll, task_generated)) / len(task_rows),
                    "cohens_kappa": task_kappa,
                },
            }
        cll_answers = [row["top_answer"] for row in model_rows]
        generated_answers = [row["constrained_generation_answer"] for row in model_rows]
        exact = sum(a == b for a, b in zip(cll_answers, generated_answers)) / len(model_rows)
        kappa = cohens_kappa(cll_answers, generated_answers, CARDINAL)
        cll_qualified_count += int(cll_all_pass)
        generation_qualified_count += int(generation_all_pass)
        task_verdict_conflict = any(
            value["cll_gate"] != value["generation_gate"] for value in tasks.values()
        )
        contract_conflict = cll_all_pass != generation_all_pass or task_verdict_conflict
        kappa_gate = kappa is not None and kappa >= scoring["agreement"]["kappa_min"]
        if contract_conflict or not kappa_gate:
            label = "MEASUREMENT_CONTRACT_DEPENDENT"
        elif cll_all_pass and generation_all_pass:
            label = "ATOMICALLY_QUALIFIED"
        else:
            label = FAIL_LABELS[failed_tasks[0]]
        models[model_key] = {
            "label": label,
            "tasks": tasks,
            "agreement": {
                "exact": exact,
                "cohens_kappa": kappa,
                "kappa_gate": kappa_gate,
                "contract_pass_direction_conflict": contract_conflict,
                "undefined_degenerate_kappa": kappa is None,
            },
            "failed_tasks": failed_tasks,
        }
    qualified_count = sum(model["label"] == "ATOMICALLY_QUALIFIED" for model in models.values())
    if qualified_count >= 2:
        decision = "ATOMIC_COHORT_GO"
    elif max(cll_qualified_count, generation_qualified_count) >= 2:
        decision = "MEASUREMENT_CONTRACT_NO_GO"
    else:
        decision = "CAPABILITY_COHORT_NO_GO"
    result = {
        "decision": decision,
        "qualified_count": qualified_count,
        "cll_qualified_count": cll_qualified_count,
        "constrained_generation_qualified_count": generation_qualified_count,
        "models": models,
    }
    write_json(ARTIFACTS / "atomic" / "adjudication.json", result)
    return result
