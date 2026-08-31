from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import yaml

from capability_gate.artifacts import read_jsonl, write_json
from capability_gate.paths import ARTIFACTS, CONFIGS, REPORTS
from capability_gate.recovery.governance import engineering_cohort_decision
from capability_gate.statistics.joint import analyze_joint_rows
from capability_gate.statistics.metrics import atomic_task_metrics, cohens_kappa

CARDINAL = ("north", "south", "east", "west")
FAIL_LABELS = {
    "direct_visual_relation": "ATOMIC_VISUAL_FAIL",
    "direct_text_relation": "ATOMIC_TEXT_FAIL",
    "direction_reversal": "ATOMIC_DIRECTION_FAIL",
    "cross_modal_bridge_binding": "ATOMIC_BINDING_FAIL",
}


def adjudicate_engineering_recovery() -> dict[str, Any]:
    path = ARTIFACTS / "engineering_recovery/manifests/engineering_recovery_smoke.json"
    smoke = json.loads(path.read_text(encoding="utf-8"))
    statuses = {key: value["status"] for key, value in smoke["model_results"].items()}
    cohort = engineering_cohort_decision(statuses)
    result = {
        "schema_version": 1,
        **cohort,
        "model_statuses": statuses,
        "formal_atomic_outputs_at_adjudication": 0,
        "formal_joint_outputs_at_adjudication": 0,
        "scientific_capability_conclusion": False,
        "activation_patching_executed": False,
        "third_model_disposition": (
            {key: "NOT_EVALUABLE_BY_ENGINEERING_BLOCK" for key in cohort["blocked_models"]}
            if cohort["passed_count"] == 2
            else {}
        ),
    }
    artifact_path = ARTIFACTS / "engineering_recovery/manifests/engineering_cohort_decision.json"
    write_json(artifact_path, result)
    report_path = REPORTS / "recovery/engineering_cohort_decision.yaml"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_adapter_report(smoke, result)
    return result


def _write_adapter_report(smoke: dict[str, Any], cohort: dict[str, Any]) -> None:
    lines = [
        "# Adapter Recovery Report",
        "",
        (
            "No engineering-scene task accuracy is reported. These results establish transport and "
            "measurement implementation only; they are not model-capability evidence."
        ),
        "",
        "| Model | Status | Native class | Processor | Meta params/buffers | Device map | Visual | Text | CLL | Constrained | Determinism | Peak VRAM | Runtime |",
        "|---|---|---|---|---|---|---|---|---|---|---|---:|---:|",
    ]
    for model_key, result in smoke["model_results"].items():
        metadata = result.get("model_metadata", {})
        material = metadata.get("materialization", {})
        gates = result.get("gates", {})
        lines.append(
            f"| {model_key} | {result['status']} | {metadata.get('model_class', 'NA')} | "
            f"{metadata.get('processor_class', 'NA')} | "
            f"{material.get('meta_parameter_count', 'NA')}/{material.get('meta_buffer_count', 'NA')} | "
            f"`{metadata.get('resolved_device_map', 'NA')}` | "
            f"{gates.get('real_visual_forward', False)} | {gates.get('text_only_forward', False)} | "
            f"{gates.get('cll_finite', False)} | "
            f"{gates.get('constrained_generation_allowed', False)} | "
            f"{result.get('deterministic_rerun_agreement', 0.0):.2f} | "
            f"{result.get('peak_vram_bytes', 0)} | {result.get('runtime_seconds', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Engineering cohort: **{cohort['decision']}**; passed families: "
                f"**{cohort['passed_count']} / 3**."
            ),
            "",
            "Activation patching was not run. No fourth model or replacement checkpoint was used.",
        ]
    )
    path = REPORTS / "recovery/adapter_recovery_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def adjudicate_atomic_v2() -> dict[str, Any]:
    root = ARTIFACTS / "recovery_qualification/atomic"
    run_path = root / "run_status.json"
    status = json.loads(run_path.read_text(encoding="utf-8"))
    prediction_paths = sorted((root / "predictions").glob("*.jsonl"))
    if status["status"] != "complete":
        result = {
            "decision": status.get("upstream_decision", "ENGINEERING_COHORT_NO_GO"),
            "qualified_count": 0,
            "models": {},
            "status": "NOT_RUN_BY_ENGINEERING_GATE",
        }
        write_json(root / "adjudication.json", result)
        _write_atomic_report(result)
        return result
    rows = [row for path in prediction_paths for row in read_jsonl(path)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_key"]].append(row)
    scoring = yaml.safe_load((CONFIGS / "scoring.yaml").read_text(encoding="utf-8"))
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
            left = [row["top_answer"] for row in task_rows]
            right = [row["constrained_generation_answer"] for row in task_rows]
            tasks[task] = {
                "cll": cll,
                "constrained_generation": generated,
                "cll_gate": cll_pass,
                "generation_gate": generation_pass,
                "agreement": {
                    "exact": sum(a == b for a, b in zip(left, right)) / len(left),
                    "cohens_kappa": cohens_kappa(left, right, CARDINAL),
                },
            }
        left = [row["top_answer"] for row in model_rows]
        right = [row["constrained_generation_answer"] for row in model_rows]
        exact = sum(a == b for a, b in zip(left, right)) / len(left)
        kappa = cohens_kappa(left, right, CARDINAL)
        cll_qualified_count += int(cll_all_pass)
        generation_qualified_count += int(generation_all_pass)
        contract_conflict = cll_all_pass != generation_all_pass or any(
            value["cll_gate"] != value["generation_gate"] for value in tasks.values()
        )
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
            },
            "failed_tasks": failed_tasks,
        }
    qualified_count = sum(value["label"] == "ATOMICALLY_QUALIFIED" for value in models.values())
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
    write_json(root / "adjudication.json", result)
    _write_atomic_report(result)
    return result


def _write_atomic_report(result: dict[str, Any]) -> None:
    lines = ["# Atomic Qualification v2", "", f"Decision: **{result['decision']}**", ""]
    if not result.get("models"):
        lines.extend(["**NOT_RUN_BY_ENGINEERING_GATE**", ""])
    for model_key, model in result.get("models", {}).items():
        lines.extend(
            [
                f"## {model_key}",
                "",
                f"Qualification label: `{model['label']}`",
                "",
                "| Task | Contract | n | Accuracy | One-sided 95% exact lower | Gate |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for task, metrics in model["tasks"].items():
            for contract, gate_key in (
                ("cll", "cll_gate"),
                ("constrained_generation", "generation_gate"),
            ):
                values = metrics[contract]
                lines.append(
                    f"| {task} | {contract} | {values['n']} | {values['accuracy']:.4f} | "
                    f"{values['one_sided_95_exact_lower']:.4f} | {metrics[gate_key]} |"
                )
        lines.extend(
            [
                "",
                (
                    f"Overall exact agreement: {model['agreement']['exact']:.4f}; "
                    f"Cohen's kappa: {model['agreement']['cohens_kappa']}."
                ),
                "",
            ]
        )
    path = REPORTS / "recovery/atomic_qualification_v2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_joint_v2() -> dict[str, Any]:
    root = ARTIFACTS / "recovery_qualification/joint"
    status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        result = {
            "decision": status.get("upstream_decision", "CAPABILITY_COHORT_NO_GO"),
            "status": "JOINT_NOT_RUN_BY_UPSTREAM_GATE",
            "models": {},
            "exact_next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
        }
    else:
        predictions = [
            row
            for path in sorted((root / "predictions").glob("*.jsonl"))
            for row in read_jsonl(path)
        ]
        retention = [
            row for path in sorted((root / "retention").glob("*.jsonl")) for row in read_jsonl(path)
        ]
        result = analyze_joint_rows(predictions, retention)
    write_json(root / "analysis.json", result)
    _write_joint_report(result)
    return result


def _write_joint_report(result: dict[str, Any]) -> None:
    lines = [
        "# Joint Composition Screen v2",
        "",
        f"Decision: **{result['decision']}**",
        "",
    ]
    if not result.get("models"):
        lines.extend(
            [
                "**NOT_RUN_BY_UPSTREAM_GATE**",
                "",
                f"Upstream status: `{result.get('status', result['decision'])}`.",
                "",
            ]
        )
    for model_key, model in result.get("models", {}).items():
        lines.extend(
            [
                f"## {model_key}",
                "",
                "| Contract | Joint | Image only | Text only | Question only | Advantage |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for contract in ("cll", "generation"):
            accuracy = model["accuracy"][contract]
            lines.append(
                f"| {contract} | {accuracy['joint']:.4f} | {accuracy['image_only']:.4f} | "
                f"{accuracy['text_only']:.4f} | {accuracy['question_only']:.4f} | "
                f"{model['joint_advantage'][contract]:.4f} |"
            )
        psi = model["psi"]
        lines.extend(
            [
                "",
                (
                    f"Ψ mean={psi['mean']:.6f}, 95% quartet-bootstrap CI "
                    f"[{psi['lower']:.6f}, {psi['upper']:.6f}]; atomic retention="
                    f"{model['retention']['retained']}; CLL/generation agreement="
                    f"{model['measurement_agreement']['exact']:.4f}."
                ),
                "",
            ]
        )
    path = REPORTS / "recovery/joint_composition_screen_v2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
