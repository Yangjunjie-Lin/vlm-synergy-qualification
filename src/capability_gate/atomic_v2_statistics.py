"""Read-only, pre-outcome adjudication for the clean Atomic v2 protocol.

The historical scoring functions and Joint implementation are deliberately reused.
This module never writes historical results and never authorizes activation patching.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean, median
from typing import Any

import yaml

from capability_gate.paths import CONFIGS
from capability_gate.statistics.joint import analyze_joint_rows
from capability_gate.statistics.metrics import atomic_task_metrics, cohens_kappa

MODEL_KEYS = ("qwen2_5_vl_7b", "glm4_1v_9b")
CARDINAL = ("north", "south", "east", "west")
DIAGONAL = ("northeast", "northwest", "southeast", "southwest")
FAIL_LABELS = {
    "direct_visual_relation": "ATOMIC_VISUAL_FAIL",
    "direct_text_relation": "ATOMIC_TEXT_FAIL",
    "direction_reversal": "ATOMIC_DIRECTION_FAIL",
    "cross_modal_bridge_binding": "ATOMIC_BINDING_FAIL",
}
TASK_PRIORITY = tuple(FAIL_LABELS)
MODES = ("joint", "image_only", "text_only", "question_only")
CONDITIONS = ("I0T0", "I0T1", "I1T0", "I1T1")
CONTRACTS = {
    "cll": "top_answer",
    "constrained_generation": "constrained_generation_answer",
}


def _scoring() -> dict[str, Any]:
    scoring = yaml.safe_load((CONFIGS / "scoring.yaml").read_text(encoding="utf-8"))
    expected = {
        task: {
            "accuracy_min": 0.90 if task == "direct_text_relation" else 0.80,
            "lower_bound_min": 0.82 if task == "direct_text_relation" else 0.70,
        }
        for task in TASK_PRIORITY
    }
    if scoring["atomic_gates"] != expected or scoring["agreement"]["kappa_min"] != 0.80:
        raise RuntimeError("ATOMIC_V2_FROZEN_THRESHOLDS_CHANGED")
    if scoring["statistics"] != {
        "confidence": 0.95,
        "atomic_interval": "one_sided_clopper_pearson",
    }:
        raise RuntimeError("ATOMIC_V2_FROZEN_STATISTICS_CHANGED")
    return scoring


def _measurement_failure(errors: Sequence[str], stage: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "capability_gate_atomic_v2",
        "stage": stage,
        "status": "INVALID_OR_INCOMPLETE_MEASUREMENT",
        "decision": "MEASUREMENT_IMPLEMENTATION_NO_GO",
        "q1_potential": "NOT_EVALUATED_MEASUREMENT_FAILURE",
        "exact_next_action": "TERMINATE_AFTER_FINAL_MEASUREMENT_FAILURE",
        "scientific_capability_conclusion": False,
        "metrics_computed": False,
        "qualified_count": 0,
        "models": {},
        "validation_errors": list(errors),
        "joint_authorized": False,
        "activation_patching_executed": False,
        "activation_patching_authorized": False,
    }


def _score_errors(row: Mapping[str, Any], labels: Sequence[str], index: int) -> list[str]:
    errors = []
    for key in ("target", "top_answer", "constrained_generation_answer"):
        if row.get(key) not in labels:
            errors.append(f"row {index}: missing or invalid {key}")
    position = row.get("correct_option_position")
    if type(position) is not int or position not in range(4):
        errors.append(f"row {index}: invalid correct_option_position")
    scores = row.get("normalized_log_likelihood")
    if not isinstance(scores, Mapping) or set(scores) != set(labels):
        errors.append(f"row {index}: missing exact candidate likelihood set")
    elif any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in scores.values()
    ):
        errors.append(f"row {index}: non-finite candidate likelihood")
    margin = row.get("target_margin")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
    ):
        errors.append(f"row {index}: missing or non-finite target margin")
    elif not errors:
        target = row["target"]
        recomputed = scores[target] - mean(scores[label] for label in labels if label != target)
        if not math.isclose(margin, recomputed, rel_tol=1e-7, abs_tol=1e-7):
            errors.append(f"row {index}: target margin inconsistent with candidate likelihoods")
    return errors


def _atomic_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    if len(rows) != 512:
        errors.append(f"expected 512 Atomic rows, found {len(rows)}")
    if {row.get("model_key") for row in rows} != set(MODEL_KEYS):
        errors.append("expected exactly the frozen Qwen and GLM model keys")
    for index, row in enumerate(rows):
        errors.extend(_score_errors(row, CARDINAL, index))
        if not isinstance(row.get("scene_id"), str) or not row["scene_id"]:
            errors.append(f"row {index}: scene_id required")
        if row.get("task") not in TASK_PRIORITY:
            errors.append(f"row {index}: unknown Atomic task")
    signatures = {}
    for model_key in MODEL_KEYS:
        model_rows = [row for row in rows if row.get("model_key") == model_key]
        if len(model_rows) != 256:
            errors.append(f"{model_key}: expected 256 rows, found {len(model_rows)}")
        scene_ids = [row.get("scene_id") for row in model_rows]
        if len(set(scene_ids)) != len(scene_ids):
            errors.append(f"{model_key}: duplicate scene_id")
        counts = Counter(row.get("task") for row in model_rows)
        if counts != {task: 64 for task in TASK_PRIORITY}:
            errors.append(f"{model_key}: each of the four tasks must contain 64 rows")
        signatures[model_key] = {
            row.get("scene_id"): (
                row.get("task"), row.get("target"), row.get("correct_option_position")
            )
            for row in model_rows
        }
    if signatures[MODEL_KEYS[0]] != signatures[MODEL_KEYS[1]]:
        errors.append("the two model cohorts do not contain the identical scene/target/option set")
    return errors


def _summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "values": list(values),
    }


def _margins(rows: Sequence[Mapping[str, Any]], prediction_key: str) -> dict[str, Any]:
    selected = []
    top_gaps = []
    for row in rows:
        scores = row["normalized_log_likelihood"]
        answer = row[prediction_key]
        selected.append(scores[answer] - mean(v for k, v in scores.items() if k != answer))
        ranked = sorted(scores.values(), reverse=True)
        top_gaps.append(ranked[0] - ranked[1])
    return {
        "target_vs_mean_other": _summary([row["target_margin"] for row in rows]),
        "contract_selected_vs_mean_other": _summary(selected),
        "cll_top_vs_runner_up": _summary(top_gaps),
    }


def _per_answer(
    rows: Sequence[Mapping[str, Any]], prediction_key: str, labels: Sequence[str]
) -> dict[str, Any]:
    result = {}
    for label in labels:
        subset = [row for row in rows if row["target"] == label]
        correct = sum(row[prediction_key] == label for row in subset)
        result[label] = {
            "n": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else None,
        }
    return result


def _agreement(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    left = [row["top_answer"] for row in rows]
    right = [row["constrained_generation_answer"] for row in rows]
    kappa = cohens_kappa(left, right, labels)
    return {
        "exact": sum(a == b for a, b in zip(left, right)) / len(rows),
        "cohens_kappa": kappa,
        "undefined_degenerate_kappa": kappa is None,
        "constant_common_answer": (
            left[0] if len(set(left)) == 1 and left == right else None
        ),
    }


def adjudicate_atomic_rows_v2(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Adjudicate only a complete paired cohort; do not score partial outputs.

    There is no additional preregistered Atomic directional contrast beyond
    per-task contract-gate disagreement. Undefined kappa is not disagreement,
    but it cannot satisfy the frozen qualification kappa gate.
    """
    rows = list(predictions)
    errors = _atomic_errors(rows)
    if errors:
        return _measurement_failure(errors, "atomic")
    scoring = _scoring()
    models = {}
    for model_key in MODEL_KEYS:
        model_rows = [row for row in rows if row["model_key"] == model_key]
        tasks = {}
        failed_tasks = []
        conflicts = []
        for task in TASK_PRIORITY:
            task_rows = [row for row in model_rows if row["task"] == task]
            gate = scoring["atomic_gates"][task]
            result = {}
            passes = {}
            for contract, key in CONTRACTS.items():
                metrics = atomic_task_metrics(task_rows, CARDINAL, key)
                metrics["per_answer"] = _per_answer(task_rows, key, CARDINAL)
                metrics["per_answer_accuracy"] = {
                    label: values["accuracy"] for label, values in metrics["per_answer"].items()
                }
                metrics["candidate_margin"] = _margins(task_rows, key)
                passes[contract] = (
                    metrics["accuracy"] >= gate["accuracy_min"]
                    and metrics["one_sided_95_exact_lower"] >= gate["lower_bound_min"]
                )
                result[contract] = metrics
            conflict = passes["cll"] != passes["constrained_generation"]
            if not all(passes.values()):
                failed_tasks.append(task)
            if conflict:
                conflicts.append(task)
            tasks[task] = {
                **result,
                "n": len(task_rows),
                "cll_gate": passes["cll"],
                "generation_gate": passes["constrained_generation"],
                "agreement": _agreement(task_rows, CARDINAL),
                "contract_direction_conflict": conflict,
                "thresholds": gate,
            }
        agreement = _agreement(model_rows, CARDINAL)
        kappa = agreement["cohens_kappa"]
        low_defined_kappa = kappa is not None and kappa < scoring["agreement"]["kappa_min"]
        agreement["kappa_gate"] = kappa is not None and not low_defined_kappa
        agreement["contract_direction_conflict"] = bool(conflicts)
        agreement["kappa_gate_scope"] = "model_overall_not_per_task"
        if conflicts or low_defined_kappa:
            label = "MEASUREMENT_CONTRACT_DEPENDENT"
        elif not failed_tasks and agreement["kappa_gate"]:
            label = "ATOMICALLY_QUALIFIED"
        elif failed_tasks:
            label = FAIL_LABELS[failed_tasks[0]]
        else:
            # A balanced four-answer dataset cannot pass all gates constantly.
            # Fail safely for anomalous inputs instead of interpreting 0/0 as 1.
            return _measurement_failure(
                [f"{model_key}: undefined kappa despite all task gates passing"], "atomic"
            )
        models[model_key] = {
            "label": label,
            "n": len(model_rows),
            "tasks": tasks,
            "failed_tasks": failed_tasks,
            "failed_task_labels": [FAIL_LABELS[task] for task in failed_tasks],
            "task_gate_conflicts": conflicts,
            "defined_kappa_below_threshold": low_defined_kappa,
            "additional_preregistered_directional_conflict": False,
            "agreement": agreement,
        }
    count = sum(model["label"] == "ATOMICALLY_QUALIFIED" for model in models.values())
    dependent = any(
        model["label"] == "MEASUREMENT_CONTRACT_DEPENDENT" for model in models.values()
    )
    go = count == 2
    return {
        "schema_version": 1,
        "protocol": "capability_gate_atomic_v2",
        "stage": "atomic",
        "status": "COMPLETE",
        "decision": "ATOMIC_COHORT_GO" if go else (
            "MEASUREMENT_CONTRACT_NO_GO" if dependent else "CAPABILITY_COHORT_NO_GO"
        ),
        "q1_potential": None if go else (
            "NOT_EVALUATED_MEASUREMENT_FAILURE" if dependent else "NO_FEASIBLE_COHORT"
        ),
        "q1_evaluation_status": "PENDING_JOINT" if go else "FINAL",
        "exact_next_action": "RUN_FROZEN_JOINT_COMPOSITION_SCREEN" if go else (
            "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
        ),
        "scientific_capability_conclusion": not dependent,
        "metrics_computed": True,
        "qualified_count": count,
        "models": models,
        "joint_authorized": go,
        "thresholds": {
            "atomic_gates": scoring["atomic_gates"],
            "agreement": scoring["agreement"],
            "statistics": scoring["statistics"],
        },
        "activation_patching_executed": False,
        "activation_patching_authorized": False,
    }


def require_atomic_v2_go(result: Mapping[str, Any]) -> None:
    """Require the complete, exact two-family cohort before any Joint inference."""
    models = result.get("models", {})
    if not (
        result.get("status") == "COMPLETE"
        and result.get("decision") == "ATOMIC_COHORT_GO"
        and result.get("qualified_count") == 2
        and result.get("joint_authorized") is True
        and set(models) == set(MODEL_KEYS)
        and all(model.get("label") == "ATOMICALLY_QUALIFIED" for model in models.values())
    ):
        raise RuntimeError("JOINT_BLOCKED_BY_ATOMIC_V2_GATE: both frozen families must qualify")


def guard_activation_patching_v2(*_args: Any, **_kwargs: Any) -> None:
    """Patching is forbidden throughout this task, even after a Joint GO."""
    raise RuntimeError("ACTIVATION_PATCHING_FORBIDDEN_IN_ATOMIC_V2_TASK")


def _joint_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors = []
    if len(rows) != 4096:
        errors.append(f"expected 4096 Joint rows (2 x 128 x 4 x 4), found {len(rows)}")
    if {row.get("model_key") for row in rows} != set(MODEL_KEYS):
        errors.append("Joint requires exactly the frozen Qwen and GLM models")
    expected_cells = {(mode, condition) for mode in MODES for condition in CONDITIONS}
    quartet_sets = {}
    signatures = {}
    for index, row in enumerate(rows):
        errors.extend(_score_errors(row, DIAGONAL, index))
        if not isinstance(row.get("base_quartet_id"), str) or not row["base_quartet_id"]:
            errors.append(f"row {index}: base_quartet_id required")
        if row.get("psi_fixed_target") not in DIAGONAL:
            errors.append(f"row {index}: invalid fixed Psi target")
        if row.get("axis_map") not in {
            "horizontal_image_vertical_text", "vertical_image_horizontal_text"
        }:
            errors.append(f"row {index}: invalid axis_map")
        if row.get("template_id") not in range(4):
            errors.append(f"row {index}: invalid frozen template_id")
    for key in MODEL_KEYS:
        model_rows = [row for row in rows if row.get("model_key") == key]
        signatures[key] = {
            (row.get("base_quartet_id"), row.get("mode"), row.get("condition")): (
                row.get("target"), row.get("correct_option_position"),
                row.get("axis_map"), row.get("template_id"), row.get("psi_fixed_target"),
            )
            for row in model_rows
        }
        quartet_ids = {row.get("base_quartet_id") for row in model_rows}
        quartet_sets[key] = quartet_ids
        if len(model_rows) != 2048 or len(quartet_ids) != 128:
            errors.append(f"{key}: expected 128 complete quartets / 2048 rows")
        for quartet_id in quartet_ids:
            cells = [row for row in model_rows if row.get("base_quartet_id") == quartet_id]
            actual = {(row.get("mode"), row.get("condition")) for row in cells}
            if len(cells) != 16 or actual != expected_cells:
                errors.append(f"{key}/{quartet_id}: incomplete or duplicate quartet/mode cells")
            for condition in CONDITIONS:
                targets = {
                    (row.get("target"), row.get("correct_option_position"))
                    for row in cells if row.get("condition") == condition
                }
                if len(targets) != 1:
                    errors.append(f"{key}/{quartet_id}/{condition}: target/option differs by mode")
            if {row.get("target") for row in cells} != set(DIAGONAL):
                errors.append(f"{key}/{quartet_id}: quartet must cover all four answers")
            for field in ("axis_map", "template_id"):
                if len({row.get(field) for row in cells}) != 1:
                    errors.append(f"{key}/{quartet_id}: inconsistent {field}")
            fixed = {row.get("psi_fixed_target") for row in cells}
            fixed_expected = {
                row.get("target") for row in cells if row.get("condition") == "I1T1"
            }
            if len(fixed) != 1 or fixed != fixed_expected:
                errors.append(f"{key}/{quartet_id}: inconsistent I1T1 fixed Psi target")
    if quartet_sets[MODEL_KEYS[0]] != quartet_sets[MODEL_KEYS[1]]:
        errors.append("Joint model quartet sets differ")
    if signatures[MODEL_KEYS[0]] != signatures[MODEL_KEYS[1]]:
        errors.append("Joint paired model conditions/targets/options/metadata differ")
    return errors


def _joint_strata(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    result = {}
    for mode in MODES:
        subset = [row for row in rows if row["mode"] == mode]
        metrics = atomic_task_metrics(subset, DIAGONAL, key)
        metrics["per_answer"] = _per_answer(subset, key, DIAGONAL)
        metrics["candidate_margin"] = _margins(subset, key)
        metrics["by_template"] = {
            str(template): {
                "n": len(group),
                "accuracy": sum(row[key] == row["target"] for row in group) / len(group),
                "per_answer": _per_answer(group, key, DIAGONAL),
            }
            for template in sorted({row["template_id"] for row in subset})
            for group in ([row for row in subset if row["template_id"] == template],)
        }
        result[mode] = metrics
    return result


def analyze_joint_rows_v2(
    predictions: Sequence[Mapping[str, Any]],
    retention_predictions: Sequence[Mapping[str, Any]],
    atomic_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate complete inputs, then call the unchanged frozen Joint analysis."""
    require_atomic_v2_go(atomic_result)
    rows = list(predictions)
    retention_rows = list(retention_predictions)
    errors = _joint_errors(rows) + _atomic_errors(retention_rows)
    if errors:
        return _measurement_failure(errors, "joint")
    result = analyze_joint_rows(rows, retention_rows)
    original_decision = result["decision"]
    retention = adjudicate_atomic_rows_v2(retention_rows)
    directional = any(
        model["diagnostics"]["directional_measurement_conflict"]
        for model in result["models"].values()
    )
    contract_dependent = (
        directional
        or original_decision == "MEASUREMENT_CONTRACT_NO_GO"
        or retention["decision"] == "MEASUREMENT_CONTRACT_NO_GO"
    )
    if contract_dependent:
        decision = "MEASUREMENT_CONTRACT_NO_GO"
        q1 = "NOT_EVALUATED_MEASUREMENT_FAILURE"
    elif original_decision == "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION":
        decision = original_decision
        q1 = "MECHANISTIC_STUDY_FEASIBLE"
    elif original_decision == "BLOCKED_BY_MODEL_ADAPTER":
        decision = "MEASUREMENT_IMPLEMENTATION_NO_GO"
        q1 = "NOT_EVALUATED_MEASUREMENT_FAILURE"
    else:
        decision = "JOINT_COMPOSITION_NO_GO"
        q1 = "NO_MECHANISTIC_STUDY_BASIS"
    for model_key, model in result["models"].items():
        model_rows = [row for row in rows if row["model_key"] == model_key]
        model["complete_metrics"] = {
            contract: _joint_strata(model_rows, key) for contract, key in CONTRACTS.items()
        }
        model["agreement_by_mode"] = {
            mode: _agreement([row for row in model_rows if row["mode"] == mode], DIAGONAL)
            for mode in MODES
        }
    go = decision == "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION"
    return {
        **result,
        "schema_version": 1,
        "protocol": "capability_gate_atomic_v2",
        "stage": "joint",
        "status": "COMPLETE",
        "frozen_joint_analysis_decision": original_decision,
        "decision": decision,
        "q1_potential": q1,
        "exact_next_action": "BUILD_HELD_OUT_MECHANISTIC_PILOT" if go else (
            "TERMINATE_AFTER_FINAL_MEASUREMENT_FAILURE"
            if decision == "MEASUREMENT_IMPLEMENTATION_NO_GO"
            else "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
        ),
        "atomic_retention_adjudication": retention,
        "scientific_capability_conclusion": True,
        "joint_composition_conclusion_admissible": not contract_dependent,
        "metrics_computed": True,
        "activation_patching_executed": False,
        "activation_patching_authorized": False,
        "new_mechanistic_preregistration_allowed": go,
    }
