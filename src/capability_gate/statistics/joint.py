from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import yaml

from capability_gate.artifacts import read_jsonl, write_json
from capability_gate.models.registry import load_model_specs
from capability_gate.paths import ARTIFACTS, CONFIGS
from capability_gate.statistics.metrics import (
    atomic_task_metrics,
    bootstrap_mean_ci,
    cohens_kappa,
    option_position_dependence,
)

CARDINAL = ("north", "south", "east", "west")
DIAGONAL = ("northeast", "northwest", "southeast", "southwest")


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def _accuracy(rows: Sequence[dict[str, Any]], key: str) -> float:
    return sum(row[key] == row["target"] for row in rows) / len(rows)


def _advantage(rows: Sequence[dict[str, Any]], key: str) -> tuple[float, dict[str, float]]:
    accuracies = {
        mode: _accuracy([row for row in rows if row["mode"] == mode], key)
        for mode in ("joint", "image_only", "text_only", "question_only")
    }
    return accuracies["joint"] - max(accuracies["image_only"], accuracies["text_only"]), accuracies


def _fixed_margin(row: dict[str, Any]) -> float:
    scores = row["normalized_log_likelihood"]
    target = row["psi_fixed_target"]
    return scores[target] - sum(value for key, value in scores.items() if key != target) / 3.0


def _psi(rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    joint = [row for row in rows if row["mode"] == "joint"]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in joint:
        grouped[row["base_quartet_id"]][row["condition"]] = row
    values = []
    for base_id, cells in grouped.items():
        if set(cells) != {"I0T0", "I0T1", "I1T0", "I1T1"}:
            raise RuntimeError(f"incomplete quartet {base_id}")
        margins = {condition: _fixed_margin(row) for condition, row in cells.items()}
        values.append(margins["I1T1"] - margins["I1T0"] - margins["I0T1"] + margins["I0T0"])
    ci = bootstrap_mean_ci(
        values,
        resamples=config["construction"]["bootstrap_resamples"],
        seed=config["construction"]["bootstrap_seed"],
    )
    return {**ci, "quartets": len(values), "quartet_values": values}


def _components(answer: str) -> tuple[str, str]:
    vertical = "north" if answer.startswith("north") else "south"
    horizontal = "east" if answer.endswith("east") else "west"
    return horizontal, vertical


def _factor_accuracy(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    image_hits = []
    text_hits = []
    for row in rows:
        if row["mode"] != "joint":
            continue
        pred_h, pred_v = _components(row[key])
        target_h, target_v = _components(row["target"])
        if row["axis_map"] == "horizontal_image_vertical_text":
            image_hits.append(pred_h == target_h)
            text_hits.append(pred_v == target_v)
        else:
            image_hits.append(pred_v == target_v)
            text_hits.append(pred_h == target_h)
    image_accuracy = sum(image_hits) / len(image_hits)
    text_accuracy = sum(text_hits) / len(text_hits)
    return {
        "image_factor_accuracy": image_accuracy,
        "text_factor_accuracy": text_accuracy,
        "macro_factor_accuracy": (image_accuracy + text_accuracy) / 2.0,
    }


def _retention(rows: Sequence[dict[str, Any]], scoring: dict[str, Any]) -> dict[str, Any]:
    tasks = {}
    cll_all = True
    generation_all = True
    conflict = False
    for task, gate in scoring["atomic_gates"].items():
        task_rows = [row for row in rows if row["task"] == task]
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
        cll_all &= cll_pass
        generation_all &= generation_pass
        conflict |= cll_pass != generation_pass
        tasks[task] = {
            "cll": cll,
            "constrained_generation": generated,
            "cll_gate": cll_pass,
            "generation_gate": generation_pass,
        }
    left = [row["top_answer"] for row in rows]
    right = [row["constrained_generation_answer"] for row in rows]
    kappa = cohens_kappa(left, right, CARDINAL)
    robust = cll_all and generation_all and not conflict and kappa is not None and kappa >= 0.80
    return {"retained": robust, "cohens_kappa": kappa, "verdict_conflict": conflict, "tasks": tasks}


def _stratum_advantages(rows: Sequence[dict[str, Any]], key: str, field: str) -> dict[str, float]:
    result = {}
    values = sorted({str(row[field]) for row in rows if row["mode"] == "joint"})
    for value in values:
        subset = [row for row in rows if str(row[field]) == value]
        result[value] = _advantage(subset, key)[0]
    return result


def _diagnostics(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    joint = [row for row in rows if row["mode"] == "joint"]
    dependence = option_position_dependence(
        [row["correct_option_position"] for row in joint],
        [row[key] == row["target"] for row in joint],
    )
    severe_option = dependence["accuracy_range"] > 0.25 or (
        dependence["p_value"] < 0.01 and dependence["cramers_v"] > 0.20
    )
    class_advantage = _stratum_advantages(rows, key, "target")
    template_advantage = _stratum_advantages(rows, key, "template_id")

    def driven(advantages: dict[str, float], field: str) -> bool:
        positives = [value for value, advantage in advantages.items() if advantage > 0]
        if len(positives) != 1:
            return False
        remaining = [row for row in rows if str(row[field]) != positives[0]]
        return bool(remaining) and _advantage(remaining, key)[0] <= 0

    return {
        "option_position": dependence,
        "severe_option_position_bias": severe_option,
        "advantage_by_answer_class": class_advantage,
        "single_answer_class_driven": driven(class_advantage, "target"),
        "advantage_by_template": template_advantage,
        "single_template_driven": driven(template_advantage, "template_id"),
    }


def analyze_joint_rows(
    predictions: Sequence[dict[str, Any]],
    retention_predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    config = _load("joint_screen.yaml")
    scoring = _load("scoring.yaml")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retention_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["model_key"]].append(row)
    for row in retention_predictions:
        retention_grouped[row["model_key"]].append(row)

    models = {}
    pair_sets = {"cll": set(), "generation": set()}
    for model_key, rows in grouped.items():
        cll_advantage, cll_accuracy = _advantage(rows, "top_answer")
        generation_advantage, generation_accuracy = _advantage(
            rows, "constrained_generation_answer"
        )
        psi = _psi(rows, config)
        retention = _retention(retention_grouped[model_key], scoring)
        cll_diagnostics = _diagnostics(rows, "top_answer")
        generation_diagnostics = _diagnostics(rows, "constrained_generation_answer")
        cll_factor = _factor_accuracy(rows, "top_answer")
        generation_factor = _factor_accuracy(rows, "constrained_generation_answer")
        cll_answers = [row["top_answer"] for row in rows if row["mode"] == "joint"]
        generated_answers = [
            row["constrained_generation_answer"] for row in rows if row["mode"] == "joint"
        ]
        agreement_kappa = cohens_kappa(cll_answers, generated_answers, DIAGONAL)
        directional_conflict = (
            (cll_advantage > 0) != (generation_advantage > 0)
            or (cll_factor["image_factor_accuracy"] >= 0.5)
            != (generation_factor["image_factor_accuracy"] >= 0.5)
            or (cll_factor["text_factor_accuracy"] >= 0.5)
            != (generation_factor["text_factor_accuracy"] >= 0.5)
        )
        bias_cll = (
            any(
                cll_diagnostics[key]
                for key in (
                    "severe_option_position_bias",
                    "single_answer_class_driven",
                    "single_template_driven",
                )
            )
            or directional_conflict
        )
        bias_generation = (
            any(
                generation_diagnostics[key]
                for key in (
                    "severe_option_position_bias",
                    "single_answer_class_driven",
                    "single_template_driven",
                )
            )
            or directional_conflict
        )

        role = {}
        for contract, accuracy, advantage, biased in (
            ("cll", cll_accuracy, cll_advantage, bias_cll),
            ("generation", generation_accuracy, generation_advantage, bias_generation),
        ):
            strong = (
                accuracy["joint"] >= 0.50
                and advantage >= 0.10
                and psi["mean"] > 0
                and psi["lower"] > 0
                and retention["retained"]
                and not biased
            )
            support = (
                accuracy["joint"] >= 0.40
                and advantage > 0
                and psi["mean"] >= 0
                and retention["retained"]
                and not biased
            )
            role[contract] = {"strong": strong, "support": support}
        models[model_key] = {
            "accuracy": {"cll": cll_accuracy, "generation": generation_accuracy},
            "joint_advantage": {"cll": cll_advantage, "generation": generation_advantage},
            "psi": psi,
            "factor_bit_accuracy": {"cll": cll_factor, "generation": generation_factor},
            "retention": retention,
            "diagnostics": {
                "cll": cll_diagnostics,
                "generation": generation_diagnostics,
                "directional_measurement_conflict": directional_conflict,
            },
            "measurement_agreement": {
                "exact": sum(a == b for a, b in zip(cll_answers, generated_answers))
                / len(cll_answers),
                "cohens_kappa": agreement_kappa,
            },
            "role": role,
        }

    keys = sorted(models)
    for contract, pairs in pair_sets.items():
        for discovery in keys:
            for replication in keys:
                if discovery == replication:
                    continue
                if (
                    models[discovery]["role"][contract]["strong"]
                    and models[replication]["role"][contract]["support"]
                ):
                    pairs.add((discovery, replication))
    shared_pairs = pair_sets["cll"] & pair_sets["generation"]
    any_pairs = pair_sets["cll"] | pair_sets["generation"]
    selected = None
    if shared_pairs:
        specs = {spec["key"]: spec for spec in load_model_specs()}
        eligible_discovery = sorted(
            {pair[0] for pair in shared_pairs if specs[pair[0]].get("complete_activation_hooks")},
            key=lambda key: (specs[key]["parameters_billions"], key),
        )
        if not eligible_discovery:
            decision = "BLOCKED_BY_MODEL_ADAPTER"
        else:
            discovery = eligible_discovery[0]
            replication = min(pair[1] for pair in shared_pairs if pair[0] == discovery)
            selected = {"discovery_model": discovery, "replication_model": replication}
            decision = "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION"
    elif any_pairs:
        decision = "MEASUREMENT_CONTRACT_NO_GO"
    else:
        decision = "JOINT_COMPOSITION_NO_GO"
    result = {
        "decision": decision,
        "models": models,
        "qualifying_pairs": {
            contract: [list(pair) for pair in sorted(pairs)]
            for contract, pairs in pair_sets.items()
        },
        "shared_qualifying_pairs": [list(pair) for pair in sorted(shared_pairs)],
        "selected_models": selected,
        "exact_next_action": (
            "BUILD_HELD_OUT_MECHANISTIC_PILOT"
            if decision == "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION"
            else "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
        ),
    }
    return result


def analyze_joint() -> dict[str, Any]:
    status = json.loads(
        (ARTIFACTS / "manifests" / "joint_screen_status.json").read_text(encoding="utf-8")
    )
    if status["status"] != "complete":
        result = {
            "decision": status["upstream_decision"],
            "status": "JOINT_NOT_RUN_BY_ATOMIC_STOP_RULE",
            "models": {},
            "exact_next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
        }
        write_json(ARTIFACTS / "joint" / "analysis.json", result)
        return result
    result = analyze_joint_rows(
        read_jsonl(ARTIFACTS / "joint" / "predictions.jsonl"),
        read_jsonl(ARTIFACTS / "joint" / "atomic_retention_predictions.jsonl"),
    )
    write_json(ARTIFACTS / "joint" / "analysis.json", result)
    return result
