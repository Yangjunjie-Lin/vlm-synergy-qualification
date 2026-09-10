from copy import deepcopy

import pytest

from capability_gate.atomic_v2_statistics import (
    CARDINAL,
    DIAGONAL,
    MODEL_KEYS,
    MODES,
    TASK_PRIORITY,
    adjudicate_atomic_rows_v2,
    analyze_joint_rows_v2,
    guard_activation_patching_v2,
    require_atomic_v2_go,
)


def _rows() -> list[dict]:
    result = []
    for model in MODEL_KEYS:
        for task in TASK_PRIORITY:
            for index in range(64):
                target = CARDINAL[index % 4]
                scores = {answer: -1.0 if answer == target else -3.0 for answer in CARDINAL}
                result.append({
                    "model_key": model,
                    "task": task,
                    "scene_id": f"fresh-v2-{task}-{index}",
                    "target": target,
                    "top_answer": target,
                    "constrained_generation_answer": target,
                    "correct_option_position": (index // 4) % 4,
                    "normalized_log_likelihood": scores,
                    "target_margin": 2.0,
                })
    return result


def _set_prediction(row: dict, answer: str, generation: str | None = None) -> None:
    row["top_answer"] = answer
    row["constrained_generation_answer"] = answer if generation is None else generation
    scores = {candidate: -1.0 if candidate == answer else -3.0 for candidate in CARDINAL}
    row["normalized_log_likelihood"] = scores
    row["target_margin"] = scores[row["target"]] - sum(
        value for candidate, value in scores.items() if candidate != row["target"]
    ) / 3.0


def test_two_complete_models_pass_and_joint_is_authorized() -> None:
    result = adjudicate_atomic_rows_v2(_rows())
    assert result["decision"] == "ATOMIC_COHORT_GO"
    assert result["qualified_count"] == 2
    assert result["q1_potential"] is None  # Mechanistic feasibility awaits Joint.
    require_atomic_v2_go(result)
    metrics = result["models"][MODEL_KEYS[0]]["tasks"][TASK_PRIORITY[0]]
    assert metrics["cll"]["one_sided_95_exact_lower"] > 0.95
    assert metrics["cll"]["per_answer_accuracy"] == dict.fromkeys(CARDINAL, 1.0)
    assert metrics["agreement"]["cohens_kappa"] == 1.0
    assert metrics["cll"]["candidate_margin"]["target_vs_mean_other"]["mean"] == 2.0


def test_common_25_percent_failure_with_kappa_one_is_not_contract_dependent() -> None:
    rows = _rows()
    for row in rows:
        index = int(row["scene_id"].rsplit("-", 1)[1])
        # A nonconstant shared permutation fixes north and cycles the other classes.
        shared = {"north": "north", "south": "east", "east": "west", "west": "south"}
        _set_prediction(row, shared[CARDINAL[index % 4]])
    result = adjudicate_atomic_rows_v2(rows)
    assert result["decision"] == "CAPABILITY_COHORT_NO_GO"
    assert result["q1_potential"] == "NO_FEASIBLE_COHORT"
    for model in result["models"].values():
        assert model["label"] == "ATOMIC_VISUAL_FAIL"
        assert model["agreement"]["cohens_kappa"] == 1.0
        assert model["failed_tasks"] == list(TASK_PRIORITY)
        assert len(model["failed_task_labels"]) == 4
        for task in model["tasks"].values():
            assert task["cll"]["accuracy"] == 0.25
            assert task["constrained_generation"]["accuracy"] == 0.25


def test_constant_common_answer_preserves_undefined_kappa_and_task_failures() -> None:
    rows = _rows()
    for row in rows:
        _set_prediction(row, "east")
    result = adjudicate_atomic_rows_v2(rows)
    assert result["decision"] == "CAPABILITY_COHORT_NO_GO"
    assert result["qualified_count"] == 0
    for model in result["models"].values():
        assert model["label"] == "ATOMIC_VISUAL_FAIL"
        assert model["agreement"]["cohens_kappa"] is None
        assert model["agreement"]["exact"] == 1.0
        assert model["agreement"]["constant_common_answer"] == "east"
        assert not model["agreement"]["kappa_gate"]


def test_actual_task_gate_conflict_takes_primary_priority_and_lists_failed_task() -> None:
    rows = _rows()
    for row in rows:
        if row["model_key"] == MODEL_KEYS[0] and row["task"] == "direct_text_relation":
            row["constrained_generation_answer"] = "east"
    result = adjudicate_atomic_rows_v2(rows)
    model = result["models"][MODEL_KEYS[0]]
    assert model["label"] == "MEASUREMENT_CONTRACT_DEPENDENT"
    assert model["task_gate_conflicts"] == ["direct_text_relation"]
    assert model["failed_tasks"] == ["direct_text_relation"]
    assert result["decision"] == "MEASUREMENT_CONTRACT_NO_GO"
    assert result["exact_next_action"] == "TERMINATE_CROSS_MODAL_SYNERGY_LINE"


def test_defined_low_kappa_is_contract_dependent_even_if_both_gates_fail() -> None:
    rows = _rows()
    for row in rows:
        _set_prediction(row, "east", "west")
    result = adjudicate_atomic_rows_v2(rows)
    for model in result["models"].values():
        assert model["agreement"]["cohens_kappa"] == 0.0
        assert not model["task_gate_conflicts"]
        assert model["label"] == "MEASUREMENT_CONTRACT_DEPENDENT"
        assert model["failed_tasks"] == list(TASK_PRIORITY)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "third_model", "wrong_task"])
def test_incomplete_or_contaminated_cohort_is_not_a_capability_conclusion(mutation: str) -> None:
    rows = _rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif mutation == "third_model":
        rows[0]["model_key"] = "phi4_multimodal_5_6b"
    else:
        rows[0]["task"] = "fifth_task"
    result = adjudicate_atomic_rows_v2(rows)
    assert result["decision"] == "MEASUREMENT_IMPLEMENTATION_NO_GO"
    assert result["q1_potential"] == "NOT_EVALUATED_MEASUREMENT_FAILURE"
    assert not result["scientific_capability_conclusion"]
    assert not result["metrics_computed"]
    assert result["models"] == {}


def test_likelihood_or_margin_corruption_blocks_adjudication() -> None:
    rows = _rows()
    rows[0]["normalized_log_likelihood"]["north"] = float("nan")
    assert adjudicate_atomic_rows_v2(rows)["decision"] == "MEASUREMENT_IMPLEMENTATION_NO_GO"
    rows = _rows()
    rows[0]["target_margin"] = 99.0
    assert adjudicate_atomic_rows_v2(rows)["decision"] == "MEASUREMENT_IMPLEMENTATION_NO_GO"


@pytest.mark.parametrize(
    ("task", "successes", "passes"),
    [
        ("direct_visual_relation", 51, False),
        ("direct_visual_relation", 52, True),
        ("direct_text_relation", 57, False),
        ("direct_text_relation", 58, True),
    ],
)
def test_exact_frozen_gate_boundaries(task: str, successes: int, passes: bool) -> None:
    rows = _rows()
    for row in rows:
        index = int(row["scene_id"].rsplit("-", 1)[1])
        if row["task"] == task and index >= successes:
            _set_prediction(row, CARDINAL[(CARDINAL.index(row["target"]) + 1) % 4])
    result = adjudicate_atomic_rows_v2(rows)
    for model in result["models"].values():
        assert model["tasks"][task]["cll_gate"] is passes
        assert model["tasks"][task]["generation_gate"] is passes


def test_one_qualified_model_cannot_authorize_joint() -> None:
    rows = _rows()
    for row in rows:
        if row["model_key"] == MODEL_KEYS[1]:
            _set_prediction(row, "east")
    result = adjudicate_atomic_rows_v2(rows)
    assert result["qualified_count"] == 1
    with pytest.raises(RuntimeError, match="JOINT_BLOCKED_BY_ATOMIC_V2_GATE"):
        require_atomic_v2_go(result)
    with pytest.raises(RuntimeError, match="JOINT_BLOCKED_BY_ATOMIC_V2_GATE"):
        analyze_joint_rows_v2([], [], result)


def test_missing_joint_results_after_atomic_go_are_measurement_failure() -> None:
    atomic = adjudicate_atomic_rows_v2(_rows())
    result = analyze_joint_rows_v2([], _rows(), atomic)
    assert result["decision"] == "MEASUREMENT_IMPLEMENTATION_NO_GO"
    assert result["q1_potential"] == "NOT_EVALUATED_MEASUREMENT_FAILURE"


def _joint_rows(*, passes: bool) -> list[dict]:
    result = []
    targets = {
        "I0T0": "northeast", "I0T1": "southeast", "I1T0": "northwest", "I1T1": "southwest"
    }
    for model in MODEL_KEYS:
        for index in range(128):
            for mode in MODES:
                for condition, target in targets.items():
                    answer = target if passes and mode == "joint" else "northeast"
                    scores = {label: -1.0 if label == answer else -3.0 for label in DIAGONAL}
                    result.append({
                        "model_key": model,
                        "base_quartet_id": f"synthetic-test-quartet-{index}",
                        "mode": mode,
                        "condition": condition,
                        "target": target,
                        "correct_option_position": DIAGONAL.index(target),
                        "top_answer": answer,
                        "constrained_generation_answer": answer,
                        "normalized_log_likelihood": scores,
                        "target_margin": scores[target] - sum(
                            value for label, value in scores.items() if label != target
                        ) / 3.0,
                        "psi_fixed_target": "southwest",
                        "axis_map": "horizontal_image_vertical_text",
                        "template_id": index % 4,
                    })
    return result


@pytest.mark.parametrize("passes", [False, True])
def test_complete_joint_uses_frozen_analysis_and_correct_final_mapping(passes: bool) -> None:
    retention = _rows()
    atomic = adjudicate_atomic_rows_v2(retention)
    result = analyze_joint_rows_v2(_joint_rows(passes=passes), retention, atomic)
    assert result["q1_potential"] == (
        "MECHANISTIC_STUDY_FEASIBLE" if passes else "NO_MECHANISTIC_STUDY_BASIS"
    )
    assert not result["activation_patching_authorized"]
    assert not result["activation_patching_executed"]
    for model in result["models"].values():
        assert model["psi"]["quartets"] == 128
        assert model["psi"]["seed"] == 9041723
        assert model["psi"]["resamples"] == 50000
        assert set(model["complete_metrics"]["cll"]["joint"]["by_template"]) == {
            "0", "1", "2", "3"
        }
        assert model["complete_metrics"]["cll"]["joint"]["n"] == 512


@pytest.mark.parametrize("joint_go", [False, True])
def test_patching_is_blocked_before_and_after_joint_go_in_this_task(joint_go: bool) -> None:
    with pytest.raises(RuntimeError, match="ACTIVATION_PATCHING_FORBIDDEN"):
        guard_activation_patching_v2(joint_go=joint_go)
