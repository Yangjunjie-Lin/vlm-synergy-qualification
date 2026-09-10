from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from capability_gate.atomic_v2_data import (
    CARDINAL,
    QUESTION_FORMS,
    TASKS,
    TEXT_FORMS,
    DataValidationError,
    _aliases,
    _assert_pre_outcome,
    _obj,
    _row_failures,
    grouped_shortcut_probes,
    symbolic_atomic_oracle,
)


@pytest.fixture
def visual_row() -> dict:
    query, reference, first, second = _aliases(
        "unit-test-not-a-formal-scene", ["query", "reference", "distractor", "distractor"])
    return {
        "schema_version": 2, "protocol": "capability_gate_atomic_v2",
        "scene_id": "unit-test-not-a-formal-scene", "task": TASKS[0],
        "query_names": [query, reference], "query_name": query, "reference_name": reference,
        "entities": [_obj(query, "query", (256, 106), "azure", "circle"),
                     _obj(reference, "reference", (256, 256), "black", "circle"),
                     _obj(first, "distractor", (82, 96), "violet", "square"),
                     _obj(second, "distractor", (430, 96), "violet", "triangle")],
        "requires_image": True, "template_id": 0, "premise": "",
        "question": QUESTION_FORMS[0].format(a=query, b=reference),
        "question_without_premise": QUESTION_FORMS[0].format(a=query, b=reference),
        "options": list(CARDINAL), "target": "north", "symbolic_answer": "north",
        "correct_option_position": 0, "entity_count": 4, "color": "azure", "shape": "circle",
        "image_path": "unit-test.png",
    }


def test_valid_visual_scene_has_no_failures(visual_row):
    assert _row_failures(visual_row) == []
    assert symbolic_atomic_oracle(visual_row) == "north"


def test_alias_collision_rejected(visual_row):
    visual_row["entities"][2]["name"] = visual_row["query_name"]
    assert "within_scene_alias_collision" in _row_failures(visual_row)


def test_query_equals_reference_rejected(visual_row):
    visual_row["query_names"][1] = visual_row["query_names"][0]
    visual_row["reference_name"] = visual_row["query_name"]
    assert "query_reference_equality" in _row_failures(visual_row)
    assert "self_relation" in _row_failures(visual_row)


def test_duplicate_distractors_rejected(visual_row):
    visual_row["entities"][3]["name"] = visual_row["entities"][2]["name"]
    assert "duplicate_distractor" in _row_failures(visual_row)


def test_self_relation_geometry_rejected(visual_row):
    visual_row["entities"][0].update(x=256, y=256)
    assert "symbolic_oracle_error" in _row_failures(visual_row)


def test_symbolic_target_is_not_used_as_oracle(visual_row):
    visual_row.update(target="south", symbolic_answer="south", correct_option_position=1)
    assert symbolic_atomic_oracle(visual_row) == "north"
    assert "symbolic_target_mismatch" in _row_failures(visual_row)


def test_reversal_checks_ordered_identities_and_strict_inverse(visual_row):
    query, reference = visual_row["query_names"]
    visual_row.update(task=TASKS[2], requires_image=False,
                      premise=TEXT_FORMS[0].format(b=reference, c=query, relation="south"))
    for entity in visual_row["entities"]:
        entity.update(drawn=False, display_name="")
    visual_row["question"] = ("The image is unrelated decoration; use the text statement.\n"
                              + visual_row["premise"] + "\n"
                              + QUESTION_FORMS[0].format(a=query, b=reference))
    assert symbolic_atomic_oracle(visual_row) == "north"
    assert _row_failures(visual_row) == []
    visual_row["premise"] = TEXT_FORMS[0].format(b=query, c=reference, relation="south")
    assert "text_fact_direction_contract" in _row_failures(visual_row)
    assert "symbolic_target_mismatch" in _row_failures(visual_row)


def test_actual_entities_must_be_drawn(visual_row):
    visual_row["entities"][0]["drawn"] = False
    assert "query_or_reference_not_drawn" in _row_failures(visual_row)


def test_distractor_occlusion_rejected(visual_row):
    visual_row["entities"][2].update(x=256, y=106)
    assert "object_overlap_or_occlusion" in _row_failures(visual_row)


def test_question_entity_mismatch_rejected(visual_row):
    visual_row["question"] = "Where is a different entity relative to the reference?"
    assert "question_identity_or_template_mismatch" in _row_failures(visual_row)


def test_hmac_aliases_are_alpha_unique_and_not_semantic():
    aliases = _aliases("unit-test-aliases", ["query", "reference"] + ["distractor"] * 100)
    assert len(set(aliases)) == len(aliases)
    assert all(alias.isalpha() and len(alias) == 10 for alias in aliases)
    assert all(not any(direction in alias for direction in CARDINAL) for alias in aliases)
    assert aliases == _aliases("unit-test-aliases", ["query", "reference"] + ["distractor"] * 100)


def test_preoutcome_barrier_rejects_any_partial_formal_output(tmp_path: Path):
    path = tmp_path / "artifacts/atomic_v2/formal/qwen/predictions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"partial": true}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="regeneration prohibited"):
        _assert_pre_outcome(tmp_path)


def test_preoutcome_barrier_rejects_formal_lock(tmp_path: Path):
    path = tmp_path / "artifacts/atomic_v2/formal_run_lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"frozen": True}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="already frozen"):
        _assert_pre_outcome(tmp_path)


def test_numeric_probe_detects_index_modulo_answer_encoding(visual_row):
    rows = []
    for index in range(64):
        row = copy.deepcopy(visual_row)
        row.update(scene_id=f"test-{index}", scene_index=index + 1,
                   nuisance_block_id=f"test-block-{index // 4}", target=CARDINAL[index % 4],
                   correct_option_position=index % 4)
        rows.append(row)
    result = grouped_shortcut_probes(rows)
    assert result["probes"]["scene_index"]["cross_validated_accuracy"] > 0.30
    assert not result["probes"]["scene_index"]["gate"]
    assert result["probes"]["scene_index"]["prediction_count"] == 64
    assert result["probes"]["scene_index"]["classifier"] == "unconditional_train_only_ridge"


def test_all_cv_predictions_are_held_out_by_nuisance_group(visual_row):
    rows = []
    for index in range(64):
        row = copy.deepcopy(visual_row)
        row.update(scene_id=f"test-{index}", scene_index=index + 1,
                   nuisance_block_id=f"test-block-{index // 4}", target=CARDINAL[index % 4],
                   correct_option_position=index % 4)
        rows.append(row)
    report = grouped_shortcut_probes(rows)
    groups = report["held_out_groups"]
    assert len({group for fold in groups for group in fold}) == 16
    by_id = {row["scene_id"]: row for row in rows}
    for probe in report["probes"].values():
        for prediction in probe["predictions"]:
            assert by_id[prediction["scene_id"]]["nuisance_block_id"] in groups[prediction["fold"]]
