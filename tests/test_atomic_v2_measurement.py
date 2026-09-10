"""Synthetic unit fixtures are never processor snapshots or model evidence."""

from __future__ import annotations

import copy

import pytest
from PIL import Image

from capability_gate.atomic_v2_measurement import (
    CANDIDATES,
    GENERATION_SUFFIX,
    IMAGE_MARKERS,
    GlmAtomicV2Adapter,
    QwenAtomicV2Adapter,
    _preoutcome_gate,
    _read_verified_existing,
    _write_new,
    adapter_for,
    contract_control_cases,
    prompt_user,
    validate_contract_control,
    validate_rendered_prompt,
)
from capability_gate.recovery.adapters import (
    DESCRIPTORS,
    GlmRecoveryAdapter,
    MeasurementImplementationError,
    QwenRecoveryAdapter,
)

SYSTEM = "Return exactly one allowed word without explanation."
QUESTION = "Where is abcdefghij relative to klmnopqrst?"
SCHEMA = "Return exactly one allowed lowercase direction word and nothing else."


def _rendered(model: str, *, visual: bool = False, question: str = QUESTION) -> str:
    user = prompt_user(question, CANDIDATES, SCHEMA)
    block = "".join(IMAGE_MARKERS[model]) if visual else ""
    if model == "qwen2_5_vl_7b":
        return (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{block}{user}<|im_end|>\n{GENERATION_SUFFIX[model]}"
        )
    return (
        f"[gMASK]<sop><|system|>\n{SYSTEM}<|user|>\n"
        f"{block}{user}{GENERATION_SUFFIX[model]}"
    )


def _validate(model: str = "glm4_1v_9b", *, visual: bool = False, **overrides):
    kwargs = {
        "model_key": model, "rendered": _rendered(model, visual=visual),
        "encoded": {"input_ids": [[1, 2, 3]], **({"pixel_values": [1]} if visual else {})},
        "system": SYSTEM, "user": prompt_user(QUESTION, CANDIDATES, SCHEMA),
        "question": QUESTION, "options": CANDIDATES, "visual": visual,
        "query_name": "abcdefghij", "reference_name": "klmnopqrst",
    }
    kwargs.update(overrides)
    return validate_rendered_prompt(**kwargs)


@pytest.mark.parametrize("model", ["qwen2_5_vl_7b", "glm4_1v_9b"])
@pytest.mark.parametrize("visual", [True, False])
def test_plain_official_shape_with_exact_source_content_is_valid(model, visual):
    assert _validate(model, visual=visual)["overall_gate"]


def test_glm_python_repr_invalidates_renderer():
    rendered = _rendered("glm4_1v_9b").replace(SYSTEM, str([{"type": "text", "text": SYSTEM}]))
    checks = _validate(rendered=rendered)
    assert not checks["overall_gate"]
    assert not checks["gates"]["system_is_plain_text_not_python_repr"]


def test_visual_task_missing_image_token_fails():
    assert not _validate(visual=True, rendered=_rendered("glm4_1v_9b"))["overall_gate"]


def test_text_task_with_image_token_fails():
    assert not _validate(rendered=_rendered("glm4_1v_9b", visual=True))["overall_gate"]


@pytest.mark.parametrize("change", ["hidden_answer", "duplicate_system", "question", "suffix"])
def test_hidden_content_duplicate_system_wrong_question_and_generation_boundary_fail(change):
    rendered = _rendered("glm4_1v_9b")
    if change == "hidden_answer":
        rendered = rendered.replace("<|assistant|>", "Hidden answer: north<|assistant|>")
    elif change == "duplicate_system":
        rendered = rendered.replace(SYSTEM, SYSTEM + SYSTEM)
    elif change == "question":
        rendered = rendered.replace(QUESTION, "Where is something else?")
    else:
        rendered += "<think>"
    assert not _validate(rendered=rendered)["overall_gate"]


def test_equal_query_reference_and_empty_tokenization_fail():
    assert not _validate(reference_name="abcdefghij")["overall_gate"]
    assert not _validate(encoded={"input_ids": [[]]})["overall_gate"]


def test_duplicate_candidate_list_fails():
    rendered = _rendered("glm4_1v_9b").replace(
        "<|assistant|>", "Allowed answers in displayed order: north, south, east, west.<|assistant|>"
    )
    assert not _validate(rendered=rendered)["gates"]["candidate_list_once"]


def test_frozen_qwen_scoring_renderer_and_glm_native_loader_are_inherited():
    assert QwenAtomicV2Adapter._encode is QwenRecoveryAdapter._encode
    assert QwenAtomicV2Adapter.score_and_generate is QwenRecoveryAdapter.score_and_generate
    assert GlmAtomicV2Adapter.score_and_generate is GlmRecoveryAdapter.score_and_generate
    assert GlmAtomicV2Adapter._model_loader is GlmRecoveryAdapter._model_loader
    assert adapter_for("glm4_1v_9b").descriptor == DESCRIPTORS["glm4_1v_9b"]
    for forbidden in ("phi4_multimodal_5_6b", "fourth_model"):
        with pytest.raises(ValueError):
            adapter_for(forbidden)


def test_glm_uses_actual_processor_contract_with_string_system():
    calls = []

    def apply_chat_template(messages, **kwargs):
        calls.append((messages, kwargs))
        return {"input_ids": [[1]]} if kwargs.get("tokenize") else "rendered"

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['tokenize'] is False
            return apply_chat_template(messages, **kwargs)

        def __call__(self, **kwargs):
            calls.append(kwargs)
            assert kwargs['add_special_tokens'] is False
            assert kwargs['text'] == ['rendered']
            assert len(kwargs['images']) == 1
            return {'input_ids': [[1]]}

    adapter = GlmAtomicV2Adapter()
    adapter.processor = Processor()
    adapter._encode(SYSTEM, "user", Image.new("RGB", (10, 10)))
    assert len(calls) == 2
    for messages, kwargs in calls[:1]:
        assert messages[0] == {"role": "system", "content": SYSTEM}
        assert len([item for item in messages[1]["content"] if item["type"] == "image"]) == 1
        assert kwargs["add_generation_prompt"] is True
    assert calls[1]["return_tensors"] == "pt"


def _control_records():
    return [
        {
            "case_id": case["case_id"], "prompt_validation": {"overall_gate": True},
            "result": {
                "rendered_prompt": "unit-test-only", "candidate_ranking": list(CANDIDATES),
                "top_answer": case["target"], "target_margin": 1.0,
                "constrained_answer": case["target"], "constrained_generation_token_ids": [10, 2],
                "runtime_seconds": 0.1, "input_sequence_length": 10,
                "candidate_scores": [
                    {"candidate": answer, "candidate_token_ids": [10 + index],
                     "candidate_token_count": 1, "token_log_probabilities": [-1.0],
                     "raw_log_likelihood": -1.0, "normalized_log_likelihood": -1.0}
                    for index, answer in enumerate(CANDIDATES)
                ],
            },
        }
        for case in contract_control_cases()
    ]


def test_controls_are_sixteen_disjoint_engineering_cases_and_no_scientific_accuracy():
    cases = contract_control_cases()
    assert len({case["case_id"] for case in cases}) == 16
    assert all(sum(case["target"] == answer for case in cases) == 4 for answer in CANDIDATES)
    result = validate_contract_control(cases, _control_records())
    assert result["overall_gate"]
    assert result["scientific_accuracy_reported"] is False
    assert "accuracy" not in result


def test_constant_constrained_answer_fails_controls():
    records = _control_records()
    for record in records:
        record["result"]["constrained_answer"] = "east"
    result = validate_contract_control(contract_control_cases(), records)
    assert result["status"] == "MEASUREMENT_IMPLEMENTATION_NO_GO"
    assert result["constant_one_answer_degeneration"] is True
    assert not result["gates"]["all_four_answers_can_be_generated"]


@pytest.mark.parametrize("mutation", ["empty_ids", "nonfinite", "invalid", "partial", "duplicate"])
def test_control_contract_gate_rejects_broken_or_incomplete_results(mutation):
    records = copy.deepcopy(_control_records())
    if mutation == "empty_ids":
        records[0]["result"]["candidate_scores"][0]["candidate_token_ids"] = []
    elif mutation == "nonfinite":
        records[0]["result"]["candidate_scores"][0]["normalized_log_likelihood"] = float("nan")
    elif mutation == "invalid":
        records[0]["result"]["constrained_answer"] = "northeast"
    elif mutation == "partial":
        records.pop()
    else:
        records[1]["case_id"] = records[0]["case_id"]
    assert not validate_contract_control(contract_control_cases(), records)["overall_gate"]


def test_measurement_artifact_cannot_be_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    _write_new(path, {"files": [], "overall_gate": False})
    with pytest.raises(FileExistsError):
        _write_new(path, {"overall_gate": True})
    assert _read_verified_existing(path, tmp_path)["overall_gate"] is False


def test_failed_data_gate_blocks_model_measurement(tmp_path):
    _write_new(tmp_path / "artifacts/atomic_v2/data_validation.json", {"overall_gate": False})
    with pytest.raises(MeasurementImplementationError, match="ATOMIC_V2_DATA_INVALID"):
        _preoutcome_gate(tmp_path)
