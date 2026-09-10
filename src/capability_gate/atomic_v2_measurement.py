"""Isolated, pre-outcome renderer and answer-contract checks for Atomic v2.

Scientific scoring and generation are inherited unchanged from recovery adapters.
This module never reads task correctness to repair a prompt or adapter. Historical
adapters and their outputs are intentionally not edited.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import re
import subprocess
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from capability_gate.artifacts import (
    canonical_json,
    read_jsonl,
    runtime_metadata,
    sha256_file,
    sha256_text,
)
from capability_gate.paths import ROOT
from capability_gate.recovery.adapters import (
    GlmRecoveryAdapter,
    MeasurementImplementationError,
    NativeRecoveryAdapter,
    QwenRecoveryAdapter,
)

MODEL_KEYS = ("qwen2_5_vl_7b", "glm4_1v_9b")
CANDIDATES = ("north", "south", "east", "west")
TASKS = (
    "direct_visual_relation",
    "direct_text_relation",
    "direction_reversal",
    "cross_modal_bridge_binding",
)
VISUAL_TASKS = {"direct_visual_relation", "cross_modal_bridge_binding"}
CONTROL_NAMESPACE = uuid.UUID("0948951f-f136-4cb9-a5e2-5ed0348cc068")
IMAGE_MARKERS = {
    "qwen2_5_vl_7b": ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"),
    "glm4_1v_9b": ("<|begin_of_image|>", "<|image|>", "<|end_of_image|>"),
}
GENERATION_SUFFIX = {
    "qwen2_5_vl_7b": "<|im_start|>assistant\n",
    "glm4_1v_9b": "<|assistant|>\n",
}


class QwenAtomicV2Adapter(QwenRecoveryAdapter):
    """An explicit v2 identity with exactly the historical Qwen implementation."""


class GlmAtomicV2Adapter(GlmRecoveryAdapter):
    """One engineering correction: GLM's official template expects system str."""

    def _encode(self, system: str, user: str, image: Image.Image | None) -> tuple[str, Any]:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        encoded = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return rendered, encoded


def adapter_for(model_key: str) -> NativeRecoveryAdapter:
    factories = {
        "qwen2_5_vl_7b": QwenAtomicV2Adapter,
        "glm4_1v_9b": GlmAtomicV2Adapter,
    }
    if model_key not in factories:
        raise ValueError(f"Atomic v2 permits only frozen Qwen and GLM, not {model_key}")
    return factories[model_key]()


def prompt_user(question: str, options: Sequence[str], schema: str) -> str:
    return (
        f"{question}\nAllowed answers in displayed order: {', '.join(options)}.\n"
        f"Answer schema: {schema}"
    )


def _ids(encoded: Mapping[str, Any]) -> list[int]:
    values = encoded.get("input_ids", [])
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            return []
        values = values[0]
    return list(values)


def _plain_prompt(rendered: str, model_key: str) -> str:
    # Permit only official role/image wrappers; arbitrary hidden text is retained
    # and therefore makes the exact normalized source-content comparison fail.
    value = rendered
    wrappers = [*IMAGE_MARKERS[model_key], GENERATION_SUFFIX[model_key]]
    if model_key == "qwen2_5_vl_7b":
        wrappers += ["<|im_start|>system\n", "<|im_start|>user\n", "<|im_end|>"]
    else:
        wrappers += ["[gMASK]<sop>", "<|system|>\n", "<|user|>\n"]
    for wrapper in wrappers:
        value = value.replace(wrapper, " ")
    return " ".join(value.split())


def validate_rendered_prompt(
    *,
    model_key: str,
    rendered: str,
    encoded: Mapping[str, Any],
    system: str,
    user: str,
    question: str,
    options: Sequence[str],
    visual: bool,
    query_name: str | None = None,
    reference_name: str | None = None,
    engineering_only: bool = False,
) -> dict[str, Any]:
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unknown model: {model_key}")
    python_repr = bool(
        re.search(r"\[\s*\{\s*['\"](?:type|text)['\"]\s*:", rendered)
        or re.search(r"\{\s*['\"]type['\"]\s*:\s*['\"]text['\"]", rendered)
    )
    expected_images = 1 if visual else 0
    counts = {marker: rendered.count(marker) for marker in IMAGE_MARKERS[model_key]}
    all_image_markers = {marker for markers in IMAGE_MARKERS.values() for marker in markers}
    unexpected_images = any(
        marker in rendered for marker in all_image_markers - set(IMAGE_MARKERS[model_key])
    )
    candidates_line = f"Allowed answers in displayed order: {', '.join(options)}."
    input_ids = _ids(encoded)
    visual_keys = sorted(
        key for key in encoded if key.startswith(("pixel_", "image_", "input_image"))
    )
    gates = {
        "system_instruction_once": bool(system) and rendered.count(system) == 1,
        "system_is_plain_text_not_python_repr": not python_repr,
        "correct_image_block_count": all(count == expected_images for count in counts.values())
        and not unexpected_images,
        "visual_tensor_presence_matches_task": bool(visual_keys) == visual,
        "query_reference_distinct": engineering_only
        or bool(query_name and reference_name and query_name != reference_name),
        "question_matches_data_row": bool(question)
        and user.startswith(question + "\n")
        and rendered.count(question) == 1,
        "candidate_list_once": rendered.count(candidates_line) == 1
        and rendered.count("Allowed answers in displayed order:") == 1
        and len(options) == 4
        and set(options) == set(CANDIDATES),
        "generation_prompt_at_end": rendered.endswith(GENERATION_SUFFIX[model_key])
        and rendered.count(GENERATION_SUFFIX[model_key]) == 1,
        "no_hidden_answer_or_added_content": _plain_prompt(rendered, model_key)
        == " ".join(f"{system} {user}".split()),
        "tokenized_prompt_nonempty": bool(input_ids)
        and all(isinstance(value, int) for value in input_ids),
    }
    return {
        "overall_gate": all(gates.values()),
        "gates": gates,
        "rendered_prompt_sha256": sha256_text(rendered),
        "input_ids_sha256": sha256_text(canonical_json(input_ids)),
        "input_sequence_length": len(input_ids),
        "image_block_counts": counts,
        "visual_input_keys": visual_keys,
    }


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _write_new_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _files(root: Path, paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]


def _read_verified_existing(path: Path, root: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    for record in value.get("files", []):
        artifact = root / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise MeasurementImplementationError(f"immutable measurement artifact changed: {artifact}")
    return value


def _preoutcome_gate(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation = json.loads(
        (root / "artifacts/atomic_v2/data_validation.json").read_text(encoding="utf-8")
    )
    passed = validation.get("overall_gate") is True or validation.get("status") in {
        "ATOMIC_V2_DATA_VALID", "PASS"
    }
    if not passed:
        raise MeasurementImplementationError("ATOMIC_V2_DATA_INVALID: measurement is forbidden")
    formal = root / "artifacts/atomic_v2/formal"
    if formal.exists() and any(path.stat().st_size for path in formal.rglob("*.jsonl")):
        raise MeasurementImplementationError("renderer/control changes after formal outputs forbidden")
    scenes = read_jsonl(root / "artifacts/data_v2/atomic_qualification/scenes.jsonl")
    if len(scenes) != 256 or Counter(row["task"] for row in scenes) != {
        task: 64 for task in TASKS
    }:
        raise MeasurementImplementationError("formal Atomic v2 requires exactly four tasks x64")
    old_ids = {
        row["scene_id"]
        for row in read_jsonl(root / "artifacts/data/atomic_qualification/scenes.jsonl")
    }
    if old_ids & {row["scene_id"] for row in scenes}:
        raise MeasurementImplementationError("Atomic v1 scene IDs may not enter v2")
    return validation, scenes


def _atomic_config(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "configs/atomic_tasks.yaml").read_text(encoding="utf-8"))


def _processor_record(adapter: NativeRecoveryAdapter) -> dict[str, Any]:
    template = getattr(adapter.processor, "chat_template", None)
    if template is None:
        template = getattr(adapter.processor.tokenizer, "chat_template", None)
    descriptor = adapter.descriptor
    return {
        "model_key": descriptor.key,
        "model_id": descriptor.model_id,
        "model_revision": descriptor.revision,
        "processor_revision": descriptor.processor_revision,
        "tokenizer_revision": descriptor.processor_revision,
        "processor_class": type(adapter.processor).__name__,
        "tokenizer_class": type(adapter.processor.tokenizer).__name__,
        "official_chat_template": template,
        "official_chat_template_sha256": sha256_text(canonical_json(template)),
        "native_model_class": descriptor.expected_model_class,
        "renderer_adapter_sha256": sha256_file(Path(__file__)),
        "inherited_scoring_adapter_sha256": sha256_file(
            Path(__file__).parent / "recovery/adapters.py"
        ),
    }


def _scene_names(scene: Mapping[str, Any]) -> tuple[str | None, str | None]:
    names = scene.get("query_names", [])
    query = scene.get("query_name", names[0] if len(names) > 0 else None)
    reference = scene.get("reference_name", names[1] if len(names) > 1 else None)
    return query, reference


def validate_model_renderers(model_key: str, root: Path = ROOT) -> dict[str, Any]:
    """Load actual frozen processor, not weights; save eight immutable snapshots."""
    adapter = adapter_for(model_key)
    destination = root / "artifacts/atomic_v2/renderer_snapshots" / model_key
    manifest_path = destination / "manifest.json"
    existing = _read_verified_existing(manifest_path, root)
    if existing is not None:
        return existing
    _, scenes = _preoutcome_gate(root)
    config = _atomic_config(root)
    selected = [row for task in TASKS for row in [r for r in scenes if r["task"] == task][:2]]
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    failure = None
    started = time.perf_counter()
    stderr_path = destination / "stderr.log"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("x", encoding="utf-8") as stderr, contextlib.redirect_stderr(stderr):
        try:
            adapter.load_processor()
            processor = _processor_record(adapter)
            for index, scene in enumerate(selected):
                image = None
                if scene.get("requires_image", scene["task"] in VISUAL_TASKS):
                    with Image.open(root / scene["image_path"]) as source:
                        image = source.convert("RGB")
                user = prompt_user(scene["question"], scene["options"], config["answer_schema"])
                rendered, encoded = adapter._encode(config["system_instruction"], user, image)
                query, reference = _scene_names(scene)
                checks = validate_rendered_prompt(
                    model_key=model_key, rendered=rendered, encoded=encoded,
                    system=config["system_instruction"], user=user, question=scene["question"],
                    options=scene["options"], visual=scene["task"] in VISUAL_TASKS,
                    query_name=query, reference_name=reference,
                )
                snapshot = {
                    "schema_version": 1, "model_key": model_key, "scene_id": scene["scene_id"],
                    "task": scene["task"], "prompt": {"system": config["system_instruction"],
                    "user": user}, "rendered_prompt": rendered, "input_ids": _ids(encoded),
                    "validation": checks, "processor": processor,
                }
                stem = f"{scene['task']}_{index % 2 + 1:02d}"
                snapshot_path = destination / f"{stem}.json"
                _write_new(snapshot_path, snapshot)
                family = "glm" if model_key == "glm4_1v_9b" else "qwen"
                test_path = root / f"tests/snapshots/{family}_atomic_prompts/{stem}.txt"
                _write_new_text(test_path, rendered)
                paths.extend([snapshot_path, test_path])
                records.append({"scene_id": scene["scene_id"], "task": scene["task"], **checks})
        except Exception as error:  # noqa: BLE001 - preserve all actual processor failures
            failure = {"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()}
            processor = _processor_record(adapter) if adapter.processor is not None else None
        finally:
            adapter.close()
    paths.append(stderr_path)
    passed = len(records) == 8 and all(row["overall_gate"] for row in records) and failure is None
    manifest = {
        "schema_version": 1, "model_key": model_key,
        "status": "RENDERER_VALIDATION_PASS" if passed else "MEASUREMENT_IMPLEMENTATION_NO_GO",
        "overall_gate": passed, "processor": processor, "snapshots": records,
        "snapshot_count": len(records), "failure": failure,
        "runtime_seconds": time.perf_counter() - started,
        "model_forward_passes": 0, "files": _files(root, paths), **runtime_metadata(root),
    }
    _write_new(manifest_path, manifest)
    return manifest


def contract_control_cases() -> list[dict[str, Any]]:
    """Sixteen explicit engineering instructions, never scientific task scenes."""
    return [
        {
            "case_id": str(uuid.uuid5(CONTROL_NAMESPACE, f"contract-control-{repetition}-{answer}")),
            "split": "contract_control", "engineering_only": True, "target": answer,
            "question": f'For this output-contract test, return exactly "{answer}".',
            "options": list(CANDIDATES), "image_path": None,
        }
        for repetition in range(4) for answer in CANDIDATES
    ]


def validate_contract_control(
    cases: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    required_result_fields = {
        "rendered_prompt", "candidate_scores", "candidate_ranking", "top_answer",
        "target_margin", "constrained_answer", "constrained_generation_token_ids",
        "runtime_seconds", "input_sequence_length",
    }
    case_ids = {case["case_id"] for case in cases}
    results = [record.get("result", {}) for record in records]
    scores = [score for result in results for score in result.get("candidate_scores", [])]
    answers = [result.get("constrained_answer") for result in results]
    complete = (
        len(cases) == len(records) == 16
        and len(case_ids) == 16
        and Counter(record.get("case_id") for record in records) == Counter(case_ids)
        and all(required_result_fields <= set(result) for result in results)
        and all(record.get("prompt_validation", {}).get("overall_gate") is True for record in records)
        and all({s.get("candidate") for s in result.get("candidate_scores", [])}
                == set(CANDIDATES) and len(result.get("candidate_scores", [])) == 4
                for result in results)
    )
    invalid_count = sum(answer not in CANDIDATES for answer in answers)
    constant = len(set(answers)) == 1 and bool(answers)
    gates = {
        "all_candidate_token_sequences_nonempty": len(scores) == 64 and all(
            score.get("candidate_token_count", 0) > 0
            and len(score.get("candidate_token_ids", [])) == score.get("candidate_token_count")
            and len(score.get("token_log_probabilities", [])) == score.get("candidate_token_count")
            for score in scores
        ),
        "all_cll_scores_finite": len(scores) == 64 and all(
            all(isinstance(value, (int, float)) and math.isfinite(value) for value in [
                score.get("normalized_log_likelihood"), score.get("raw_log_likelihood"),
                *score.get("token_log_probabilities", [])]) for score in scores
        ),
        "all_four_answers_can_be_generated": set(answers) == set(CANDIDATES),
        "invalid_constrained_output_count_zero": invalid_count == 0,
        "constant_one_answer_degeneration_false": not constant,
        "artifact_completeness": complete,
    }
    passed = all(gates.values())
    return {
        "status": "CONTRACT_CONTROL_PASS" if passed else "MEASUREMENT_IMPLEMENTATION_NO_GO",
        "overall_gate": passed, "gates": gates, "case_count": len(cases),
        "completed_case_count": len(records), "invalid_constrained_output_count": invalid_count,
        "constant_one_answer_degeneration": constant,
        "generated_answer_counts": dict(Counter(str(answer) for answer in answers)),
        "artifact_completeness": 1.0 if complete else 0.0,
        "engineering_only": True, "scientific_accuracy_reported": False,
    }


def _environment(root: Path) -> dict[str, Any]:
    packages = sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    payload = {**runtime_metadata(root), "packages": packages, "gpu_inventory": gpu}
    return {**payload, "environment_sha256": sha256_text(canonical_json(payload))}


def run_contract_control(model_key: str, root: Path = ROOT) -> dict[str, Any]:
    """Run one real frozen model and release it; call in separate model environments."""
    adapter = adapter_for(model_key)
    destination = root / "artifacts/atomic_v2/contract_control" / model_key
    manifest_path = destination / "manifest.json"
    existing = _read_verified_existing(manifest_path, root)
    if existing is not None:
        return existing
    _, scenes = _preoutcome_gate(root)
    # Both processor gates must precede any engineering or formal model inference.
    for key in MODEL_KEYS:
        renderer_path = root / "artifacts/atomic_v2/renderer_snapshots" / key / "manifest.json"
        renderer = _read_verified_existing(renderer_path, root)
        if renderer is None or renderer.get("overall_gate") is not True:
            raise MeasurementImplementationError(f"renderer gate not passed: {key}")
    config = _atomic_config(root)
    scoring = yaml.safe_load((root / "configs/scoring.yaml").read_text(encoding="utf-8"))
    cases = contract_control_cases()
    if {case["case_id"] for case in cases} & {row["scene_id"] for row in scenes}:
        raise MeasurementImplementationError("engineering controls overlap formal scene IDs")
    cases_path = destination / "cases.json"
    _write_new(cases_path, cases)
    environment_path = destination / "environment.json"
    _write_new(environment_path, _environment(root))
    records: list[dict[str, Any]] = []
    paths = [cases_path, environment_path]
    failure = None
    metadata = None
    started = time.perf_counter()
    stderr_path = destination / "stderr.log"
    predictions_path = destination / "predictions.jsonl"
    with (stderr_path.open("x", encoding="utf-8") as stderr,
          predictions_path.open("x", encoding="utf-8", newline="\n") as predictions,
          contextlib.redirect_stderr(stderr)):
        try:
            adapter.load()
            adapter.verify_cached_weights()
            metadata = adapter.runtime_metadata()
            for case in cases:
                user = prompt_user(case["question"], case["options"], config["answer_schema"])
                rendered, encoded = adapter._encode(config["system_instruction"], user, None)
                checks = validate_rendered_prompt(
                    model_key=model_key, rendered=rendered, encoded=encoded,
                    system=config["system_instruction"], user=user, question=case["question"],
                    options=case["options"], visual=False, engineering_only=True,
                )
                if not checks["overall_gate"]:
                    raise MeasurementImplementationError(f"control prompt invalid: {checks}")
                result = adapter.score_and_generate(
                    system=config["system_instruction"], user=user, image_path=None,
                    candidates=list(CANDIDATES), target=case["target"],
                    candidate_prefix=scoring["primary"]["candidate_prefix"],
                )
                record = {"schema_version": 1, "model_key": model_key, "case_id": case["case_id"],
                          "engineering_only": True, "result": result, "prompt_validation": checks}
                record["artifact_sha256"] = sha256_text(canonical_json(record))
                predictions.write(canonical_json(record) + "\n")
                predictions.flush()
                records.append(record)
            metadata = adapter.runtime_metadata()
        except Exception as error:  # noqa: BLE001 - null, failure and partial outputs are immutable
            failure = {"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()}
        finally:
            release = adapter.close()
    paths.extend([stderr_path, predictions_path])
    result = validate_contract_control(cases, records)
    if failure is not None:
        result.update({"status": "MEASUREMENT_IMPLEMENTATION_NO_GO", "overall_gate": False})
    runtime_path = destination / "runtime.json"
    _write_new(runtime_path, {"model_metadata": metadata, "release": release,
                             "runtime_seconds": time.perf_counter() - started, "failure": failure})
    paths.append(runtime_path)
    manifest = {
        "schema_version": 1, "model_key": model_key, **result, "model_metadata": metadata,
        "failure": failure, "release": release, "files": _files(root, paths),
        "renderer_adapter_repairs_used": 1 if model_key == "glm4_1v_9b" else 0,
        "frozen_scoring_inherited_unchanged": True, **runtime_metadata(root),
    }
    _write_new(manifest_path, manifest)
    return manifest


def aggregate_measurement_validation(root: Path = ROOT) -> dict[str, Any]:
    models = {}
    for key in MODEL_KEYS:
        models[key] = {}
        for kind in ("renderer_snapshots", "contract_control"):
            path = root / "artifacts/atomic_v2" / kind / key / "manifest.json"
            models[key][kind] = _read_verified_existing(path, root)
    complete = all(value and value.get("overall_gate") is True
                   for model in models.values() for value in model.values())
    return {
        "schema_version": 1, "models": models, "overall_gate": complete,
        "status": "MEASUREMENT_VALIDATION_PASS" if complete else "MEASUREMENT_IMPLEMENTATION_NO_GO",
        "scientific_accuracy_reported": False, "formal_model_outputs_produced": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("renderers", "contract-control", "aggregate"))
    parser.add_argument("--model-key", choices=MODEL_KEYS)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.operation == "aggregate":
        result = aggregate_measurement_validation(args.root)
    else:
        if args.model_key is None:
            parser.error("--model-key is required in an isolated model environment")
        function = validate_model_renderers if args.operation == "renderers" else run_contract_control
        result = function(args.model_key, args.root)
    print(canonical_json(result))
    if result.get("overall_gate") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
