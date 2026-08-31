from __future__ import annotations

import json
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from capability_gate.artifacts import (
    canonical_json,
    config_hash,
    read_jsonl,
    sha256_text,
    write_json,
)
from capability_gate.models.adapters import PromptPayload, TransformersVLM
from capability_gate.models.registry import frozen_registry, load_model_specs
from capability_gate.paths import ARTIFACTS, CONFIGS, ROOT


def _load_yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def _experiment_hash() -> str:
    return config_hash(
        [
            CONFIGS / "models.yaml",
            CONFIGS / "atomic_tasks.yaml",
            CONFIGS / "joint_screen.yaml",
            CONFIGS / "scoring.yaml",
        ]
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def _prompt(system: str, question: str, options: Sequence[str], schema: str) -> PromptPayload:
    user = (
        f"{question}\nAllowed answers in displayed order: {', '.join(options)}.\n"
        f"Answer schema: {schema}"
    )
    return PromptPayload(system=system, user=user)


def _image(relative_path: str | None) -> Image.Image | None:
    if not relative_path:
        return None
    with Image.open(ROOT / relative_path) as source:
        return source.convert("RGB")


def _prediction_row(
    spec: dict[str, Any],
    scene: dict[str, Any],
    payload: PromptPayload,
    result: dict[str, Any],
    *,
    mode: str,
    image_path: str | None,
    image_hash: str | None,
) -> dict[str, Any]:
    candidate_scores = result["candidate_scores"]
    normalized = {
        score["candidate"]: score["normalized_log_likelihood"] for score in candidate_scores
    }
    target_margin = normalized[scene["target"]] - sum(
        value for candidate, value in normalized.items() if candidate != scene["target"]
    ) / (len(normalized) - 1)
    return {
        "schema_version": 1,
        "model_key": spec["key"],
        "model": spec["model_id"],
        "revision": spec["revision"],
        "scene_id": scene.get("scene_id", scene.get("base_quartet_id")),
        "base_quartet_id": scene.get("base_quartet_id"),
        "condition": scene.get("condition"),
        "task": scene["task"],
        "mode": mode,
        "image_path": image_path,
        "image_hash": image_hash,
        "prompt": result["rendered_prompt"],
        "prompt_hash": sha256_text(result["rendered_prompt"]),
        "candidate_text": [score["candidate"] for score in candidate_scores],
        "candidate_token_ids": {
            score["candidate"]: score["candidate_token_ids"] for score in candidate_scores
        },
        "raw_log_likelihood": {
            score["candidate"]: score["raw_log_likelihood"] for score in candidate_scores
        },
        "normalized_log_likelihood": normalized,
        "candidate_ranking": result["candidate_ranking"],
        "top_answer": result["top_answer"],
        "target": scene["target"],
        "target_margin": target_margin,
        "correct_option_position": scene["correct_option_position"],
        "constrained_generation_answer": result["constrained_generation_answer"],
        "constrained_generation_token_ids": result["constrained_generation_token_ids"],
        "runtime_seconds": result["runtime_seconds"],
        "config_hash": _experiment_hash(),
        "prompt_contract": {"system": payload.system, "user": payload.user},
    }


def _classify_failure(error: BaseException) -> str:
    message = str(error).lower()
    compute_markers = (
        "out of memory",
        "cuda is unavailable",
        "cuda error",
        "not enough memory",
        "max_memory",
        "device-side",
    )
    return (
        "BLOCKED_BY_COMPUTE"
        if any(marker in message for marker in compute_markers)
        else "BLOCKED_BY_MODEL_ADAPTER"
    )


def run_engineering_smoke() -> dict[str, Any]:
    registry = frozen_registry()
    specs = load_model_specs()
    atomic_config = _load_yaml("atomic_tasks.yaml")
    scoring = _load_yaml("scoring.yaml")
    scenes = read_jsonl(ARTIFACTS / "data" / "engineering_smoke" / "scenes.jsonl")
    output = ARTIFACTS / "atomic" / "engineering_smoke_predictions.jsonl"
    existing_manifest = ARTIFACTS / "manifests" / "engineering_smoke.json"
    if existing_manifest.exists():
        return json.loads(existing_manifest.read_text(encoding="utf-8"))
    attempts = []
    model_status = {}
    if output.exists():
        raise RuntimeError("engineering smoke is one-shot and already has prediction output")

    for spec in specs:
        passed = False
        failures = []
        for attempt_index, profile in enumerate(spec["load_profiles"][:2], start=1):
            adapter = TransformersVLM(spec, profile)
            try:
                adapter.load()
                runtime_metadata = adapter.runtime_metadata()
                for scene in scenes:
                    payload = _prompt(
                        atomic_config["system_instruction"],
                        scene["question"],
                        scene["options"],
                        atomic_config["answer_schema"],
                    )
                    result = adapter.score_and_generate(
                        payload,
                        _image(scene["image_path"]),
                        atomic_config["candidates"],
                        scene["target"],
                        candidate_prefix=scoring["primary"]["candidate_prefix"],
                    )
                    row = _prediction_row(
                        spec,
                        scene,
                        payload,
                        result,
                        mode="engineering_smoke",
                        image_path=scene["image_path"],
                        image_hash=scene["image_sha256"],
                    )
                    row["engineering_attempt"] = attempt_index
                    _append_jsonl(output, row)
                passed = True
                model_status[spec["key"]] = {
                    "status": "ENGINEERING_SMOKE_PASS",
                    "successful_attempt": attempt_index,
                    "successful_profile": profile,
                    "runtime_metadata": runtime_metadata,
                    "scenes_completed": len(scenes),
                }
                attempts.append(
                    {
                        "model_key": spec["key"],
                        "attempt": attempt_index,
                        "status": "pass",
                        "profile": profile,
                    }
                )
                break
            except Exception as error:  # noqa: BLE001 - every smoke failure must be preserved
                failure = {
                    "model_key": spec["key"],
                    "attempt": attempt_index,
                    "status": "fail",
                    "profile": profile,
                    "failure_class": _classify_failure(error),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                attempts.append(failure)
                failures.append(failure)
            finally:
                adapter.close()
        if not passed:
            model_status[spec["key"]] = {
                "status": "ENGINEERING_SMOKE_FAIL",
                "failures": failures,
            }

    failures = [state for state in model_status.values() if state["status"].endswith("FAIL")]
    blocking_decision = None
    reason = None
    if failures:
        classes = {failure["failure_class"] for state in failures for failure in state["failures"]}
        blocking_decision = (
            "BLOCKED_BY_COMPUTE" if "BLOCKED_BY_COMPUTE" in classes else "BLOCKED_BY_MODEL_ADAPTER"
        )
        reason = "at least one of the three prospectively frozen families failed real forward/measurement smoke"
    manifest = {
        "schema_version": 1,
        "registry_config_hash": registry["config_hash"],
        "prompt_frozen_for_formal_qualification": not failures,
        "prompt_changed_from_accuracy": False,
        "model_status": model_status,
        "attempts": attempts,
        "blocking_decision": blocking_decision,
        "reason": reason,
    }
    write_json(ARTIFACTS / "manifests" / "engineering_smoke.json", manifest)
    return manifest


def _successful_profiles() -> dict[str, dict[str, Any]]:
    path = ARTIFACTS / "manifests" / "engineering_smoke.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("blocking_decision"):
        raise RuntimeError(state["blocking_decision"])
    return {key: value["successful_profile"] for key, value in state["model_status"].items()}


def run_atomic_qualification() -> dict[str, Any]:
    specs = load_model_specs()
    try:
        profiles = _successful_profiles()
    except RuntimeError as error:
        state = {
            "status": "NOT_RUN_BY_ENGINEERING_STOP_RULE",
            "upstream_decision": str(error),
            "model_forward_passes": 0,
        }
        write_json(ARTIFACTS / "manifests" / "atomic_run.json", state)
        return state
    atomic_config = _load_yaml("atomic_tasks.yaml")
    scoring = _load_yaml("scoring.yaml")
    scenes = read_jsonl(ARTIFACTS / "data" / "atomic_qualification" / "scenes.jsonl")
    output = ARTIFACTS / "atomic" / "predictions.jsonl"
    existing_manifest = ARTIFACTS / "manifests" / "atomic_run.json"
    if existing_manifest.exists():
        return json.loads(existing_manifest.read_text(encoding="utf-8"))
    if output.exists():
        raise RuntimeError("formal atomic qualification is one-shot and already exists")
    completed = 0
    for spec in specs:
        adapter = TransformersVLM(spec, profiles[spec["key"]])
        try:
            adapter.load()
            for scene in scenes:
                payload = _prompt(
                    atomic_config["system_instruction"],
                    scene["question"],
                    scene["options"],
                    atomic_config["answer_schema"],
                )
                result = adapter.score_and_generate(
                    payload,
                    _image(scene["image_path"]),
                    atomic_config["candidates"],
                    scene["target"],
                    candidate_prefix=scoring["primary"]["candidate_prefix"],
                )
                _append_jsonl(
                    output,
                    _prediction_row(
                        spec,
                        scene,
                        payload,
                        result,
                        mode="atomic_qualification",
                        image_path=scene["image_path"],
                        image_hash=scene["image_sha256"],
                    ),
                )
                completed += 1
        except Exception as error:
            failure = {
                "blocking_decision": _classify_failure(error),
                "model_key": spec["key"],
                "completed_prediction_rows_preserved": completed,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(ARTIFACTS / "manifests" / "atomic_runtime_failure.json", failure)
            raise
        finally:
            adapter.close()
    manifest = {"status": "complete", "rows": completed, "models": len(specs)}
    write_json(ARTIFACTS / "manifests" / "atomic_run.json", manifest)
    return manifest


def run_joint_screen() -> dict[str, Any]:
    existing_status = ARTIFACTS / "manifests" / "joint_screen_status.json"
    if existing_status.exists():
        return json.loads(existing_status.read_text(encoding="utf-8"))
    adjudication_path = ARTIFACTS / "atomic" / "adjudication.json"
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    if adjudication["decision"] != "ATOMIC_COHORT_GO":
        state = {
            "status": "NOT_RUN_BY_STOP_RULE",
            "upstream_decision": adjudication["decision"],
            "model_forward_passes": 0,
        }
        write_json(ARTIFACTS / "manifests" / "joint_screen_status.json", state)
        return state

    profiles = _successful_profiles()
    specs = {
        spec["key"]: spec
        for spec in load_model_specs()
        if adjudication["models"][spec["key"]]["label"] == "ATOMICALLY_QUALIFIED"
    }
    joint_config = _load_yaml("joint_screen.yaml")
    atomic_config = _load_yaml("atomic_tasks.yaml")
    scoring = _load_yaml("scoring.yaml")
    quartets = read_jsonl(ARTIFACTS / "data" / "joint_composition_screen" / "quartets.jsonl")
    atomic_scenes = read_jsonl(ARTIFACTS / "data" / "atomic_qualification" / "scenes.jsonl")
    output = ARTIFACTS / "joint" / "predictions.jsonl"
    retention_output = ARTIFACTS / "joint" / "atomic_retention_predictions.jsonl"
    if output.exists() or retention_output.exists():
        raise RuntimeError("joint screen is one-shot and output already exists")
    completed = 0
    for model_key, spec in specs.items():
        adapter = TransformersVLM(spec, profiles[model_key])
        cache: dict[str, dict[str, Any]] = {}
        try:
            adapter.load()
            for quartet in quartets:
                for condition in quartet["conditions"]:
                    logical = {
                        **condition,
                        "base_quartet_id": quartet["base_quartet_id"],
                        "task": "joint_composition",
                        "options": quartet["options"],
                        "psi_fixed_target": quartet["psi_fixed_target"],
                        "axis_map": quartet["axis_map"],
                    }
                    modes = {
                        "joint": (
                            condition["question"],
                            condition["image_path"],
                            condition["image_sha256"],
                        ),
                        "image_only": (
                            condition["question_without_premise"],
                            condition["image_path"],
                            condition["image_sha256"],
                        ),
                        "text_only": (condition["question"], None, None),
                        "question_only": (condition["question_without_premise"], None, None),
                    }
                    for mode, (question, image_path, image_hash) in modes.items():
                        payload = _prompt(
                            joint_config["system_instruction"],
                            question,
                            quartet["options"],
                            joint_config["answer_schema"],
                        )
                        cache_key = sha256_text(
                            canonical_json(
                                {
                                    "model": model_key,
                                    "mode": mode,
                                    "prompt": payload.user,
                                    "image_hash": image_hash,
                                }
                            )
                        )
                        if cache_key not in cache:
                            cache[cache_key] = adapter.score_and_generate(
                                payload,
                                _image(image_path),
                                joint_config["candidates"],
                                condition["target"],
                                candidate_prefix=scoring["primary"]["candidate_prefix"],
                            )
                        row = _prediction_row(
                            spec,
                            logical,
                            payload,
                            cache[cache_key],
                            mode=mode,
                            image_path=image_path,
                            image_hash=image_hash,
                        )
                        row.update(
                            {
                                "psi_fixed_target": quartet["psi_fixed_target"],
                                "axis_map": quartet["axis_map"],
                                "image_bit": condition["image_bit"],
                                "text_bit": condition["text_bit"],
                                "inference_cache_key": cache_key,
                            }
                        )
                        _append_jsonl(output, row)
                        completed += 1
            retention_system = (
                joint_config["system_instruction"]
                + " For direct control items, use the supplied direct evidence and a cardinal answer."
            )
            for scene in atomic_scenes:
                payload = _prompt(
                    retention_system,
                    scene["question"],
                    scene["options"],
                    atomic_config["answer_schema"],
                )
                result = adapter.score_and_generate(
                    payload,
                    _image(scene["image_path"]),
                    atomic_config["candidates"],
                    scene["target"],
                    candidate_prefix=scoring["primary"]["candidate_prefix"],
                )
                _append_jsonl(
                    retention_output,
                    _prediction_row(
                        spec,
                        scene,
                        payload,
                        result,
                        mode="joint_context_atomic_retention",
                        image_path=scene["image_path"],
                        image_hash=scene["image_sha256"],
                    ),
                )
        except Exception as error:
            failure = {
                "blocking_decision": _classify_failure(error),
                "model_key": model_key,
                "completed_prediction_rows_preserved": completed,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            write_json(ARTIFACTS / "manifests" / "joint_runtime_failure.json", failure)
            raise
        finally:
            adapter.close()
    state = {"status": "complete", "prediction_rows": completed, "qualified_models": list(specs)}
    write_json(ARTIFACTS / "manifests" / "joint_screen_status.json", state)
    return state
