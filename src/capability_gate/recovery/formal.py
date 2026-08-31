from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from capability_gate.artifacts import (
    canonical_json,
    config_hash,
    read_jsonl,
    sha256_text,
    write_json,
)
from capability_gate.paths import ARTIFACTS, CONFIGS, REPORTS, ROOT
from capability_gate.recovery.adapters import DESCRIPTORS
from capability_gate.recovery.environments import ENV_NAMES
from capability_gate.recovery.governance import authorize_atomic, authorize_joint
from capability_gate.recovery.smoke import WorkerSession

ATOMIC_ROOT = ARTIFACTS / "recovery_qualification/atomic"
JOINT_ROOT = ARTIFACTS / "recovery_qualification/joint"

COMPUTE_RUNTIME_SIGNATURES = (
    "cuda out of memory",
    "illegal memory access",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)


def classify_formal_runtime_failure(error: BaseException, stderr_text: str) -> str:
    """Separate CUDA/VRAM execution failures from adapter failures.

    This classification is engineering-only. It must never inspect predictions or
    task correctness.
    """

    evidence = f"{type(error).__name__}: {error}\n{stderr_text}".lower()
    if any(signature in evidence for signature in COMPUTE_RUNTIME_SIGNATURES):
        return "BLOCKED_BY_COMPUTE"
    return "BLOCKED_BY_MODEL_ADAPTER"


def _experiment_hash() -> str:
    return config_hash(
        [
            CONFIGS / "models.yaml",
            CONFIGS / "atomic_tasks.yaml",
            CONFIGS / "joint_screen.yaml",
            CONFIGS / "scoring.yaml",
        ]
    )


def _assert_governance_committed(report: Path) -> str:
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise RuntimeError("formal execution requires a clean, committed engineering decision")
    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "log",
            "-1",
            "--format=%H",
            "--",
            report.relative_to(ROOT).as_posix(),
        ],
        text=True,
    ).strip()
    if not commit:
        raise RuntimeError("engineering decision must be committed before formal output")
    return commit


def _worker_request(
    model_key: str,
    request_id: str,
    *,
    operation: str,
    system: str,
    user: str,
    candidates: list[str],
    target: str,
    image_path: str | None,
) -> dict[str, Any]:
    descriptor = DESCRIPTORS[model_key]
    return {
        "schema_version": 1,
        "request_id": request_id,
        "model_key": model_key,
        "model_revision": descriptor.revision,
        "processor_revision": descriptor.processor_revision,
        "image_path": image_path,
        "prompt": {"system": system, "user": user},
        "candidates": candidates,
        "target": target,
        "operation": operation,
    }


def _prompt_user(question: str, options: list[str], schema: str) -> str:
    return (
        f"{question}\nAllowed answers in displayed order: {', '.join(options)}.\n"
        f"Answer schema: {schema}"
    )


def _prediction_row(
    *,
    model_key: str,
    logical: dict[str, Any],
    response: dict[str, Any],
    mode: str,
    image_path: str | None,
    image_hash: str | None,
    prompt: dict[str, str],
) -> dict[str, Any]:
    scores = response["candidate_scores"]
    normalized = {score["candidate"]: score["normalized_log_likelihood"] for score in scores}
    metadata = response["model_metadata"]
    target = logical["target"]
    target_margin = normalized[target] - sum(
        value for candidate, value in normalized.items() if candidate != target
    ) / (len(normalized) - 1)
    return {
        "schema_version": 2,
        "model_key": model_key,
        "model": DESCRIPTORS[model_key].model_id,
        "revision": DESCRIPTORS[model_key].revision,
        "scene_id": logical.get("scene_id", logical.get("base_quartet_id")),
        "base_quartet_id": logical.get("base_quartet_id"),
        "condition": logical.get("condition"),
        "task": logical["task"],
        "mode": mode,
        "image_path": image_path,
        "image_hash": image_hash,
        "prompt": metadata["rendered_prompt"],
        "prompt_hash": sha256_text(metadata["rendered_prompt"]),
        "candidate_text": [score["candidate"] for score in scores],
        "candidate_token_ids": {
            score["candidate"]: score["candidate_token_ids"] for score in scores
        },
        "raw_log_likelihood": {score["candidate"]: score["raw_log_likelihood"] for score in scores},
        "normalized_log_likelihood": normalized,
        "candidate_ranking": metadata["candidate_ranking"],
        "top_answer": metadata["top_answer"],
        "target": target,
        "target_margin": target_margin,
        "correct_option_position": logical["correct_option_position"],
        "constrained_generation_answer": response["constrained_answer"],
        "constrained_generation_token_ids": metadata["constrained_generation_token_ids"],
        "runtime_seconds": response["runtime"]["forward_and_generation_seconds"],
        "config_hash": _experiment_hash(),
        "prompt_contract": prompt,
        "worker_artifact_hash": response["artifact_hash"],
    }


def _write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def run_atomic_qualification_v2() -> dict[str, Any]:
    existing = ATOMIC_ROOT / "run_status.json"
    if existing.exists():
        return json.loads(existing.read_text(encoding="utf-8"))
    decision_path = REPORTS / "recovery/engineering_cohort_decision.yaml"
    engineering = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
    try:
        model_keys = authorize_atomic(engineering)
    except RuntimeError:
        state = {
            "status": "NOT_RUN_BY_ENGINEERING_GATE",
            "upstream_decision": engineering["decision"],
            "model_forward_passes": 0,
            "authorized_models": [],
        }
        write_json(existing, state)
        return state
    decision_commit = _assert_governance_committed(decision_path)
    atomic = yaml.safe_load((CONFIGS / "atomic_tasks.yaml").read_text(encoding="utf-8"))
    scenes = read_jsonl(ARTIFACTS / "data/atomic_qualification/scenes.jsonl")
    if len(scenes) != 256:
        raise RuntimeError("frozen Atomic qualification must contain 256 scenes")
    completed = 0
    models_completed = []
    for model_key in model_keys:
        output = ATOMIC_ROOT / f"predictions/{ENV_NAMES[model_key]}.jsonl"
        if output.exists():
            raise RuntimeError(f"Atomic v2 output already exists: {output}")
        stderr_path = ATOMIC_ROOT / f"runtime/{ENV_NAMES[model_key]}.stderr.log"
        session = WorkerSession(model_key, stderr_path)
        try:
            load = session.request(
                _worker_request(
                    model_key,
                    f"atomic-v2-{model_key}-load",
                    operation="load_preflight",
                    system=atomic["system_instruction"],
                    user="Atomic v2 frozen-load preflight.",
                    candidates=list(atomic["candidates"]),
                    target=atomic["candidates"][0],
                    image_path=None,
                )
            )
            if load["status"] != "LOAD_PREFLIGHT_PASS":
                raise RuntimeError(load["traceback"] or load["error_class"])
            for index, scene in enumerate(scenes):
                user = _prompt_user(
                    scene["question"], list(scene["options"]), atomic["answer_schema"]
                )
                prompt = {"system": atomic["system_instruction"], "user": user}
                image_path = str(ROOT / scene["image_path"]) if scene["image_path"] else None
                response = session.request(
                    _worker_request(
                        model_key,
                        f"atomic-v2-{model_key}-{index:03d}",
                        operation="score_and_generate",
                        system=prompt["system"],
                        user=prompt["user"],
                        candidates=list(atomic["candidates"]),
                        target=scene["target"],
                        image_path=image_path,
                    )
                )
                if response["status"] != "SCORE_AND_GENERATE_PASS":
                    raise RuntimeError(response["traceback"] or response["error_class"])
                _write_row(
                    output,
                    _prediction_row(
                        model_key=model_key,
                        logical=scene,
                        response=response,
                        mode="atomic_qualification_v2",
                        image_path=scene["image_path"],
                        image_hash=scene["image_sha256"],
                        prompt=prompt,
                    ),
                )
                completed += 1
            models_completed.append(model_key)
        except Exception as error:  # noqa: BLE001 - preserve partial formal sequence
            session.stderr_handle.flush()
            stderr_text = stderr_path.read_text(encoding="utf-8")
            failure = {
                "status": "FORMAL_ATOMIC_RUNTIME_FAIL",
                "block_class": classify_formal_runtime_failure(error, stderr_text),
                "model_key": model_key,
                "completed_prediction_rows_preserved": completed,
                "error": f"{type(error).__name__}: {error}",
                "engineering_decision_commit": decision_commit,
            }
            write_json(ATOMIC_ROOT / "runtime_failure.json", failure)
            write_json(existing, failure)
            return failure
        finally:
            release = session.close()
            write_json(
                ATOMIC_ROOT / f"runtime/{ENV_NAMES[model_key]}.json",
                {"model_key": model_key, "release": release},
            )
    state = {
        "status": "complete",
        "rows": completed,
        "models": models_completed,
        "engineering_decision_commit": decision_commit,
        "formal_contract_modified": False,
    }
    write_json(existing, state)
    return state


def run_joint_screen_v2() -> dict[str, Any]:
    existing = JOINT_ROOT / "run_status.json"
    if existing.exists():
        return json.loads(existing.read_text(encoding="utf-8"))
    atomic_path = ATOMIC_ROOT / "adjudication.json"
    atomic = json.loads(atomic_path.read_text(encoding="utf-8"))
    labels = {key: value["label"] for key, value in atomic.get("models", {}).items()}
    try:
        model_keys = authorize_joint(labels)
    except RuntimeError:
        state = {
            "status": "NOT_RUN_BY_ATOMIC_GATE",
            "upstream_decision": atomic["decision"],
            "model_forward_passes": 0,
            "authorized_models": [],
        }
        write_json(existing, state)
        _write_joint_not_run(state)
        return state
    atomic_commit = _assert_governance_committed(REPORTS / "recovery/atomic_qualification_v2.md")
    joint = yaml.safe_load((CONFIGS / "joint_screen.yaml").read_text(encoding="utf-8"))
    atomic_config = yaml.safe_load((CONFIGS / "atomic_tasks.yaml").read_text(encoding="utf-8"))
    quartets = read_jsonl(ARTIFACTS / "data/joint_composition_screen/quartets.jsonl")
    atomic_scenes = read_jsonl(ARTIFACTS / "data/atomic_qualification/scenes.jsonl")
    if len(quartets) != 128:
        raise RuntimeError("frozen Joint screen must contain 128 quartets")
    completed = 0
    for model_key in model_keys:
        output = JOINT_ROOT / f"predictions/{ENV_NAMES[model_key]}.jsonl"
        retention_output = JOINT_ROOT / f"retention/{ENV_NAMES[model_key]}.jsonl"
        stderr_path = JOINT_ROOT / f"runtime/{ENV_NAMES[model_key]}.stderr.log"
        session = WorkerSession(model_key, stderr_path)
        cache: dict[str, dict[str, Any]] = {}
        try:
            load = session.request(
                _worker_request(
                    model_key,
                    f"joint-v2-{model_key}-load",
                    operation="load_preflight",
                    system=joint["system_instruction"],
                    user="Joint v2 frozen-load preflight.",
                    candidates=list(joint["candidates"]),
                    target=joint["candidates"][0],
                    image_path=None,
                )
            )
            if load["status"] != "LOAD_PREFLIGHT_PASS":
                raise RuntimeError(load["traceback"] or load["error_class"])
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
                        user = _prompt_user(
                            question, list(quartet["options"]), joint["answer_schema"]
                        )
                        prompt = {"system": joint["system_instruction"], "user": user}
                        cache_key = sha256_text(
                            canonical_json(
                                {
                                    "prompt": prompt,
                                    "image_path": image_path,
                                    "candidates": joint["candidates"],
                                }
                            )
                        )
                        if cache_key not in cache:
                            cache[cache_key] = session.request(
                                _worker_request(
                                    model_key,
                                    f"joint-v2-{model_key}-{completed:06d}",
                                    operation="score_and_generate",
                                    system=prompt["system"],
                                    user=prompt["user"],
                                    candidates=list(joint["candidates"]),
                                    target=condition["target"],
                                    image_path=str(ROOT / image_path) if image_path else None,
                                )
                            )
                        response = cache[cache_key]
                        if response["status"] != "SCORE_AND_GENERATE_PASS":
                            raise RuntimeError(response["traceback"] or response["error_class"])
                        row = _prediction_row(
                            model_key=model_key,
                            logical=logical,
                            response=response,
                            mode=mode,
                            image_path=image_path,
                            image_hash=image_hash,
                            prompt=prompt,
                        )
                        row.update(
                            {
                                "psi_fixed_target": quartet["psi_fixed_target"],
                                "axis_map": quartet["axis_map"],
                                "image_bit": condition["image_bit"],
                                "text_bit": condition["text_bit"],
                                "inference_cache_key": cache_key,
                                "template_id": quartet["template_id"],
                            }
                        )
                        _write_row(output, row)
                        completed += 1
            retention_system = (
                joint["system_instruction"]
                + " For direct control items, use the supplied direct evidence and a cardinal answer."
            )
            for index, scene in enumerate(atomic_scenes):
                user = _prompt_user(
                    scene["question"], list(scene["options"]), atomic_config["answer_schema"]
                )
                prompt = {"system": retention_system, "user": user}
                response = session.request(
                    _worker_request(
                        model_key,
                        f"joint-v2-retention-{model_key}-{index:03d}",
                        operation="score_and_generate",
                        system=prompt["system"],
                        user=prompt["user"],
                        candidates=list(atomic_config["candidates"]),
                        target=scene["target"],
                        image_path=str(ROOT / scene["image_path"]) if scene["image_path"] else None,
                    )
                )
                if response["status"] != "SCORE_AND_GENERATE_PASS":
                    raise RuntimeError(response["traceback"] or response["error_class"])
                _write_row(
                    retention_output,
                    _prediction_row(
                        model_key=model_key,
                        logical=scene,
                        response=response,
                        mode="joint_context_atomic_retention",
                        image_path=scene["image_path"],
                        image_hash=scene["image_sha256"],
                        prompt=prompt,
                    ),
                )
        except Exception as error:  # noqa: BLE001 - preserve partial joint sequence
            session.stderr_handle.flush()
            stderr_text = stderr_path.read_text(encoding="utf-8")
            failure = {
                "status": "FORMAL_JOINT_RUNTIME_FAIL",
                "block_class": classify_formal_runtime_failure(error, stderr_text),
                "model_key": model_key,
                "completed_prediction_rows_preserved": completed,
                "error": f"{type(error).__name__}: {error}",
                "atomic_decision_commit": atomic_commit,
            }
            write_json(JOINT_ROOT / "runtime_failure.json", failure)
            write_json(existing, failure)
            return failure
        finally:
            release = session.close()
            write_json(
                JOINT_ROOT / f"runtime/{ENV_NAMES[model_key]}.json",
                {"model_key": model_key, "release": release},
            )
    state = {
        "status": "complete",
        "prediction_rows": completed,
        "qualified_models": model_keys,
        "atomic_decision_commit": atomic_commit,
        "formal_contract_modified": False,
    }
    write_json(existing, state)
    return state


def _write_joint_not_run(state: dict[str, Any]) -> None:
    path = REPORTS / "recovery/joint_composition_screen_v2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Joint Composition Screen v2\n\n"
        "**NOT_RUN_BY_ATOMIC_GATE**\n\n"
        f"Upstream gate: `{state['upstream_decision']}`.\n",
        encoding="utf-8",
    )
