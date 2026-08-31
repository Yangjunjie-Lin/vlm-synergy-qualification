from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

from capability_gate.artifacts import canonical_json, write_json
from capability_gate.paths import ARTIFACTS, ROOT
from capability_gate.recovery.adapters import DESCRIPTORS, response_hash
from capability_gate.recovery.contract import RESPONSE_FIELDS, validate_worker_response
from capability_gate.recovery.environments import ENV_NAMES, worker_python, worker_script
from capability_gate.recovery.smoke_data import generate_recovery_smoke_scenes

RECOVERY = ARTIFACTS / "engineering_recovery"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def _gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.splitlines()[0].strip())
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


class WorkerSession:
    def __init__(self, model_key: str, stderr_path: Path):
        self.model_key = model_key
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_handle = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(worker_python(model_key)), str(worker_script(model_key))],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_handle,
            text=True,
            bufsize=1,
        )

    def request(self, value: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        self.process.stdin.write(canonical_json(value) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"worker {self.model_key} exited without a response: {self.process.poll()}"
            )
        response = json.loads(line)
        validate_worker_response(response)
        if response_hash(response) != response["artifact_hash"]:
            raise RuntimeError("worker response artifact hash mismatch")
        return response

    def close(self) -> dict[str, Any]:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            returncode = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            returncode = self.process.wait(timeout=30)
        finally:
            self.stderr_handle.close()
        return {"worker_returncode": returncode, "gpu_used_mib_after_worker": _gpu_used_mib()}


def _request(
    model_key: str,
    request_id: str,
    operation: str,
    *,
    image_path: str | None = None,
    user: str = "Engineering recovery preflight.",
    target: str = "north",
) -> dict[str, Any]:
    descriptor = DESCRIPTORS[model_key]
    return {
        "schema_version": 1,
        "request_id": request_id,
        "model_key": model_key,
        "model_revision": descriptor.revision,
        "processor_revision": descriptor.processor_revision,
        "image_path": image_path,
        "prompt": {
            "system": (
                "Engineering transport validation only. Return exactly one allowed lowercase "
                "direction word and no explanation."
            ),
            "user": user + "\nAllowed answers in displayed order: north, south, east, west.\n"
            "Answer schema: Return exactly one allowed lowercase direction word and nothing else.",
        },
        "candidates": ["north", "south", "east", "west"],
        "target": target,
        "operation": operation,
    }


def _score_response_valid(response: dict[str, Any], allowed: set[str]) -> bool:
    scores = response["candidate_scores"]
    return (
        response["status"] == "SCORE_AND_GENERATE_PASS"
        and len(scores) == len(allowed)
        and {score["candidate"] for score in scores} == allowed
        and all(score["candidate_token_count"] > 0 for score in scores)
        and all(math.isfinite(score["normalized_log_likelihood"]) for score in scores)
        and response["constrained_answer"] in allowed
        and set(response) == RESPONSE_FIELDS
    )


def _deterministic_agreement(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_scores = {
        value["candidate"]: value["normalized_log_likelihood"]
        for value in first["candidate_scores"]
    }
    second_scores = {
        value["candidate"]: value["normalized_log_likelihood"]
        for value in second["candidate_scores"]
    }
    score_agreement = set(first_scores) == set(second_scores) and all(
        math.isclose(first_scores[key], second_scores[key], rel_tol=0.0, abs_tol=1e-6)
        for key in first_scores
    )
    metadata_agreement = (
        first["model_metadata"].get("candidate_ranking")
        == second["model_metadata"].get("candidate_ranking")
        and first["constrained_answer"] == second["constrained_answer"]
    )
    return 1.0 if score_agreement and metadata_agreement else 0.0


def _load_gate(metadata: dict[str, Any]) -> dict[str, bool]:
    materialization = metadata.get("materialization", {})
    placement = metadata.get("placement_inventory", {})
    return {
        "exact_revision_verified": metadata.get("model_revision")
        == DESCRIPTORS[metadata["model_key"]].revision,
        "weight_hashes_verified": metadata.get("weight_hashes_verified") is True,
        "native_model_class_verified": metadata.get("native_model_class_verified") is True,
        "processor_class_verified": metadata.get("processor_class_verified") is True,
        "no_meta_parameters": materialization.get("meta_parameter_count") == 0,
        "no_meta_buffers": materialization.get("meta_buffer_count") == 0,
        "no_missing_weights": materialization.get("missing_weight_count") == 0,
        "no_unexpected_weights": materialization.get("unexpected_weight_count") == 0,
        "no_silent_fallback": (
            metadata.get("placement_policy") == "EXPLICIT_SINGLE_GPU_4BIT_NO_CPU_DISK_OFFLOAD"
            and not placement.get("disk_offloaded_modules")
            and metadata.get("resolved_device_map") in ({"": "0"}, {"": "cuda:0"})
        ),
    }


def run_adapter_recovery_smoke() -> dict[str, Any]:
    historical_path = RECOVERY / "manifests/historical_block_verification.json"
    diagnosis_path = RECOVERY / "manifests/root_cause_diagnosis.json"
    environment_path = RECOVERY / "manifests/environment_verification.json"
    for required in (historical_path, diagnosis_path, environment_path):
        if not required.exists():
            raise RuntimeError(
                f"required recovery preflight is absent: {required.relative_to(ROOT)}"
            )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    environments = json.loads(environment_path.read_text(encoding="utf-8"))
    if not historical["overall_gate"] or not diagnosis["overall_gate"]:
        raise RuntimeError("historical or root-cause preflight gate failed")
    existing = RECOVERY / "manifests/engineering_recovery_smoke.json"
    if existing.exists():
        return json.loads(existing.read_text(encoding="utf-8"))
    scenes = generate_recovery_smoke_scenes()
    if len(scenes) != 12:
        raise RuntimeError("engineering recovery smoke must contain exactly 12 scenes")
    allowed = {"north", "south", "east", "west"}
    model_results = {}
    for model_key in DESCRIPTORS:
        env_name = ENV_NAMES[model_key]
        load_path = RECOVERY / f"load_attempts/{env_name}.jsonl"
        prediction_path = RECOVERY / f"smoke_predictions/{env_name}.jsonl"
        runtime_path = RECOVERY / f"runtime_metadata/{env_name}.json"
        stderr_path = RECOVERY / f"runtime_metadata/{env_name}.stderr.log"
        env_record = environments["models"][model_key]
        if env_record["status"] != "DEPENDENCY_PREFLIGHT_PASS":
            model_results[model_key] = {
                "status": "BLOCKED_BY_DEPENDENCY",
                "reason": env_record["preflight"],
                "load_attempts": 0,
                "scientific_capability_conclusion": False,
            }
            continue
        session = WorkerSession(model_key, stderr_path)
        started = time.perf_counter()
        peak_vram = 0
        responses: list[dict[str, Any]] = []
        close_state: dict[str, Any] = {}
        try:
            load_response = session.request(
                _request(model_key, f"{model_key}-load-1", "load_preflight")
            )
            _append_jsonl(load_path, load_response)
            if load_response["status"] != "LOAD_PREFLIGHT_PASS":
                model_results[model_key] = {
                    "status": load_response["error_class"] or "BLOCKED_BY_MODEL_ADAPTER",
                    "reason": load_response["traceback"],
                    "load_attempts": 1,
                    "scientific_capability_conclusion": False,
                }
                continue
            metadata = load_response["model_metadata"]
            load_gates = _load_gate(metadata)
            peak_vram = max(peak_vram, int(load_response["peak_vram"] or 0))
            for index, scene in enumerate(scenes):
                response = session.request(
                    _request(
                        model_key,
                        f"{model_key}-visual-{index:02d}",
                        "score_and_generate",
                        image_path=str(ROOT / scene["image_path"]),
                        user=scene["prompt"],
                        target=scene["target"],
                    )
                )
                _append_jsonl(prediction_path, response)
                responses.append(response)
                peak_vram = max(peak_vram, int(response["peak_vram"] or 0))
                if not _score_response_valid(response, allowed):
                    break
            text_response = None
            rerun_response = None
            if len(responses) == len(scenes) and all(
                _score_response_valid(response, allowed) for response in responses
            ):
                text_response = session.request(
                    _request(
                        model_key,
                        f"{model_key}-text-only",
                        "score_and_generate",
                        user=(
                            "Engineering text-only transport check. The control token says north. "
                            "Return one allowed answer."
                        ),
                    )
                )
                _append_jsonl(prediction_path, text_response)
                rerun_scene = scenes[0]
                rerun_response = session.request(
                    _request(
                        model_key,
                        f"{model_key}-deterministic-rerun",
                        "score_and_generate",
                        image_path=str(ROOT / rerun_scene["image_path"]),
                        user=rerun_scene["prompt"],
                        target=rerun_scene["target"],
                    )
                )
                _append_jsonl(prediction_path, rerun_response)
            visual_gate = (
                len(responses) == 12
                and all(_score_response_valid(response, allowed) for response in responses)
                and all(
                    response["model_metadata"].get("vision_forward_observed") is True
                    for response in responses
                )
            )
            text_gate = bool(
                text_response
                and _score_response_valid(text_response, allowed)
                and text_response["model_metadata"].get("text_only_forward") is True
            )
            determinism = (
                _deterministic_agreement(responses[0], rerun_response)
                if responses and rerun_response
                else 0.0
            )
            artifact_completeness = (
                1.0
                if responses
                and text_response
                and rerun_response
                and all(set(response) == RESPONSE_FIELDS for response in responses)
                and set(text_response) == RESPONSE_FIELDS
                and set(rerun_response) == RESPONSE_FIELDS
                else 0.0
            )
            gates = {
                **load_gates,
                "real_visual_forward": visual_gate,
                "text_only_forward": text_gate,
                "candidate_tokens_nonempty": all(
                    score["candidate_token_count"] > 0
                    for response in responses
                    for score in response["candidate_scores"]
                ),
                "cll_finite": all(
                    math.isfinite(score["normalized_log_likelihood"])
                    for response in responses
                    for score in response["candidate_scores"]
                ),
                "constrained_generation_allowed": all(
                    response["constrained_answer"] in allowed for response in responses
                ),
                "deterministic_rerun_agreement": determinism == 1.0,
                "artifact_completeness": artifact_completeness == 1.0,
                "runtime_recorded": time.perf_counter() > started,
                "peak_vram_recorded": peak_vram > 0,
            }
            status = (
                "ENGINEERING_RECOVERY_PASS"
                if all(gates.values())
                else "MEASUREMENT_IMPLEMENTATION_FAIL"
            )
            model_results[model_key] = {
                "status": status,
                "load_attempts": 1,
                "gates": gates,
                "deterministic_rerun_agreement": determinism,
                "artifact_completeness": artifact_completeness,
                "scene_count": len(responses),
                "peak_vram_bytes": peak_vram,
                "runtime_seconds": time.perf_counter() - started,
                "model_metadata": metadata,
                "scientific_capability_conclusion": False,
                "engineering_scene_accuracy_reported": False,
            }
        except Exception as error:  # noqa: BLE001 - isolate and preserve worker failure
            model_results[model_key] = {
                "status": "BLOCKED_BY_MODEL_ADAPTER",
                "reason": f"{type(error).__name__}: {error}",
                "load_attempts": 1,
                "scientific_capability_conclusion": False,
            }
        finally:
            close_state = session.close()
            write_json(
                runtime_path,
                {
                    "schema_version": 1,
                    "model_key": model_key,
                    "sequential_worker_release": close_state,
                    "total_runtime_seconds": time.perf_counter() - started,
                },
            )
            if model_key in model_results:
                model_results[model_key]["worker_release"] = close_state
    result = {
        "schema_version": 1,
        "scene_count": len(scenes),
        "engineering_only": True,
        "engineering_scene_accuracy_reported": False,
        "formal_atomic_outputs_before_smoke": 0,
        "formal_joint_outputs_before_smoke": 0,
        "model_results": model_results,
        "activation_patching_executed": False,
        "fourth_model_used": False,
        "scientific_capability_conclusion": False,
    }
    write_json(existing, result)
    return result
