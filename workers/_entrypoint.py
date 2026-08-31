from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _dependency_check(model_key: str) -> int:
    from capability_gate.recovery.dependencies import dependency_preflight

    result = dependency_preflight(model_key)
    print(_json_line(result), flush=True)
    return 0 if result["status"] == "DEPENDENCY_PREFLIGHT_PASS" else 2


def _peak_ram_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset
    except (AttributeError, ImportError):
        return None


def _failure_class(error: BaseException) -> str:
    from capability_gate.recovery.adapters import (
        DependencyPreflightError,
        FrozenRevisionError,
        MeasurementImplementationError,
        MetaTensorError,
    )

    if isinstance(error, DependencyPreflightError):
        return "BLOCKED_BY_DEPENDENCY"
    if isinstance(error, MeasurementImplementationError):
        return "MEASUREMENT_IMPLEMENTATION_FAIL"
    if isinstance(error, MetaTensorError):
        return "BLOCKED_BY_MODEL_ADAPTER"
    if isinstance(error, FrozenRevisionError):
        return "BLOCKED_BY_MODEL_ADAPTER"
    message = str(error).lower()
    compute_markers = (
        "out of memory",
        "cuda is unavailable",
        "cuda error",
        "cublas",
        "not enough memory",
        "allocation",
    )
    if any(marker in message for marker in compute_markers):
        return "BLOCKED_BY_COMPUTE"
    if "only tensors of floating point dtype can require gradients" in message:
        return "MEASUREMENT_IMPLEMENTATION_FAIL"
    return "BLOCKED_BY_MODEL_ADAPTER"


def _response(
    *,
    request_id: str | None,
    status: str,
    model_metadata: dict[str, Any] | None = None,
    candidate_scores: list[dict[str, Any]] | None = None,
    constrained_answer: str | None = None,
    runtime: dict[str, Any] | None = None,
    peak_vram: float | None = None,
    resolved_device_map: dict[str, Any] | None = None,
    error_class: str | None = None,
    error_traceback: str | None = None,
) -> dict[str, Any]:
    from capability_gate.recovery.adapters import response_hash

    value = {
        "schema_version": 1,
        "request_id": request_id,
        "status": status,
        "model_metadata": model_metadata or {},
        "candidate_scores": candidate_scores or [],
        "constrained_answer": constrained_answer,
        "runtime": runtime or {},
        "peak_vram": peak_vram,
        "resolved_device_map": resolved_device_map,
        "artifact_hash": None,
        "error_class": error_class,
        "traceback": error_traceback,
    }
    value["artifact_hash"] = response_hash(value)
    return value


def _serve(model_key: str) -> int:
    from capability_gate.recovery.adapters import DESCRIPTORS, adapter_for
    from capability_gate.recovery.contract import WorkerRequest, validate_worker_response
    from capability_gate.recovery.dependencies import dependency_preflight

    descriptor = DESCRIPTORS[model_key]
    adapter = None
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            started = time.perf_counter()
            request_id = None
            try:
                request = WorkerRequest.from_dict(json.loads(line))
                request_id = request.request_id
                if request.model_key != model_key:
                    raise ValueError(
                        f"worker {model_key} refuses request for {request.model_key}"
                    )
                if request.model_revision != descriptor.revision:
                    raise ValueError("worker refuses non-frozen model revision")
                if request.processor_revision != descriptor.processor_revision:
                    raise ValueError("worker refuses non-frozen processor revision")
                if request.operation == "dependency_preflight":
                    preflight = dependency_preflight(model_key)
                    response = _response(
                        request_id=request_id,
                        status=preflight["status"],
                        model_metadata={"dependency_preflight": preflight},
                        runtime={
                            "operation": request.operation,
                            "seconds": time.perf_counter() - started,
                            "peak_ram_bytes": _peak_ram_bytes(),
                        },
                        error_class=(
                            None
                            if preflight["status"] == "DEPENDENCY_PREFLIGHT_PASS"
                            else "BLOCKED_BY_DEPENDENCY"
                        ),
                    )
                else:
                    if adapter is None:
                        adapter = adapter_for(model_key)
                    if request.operation == "processor_preflight":
                        adapter.load_processor()
                        response = _response(
                            request_id=request_id,
                            status="PROCESSOR_PREFLIGHT_PASS",
                            model_metadata={
                                "model_key": model_key,
                                "processor_revision": descriptor.processor_revision,
                                "processor_class": type(adapter.processor).__name__,
                                "processor_class_verified": (
                                    type(adapter.processor).__name__
                                    == descriptor.expected_processor_class
                                ),
                            },
                            runtime={
                                "operation": request.operation,
                                "seconds": time.perf_counter() - started,
                                "peak_ram_bytes": _peak_ram_bytes(),
                            },
                        )
                    else:
                        if adapter.model is None:
                            adapter.load()
                            adapter.verify_cached_weights()
                        metadata = adapter.runtime_metadata()
                        if request.operation == "load_preflight":
                            response = _response(
                                request_id=request_id,
                                status="LOAD_PREFLIGHT_PASS",
                                model_metadata=metadata,
                                runtime={
                                    "operation": request.operation,
                                    "seconds": time.perf_counter() - started,
                                    "load_seconds": adapter.load_seconds,
                                    "peak_ram_bytes": _peak_ram_bytes(),
                                },
                                peak_vram=metadata["peak_vram_bytes"],
                                resolved_device_map=metadata["resolved_device_map"],
                            )
                        elif request.operation == "score_and_generate":
                            result = adapter.score_and_generate(
                                system=request.prompt["system"],
                                user=request.prompt["user"],
                                image_path=request.image_path,
                                candidates=request.candidates,
                                target=request.target,
                            )
                            metadata = adapter.runtime_metadata()
                            response = _response(
                                request_id=request_id,
                                status="SCORE_AND_GENERATE_PASS",
                                model_metadata={
                                    **metadata,
                                    "rendered_prompt": result["rendered_prompt"],
                                    "candidate_ranking": result["candidate_ranking"],
                                    "top_answer": result["top_answer"],
                                    "target_margin": result["target_margin"],
                                    "constrained_generation_token_ids": result[
                                        "constrained_generation_token_ids"
                                    ],
                                    "visual_input_keys": result["visual_input_keys"],
                                    "vision_forward_observed": result[
                                        "vision_forward_observed"
                                    ],
                                    "text_only_forward": result["text_only_forward"],
                                },
                                candidate_scores=result["candidate_scores"],
                                constrained_answer=result["constrained_answer"],
                                runtime={
                                    "operation": request.operation,
                                    "seconds": time.perf_counter() - started,
                                    "forward_and_generation_seconds": result["runtime_seconds"],
                                    "load_seconds": adapter.load_seconds,
                                    "peak_ram_bytes": _peak_ram_bytes(),
                                },
                                peak_vram=metadata["peak_vram_bytes"],
                                resolved_device_map=metadata["resolved_device_map"],
                            )
                        else:
                            raise ValueError(f"unsupported worker operation: {request.operation}")
            except Exception as error:  # noqa: BLE001 - worker must serialize all failures
                response = _response(
                    request_id=request_id,
                    status="WORKER_OPERATION_FAIL",
                    runtime={
                        "seconds": time.perf_counter() - started,
                        "peak_ram_bytes": _peak_ram_bytes(),
                    },
                    error_class=_failure_class(error),
                    error_traceback=traceback.format_exc(),
                )
            validate_worker_response(response)
            print(_json_line(response), flush=True)
    finally:
        if adapter is not None:
            adapter.close()
    return 0


def main(model_key: str) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-dependencies", action="store_true")
    args = parser.parse_args()
    if args.check_dependencies:
        return _dependency_check(model_key)
    return _serve(model_key)
