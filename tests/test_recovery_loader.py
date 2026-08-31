from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from capability_gate.recovery.adapters import (
    DESCRIPTORS,
    QwenRecoveryAdapter,
    adapter_for,
    inspect_materialization,
)
from capability_gate.recovery.placement import validate_no_auto_offload
from capability_gate.recovery.smoke import _repair_retry_models


class _HookableModule:
    def __init__(self) -> None:
        self.hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return SimpleNamespace(remove=lambda: None)


def test_nested_native_visual_module_is_accepted_for_forward_proof() -> None:
    adapter = QwenRecoveryAdapter()
    visual = _HookableModule()
    adapter.model = SimpleNamespace(named_modules=lambda: [("model.visual", visual)])

    adapter._install_vision_hooks()

    assert visual.hook is not None
    visual.hook(visual, (), None)
    assert adapter._vision_observed is True


def test_repair_retry_is_limited_to_exact_hook_failure() -> None:
    manifest = {
        "model_results": {
            "qwen2_5_vl_7b": {
                "status": "MEASUREMENT_IMPLEMENTATION_FAIL",
                "reason": "trace: no vision module found for forward proof hook",
            },
            "glm4_1v_9b": {
                "status": "BLOCKED_BY_COMPUTE",
                "reason": "CUDA out of memory",
            },
            "phi4_multimodal_5_6b": {
                "status": "ENGINEERING_RECOVERY_PASS",
            },
        }
    }

    assert _repair_retry_models(manifest) == {"qwen2_5_vl_7b"}


def test_native_model_loader_classes_are_frozen() -> None:
    assert DESCRIPTORS["qwen2_5_vl_7b"].loader_class == "Qwen2_5_VLForConditionalGeneration"
    assert DESCRIPTORS["glm4_1v_9b"].loader_class == "Glm4vForConditionalGeneration"
    assert DESCRIPTORS["phi4_multimodal_5_6b"].loader_class == "AutoModelForCausalLM"
    assert DESCRIPTORS["phi4_multimodal_5_6b"].expected_model_class == "Phi4MMForCausalLM"


def test_meta_parameters_and_buffers_are_detected_before_forward() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.empty(2, device="meta")))
    model.register_buffer("cache", torch.empty(1, device="meta"))
    result = inspect_materialization(model, {"missing_keys": [], "unexpected_keys": []})
    assert result["meta_parameter_count"] == 1
    assert result["meta_buffer_count"] == 1


def test_historical_auto_offload_is_rejected() -> None:
    quantization = SimpleNamespace(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
    with pytest.raises(ValueError):
        validate_no_auto_offload(
            {
                "device_map": "auto",
                "max_memory": {0: "4GiB", "cpu": "24GiB"},
                "quantization_config": quantization,
            }
        )


def test_worker_failure_does_not_pollute_next_request() -> None:
    root = Path(__file__).resolve().parents[1]
    descriptor = DESCRIPTORS["qwen2_5_vl_7b"]
    base = {
        "schema_version": 1,
        "model_key": descriptor.key,
        "model_revision": descriptor.revision,
        "processor_revision": descriptor.processor_revision,
        "image_path": None,
        "prompt": {"system": "s", "user": "u"},
        "candidates": ["north", "south"],
        "target": "north",
        "operation": "dependency_preflight",
    }
    invalid = {**base, "model_revision": "0" * 40, "request_id": "bad"}
    valid = {**base, "request_id": "good"}
    completed = subprocess.run(
        [sys.executable, str(root / "workers" / "qwen_worker.py")],
        input="\n".join((json.dumps(invalid), json.dumps(valid))) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == 2
    assert responses[0]["status"] == "WORKER_OPERATION_FAIL"
    assert responses[1]["request_id"] == "good"
    assert responses[1]["status"].startswith("DEPENDENCY_PREFLIGHT_")


def test_sequential_close_requests_vram_release() -> None:
    calls: list[str] = []

    class FakeCuda:
        def is_available(self) -> bool:
            return True

        def memory_allocated(self, _index: int | None = None) -> int:
            return 100 if "empty" not in calls else 0

        def empty_cache(self) -> None:
            calls.append("empty")

        def synchronize(self, _index: int | None = None) -> None:
            calls.append("synchronize")

    adapter = QwenRecoveryAdapter()
    adapter.torch = SimpleNamespace(cuda=FakeCuda())
    adapter.model = object()
    adapter.processor = object()
    result = adapter.close()
    assert result["released"]
    assert calls == ["empty", "synchronize"]


def test_fourth_model_is_forbidden() -> None:
    with pytest.raises(ValueError, match="fourth"):
        adapter_for("replacement_family")
