from __future__ import annotations

import gc
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from capability_gate.artifacts import canonical_json, sha256_file, sha256_text
from capability_gate.models.adapters import PrefixTrie
from capability_gate.recovery.dependencies import DEPENDENCY_SPECS, dependency_preflight
from capability_gate.recovery.placement import single_gpu_4bit_model_kwargs


class DependencyPreflightError(RuntimeError):
    pass


class FrozenRevisionError(RuntimeError):
    pass


class MetaTensorError(RuntimeError):
    pass


class MeasurementImplementationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    model_id: str
    revision: str
    processor_revision: str
    loader_class: str
    expected_model_class: str
    expected_processor_class: str
    trust_remote_code: bool
    python_minor: tuple[int, int]
    required_packages: Mapping[str, str]
    weight_names: tuple[str, ...]
    weight_hashes: tuple[str, ...]


DESCRIPTORS: dict[str, ModelDescriptor] = {
    "qwen2_5_vl_7b": ModelDescriptor(
        key="qwen2_5_vl_7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        processor_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        loader_class="Qwen2_5_VLForConditionalGeneration",
        expected_model_class="Qwen2_5_VLForConditionalGeneration",
        expected_processor_class="Qwen2_5_VLProcessor",
        trust_remote_code=False,
        python_minor=(3, 11),
        required_packages=DEPENDENCY_SPECS["qwen2_5_vl_7b"],
        weight_names=tuple(f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)),
        weight_hashes=(
            "e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e",
            "a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024",
            "111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507",
            "ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed",
            "0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67",
        ),
    ),
    "glm4_1v_9b": ModelDescriptor(
        key="glm4_1v_9b",
        model_id="zai-org/GLM-4.1V-9B-Thinking",
        revision="3c1471e51dc811b589d4d12b1c1c7c1c941267c2",
        processor_revision="3c1471e51dc811b589d4d12b1c1c7c1c941267c2",
        loader_class="Glm4vForConditionalGeneration",
        expected_model_class="Glm4vForConditionalGeneration",
        expected_processor_class="Glm4vProcessor",
        trust_remote_code=False,
        python_minor=(3, 11),
        required_packages=DEPENDENCY_SPECS["glm4_1v_9b"],
        weight_names=tuple(f"model-{index:05d}-of-00004.safetensors" for index in range(1, 5)),
        weight_hashes=(
            "56708b3541e1f69c9843ae19eaa23720c5cee9f3d0c9ed101c3b9b43b40974a1",
            "7c353db4141fdd9a0c6b990d50040a28812bb56dc91c42154abc5a4ee81fe698",
            "f3fab833fe4cad6cbce924908e1a58320217295417bbf87aa810fff6e4b634a4",
            "b3c7723fac7de60f2b67e4f84f2d8648d7431ff32c95c630dcc17790091005eb",
        ),
    ),
    "phi4_multimodal_5_6b": ModelDescriptor(
        key="phi4_multimodal_5_6b",
        model_id="microsoft/Phi-4-multimodal-instruct",
        revision="93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        processor_revision="93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        loader_class="AutoModelForCausalLM",
        expected_model_class="Phi4MMForCausalLM",
        expected_processor_class="Phi4MMProcessor",
        trust_remote_code=True,
        python_minor=(3, 10),
        required_packages=DEPENDENCY_SPECS["phi4_multimodal_5_6b"],
        weight_names=tuple(f"model-{index:05d}-of-00003.safetensors" for index in range(1, 4)),
        weight_hashes=(
            "c46bb03332d82f6a3eaf85bd20af388dd4d4d68b198c2203c965c7381a466094",
            "b3e812c0c8acef4e7f5e34d6c9f77a7640ee4a2b93ea351921365ac62f19918d",
            "7be96b7339303752634b202d3f377bcf312a03046586eca6cea23347ace1e65a",
        ),
    ),
}


def _tensor_is_meta(value: Any) -> bool:
    return bool(getattr(value, "is_meta", False)) or getattr(
        getattr(value, "device", None), "type", None
    ) == "meta"


def inspect_materialization(model: Any, loading_info: Mapping[str, Any]) -> dict[str, Any]:
    meta_parameters = [name for name, parameter in model.named_parameters() if _tensor_is_meta(parameter)]
    meta_buffers = [name for name, buffer in model.named_buffers() if _tensor_is_meta(buffer)]
    unmaterialized_buffers = [
        name
        for name, buffer in model.named_buffers()
        if type(buffer).__name__ in {"UninitializedBuffer", "UninitializedParameter"}
    ]
    return {
        "meta_parameter_count": len(meta_parameters),
        "meta_buffer_count": len(meta_buffers),
        "missing_weight_count": len(loading_info.get("missing_keys", [])),
        "unexpected_weight_count": len(loading_info.get("unexpected_keys", [])),
        "mismatched_weight_count": len(loading_info.get("mismatched_keys", [])),
        "meta_parameters": meta_parameters,
        "meta_buffers": meta_buffers,
        "unmaterialized_buffers": unmaterialized_buffers,
        "missing_weights": list(loading_info.get("missing_keys", [])),
        "unexpected_weights": list(loading_info.get("unexpected_keys", [])),
        "mismatched_weights": list(loading_info.get("mismatched_keys", [])),
    }


def _device_map(model: Any) -> dict[str, str]:
    resolved = getattr(model, "hf_device_map", None)
    if isinstance(resolved, dict):
        return {str(key): str(value) for key, value in resolved.items()}
    devices = {str(parameter.device) for parameter in model.parameters()}
    return {"<parameter_devices>": ",".join(sorted(devices))}


def placement_inventory(model: Any) -> dict[str, Any]:
    quantized_gpu_modules: set[str] = set()
    full_precision_cpu_modules: set[str] = set()
    disk_offloaded_modules: set[str] = set()
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        device = getattr(weight, "device", None)
        class_name = type(module).__name__.lower()
        if "4bit" in class_name and getattr(device, "type", None) == "cuda":
            quantized_gpu_modules.add(name)
        elif weight is not None and getattr(device, "type", None) == "cpu":
            full_precision_cpu_modules.add(name)
    for name, placement in _device_map(model).items():
        if str(placement) == "disk":
            disk_offloaded_modules.add(name)
    return {
        "quantized_gpu_modules": sorted(quantized_gpu_modules),
        "full_precision_cpu_modules": sorted(full_precision_cpu_modules),
        "disk_offloaded_modules": sorted(disk_offloaded_modules),
    }


class NativeRecoveryAdapter:
    """Base for frozen model-specific adapters used only by isolated workers."""

    def __init__(self, descriptor: ModelDescriptor):
        self.descriptor = descriptor
        self.model: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.loading_info: dict[str, Any] = {}
        self.materialization: dict[str, Any] = {}
        self.load_seconds: float | None = None
        self.weight_manifest: list[dict[str, Any]] = []
        self._vision_observed = False
        self._vision_hook_handles: list[Any] = []

    def _processor_kwargs(self) -> dict[str, Any]:
        return {
            "revision": self.descriptor.processor_revision,
            "trust_remote_code": self.descriptor.trust_remote_code,
        }

    def load_processor(self) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.descriptor.model_id, **self._processor_kwargs()
        )
        actual = type(self.processor).__name__
        if actual != self.descriptor.expected_processor_class:
            raise MeasurementImplementationError(
                f"processor class mismatch: expected {self.descriptor.expected_processor_class}, got {actual}"
            )

    def _model_loader(self) -> Any:
        raise NotImplementedError

    def _model_kwargs(self) -> dict[str, Any]:
        return single_gpu_4bit_model_kwargs(
            torch=self.torch,
            revision=self.descriptor.revision,
            trust_remote_code=self.descriptor.trust_remote_code,
            attn_implementation="eager",
        )

    def load(self) -> None:
        preflight = dependency_preflight(self.descriptor.key)
        if preflight["status"] != "DEPENDENCY_PREFLIGHT_PASS":
            raise DependencyPreflightError(canonical_json(preflight))
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the frozen recovery model")
        self.torch = torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        if self.processor is None:
            self.load_processor()
        loaded = self._model_loader().from_pretrained(
            self.descriptor.model_id, **self._model_kwargs()
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise MeasurementImplementationError("output_loading_info did not return model and info")
        self.model, self.loading_info = loaded
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        actual = type(self.model).__name__
        if actual != self.descriptor.expected_model_class:
            raise MeasurementImplementationError(
                f"native model class mismatch: expected {self.descriptor.expected_model_class}, got {actual}"
            )
        self.materialization = inspect_materialization(self.model, self.loading_info)
        failed_counts = {
            key: self.materialization[key]
            for key in (
                "meta_parameter_count",
                "meta_buffer_count",
                "missing_weight_count",
                "unexpected_weight_count",
                "mismatched_weight_count",
            )
            if self.materialization[key]
        }
        if failed_counts:
            raise MetaTensorError(
                f"ENGINEERING_RECOVERY_FAIL_META_TENSOR_OR_WEIGHTS: {failed_counts}"
            )
        self._install_vision_hooks()

    def verify_cached_weights(self) -> list[dict[str, Any]]:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                self.descriptor.model_id,
                revision=self.descriptor.revision,
                allow_patterns=list(self.descriptor.weight_names),
                local_files_only=True,
            )
        )
        manifest = []
        for name, expected in zip(self.descriptor.weight_names, self.descriptor.weight_hashes):
            path = snapshot / name
            if not path.is_file():
                raise FrozenRevisionError(f"cached weight missing: {name}")
            actual = sha256_file(path)
            if actual != expected:
                raise FrozenRevisionError(
                    f"weight hash mismatch for {name}: expected {expected}, got {actual}"
                )
            manifest.append({"path": name, "bytes": path.stat().st_size, "sha256": actual})
        self.weight_manifest = manifest
        return manifest

    def _install_vision_hooks(self) -> None:
        candidates: list[tuple[str, Any]] = []
        for name, module in self.model.named_modules():
            lowered = name.lower()
            if name == "visual" or any(
                marker in lowered for marker in ("vision", "image_embed", "img_processor")
            ):
                candidates.append((name, module))
        if not candidates:
            raise MeasurementImplementationError("no vision module found for forward proof hook")

        def observed(_module: Any, _inputs: Any, _output: Any) -> None:
            self._vision_observed = True

        seen: set[int] = set()
        for _name, module in candidates:
            if id(module) not in seen:
                self._vision_hook_handles.append(module.register_forward_hook(observed))
                seen.add(id(module))

    def _encode(self, system: str, user: str, image: Image.Image | None) -> tuple[str, Any]:
        raise NotImplementedError

    def _model_device(self) -> Any:
        for parameter in self.model.parameters():
            if parameter.device.type == "cuda":
                return parameter.device
        raise MeasurementImplementationError("recovery model has no CUDA parameters")

    def _to_model(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        device = self._model_device()
        result = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(device)
                if getattr(value, "is_floating_point", lambda: False)():
                    value = value.to(self.torch.bfloat16)
            result[key] = value
        return result

    def _candidate_ids(self, candidate_prefix: str, candidate: str) -> list[int]:
        ids = self.processor.tokenizer.encode(
            candidate_prefix + candidate, add_special_tokens=False
        )
        if not ids:
            raise MeasurementImplementationError(f"candidate tokenization is empty: {candidate}")
        return list(ids)

    def _append_candidate(self, prompt_inputs: Mapping[str, Any], ids: Sequence[int]) -> dict[str, Any]:
        torch = self.torch
        full = dict(prompt_inputs)
        input_ids = prompt_inputs["input_ids"]
        suffix = torch.tensor([list(ids)], dtype=input_ids.dtype, device=input_ids.device)
        full["input_ids"] = torch.cat([input_ids, suffix], dim=1)
        if "attention_mask" in prompt_inputs:
            mask = prompt_inputs["attention_mask"]
            full["attention_mask"] = torch.cat(
                [mask, torch.ones((mask.shape[0], len(ids)), dtype=mask.dtype, device=mask.device)],
                dim=1,
            )
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in prompt_inputs:
                values = prompt_inputs[key]
                full[key] = torch.cat(
                    [
                        values,
                        torch.zeros(
                            (values.shape[0], len(ids)), dtype=values.dtype, device=values.device
                        ),
                    ],
                    dim=1,
                )
        full.pop("position_ids", None)
        return full

    def score_and_generate(
        self,
        *,
        system: str,
        user: str,
        image_path: str | None,
        candidates: Sequence[str],
        target: str,
        candidate_prefix: str = " ",
    ) -> dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("adapter is not loaded")
        if target not in candidates:
            raise MeasurementImplementationError("target is not in the allowed candidate set")
        image = None
        if image_path is not None:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        started = time.perf_counter()
        self._vision_observed = False
        rendered, encoded = self._encode(system, user, image)
        visual_keys = sorted(
            key
            for key in encoded
            if key.startswith(("pixel_", "image_", "input_image"))
        )
        if image is not None and not visual_keys:
            raise MeasurementImplementationError("processor produced no visual tensor fields")
        prompt_inputs = self._to_model(encoded)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        scores = []
        candidate_sequences = []
        for order, candidate in enumerate(candidates):
            candidate_ids = self._candidate_ids(candidate_prefix, candidate)
            candidate_sequences.append(candidate_ids)
            full_inputs = self._append_candidate(prompt_inputs, candidate_ids)
            with self.torch.inference_mode():
                outputs = self.model(**full_inputs, return_dict=True, use_cache=False)
                logits = outputs.logits[0]
                log_probs = self.torch.log_softmax(logits.float(), dim=-1)
                token_scores = [
                    float(log_probs[prompt_length + offset - 1, token_id].item())
                    for offset, token_id in enumerate(candidate_ids)
                ]
            if not token_scores or not all(math.isfinite(value) for value in token_scores):
                raise MeasurementImplementationError(
                    f"candidate has empty or non-finite CLL: {candidate}"
                )
            raw = sum(token_scores)
            scores.append(
                {
                    "candidate": candidate,
                    "candidate_token_ids": candidate_ids,
                    "candidate_token_count": len(candidate_ids),
                    "token_log_probabilities": token_scores,
                    "raw_log_likelihood": raw,
                    "normalized_log_likelihood": raw / len(token_scores),
                    "frozen_order": order,
                }
            )
            del outputs, logits, log_probs, full_inputs
        if image is not None and not self._vision_observed:
            raise MeasurementImplementationError(
                "visual tensors were present but no vision module forward hook fired"
            )
        ranking = sorted(
            scores,
            key=lambda score: (-score["normalized_log_likelihood"], score["frozen_order"]),
        )
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.processor.tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        if not eos_ids or any(value is None for value in eos_ids):
            raise MeasurementImplementationError("model has no usable EOS token")
        trie = PrefixTrie(candidate_sequences, eos_ids)

        def allowed_tokens(_batch_id: int, input_ids: Any) -> list[int]:
            sequence = input_ids[0] if getattr(input_ids, "ndim", 1) == 2 else input_ids
            return trie.allowed(sequence[prompt_length:].tolist())

        with self.torch.inference_mode():
            generated = self.model.generate(
                **prompt_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max(map(len, candidate_sequences)) + 1,
                prefix_allowed_tokens_fn=allowed_tokens,
                pad_token_id=eos_ids[0],
            )
        generated_suffix = generated[0, prompt_length:].tolist()
        constrained_answer = None
        for candidate, sequence in zip(candidates, candidate_sequences):
            if generated_suffix[: len(sequence)] == sequence:
                constrained_answer = candidate
                break
        if constrained_answer not in candidates:
            raise MeasurementImplementationError(
                f"constrained generation escaped allowed answers: {generated_suffix}"
            )
        target_score = next(
            score["normalized_log_likelihood"]
            for score in scores
            if score["candidate"] == target
        )
        other_scores = [
            score["normalized_log_likelihood"]
            for score in scores
            if score["candidate"] != target
        ]
        return {
            "rendered_prompt": rendered,
            "candidate_scores": scores,
            "candidate_ranking": [score["candidate"] for score in ranking],
            "top_answer": ranking[0]["candidate"],
            "target_margin": target_score - sum(other_scores) / len(other_scores),
            "constrained_answer": constrained_answer,
            "constrained_generation_token_ids": generated_suffix,
            "runtime_seconds": time.perf_counter() - started,
            "visual_input_keys": visual_keys,
            "vision_forward_observed": self._vision_observed if image is not None else None,
            "text_only_forward": image is None,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("adapter is not loaded")
        gpu = self.torch.cuda.get_device_properties(0)
        placement = placement_inventory(self.model)
        return {
            "model_key": self.descriptor.key,
            "model_id": self.descriptor.model_id,
            "model_revision": self.descriptor.revision,
            "processor_revision": self.descriptor.processor_revision,
            "loader_class": self.descriptor.loader_class,
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "native_model_class_verified": (
                type(self.model).__name__ == self.descriptor.expected_model_class
            ),
            "processor_class_verified": (
                type(self.processor).__name__ == self.descriptor.expected_processor_class
            ),
            "placement_policy": "EXPLICIT_SINGLE_GPU_4BIT_NO_CPU_DISK_OFFLOAD",
            "resolved_device_map": _device_map(self.model),
            "materialization": self.materialization,
            "placement_inventory": placement,
            "weight_manifest": self.weight_manifest,
            "weight_hashes_verified": bool(self.weight_manifest),
            "load_seconds": self.load_seconds,
            "peak_vram_bytes": self.torch.cuda.max_memory_allocated(0),
            "peak_reserved_vram_bytes": self.torch.cuda.max_memory_reserved(0),
            "gpu": {"name": gpu.name, "total_vram_bytes": gpu.total_memory},
            "hf_home": os.environ.get("HF_HOME"),
        }

    def close(self) -> dict[str, Any]:
        before = None
        if self.torch is not None and self.torch.cuda.is_available():
            before = self.torch.cuda.memory_allocated(0)
        for handle in self._vision_hook_handles:
            handle.remove()
        self._vision_hook_handles.clear()
        self.model = None
        self.processor = None
        self.loading_info = {}
        gc.collect()
        after = None
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.synchronize(0)
            after = self.torch.cuda.memory_allocated(0)
        return {
            "allocated_before_release_bytes": before,
            "allocated_after_release_bytes": after,
            "released": before is None or after is None or after <= before,
        }


class QwenRecoveryAdapter(NativeRecoveryAdapter):
    def __init__(self) -> None:
        super().__init__(DESCRIPTORS["qwen2_5_vl_7b"])

    def _processor_kwargs(self) -> dict[str, Any]:
        return {
            **super()._processor_kwargs(),
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 512 * 28 * 28,
        }

    def _model_loader(self) -> Any:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration

    def _encode(self, system: str, user: str, image: Image.Image | None) -> tuple[str, Any]:
        from qwen_vl_utils import process_vision_info

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": content},
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        encoded = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return rendered, encoded


class GlmRecoveryAdapter(NativeRecoveryAdapter):
    def __init__(self) -> None:
        super().__init__(DESCRIPTORS["glm4_1v_9b"])

    def _processor_kwargs(self) -> dict[str, Any]:
        return {**super()._processor_kwargs(), "use_fast": True}

    def _model_loader(self) -> Any:
        from transformers import Glm4vForConditionalGeneration

        return Glm4vForConditionalGeneration

    def _encode(self, system: str, user: str, image: Image.Image | None) -> tuple[str, Any]:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
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


class PhiRecoveryAdapter(NativeRecoveryAdapter):
    def __init__(self) -> None:
        super().__init__(DESCRIPTORS["phi4_multimodal_5_6b"])

    def _model_loader(self) -> Any:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM

    def _encode(self, system: str, user: str, image: Image.Image | None) -> tuple[str, Any]:
        image_token = "<|image_1|>" if image is not None else ""
        rendered = (
            f"<|system|>{system}<|end|><|user|>{image_token}{user}<|end|>"
            "<|assistant|>"
        )
        kwargs: dict[str, Any] = {"text": [rendered], "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        return rendered, self.processor(**kwargs)


def adapter_for(model_key: str) -> NativeRecoveryAdapter:
    adapters = {
        "qwen2_5_vl_7b": QwenRecoveryAdapter,
        "glm4_1v_9b": GlmRecoveryAdapter,
        "phi4_multimodal_5_6b": PhiRecoveryAdapter,
    }
    try:
        return adapters[model_key]()
    except KeyError as error:
        raise ValueError(f"fourth or unknown model is forbidden: {model_key}") from error


def response_hash(response: Mapping[str, Any]) -> str:
    payload = dict(response)
    payload["artifact_hash"] = None
    return sha256_text(canonical_json(payload))
