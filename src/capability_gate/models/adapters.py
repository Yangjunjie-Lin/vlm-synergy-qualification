from __future__ import annotations

import gc
import importlib.metadata
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image

from capability_gate.artifacts import canonical_json, sha256_text


@dataclass(frozen=True)
class PromptPayload:
    system: str
    user: str


class PrefixTrie:
    def __init__(self, sequences: Sequence[Sequence[int]], eos_ids: Sequence[int]):
        self.sequences = [tuple(sequence) for sequence in sequences]
        self.eos_ids = list(eos_ids)

    def allowed(self, prefix: Sequence[int]) -> list[int]:
        prefix_tuple = tuple(prefix)
        next_tokens = {
            sequence[len(prefix_tuple)]
            for sequence in self.sequences
            if len(sequence) > len(prefix_tuple) and sequence[: len(prefix_tuple)] == prefix_tuple
        }
        complete = any(sequence == prefix_tuple for sequence in self.sequences)
        if complete:
            next_tokens.update(self.eos_ids)
        return sorted(next_tokens) if next_tokens else self.eos_ids


class TransformersVLM:
    """Frozen-revision VLM adapter implementing two genuinely separate contracts."""

    def __init__(self, spec: dict[str, Any], profile: dict[str, Any] | None = None):
        self.spec = spec
        self.profile = profile or spec["load_profiles"][0]
        self.model = None
        self.processor = None
        self.torch = None

    def load(self) -> None:
        raise RuntimeError(
            "HISTORICAL_GENERIC_4BIT_AUTO_OFFLOAD_PATH_DISABLED: use an isolated recovery worker"
        )

    def close(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def runtime_metadata(self) -> dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("adapter is not loaded")
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            device_map = {str(key): str(value) for key, value in device_map.items()}
        gpu = {}
        if self.torch.cuda.is_available():
            properties = self.torch.cuda.get_device_properties(0)
            gpu = {
                "name": properties.name,
                "total_vram_bytes": properties.total_memory,
                "max_memory_allocated_bytes": self.torch.cuda.max_memory_allocated(0),
                "max_memory_reserved_bytes": self.torch.cuda.max_memory_reserved(0),
            }
        versions = {}
        for package in ("torch", "transformers", "accelerate", "bitsandbytes"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        weight_hashes = self.spec["expected_weight_sha256"]
        return {
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "tokenizer_class": type(getattr(self.processor, "tokenizer", None)).__name__,
            "model_revision": self.spec["revision"],
            "processor_revision": self.spec["processor_revision"],
            "tokenizer_revision": self.spec["tokenizer_revision"],
            "dtype": self.spec["dtype"],
            "quantization": self.spec["quantization"],
            "device_map_policy": self.spec["device_map"],
            "resolved_device_map": device_map,
            "load_profile": self.profile,
            "expected_weight_sha256": weight_hashes,
            "weight_manifest_sha256": sha256_text(canonical_json(weight_hashes)),
            "gpu": gpu,
            "software_versions": versions,
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        }

    def _render(self, payload: PromptPayload, image: Image.Image | None) -> str:
        if self.spec["adapter"] == "phi4_multimodal":
            image_token = "<|image_1|>\n" if image is not None else ""
            return (
                f"<|system|>\n{payload.system}<|end|>\n"
                f"<|user|>\n{image_token}{payload.user}<|end|>\n<|assistant|>\n"
            )
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": payload.user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": payload.system}]},
            {"role": "user", "content": content},
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _encode(self, rendered: str, image: Image.Image | None, suffix: str = "") -> Any:
        kwargs: dict[str, Any] = {
            "text": [rendered + suffix],
            "return_tensors": "pt",
            "padding": True,
        }
        if image is not None:
            kwargs["images"] = [image]
        encoded = self.processor(**kwargs)
        return encoded

    def _input_device(self):
        for parameter in self.model.parameters():
            if parameter.device.type not in {"meta", "cpu"}:
                return parameter.device
        return next(self.model.parameters()).device

    def _to_device(self, encoded: Any) -> dict[str, Any]:
        device = self._input_device()
        result = {}
        for key, value in encoded.items():
            if hasattr(value, "to"):
                value = value.to(device)
                if getattr(value, "is_floating_point", lambda: False)():
                    value = value.to(self.torch.bfloat16)
            result[key] = value
        return result

    def _candidate_encoding(
        self,
        rendered: str,
        image: Image.Image | None,
        candidate: str,
        prompt_ids: Any,
        prefix: str,
    ) -> tuple[Any, list[int]]:
        full = self._encode(rendered, image, prefix + candidate)
        full_ids = full["input_ids"][0]
        prompt_length = prompt_ids.shape[0]
        if full_ids.shape[0] <= prompt_length or not self.torch.equal(
            full_ids[:prompt_length], prompt_ids
        ):
            raise RuntimeError(
                f"candidate tokenization is not a prompt-preserving continuation for {candidate!r}"
            )
        return full, full_ids[prompt_length:].tolist()

    def score_and_generate(
        self,
        payload: PromptPayload,
        image: Image.Image | None,
        candidates: Sequence[str],
        target: str,
        *,
        candidate_prefix: str = " ",
    ) -> dict[str, Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("adapter is not loaded")
        started = time.perf_counter()
        rendered = self._render(payload, image)
        prompt_encoded = self._encode(rendered, image)
        prompt_ids = prompt_encoded["input_ids"][0]
        prompt_length = prompt_ids.shape[0]
        scores = []
        candidate_sequences = []
        for order, candidate in enumerate(candidates):
            full, candidate_ids = self._candidate_encoding(
                rendered, image, candidate, prompt_ids, candidate_prefix
            )
            candidate_sequences.append(candidate_ids)
            inputs = self._to_device(full)
            with self.torch.inference_mode():
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits[0]
                log_probs = self.torch.log_softmax(logits.float(), dim=-1)
                token_scores = [
                    float(log_probs[prompt_length + offset - 1, token_id].item())
                    for offset, token_id in enumerate(candidate_ids)
                ]
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
            del outputs, logits, log_probs, inputs, full
        ranking = sorted(
            scores,
            key=lambda score: (-score["normalized_log_likelihood"], score["frozen_order"]),
        )
        top_answer = ranking[0]["candidate"]

        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.processor.tokenizer.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        trie = PrefixTrie(candidate_sequences, eos_ids)
        generation_inputs = self._to_device(prompt_encoded)
        start_length = generation_inputs["input_ids"].shape[1]

        def allowed_tokens(_batch_id: int, input_ids: Any) -> list[int]:
            sequence = input_ids[0] if getattr(input_ids, "ndim", 1) == 2 else input_ids
            return trie.allowed(sequence[start_length:].tolist())

        with self.torch.inference_mode():
            generated = self.model.generate(
                **generation_inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max(map(len, candidate_sequences)) + 1,
                prefix_allowed_tokens_fn=allowed_tokens,
                pad_token_id=eos_ids[0],
            )
        generated_suffix = generated[0, start_length:].tolist()
        constrained_answer = None
        for candidate, sequence in zip(candidates, candidate_sequences):
            if generated_suffix[: len(sequence)] == sequence:
                constrained_answer = candidate
                break
        if constrained_answer is None:
            raise RuntimeError(f"grammar generation returned invalid suffix: {generated_suffix}")
        by_candidate = {score["candidate"]: score for score in scores}
        target_score = by_candidate[target]["normalized_log_likelihood"]
        other_mean = sum(
            score["normalized_log_likelihood"] for score in scores if score["candidate"] != target
        ) / (len(scores) - 1)
        return {
            "rendered_prompt": rendered,
            "candidate_scores": scores,
            "candidate_ranking": [score["candidate"] for score in ranking],
            "top_answer": top_answer,
            "target_margin": target_score - other_mean,
            "constrained_generation_answer": constrained_answer,
            "constrained_generation_token_ids": generated_suffix,
            "runtime_seconds": time.perf_counter() - started,
        }
