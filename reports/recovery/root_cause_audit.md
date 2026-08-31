# Root-Cause Audit

This audit was frozen before any recovery load attempt. Its evidence is the complete traceback
set in `artifacts/manifests/engineering_smoke.json` at commit
`32a8333bf0106579834fb7a0747c81d67b49ac10`.

## Qwen2.5-VL

Both old attempts failed while Accelerate attached dispatch hooks and requested a state dict
from a bitsandbytes 4-bit module. `QuantState.as_dict()` called `.item()` on a meta-valued
offset and raised `Tensor.item() cannot be called on meta tensors`. Attempt 2 only lowered GPU
`max_memory` from 4600 MiB to 4100 MiB and changed the offload folder. It retained NF4,
`device_map=auto`, CPU offload flags, and the same hook path, so it was not a root-cause repair.

Recovery uses `Qwen2_5_VLForConditionalGeneration`, bounded visual tokens, candidate
microbatch 1, and a one-GPU quantized placement without CPU/disk auto-offload. A local capacity
failure is classified as compute, not model capability.

## GLM-4.1V

The complete GLM traceback has the same terminal chain and error as Qwen: Accelerate dispatch
hook attachment, module state-dict collection, bitsandbytes quant-state packing, and `.item()`
on a meta tensor. The Qwen and GLM observations therefore instantiate one shared quantized
auto-offload root cause; they are not two independent adapter defects. GLM attempt 2 likewise
changed only the memory profile and offload folder.

Recovery uses `Glm4vForConditionalGeneration` and the model-specific
`apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True,
return_tensors="pt")` contract. It does not assume the generic parallel text/image processor
layout is equivalent.

## Phi-4-Multimodal

Both old attempts stopped before model construction because remote modeling code required the
missing `backoff` package. Neither attempt installed it; changing `max_memory` was unrelated to
dependency resolution. This is an incomplete inference dependency set and a packaging /
environment failure.

Recovery isolates the official Python 3.10 / Torch 2.6.0 / Transformers 4.48.2 stack, pins
`backoff==2.2.1`, and runs an exhaustive dependency preflight before any
`from_pretrained` call. FlashAttention is optional for the Windows smoke; SDPA or eager is
recorded when used.

None of these diagnoses is capability evidence, and no activation patching is in scope.
