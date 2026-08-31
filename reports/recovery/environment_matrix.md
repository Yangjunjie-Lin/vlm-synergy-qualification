# Environment Matrix

The three workers are isolated. Dependency preflight is not a model-load or scientific attempt.

| Model | Python | Torch | Transformers | Accelerate | bitsandbytes | Additional | Preflight |
|---|---|---|---|---|---|---|---|
| qwen2_5_vl_7b | 3.11 | 2.6.0+cu124 | 4.57.6 | 1.14.0 | 0.50.2 | qwen-vl-utils=0.0.14, torchvision=0.21.0+cu124, Pillow=12.3.0 | DEPENDENCY_PREFLIGHT_PASS |
| glm4_1v_9b | 3.11 | 2.6.0+cu124 | 4.57.6 | 1.14.0 | 0.50.2 | sentencepiece=0.2.1, torchvision=0.21.0+cu124, Pillow=12.3.0 | DEPENDENCY_PREFLIGHT_PASS |
| phi4_multimodal_5_6b | 3.10 | 2.6.0+cu124 | 4.48.2 | 1.3.0 | 0.45.2 | backoff=2.2.1, peft=0.13.2, soundfile=0.13.1, Pillow=11.1.0, SciPy=1.15.1, torchvision=0.21.0+cu124 | DEPENDENCY_PREFLIGHT_PASS |

Qwen and GLM use eager attention in their pinned Transformers 4.57.6 workers. Phi uses the official Python 3.10 / Transformers 4.48.2 stack; FlashAttention is optional on Windows and the recorded fallback is SDPA or eager.

Overall dependency gate: **True**.
