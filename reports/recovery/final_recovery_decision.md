# CapabilityGate Final Recovery Decision

# 1. Final Decision

**BLOCKED_BY_COMPUTE**

# 2. Historical Integrity

Frozen branch `codex/three-family-capability-screen`, commit `32a8333bf0106579834fb7a0747c81d67b49ac10`, annotated tag `capability-gate-adapter-block-2026-08-31`, historical blocker reports, and old empty formal artifacts remain unchanged. SynergyTrace's frozen tag remains unmoved; ReCoAlign and vlm-construct-audit remain archived.

# 3. Root-Cause Matrix

| Model | Original failure | Actual root cause | Old attempts fixed root cause? | Recovery action | Result |
|---|---|---|---|---|---|
| qwen2_5_vl_7b | Tensor.item() cannot be called on meta tensors | QUANTIZED_AUTO_OFFLOAD_META_QUANT_STATE | False | `{'native_class': 'Qwen2_5_VLForConditionalGeneration', 'primary': '4-bit GPU-only placement on one CUDA device, no CPU or disk auto-offload', 'fallback': 'official explicit device map only after a recorded GPU OOM', 'local_compute_boundary': 'BLOCKED_BY_COMPUTE_LOCAL_GPU'}` | ENGINEERING_RECOVERY_PASS |
| glm4_1v_9b | Tensor.item() cannot be called on meta tensors | QUANTIZED_AUTO_OFFLOAD_META_QUANT_STATE | False | `{'native_class': 'Glm4vForConditionalGeneration', 'processor_contract': 'apply_chat_template returns tokenized multimodal tensors', 'primary': '4-bit GPU-only placement on one CUDA device, no CPU or disk auto-offload', 'fallback': 'official explicit device map only after a recorded GPU OOM', 'local_compute_boundary': 'BLOCKED_BY_COMPUTE_LOCAL_GPU'}` | ENGINEERING_RECOVERY_PASS |
| phi4_multimodal_5_6b | missing dependency: backoff | INCOMPLETE_INFERENCE_DEPENDENCY_SET | False | `{'isolated_python': '3.10', 'dependency_preflight_before_load': 'mandatory', 'native_class': 'AutoModelForCausalLM', 'trust_remote_code': True, 'backoff_version': '2.2.1', 'attention_fallback': 'eager_or_sdpa_on_windows'}` | BLOCKED_BY_MODEL_ADAPTER |

# 4. Environment Matrix

Full pinned Python, Torch, Transformers, Accelerate, bitsandbytes, and additional dependency versions are in `reports/recovery/environment_matrix.md`.

| Model | Python | Torch | Transformers | Accelerate | bitsandbytes | Preflight |
|---|---|---|---|---|---|---|
| qwen2_5_vl_7b | 3.11 | 2.6.0+cu124 | 4.57.6 | 1.14.0 | 0.50.2 | DEPENDENCY_PREFLIGHT_PASS |
| glm4_1v_9b | 3.11 | 2.6.0+cu124 | 4.57.6 | 1.14.0 | 0.50.2 | DEPENDENCY_PREFLIGHT_PASS |
| phi4_multimodal_5_6b | 3.10 | 2.6.0+cu124 | 4.48.2 | 1.3.0 | 0.45.2 | DEPENDENCY_PREFLIGHT_PASS |

# 5. Adapter Recovery

| Model | Load status | Native class | Processor | Meta tensors | Device map | Visual forward | CLL | Constrained | Determinism | Peak VRAM | Runtime |
|---|---|---|---|---:|---|---|---|---|---:|---:|---:|
| qwen2_5_vl_7b | ENGINEERING_RECOVERY_PASS | Qwen2_5_VLForConditionalGeneration | Qwen2_5_VLProcessor | 0 | `{'': '0'}` | True | True | True | 1.00 | 6586147328 | 1047.844 |
| glm4_1v_9b | ENGINEERING_RECOVERY_PASS | Glm4vForConditionalGeneration | Glm4vProcessor | 0 | `{'': '0'}` | True | True | True | 1.00 | 7851716608 | 2345.045 |
| phi4_multimodal_5_6b | BLOCKED_BY_MODEL_ADAPTER | Phi4MMForCausalLM | Phi4MMProcessor | 0 | `{'': '0'}` | False | True | False | 0.00 | 5998003712 | 68.325 |

# 6. Engineering Cohort

**2 / 3** families passed. Cohort gate: **PARTIAL_ENGINEERING_COHORT_GO**. Blocked models and engineering reasons are `{'phi4_multimodal_5_6b': 'BLOCKED_BY_MODEL_ADAPTER'}`. No engineering result is treated as model-capability evidence.

# 7. Atomic Qualification

**ATOMIC_INCOMPLETE_BY_COMPUTE_BLOCK**. Preserved rows: 15; formal rows used for scientific metrics: 0. No Atomic capability conclusion was made.

# 8. Joint Composition

Not run; upstream gate: `BLOCKED_BY_COMPUTE`.

# 9. Q1 Potential

**NOT_EVALUATED_ENGINEERING_BLOCK**

# 10. Exact Next Action

**MIGRATE_SAME_FROZEN_COHORT_TO_ADEQUATE_GPU**

The historical decision was produced before any formal forward. This recovery changed only engineering paths; all formal scientific parameters remained frozen. Atomic started: **True**; Atomic completed: **False**. Joint ran: **False**. A model-capability conclusion was produced: **False**. Activation patching was not run.
