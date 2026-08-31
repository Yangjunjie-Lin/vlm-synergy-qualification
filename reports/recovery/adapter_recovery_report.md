# Adapter Recovery Report

No engineering-scene task accuracy is reported. These results establish transport and measurement implementation only; they are not model-capability evidence.

| Model | Status | Native class | Processor | Meta params/buffers | Device map | Visual | Text | CLL | Constrained | Determinism | Peak VRAM | Runtime |
|---|---|---|---|---|---|---|---|---|---|---|---:|---:|
| qwen2_5_vl_7b | ENGINEERING_RECOVERY_PASS | Qwen2_5_VLForConditionalGeneration | Qwen2_5_VLProcessor | 0/0 | `{'': '0'}` | True | True | True | True | 1.00 | 6586147328 | 1047.844 |
| glm4_1v_9b | ENGINEERING_RECOVERY_PASS | Glm4vForConditionalGeneration | Glm4vProcessor | 0/0 | `{'': '0'}` | True | True | True | True | 1.00 | 7851716608 | 2345.045 |
| phi4_multimodal_5_6b | BLOCKED_BY_MODEL_ADAPTER | Phi4MMForCausalLM | Phi4MMProcessor | 0/0 | `{'': '0'}` | False | False | True | False | 0.00 | 5998003712 | 68.325 |

Engineering cohort: **PARTIAL_ENGINEERING_COHORT_GO**; passed families: **2 / 3**.

Activation patching was not run. No fourth model or replacement checkpoint was used.
