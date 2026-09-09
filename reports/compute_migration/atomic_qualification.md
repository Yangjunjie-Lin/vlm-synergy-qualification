# Migrated Atomic Qualification

Decision: **CAPABILITY_COHORT_NO_GO**

## qwen2_5_vl_7b

Label: `ATOMIC_TEXT_FAIL`

| Task | Contract | n | Accuracy | One-sided 95% lower | Gate |
|---|---|---:|---:|---:|---|
| direct_visual_relation | cll | 64 | 0.9375 | 0.8627 | True |
| direct_visual_relation | constrained_generation | 64 | 0.9375 | 0.8627 | True |
| direct_text_relation | cll | 64 | 0.8281 | 0.7316 | False |
| direct_text_relation | constrained_generation | 64 | 0.8125 | 0.7140 | False |
| direction_reversal | cll | 64 | 0.7344 | 0.6286 | False |
| direction_reversal | constrained_generation | 64 | 0.6719 | 0.5629 | False |
| cross_modal_bridge_binding | cll | 64 | 0.9844 | 0.9280 | True |
| cross_modal_bridge_binding | constrained_generation | 64 | 0.9844 | 0.9280 | True |

Exact agreement=0.9727; Cohen's κ=0.9626759976672499.

## glm4_1v_9b

Label: `MEASUREMENT_CONTRACT_DEPENDENT`

| Task | Contract | n | Accuracy | One-sided 95% lower | Gate |
|---|---|---:|---:|---:|---|
| direct_visual_relation | cll | 64 | 0.2500 | 0.1635 | False |
| direct_visual_relation | constrained_generation | 64 | 0.2500 | 0.1635 | False |
| direct_text_relation | cll | 64 | 0.2500 | 0.1635 | False |
| direct_text_relation | constrained_generation | 64 | 0.2500 | 0.1635 | False |
| direction_reversal | cll | 64 | 0.2500 | 0.1635 | False |
| direction_reversal | constrained_generation | 64 | 0.2500 | 0.1635 | False |
| cross_modal_bridge_binding | cll | 64 | 0.2500 | 0.1635 | False |
| cross_modal_bridge_binding | constrained_generation | 64 | 0.2500 | 0.1635 | False |

Exact agreement=1.0000; Cohen's κ=None.

