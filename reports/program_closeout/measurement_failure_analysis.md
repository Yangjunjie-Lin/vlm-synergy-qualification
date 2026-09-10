# Measurement failure analysis

Failure class: `MEASUREMENT_CONTRACT_FAILURE`.

Protocol decision: `MEASUREMENT_IMPLEMENTATION_NO_GO`.

Causal diagnosis: `NOT_IDENTIFIED`.

This is an evidence-only analysis of the frozen observations at
`eb12bd64da6cb37d4e43631e4f2e08492a9007fb`. No control is rerun, no processor or
model is loaded, and no alternative answer contract is evaluated for this report.

## What was observed

Fresh Atomic v2 data validation and both model-specific renderer validations
passed. Both frozen model families then completed the 16 engineering-only cases.
The controls directly specified a legal output, with each of four directions
requested four times. They were not Atomic scientific task evaluations.

| Engineering observation | Qwen | GLM |
|---|---|---|
| Completed controls | 16/16 | 16/16 |
| Candidate encodings nonempty | PASS | PASS |
| All CLL scores finite | PASS | PASS |
| CLL top-answer counts | Four of each direction | `east`: 16/16 |
| Constrained-generation counts | Four of each direction | `east`: 16/16 |
| Invalid constrained outputs | 0 | 0 |
| All four legal answers generated | PASS | FAIL |
| Nonconstant output | PASS | FAIL |
| Artifact completeness | 1.00 | 1.00 |
| Frozen engineering gate | PASS | FAIL |

The unchanged [control report](../atomic_v2/contract_control.md) and raw
[Qwen controls](../../artifacts/atomic_v2/contract_control/qwen2_5_vl_7b/predictions.jsonl)
and [GLM controls](../../artifacts/atomic_v2/contract_control/glm4_1v_9b/predictions.jsonl)
are the source evidence. Control success or failure is not reported as scientific
task accuracy.

Agreement between GLM's CLL selections and generated answers does not make a
constant, nonresponsive control a valid capability measurement. It also does
not establish a disagreement-based measurement-contract-dependent scientific
classification. For identical constant answer marginals, Cohen's kappa is
undefined, not a manufactured value of one. No missing formal kappa or accuracy
is imputed.

## What remains unresolved

Model-contract incompatibility, a first-token prior, a reasoning-output-structure
mismatch, quantized-logit bias, or an unidentified implementation factor are
possible explanations only. The observed 16/16 `east` selections do not isolate
any one cause. In particular, the dominant answer does not prove that GLM lacks
spatial understanding or atomic capability.

Whether native reasoning-output parsing, different quantization, or another
preregistered contract would change the outcome is unresolved. These questions
do not authorize parsing existing outputs into a new result, changing the
template, repeating controls, or starting any experiment in this line.

## Why the line stops

The frozen measurement stopping rule was triggered before formal Atomic. The
permitted renderer/adapter correction was already used. Continuing with new
candidate words, prefixes, parsers, reasoning extraction, prompts, templates,
quantization, checkpoints, revisions or models would constitute a new,
unauthorized measurement attempt.

Atomic v1 had distinct protocol-invalidity findings, preserved in its
[adjudication](../../research/atomic_v2/atomic_v1_protocol_adjudication.yaml).
Its invalidity must not be collapsed into the v2 measurement-contract failure.
Neither record supplies a valid cohort capability conclusion. Formal Atomic,
Joint composition and mechanism results were not observed. The terminal action
is evidence preservation and no further experiments on this line.
