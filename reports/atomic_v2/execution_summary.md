# Atomic v2 execution closeout

## Authoritative decision

**MEASUREMENT_IMPLEMENTATION_NO_GO**

`q1_potential: NOT_EVALUATED_MEASUREMENT_FAILURE`

`exact_next_action: TERMINATE_AFTER_FINAL_MEASUREMENT_FAILURE`

The real frozen Qwen and GLM models each executed all 16 engineering-only contract
controls on the original Compshare RTX 3090 instance, sequentially. This is actual
model execution, not mock inference. The engineering gate failed, so the protocol
correctly forbids formal Atomic qualification. No valid v2 capability/cohort or
Joint-composition conclusion was measured. Mechanism result: **NOT EXECUTED**.

## Preserved historical protocol

Atomic v1 at `eca06a65e9ea7c384ff17859bcd9b28c79630219` remains
`INVALID_FOR_COHORT_CONCLUSION`, not a clean capability NO-GO. Its tag is
`capability-gate-atomic-v1-protocol-invalid-2026-09-10`, annotated object
`bce668ac9f95f2cd0ee4639c45985eff478e3752`. All 674 protected files and old tags
were verified unchanged. Both v1 256-row predictions retain their audit hashes:

| Family | v1 prediction SHA-256 |
|---|---|
| Qwen | `d41ad33bece21d308352d489a8d6deb4aaa2df371aab9b59749137f990772dba` |
| GLM | `cd4e4be6649d4e604bb201698549fc2a89328215ffff15ca6d83e74fec6aebf5` |

The invalidating evidence remains: 28 alias-collision scenes, four direct-visual
self-relations with directional targets, all 256 GLM prompts containing Python
object repr, constant GLM east outputs, and erroneous classification of shared
failure as contract dependence. Identical constant marginals have undefined kappa,
not a fabricated kappa=1. The old 15 local partial rows were not reused. Historical
repositories and Archive settings were not changed.

## Fresh data

256 new Atomic scenes (4 tasks ×64); each task has 16 of every answer and every
correct-option position. Alias collisions, duplicate distractors, self-relations,
query/reference equality, and v1 UUID/name/image-hash overlaps are all zero.
There are 256 unique new image hashes. Symbolic oracle accuracy is 1.00.

| Frozen shortcut probe | Pooled 256-row out-of-fold accuracy |
|---|---:|
| scene index | 0.23046875 |
| entity names | 0.23046875 |
| evidence-free question text | 0.21484375 |
| template | 0.25 |
| option position | 0.25 |
| option order | 0.25 |
| image filename | 0.25 |
| non-evidence metadata | 0.25 |

All meet the pre-outcome ≤0.30 gates. Per-task diagnostics and every held-out
prediction are retained, not hidden. Descriptor-aware binding probes also pass.
These finite probes do not prove absence of every possible nonlinear shortcut.

Old Joint had two invalid quartets; it was not locally patched. A wholly new
128-quartet batch was generated and hashed before any model inference, with zero
old UUID overlap, oracle 1.00 and non-identifying unimodal conditions. No full-batch
regeneration was used. Its data were **not run as a Joint screen**.

## Renderers and actual model controls

Each family has eight actual official-processor prompt snapshots (two per task),
with ordinary system text once, the exact row question and candidate list once,
one logical image block in visual tasks, no image in text tasks, and nonempty
tokenization. GLM retains `Glm4vForConditionalGeneration`; Qwen retains
`Qwen2_5_VLForConditionalGeneration`. No checkpoint or revision was changed:

- Qwen/Qwen2.5-VL-7B-Instruct: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- zai-org/GLM-4.1V-9B-Thinking: `3c1471e51dc811b589d4d12b1c1c7c1c941267c2`.

Actual loaded weight hashes were verified for both models. Initial GLM processor
validation failed before any forward because Transformers' generic multimodal
tokenization wrapper iterated a plain system string. The sole pre-control
compatibility retry renders the unchanged official template, then calls the
official GLM processor with text/images and no duplicate special tokens. Initial
failure files and initial stopping documents are retained with an archive map;
they are not the current final decision. No further renderer retry is permitted.

| Engineering check | Qwen | GLM |
|---|---|---|
| cases completed | 16/16 | 16/16 |
| all candidate token sequences nonempty | PASS | PASS |
| all CLL scores finite | PASS | PASS |
| CLL answer counts | four of each direction | east:16 |
| constrained-generation counts | four of each direction | east:16 |
| invalid constrained outputs | 0 | 0 |
| all four answers generated | PASS | FAIL |
| nonconstant output | PASS | FAIL |
| artifact completeness | 1.00 | 1.00 |
| engineering gate | PASS | MEASUREMENT_IMPLEMENTATION_NO_GO |

No contract-control accuracy is presented as a scientific task result. The two
models agree on the constant GLM outputs; this is an engineering-contract failure,
not evidence that GLM lacks all atomic capabilities.

## Formal-stage status and reproducibility

Formal Atomic: **NOT EXECUTED**, 0/256 for each family, blocked by GLM's engineering
gate. All per-task scientific metrics and qualified count are **NOT EVALUATED**,
not zero accuracies and not a valid cohort rejection. The formal run lock and
prerun tag were intentionally not created because their prerequisite failed.

Joint, Atomic retention, Phi, any fourth model, and activation patching:
**NOT EXECUTED**. No new checkpoint, prompt, threshold, Atomic v3 or rescue project
is authorized. The only exact next action is the terminal action above.

Full raw controls, stderr, environment, weights, snapshots, data manifests and
failure history are in `artifacts/atomic_v2/`. The remote result commit is
`84c0fb16684db77a3e28a7578f8e31015b6f6b14`; its 601-entry final manifest recomputed
successfully after local retrieval. Both legacy verifiers run read-only, preserving
their historical manifests. The post-run terminal-command guard is engineering
only: rerunning the total command verifies the recorded result without reopening
the experiment or regenerating validation artifacts.

The existing instance was left powered on pending explicit shutdown confirmation;
models were released. Its observed rate was ¥1.19/hour. Artifacts have been
retrieved locally and the result history pushed to the requested GitHub branch.
