# FINAL PROGRAM DECISION

decision:
TERMINATE_CURRENT_CROSS_MODAL_SYNERGY_LINE

source_protocol_decision:
MEASUREMENT_IMPLEMENTATION_NO_GO

capability_result:
NOT EVALUATED

joint_composition_result:
NOT EXECUTED

mechanism_result:
NOT EXECUTED

q1_readiness:
NO-GO

exact_next_action:
ARCHIVE EVIDENCE AND PERFORM NO FURTHER COMPUTE

## Authoritative evidence and reason for termination

The source is `Yangjunjie-Lin/vlm-synergy-qualification`, branch
`codex/atomic-protocol-v2-clean-rerun`, commit
`eb12bd64da6cb37d4e43631e4f2e08492a9007fb`. The unchanged
[Atomic v2 final decision](../../artifacts/atomic_v2/final_decision.json) records
`MEASUREMENT_IMPLEMENTATION_NO_GO`, `NOT_EVALUATED_MEASUREMENT_FAILURE`, and
`TERMINATE_AFTER_FINAL_MEASUREMENT_FAILURE`.

Fresh Atomic v2 data and the corrected official model renderers passed their
recorded validation gates. Qwen completed and passed all 16 engineering controls,
with four outputs of each direction under both contracts. GLM also completed all
16 controls, but both CLL top answer and constrained generation were `east` in
every case. GLM failed the all-four-answers and nonconstant-output engineering
gates. The permitted renderer correction had already been used. The protocol
therefore stopped before formal Atomic qualification; it did not identify a model
capability failure.

## Scientific boundary

Atomic v1 remains `INVALID_FOR_COHORT_CONCLUSION`. Its two 256-row prediction
files, hashes, and alias-collision, self-relation, renderer and adjudication-error
evidence remain historical evidence, not a clean capability NO-GO.

Atomic v2 formal prediction rows are zero. Joint rows and activation-patching
rows are zero. The formal run lock and Atomic v2 prerun tag were not created.
Qwen's passing engineering control is not full Atomic qualification. No valid GLM
Atomic, cohort capability, Joint composition, cross-modal synergy, or mechanism
conclusion was established. The cause of GLM's constant answer is not identified.

The binding [claim boundary](../../research/program_closeout/final_claim_boundary.yaml)
and [failure taxonomy](../../research/program_closeout/failure_taxonomy.yaml)
prohibit turning a measurement failure into a model failure or turning engineering
completeness into a scientific contribution. The four historical repositories
are separate records; no pooled or unregistered statistical conclusion is made.

## Terminal scope

This terminates the current cross-modal synergy research line. It is not a
permanent negative judgment about every VLM research question. Any future study
with a new answer contract must be completely independent, undergo a new novelty
audit and preregistration, and receive separate authorization. It must not be
called Atomic v3, repair, rescue, or the next stage of this project. This closeout
does not initiate such a study. The current repository will conduct no further
model experiments.

Only evidence preservation, non-model integrity verification, compute shutdown,
and repository closeout are authorized. Existing partial, null and failure
artifacts must remain intact. Terminal reruns may only perform read-only
verification. Operational shutdown and archive claims require actual provider
confirmation, recorded in the [compute shutdown record](../../artifacts/program_closeout/compute_shutdown_record.json)
and [repository archive report](repository_archive_report.md); this document does
not substitute for those confirmations.

Final operational action: `NO_FURTHER_EXPERIMENTS_ON_THIS_LINE`.
