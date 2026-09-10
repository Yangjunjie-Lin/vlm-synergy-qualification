# Evidence lineage and preservation boundary

The authoritative pre-closeout source is commit
`eb12bd64da6cb37d4e43631e4f2e08492a9007fb` on
`codex/atomic-protocol-v2-clean-rerun` in
`Yangjunjie-Lin/vlm-synergy-qualification`. Closeout adds provenance and terminal
policy; it does not recalculate scientific outcomes, reinterpret controls as
formal tests, or alter historical evidence.

## Four independent historical records

| Record | Preserved status | Admissible role and boundary |
|---|---|---|
| ReCoAlign | `ARCHIVED_NEGATIVE_EVIDENCE_RELEASE` | `MOTIVATING_NEGATIVE_CASE_ONLY`; no modification or unarchive |
| VLM Construct Audit | `TIER0_INCONCLUSIVE` | Scientific pilot `NOT_AUTHORIZED`; no modification or unarchive |
| SynergyTrace | `BEHAVIORAL_NO_GO` | Activation patching `NOT_EXECUTED`; no modification or unarchive |
| CapabilityGate / VLM Synergy Qualification | `MEASUREMENT_IMPLEMENTATION_NO_GO` | Corrected Atomic v2 formal qualification, Joint and activation patching `NOT_EXECUTED` |

These records are not pooled into a statistical conclusion. Their naming and
lineage do not transfer the admissibility of one study to another. The
[machine-readable lineage](../../research/program_closeout/research_lineage.yaml)
defines these boundaries; independently observed repository identifiers, commits,
archive flags and hashes belong in the
[cross-repository provenance record](../../artifacts/program_closeout/cross_repository_provenance.json).

## Atomic v1: preserved but scientifically invalid for a cohort conclusion

The frozen v1 commit is `eca06a65e9ea7c384ff17859bcd9b28c79630219`, protected by
annotated tag `capability-gate-atomic-v1-protocol-invalid-2026-09-10`. Its recorded
annotated tag object is `bce668ac9f95f2cd0ee4639c45985eff478e3752`.

| Formal historical predictions | Rows | Recorded SHA-256 |
|---|---:|---|
| Qwen | 256 | `d41ad33bece21d308352d489a8d6deb4aaa2df371aab9b59749137f990772dba` |
| GLM | 256 | `cd4e4be6649d4e604bb201698549fc2a89328215ffff15ca6d83e74fec6aebf5` |

The [v1 protocol adjudication](../../research/atomic_v2/atomic_v1_protocol_adjudication.yaml)
and [v2 execution summary](../atomic_v2/execution_summary.md) preserve 28
alias-collision scenes, four direct-visual self-relations with directional
targets, all 256 old GLM prompts rendered with Python-object representation,
degenerate GLM outputs and classification inconsistency. The historical label
`CAPABILITY_COHORT_NO_GO` is retained as a historical artifact, but scientific
admissibility is `INVALID_FOR_COHORT_CONCLUSION`. It is not a clean capability
NO-GO. The old 15 local partial rows were not reused as formal v2 results.

## Atomic v2: valid engineering observations, no formal capability endpoint

The [data validation](../atomic_v2/data_validation.md),
[renderer validation](../atomic_v2/renderer_validation.md),
[contract controls](../atomic_v2/contract_control.md), and
[final decision](../../artifacts/atomic_v2/final_decision.json) form the source
chain. Fresh Atomic data comprise 256 scenes across four tasks, with no v1 UUID
overlap, no alias collision or self-relation, balanced answers and option
positions, and symbolic oracle 1.00. The stored pooled held-out shortcut gates
passed; these finite checks do not establish universal absence of shortcuts.

The old Joint data contained two invalid quartets. A fresh 128-quartet dataset
was frozen before model controls, but no Joint screen was run. Generated data
must not be counted as model prediction rows.

Both model-specific renderer validations passed after the recorded permitted
correction. Initial renderer-failure artifacts and the archive map remain part
of the evidence. Qwen passed 16 engineering controls; GLM completed 16 controls
but returned `east` in all CLL and constrained-generation selections. Thus
formal Atomic v2, Joint and activation patching remain unexecuted. The missing
formal run lock and absent prerun tag are the correct consequence of the gate,
not evidence to be filled by a retrospective run.

## Terminal provenance

The terminal tag is named
`capability-gate-terminal-measurement-no-go-2026-09-10`. Its actual object and
target, final commit, remote branch, GitHub API, `ls-remote` and fresh-clone
verification are recorded by the
[integrity verification](../../artifacts/program_closeout/final_integrity_verification.json)
and [repository archive report](repository_archive_report.md). No future SHA,
successful archival or shutdown is assumed here.

The [canonical artifact index](../../artifacts/program_closeout/canonical_artifact_index.json)
and [terminal manifest](../../artifacts/program_closeout/terminal_manifest.json)
provide the closeout hash inventory and recomputation contract. All earlier
predictions, manifests, completion records, stderr, partial/null/failure artifacts,
adjudications, final reports and tags must remain unchanged and reachable.
