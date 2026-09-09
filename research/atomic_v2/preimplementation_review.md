# Independent preimplementation review — no v2 outcomes

These issues must be resolved explicitly before data generation or formal freeze.
They are not changes already implemented, and no data/renderer gate is claimed.

1. Alias uniqueness applies to distinct entity identities. Repeating a query's
   name in a question and in its image label is not an alias collision. Blindly
   concatenating query/reference names with all entity names double-counts the
   same identities.
2. The frozen bridge construct selects an **unlabeled** image object by a unique
   color-shape descriptor. Painting the text's bridge alias directly on the target
   would change it into name matching. Preserve the frozen construct and specify
   internal entity aliases versus rendered display labels explicitly.
3. Full text-task `question` includes the legitimate premise. A question-only
   shortcut excludes evidence; a metadata shortcut must explicitly exclude target,
   symbolic answer, coordinates and other legitimate answer-bearing evidence.
   Freeze feature allowlists and actual grouped cross-validation, not the old
   in-sample conditional-majority bound. Include descriptor-aware text-only probes
   for binding: the old `(template, color, shape)` shortcut bound is 0.50.
4. A rendered image placeholder appears once; tokenized visual patch IDs expand
   to many tokens. Snapshots must distinguish those two representations. The v2
   text-only requirement must use no image tensor/token while retaining frozen
   question semantics; v1 supplied a matched decorative image.
5. Preserve undefined kappa for identical constant marginal predictions. Do not
   manufacture kappa=1. No exact disagreement exists in that case. The old code
   gates model-wide kappa, while task-level kappa is reported; avoid silently
   changing the scope of the frozen gate.
6. The frozen Joint statistical implementation and YAML diagnostic descriptions
   are not identical. Preserve statistical definitions and obtain direction before
   any material change; do not silently reinterpret exclusions after outcomes.
7. Incomplete runtime, absent compute, or unrun engineering controls cannot be
   classified as a valid model-capability NO-GO.

Compute preflight found the previous 3090 instance **powered off**. Restarting the
same pay-as-you-go instance requires confirmation before the billable UI action.
No v2 data, renderer change, control output, formal output or new scientific
decision exists at this pause. No automatic task or background model run was
created. No historical repository state was changed. Remote `vlm-synergy-trace`
was observed already unarchived; the other two historical repositories remain
archived. This task did not change any archive setting.
