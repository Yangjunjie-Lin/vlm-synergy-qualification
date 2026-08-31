# CapabilityGate

**Capability-Qualified Screening for Cross-Modal Composition in VLMs**

CapabilityGate is a one-shot, preregistered qualification screen for three
independent, locally runnable open-weight vision-language-model families. It
first tests four directly measured atomic capabilities. The joint composition
screen is unreachable unless at least two families pass every atomic gate.

This repository is independent from the frozen SynergyTrace pilot. It neither
amends nor reruns that pilot and contains no activation-patching experiment.

## Commands

```text
python -m capability_gate freeze-models
python -m capability_gate generate-data
python -m capability_gate validate-data
python -m capability_gate run-engineering-smoke
python -m capability_gate run-atomic-qualification
python -m capability_gate adjudicate-atomic
python -m capability_gate run-joint-screen
python -m capability_gate analyze-joint
python -m capability_gate build-report
python -m capability_gate verify-artifacts
```

Install with `python -m pip install -e .[inference,test]`, then run the complete
stopping pipeline with `make capability-screen`. Model weights and generated
prediction rows are intentionally excluded from Git; signed manifests and
reports remain tracked.

## Scientific boundary

The screen can establish model-cohort eligibility for a later, separately
preregistered mechanism study. Behavioral qualification is not circuit or
mechanistic evidence. See `docs/claim_boundary.md` and the frozen stop rules.

