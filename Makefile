.PHONY: capability-screen test

capability-screen:
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

test:
	python -m pytest

