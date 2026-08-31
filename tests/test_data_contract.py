import json

import pytest

from capability_gate.paths import ARTIFACTS


def test_generated_data_contract_if_present() -> None:
    path = ARTIFACTS / "manifests" / "data_validation.json"
    if not path.exists():
        pytest.skip("data is generated only after the model registry is frozen")
    validation = json.loads(path.read_text(encoding="utf-8"))
    assert validation["overall_gate"]
    assert validation["counts"] == {
        "engineering_smoke": 12,
        "atomic": 256,
        "joint_base_quartets": 128,
        "joint_conditions": 512,
    }
    for task in validation["atomic_tasks"].values():
        assert task["symbolic_accuracy"] == 1.0
        assert max(task["shortcut_accuracy"].values()) <= 0.30
