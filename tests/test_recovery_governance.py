from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from capability_gate.artifacts import sha256_file
from capability_gate.recovery.formal import classify_formal_runtime_failure
from capability_gate.recovery.governance import (
    authorize_atomic,
    authorize_joint,
    decision_policy,
    engineering_cohort_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def test_engineering_block_is_not_mapped_to_scientific_no_go() -> None:
    assert decision_policy("BLOCKED_BY_MODEL_ADAPTER") == {
        "q1_potential": "NOT_EVALUATED",
        "next_action": "ROOT_CAUSE_ADAPTER_RECOVERY",
    }
    assert decision_policy("CAPABILITY_COHORT_NO_GO")["q1_potential"] == "NO_FEASIBLE_COHORT"


def test_atomic_requires_two_engineering_passes() -> None:
    statuses = {
        "qwen2_5_vl_7b": "ENGINEERING_RECOVERY_PASS",
        "glm4_1v_9b": "BLOCKED_BY_COMPUTE",
        "phi4_multimodal_5_6b": "BLOCKED_BY_DEPENDENCY",
    }
    with pytest.raises(RuntimeError, match="ENGINEERING_GATE"):
        authorize_atomic(engineering_cohort_decision(statuses))
    statuses["glm4_1v_9b"] = "ENGINEERING_RECOVERY_PASS"
    assert len(authorize_atomic(engineering_cohort_decision(statuses))) == 2


def test_joint_requires_two_atomic_qualifications() -> None:
    with pytest.raises(RuntimeError, match="ATOMIC_GATE"):
        authorize_joint(
            {
                "qwen2_5_vl_7b": "ATOMICALLY_QUALIFIED",
                "glm4_1v_9b": "ATOMIC_VISUAL_FAIL",
            }
        )
    assert len(
        authorize_joint(
            {
                "qwen2_5_vl_7b": "ATOMICALLY_QUALIFIED",
                "glm4_1v_9b": "ATOMICALLY_QUALIFIED",
            }
        )
    ) == 2


def test_historical_commit_tag_and_blocker_artifacts_are_unchanged() -> None:
    target = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-list", "-n", "1", "capability-gate-adapter-block-2026-08-31"],
        text=True,
    ).strip()
    assert target == "32a8333bf0106579834fb7a0747c81d67b49ac10"
    assert sha256_file(ROOT / "reports" / "final_qualification_decision.md") == (
        "c6b62482b2a40b8d5ba33fb535e31383e7779c18f730bb0ecace4fae88b65098"
    )
    assert sha256_file(ROOT / "artifacts" / "manifests" / "engineering_smoke.json") == (
        "0ba3c4df6e65c41d831f86f52db0ff3028be5532814c26e0351402e17f8b0ba3"
    )


def test_formal_outputs_remain_zero_before_engineering_go() -> None:
    atomic = json.loads((ROOT / "artifacts/manifests/atomic_run.json").read_text())
    joint = json.loads((ROOT / "artifacts/manifests/joint_screen_status.json").read_text())
    assert atomic["model_forward_passes"] == 0
    assert joint["model_forward_passes"] == 0
    assert not (ROOT / "artifacts/atomic/predictions.jsonl").exists()
    assert not (ROOT / "artifacts/joint/predictions.jsonl").exists()


def test_activation_patching_is_forbidden_and_absent() -> None:
    boundary = (ROOT / "research/engineering_recovery/recovery_amendment.yaml").read_text()
    assert "activation_patching: FORBIDDEN" in boundary
    assert not any("patch" in path.name.lower() for path in (ROOT / "src/capability_gate").rglob("*.py"))


def test_formal_cuda_kernel_failure_is_compute_not_capability() -> None:
    error = RuntimeError("worker exited without a response")
    stderr = "Error an illegal memory access was encountered in bitsandbytes/csrc/ops.cu"
    assert classify_formal_runtime_failure(error, stderr) == "BLOCKED_BY_COMPUTE"


def test_unknown_formal_worker_failure_remains_adapter_block() -> None:
    error = RuntimeError("worker exited without a response")
    assert classify_formal_runtime_failure(error, "") == "BLOCKED_BY_MODEL_ADAPTER"
