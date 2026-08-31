from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ENGINEERING_STATUSES = frozenset(
    {
        "ENGINEERING_RECOVERY_PASS",
        "BLOCKED_BY_COMPUTE",
        "BLOCKED_BY_MODEL_ADAPTER",
        "BLOCKED_BY_DEPENDENCY",
        "MEASUREMENT_IMPLEMENTATION_FAIL",
    }
)

_DECISION_POLICIES: dict[str, dict[str, str]] = {
    "BLOCKED_BY_MODEL_ADAPTER": {
        "q1_potential": "NOT_EVALUATED",
        "next_action": "ROOT_CAUSE_ADAPTER_RECOVERY",
    },
    "BLOCKED_BY_COMPUTE": {
        "q1_potential": "NOT_EVALUATED",
        "next_action": "MIGRATE_SAME_FROZEN_COHORT_TO_ADEQUATE_COMPUTE",
    },
    "CAPABILITY_COHORT_NO_GO": {
        "q1_potential": "NO_FEASIBLE_COHORT",
        "next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
    },
    "JOINT_COMPOSITION_NO_GO": {
        "q1_potential": "NO_FEASIBLE_COHORT",
        "next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
    },
    "MEASUREMENT_CONTRACT_NO_GO": {
        "q1_potential": "NO_FEASIBLE_COHORT",
        "next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
    },
    "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION": {
        "q1_potential": "MECHANISTIC_STUDY_FEASIBLE",
        "next_action": "BUILD_HELD_OUT_MECHANISTIC_PILOT",
    },
}


def decision_policy(decision: str) -> dict[str, str]:
    """Return the frozen scientific interpretation for an adjudication label."""

    try:
        return dict(_DECISION_POLICIES[decision])
    except KeyError as error:
        raise ValueError(f"unsupported recovery decision: {decision}") from error


def engineering_cohort_decision(model_statuses: Mapping[str, str]) -> dict[str, Any]:
    """Adjudicate engineering eligibility without making a capability claim."""

    if set(model_statuses) != {"qwen2_5_vl_7b", "glm4_1v_9b", "phi4_multimodal_5_6b"}:
        raise ValueError("engineering recovery must adjudicate exactly the three frozen models")
    invalid = {key: value for key, value in model_statuses.items() if value not in ENGINEERING_STATUSES}
    if invalid:
        raise ValueError(f"invalid engineering status values: {invalid}")
    passed = sorted(
        key for key, value in model_statuses.items() if value == "ENGINEERING_RECOVERY_PASS"
    )
    blocked = {key: value for key, value in model_statuses.items() if key not in passed}
    if len(passed) == 3:
        gate = "ENGINEERING_COHORT_GO"
    elif len(passed) == 2:
        gate = "PARTIAL_ENGINEERING_COHORT_GO"
    else:
        gate = "ENGINEERING_COHORT_NO_GO"
    return {
        "decision": gate,
        "passed_count": len(passed),
        "passed_models": passed,
        "blocked_models": blocked,
        "atomic_authorized": len(passed) >= 2,
        "atomic_authorized_models": passed if len(passed) >= 2 else [],
        "scientific_capability_conclusion": False,
        "activation_patching_authorized": False,
        "fourth_model_authorized": False,
    }
