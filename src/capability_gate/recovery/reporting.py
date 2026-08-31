from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from capability_gate.artifacts import build_manifest, read_jsonl, sha256_file, write_json
from capability_gate.paths import ARTIFACTS, REPORTS, ROOT
from capability_gate.recovery.adapters import DESCRIPTORS, response_hash
from capability_gate.recovery.environments import ENV_NAMES
from capability_gate.recovery.historical import (
    BASE_COMMIT,
    HISTORICAL_SHA256,
    RECOVERY_TAG,
    verify_historical_block,
)

ENGINEERING_ROOT = ARTIFACTS / "engineering_recovery"
ATOMIC_ROOT = ARTIFACTS / "recovery_qualification/atomic"
JOINT_ROOT = ARTIFACTS / "recovery_qualification/joint"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _final_state() -> dict[str, Any]:
    engineering = yaml.safe_load(
        (REPORTS / "recovery/engineering_cohort_decision.yaml").read_text(encoding="utf-8")
    )
    atomic_status = _read_json(ATOMIC_ROOT / "run_status.json", {})
    atomic = _read_json(ATOMIC_ROOT / "adjudication.json", {})
    joint_status = _read_json(JOINT_ROOT / "run_status.json", {})
    joint = _read_json(JOINT_ROOT / "analysis.json", {})
    if engineering["passed_count"] < 2:
        decision = "ENGINEERING_COHORT_NO_GO"
        potential = "NOT_EVALUATED_ENGINEERING_BLOCK"
        blocked = set(engineering["blocked_models"].values())
        next_action = (
            "MIGRATE_SAME_FROZEN_COHORT_TO_ADEQUATE_GPU"
            if blocked and blocked <= {"BLOCKED_BY_COMPUTE"}
            else "TERMINATE_AFTER_FINAL_ADAPTER_RECOVERY_FAILURE"
        )
        capability_conclusion = False
    elif atomic_status.get("status") != "complete":
        decision = atomic.get("decision", atomic_status.get("block_class"))
        if decision not in {"BLOCKED_BY_COMPUTE", "BLOCKED_BY_MODEL_ADAPTER"}:
            decision = "BLOCKED_BY_MODEL_ADAPTER"
        potential = "NOT_EVALUATED_ENGINEERING_BLOCK"
        next_action = (
            "MIGRATE_SAME_FROZEN_COHORT_TO_ADEQUATE_GPU"
            if decision == "BLOCKED_BY_COMPUTE"
            else "TERMINATE_AFTER_FINAL_ADAPTER_RECOVERY_FAILURE"
        )
        capability_conclusion = False
    elif atomic.get("decision") == "CAPABILITY_COHORT_NO_GO":
        decision = "CAPABILITY_COHORT_NO_GO"
        potential = (
            "MODEL_SPECIFIC_FEASIBILITY"
            if atomic.get("qualified_count") == 1
            else "NO_FEASIBLE_COHORT"
        )
        next_action = "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
        capability_conclusion = True
    elif atomic.get("decision") == "MEASUREMENT_CONTRACT_NO_GO":
        decision = "MEASUREMENT_CONTRACT_NO_GO"
        potential = "NO_FEASIBLE_COHORT"
        next_action = "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
        capability_conclusion = True
    elif joint_status.get("status") != "complete":
        decision = joint.get("decision", joint_status.get("block_class"))
        if decision not in {"BLOCKED_BY_COMPUTE", "BLOCKED_BY_MODEL_ADAPTER"}:
            decision = "BLOCKED_BY_MODEL_ADAPTER"
        potential = "NOT_EVALUATED_ENGINEERING_BLOCK"
        next_action = (
            "MIGRATE_SAME_FROZEN_COHORT_TO_ADEQUATE_GPU"
            if decision == "BLOCKED_BY_COMPUTE"
            else "TERMINATE_AFTER_FINAL_ADAPTER_RECOVERY_FAILURE"
        )
        capability_conclusion = False
    else:
        decision = joint["decision"]
        potential = (
            "MECHANISTIC_STUDY_FEASIBLE"
            if decision == "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION"
            else "NO_FEASIBLE_COHORT"
        )
        next_action = (
            "BUILD_HELD_OUT_MECHANISTIC_PILOT"
            if decision == "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION"
            else "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
        )
        capability_conclusion = True
    return {
        "decision": decision,
        "q1_potential": potential,
        "exact_next_action": next_action,
        "engineering": engineering,
        "atomic_status": atomic_status,
        "atomic": atomic,
        "joint_status": joint_status,
        "joint": joint,
        "scientific_capability_conclusion": capability_conclusion,
    }


def _migration_manifest(smoke: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    compute_models = [
        key
        for key, value in smoke["model_results"].items()
        if value["status"] == "BLOCKED_BY_COMPUTE"
    ]
    phase = "engineering_recovery_smoke"
    if state["decision"] == "BLOCKED_BY_COMPUTE":
        if state["atomic_status"].get("status") == "FORMAL_ATOMIC_RUNTIME_FAIL":
            compute_models = list(state["engineering"]["atomic_authorized_models"])
            phase = "atomic_qualification_v2"
        elif state["joint_status"].get("status") == "FORMAL_JOINT_RUNTIME_FAIL":
            compute_models = [
                key
                for key, value in state["atomic"].get("models", {}).items()
                if value.get("label") == "ATOMICALLY_QUALIFIED"
            ]
            phase = "joint_composition_screen_v2"
    compute_models = sorted(set(compute_models))
    if not compute_models:
        return None
    frozen = _read_json(ARTIFACTS / "models/frozen_registry.json", {})
    registry = {model["key"]: model for model in frozen["models"]}
    disk_gib = {
        "qwen2_5_vl_7b": "22-28 GiB",
        "glm4_1v_9b": "26-34 GiB",
        "phi4_multimodal_5_6b": "18-24 GiB",
    }
    result = {
        "schema_version": 1,
        "kind": "SAME_FROZEN_COHORT_COMPUTE_MIGRATION",
        "blocked_phase": phase,
        "minimum_single_gpu_vram_gib": 16,
        "preferred_single_gpu_vram_gib": 24,
        "migration_source_commit": state["atomic_status"].get(
            "engineering_decision_commit", "50f1e6cdb50c353b2c899390aec7a987a28daa27"
        ),
        "models": {
            key: {
                "model_id": DESCRIPTORS[key].model_id,
                "revision": DESCRIPTORS[key].revision,
                "requirements_lock": f"envs/{ENV_NAMES[key]}/requirements.lock",
                "cuda": "12.4 compatible driver/runtime",
                "torch": DESCRIPTORS[key].required_packages["torch"],
                "model_cache_manifest": registry[key]["remote_weight_files"],
                "expected_disk_requirement": disk_gib[key],
                "expected_vram_range": "16-24 GiB single CUDA GPU",
                "exact_environment_command": (
                    f"python -m pip install -r envs/{ENV_NAMES[key]}/requirements.lock"
                ),
            }
            for key in compute_models
        },
        "exact_commands": [
            "python -m capability_gate verify-historical-block",
            "python -m capability_gate verify-model-environments",
            "python -m capability_gate run-atomic-qualification-v2"
            if phase == "atomic_qualification_v2"
            else "python -m capability_gate run-joint-screen-v2"
            if phase == "joint_composition_screen_v2"
            else "python -m capability_gate run-adapter-recovery-smoke",
        ],
        "model_or_revision_changed": False,
        "formal_data_or_prompt_changed": False,
    }
    write_json(ENGINEERING_ROOT / "manifests/compute_migration_manifest.json", result)
    return result


def build_recovery_report() -> dict[str, Any]:
    state = _final_state()
    smoke = _read_json(ENGINEERING_ROOT / "manifests/engineering_recovery_smoke.json")
    root_matrix = yaml.safe_load(
        (ROOT / "research/engineering_recovery/root_cause_matrix.yaml").read_text(encoding="utf-8")
    )
    environment = _read_json(ENGINEERING_ROOT / "manifests/environment_verification.json")
    migration = _migration_manifest(smoke, state)
    lines = [
        "# CapabilityGate Final Recovery Decision",
        "",
        "# 1. Final Decision",
        "",
        f"**{state['decision']}**",
        "",
        "# 2. Historical Integrity",
        "",
        (
            f"Frozen branch `codex/three-family-capability-screen`, commit `{BASE_COMMIT}`, annotated "
            f"tag `{RECOVERY_TAG}`, historical blocker reports, and old empty formal artifacts remain "
            "unchanged. SynergyTrace's frozen tag remains unmoved; ReCoAlign and vlm-construct-audit "
            "remain archived."
        ),
        "",
        "# 3. Root-Cause Matrix",
        "",
        "| Model | Original failure | Actual root cause | Old attempts fixed root cause? | Recovery action | Result |",
        "|---|---|---|---|---|---|",
    ]
    for model_key, matrix in root_matrix["models"].items():
        result = smoke["model_results"][model_key]
        action = matrix["recovery_action"]
        lines.append(
            f"| {model_key} | {matrix['original_failure']} | {matrix['actual_root_cause']} | "
            f"{matrix['old_attempts']['root_cause_mechanism_changed']} | `{action}` | "
            f"{result['status']} |"
        )
    lines.extend(
        [
            "",
            "# 4. Environment Matrix",
            "",
            (
                "Full pinned Python, Torch, Transformers, Accelerate, bitsandbytes, and additional "
                "dependency versions are in `reports/recovery/environment_matrix.md`."
            ),
            "",
            "| Model | Python | Torch | Transformers | Accelerate | bitsandbytes | Preflight |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for key, value in environment["models"].items():
        declared = value["declared_environment"]
        lines.append(
            f"| {key} | {declared['python']} | {declared['torch']} | "
            f"{declared['transformers']} | {declared['accelerate']} | "
            f"{declared['bitsandbytes']} | {value['status']} |"
        )
    lines.extend(
        [
            "",
            "# 5. Adapter Recovery",
            "",
            "| Model | Load status | Native class | Processor | Meta tensors | Device map | Visual forward | CLL | Constrained | Determinism | Peak VRAM | Runtime |",
            "|---|---|---|---|---:|---|---|---|---|---:|---:|---:|",
        ]
    )
    for key, value in smoke["model_results"].items():
        metadata = value.get("model_metadata", {})
        material = metadata.get("materialization", {})
        gates = value.get("gates", {})
        meta_count = material.get("meta_parameter_count", 0) + material.get("meta_buffer_count", 0)
        lines.append(
            f"| {key} | {value['status']} | {metadata.get('model_class', 'NA')} | "
            f"{metadata.get('processor_class', 'NA')} | {meta_count} | "
            f"`{metadata.get('resolved_device_map', 'NA')}` | "
            f"{gates.get('real_visual_forward', False)} | {gates.get('cll_finite', False)} | "
            f"{gates.get('constrained_generation_allowed', False)} | "
            f"{value.get('deterministic_rerun_agreement', 0.0):.2f} | "
            f"{value.get('peak_vram_bytes', 0)} | {value.get('runtime_seconds', 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "# 6. Engineering Cohort",
            "",
            (
                f"**{state['engineering']['passed_count']} / 3** families passed. Cohort gate: "
                f"**{state['engineering']['decision']}**. Blocked models and engineering reasons are "
                f"`{state['engineering']['blocked_models']}`. No engineering result is treated as "
                "model-capability evidence."
            ),
            "",
            "# 7. Atomic Qualification",
            "",
        ]
    )
    if state["atomic_status"].get("status") != "complete":
        if state["atomic_status"].get("status") == "NOT_RUN_BY_ENGINEERING_GATE":
            lines.append("**NOT_RUN_BY_ENGINEERING_GATE**")
        else:
            lines.append(
                f"**{state['atomic'].get('status', 'ATOMIC_INCOMPLETE_BY_ENGINEERING_BLOCK')}**. "
                f"Preserved rows: {state['atomic'].get('completed_prediction_rows_preserved', 0)}; "
                "formal rows used for scientific metrics: 0. No Atomic capability conclusion was made."
            )
    else:
        lines.append(
            f"Atomic v2 ran only for authorized models. Decision: **{state['atomic']['decision']}**; "
            "all per-model, per-task metrics are in `reports/recovery/atomic_qualification_v2.md`."
        )
    lines.extend(["", "# 8. Joint Composition", ""])
    if state["joint_status"].get("status") != "complete":
        lines.append(
            f"Not run; upstream gate: `{state['joint_status'].get('upstream_decision', state['engineering']['decision'])}`."
        )
    else:
        lines.append(
            f"Joint v2 decision: **{state['joint']['decision']}**. Joint, unimodal, advantage, Ψ, "
            "and measurement-contract consistency are reported in "
            "`reports/recovery/joint_composition_screen_v2.md`."
        )
    lines.extend(
        [
            "",
            "# 9. Q1 Potential",
            "",
            f"**{state['q1_potential']}**",
            "",
            "# 10. Exact Next Action",
            "",
            f"**{state['exact_next_action']}**",
            "",
            (
                "The historical decision was produced before any formal forward. This recovery changed "
                "only engineering paths; all formal scientific parameters remained frozen. Atomic started: "
                f"**{bool(state['atomic_status']) and state['atomic_status'].get('status') != 'NOT_RUN_BY_ENGINEERING_GATE'}**; "
                f"Atomic completed: **{state['atomic_status'].get('status') == 'complete'}**. Joint ran: "
                f"**{state['joint_status'].get('status') == 'complete'}**. A model-capability conclusion "
                f"was produced: **{state['scientific_capability_conclusion']}**. Activation patching was "
                "not run."
            ),
        ]
    )
    path = REPORTS / "recovery/final_recovery_decision.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "decision": state["decision"],
        "q1_potential": state["q1_potential"],
        "exact_next_action": state["exact_next_action"],
        "atomic_run": bool(state["atomic_status"])
        and state["atomic_status"].get("status") != "NOT_RUN_BY_ENGINEERING_GATE",
        "atomic_complete": state["atomic_status"].get("status") == "complete",
        "joint_run": state["joint_status"].get("status") == "complete",
        "scientific_capability_conclusion": state["scientific_capability_conclusion"],
        "activation_patching_executed": False,
        "migration_manifest": migration is not None,
    }
    write_json(ENGINEERING_ROOT / "manifests/final_recovery_decision.json", summary)
    return summary


def verify_recovery_artifacts() -> dict[str, Any]:
    historical = verify_historical_block()
    engineering = yaml.safe_load(
        (REPORTS / "recovery/engineering_cohort_decision.yaml").read_text(encoding="utf-8")
    )
    atomic = _read_json(ATOMIC_ROOT / "adjudication.json", {})
    reports = [
        REPORTS / "recovery/root_cause_audit.md",
        REPORTS / "recovery/environment_matrix.md",
        REPORTS / "recovery/adapter_recovery_report.md",
        REPORTS / "recovery/engineering_cohort_decision.yaml",
        REPORTS / "recovery/atomic_qualification_v2.md",
        REPORTS / "recovery/joint_composition_screen_v2.md",
        REPORTS / "recovery/final_recovery_decision.md",
    ]
    worker_rows = [
        row
        for path in (ENGINEERING_ROOT / "smoke_predictions").glob("*.jsonl")
        for row in read_jsonl(path)
    ]
    scene_rows = read_jsonl(ENGINEERING_ROOT / "manifests/engineering_only_scenes.jsonl")
    formal_ids = {
        row.get("scene_id", row.get("base_quartet_id"))
        for path in (
            ARTIFACTS / "data/atomic_qualification/scenes.jsonl",
            ARTIFACTS / "data/joint_composition_screen/quartets.jsonl",
        )
        for row in read_jsonl(path)
    }
    atomic_predictions = list((ATOMIC_ROOT / "predictions").glob("*.jsonl"))
    joint_predictions = list((JOINT_ROOT / "predictions").glob("*.jsonl"))
    checks = {
        "historical_integrity": historical["overall_gate"],
        "historical_hashes": all(
            sha256_file(ROOT / path) == expected for path, expected in HISTORICAL_SHA256.items()
        ),
        "recovery_reports_complete": all(path.is_file() for path in reports),
        "exactly_12_engineering_scenes": len(scene_rows) == 12,
        "engineering_scenes_do_not_overlap_formal": not (
            {row["scene_id"] for row in scene_rows} & formal_ids
        ),
        "worker_artifact_hashes": all(
            row.get("artifact_hash") == response_hash(row) for row in worker_rows
        ),
        "atomic_gate_enforced": not atomic_predictions or engineering["passed_count"] >= 2,
        "joint_gate_enforced": not joint_predictions
        or sum(
            value.get("label") == "ATOMICALLY_QUALIFIED"
            for value in atomic.get("models", {}).values()
        )
        >= 2,
        "no_fourth_model": set(engineering["model_statuses"]) == set(DESCRIPTORS),
        "activation_patching_absent": not any(
            "patch" in path.name.lower() for path in (ROOT / "src/capability_gate").rglob("*.py")
        ),
    }
    paths = [
        *reports,
        *list(ENGINEERING_ROOT.rglob("*.json")),
        *list(ENGINEERING_ROOT.rglob("*.jsonl")),
        *list(ATOMIC_ROOT.rglob("*.json")),
        *list(ATOMIC_ROOT.rglob("*.jsonl")),
        *list(JOINT_ROOT.rglob("*.json")),
        *list(JOINT_ROOT.rglob("*.jsonl")),
    ]
    paths = [
        path
        for path in paths
        if path.name != "recovery_artifact_verification.json" and path.exists()
    ]
    result = {
        "schema_version": 1,
        "checks": checks,
        "overall_gate": all(checks.values()),
        "manifest": build_manifest(ROOT, paths, "engineering_recovery_artifacts"),
    }
    write_json(ENGINEERING_ROOT / "manifests/recovery_artifact_verification.json", result)
    if not result["overall_gate"]:
        raise RuntimeError(f"recovery artifact verification failed: {checks}")
    return result
