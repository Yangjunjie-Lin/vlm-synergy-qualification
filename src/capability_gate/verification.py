from __future__ import annotations

import json
from typing import Any

from capability_gate.artifacts import build_manifest, read_jsonl, write_json
from capability_gate.paths import ARTIFACTS, REPORTS, ROOT
from capability_gate.provenance import verify_historical_freeze

REQUIRED_PREDICTION_FIELDS = {
    "model",
    "revision",
    "scene_id",
    "task",
    "image_path",
    "image_hash",
    "prompt",
    "prompt_hash",
    "candidate_text",
    "candidate_token_ids",
    "raw_log_likelihood",
    "normalized_log_likelihood",
    "candidate_ranking",
    "top_answer",
    "target_margin",
    "constrained_generation_answer",
    "runtime_seconds",
    "config_hash",
}


def verify_artifacts() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    historical = verify_historical_freeze()
    checks["historical_freeze"] = historical["gate"]
    validation_path = ARTIFACTS / "manifests" / "data_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks["data_validation"] = validation["overall_gate"]
    registry = json.loads(
        (ARTIFACTS / "models" / "frozen_registry.json").read_text(encoding="utf-8")
    )
    checks["three_frozen_families"] = (
        len(registry["models"]) == 3 and len({model["family"] for model in registry["models"]}) == 3
    )
    checks["exact_revisions"] = all(len(model["revision"]) == 40 for model in registry["models"])
    checks["weight_hashes_recorded"] = all(
        len(model.get("expected_weight_sha256", [])) > 0 for model in registry["models"]
    )
    prediction_checks = {}
    for path in (
        ARTIFACTS / "atomic" / "engineering_smoke_predictions.jsonl",
        ARTIFACTS / "atomic" / "predictions.jsonl",
        ARTIFACTS / "joint" / "predictions.jsonl",
        ARTIFACTS / "joint" / "atomic_retention_predictions.jsonl",
    ):
        if path.exists():
            rows = read_jsonl(path)
            prediction_checks[path.relative_to(ROOT).as_posix()] = {
                "rows": len(rows),
                "required_fields": all(REQUIRED_PREDICTION_FIELDS <= row.keys() for row in rows),
            }
    checks["prediction_contracts"] = prediction_checks
    checks["prediction_contract_fields"] = all(
        value["required_fields"] for value in prediction_checks.values()
    )
    checks["no_activation_patching_module"] = not any(
        "patch" in path.name.lower() for path in (ROOT / "src" / "capability_gate").rglob("*.py")
    )
    final_path = ARTIFACTS / "manifests" / "final_decision.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    checks["no_mechanism_claims"] = (
        not final["mechanism_claims"] and not final["activation_patching_executed"]
    )
    checks["reports_exist"] = all(
        (REPORTS / name).exists()
        for name in (
            "atomic_qualification.md",
            "joint_composition_screen.md",
            "final_qualification_decision.md",
        )
    )
    checks["overall_gate"] = all(
        value
        for key, value in checks.items()
        if key not in {"prediction_contracts", "overall_gate"}
    )
    paths = [
        path
        for path in [*ARTIFACTS.glob("**/*.json"), *REPORTS.glob("*.md")]
        if path.name != "artifact_verification.json"
    ]
    result = {
        "schema_version": 1,
        "checks": checks,
        "manifest": build_manifest(ROOT, paths, "final_artifact_verification"),
    }
    write_json(ARTIFACTS / "manifests" / "artifact_verification.json", result)
    if not checks["overall_gate"]:
        raise RuntimeError("artifact verification failed")
    return result
