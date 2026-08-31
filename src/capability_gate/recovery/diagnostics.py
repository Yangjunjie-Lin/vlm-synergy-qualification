from __future__ import annotations

import json
from typing import Any

import yaml

from capability_gate.artifacts import sha256_file, write_json
from capability_gate.paths import ARTIFACTS, ROOT


def diagnose_adapters() -> dict[str, Any]:
    old_path = ROOT / "artifacts/manifests/engineering_smoke.json"
    matrix_path = ROOT / "research/engineering_recovery/root_cause_matrix.yaml"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    observed = {}
    for model_key, state in old["model_status"].items():
        errors = [failure["error"] for failure in state["failures"]]
        tracebacks = [failure["traceback"] for failure in state["failures"]]
        observed[model_key] = {
            "attempts": len(errors),
            "errors": errors,
            "same_error_both_attempts": len(set(errors)) == 1,
            "meta_quant_state_path": all(
                "bitsandbytes\\functional.py" in trace
                and "Tensor.item() cannot be called on meta tensors" in trace
                for trace in tracebacks
            ),
            "missing_backoff": all("backoff" in error for error in errors),
        }
    checks = {
        "matrix_frozen_before_load": matrix["frozen_before_new_model_load_attempts"] is True,
        "qwen_shared_meta_root": observed["qwen2_5_vl_7b"]["meta_quant_state_path"],
        "glm_shared_meta_root": observed["glm4_1v_9b"]["meta_quant_state_path"],
        "glm_not_independent_from_qwen": (
            matrix["models"]["glm4_1v_9b"]["independent_from_qwen_root_cause"] is False
        ),
        "phi_missing_backoff_both_attempts": observed["phi4_multimodal_5_6b"]["missing_backoff"],
        "two_attempts_each": all(value["attempts"] == 2 for value in observed.values()),
    }
    result = {
        "schema_version": 1,
        "source_sha256": sha256_file(old_path),
        "matrix_sha256": sha256_file(matrix_path),
        "observed": observed,
        "checks": checks,
        "overall_gate": all(checks.values()),
        "counts_as_model_load_attempt": False,
        "scientific_capability_conclusion": False,
    }
    write_json(ARTIFACTS / "engineering_recovery/manifests/root_cause_diagnosis.json", result)
    if not result["overall_gate"]:
        raise RuntimeError("root-cause diagnosis does not match frozen evidence")
    return result
