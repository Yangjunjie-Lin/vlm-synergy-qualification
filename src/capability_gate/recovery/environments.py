from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from capability_gate.artifacts import sha256_file, write_json
from capability_gate.paths import ARTIFACTS, REPORTS, ROOT
from capability_gate.recovery.adapters import DESCRIPTORS

ENV_NAMES = {
    "qwen2_5_vl_7b": "qwen",
    "glm4_1v_9b": "glm",
    "phi4_multimodal_5_6b": "phi",
}
WORKER_NAMES = {
    "qwen2_5_vl_7b": "qwen_worker.py",
    "glm4_1v_9b": "glm_worker.py",
    "phi4_multimodal_5_6b": "phi_worker.py",
}


def worker_python(model_key: str) -> Path:
    return ROOT / "envs" / ENV_NAMES[model_key] / ".venv" / "Scripts" / "python.exe"


def worker_script(model_key: str) -> Path:
    return ROOT / "workers" / WORKER_NAMES[model_key]


def _parse_lock(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(("#", " ", "--"))
    ]


def verify_model_environments() -> dict[str, Any]:
    output_dir = ARTIFACTS / "engineering_recovery/dependency_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_key, env_name in ENV_NAMES.items():
        python = worker_python(model_key)
        lock = ROOT / "envs" / env_name / "requirements.lock"
        environment = yaml.safe_load(
            (ROOT / "envs" / env_name / "environment.yaml").read_text(encoding="utf-8")
        )
        record: dict[str, Any] = {
            "model_key": model_key,
            "python_executable": str(python),
            "python_exists": python.is_file(),
            "lock_sha256": sha256_file(lock),
            "lock_package_count": len(_parse_lock(lock)),
            "lock_all_exact": all("==" in line for line in _parse_lock(lock)),
            "declared_environment": environment,
            "counts_as_model_load_attempt": False,
        }
        if python.is_file():
            completed = subprocess.run(
                [str(python), str(worker_script(model_key)), "--check-dependencies"],
                text=True,
                capture_output=True,
                check=False,
            )
            try:
                preflight = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                preflight = {
                    "status": "DEPENDENCY_PREFLIGHT_FAIL",
                    "error": completed.stderr,
                }
            record["returncode"] = completed.returncode
            record["preflight"] = preflight
            record["status"] = preflight["status"]
        else:
            record["status"] = "DEPENDENCY_PREFLIGHT_FAIL"
            record["preflight"] = {"missing_environment": str(python)}
        write_json(output_dir / f"{env_name}.json", record)
        results[model_key] = record
    overall = all(
        value["status"] == "DEPENDENCY_PREFLIGHT_PASS"
        and value["lock_all_exact"]
        and value["python_exists"]
        for value in results.values()
    )
    result = {
        "schema_version": 1,
        "models": results,
        "overall_gate": overall,
        "counts_as_model_load_attempt": False,
        "counts_as_scientific_attempt": False,
    }
    write_json(ARTIFACTS / "engineering_recovery/manifests/environment_verification.json", result)
    _write_environment_report(result)
    return result


def _write_environment_report(result: dict[str, Any]) -> None:
    REPORTS.joinpath("recovery").mkdir(parents=True, exist_ok=True)
    lines = [
        "# Environment Matrix",
        "",
        "The three workers are isolated. Dependency preflight is not a model-load or scientific attempt.",
        "",
        "| Model | Python | Torch | Transformers | Accelerate | bitsandbytes | Additional | Preflight |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model_key, record in result["models"].items():
        declared = record["declared_environment"]
        additional = ", ".join(
            f"{key}={value}" for key, value in declared.get("additional_dependencies", {}).items()
        )
        lines.append(
            f"| {model_key} | {declared['python']} | {declared['torch']} | "
            f"{declared['transformers']} | {declared['accelerate']} | "
            f"{declared['bitsandbytes']} | {additional} | {record['status']} |"
        )
    lines.extend(
        [
            "",
            (
                "Qwen and GLM use eager attention in their pinned Transformers 4.57.6 workers. "
                "Phi uses the official Python 3.10 / Transformers 4.48.2 stack; FlashAttention "
                "is optional on Windows and the recorded fallback is SDPA or eager."
            ),
            "",
            f"Overall dependency gate: **{result['overall_gate']}**.",
        ]
    )
    (REPORTS / "recovery/environment_matrix.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def environment_commands() -> dict[str, list[str]]:
    return {key: [str(worker_python(key)), str(worker_script(key))] for key in DESCRIPTORS}
