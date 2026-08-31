from __future__ import annotations

import importlib.metadata
import json
import subprocess
from typing import Any

import yaml

from capability_gate.artifacts import config_hash, runtime_metadata, write_json
from capability_gate.paths import ARTIFACTS, CONFIGS, ROOT, ensure_layout


def load_model_specs() -> list[dict[str, Any]]:
    path = CONFIGS / "models.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = payload["models"]
    if len(specs) != 3:
        raise RuntimeError("exactly three model families must be frozen")
    if len({spec["family"] for spec in specs}) != 3:
        raise RuntimeError("the three models must come from independent families")
    for spec in specs:
        revision = str(spec["revision"])
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise RuntimeError(f"{spec['key']} does not use an exact 40-character revision")
        if not 4.0 <= float(spec["parameters_billions"]) <= 12.0:
            raise RuntimeError(f"{spec['key']} is outside the frozen 4B-12B range")
        if spec.get("selection_used_task_accuracy"):
            raise RuntimeError("task-accuracy-based checkpoint selection is forbidden")
    return specs


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    inventory = []
    for line in output.splitlines():
        name, total, free, driver = [part.strip() for part in line.split(",")]
        inventory.append(
            {
                "name": name,
                "memory_total_mib": total,
                "memory_free_mib": free,
                "driver": driver,
            }
        )
    return inventory


def freeze_registry(verify_remote: bool = True) -> dict[str, Any]:
    ensure_layout()
    existing = ARTIFACTS / "models" / "frozen_registry.json"
    if existing.exists():
        return json.loads(existing.read_text(encoding="utf-8"))
    specs = load_model_specs()
    models = []
    for spec in specs:
        record = dict(spec)
        if verify_remote:
            try:
                from huggingface_hub import HfApi

                info = HfApi().model_info(
                    spec["model_id"], revision=spec["revision"], files_metadata=True
                )
                record["remote_verified_sha"] = info.sha
                record["remote_gated"] = bool(info.gated)
                record["remote_private"] = bool(info.private)
                if info.sha != spec["revision"]:
                    raise RuntimeError(f"remote revision mismatch for {spec['key']}")
                weights = []
                for sibling in info.siblings or []:
                    if sibling.rfilename.endswith((".safetensors", ".bin")):
                        lfs = getattr(sibling, "lfs", None)
                        weights.append(
                            {
                                "path": sibling.rfilename,
                                "bytes": getattr(lfs, "size", None) if lfs else None,
                                "sha256": getattr(lfs, "sha256", None) if lfs else None,
                            }
                        )
                record["remote_weight_files"] = weights
            except Exception as error:  # noqa: BLE001 - preserve remote audit failure
                record["remote_verification_error"] = f"{type(error).__name__}: {error}"
        models.append(record)
    try:
        import torch

        torch_state = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
        }
    except Exception as error:  # noqa: BLE001 - torch may fail through native loaders
        torch_state = {"error": f"{type(error).__name__}: {error}"}
    result = {
        "schema_version": 1,
        "registry_status": "FROZEN_BEFORE_TASK_OUTPUTS",
        "selection_basis": [
            "official model card",
            "license",
            "parameter count",
            "architecture independence",
            "current-hardware runnability",
        ],
        "selection_used_task_accuracy": False,
        "models": models,
        "hardware": {"gpus": _gpu_inventory(), "torch": torch_state},
        "software": _package_versions(
            [
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "bitsandbytes",
                "huggingface-hub",
            ]
        ),
        "config_hash": config_hash(
            [
                CONFIGS / "models.yaml",
                CONFIGS / "atomic_tasks.yaml",
                CONFIGS / "joint_screen.yaml",
                CONFIGS / "scoring.yaml",
            ]
        ),
        **runtime_metadata(ROOT),
    }
    output = ARTIFACTS / "models" / "frozen_registry.json"
    write_json(output, result)
    return result


def frozen_registry() -> dict[str, Any]:
    path = ARTIFACTS / "models" / "frozen_registry.json"
    if not path.exists():
        raise RuntimeError("model registry is not frozen; run freeze-models first")
    return json.loads(path.read_text(encoding="utf-8"))
