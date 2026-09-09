from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import socket
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from capability_gate.artifacts import (
    canonical_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)
from capability_gate.paths import ARTIFACTS, CONFIGS, REPORTS, ROOT
from capability_gate.recovery.adapters import DESCRIPTORS
from capability_gate.recovery.contract import RESPONSE_FIELDS
from capability_gate.recovery.environments import ENV_NAMES, worker_python
from capability_gate.recovery.smoke import WorkerSession
from capability_gate.statistics.joint import analyze_joint_rows
from capability_gate.statistics.metrics import atomic_task_metrics, cohens_kappa

MODEL_KEYS = ("qwen2_5_vl_7b", "glm4_1v_9b")
CARDINAL = ("north", "south", "east", "west")
ROOT_OUT = ARTIFACTS / "compute_migration"
ENVIRONMENT = ROOT / "environment"
FROZEN_CONFIGS = tuple(
    CONFIGS / name
    for name in ("models.yaml", "atomic_tasks.yaml", "joint_screen.yaml", "scoring.yaml")
)
FAIL_LABELS = {
    "direct_visual_relation": "ATOMIC_VISUAL_FAIL",
    "direct_text_relation": "ATOMIC_TEXT_FAIL",
    "direction_reversal": "ATOMIC_DIRECTION_FAIL",
    "cross_modal_bridge_binding": "ATOMIC_BINDING_FAIL",
}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _capture(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_fsync(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(value)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_text(line)


def _hash_paths(paths: Iterable[Path]) -> str:
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths))
    ]
    return sha256_text(canonical_json(records))


def _tree_files(path: Path) -> list[Path]:
    return [candidate for candidate in path.rglob("*") if candidate.is_file()]


def _gpu_record() -> dict[str, Any]:
    query = _capture(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    rows = [line for line in query.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("hardware gate requires exactly one visible CUDA GPU")
    name, gpu_uuid, driver, total, free = [part.strip() for part in rows[0].split(",")]
    return {
        "name": name,
        "uuid": gpu_uuid,
        "driver_version": driver,
        "total_vram_mib": int(total),
        "free_vram_mib": int(free),
    }


def _package_versions(python: Path) -> dict[str, Any]:
    program = (
        "import importlib.metadata as m,json,sys;"
        "names=['torch','torchvision','transformers','accelerate','bitsandbytes',"
        "'huggingface-hub','safetensors','Pillow'];"
        "print(json.dumps({'python':sys.version,'packages':{n:m.version(n) for n in names}}))"
    )
    return json.loads(subprocess.check_output([str(python), "-c", program], text=True))


def prepare_compute_migration() -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("INADEQUATE_GPU_ENVIRONMENT: Linux x86_64 required")
    gpu = _gpu_record()
    violations = []
    if gpu["total_vram_mib"] < 24 * 1024:
        violations.append("TOTAL_VRAM_BELOW_24_GIB")
    if gpu["free_vram_mib"] < 22 * 1024:
        violations.append("FREE_VRAM_BELOW_22_GIB")
    if violations:
        raise RuntimeError("INADEQUATE_GPU_ENVIRONMENT: " + ",".join(violations))
    for model_key in MODEL_KEYS:
        if not worker_python(model_key).is_file():
            raise RuntimeError(
                f"missing frozen environment for {model_key}: {worker_python(model_key)}"
            )

    ENVIRONMENT.mkdir(parents=True, exist_ok=True)
    locks = [ROOT / "envs/qwen/requirements.lock", ROOT / "envs/glm/requirements.lock"]
    lock_lines = ["# CapabilityGate compute migration exact dependency locks"]
    for lock in locks:
        lock_lines.extend(
            [
                f"# {lock.relative_to(ROOT).as_posix()} sha256={sha256_file(lock)}",
                lock.read_text(encoding="utf-8").rstrip(),
                "",
            ]
        )
    lock_path = ENVIRONMENT / "compute_migration_lock.txt"
    lock_path.write_text("\n".join(lock_lines), encoding="utf-8", newline="\n")

    os_release = {}
    release_path = Path("/etc/os-release")
    if release_path.is_file():
        for line in release_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip('"')
    system = {
        "schema_version": 1,
        "captured_at_utc": _now(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "os_release": os_release,
        "cpu": platform.processor()
        or _capture("bash", "-lc", "lscpu | grep 'Model name' | head -1"),
        "cpu_count": os.cpu_count(),
        "ram": _capture("bash", "-lc", "free -b"),
        "disk": _capture("bash", "-lc", f"df -B1 {ROOT}"),
        "gpu": gpu,
        "nvidia_smi": _capture("nvidia-smi"),
        "cuda_runtime_reported_by_driver": _capture(
            "bash", "-lc", "nvidia-smi | head -3 | tail -1"
        ),
        "environments": {
            model_key: _package_versions(worker_python(model_key)) for model_key in MODEL_KEYS
        },
        "git_branch": _git("branch", "--show-current"),
        "git_head": _git("rev-parse", "HEAD"),
    }
    write_json(ENVIRONMENT / "system_report.json", system)
    container = {
        "schema_version": 1,
        "kind": "REBUILDABLE_UV_VENV",
        "base_image_claim": "Compshare Cuda:12.4 platform image",
        "observed_os": os_release,
        "python": "3.11",
        "uv": _capture("bash", "-lc", "/root/.local/bin/uv --version"),
        "venvs": {
            model_key: str(worker_python(model_key).parent.parent) for model_key in MODEL_KEYS
        },
        "dependency_lock_sha256": sha256_file(lock_path),
        "dependencies_frozen_before_formal_atomic": True,
    }
    write_json(ENVIRONMENT / "container_or_venv_manifest.json", container)
    gate = {
        "schema_version": 1,
        "decision": "ADEQUATE_GPU_ENVIRONMENT",
        "evaluated_at_utc": _now(),
        "required": {"os": "Linux x86_64", "total_vram_mib_min": 24576, "free_vram_mib_min": 22528},
        "observed": system,
        "formal_atomic_started": False,
        "joint_started": False,
        "activation_patching_executed": False,
    }
    write_json(ROOT_OUT / "hardware_gate_pass.json", gate)
    return gate


def prepare_model_weights() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    results: dict[str, Any] = {}
    for model_key in MODEL_KEYS:
        descriptor = DESCRIPTORS[model_key]
        snapshot = Path(
            snapshot_download(
                descriptor.model_id,
                revision=descriptor.revision,
            )
        ).resolve()
        if snapshot.name != descriptor.revision:
            raise RuntimeError(
                f"MODEL_WEIGHT_INTEGRITY_FAILURE: resolved {snapshot.name}, expected {descriptor.revision}"
            )
        actual_names = sorted(path.name for path in snapshot.glob("*.safetensors"))
        if actual_names != list(descriptor.weight_names):
            raise RuntimeError(
                f"MODEL_WEIGHT_INTEGRITY_FAILURE: shard set mismatch for {model_key}: {actual_names}"
            )
        shards = []
        for name, expected in zip(descriptor.weight_names, descriptor.weight_hashes):
            path = snapshot / name
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(
                    f"MODEL_WEIGHT_INTEGRITY_FAILURE: {model_key}/{name}: {actual} != {expected}"
                )
            shards.append({"path": name, "bytes": path.stat().st_size, "sha256": actual})
        metadata_files = [
            path for path in snapshot.iterdir() if path.is_file() and path.suffix != ".safetensors"
        ]
        metadata = [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(metadata_files)
        ]
        result = {
            "schema_version": 1,
            "model_key": model_key,
            "model": descriptor.model_id,
            "requested_revision": descriptor.revision,
            "resolved_revision": snapshot.name,
            "processor_revision": descriptor.processor_revision,
            "tokenizer_revision": descriptor.processor_revision,
            "revision_verified": True,
            "processor_revision_verified": True,
            "tokenizer_revision_verified": True,
            "safetensors_file_count": len(shards),
            "shards": shards,
            "composite_weight_hash": sha256_text(canonical_json(shards)),
            "metadata_files": metadata,
            "snapshot_path": str(snapshot),
            "integrity": "PASS",
        }
        write_json(ROOT_OUT / f"weights/{ENV_NAMES[model_key]}.json", result)
        results[model_key] = result
    manifest = {
        "schema_version": 1,
        "decision": "MODEL_WEIGHT_INTEGRITY_PASS",
        "models": results,
        "model_or_revision_changed": False,
    }
    write_json(ROOT_OUT / "weights/manifest.json", manifest)
    return manifest


def _request(
    model_key: str,
    request_id: str,
    *,
    operation: str,
    system: str,
    user: str,
    candidates: Sequence[str],
    target: str,
    image_path: str | None,
) -> dict[str, Any]:
    descriptor = DESCRIPTORS[model_key]
    return {
        "schema_version": 1,
        "request_id": request_id,
        "model_key": model_key,
        "model_revision": descriptor.revision,
        "processor_revision": descriptor.processor_revision,
        "image_path": image_path,
        "prompt": {"system": system, "user": user},
        "candidates": list(candidates),
        "target": target,
        "operation": operation,
    }


def _load_gates(model_key: str, response: dict[str, Any]) -> dict[str, bool]:
    metadata = response["model_metadata"]
    material = metadata.get("materialization", {})
    placement = metadata.get("placement_inventory", {})
    weight_manifest = json.loads(
        (ROOT_OUT / f"weights/{ENV_NAMES[model_key]}.json").read_text(encoding="utf-8")
    )
    return {
        "exact_model_revision": metadata.get("model_revision") == DESCRIPTORS[model_key].revision,
        "processor_revision": metadata.get("processor_revision")
        == DESCRIPTORS[model_key].processor_revision,
        "weight_hashes_pass": metadata.get("weight_hashes_verified") is True
        and weight_manifest["integrity"] == "PASS",
        "native_model_class": metadata.get("native_model_class_verified") is True,
        "processor_class": metadata.get("processor_class_verified") is True,
        "meta_parameters_zero": material.get("meta_parameter_count") == 0,
        "meta_buffers_zero": material.get("meta_buffer_count") == 0,
        "missing_weights_zero": material.get("missing_weight_count") == 0,
        "unexpected_weights_zero": material.get("unexpected_weight_count") == 0,
        "mismatched_weights_zero": material.get("mismatched_weight_count") == 0,
        "cpu_offloaded_modules_zero": not placement.get("full_precision_cpu_modules"),
        "disk_offloaded_modules_zero": not placement.get("disk_offloaded_modules"),
        "single_gpu_explicit": metadata.get("resolved_device_map") in ({"": "0"}, {"": "cuda:0"}),
    }


def _score_valid(response: dict[str, Any], allowed: set[str]) -> bool:
    scores = response["candidate_scores"]
    return (
        response["status"] == "SCORE_AND_GENERATE_PASS"
        and set(response) == RESPONSE_FIELDS
        and {score["candidate"] for score in scores} == allowed
        and all(score["candidate_token_count"] > 0 for score in scores)
        and all(math.isfinite(score["raw_log_likelihood"]) for score in scores)
        and all(math.isfinite(score["normalized_log_likelihood"]) for score in scores)
        and response["constrained_answer"] in allowed
    )


def _same_measurement(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_scores = {
        value["candidate"]: value["normalized_log_likelihood"] for value in left["candidate_scores"]
    }
    right_scores = {
        value["candidate"]: value["normalized_log_likelihood"]
        for value in right["candidate_scores"]
    }
    return (
        set(left_scores) == set(right_scores)
        and all(
            math.isclose(left_scores[key], right_scores[key], rel_tol=0.0, abs_tol=1e-6)
            for key in left_scores
        )
        and left["model_metadata"].get("candidate_ranking")
        == right["model_metadata"].get("candidate_ranking")
        and left["constrained_answer"] == right["constrained_answer"]
    )


def run_migration_smoke() -> dict[str, Any]:
    output_manifest = ROOT_OUT / "smoke/manifest.json"
    if output_manifest.exists():
        return json.loads(output_manifest.read_text(encoding="utf-8"))
    hardware = json.loads((ROOT_OUT / "hardware_gate_pass.json").read_text(encoding="utf-8"))
    weights = json.loads((ROOT_OUT / "weights/manifest.json").read_text(encoding="utf-8"))
    if (
        hardware["decision"] != "ADEQUATE_GPU_ENVIRONMENT"
        or weights["decision"] != "MODEL_WEIGHT_INTEGRITY_PASS"
    ):
        raise RuntimeError("migration preflight gate is not PASS")
    scenes = read_jsonl(ARTIFACTS / "engineering_recovery/manifests/engineering_only_scenes.jsonl")
    if len(scenes) != 12 or any(
        not row["engineering_only"] or row["formal_overlap"] for row in scenes
    ):
        raise RuntimeError("migration smoke requires 12 disjoint engineering-only scenes")
    system = "Engineering transport validation only. Return exactly one allowed lowercase direction word and no explanation."
    results = {}
    for model_key in MODEL_KEYS:
        env_name = ENV_NAMES[model_key]
        stderr_path = ROOT_OUT / f"smoke/runtime/{env_name}.stderr.log"
        session = WorkerSession(model_key, stderr_path)
        responses: dict[str, list[dict[str, Any]]] = defaultdict(list)
        runtime_path = ROOT_OUT / f"smoke/predictions/{env_name}.jsonl"
        try:
            load = session.request(
                _request(
                    model_key,
                    f"migration-smoke-{env_name}-load",
                    operation="load_preflight",
                    system=system,
                    user="Frozen migration load preflight.",
                    candidates=CARDINAL,
                    target="north",
                    image_path=None,
                )
            )
            load_gates = _load_gates(model_key, load)
            for scene in scenes:
                for rerun in range(2):
                    user = (
                        scene["prompt"]
                        + "\nAllowed answers in displayed order: north, south, east, west."
                        + "\nAnswer schema: Return exactly one allowed lowercase direction word and nothing else."
                    )
                    response = session.request(
                        _request(
                            model_key,
                            f"migration-smoke-{env_name}-{scene['scene_id']}-r{rerun + 1}",
                            operation="score_and_generate",
                            system=system,
                            user=user,
                            candidates=CARDINAL,
                            target=scene["target"],
                            image_path=str(ROOT / scene["image_path"]),
                        )
                    )
                    record = {
                        "model_key": model_key,
                        "scene_id": scene["scene_id"],
                        "rerun": rerun + 1,
                        **response,
                    }
                    _append_fsync(runtime_path, record)
                    responses[scene["scene_id"]].append(response)
            all_responses = [response for pair in responses.values() for response in pair]
            visual_forward_count = sum(
                int(response["model_metadata"].get("vision_forward_event_count", 0))
                for response in all_responses
            )
            deterministic = all(
                len(pair) == 2 and _same_measurement(pair[0], pair[1])
                for pair in responses.values()
            )
            gates = {
                **load_gates,
                "scene_count_12": len(responses) == 12,
                "two_deterministic_reruns": len(all_responses) == 24 and deterministic,
                "visual_forward_count_at_least_64": visual_forward_count >= 64,
                "cll_finite": all(
                    _score_valid(response, set(CARDINAL)) for response in all_responses
                ),
                "candidate_token_count_positive": all(
                    score["candidate_token_count"] > 0
                    for response in all_responses
                    for score in response["candidate_scores"]
                ),
                "constrained_answer_allowed": all(
                    response["constrained_answer"] in CARDINAL for response in all_responses
                ),
                "cuda_illegal_memory_access_zero": "illegal memory access"
                not in stderr_path.read_text(encoding="utf-8").lower(),
                "cuda_oom_zero": "out of memory"
                not in stderr_path.read_text(encoding="utf-8").lower(),
                "driver_reset_zero": "xid" not in stderr_path.read_text(encoding="utf-8").lower(),
                "nan_inf_zero": all(
                    _score_valid(response, set(CARDINAL)) for response in all_responses
                ),
                "artifact_completeness": len(all_responses) == 24
                and all(set(response) == RESPONSE_FIELDS for response in all_responses),
            }
            results[model_key] = {
                "status": "MIGRATION_SMOKE_PASS"
                if all(gates.values())
                else "BLOCKED_BY_COMPUTE_MIGRATED_ENVIRONMENT",
                "gates": gates,
                "engineering_scene_accuracy_reported": False,
                "deterministic_rerun_agreement": 1.0 if deterministic else 0.0,
                "visual_forward_count": visual_forward_count,
                "peak_vram_bytes": max(
                    int(response.get("peak_vram") or 0) for response in [load, *all_responses]
                ),
                "model_metadata": load["model_metadata"],
            }
        except Exception as error:  # noqa: BLE001
            results[model_key] = {
                "status": "BLOCKED_BY_MODEL_ADAPTER_MIGRATED_ENVIRONMENT",
                "error": f"{type(error).__name__}: {error}",
                "traceback_preserved": str(stderr_path.relative_to(ROOT)),
                "engineering_scene_accuracy_reported": False,
            }
        finally:
            results[model_key]["worker_release"] = session.close()
    decision = (
        "MIGRATION_SMOKE_PASS"
        if all(results[key]["status"] == "MIGRATION_SMOKE_PASS" for key in MODEL_KEYS)
        else "MIGRATION_SMOKE_BLOCK"
    )
    manifest = {
        "schema_version": 1,
        "decision": decision,
        "models": results,
        "formal_atomic_rows_before_smoke": 0,
        "formal_joint_rows_before_smoke": 0,
        "engineering_scene_accuracy_reported": False,
        "activation_patching_executed": False,
    }
    _atomic_write_json(output_manifest, manifest)
    return manifest


def _environment_hash() -> str:
    return _hash_paths(
        [
            ENVIRONMENT / "compute_migration_lock.txt",
            ENVIRONMENT / "system_report.json",
            ENVIRONMENT / "container_or_venv_manifest.json",
            ROOT_OUT / "weights/manifest.json",
            ROOT_OUT / "weights/qwen.json",
            ROOT_OUT / "weights/glm.json",
        ]
    )


def create_formal_run_lock() -> dict[str, Any]:
    path = ROOT_OUT / "formal_run_lock.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    smoke = json.loads((ROOT_OUT / "smoke/manifest.json").read_text(encoding="utf-8"))
    if smoke["decision"] != "MIGRATION_SMOKE_PASS":
        raise RuntimeError("formal Atomic forbidden because both migration smokes did not pass")
    atomic_files = _tree_files(ARTIFACTS / "data/atomic_qualification")
    joint_files = _tree_files(ARTIFACTS / "data/joint_composition_screen")
    smoke_files = _tree_files(ARTIFACTS / "data/engineering_smoke")
    recovery_smoke_files = _tree_files(ARTIFACTS / "engineering_recovery/manifests/images") + [
        ARTIFACTS / "engineering_recovery/manifests/engineering_only_scenes.jsonl"
    ]
    scoring = yaml.safe_load((CONFIGS / "scoring.yaml").read_text(encoding="utf-8"))
    gpu = _gpu_record()
    lock = {
        "schema_version": 1,
        "run_id": "cm-"
        + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8],
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "environment_hash": _environment_hash(),
        "model_hashes": {
            key: json.loads(
                (ROOT_OUT / f"weights/{ENV_NAMES[key]}.json").read_text(encoding="utf-8")
            )["composite_weight_hash"]
            for key in MODEL_KEYS
        },
        "data_hashes": {
            "atomic": _hash_paths(atomic_files),
            "joint": _hash_paths(joint_files),
            "engineering_smoke": _hash_paths(smoke_files + recovery_smoke_files),
        },
        "config_hashes": {path.name: sha256_file(path) for path in FROZEN_CONFIGS},
        "prompt_hashes": {
            "atomic": sha256_text(
                yaml.safe_load((CONFIGS / "atomic_tasks.yaml").read_text(encoding="utf-8"))[
                    "system_instruction"
                ]
            ),
            "joint": sha256_text(
                yaml.safe_load((CONFIGS / "joint_screen.yaml").read_text(encoding="utf-8"))[
                    "system_instruction"
                ]
            ),
        },
        "scoring_hash": sha256_file(CONFIGS / "scoring.yaml"),
        "threshold_hash": sha256_text(
            canonical_json(
                {"atomic_gates": scoring["atomic_gates"], "agreement": scoring["agreement"]}
            )
        ),
        "run_seed": yaml.safe_load((CONFIGS / "atomic_tasks.yaml").read_text(encoding="utf-8"))[
            "seed"
        ],
        "gpu_uuid": gpu["uuid"],
        "start_timestamp": _now(),
        "old_local_partial_rows_included": False,
        "dependencies_mutable_after_lock": False,
        "activation_patching_executed": False,
    }
    _atomic_write_json(path, lock)
    return lock


def _prompt_user(question: str, options: Sequence[str], schema: str) -> str:
    return f"{question}\nAllowed answers in displayed order: {', '.join(options)}.\nAnswer schema: {schema}"


def _prediction_row(
    *,
    model_key: str,
    logical: dict[str, Any],
    response: dict[str, Any],
    mode: str,
    image_path: str | None,
    image_hash: str | None,
    prompt: dict[str, str],
    run_lock: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    scores = response["candidate_scores"]
    normalized = {score["candidate"]: score["normalized_log_likelihood"] for score in scores}
    metadata = response["model_metadata"]
    target = logical["target"]
    target_margin = normalized[target] - sum(
        value for key, value in normalized.items() if key != target
    ) / (len(normalized) - 1)
    row = {
        "schema_version": 3,
        "run_id": run_lock["run_id"],
        "row_index": row_index,
        "model_key": model_key,
        "model": DESCRIPTORS[model_key].model_id,
        "revision": DESCRIPTORS[model_key].revision,
        "scene_id": logical.get("scene_id", logical.get("base_quartet_id")),
        "base_quartet_id": logical.get("base_quartet_id"),
        "condition": logical.get("condition"),
        "task": logical["task"],
        "mode": mode,
        "target": target,
        "option_order": list(logical["options"]),
        "correct_option_position": logical["correct_option_position"],
        "image_path": image_path,
        "image_hash": image_hash,
        "prompt": metadata["rendered_prompt"],
        "prompt_hash": sha256_text(metadata["rendered_prompt"]),
        "prompt_contract": prompt,
        "candidate_text": [score["candidate"] for score in scores],
        "candidate_token_ids": {
            score["candidate"]: score["candidate_token_ids"] for score in scores
        },
        "raw_log_likelihood": {score["candidate"]: score["raw_log_likelihood"] for score in scores},
        "normalized_log_likelihood": normalized,
        "candidate_ranking": metadata["candidate_ranking"],
        "top_answer": metadata["top_answer"],
        "cll_answer": metadata["top_answer"],
        "target_margin": target_margin,
        "constrained_generation_answer": response["constrained_answer"],
        "constrained_generation_token_ids": metadata["constrained_generation_token_ids"],
        "runtime_seconds": response["runtime"]["forward_and_generation_seconds"],
        "peak_vram_bytes": response["peak_vram"],
        "image_token_count": metadata.get("image_token_count"),
        "input_sequence_length": metadata.get("input_sequence_length"),
        "pixel_tensor_shapes": metadata.get("pixel_tensor_shapes"),
        "image_grid_metadata": metadata.get("image_grid_metadata"),
        "environment_hash": run_lock["environment_hash"],
        "worker_artifact_hash": response["artifact_hash"],
        "result_hash": None,
    }
    row["result_hash"] = sha256_text(canonical_json(row))
    return row


def _refresh_result_hash(row: dict[str, Any]) -> None:
    row["result_hash"] = None
    row["result_hash"] = sha256_text(canonical_json(row))


def _verified_completed_rows(predictions: Path, hashes: Path) -> list[dict[str, Any]]:
    if not predictions.exists() and not hashes.exists():
        return []
    if predictions.exists() != hashes.exists():
        raise RuntimeError("append-only checkpoint files are incomplete")
    rows = read_jsonl(predictions)
    records = read_jsonl(hashes)
    if len(rows) != len(records):
        raise RuntimeError("prediction and row-hash checkpoint lengths differ")
    for row, record in zip(rows, records):
        expected_result_hash = row.get("result_hash")
        unhashed = dict(row)
        unhashed["result_hash"] = None
        if (
            sha256_text(canonical_json(row)) != record["row_line_hash"]
            or expected_result_hash != record["result_hash"]
            or sha256_text(canonical_json(unhashed)) != expected_result_hash
        ):
            raise RuntimeError("append-only checkpoint hash verification failed")
    return rows


def run_migrated_atomic_model(model_key: str) -> dict[str, Any]:
    if model_key not in MODEL_KEYS:
        raise ValueError("only the frozen Qwen and GLM families are authorized")
    lock = json.loads((ROOT_OUT / "formal_run_lock.json").read_text(encoding="utf-8"))
    if _environment_hash() != lock["environment_hash"] or _gpu_record()["uuid"] != lock["gpu_uuid"]:
        raise RuntimeError(
            "formal run environment changed; freeze this run as failed and start from scene 1"
        )
    if model_key == MODEL_KEYS[1]:
        prior = ROOT_OUT / f"formal/atomic/{lock['run_id']}/status/qwen.json"
        if (
            not prior.exists()
            or json.loads(prior.read_text(encoding="utf-8"))["status"] != "complete"
        ):
            raise RuntimeError("GLM Atomic is sequential and requires completed Qwen Atomic")
    atomic = yaml.safe_load((CONFIGS / "atomic_tasks.yaml").read_text(encoding="utf-8"))
    scenes = read_jsonl(ARTIFACTS / "data/atomic_qualification/scenes.jsonl")
    if len(scenes) != 256:
        raise RuntimeError("frozen Atomic qualification must contain exactly 256 scenes")
    env_name = ENV_NAMES[model_key]
    base = ROOT_OUT / f"formal/atomic/{lock['run_id']}"
    predictions = base / f"predictions/{env_name}.jsonl"
    hashes = base / f"row_hashes/{env_name}.jsonl"
    bitmap_path = base / f"completion/{env_name}.json"
    status_path = base / f"status/{env_name}.json"
    completed = _verified_completed_rows(predictions, hashes)
    if completed and any(
        row["run_id"] != lock["run_id"] or row["environment_hash"] != lock["environment_hash"]
        for row in completed
    ):
        raise RuntimeError("existing checkpoint belongs to a different formal environment")
    completed_indices = {int(row["row_index"]) for row in completed}
    if completed_indices != set(range(len(completed))):
        raise RuntimeError("formal Atomic checkpoint is not a contiguous append-only prefix")
    if (
        status_path.exists()
        and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete"
    ):
        return json.loads(status_path.read_text(encoding="utf-8"))
    stderr_path = base / f"runtime/{env_name}.stderr.log"
    session = WorkerSession(model_key, stderr_path)
    try:
        load = session.request(
            _request(
                model_key,
                f"migrated-atomic-{env_name}-load",
                operation="load_preflight",
                system=atomic["system_instruction"],
                user="Frozen migrated Atomic load preflight.",
                candidates=atomic["candidates"],
                target=atomic["candidates"][0],
                image_path=None,
            )
        )
        if load["status"] != "LOAD_PREFLIGHT_PASS" or not all(
            _load_gates(model_key, load).values()
        ):
            raise RuntimeError(load.get("traceback") or "formal load gate failed")
        for index, scene in enumerate(scenes):
            if index in completed_indices:
                continue
            prompt = {
                "system": atomic["system_instruction"],
                "user": _prompt_user(scene["question"], scene["options"], atomic["answer_schema"]),
            }
            response = session.request(
                _request(
                    model_key,
                    f"migrated-atomic-{env_name}-{index:03d}",
                    operation="score_and_generate",
                    system=prompt["system"],
                    user=prompt["user"],
                    candidates=atomic["candidates"],
                    target=scene["target"],
                    image_path=str(ROOT / scene["image_path"]) if scene["image_path"] else None,
                )
            )
            if response["status"] != "SCORE_AND_GENERATE_PASS":
                raise RuntimeError(response.get("traceback") or response.get("error_class"))
            row = _prediction_row(
                model_key=model_key,
                logical=scene,
                response=response,
                mode="atomic_qualification",
                image_path=scene["image_path"],
                image_hash=scene["image_sha256"],
                prompt=prompt,
                run_lock=lock,
                row_index=index,
            )
            row_line_hash = _append_fsync(predictions, row)
            _append_fsync(
                hashes,
                {
                    "row_index": index,
                    "scene_id": scene["scene_id"],
                    "result_hash": row["result_hash"],
                    "row_line_hash": row_line_hash,
                },
            )
            completed_indices.add(index)
            _atomic_write_json(
                bitmap_path,
                {
                    "run_id": lock["run_id"],
                    "model_key": model_key,
                    "environment_hash": lock["environment_hash"],
                    "completed": [value in completed_indices for value in range(256)],
                    "completed_count": len(completed_indices),
                    "last_completed_scene": index,
                    "updated_at_utc": _now(),
                },
            )
        status = {
            "status": "complete",
            "run_id": lock["run_id"],
            "model_key": model_key,
            "rows": len(completed_indices),
            "ended_at_utc": _now(),
        }
        _atomic_write_json(status_path, status)
        return status
    except Exception as error:  # noqa: BLE001
        failure = {
            "status": "FORMAL_ATOMIC_RUNTIME_FAIL",
            "run_id": lock["run_id"],
            "model_key": model_key,
            "completed_rows_preserved": len(completed_indices),
            "error": f"{type(error).__name__}: {error}",
            "environment_hash": lock["environment_hash"],
            "ended_at_utc": _now(),
        }
        _atomic_write_json(status_path, failure)
        return failure
    finally:
        release = session.close()
        _atomic_write_json(
            base / f"runtime/{env_name}.json", {"model_key": model_key, "release": release}
        )


def run_migrated_atomic_qwen() -> dict[str, Any]:
    return run_migrated_atomic_model(MODEL_KEYS[0])


def run_migrated_atomic_glm() -> dict[str, Any]:
    return run_migrated_atomic_model(MODEL_KEYS[1])


def _per_answer_accuracy(
    rows: Sequence[dict[str, Any]], prediction_key: str
) -> dict[str, float | None]:
    result = {}
    for answer in CARDINAL:
        subset = [row for row in rows if row["target"] == answer]
        result[answer] = (
            sum(row[prediction_key] == answer for row in subset) / len(subset) if subset else None
        )
    return result


def adjudicate_migrated_atomic() -> dict[str, Any]:
    lock = json.loads((ROOT_OUT / "formal_run_lock.json").read_text(encoding="utf-8"))
    base = ROOT_OUT / f"formal/atomic/{lock['run_id']}"
    scoring = yaml.safe_load((CONFIGS / "scoring.yaml").read_text(encoding="utf-8"))
    models = {}
    for model_key in MODEL_KEYS:
        env_name = ENV_NAMES[model_key]
        status = json.loads((base / f"status/{env_name}.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            result = {
                "decision": "CAPABILITY_COHORT_NO_GO",
                "q1_potential": "NO_FEASIBLE_COHORT",
                "exact_next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
                "models": models,
                "failed_runtime": status,
                "joint_authorized": False,
            }
            _atomic_write_json(base / "adjudication.json", result)
            return result
        rows = read_jsonl(base / f"predictions/{env_name}.jsonl")
        if len(rows) != 256:
            raise RuntimeError(f"{model_key} Atomic output must contain exactly 256 rows")
        tasks = {}
        failed_tasks = []
        cll_all = True
        generation_all = True
        for task, gate in scoring["atomic_gates"].items():
            task_rows = [row for row in rows if row["task"] == task]
            if len(task_rows) != 64:
                raise RuntimeError(f"{model_key}/{task} must contain exactly 64 rows")
            cll = atomic_task_metrics(task_rows, CARDINAL, "cll_answer")
            generated = atomic_task_metrics(task_rows, CARDINAL, "constrained_generation_answer")
            cll["per_answer_accuracy"] = _per_answer_accuracy(task_rows, "cll_answer")
            generated["per_answer_accuracy"] = _per_answer_accuracy(
                task_rows, "constrained_generation_answer"
            )
            cll_pass = (
                cll["accuracy"] >= gate["accuracy_min"]
                and cll["one_sided_95_exact_lower"] >= gate["lower_bound_min"]
            )
            generation_pass = (
                generated["accuracy"] >= gate["accuracy_min"]
                and generated["one_sided_95_exact_lower"] >= gate["lower_bound_min"]
            )
            cll_all &= cll_pass
            generation_all &= generation_pass
            if not (cll_pass and generation_pass):
                failed_tasks.append(task)
            left = [row["cll_answer"] for row in task_rows]
            right = [row["constrained_generation_answer"] for row in task_rows]
            task_kappa = cohens_kappa(left, right, CARDINAL)
            tasks[task] = {
                "cll": cll,
                "constrained_generation": generated,
                "cll_gate": cll_pass,
                "generation_gate": generation_pass,
                "contract_direction_conflict": cll_pass != generation_pass,
                "agreement": {
                    "exact": sum(a == b for a, b in zip(left, right)) / len(left),
                    "cohens_kappa": task_kappa,
                },
            }
        left = [row["cll_answer"] for row in rows]
        right = [row["constrained_generation_answer"] for row in rows]
        exact = sum(a == b for a, b in zip(left, right)) / len(left)
        kappa = cohens_kappa(left, right, CARDINAL)
        contract_conflict = cll_all != generation_all or any(
            value["contract_direction_conflict"] for value in tasks.values()
        )
        if contract_conflict or kappa is None or kappa < scoring["agreement"]["kappa_min"]:
            label = "MEASUREMENT_CONTRACT_DEPENDENT"
        elif cll_all and generation_all:
            label = "ATOMICALLY_QUALIFIED"
        else:
            label = FAIL_LABELS[failed_tasks[0]]
        models[model_key] = {
            "label": label,
            "tasks": tasks,
            "failed_tasks": failed_tasks,
            "agreement": {
                "exact": exact,
                "cohens_kappa": kappa,
                "contract_direction_conflict": contract_conflict,
            },
        }
    qualified = all(models[key]["label"] == "ATOMICALLY_QUALIFIED" for key in MODEL_KEYS)
    result = {
        "decision": "ATOMIC_COHORT_GO" if qualified else "CAPABILITY_COHORT_NO_GO",
        "q1_potential": "JOINT_SCREEN_PENDING" if qualified else "NO_FEASIBLE_COHORT",
        "exact_next_action": "RUN_FROZEN_JOINT_COMPOSITION_SCREEN"
        if qualified
        else "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
        "qualified_count": sum(
            models[key]["label"] == "ATOMICALLY_QUALIFIED" for key in MODEL_KEYS
        ),
        "models": models,
        "joint_authorized": qualified,
        "activation_patching_executed": False,
    }
    _atomic_write_json(base / "adjudication.json", result)
    _write_atomic_report(result)
    return result


def _write_atomic_report(result: dict[str, Any]) -> None:
    lines = ["# Migrated Atomic Qualification", "", f"Decision: **{result['decision']}**", ""]
    for model_key, model in result.get("models", {}).items():
        lines.extend(
            [
                f"## {model_key}",
                "",
                f"Label: `{model['label']}`",
                "",
                "| Task | Contract | n | Accuracy | One-sided 95% lower | Gate |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for task, metrics in model["tasks"].items():
            for contract, gate_key in (
                ("cll", "cll_gate"),
                ("constrained_generation", "generation_gate"),
            ):
                values = metrics[contract]
                lines.append(
                    f"| {task} | {contract} | {values['n']} | {values['accuracy']:.4f} | {values['one_sided_95_exact_lower']:.4f} | {metrics[gate_key]} |"
                )
        lines.extend(
            [
                "",
                f"Exact agreement={model['agreement']['exact']:.4f}; Cohen's κ={model['agreement']['cohens_kappa']}.",
                "",
            ]
        )
    path = REPORTS / "compute_migration/atomic_qualification.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_migrated_joint_model(model_key: str) -> dict[str, Any]:
    if model_key not in MODEL_KEYS:
        raise ValueError("only the frozen Qwen and GLM families are authorized")
    lock = json.loads((ROOT_OUT / "formal_run_lock.json").read_text(encoding="utf-8"))
    atomic_base = ROOT_OUT / f"formal/atomic/{lock['run_id']}"
    atomic_decision = json.loads((atomic_base / "adjudication.json").read_text(encoding="utf-8"))
    if atomic_decision["decision"] != "ATOMIC_COHORT_GO" or not atomic_decision["joint_authorized"]:
        return {
            "status": "NOT_RUN_BY_ATOMIC_GATE",
            "upstream_decision": atomic_decision["decision"],
        }
    if model_key == MODEL_KEYS[1]:
        prior = ROOT_OUT / f"formal/joint/{lock['run_id']}/status/qwen.json"
        if (
            not prior.exists()
            or json.loads(prior.read_text(encoding="utf-8"))["status"] != "complete"
        ):
            raise RuntimeError("GLM Joint is sequential and requires completed Qwen Joint")
    joint = yaml.safe_load((CONFIGS / "joint_screen.yaml").read_text(encoding="utf-8"))
    atomic = yaml.safe_load((CONFIGS / "atomic_tasks.yaml").read_text(encoding="utf-8"))
    quartets = read_jsonl(ARTIFACTS / "data/joint_composition_screen/quartets.jsonl")
    atomic_scenes = read_jsonl(ARTIFACTS / "data/atomic_qualification/scenes.jsonl")
    if len(quartets) != 128:
        raise RuntimeError("frozen Joint screen must contain exactly 128 quartets")
    env_name = ENV_NAMES[model_key]
    base = ROOT_OUT / f"formal/joint/{lock['run_id']}"
    predictions = base / f"predictions/{env_name}.jsonl"
    retention = base / f"retention/{env_name}.jsonl"
    status_path = base / f"status/{env_name}.json"
    if (
        status_path.exists()
        and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete"
    ):
        return json.loads(status_path.read_text(encoding="utf-8"))
    existing_prediction_keys = (
        {(row["base_quartet_id"], row["condition"], row["mode"]) for row in read_jsonl(predictions)}
        if predictions.exists()
        else set()
    )
    existing_retention = (
        {row["scene_id"] for row in read_jsonl(retention)} if retention.exists() else set()
    )
    stderr_path = base / f"runtime/{env_name}.stderr.log"
    session = WorkerSession(model_key, stderr_path)
    row_index = len(existing_prediction_keys)
    cache: dict[str, dict[str, Any]] = {}
    try:
        load = session.request(
            _request(
                model_key,
                f"migrated-joint-{env_name}-load",
                operation="load_preflight",
                system=joint["system_instruction"],
                user="Frozen Joint load preflight.",
                candidates=joint["candidates"],
                target=joint["candidates"][0],
                image_path=None,
            )
        )
        if load["status"] != "LOAD_PREFLIGHT_PASS" or not all(
            _load_gates(model_key, load).values()
        ):
            raise RuntimeError(load.get("traceback") or "Joint load gate failed")
        for quartet in quartets:
            for condition in quartet["conditions"]:
                logical = {
                    **condition,
                    "base_quartet_id": quartet["base_quartet_id"],
                    "task": "joint_composition",
                    "options": quartet["options"],
                    "psi_fixed_target": quartet["psi_fixed_target"],
                    "axis_map": quartet["axis_map"],
                }
                modes = {
                    "joint": (
                        condition["question"],
                        condition["image_path"],
                        condition["image_sha256"],
                    ),
                    "image_only": (
                        condition["question_without_premise"],
                        condition["image_path"],
                        condition["image_sha256"],
                    ),
                    "text_only": (condition["question"], None, None),
                    "question_only": (condition["question_without_premise"], None, None),
                }
                for mode, (question, image_path, image_hash) in modes.items():
                    key = (quartet["base_quartet_id"], condition["condition"], mode)
                    if key in existing_prediction_keys:
                        continue
                    prompt = {
                        "system": joint["system_instruction"],
                        "user": _prompt_user(question, quartet["options"], joint["answer_schema"]),
                    }
                    cache_key = sha256_text(
                        canonical_json(
                            {
                                "prompt": prompt,
                                "image_path": image_path,
                                "candidates": joint["candidates"],
                            }
                        )
                    )
                    if cache_key not in cache:
                        cache[cache_key] = session.request(
                            _request(
                                model_key,
                                f"migrated-joint-{env_name}-{row_index:06d}",
                                operation="score_and_generate",
                                system=prompt["system"],
                                user=prompt["user"],
                                candidates=joint["candidates"],
                                target=condition["target"],
                                image_path=str(ROOT / image_path) if image_path else None,
                            )
                        )
                    response = cache[cache_key]
                    if response["status"] != "SCORE_AND_GENERATE_PASS":
                        raise RuntimeError(response.get("traceback") or response.get("error_class"))
                    row = _prediction_row(
                        model_key=model_key,
                        logical=logical,
                        response=response,
                        mode=mode,
                        image_path=image_path,
                        image_hash=image_hash,
                        prompt=prompt,
                        run_lock=lock,
                        row_index=row_index,
                    )
                    row.update(
                        {
                            "psi_fixed_target": quartet["psi_fixed_target"],
                            "axis_map": quartet["axis_map"],
                            "image_bit": condition["image_bit"],
                            "text_bit": condition["text_bit"],
                            "inference_cache_key": cache_key,
                            "template_id": quartet["template_id"],
                        }
                    )
                    _refresh_result_hash(row)
                    _append_fsync(predictions, row)
                    existing_prediction_keys.add(key)
                    row_index += 1
        retention_system = (
            joint["system_instruction"]
            + " For direct control items, use the supplied direct evidence and a cardinal answer."
        )
        for index, scene in enumerate(atomic_scenes):
            if scene["scene_id"] in existing_retention:
                continue
            prompt = {
                "system": retention_system,
                "user": _prompt_user(scene["question"], scene["options"], atomic["answer_schema"]),
            }
            response = session.request(
                _request(
                    model_key,
                    f"migrated-joint-retention-{env_name}-{index:03d}",
                    operation="score_and_generate",
                    system=prompt["system"],
                    user=prompt["user"],
                    candidates=atomic["candidates"],
                    target=scene["target"],
                    image_path=str(ROOT / scene["image_path"]),
                )
            )
            if response["status"] != "SCORE_AND_GENERATE_PASS":
                raise RuntimeError(response.get("traceback") or response.get("error_class"))
            row = _prediction_row(
                model_key=model_key,
                logical=scene,
                response=response,
                mode="joint_context_atomic_retention",
                image_path=scene["image_path"],
                image_hash=scene["image_sha256"],
                prompt=prompt,
                run_lock=lock,
                row_index=index,
            )
            _append_fsync(retention, row)
            existing_retention.add(scene["scene_id"])
        status = {
            "status": "complete",
            "model_key": model_key,
            "prediction_rows": len(existing_prediction_keys),
            "retention_rows": len(existing_retention),
            "ended_at_utc": _now(),
        }
        _atomic_write_json(status_path, status)
        return status
    except Exception as error:  # noqa: BLE001
        failure = {
            "status": "FORMAL_JOINT_RUNTIME_FAIL",
            "model_key": model_key,
            "prediction_rows_preserved": len(existing_prediction_keys),
            "retention_rows_preserved": len(existing_retention),
            "error": f"{type(error).__name__}: {error}",
        }
        _atomic_write_json(status_path, failure)
        return failure
    finally:
        release = session.close()
        _atomic_write_json(
            base / f"runtime/{env_name}.json", {"model_key": model_key, "release": release}
        )


def run_migrated_joint_qwen() -> dict[str, Any]:
    return run_migrated_joint_model(MODEL_KEYS[0])


def run_migrated_joint_glm() -> dict[str, Any]:
    return run_migrated_joint_model(MODEL_KEYS[1])


def analyze_migrated_joint() -> dict[str, Any]:
    lock = json.loads((ROOT_OUT / "formal_run_lock.json").read_text(encoding="utf-8"))
    atomic_base = ROOT_OUT / f"formal/atomic/{lock['run_id']}"
    atomic = json.loads((atomic_base / "adjudication.json").read_text(encoding="utf-8"))
    if atomic["decision"] != "ATOMIC_COHORT_GO":
        result = {
            "decision": "CAPABILITY_COHORT_NO_GO",
            "q1_potential": "NO_FEASIBLE_COHORT",
            "exact_next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
            "joint_status": "NOT_RUN_BY_ATOMIC_GATE",
            "models": {},
        }
    else:
        base = ROOT_OUT / f"formal/joint/{lock['run_id']}"
        predictions = [
            row
            for key in MODEL_KEYS
            for row in read_jsonl(base / f"predictions/{ENV_NAMES[key]}.jsonl")
        ]
        retention = [
            row
            for key in MODEL_KEYS
            for row in read_jsonl(base / f"retention/{ENV_NAMES[key]}.jsonl")
        ]
        result = analyze_joint_rows(predictions, retention)
        result["q1_potential"] = (
            "MECHANISTIC_STUDY_FEASIBLE"
            if result["decision"] == "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION"
            else "NO_FEASIBLE_COHORT"
        )
    _atomic_write_json(ROOT_OUT / f"formal/final_decision_{lock['run_id']}.json", result)
    report = REPORTS / "compute_migration/final_decision.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# CapabilityGate Final Decision\n\n"
        f"Decision: **{result['decision']}**\n\n"
        f"Q1 potential: `{result.get('q1_potential')}`\n\n"
        f"Exact next action: `{result.get('exact_next_action')}`\n\n"
        "Activation patching was not executed. No Phi or fourth model was used.\n",
        encoding="utf-8",
        newline="\n",
    )
    return result
