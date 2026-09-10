"""Append-only Atomic Protocol v2 orchestration; never writes v1 artifacts."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

import yaml

from capability_gate.artifacts import canonical_json, read_jsonl, sha256_file, sha256_text
from capability_gate.compute_migration import _gpu_record, _package_versions, _prompt_user
from capability_gate.paths import ROOT
from capability_gate.recovery.adapters import DESCRIPTORS

SOURCE = "eca06a65e9ea7c384ff17859bcd9b28c79630219"
V1_TAG = "capability-gate-atomic-v1-protocol-invalid-2026-09-10"
V2_TAG = "capability-gate-atomic-v2-prerun-freeze"
KEYS = ("qwen2_5_vl_7b", "glm4_1v_9b")
NAMES = {KEYS[0]: "qwen", KEYS[1]: "glm"}
OUT = ROOT / "artifacts/atomic_v2"
REPORT = ROOT / "reports/atomic_v2"
DATA = ROOT / "artifacts/data_v2"


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def git(*args):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(ROOT), *args], text=True
    ).strip()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_once(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"IMMUTABLE_ARTIFACT_ALREADY_EXISTS: {path}")
        return value
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return value


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def snapshot_files(paths):
    return [
        {
            "path": p.relative_to(ROOT).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(set(paths))
        if p.is_file()
    ]


def _scientific_files():
    paths = []
    for folder in (
        ROOT / "configs",
        ROOT / "src/capability_gate",
        ROOT / "workers",
        ROOT / "research/atomic_v2",
        DATA,
        OUT / "renderer_snapshots",
    ):
        paths.extend(
            p
            for p in folder.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
        )
    return paths


def _stable_file_hash(path):
    """Source/config text hash with canonical LF; artifact/weight hashes remain raw."""
    content = path.read_bytes()
    if path.suffix in {".py", ".yaml", ".md", ".txt"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def frozen_inputs():
    return {p.relative_to(ROOT).as_posix(): _stable_file_hash(p) for p in _scientific_files()}


def verify_v1():
    """Read-only verification replaces historical commands which rewrote manifests."""
    names = git("ls-tree", "-r", "--name-only", SOURCE).splitlines()
    protected = [
        p
        for p in names
        if p.startswith(
            ("artifacts/", "reports/", "research/", "configs/", "environment/", "envs/")
        )
    ]
    changed = set(git("diff", SOURCE, "--name-only").splitlines())
    collisions = sorted(changed.intersection(protected))
    missing = [p for p in protected if not (ROOT / p).is_file()]
    if collisions or missing:
        raise RuntimeError(f"V1_CHANGED: changed={collisions}; missing={missing}")
    if git("cat-file", "-t", V1_TAG) != "tag" or git("rev-parse", V1_TAG + "^{}") != SOURCE:
        raise RuntimeError("V1_TAG_CHANGED")
    initial = load(OUT / "manifests/initial_preflight.json")
    if git("rev-parse", V1_TAG) != initial["invalid_v1_tag"]["annotated_object"]:
        raise RuntimeError("V1_TAG_OBJECT_CHANGED")
    old = initial["old_annotated_tag"]
    if git("rev-parse", old["name"]) != old["object"]:
        raise RuntimeError("OLD_TAG_CHANGED")
    audit = load(ROOT / "artifacts/compute_migration/execution_audit.json")
    base = ROOT / "artifacts/compute_migration/formal/atomic" / audit["run_id"]
    results = {}
    for name in NAMES.values():
        path = base / f"predictions/{name}.jsonl"
        rows = read_jsonl(path)
        if sha256_file(path) != audit["formal_outputs"][name + "_predictions_sha256"]:
            raise RuntimeError("V1_PREDICTION_HASH_CHANGED")
        if len(rows) != 256 or len({r["scene_id"] for r in rows}) != 256:
            raise RuntimeError("V1_INCOMPLETE")
        bitmap = load(base / f"completion/{name}.json")
        results[name] = {
            "rows": len(rows),
            "sha256": sha256_file(path),
            "bitmap_sha256": sha256_file(base / f"completion/{name}.json"),
            "bitmap_record": bitmap,
        }
    return {
        "schema_version": 1,
        "overall_gate": True,
        "source_commit": SOURCE,
        "protected_files": len(protected),
        "changed_old_files": [],
        "v1_results": results,
        "historical_results_overwritten": False,
        "verification_mode": "READ_ONLY_NO_HISTORICAL_MANIFEST_REWRITE",
    }


def freeze_invalid_atomic_v1():
    result = verify_v1()
    save_once(OUT / "manifests/v1_preservation.json", result)
    return result


def repair_glm_processor_v2():
    """One pre-control engineering retry, with the initial failure fully preserved."""
    record_path = OUT / "renderer_repair_record.json"
    if record_path.exists():
        return load(record_path)
    if (OUT / "formal_run_lock.json").exists() or any((OUT / "formal").rglob("*.jsonl")):
        raise RuntimeError("RENDERER_REPAIR_AFTER_FORMAL_OUTPUT_FORBIDDEN")
    if any((OUT / "contract_control").rglob("predictions.jsonl")):
        raise RuntimeError("THIS_REPAIR_MUST_PRECEDE_ALL_CONTROL_OUTPUTS")
    failure = load(OUT / "renderer_snapshots/glm4_1v_9b/manifest.json")
    if failure.get("failure", {}).get("type") != "TypeError" or failure.get("snapshot_count") != 0:
        raise RuntimeError("UNREGISTERED_RENDERER_REPAIR_FAILURE_SIGNATURE")
    if "string indices must be integers" not in failure["failure"]["message"]:
        raise RuntimeError("UNREGISTERED_RENDERER_REPAIR_FAILURE_SIGNATURE")
    if load(OUT / "renderer_snapshots/qwen2_5_vl_7b/manifest.json")["overall_gate"] is not True:
        raise RuntimeError("QWEN_RENDERER_NOT_VERIFIED")
    archive = OUT / "attempts/initial_renderer_failure"
    mapping = []
    paths = [
        OUT / "final_decision.json",
        OUT / "manifests/final_manifest.json",
        REPORT / "final_decision.md",
        REPORT / "atomic_qualification.md",
        REPORT / "joint_composition.md",
        REPORT / "renderer_validation.md",
    ]
    for path in paths:
        if path.exists():
            relative = path.relative_to(ROOT)
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError("INITIAL_FAILURE_ARCHIVE_ALREADY_EXISTS")
            digest = sha256_file(path)
            path.rename(destination)
            mapping.append(
                {
                    "original_path": relative.as_posix(),
                    "archived_path": destination.relative_to(ROOT).as_posix(),
                    "sha256": digest,
                }
            )
    record = {
        "schema_version": 1,
        "engineering_retry_count": 1,
        "maximum_retries": 1,
        "initial_attempt": "v2 system-str implementation; zero model forwards",
        "failure_signature": "Transformers 4.57.6 multimodal tokenize=True iterates system str",
        "repair": "official GLM apply_chat_template(tokenize=False) then official processor(text,images,add_special_tokens=False)",
        "scientific_prompt_changed": False,
        "candidate_contract_changed": False,
        "data_changed": False,
        "model_outputs_before_repair": 0,
        "initial_failure_snapshots_preserved_at_original_paths": True,
        "archived_terminal_documents": mapping,
        "timestamp": now(),
    }
    save_once(record_path, record)
    _commit(
        [OUT, REPORT],
        "audit: preserve initial renderer failure and authorize sole pre-control correction",
    )
    return record


def python_for(key):
    if key not in KEYS:
        raise RuntimeError("UNAUTHORIZED_MODEL")
    override = os.environ.get("CAPABILITY_V2_" + NAMES[key].upper() + "_PYTHON")
    if override:
        return Path(override)
    return (
        ROOT
        / "envs"
        / NAMES[key]
        / ".venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )


def _subprocess(module, args, logfile):
    key = args[args.index("--model-key") + 1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    if logfile.exists():
        raise RuntimeError("RUNTIME_LOG_ALREADY_EXISTS")
    with logfile.open("x", encoding="utf-8", newline="\n") as handle:
        result = subprocess.run(
            [str(python_for(key)), "-m", module, *args],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return result.returncode


def validate_model_renderers():
    from capability_gate.atomic_v2_data import validate_atomic_v2_data
    from capability_gate.atomic_v2_measurement import renderer_manifest_path

    validate_atomic_v2_data()
    results = {}
    for key in KEYS:
        path = renderer_manifest_path(ROOT, key)
        if not path.exists():
            _subprocess(
                "capability_gate.atomic_v2_measurement",
                ["renderers", "--model-key", key, "--root", str(ROOT)],
                path.parent / "process.log",
            )
        if not path.exists():
            raise RuntimeError(f"RENDERER_DID_NOT_PRODUCE_MANIFEST: {key}")
        results[key] = load(path)
    return results


def run_contract_control():
    results = {}
    for key in KEYS:
        path = OUT / "contract_control" / key / "manifest.json"
        if not path.exists():
            _subprocess(
                "capability_gate.atomic_v2_measurement",
                ["contract-control", "--model-key", key, "--root", str(ROOT)],
                OUT / "contract_control" / key / "process.log",
            )
        if not path.exists():
            raise RuntimeError(f"CONTROL_DID_NOT_PRODUCE_MANIFEST: {key}")
        results[key] = load(path)
    return results


def _measurement_gate():
    from capability_gate.atomic_v2_measurement import aggregate_measurement_validation

    result = aggregate_measurement_validation(ROOT)
    if not result.get("overall_gate", False):
        raise RuntimeError("MEASUREMENT_IMPLEMENTATION_NO_GO")
    return result


def _commit(paths, message):
    relative = [str(Path(p).relative_to(ROOT)) if Path(p).is_absolute() else str(p) for p in paths]
    relative = [p for p in relative if (ROOT / p).exists()]
    if relative:
        git("add", "--", *relative)
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", message)
    return git("rev-parse", "HEAD")


def freeze_atomic_v2_run():
    from capability_gate.atomic_v2_data import (
        validate_atomic_v2_data,
        validate_joint_data_before_atomic_v2,
    )

    verify_v1()
    validate_atomic_v2_data()
    validate_joint_data_before_atomic_v2()
    _measurement_gate()
    lockpath = OUT / "formal_run_lock.json"
    if lockpath.exists():
        check_lock()
        return load(lockpath)
    if any((OUT / "formal").rglob("*.jsonl")):
        raise RuntimeError("FORMAL_OUTPUT_BEFORE_LOCK")
    if git("status", "--porcelain"):
        raise RuntimeError("COMMIT_VALIDATED_DATA_RENDERERS_AND_CONTROLS_BEFORE_FREEZE")
    gpu = _gpu_record()
    if (
        platform.system() != "Linux"
        or gpu["total_vram_mib"] < 24576
        or gpu["free_vram_mib"] < 22528
    ):
        raise RuntimeError("INADEQUATE_GPU_ENVIRONMENT")
    environment = {key: _package_versions(python_for(key)) for key in KEYS}
    weights = {}
    for key in KEYS:
        descriptor = DESCRIPTORS[key]
        previous = load(ROOT / f"artifacts/compute_migration/weights/{NAMES[key]}.json")
        snapshot = Path(previous["snapshot_path"])
        shards = []
        for name, expected in zip(descriptor.weight_names, descriptor.weight_hashes):
            shard = snapshot / name
            actual = sha256_file(shard)
            if actual != expected:
                raise RuntimeError("FROZEN_WEIGHT_HASH_MISMATCH")
            shards.append({"path": name, "bytes": shard.stat().st_size, "sha256": actual})
        weights[key] = {
            "model": descriptor.model_id,
            "revision": descriptor.revision,
            "processor_revision": descriptor.processor_revision,
            "tokenizer_revision": descriptor.processor_revision,
            "shards": shards,
            "composite_weight_hash": sha256_text(canonical_json(shards)),
        }
    scoring = yaml.safe_load((ROOT / "configs/scoring.yaml").read_text())
    manifests = snapshot_files((OUT / "manifests").glob("*data*manifest.json"))
    manifests += snapshot_files([OUT / "manifests/image_manifest.json"])
    lock = {
        "schema_version": 2,
        "protocol": "capability_gate_atomic_v2",
        "run_id": "atomic-v2-"
        + dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8],
        "git_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "source_commit": SOURCE,
        "data_and_image_manifests": manifests,
        "model_weights": weights,
        "environment": environment,
        "environment_hash": sha256_text(canonical_json(environment)),
        "gpu": gpu,
        "gpu_uuid": gpu["uuid"],
        "frozen_inputs": frozen_inputs(),
        "rendered_prompt_snapshot_hashes": snapshot_files(
            (OUT / "renderer_snapshots").rglob("*.json")
        ),
        "contract_control_hashes": snapshot_files((OUT / "contract_control").rglob("*")),
        "config_hashes": {p.name: _stable_file_hash(p) for p in (ROOT / "configs").glob("*.yaml")},
        "adjudicator_hash": _stable_file_hash(ROOT / "src/capability_gate/atomic_v2_statistics.py"),
        "statistical_code_hashes": {
            p.name: _stable_file_hash(p)
            for p in (ROOT / "src/capability_gate/statistics").glob("*.py")
        },
        "thresholds": scoring,
        "run_seed": 61049273,
        "bootstrap_seed": 9041723,
        "start_timestamp": now(),
        "v1_rows_reused": False,
        "activation_patching_executed": False,
        "model_order": list(KEYS),
        "hash_policy": "raw artifacts and weights; source/config text canonical LF",
    }
    save_once(lockpath, lock)
    _commit([lockpath], "freeze: freeze atomic-v2 scientific run")
    if git("tag", "--list", V2_TAG):
        raise RuntimeError("PRERUN_TAG_ALREADY_EXISTS_DO_NOT_MOVE")
    git(
        "tag",
        "-a",
        V2_TAG,
        "-m",
        "Freeze validated Atomic v2 data, renderers and measurement contracts before formal outputs",
    )
    return lock


def check_lock():
    lock = load(OUT / "formal_run_lock.json")
    if git("cat-file", "-t", V2_TAG) != "tag":
        raise RuntimeError("MISSING_ANNOTATED_PRERUN_TAG")
    frozen = git("rev-parse", V2_TAG + "^{}")
    if git("rev-parse", frozen + "^") != lock["git_commit"]:
        raise RuntimeError("PRERUN_TAG_CHRONOLOGY_INVALID")
    subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", frozen, "HEAD"], check=True
    )
    if frozen_inputs() != lock["frozen_inputs"]:
        raise RuntimeError("FROZEN_SCIENTIFIC_INPUT_CHANGED")
    for record in lock["contract_control_hashes"]:
        if sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError("CONTRACT_CONTROL_CHANGED")
    return lock


def _run_dir(lock, kind):
    return OUT / "formal" / kind / lock["run_id"]


def _verify_new_rows(rows, scenes, lock, key):
    if len(rows) != len(scenes):
        raise RuntimeError("FORMAL_INCOMPLETE")
    old_ids = {
        r["scene_id"] for r in read_jsonl(ROOT / "artifacts/data/atomic_qualification/scenes.jsonl")
    }
    for index, (row, scene) in enumerate(zip(rows, scenes)):
        if row["row_index"] != index or row["scene_id"] != scene["scene_id"]:
            raise RuntimeError("V2_MUST_START_AT_SCENE_1_AND_PRESERVE_ORDER")
        if row["scene_id"] in old_ids or row["run_id"] != lock["run_id"] or row["model_key"] != key:
            raise RuntimeError("V1_OR_FOREIGN_ROWS_FORBIDDEN")
        payload = dict(row)
        expected = payload.pop("result_hash")
        if sha256_text(canonical_json(payload)) != expected:
            raise RuntimeError("RESULT_HASH_MISMATCH")


def run_atomic_model(key):
    from capability_gate.atomic_v2_measurement import adapter_for

    lock = check_lock()
    verify_v1()
    if key not in KEYS:
        raise RuntimeError("UNAUTHORIZED_MODEL")
    base = _run_dir(lock, "atomic")
    pred = base / f"predictions/{NAMES[key]}.jsonl"
    if pred.exists() or (base / f"status/{NAMES[key]}.json").exists():
        raise RuntimeError("FORMAL_RESULTS_MUST_NOT_BE_OVERWRITTEN_OR_RESTARTED")
    if key == KEYS[1] and load(base / "status/qwen.json")["status"] != "complete":
        raise RuntimeError("QWEN_MUST_COMPLETE_FIRST")
    gpu = _gpu_record()
    if gpu["uuid"] != lock["gpu_uuid"] or gpu["free_vram_mib"] < 22528:
        raise RuntimeError("GPU_CHANGED_OR_ANOTHER_MODEL_IS_RESIDENT")
    if _package_versions(Path(sys.executable)) != lock["environment"][key]:
        raise RuntimeError("MODEL_ENVIRONMENT_CHANGED")
    scenes = read_jsonl(DATA / "atomic_qualification/scenes.jsonl")
    if len(scenes) != 256:
        raise RuntimeError("V2_DATA_INCOMPLETE")
    config = yaml.safe_load((ROOT / "configs/atomic_tasks.yaml").read_text())
    runtime = base / f"runtime/{NAMES[key]}"
    runtime.mkdir(parents=True, exist_ok=True)
    adapter = adapter_for(key)
    started = now()
    completed = []
    with (runtime / "stderr.log").open("x", encoding="utf-8", newline="\n") as stderr:
        try:
            with contextlib.redirect_stderr(stderr):
                adapter.load()
                adapter.verify_cached_weights()
                save_once(runtime / "load.json", adapter.runtime_metadata())
                for index, scene in enumerate(scenes):
                    prompt = {
                        "system": config["system_instruction"],
                        "user": _prompt_user(
                            scene["question"], scene["options"], config["answer_schema"]
                        ),
                    }
                    image_path = scene["image_path"] if scene.get("requires_image", True) else None
                    result = adapter.score_and_generate(
                        system=prompt["system"],
                        user=prompt["user"],
                        image_path=str(ROOT / image_path) if image_path else None,
                        candidates=config["candidates"],
                        target=scene["target"],
                    )
                    row = prediction_row(
                        key, scene, result, prompt, image_path, lock, index, "atomic_qualification"
                    )
                    append(pred, row)
                    completed.append(scene["scene_id"])
                    save_once(
                        base / f"completion_bitmap/{NAMES[key]}/{index + 1:03d}.json",
                        {
                            "completed_count": index + 1,
                            "scene_ids": completed.copy(),
                            "last_result_hash": row["result_hash"],
                        },
                    )
                    append(
                        runtime / "progress.jsonl",
                        {"timestamp": now(), "completed": index + 1, "total": 256},
                    )
                    print(f"{NAMES[key]} Atomic v2 {index + 1}/256", flush=True)
                _verify_new_rows(read_jsonl(pred), scenes, lock, key)
                status = {
                    "status": "complete",
                    "model_key": key,
                    "rows": 256,
                    "prediction_sha256": sha256_file(pred),
                    "started_at": started,
                    "ended_at": now(),
                }
        except Exception:  # noqa: BLE001 - preserve every formal runtime failure
            status = {
                "status": "MEASUREMENT_IMPLEMENTATION_NO_GO",
                "model_key": key,
                "rows_preserved": len(completed),
                "traceback": traceback.format_exc(),
                "started_at": started,
                "ended_at": now(),
                "capability_conclusion_valid": False,
            }
        finally:
            save_once(runtime / "release.json", adapter.close())
        save_once(base / f"status/{NAMES[key]}.json", status)
    return status


def prediction_row(key, scene, result, prompt, image_path, lock, index, mode):
    scores = result["candidate_scores"]
    row = {
        "schema_version": 4,
        "protocol": "capability_gate_atomic_v2",
        "run_id": lock["run_id"],
        "row_index": index,
        "model_key": key,
        "model": DESCRIPTORS[key].model_id,
        "revision": DESCRIPTORS[key].revision,
        "scene_id": scene.get("scene_id", scene.get("base_quartet_id")),
        "base_quartet_id": scene.get("base_quartet_id"),
        "condition": scene.get("condition"),
        "task": scene["task"],
        "target": scene["target"],
        "template_id": scene.get("template_id"),
        "mode": mode,
        "correct_option_position": scene["correct_option_position"],
        "options": scene["options"],
        "image_path": image_path,
        "image_hash": sha256_file(ROOT / image_path) if image_path else None,
        "prompt": result["rendered_prompt"],
        "prompt_hash": sha256_text(result["rendered_prompt"]),
        "prompt_contract": prompt,
        "candidate_text": [s["candidate"] for s in scores],
        "candidate_token_ids": {s["candidate"]: s["candidate_token_ids"] for s in scores},
        "raw_log_likelihood": {s["candidate"]: s["raw_log_likelihood"] for s in scores},
        "normalized_log_likelihood": {
            s["candidate"]: s["normalized_log_likelihood"] for s in scores
        },
        "candidate_ranking": result["candidate_ranking"],
        "top_answer": result["top_answer"],
        "constrained_generation_answer": result["constrained_answer"],
        "target_margin": result["target_margin"],
        "runtime_seconds": result["runtime_seconds"],
        "raw_adapter_result": result,
        "environment_hash": lock["environment_hash"],
        "config_hash": sha256_text(canonical_json(lock["config_hashes"])),
        "created_at": now(),
    }
    for field in ("axis_map", "psi_fixed_target", "image_bit", "text_bit"):
        if field in scene:
            row[field] = scene[field]
    row["result_hash"] = sha256_text(canonical_json(row))
    return row


def run_atomic_v2():
    lock = check_lock()
    base = _run_dir(lock, "atomic")
    results = {}
    for key in KEYS:
        status = base / f"status/{NAMES[key]}.json"
        if not status.exists():
            _subprocess(
                "capability_gate.atomic_v2_protocol",
                ["model-atomic", "--model-key", key],
                base / f"runtime/{NAMES[key]}/process.log",
            )
        if not status.exists():
            raise RuntimeError("FORMAL_PROCESS_FAILED_WITHOUT_STATUS")
        results[key] = load(status)
        if results[key]["status"] != "complete":
            break
        _commit([base], f"results: execute full {NAMES[key].upper()} atomic-v2")
    return results


def adjudicate_atomic_v2():
    from capability_gate.atomic_v2_statistics import adjudicate_atomic_rows_v2

    lock = check_lock()
    base = _run_dir(lock, "atomic")
    for key in KEYS:
        status_path = base / f"status/{NAMES[key]}.json"
        prediction_path = base / f"predictions/{NAMES[key]}.jsonl"
        status = load(status_path) if status_path.exists() else {}
        if (
            status.get("status") != "complete"
            or not prediction_path.exists()
            or status.get("prediction_sha256") != sha256_file(prediction_path)
        ):
            result = adjudicate_atomic_rows_v2([])
            result["execution_error"] = f"Missing complete status or verified hash: {key}"
            return save_once(OUT / "statistics/atomic_adjudication.json", result)
    rows = [r for p in sorted((base / "predictions").glob("*.jsonl")) for r in read_jsonl(p)]
    for key in KEYS:
        subset = [r for r in rows if r["model_key"] == key]
        if len(subset) == 256:
            _verify_new_rows(
                subset, read_jsonl(DATA / "atomic_qualification/scenes.jsonl"), lock, key
            )
    result = adjudicate_atomic_rows_v2(rows)
    save_once(OUT / "statistics/atomic_adjudication.json", result)
    return result


def run_joint_after_atomic_v2():
    from capability_gate.atomic_v2_statistics import require_atomic_v2_go

    atomic = load(OUT / "statistics/atomic_adjudication.json")
    if atomic["decision"] != "ATOMIC_COHORT_GO":
        return save_once(
            OUT / "statistics/joint_execution.json",
            {
                "status": "NOT_RUN_BY_ATOMIC_GATE",
                "upstream_decision": atomic["decision"],
                "joint_executed": False,
                "activation_patching_executed": False,
            },
        )
    require_atomic_v2_go(atomic)
    lock = check_lock()
    base = _run_dir(lock, "joint")
    results = {}
    for key in KEYS:
        status = base / f"status/{NAMES[key]}.json"
        if not status.exists():
            _subprocess(
                "capability_gate.atomic_v2_protocol",
                ["model-joint", "--model-key", key],
                base / f"runtime/{NAMES[key]}/process.log",
            )
        results[key] = load(status)
        if results[key]["status"] != "complete":
            break
        _commit([base], f"results: execute {NAMES[key]} joint only after atomic cohort GO")
    save_once(OUT / "statistics/joint_execution.json", results)
    return results


def run_joint_model(key):
    from capability_gate.atomic_v2_measurement import adapter_for
    from capability_gate.atomic_v2_statistics import require_atomic_v2_go

    require_atomic_v2_go(load(OUT / "statistics/atomic_adjudication.json"))
    lock = check_lock()
    if key not in KEYS:
        raise RuntimeError("UNAUTHORIZED_MODEL")
    base = _run_dir(lock, "joint")
    pred = base / f"predictions/{NAMES[key]}.jsonl"
    retention = base / f"retention/{NAMES[key]}.jsonl"
    if pred.exists() or retention.exists():
        raise RuntimeError("FORMAL_RESULTS_MUST_NOT_BE_OVERWRITTEN")
    gpu = _gpu_record()
    if gpu["uuid"] != lock["gpu_uuid"] or gpu["free_vram_mib"] < 22528:
        raise RuntimeError("GPU_CHANGED_OR_MODEL_RESIDENT")
    if _package_versions(Path(sys.executable)) != lock["environment"][key]:
        raise RuntimeError("MODEL_ENVIRONMENT_CHANGED")
    joint = yaml.safe_load((ROOT / "configs/joint_screen.yaml").read_text())
    atomic = yaml.safe_load((ROOT / "configs/atomic_tasks.yaml").read_text())
    quartets = read_jsonl(DATA / "joint_composition_screen/quartets.jsonl")
    if len(quartets) != 128:
        raise RuntimeError("JOINT_DATA_INCOMPLETE")
    adapter = adapter_for(key)
    runtime = base / f"runtime/{NAMES[key]}"
    runtime.mkdir(parents=True, exist_ok=True)
    index = 0
    try:
        adapter.load()
        adapter.verify_cached_weights()
        save_once(runtime / "load.json", adapter.runtime_metadata())
        for quartet in quartets:
            for condition in quartet["conditions"]:
                logical = {
                    **condition,
                    "task": "joint_composition",
                    "options": quartet["options"],
                    **{
                        k: quartet[k]
                        for k in ("base_quartet_id", "template_id", "axis_map", "psi_fixed_target")
                    },
                }
                for mode in ("joint", "image_only", "text_only", "question_only"):
                    question = (
                        condition["question"]
                        if mode in {"joint", "text_only"}
                        else condition["question_without_premise"]
                    )
                    image_path = (
                        condition["image_path"] if mode in {"joint", "image_only"} else None
                    )
                    prompt = {
                        "system": joint["system_instruction"],
                        "user": _prompt_user(question, quartet["options"], joint["answer_schema"]),
                    }
                    result = adapter.score_and_generate(
                        **prompt,
                        image_path=str(ROOT / image_path) if image_path else None,
                        candidates=joint["candidates"],
                        target=condition["target"],
                    )
                    append(
                        pred,
                        prediction_row(key, logical, result, prompt, image_path, lock, index, mode),
                    )
                    index += 1
            print(f"{NAMES[key]} Joint rows {index}/2048", flush=True)
        for index, scene in enumerate(read_jsonl(DATA / "atomic_qualification/scenes.jsonl")):
            prompt = {
                "system": joint["system_instruction"]
                + " For direct control items, use the supplied direct evidence and a cardinal answer.",
                "user": _prompt_user(scene["question"], scene["options"], atomic["answer_schema"]),
            }
            image_path = scene["image_path"] if scene.get("requires_image", True) else None
            result = adapter.score_and_generate(
                **prompt,
                image_path=str(ROOT / image_path) if image_path else None,
                candidates=atomic["candidates"],
                target=scene["target"],
            )
            append(
                retention,
                prediction_row(
                    key, scene, result, prompt, image_path, lock, index, "atomic_retention"
                ),
            )
        status = {
            "status": "complete",
            "prediction_rows": 2048,
            "retention_rows": 256,
            "predictions_sha256": sha256_file(pred),
            "retention_sha256": sha256_file(retention),
        }
    except Exception:  # noqa: BLE001 - preserve every formal runtime failure
        status = {
            "status": "MEASUREMENT_IMPLEMENTATION_NO_GO",
            "traceback": traceback.format_exc(),
            "rows_preserved": index,
        }
    finally:
        save_once(runtime / "release.json", adapter.close())
    save_once(base / f"status/{NAMES[key]}.json", status)
    return status


def analyze_joint_after_atomic_v2():
    from capability_gate.atomic_v2_statistics import analyze_joint_rows_v2

    atomic = load(OUT / "statistics/atomic_adjudication.json")
    if atomic["decision"] != "ATOMIC_COHORT_GO":
        result = {**atomic, "joint_status": "NOT_RUN_BY_ATOMIC_GATE"}
    else:
        base = _run_dir(check_lock(), "joint")
        for key in KEYS:
            status_path = base / f"status/{NAMES[key]}.json"
            status = load(status_path) if status_path.exists() else {}
            for kind, field in (
                ("predictions", "predictions_sha256"),
                ("retention", "retention_sha256"),
            ):
                path = base / f"{kind}/{NAMES[key]}.jsonl"
                if (
                    status.get("status") != "complete"
                    or not path.exists()
                    or status.get(field) != sha256_file(path)
                ):
                    result = analyze_joint_rows_v2([], [], atomic)
                    result["execution_error"] = f"Missing complete Joint status or hash: {key}"
                    return save_once(OUT / "statistics/joint_analysis.json", result)
        predictions = [r for p in (base / "predictions").glob("*.jsonl") for r in read_jsonl(p)]
        retention = [r for p in (base / "retention").glob("*.jsonl") for r in read_jsonl(p)]
        result = analyze_joint_rows_v2(predictions, retention, atomic)
    return save_once(OUT / "statistics/joint_analysis.json", result)


def _report_once(name, title, result):
    path = REPORT / name
    text = (
        f"# {title}\n\n"
        + "```json\n"
        + json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n```\n"
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def build_atomic_v2_report():
    verify_v1()
    final_path = OUT / "final_decision.json"
    if final_path.exists():
        return load(final_path)
    if (OUT / "statistics/joint_analysis.json").exists():
        source = load(OUT / "statistics/joint_analysis.json")
    elif (OUT / "statistics/atomic_adjudication.json").exists():
        source = load(OUT / "statistics/atomic_adjudication.json")
        if source["decision"] == "ATOMIC_COHORT_GO":
            raise RuntimeError("JOINT_REQUIRED_BEFORE_FINAL_REPORT")
    else:
        data_records = [
            load(p)
            for p in (OUT / "data_validation.json", OUT / "joint_data_validation.json")
            if p.exists()
        ]
        renderer_records = [load(p) for p in (OUT / "renderer_snapshots").glob("*/manifest.json")]
        control_records = [load(p) for p in (OUT / "contract_control").glob("*/manifest.json")]
        if any(r.get("overall_gate") is False for r in data_records):
            source = {
                "decision": "ATOMIC_V2_DATA_INVALID",
                "q1_potential": "NOT_EVALUATED_MEASUREMENT_FAILURE",
                "exact_next_action": "TERMINATE_CROSS_MODAL_SYNERGY_LINE",
            }
        else:
            if not any(
                r.get("overall_gate") is False for r in [*renderer_records, *control_records]
            ):
                raise RuntimeError("NO_TERMINAL_DECISION_WHILE_REQUIRED_WORK_IS_UNRUN")
            source = {
                "decision": "MEASUREMENT_IMPLEMENTATION_NO_GO",
                "q1_potential": "NOT_EVALUATED_MEASUREMENT_FAILURE",
                "exact_next_action": "TERMINATE_AFTER_FINAL_MEASUREMENT_FAILURE",
            }
    result = {k: source.get(k) for k in ("decision", "q1_potential", "exact_next_action")}
    if None in result.values():
        raise RuntimeError("INCOMPLETE_FINAL_CLASSIFICATION")
    result.update(
        {
            "schema_version": 2,
            "protocol": "capability_gate_atomic_v2",
            "created_at": now(),
            "historical_protocol": "INVALID_FOR_COHORT_CONCLUSION",
            "v1_results_preserved": True,
            "activation_patching_executed": False,
            "mechanism_result": "NOT_EXECUTED",
            "formal_v1_rows_reused": False,
        }
    )
    for filename, title, path in (
        (
            "atomic_qualification.md",
            "Clean Atomic v2 qualification",
            OUT / "statistics/atomic_adjudication.json",
        ),
        (
            "joint_composition.md",
            "Conditional Joint composition",
            OUT / "statistics/joint_analysis.json",
        ),
    ):
        _report_once(
            filename,
            title,
            load(path)
            if path.exists()
            else {"status": "NOT_EXECUTED", "blocking_gate": result["decision"]},
        )
    _report_once("final_decision.md", "Atomic v2 final decision", result)
    save_once(final_path, result)
    manifest_paths = [
        p
        for folder in (OUT, DATA, REPORT, ROOT / "research/atomic_v2", ROOT / "tests/snapshots")
        for p in folder.rglob("*")
        if p.is_file() and p.name not in {"final_manifest.json", "verification.json"}
    ]
    save_once(
        OUT / "manifests/final_manifest.json",
        {"schema_version": 1, "files": snapshot_files(manifest_paths)},
    )
    return result


def verify_atomic_v2_artifacts():
    result = verify_v1()
    manifest = load(OUT / "manifests/final_manifest.json")
    for record in manifest["files"]:
        if sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"FINAL_MANIFEST_HASH_MISMATCH: {record['path']}")
    final = load(OUT / "final_decision.json")
    if final["activation_patching_executed"]:
        raise RuntimeError("ACTIVATION_PATCHING_FORBIDDEN")
    result.update(
        {
            "final_manifest_recomputed": True,
            "final_artifacts": len(manifest["files"]),
            "final_decision": final,
            "overall_gate": True,
        }
    )
    return result


def clean_rerun():
    """Single-entry gated execution; terminal failures stop without outcome tuning."""
    if (OUT / "final_decision.json").exists():
        verified = verify_atomic_v2_artifacts()
        return {
            "final": load(OUT / "final_decision.json"),
            "verification_gate": verified["overall_gate"],
            "terminal_run_reused_read_only": True,
        }
    from capability_gate.atomic_v2_data import (
        DataValidationError,
        generate_atomic_v2_data,
        validate_atomic_v2_data,
        validate_joint_data_before_atomic_v2,
    )

    verify_v1()
    freeze_invalid_atomic_v1()
    try:
        generate_atomic_v2_data()
    except DataValidationError:
        if not (OUT / "data_validation.json").exists():
            raise
    atomic_data = validate_atomic_v2_data(raise_on_invalid=False)
    joint_data = validate_joint_data_before_atomic_v2(raise_on_invalid=False)
    _commit(
        [
            DATA,
            OUT / "data",
            OUT / "manifests",
            OUT / "data_validation.json",
            OUT / "joint_data_validation.json",
            REPORT / "data_validation.md",
        ],
        "data: validate fresh atomic and joint data before any model output",
    )
    if not atomic_data.get("overall_gate", False) or not joint_data.get("overall_gate", False):
        return _finish_terminal()
    renderers = validate_model_renderers()
    _report_once("renderer_validation.md", "Official renderer validation", renderers)
    _commit(
        [OUT / "renderer_snapshots", ROOT / "tests/snapshots", REPORT / "renderer_validation.md"],
        "test: add actual renderer snapshots before contract control",
    )
    if not all(r.get("overall_gate", False) for r in renderers.values()):
        return _finish_terminal()
    controls = run_contract_control()
    _report_once(
        "contract_control.md",
        "Engineering-only contract control; not scientific accuracy",
        controls,
    )
    _commit(
        [OUT / "contract_control", REPORT / "contract_control.md"],
        "test: execute engineering-only contract-control for both frozen models",
    )
    if all(r.get("overall_gate", False) for r in controls.values()):
        freeze_atomic_v2_run()
        run_atomic_v2()
        adjudicate_atomic_v2()
        _commit([OUT / "statistics"], "results: adjudicate clean atomic cohort")
        run_joint_after_atomic_v2()
        analyze_joint_after_atomic_v2()
    return _finish_terminal()


def _finish_terminal():
    result = build_atomic_v2_report()
    verification = verify_atomic_v2_artifacts()
    _commit([OUT, REPORT], "results: publish final atomic-v2 decision")
    return {"final": result, "verification_gate": verification["overall_gate"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["model-atomic", "model-joint"])
    parser.add_argument("--model-key", required=True, choices=KEYS)
    args = parser.parse_args()
    result = (
        run_atomic_model(args.model_key)
        if args.operation == "model-atomic"
        else run_joint_model(args.model_key)
    )
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
