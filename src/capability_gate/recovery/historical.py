from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from capability_gate.artifacts import sha256_file, write_json
from capability_gate.paths import ARTIFACTS, ROOT

BASE_COMMIT = "32a8333bf0106579834fb7a0747c81d67b49ac10"
FROZEN_BRANCH = "codex/three-family-capability-screen"
RECOVERY_TAG = "capability-gate-adapter-block-2026-08-31"
RECOVERY_TAG_MESSAGE = (
    "Freeze pre-forward BLOCKED_BY_MODEL_ADAPTER state before root-cause recovery"
)
SYNERGY_TAG = "synergy-trace-behavioral-no-go-2026-08-31"
SYNERGY_TARGET = "88741f0301c61c0abf3e63a5903183e9664af552"

HISTORICAL_SHA256 = {
    "reports/final_qualification_decision.md": (
        "c6b62482b2a40b8d5ba33fb535e31383e7779c18f730bb0ecace4fae88b65098"
    ),
    "reports/atomic_qualification.md": (
        "1624fa9a65874ea8e117fa9ecc4f89a2399cb3824e9325dd35b0d0b0e14fde0f"
    ),
    "reports/joint_composition_screen.md": (
        "ea537e0f552f1177344c5d77f41bb5cfb6712bfc9630d1541b98fe29f10119ef"
    ),
    "artifacts/manifests/engineering_smoke.json": (
        "0ba3c4df6e65c41d831f86f52db0ff3028be5532814c26e0351402e17f8b0ba3"
    ),
    "artifacts/manifests/final_decision.json": (
        "d7860816324cfb2923ec904af412b73e9c99be0c172623e117fac9e5e145ddae"
    ),
    "artifacts/manifests/atomic_run.json": (
        "3745a0c01f2fb0f13c825d26e61e19d594e314a445bdf95217dba5c99ee81c0e"
    ),
    "artifacts/manifests/joint_screen_status.json": (
        "8483d02b0362181db2c05c19a3e9f37c0da0f0a326a54254ffb898ae457560f4"
    ),
}


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _ls_remote(url: str, *patterns: str) -> dict[str, str]:
    output = subprocess.check_output(
        ["git", "ls-remote", url, *patterns], text=True, stderr=subprocess.STDOUT
    )
    return {ref: sha for sha, ref in (line.split() for line in output.splitlines())}


def _github_archived(repository: str) -> bool:
    output = subprocess.check_output(
        ["gh", "repo", "view", repository, "--json", "isArchived"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return bool(json.loads(output)["isArchived"])


def verify_historical_block() -> dict[str, Any]:
    atomic = json.loads((ROOT / "artifacts/manifests/atomic_run.json").read_text())
    joint = json.loads((ROOT / "artifacts/manifests/joint_screen_status.json").read_text())
    final = json.loads((ROOT / "artifacts/manifests/final_decision.json").read_text())
    remote = _ls_remote(
        "https://github.com/Yangjunjie-Lin/vlm-synergy-qualification.git",
        f"refs/heads/{FROZEN_BRANCH}",
    )
    synergy = _ls_remote(
        "https://github.com/Yangjunjie-Lin/vlm-synergy-trace.git",
        f"refs/tags/{SYNERGY_TAG}",
        f"refs/tags/{SYNERGY_TAG}^{{}}",
    )
    artifact_hashes = {
        path: {"expected": expected, "actual": sha256_file(ROOT / path)}
        for path, expected in HISTORICAL_SHA256.items()
    }
    checks = {
        "local_frozen_branch_target": _git("rev-parse", FROZEN_BRANCH) == BASE_COMMIT,
        "remote_frozen_branch_target": (remote.get(f"refs/heads/{FROZEN_BRANCH}") == BASE_COMMIT),
        "recovery_tag_is_annotated": _git("cat-file", "-t", RECOVERY_TAG) == "tag",
        "recovery_tag_target": _git("rev-list", "-n", "1", RECOVERY_TAG) == BASE_COMMIT,
        "recovery_tag_message": (
            _git("for-each-ref", f"refs/tags/{RECOVERY_TAG}", "--format=%(contents:subject)")
            == RECOVERY_TAG_MESSAGE
        ),
        "historical_artifacts_unchanged": all(
            value["actual"] == value["expected"] for value in artifact_hashes.values()
        ),
        "atomic_predictions_zero": atomic.get("model_forward_passes") == 0,
        "joint_predictions_zero": joint.get("model_forward_passes") == 0,
        "atomic_prediction_file_absent": not (ROOT / "artifacts/atomic/predictions.jsonl").exists(),
        "joint_prediction_file_absent": not (ROOT / "artifacts/joint/predictions.jsonl").exists(),
        "historical_decision_adapter_block": final.get("decision") == "BLOCKED_BY_MODEL_ADAPTER",
        "historical_no_feasible_is_not_capability_result": (
            atomic.get("model_forward_passes") == 0
            and joint.get("model_forward_passes") == 0
            and final.get("decision") == "BLOCKED_BY_MODEL_ADAPTER"
        ),
        "synergy_trace_tag_unmoved": synergy.get(f"refs/tags/{SYNERGY_TAG}^{{}}") == SYNERGY_TARGET,
        "recoalign_archived": _github_archived("Yangjunjie-Lin/recoalign"),
        "vlm_construct_audit_archived": _github_archived("Yangjunjie-Lin/vlm-construct-audit"),
    }
    result = {
        "schema_version": 1,
        "base_commit": BASE_COMMIT,
        "frozen_branch": FROZEN_BRANCH,
        "recovery_tag": RECOVERY_TAG,
        "artifact_hashes": artifact_hashes,
        "checks": checks,
        "overall_gate": all(checks.values()),
    }
    path = ARTIFACTS / "engineering_recovery/manifests/historical_block_verification.json"
    write_json(path, result)
    if not result["overall_gate"]:
        raise RuntimeError(f"historical integrity verification failed: {checks}")
    return result


def historical_paths() -> list[Path]:
    return [ROOT / path for path in HISTORICAL_SHA256]
