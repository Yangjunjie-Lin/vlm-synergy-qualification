from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from capability_gate.artifacts import write_json
from capability_gate.paths import ARTIFACTS, ROOT

FROZEN_COMMIT = "88741f0301c61c0abf3e63a5903183e9664af552"
FROZEN_BRANCH = "codex/real-vlm-synergy-pilot"
FROZEN_TAG = "synergy-trace-behavioral-no-go-2026-08-31"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_historical_freeze() -> dict[str, Any]:
    old = ROOT.parent / "vlm-synergy-trace"
    branch_commit = _git(old, "rev-parse", FROZEN_BRANCH)
    tag_target = _git(old, "rev-parse", f"{FROZEN_TAG}^{{}}")
    tag_type = _git(old, "cat-file", "-t", FROZEN_TAG)
    status = _git(old, "status", "--porcelain")
    result = {
        "schema_version": 1,
        "source_repository": "https://github.com/Yangjunjie-Lin/vlm-synergy-trace",
        "local_repository": str(old),
        "frozen_branch": FROZEN_BRANCH,
        "frozen_commit": FROZEN_COMMIT,
        "branch_commit": branch_commit,
        "annotated_tag": FROZEN_TAG,
        "tag_object_type": tag_type,
        "tag_peeled_target": tag_target,
        "worktree_clean": status == "",
        "historical_decision": "BEHAVIORAL_NO_GO",
        "historical_potential": "NO_POTENTIAL under frozen configuration",
        "activation_patching_executed": False,
        "mechanism_claims_present": False,
        "gate": branch_commit == FROZEN_COMMIT
        and tag_target == FROZEN_COMMIT
        and tag_type == "tag"
        and status == "",
    }
    write_json(ARTIFACTS / "manifests" / "historical_freeze.json", result)
    if not result["gate"]:
        raise RuntimeError("frozen historical state verification failed")
    return result
