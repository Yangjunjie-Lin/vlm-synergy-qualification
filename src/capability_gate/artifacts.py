from __future__ import annotations

import datetime as datetime_module
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def config_hash(paths: Iterable[Path]) -> str:
    payload = [{"path": path.name, "sha256": sha256_file(path)} for path in sorted(paths)]
    return sha256_text(canonical_json(payload))


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(root: Path) -> dict[str, Any]:
    return {
        # timezone.utc keeps the isolated Python 3.10 Phi worker compatible.
        "created_at": datetime_module.datetime.now(
            datetime_module.timezone.utc  # noqa: UP017
        ).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
        "git_commit": git_value(root, "rev-parse", "HEAD"),
        "git_branch": git_value(root, "branch", "--show-current"),
    }


def build_manifest(root: Path, paths: Iterable[Path], kind: str) -> dict[str, Any]:
    files = []
    for path in sorted(set(paths)):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": 1, "kind": kind, **runtime_metadata(root), "files": files}
