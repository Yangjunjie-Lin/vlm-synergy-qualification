from __future__ import annotations

import importlib.metadata
from pathlib import Path

from capability_gate.recovery import dependencies


def test_phi_preflight_lists_missing_backoff(monkeypatch) -> None:
    def without_backoff(package: str) -> str:
        if package == "backoff":
            raise importlib.metadata.PackageNotFoundError(package)
        return dependencies.DEPENDENCY_SPECS["phi4_multimodal_5_6b"][package]

    monkeypatch.setattr(dependencies.importlib.metadata, "version", without_backoff)
    monkeypatch.setitem(dependencies.PYTHON_MINORS, "phi4_multimodal_5_6b", (3, 11))
    result = dependencies.dependency_preflight("phi4_multimodal_5_6b")
    assert result["status"] == "DEPENDENCY_PREFLIGHT_FAIL"
    assert result["missing_dependencies"] == ["backoff"]
    assert not result["counts_as_model_load_attempt"]


def test_phi_preflight_passes_when_all_locked_dependencies_are_present(monkeypatch) -> None:
    expected = dependencies.DEPENDENCY_SPECS["phi4_multimodal_5_6b"]
    monkeypatch.setattr(dependencies.importlib.metadata, "version", expected.__getitem__)
    monkeypatch.setitem(dependencies.PYTHON_MINORS, "phi4_multimodal_5_6b", (3, 11))
    result = dependencies.dependency_preflight("phi4_multimodal_5_6b")
    assert result["status"] == "DEPENDENCY_PREFLIGHT_PASS"
    assert result["missing_dependencies"] == []
    assert result["version_mismatches"] == []


def test_three_worker_locks_are_exact_and_rebuildable() -> None:
    root = Path(__file__).resolve().parents[1]
    locks = sorted((root / "envs").glob("*/requirements.lock"))
    assert [path.parent.name for path in locks] == ["glm", "phi", "qwen"]
    for path in locks:
        package_lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith(("#", " ", "--"))
        ]
        assert package_lines
        assert all("==" in line for line in package_lines)
