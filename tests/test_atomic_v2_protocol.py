from __future__ import annotations

import pytest

from capability_gate import atomic_v2_protocol as protocol
from capability_gate.artifacts import canonical_json, sha256_text


def test_immutable_artifact_cannot_be_overwritten(tmp_path):
    path = tmp_path / "formal.json"
    protocol.save_once(path, {"a": 1})
    protocol.save_once(path, {"a": 1})
    with pytest.raises(RuntimeError, match="IMMUTABLE"):
        protocol.save_once(path, {"a": 2})


def test_passing_controls_without_formal_outputs_cannot_produce_terminal_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(protocol, "OUT", tmp_path)
    monkeypatch.setattr(protocol, "verify_v1", dict)
    for key in protocol.KEYS:
        protocol.save_once(
            tmp_path / "contract_control" / key / "manifest.json", {"overall_gate": True}
        )
    with pytest.raises(RuntimeError, match="REQUIRED_WORK_IS_UNRUN"):
        protocol.build_atomic_v2_report()
    assert not (tmp_path / "final_decision.json").exists()


def test_v2_must_start_at_scene_one_and_reject_v1_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    old = tmp_path / "artifacts/data/atomic_qualification/scenes.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text('{"scene_id":"old"}\n', encoding="utf-8")
    row = {"row_index": 0, "scene_id": "fresh", "run_id": "v2", "model_key": "qwen"}
    row["result_hash"] = sha256_text(canonical_json(row))
    protocol._verify_new_rows([row], [{"scene_id": "fresh"}], {"run_id": "v2"}, "qwen")
    with pytest.raises(RuntimeError, match="START_AT_SCENE_1"):
        protocol._verify_new_rows(
            [{**row, "row_index": 15}], [{"scene_id": "fresh"}], {"run_id": "v2"}, "qwen"
        )
    with pytest.raises(RuntimeError, match="V1_OR_FOREIGN"):
        protocol._verify_new_rows(
            [{**row, "scene_id": "old"}], [{"scene_id": "old"}], {"run_id": "v2"}, "qwen"
        )


def test_final_manifest_can_be_recomputed_and_detects_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    monkeypatch.setattr(protocol, "OUT", tmp_path)
    monkeypatch.setattr(protocol, "verify_v1", dict)
    artifact = tmp_path / "artifact.json"
    protocol.save_once(artifact, {"raw": 1})
    protocol.save_once(tmp_path / "final_decision.json", {"activation_patching_executed": False})
    protocol.save_once(
        tmp_path / "manifests/final_manifest.json", {"files": protocol.snapshot_files([artifact])}
    )
    assert protocol.verify_atomic_v2_artifacts()["final_manifest_recomputed"]
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="FINAL_MANIFEST_HASH"):
        protocol.verify_atomic_v2_artifacts()


def test_old_tag_mutation_fails_closed(monkeypatch):
    def fake_git(*args):
        if args[0] == "ls-tree":
            return ""
        if args[0] == "diff":
            return ""
        if args[0] == "cat-file":
            return "commit"
        raise AssertionError(args)

    monkeypatch.setattr(protocol, "git", fake_git)
    with pytest.raises(RuntimeError, match="V1_TAG_CHANGED"):
        protocol.verify_v1()
