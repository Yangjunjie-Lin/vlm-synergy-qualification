"""Standard-library-only, read-only terminal evidence verification.

No processor, model, tensor library, generation function, or cloud API is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/program_closeout"
SOURCE = "eb12bd64da6cb37d4e43631e4f2e08492a9007fb"
BRANCH = "codex/atomic-protocol-v2-clean-rerun"
TAG = "capability-gate-terminal-measurement-no-go-2026-09-10"
TAG_MESSAGE = (
    "Terminal closeout after preregistered Atomic v2 measurement implementation no-go; "
    "no capability, joint, or mechanism conclusion was identified."
)
EXPECTED = {
    "decision": "MEASUREMENT_IMPLEMENTATION_NO_GO",
    "q1_potential": "NOT_EVALUATED_MEASUREMENT_FAILURE",
    "exact_next_action": "TERMINATE_AFTER_FINAL_MEASUREMENT_FAILURE",
}
OLD_TAGS = {
    "capability-gate-adapter-block-2026-08-31": "5e8094970d228da8ce6ec092ac107ff7c5e70055",
    "capability-gate-atomic-v1-protocol-invalid-2026-09-10": "bce668ac9f95f2cd0ee4639c45985eff478e3752",
}
ENTRYPOINT_CHANGES = {
    ".gitattributes",
    "src/capability_gate/__init__.py",
    "src/capability_gate/cli.py",
    "workers/_entrypoint.py",
}
KEYS = ("qwen2_5_vl_7b", "glm4_1v_9b")


def git_bytes(*args, stdin=None):
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(ROOT), *args],
        input=stdin,
        stderr=subprocess.PIPE,
    )


def git(*args):
    return git_bytes(*args).decode("utf-8").strip()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    if not Path(path).exists():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require(condition, description):
    if not condition:
        raise RuntimeError("CLOSEOUT_INTEGRITY_FAILURE: " + description)


def safe_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()), "manifest path escapes repository")
    return path


def source_inventory():
    """Return SHA256 of immutable Git-clean bytes, retaining raw manifests separately.

    This normalizes only Git's existing text transport rules; raw recorded artifact
    SHA256 checks below remain strict and are not replaced by normalized hashes.
    """
    entries = []
    for item in git_bytes("ls-tree", "-r", "-z", SOURCE).split(b"\0"):
        if not item:
            continue
        metadata, name = item.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        path = name.decode("utf-8")
        if kind == "blob" and path not in ENTRYPOINT_CHANGES:
            entries.append({"path": path, "git_blob_sha1": oid, "mode": mode})
    payload = "".join(item["git_blob_sha1"] + "\n" for item in entries).encode()
    stream = io.BytesIO(git_bytes("cat-file", "--batch", stdin=payload))
    for entry in entries:
        oid, kind, size = stream.readline().decode().strip().split()
        content = stream.read(int(size))
        require(stream.read(1) == b"\n", "invalid git batch framing")
        require(oid == entry["git_blob_sha1"] and kind == "blob", "git blob mismatch")
        entry.update(bytes=int(size), sha256_git_clean=hashlib.sha256(content).hexdigest())
    return entries


def verify_canonical_index():
    index = read(OUT / "canonical_artifact_index.json")
    entries = index["files"]
    actual_inventory = source_inventory()
    require(entries == actual_inventory, "canonical index does not describe source commit")
    require(all(safe_path(item["path"]).is_file() for item in entries), "preserved file missing")
    input_paths = "".join(item["path"] + "\n" for item in entries).encode()
    observed = git_bytes("hash-object", "--stdin-paths", stdin=input_paths).decode().splitlines()
    require(len(observed) == len(entries), "hash-object record count")
    changed = [item["path"] for item, oid in zip(entries, observed) if oid != item["git_blob_sha1"]]
    require(not changed, "preserved baseline content changed: " + repr(changed))
    return len(entries)


def verify_raw_manifest(relative):
    manifest = read(safe_path(relative))
    checked = 0
    for entry in manifest["files"]:
        path = safe_path(entry["path"])
        require(
            path.is_file() and digest(path) == entry["sha256"],
            "raw artifact hash mismatch: " + entry["path"],
        )
        if "bytes" in entry:
            require(path.stat().st_size == entry["bytes"], "artifact size mismatch")
        checked += 1
    return checked


def verify_source_protocol():
    preserved = verify_canonical_index()
    for name, oid in OLD_TAGS.items():
        require(git("cat-file", "-t", name) == "tag", "old tag is not annotated")
        require(git("rev-parse", name) == oid, "old annotated tag changed")
    require(
        git("rev-parse", "capability-gate-atomic-v1-protocol-invalid-2026-09-10^{}")
        == "eca06a65e9ea7c384ff17859bcd9b28c79630219",
        "v1 tag target changed",
    )
    require(
        not git("tag", "--list", "capability-gate-atomic-v2-prerun-freeze"),
        "unexpected formal prerun tag",
    )
    git_bytes("merge-base", "--is-ancestor", SOURCE, "HEAD")
    final = read(ROOT / "artifacts/atomic_v2/final_decision.json")
    require({key: final.get(key) for key in EXPECTED} == EXPECTED, "source final decision changed")
    require(not (ROOT / "artifacts/atomic_v2/formal_run_lock.json").exists(), "formal lock exists")
    formal = list((ROOT / "artifacts/atomic_v2/formal").rglob("*.jsonl"))
    require(sum(len(rows(path)) for path in formal) == 0, "formal v2 output exists")
    patch_files = list((ROOT / "artifacts").rglob("*patch*.jsonl"))
    require(sum(len(rows(path)) for path in patch_files) == 0, "patching output exists")
    manifest_count = verify_raw_manifest("artifacts/atomic_v2/manifests/final_manifest.json")
    supplemental = verify_raw_manifest("artifacts/atomic_v2/closeout/manifest.json")
    audit = read(ROOT / "artifacts/compute_migration/execution_audit.json")
    v1_base = ROOT / "artifacts/compute_migration/formal/atomic" / audit["run_id"]
    for name in ("qwen", "glm"):
        path = v1_base / f"predictions/{name}.jsonl"
        predictions = rows(path)
        checkpoints = rows(v1_base / f"row_hashes/{name}.jsonl")
        require(len(predictions) == len(checkpoints) == 256, "v1 row/checkpoint count")
        require(
            digest(path) == audit["formal_outputs"][name + "_predictions_sha256"],
            "v1 prediction hash",
        )
        for prediction, checkpoint in zip(predictions, checkpoints):
            unsigned = dict(prediction, result_hash=None)
            require(object_digest(unsigned) == prediction["result_hash"], "v1 result hash")
            require(object_digest(prediction) == checkpoint["row_line_hash"], "v1 checkpoint hash")
    controls = {}
    for key in KEYS:
        relative = f"artifacts/atomic_v2/contract_control/{key}/manifest.json"
        verify_raw_manifest(relative)
        control = read(ROOT / relative)
        predictions = rows(ROOT / f"artifacts/atomic_v2/contract_control/{key}/predictions.jsonl")
        require(len(predictions) == 16, "engineering control count")
        cll = Counter(row["result"]["top_answer"] for row in predictions)
        generation = Counter(row["result"]["constrained_answer"] for row in predictions)
        expected = (
            dict.fromkeys(("north", "south", "east", "west"), 4) if key == KEYS[0] else {"east": 16}
        )
        require(cll == generation == expected, "engineering control observations changed")
        require(control["overall_gate"] is (key == KEYS[0]), "engineering verdict changed")
        for row in predictions:
            unsigned = {k: v for k, v in row.items() if k != "artifact_sha256"}
            require(object_digest(unsigned) == row["artifact_sha256"], "control row hash")
        renderer = f"artifacts/atomic_v2/renderer_snapshots/{key}/"
        if key == KEYS[1]:
            renderer += "repair_01/"
        verify_raw_manifest(renderer + "manifest.json")
        snapshot = read(ROOT / renderer / "manifest.json")
        require(
            snapshot["snapshot_count"] == 8 and snapshot["overall_gate"],
            "renderer evidence changed",
        )
        controls[key] = {
            "rows": 16,
            "gate": control["overall_gate"],
            "cll_counts": dict(cll),
            "generation_counts": dict(generation),
        }
    repair = read(ROOT / "artifacts/atomic_v2/renderer_repair_record.json")
    for record in repair["archived_terminal_documents"]:
        require(
            digest(safe_path(record["archived_path"])) == record["sha256"],
            "archived failure changed",
        )
    return {
        "schema_version": 1,
        "overall_gate": True,
        "verification_root": str(ROOT),
        "source_commit": SOURCE,
        "canonical_files": preserved,
        "v1_rows_verified": 512,
        "v2_final_manifest_entries": manifest_count,
        "supplemental_entries": supplemental,
        "control_rows_verified": 32,
        "controls": controls,
        "source_final_decision": final,
        "formal_atomic_rows": 0,
        "joint_rows": 0,
        "activation_patching_rows": 0,
        "model_or_processor_loads": 0,
        "new_model_forwards": 0,
        "external_model_weights_rehashed": False,
        "verification_mode": "STANDARD_LIBRARY_ONLY_READ_ONLY",
    }


def verify_terminal():
    result = verify_source_protocol()
    count = verify_raw_manifest("artifacts/program_closeout/terminal_manifest.json")
    terminal = read(OUT / "terminal_manifest.json")
    require(
        terminal["source_commit"] == SOURCE and terminal["terminal_tag"] == TAG,
        "terminal manifest identity changed",
    )
    tag_present = bool(git("tag", "--list", TAG))
    if tag_present:
        require(git("cat-file", "-t", TAG) == "tag", "terminal tag is not annotated")
        require(
            git("rev-parse", TAG + "^{}") == git("rev-parse", "HEAD"),
            "terminal tag must identify this canonical checkout",
        )
        require(
            git("for-each-ref", "--format=%(contents)", "refs/tags/" + TAG) == TAG_MESSAGE,
            "terminal tag message changed",
        )
    result.update(
        terminal_manifest_entries=count,
        terminal_tag_present=tag_present,
        decision="TERMINATE_CURRENT_CROSS_MODAL_SYNERGY_LINE",
        exact_next_action="NO_FURTHER_EXPERIMENTS_ON_THIS_LINE",
        terminal_rerun_read_only=True,
    )
    return result


COMMANDS = {
    "verify-artifacts": verify_source_protocol,
    "verify-compute-migration-artifacts": verify_source_protocol,
    "verify-atomic-v2-artifacts": verify_source_protocol,
    "verify-program-closeout": verify_terminal,
    "atomic-v2-clean-rerun": verify_terminal,
}


def main():
    parser = argparse.ArgumentParser(description="TERMINAL ARCHIVE: read-only verification only")
    parser.add_argument("command")
    args = parser.parse_args()
    if args.command not in COMMANDS:
        parser.error(
            "NO_FURTHER_EXPERIMENTS_ON_THIS_LINE; only read-only verification is permitted"
        )
    print(json.dumps(COMMANDS[args.command](), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
