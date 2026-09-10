"""Terminal safety tests: no real processor/model is created or invoked."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys

import pytest

from capability_gate.program_closeout import COMMANDS, ROOT, SOURCE, digest, git
from capability_gate.terminal_guard import BLOCKED_FUNCTIONS, TerminalProgramStop, deny_execution


def old_experiment_commands():
    historical = git("show", SOURCE + ":src/capability_gate/cli.py")
    return sorted(
        set(re.findall(r'^\s+"([a-z][a-z0-9-]+)":', historical, flags=re.MULTILINE)) - set(COMMANDS)
    )


@pytest.mark.parametrize("command", old_experiment_commands())
def test_all_historical_non_verifier_cli_commands_reject_without_effects(command):
    path = ROOT / "artifacts/atomic_v2/final_decision.json"
    before = digest(path)
    result = subprocess.run(
        [sys.executable, "-m", "capability_gate", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "NO_FURTHER_EXPERIMENTS_ON_THIS_LINE" in result.stderr
    assert digest(path) == before


@pytest.mark.parametrize(
    "module", ["capability_gate.atomic_v2_measurement", "capability_gate.atomic_v2_protocol"]
)
def test_direct_model_module_execution_rejects_before_processor_import(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "NO_FURTHER_EXPERIMENTS_ON_THIS_LINE" in result.stderr


@pytest.mark.parametrize("module_name", list(BLOCKED_FUNCTIONS))
def test_python_api_experiment_entrypoints_are_denial_functions(module_name):
    module = importlib.import_module(module_name)
    for name in BLOCKED_FUNCTIONS[module_name]:
        if hasattr(module, name):
            function = getattr(module, name)
            assert function is deny_execution
            with pytest.raises(TerminalProgramStop):
                function()


def test_native_model_and_processor_load_are_denied_without_constructing_either():
    from capability_gate.recovery.adapters import NativeRecoveryAdapter

    assert NativeRecoveryAdapter.load is deny_execution
    assert NativeRecoveryAdapter.load_processor is deny_execution
    with pytest.raises(TerminalProgramStop):
        NativeRecoveryAdapter.load(None)
    with pytest.raises(TerminalProgramStop):
        NativeRecoveryAdapter.load_processor(None)


def test_worker_load_request_is_terminal_not_a_new_measurement_failure():
    request = {
        "schema_version": 1,
        "request_id": "terminal-denial-unit-test",
        "model_key": "qwen2_5_vl_7b",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "processor_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "image_path": None,
        "prompt": {"system": "unused", "user": "unused"},
        "candidates": ["north", "south"],
        "target": "north",
        "operation": "load_preflight",
    }
    result = subprocess.run(
        [sys.executable, str(ROOT / "workers/qwen_worker.py")],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    response = json.loads(result.stdout)
    assert response["error_class"] == "PROGRAM_TERMINATED"
    assert response["candidate_scores"] == []
    assert "NO_FURTHER_EXPERIMENTS_ON_THIS_LINE" in response["traceback"]


def test_verification_module_does_not_import_model_libraries():
    code = (
        "import sys; import capability_gate.program_closeout; "
        "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
