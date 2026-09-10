"""Permanent execution-denial layer; historical scientific source stays byte-preserved.

This is an invocation guard, not a new model adapter or answer contract. There is
no environment-variable bypass. Only pure synthetic unit helpers remain callable.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path


class TerminalProgramStop(RuntimeError):
    pass


def deny_execution(*_args, **_kwargs):
    raise TerminalProgramStop("NO_FURTHER_EXPERIMENTS_ON_THIS_LINE: program terminally closed")


BLOCKED_FUNCTIONS = {
    "capability_gate.atomic_v2_measurement": {
        "validate_model_renderers",
        "run_contract_control",
        "main",
    },
    "capability_gate.atomic_v2_data": {
        "generate_atomic_v2_data",
        "validate_atomic_v2_data",
        "validate_joint_data_before_atomic_v2",
    },
    "capability_gate.atomic_v2_protocol": {
        "repair_glm_processor_v2",
        "validate_model_renderers",
        "run_contract_control",
        "freeze_atomic_v2_run",
        "run_atomic_model",
        "run_atomic_v2",
        "run_joint_model",
        "run_joint_after_atomic_v2",
        "adjudicate_atomic_v2",
        "analyze_joint_after_atomic_v2",
        "main",
    },
    "capability_gate.data.generator": {"generate_all"},
    "capability_gate.data.validation": {"validate_all"},
    "capability_gate.models.registry": {"freeze_registry"},
    "capability_gate.models.runner": {
        "run_engineering_smoke",
        "run_atomic_qualification",
        "run_joint_screen",
    },
    "capability_gate.recovery.smoke": {"run_adapter_recovery_smoke"},
    "capability_gate.recovery.formal": {"run_atomic_qualification_v2", "run_joint_screen_v2"},
    "capability_gate.compute_migration": {
        "prepare_compute_migration",
        "prepare_model_weights",
        "run_migration_smoke",
        "create_formal_run_lock",
        "run_migrated_atomic_model",
        "run_migrated_atomic_qwen",
        "run_migrated_atomic_glm",
        "run_migrated_joint_model",
        "run_migrated_joint_qwen",
        "run_migrated_joint_glm",
        "adjudicate_migrated_atomic",
        "analyze_migrated_joint",
    },
    "capability_gate.recovery.adapters": set(),
}


class _GuardedLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        for name in BLOCKED_FUNCTIONS[module.__name__]:
            if hasattr(module, name):
                setattr(module, name, deny_execution)
        if module.__name__ == "capability_gate.recovery.adapters":
            module.NativeRecoveryAdapter.load = deny_execution
            module.NativeRecoveryAdapter.load_processor = deny_execution


class _TerminalFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in BLOCKED_FUNCTIONS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _GuardedLoader(spec.loader)
        return spec


def install():
    original_args = getattr(sys, "orig_argv", [])
    for index, value in enumerate(original_args[:-1]):
        if value == "-m" and original_args[index + 1] in {
            "capability_gate.atomic_v2_measurement",
            "capability_gate.atomic_v2_protocol",
        }:
            deny_execution()
    if Path(sys.argv[0]).name in {"atomic_v2_measurement.py", "atomic_v2_protocol.py"}:
        deny_execution()
    if not any(isinstance(finder, _TerminalFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TerminalFinder())
