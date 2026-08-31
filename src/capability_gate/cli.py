from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from capability_gate.data import generate_all, validate_all
from capability_gate.models.registry import freeze_registry
from capability_gate.models.runner import (
    run_atomic_qualification,
    run_engineering_smoke,
    run_joint_screen,
)
from capability_gate.provenance import verify_historical_freeze
from capability_gate.recovery.adjudication import (
    adjudicate_atomic_v2,
    adjudicate_engineering_recovery,
    analyze_joint_v2,
)
from capability_gate.recovery.diagnostics import diagnose_adapters
from capability_gate.recovery.environments import verify_model_environments
from capability_gate.recovery.formal import run_atomic_qualification_v2, run_joint_screen_v2
from capability_gate.recovery.historical import verify_historical_block
from capability_gate.recovery.reporting import build_recovery_report, verify_recovery_artifacts
from capability_gate.recovery.smoke import run_adapter_recovery_smoke
from capability_gate.reporting import build_reports
from capability_gate.statistics.adjudication import adjudicate_atomic
from capability_gate.statistics.joint import analyze_joint
from capability_gate.verification import verify_artifacts


def _freeze_models() -> dict[str, Any]:
    historical = verify_historical_freeze()
    registry = freeze_registry()
    return {"historical_freeze": historical, "model_registry": registry}


COMMANDS: dict[str, Callable[[], Any]] = {
    "freeze-models": _freeze_models,
    "generate-data": generate_all,
    "validate-data": validate_all,
    "run-engineering-smoke": run_engineering_smoke,
    "run-atomic-qualification": run_atomic_qualification,
    "adjudicate-atomic": adjudicate_atomic,
    "run-joint-screen": run_joint_screen,
    "analyze-joint": analyze_joint,
    "build-report": build_reports,
    "verify-artifacts": verify_artifacts,
    "verify-historical-block": verify_historical_block,
    "diagnose-adapters": diagnose_adapters,
    "verify-model-environments": verify_model_environments,
    "run-adapter-recovery-smoke": run_adapter_recovery_smoke,
    "adjudicate-engineering-recovery": adjudicate_engineering_recovery,
    "run-atomic-qualification-v2": run_atomic_qualification_v2,
    "adjudicate-atomic-v2": adjudicate_atomic_v2,
    "run-joint-screen-v2": run_joint_screen_v2,
    "analyze-joint-v2": analyze_joint_v2,
    "build-recovery-report": build_recovery_report,
    "verify-recovery-artifacts": verify_recovery_artifacts,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="capability-gate")
    parser.add_argument("command", choices=COMMANDS)
    args = parser.parse_args()
    result = COMMANDS[args.command]()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
