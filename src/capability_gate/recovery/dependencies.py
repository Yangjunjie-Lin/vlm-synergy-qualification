from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Mapping
from typing import Any

PYTHON_MINORS: dict[str, tuple[int, int]] = {
    "qwen2_5_vl_7b": (3, 11),
    "glm4_1v_9b": (3, 11),
    "phi4_multimodal_5_6b": (3, 10),
}

DEPENDENCY_SPECS: dict[str, Mapping[str, str]] = {
    "qwen2_5_vl_7b": {
        "torch": "2.6.0+cu124",
        "torchvision": "0.21.0+cu124",
        "transformers": "4.57.6",
        "accelerate": "1.14.0",
        "bitsandbytes": "0.50.2",
        "qwen-vl-utils": "0.0.14",
        "Pillow": "12.3.0",
    },
    "glm4_1v_9b": {
        "torch": "2.6.0+cu124",
        "torchvision": "0.21.0+cu124",
        "transformers": "4.57.6",
        "accelerate": "1.14.0",
        "bitsandbytes": "0.50.2",
        "Pillow": "12.3.0",
        "sentencepiece": "0.2.1",
    },
    "phi4_multimodal_5_6b": {
        "torch": "2.6.0+cu124",
        "torchvision": "0.21.0+cu124",
        "transformers": "4.48.2",
        "accelerate": "1.3.0",
        "bitsandbytes": "0.45.2",
        "backoff": "2.2.1",
        "peft": "0.13.2",
        "soundfile": "0.13.1",
        "Pillow": "11.1.0",
        "scipy": "1.15.1",
    },
}


def dependency_preflight(model_key: str) -> dict[str, Any]:
    required = DEPENDENCY_SPECS[model_key]
    python_minor = PYTHON_MINORS[model_key]
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    installed: dict[str, str | None] = {}
    for package, expected in required.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
            missing.append(package)
        installed[package] = actual
        if actual is not None and actual != expected:
            mismatched.append({"package": package, "expected": expected, "actual": actual})
    python_actual = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_ok = sys.version_info[:2] == python_minor
    return {
        "schema_version": 1,
        "model_key": model_key,
        "status": (
            "DEPENDENCY_PREFLIGHT_PASS"
            if not missing and not mismatched and python_ok
            else "DEPENDENCY_PREFLIGHT_FAIL"
        ),
        "python_expected": ".".join(map(str, python_minor)),
        "python_actual": python_actual,
        "python_compatible": python_ok,
        "required_packages": dict(required),
        "installed_packages": installed,
        "missing_dependencies": missing,
        "version_mismatches": mismatched,
        "counts_as_model_load_attempt": False,
        "counts_as_scientific_attempt": False,
    }
