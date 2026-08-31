from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def ensure_layout() -> None:
    for path in (
        ARTIFACTS / "data" / "engineering_smoke" / "images",
        ARTIFACTS / "data" / "atomic_qualification" / "images",
        ARTIFACTS / "data" / "joint_composition_screen" / "images",
        ARTIFACTS / "atomic",
        ARTIFACTS / "joint",
        ARTIFACTS / "manifests",
        ARTIFACTS / "models",
        REPORTS,
    ):
        path.mkdir(parents=True, exist_ok=True)
