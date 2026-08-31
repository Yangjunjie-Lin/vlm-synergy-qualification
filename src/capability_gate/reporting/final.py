from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from capability_gate.artifacts import build_manifest, write_json
from capability_gate.paths import ARTIFACTS, REPORTS, ROOT


def _read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _atomic_report(adjudication: dict[str, Any]) -> str:
    lines = ["# Atomic Qualification", ""]
    lines.append(f"Cohort adjudication: **{adjudication.get('decision', 'NOT_RUN')}**")
    lines.append("")
    for model_key, model in adjudication.get("models", {}).items():
        lines.extend([f"## {model_key}", "", f"Qualification label: `{model['label']}`", ""])
        lines.append("| Task | Contract | Accuracy | One-sided 95% exact lower | Gate |")
        lines.append("|---|---:|---:|---:|---:|")
        for task, metrics in model["tasks"].items():
            for contract, gate_key in (
                ("cll", "cll_gate"),
                ("constrained_generation", "generation_gate"),
            ):
                values = metrics[contract]
                lines.append(
                    f"| {task} | {contract} | {_fmt(values['accuracy'])} | "
                    f"{_fmt(values['one_sided_95_exact_lower'])} | {metrics[gate_key]} |"
                )
        lines.extend(
            [
                "",
                (
                    f"CLL/generation exact agreement: {_fmt(model['agreement']['exact'])}; "
                    f"Cohen's kappa: {_fmt(model['agreement']['cohens_kappa'])}."
                ),
                "",
                (
                    "Confusion matrices and option-position dependence are preserved in "
                    "`artifacts/atomic/adjudication.json`."
                ),
                "",
            ]
        )
    if not adjudication.get("models"):
        lines.extend(
            ["No formal atomic model output was produced because an upstream stop rule fired.", ""]
        )
    return "\n".join(lines)


def _joint_report(analysis: dict[str, Any]) -> str:
    lines = ["# Joint Composition Screen", ""]
    lines.append(f"Decision: **{analysis.get('decision', 'NOT_RUN')}**")
    lines.append("")
    if not analysis.get("models"):
        lines.extend(
            ["Joint inference was not run because the atomic cohort gate did not pass.", ""]
        )
        return "\n".join(lines)
    for model_key, model in analysis["models"].items():
        lines.extend([f"## {model_key}", ""])
        lines.append(
            "| Contract | Joint | Image only | Text only | Question only | Joint advantage |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for contract in ("cll", "generation"):
            acc = model["accuracy"][contract]
            lines.append(
                f"| {contract} | {_fmt(acc['joint'])} | {_fmt(acc['image_only'])} | "
                f"{_fmt(acc['text_only'])} | {_fmt(acc['question_only'])} | "
                f"{_fmt(model['joint_advantage'][contract])} |"
            )
        psi = model["psi"]
        lines.extend(
            [
                "",
                (
                    f"Mean Ψ: {_fmt(psi['mean'])}; quartet-bootstrap 95% CI "
                    f"[{_fmt(psi['lower'])}, {_fmt(psi['upper'])}]. Atomic retention: "
                    f"{model['retention']['retained']}."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def build_reports() -> dict[str, Any]:
    registry = _read(ARTIFACTS / "models" / "frozen_registry.json", {})
    historical = _read(ARTIFACTS / "manifests" / "historical_freeze.json", {})
    smoke = _read(ARTIFACTS / "manifests" / "engineering_smoke.json", {})
    atomic = _read(
        ARTIFACTS / "atomic" / "adjudication.json", {"decision": smoke.get("blocking_decision")}
    )
    joint = _read(ARTIFACTS / "joint" / "analysis.json", {})
    atomic_decision = atomic.get("decision")
    decision = joint.get("decision") if atomic_decision == "ATOMIC_COHORT_GO" else atomic_decision
    decision = decision or smoke.get("blocking_decision") or "BLOCKED_BY_MODEL_ADAPTER"
    qualified_count = atomic.get("qualified_count", 0)
    if decision == "QUALIFIED_FOR_NEW_MECHANISTIC_PREREGISTRATION":
        potential = "MECHANISTIC_STUDY_FEASIBLE"
        next_action = "BUILD_HELD_OUT_MECHANISTIC_PILOT"
    elif qualified_count == 1:
        potential = "MODEL_SPECIFIC_FEASIBILITY"
        next_action = "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
    else:
        potential = "NO_FEASIBLE_COHORT"
        next_action = (
            "RESOLVE_FROZEN_COHORT_BLOCKER_WITHOUT_MODEL_SUBSTITUTION"
            if decision in {"BLOCKED_BY_COMPUTE", "BLOCKED_BY_MODEL_ADAPTER"}
            else "TERMINATE_CROSS_MODAL_SYNERGY_LINE"
        )

    REPORTS.mkdir(parents=True, exist_ok=True)
    atomic_text = _atomic_report(atomic)
    joint_text = _joint_report(joint or {"decision": atomic_decision, "models": {}})
    (REPORTS / "atomic_qualification.md").write_text(atomic_text + "\n", encoding="utf-8")
    (REPORTS / "joint_composition_screen.md").write_text(joint_text + "\n", encoding="utf-8")

    lines = [
        "# CapabilityGate Final Qualification Decision",
        "",
        "# 1. Final Decision",
        "",
        f"**{decision}**",
        "",
    ]
    lines.extend(
        [
            "# 2. Frozen Historical State",
            "",
            (
                f"SynergyTrace remains `{historical.get('historical_decision', 'BEHAVIORAL_NO_GO')}` and "
                f"`{historical.get('historical_potential', 'NO_POTENTIAL under frozen configuration')}`. "
                f"Annotated tag `{historical.get('annotated_tag', 'synergy-trace-behavioral-no-go-2026-08-31')}` "
                f"peels to `{historical.get('tag_peeled_target', 'unknown')}`. Activation patching was not "
                "executed and no mechanism claim is made."
            ),
            "",
            "# 3. Model Registry",
            "",
            "| Family | Exact revision | Parameters (B) | Dtype / quantization | Adapter smoke |",
            "|---|---|---:|---|---|",
        ]
    )
    statuses = smoke.get("model_status", {})
    for model in registry.get("models", []):
        lines.append(
            f"| {model['family']} | `{model['revision']}` | {_fmt(model['parameters_billions'])} | "
            f"{model['dtype']} / {model['quantization']} | "
            f"{statuses.get(model['key'], {}).get('status', 'NOT_RUN')} |"
        )
    lines.extend(["", "# 4. Atomic Qualification", ""])
    for model_key, result in atomic.get("models", {}).items():
        lines.append(
            f"- `{model_key}`: **{result['label']}**; failed tasks: {result['failed_tasks'] or 'none'}. "
        )
    if not atomic.get("models"):
        lines.append(
            "Formal atomic qualification was not run due to the recorded upstream blocker."
        )
    lines.extend(
        [
            "",
            "Full task metrics, exact bounds, confusion matrices, and position diagnostics: `reports/atomic_qualification.md`.",
            "",
        ]
    )
    lines.extend(["# 5. Measurement Agreement", ""])
    for model_key, result in atomic.get("models", {}).items():
        agreement = result["agreement"]
        lines.append(
            f"- `{model_key}`: exact={_fmt(agreement['exact'])}, kappa={_fmt(agreement['cohens_kappa'])}, "
            f"contract-direction conflict={agreement['contract_pass_direction_conflict']}."
        )
    if not atomic.get("models"):
        lines.append("No formal paired contract outputs were available.")
    lines.extend(
        [
            "",
            "# 6. Qualified Cohort",
            "",
            f"Robustly qualified families: **{qualified_count} / 3**.",
            "",
        ]
    )
    lines.extend(["# 7. Joint Composition", ""])
    lines.append(
        "Joint screen details are in `reports/joint_composition_screen.md`; it is absent by design when the atomic gate does not pass."
    )
    lines.extend(["", "# 8. Failure Analysis", ""])
    if smoke.get("blocking_decision"):
        lines.append(
            f"The present failure is an engineering blocker: `{smoke['blocking_decision']}`. It is not a model-capability or mechanism result."
        )
    else:
        categories = {
            "ATOMIC_VISUAL_FAIL": "visual perception",
            "ATOMIC_TEXT_FAIL": "text uptake",
            "ATOMIC_DIRECTION_FAIL": "direction reversal",
            "ATOMIC_BINDING_FAIL": "bridge binding",
            "MEASUREMENT_CONTRACT_DEPENDENT": "measurement dependence",
        }
        for model_key, result in atomic.get("models", {}).items():
            if result["label"] != "ATOMICALLY_QUALIFIED":
                lines.append(f"- `{model_key}`: {categories.get(result['label'], 'composition')}")
        if decision == "JOINT_COMPOSITION_NO_GO":
            lines.append(
                "Atomic capability existed in at least two families, but the composition gate failed."
            )
    lines.extend(["", "# 9. Q1 Research Potential", "", f"**{potential}**", ""])
    lines.extend(["# 10. Exact Next Action", "", f"**{next_action}**", ""])
    lines.append(
        "Qualification behavior is not mechanism evidence; no activation patching was run."
    )
    final_text = "\n".join(lines) + "\n"
    final_path = REPORTS / "final_qualification_decision.md"
    final_path.write_text(final_text, encoding="utf-8")
    summary = {
        "decision": decision,
        "q1_research_potential": potential,
        "exact_next_action": next_action,
        "activation_patching_executed": False,
        "mechanism_claims": [],
    }
    write_json(ARTIFACTS / "manifests" / "final_decision.json", summary)
    report_paths = [
        REPORTS / "atomic_qualification.md",
        REPORTS / "joint_composition_screen.md",
        final_path,
        ARTIFACTS / "manifests" / "final_decision.json",
    ]
    write_json(
        ARTIFACTS / "manifests" / "report_manifest.json",
        build_manifest(ROOT, report_paths, "qualification_reports"),
    )
    return summary
