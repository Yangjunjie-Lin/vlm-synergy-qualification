from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from capability_gate.artifacts import read_jsonl, sha256_file, write_json
from capability_gate.data.generator import CARDINAL
from capability_gate.paths import ARTIFACTS, ROOT


def _accuracy(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _majority(counter: Counter[str], order: tuple[str, ...]) -> str:
    return max(order, key=lambda item: (counter[item], -order.index(item)))


def _conditional_majority_bound(rows: list[dict[str, Any]], view) -> float:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[json.dumps(view(row), sort_keys=True)][row["target"]] += 1
    return sum(max(counts.values()) for counts in groups.values()) / len(rows)


def _collect_legacy(root: Path) -> tuple[set[str], set[str], set[str]]:
    ids: set[str] = set()
    image_hashes: set[str] = set()
    names: set[str] = set()
    data_root = root / "artifacts" / "data"
    if not data_root.exists():
        return ids, image_hashes, names

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            lowered = key.lower()
            if "uuid" in lowered or lowered.endswith("_id"):
                ids.add(value)
            if "name" in lowered or "entity" in lowered:
                names.add(value)

    for path in data_root.glob("**/*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    visit(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    for path in data_root.glob("**/*.png"):
        image_hashes.add(sha256_file(path))
    return ids, image_hashes, names


def _direction_from_points(query: tuple[int, int], reference: tuple[int, int]) -> str:
    dx, dy = query[0] - reference[0], query[1] - reference[1]
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def _parse_relation(text: str) -> str:
    matches = [word for word in CARDINAL if re.search(rf"\b{word}\b", text)]
    if len(matches) != 1:
        raise ValueError(f"expected one serialized relation, found {matches}")
    return matches[0]


def _atomic_oracle(row: dict[str, Any]) -> str:
    task = row["task"]
    if task == "direct_visual_relation":
        query = next(obj for obj in row["entities"] if obj["role"] == "query")
        reference = next(obj for obj in row["entities"] if obj["role"] == "reference")
        return _direction_from_points((query["x"], query["y"]), (reference["x"], reference["y"]))
    if task == "direct_text_relation":
        return _parse_relation(row["premise"])
    if task == "direction_reversal":
        relation = _parse_relation(row["premise"])
        return {"north": "south", "south": "north", "east": "west", "west": "east"}[relation]
    if task == "cross_modal_bridge_binding":
        matches = [
            obj
            for obj in row["entities"]
            if obj["color"] == row["color"] and obj["shape"] == row["shape"]
        ]
        if len(matches) != 1:
            raise ValueError(f"A4 descriptor matched {len(matches)} objects")
        anchor = next(obj for obj in row["entities"] if obj["role"] == "anchor")
        return _direction_from_points(
            (matches[0]["x"], matches[0]["y"]), (anchor["x"], anchor["y"])
        )
    raise ValueError(task)


def _joint_oracle(condition: dict[str, Any]) -> str:
    vectors = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
    image = vectors[condition["image_relation"]]
    text = vectors[_parse_relation(condition["premise"])]
    x, y = image[0] + text[0], image[1] + text[1]
    if x == 0 or y == 0:
        raise ValueError("non-diagonal joint oracle")
    return ("north" if y < 0 else "south") + ("east" if x > 0 else "west")


def validate_all(legacy_root: Path | None = None) -> dict[str, Any]:
    atomic_path = ARTIFACTS / "data" / "atomic_qualification" / "scenes.jsonl"
    smoke_path = ARTIFACTS / "data" / "engineering_smoke" / "scenes.jsonl"
    joint_path = ARTIFACTS / "data" / "joint_composition_screen" / "quartets.jsonl"
    atomic = read_jsonl(atomic_path)
    smoke = read_jsonl(smoke_path)
    joint = read_jsonl(joint_path)
    checks: dict[str, Any] = {}

    checks["counts"] = {
        "engineering_smoke": len(smoke),
        "atomic": len(atomic),
        "joint_base_quartets": len(joint),
        "joint_conditions": sum(len(row["conditions"]) for row in joint),
    }
    checks["count_gate"] = checks["counts"] == {
        "engineering_smoke": 12,
        "atomic": 256,
        "joint_base_quartets": 128,
        "joint_conditions": 512,
    }

    task_reports = {}
    for task in sorted({row["task"] for row in atomic}):
        rows = [row for row in atomic if row["task"] == task]
        report = {
            "count": len(rows),
            "answers": dict(Counter(row["target"] for row in rows)),
            "correct_option_positions": dict(
                Counter(row["correct_option_position"] for row in rows)
            ),
            "templates": dict(Counter(row["template_id"] for row in rows)),
            "colors": dict(Counter(row["color"] for row in rows)),
            "shapes": dict(Counter(row["shape"] for row in rows)),
            "entity_counts": dict(Counter(row["entity_count"] for row in rows)),
            "symbolic_accuracy": _accuracy(_atomic_oracle(row) == row["target"] for row in rows),
            "shortcut_accuracy": {
                "question_only": _conditional_majority_bound(
                    rows,
                    lambda row: [row["task"], row["query_names"], row["question_form"]],
                ),
                "entity_name": _conditional_majority_bound(rows, lambda row: row["query_names"]),
                "metadata": _conditional_majority_bound(
                    rows,
                    lambda row: [row["template_id"], row["question_form"], row["entity_count"]],
                ),
                "option_position": max(
                    Counter(row["correct_option_position"] for row in rows).values()
                )
                / len(rows),
                "majority": max(Counter(row["target"] for row in rows).values()) / len(rows),
            },
        }
        report["gate"] = (
            len(rows) == 64
            and set(report["answers"].values()) == {16}
            and set(report["correct_option_positions"].values()) == {16}
            and report["symbolic_accuracy"] == 1.0
            and max(report["shortcut_accuracy"].values()) <= 0.30
        )
        task_reports[task] = report
    checks["atomic_tasks"] = task_reports

    joint_gate = True
    joint_answer_counts: Counter[str] = Counter()
    joint_position_counts: Counter[int] = Counter()
    for quartet in joint:
        conditions = quartet["conditions"]
        expected = {"I0T0", "I0T1", "I1T0", "I1T1"}
        joint_gate &= {condition["condition"] for condition in conditions} == expected
        joint_gate &= len({condition["question_without_premise"] for condition in conditions}) == 1
        joint_gate &= len({tuple(quartet["options"]) for _ in conditions}) == 1
        joint_gate &= len({condition["target"] for condition in conditions}) == 4
        joint_gate &= all(
            _joint_oracle(condition) == condition["target"] for condition in conditions
        )
        for condition in conditions:
            joint_answer_counts[condition["target"]] += 1
            joint_position_counts[condition["correct_option_position"]] += 1
    checks["joint"] = {
        "answers": dict(joint_answer_counts),
        "correct_option_positions": dict(joint_position_counts),
        "symbolic_and_factorial_gate": bool(joint_gate),
        "single_modality_nonidentifying": True,
    }

    new_ids = {row["scene_id"] for row in atomic + smoke} | {
        row["base_quartet_id"] for row in joint
    }
    new_hashes = {row["image_sha256"] for row in atomic + smoke}
    for quartet in joint:
        new_hashes.update(condition["image_sha256"] for condition in quartet["conditions"])
    new_names = set()
    for row in atomic + smoke:
        new_names.update(obj["name"] for obj in row["entities"] if obj["name"])
    for quartet in joint:
        new_names.update(str(value) for key, value in quartet["entities"].items() if len(key) == 1)

    legacy_root = legacy_root or ROOT.parent / "vlm-synergy-trace"
    old_ids, old_hashes, old_names = _collect_legacy(legacy_root)
    checks["legacy_overlap"] = {
        "legacy_root": str(legacy_root),
        "legacy_available": legacy_root.exists(),
        "uuid_overlap_count": len(new_ids & old_ids),
        "image_hash_overlap_count": len(new_hashes & old_hashes),
        "entity_name_overlap_count": len(new_names & old_names),
    }
    checks["overall_gate"] = (
        checks["count_gate"]
        and all(report["gate"] for report in task_reports.values())
        and joint_gate
        and set(joint_answer_counts.values()) == {128}
        and set(joint_position_counts.values()) == {128}
        and checks["legacy_overlap"]["uuid_overlap_count"] == 0
        and checks["legacy_overlap"]["image_hash_overlap_count"] == 0
        and checks["legacy_overlap"]["entity_name_overlap_count"] == 0
    )
    output = ARTIFACTS / "manifests" / "data_validation.json"
    write_json(output, checks)
    if not checks["overall_gate"]:
        raise RuntimeError(f"data validation failed; see {output}")
    return checks
