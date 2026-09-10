"""Pre-outcome construct repair; never imports or consumes model predictions.

The generator is a new implementation, not a filtered or renamed v1 dataset.
Displayed names refer to distinct identities; repetition of a name in the query
and its image label is not a collision. The binding task remains descriptor
binding to UNLABELED objects, with collision-safe internal identity aliases.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from capability_gate.artifacts import canonical_json, read_jsonl, sha256_file
from capability_gate.data.generator import (
    CARDINAL,
    COLORS,
    DIAGONAL,
    QUESTION_FORMS,
    REVERSE,
    RGB,
    SHAPES,
    TEXT_FORMS,
    VECTORS,
    _render,
)
from capability_gate.paths import ROOT

TASKS = (
    "direct_visual_relation",
    "direct_text_relation",
    "direction_reversal",
    "cross_modal_bridge_binding",
)
NAMESPACE = uuid.UUID("d510287f-fb80-4ee0-9234-c8215cf6f620")
ALIAS_SECRET = b"capability-gate-atomic-v2/pre-outcome/2026-09-10/alias/78210693"
SEEDS = {
    "atomic_scene_order": 92618371,
    "atomic_template_assignment": 12867043,
    "atomic_option_permutation": 68201439,
    "atomic_image": 59318721,
    "atomic_descriptor_assignment": 73194625,
    "joint_scene_order": 41297683,
    "joint_template_assignment": 18294637,
    "joint_option_permutation": 97163825,
    "joint_image": 38152749,
    "grouped_cross_validation": 24591837,
}
BRIDGE_FORMS = (
    ("The bridge entity named {name} is the {color} {shape} shown in the image. "
     "Where is that bridge entity relative to the black anchor?"),
    ("In the image, bind the name {name} to its {color} {shape}. "
     "Which direction is it from the black anchor?"),
    ("Locate {name}, the {color} {shape}, in the image. "
     "Relative to the black anchor, where is it?"),
    ("The text refers to the {color} {shape} as {name}. "
     "Choose its direction from the black anchor."),
)


class DataValidationError(RuntimeError):
    """A pre-inference hard gate, never a model-capability conclusion."""


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _uid(*parts: object) -> str:
    return str(uuid.uuid5(NAMESPACE, "/".join(map(str, parts))))


def _seed(kind: str, *parts: object) -> int:
    value = f"{SEEDS[kind]}/" + "/".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def _alias(scene_id: str, role: str, index: int, retry: int = 0) -> str:
    # No index/role/task is exposed literally: HMAC output is alphabet-only.
    message = f"{scene_id}/{role}/{index}/{retry}".encode()
    digest = hmac.new(ALIAS_SECRET, message, hashlib.sha256).digest()
    alias = "".join(chr(97 + x % 26) for x in digest[:10])
    if any(direction in alias for direction in CARDINAL):
        return _alias(scene_id, role, index, retry + 1)
    return alias


def _aliases(scene_id: str, roles: list[str]) -> list[str]:
    # A collision regenerates the entire alias assignment before rendering.
    for retry in range(256):
        aliases = [_alias(scene_id, role, i, retry) for i, role in enumerate(roles)]
        if len(aliases) == len(set(aliases)):
            return aliases
    raise DataValidationError("ATOMIC_V2_DATA_INVALID: cannot allocate unique identities")


def _assert_pre_outcome(root: Path) -> None:
    formal = root / "artifacts/atomic_v2/formal"
    if formal.exists() and any(path.stat().st_size for path in formal.rglob("*.jsonl")):
        raise DataValidationError("formal v2 output exists: construct regeneration prohibited")
    if (root / "artifacts/atomic_v2/formal_run_lock.json").exists():
        raise DataValidationError("formal run already frozen: construct regeneration prohibited")


def _obj(name: str, role: str, xy: tuple[int, int], color: str, shape: str,
         *, displayed: bool = True) -> dict[str, Any]:
    return {"name": name, "display_name": name if displayed else "", "role": role,
            "x": xy[0], "y": xy[1], "color": color, "shape": shape, "drawn": True}


def _position(origin: tuple[int, int], direction: str, distance: int) -> tuple[int, int]:
    dx, dy = VECTORS[direction]
    return origin[0] + distance * dx, origin[1] + distance * dy


def _render_entities(path: Path, entities: list[dict[str, Any]], title: str) -> None:
    _render(path, [dict(obj, name=obj["display_name"]) for obj in entities if obj["drawn"]],
            title=title)


def _atomic_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        templates = [i % 4 for i in range(16)]
        random.Random(_seed("atomic_template_assignment", task)).shuffle(templates)
        descriptors = [(color, shape) for color in COLORS for shape in SHAPES]
        random.Random(_seed("atomic_descriptor_assignment", task)).shuffle(descriptors)
        base_options = list(CARDINAL)
        random.Random(_seed("atomic_option_permutation", task)).shuffle(base_options)
        option_rotations = [i % 4 for i in range(16)]
        random.Random(_seed("atomic_option_permutation", task, "rotations")).shuffle(
            option_rotations)
        for block in range(16):
            template = templates[block]
            target_color, target_shape = descriptors[block]
            rotation = option_rotations[block]
            options = base_options[rotation:] + base_options[:rotation]
            option_seed = _seed("atomic_option_permutation", task, block)
            for answer in CARDINAL:
                scene_id = _uid("atomic", task, block, answer)
                image_seed = _seed("atomic_image", scene_id)
                rng = random.Random(image_seed)
                center = (256 + rng.randint(-5, 5), 256 + rng.randint(-5, 5))
                distance = rng.randint(140, 150)
                extras = block % 4
                roles = (["bridge", "anchor", "candidate", "candidate", "candidate"]
                         if task == TASKS[3] else ["query", "reference"])
                roles += ["distractor"] * extras
                names = _aliases(scene_id, roles)
                query_name, reference_name = names[:2]
                entities: list[dict[str, Any]] = []
                premise = ""
                query = QUESTION_FORMS[template].format(a=query_name, b=reference_name)
                question = query
                requires_image = task in {TASKS[0], TASKS[3]}
                if task == TASKS[0]:
                    entities = [
                        _obj(query_name, "query", _position(center, answer, distance),
                             target_color, target_shape),
                        _obj(reference_name, "reference", center, "black", "circle"),
                    ]
                elif task in {TASKS[1], TASKS[2]}:
                    source, destination = ((query_name, reference_name) if task == TASKS[1]
                                           else (reference_name, query_name))
                    relation = answer if task == TASKS[1] else REVERSE[answer]
                    premise = TEXT_FORMS[template].format(b=source, c=destination,
                                                          relation=relation)
                    question = ("The image is unrelated decoration; use the text statement.\n"
                                + premise + "\n" + query)
                    entities = [dict(_obj(query_name, "query", (0, 0), target_color,
                                          target_shape), drawn=False, display_name=""),
                                dict(_obj(reference_name, "reference", (0, 0), "black",
                                          "circle"), drawn=False, display_name="")]
                else:
                    entities = [_obj(reference_name, "anchor", center, "black", "circle",
                                     displayed=False)]
                    other_colors = [color for color in COLORS if color != target_color]
                    rng.shuffle(other_colors)
                    candidate_index = 0
                    for direction in CARDINAL:
                        if direction == answer:
                            name, role, color, shape = (query_name, "bridge", target_color,
                                                        target_shape)
                        else:
                            name = names[candidate_index + 2]
                            color = other_colors[candidate_index]
                            shape = SHAPES[rng.randrange(4)]
                            role = "candidate"
                            candidate_index += 1
                        entities.append(_obj(name, role, _position(center, direction, distance),
                                             color, shape, displayed=False))
                    question = BRIDGE_FORMS[template].format(name=query_name, color=target_color,
                                                            shape=target_shape)
                    # Evidence-free question is not the descriptor-bearing full prompt.
                    query = "Where is " + query_name + " relative to the black anchor?"
                corner_slots = [(82, 96), (430, 96), (82, 422)]
                rng.shuffle(corner_slots)
                for i in range(extras):
                    entity = _obj(names[len(roles) - extras + i], "distractor", corner_slots[i],
                                  "violet", SHAPES[(block + i) % 4],
                                  displayed=task != TASKS[3])
                    if not requires_image:
                        entity.update(drawn=False, display_name="")
                    entities.append(entity)
                image_path = root / "artifacts/data_v2/atomic_qualification/images" / (
                    _uid("atomic-image", scene_id) + ".png")
                if requires_image:
                    _render_entities(image_path, entities,
                                     "Bridge-binding scene" if task == TASKS[3]
                                     else "Spatial relation scene")
                else:
                    # A fresh file exists for provenance; it is never sent to the model.
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    provenance = PngInfo()
                    provenance.add_text("scene_uuid", scene_id)
                    provenance.add_text("image_seed", str(image_seed))
                    Image.new("RGB", (512, 512), (249, 248, 243)).save(
                        image_path, pnginfo=provenance)
                rows.append({
                    "schema_version": 2, "protocol": "capability_gate_atomic_v2",
                    "scene_id": scene_id, "split": "atomic_qualification", "task": task,
                    "image_path": image_path.relative_to(root).as_posix(),
                    "image_sha256": sha256_file(image_path), "requires_image": requires_image,
                    "question": question, "question_without_premise": query,
                    "premise": premise, "options": options, "target": answer,
                    "correct_option_position": options.index(answer), "symbolic_answer": answer,
                    "template_id": template, "question_form": template,
                    "nuisance_block_id": f"{task}:{block}", "entities": entities,
                    "query_names": [query_name, reference_name], "query_name": query_name,
                    "reference_name": reference_name, "color": target_color,
                    "shape": target_shape, "entity_count": len(entities),
                    "answer_surface_token_length": 1, "image_seed": image_seed,
                    "option_permutation_seed": option_seed,
                    "template_assignment_seed": _seed("atomic_template_assignment", task),
                })
    random.Random(SEEDS["atomic_scene_order"]).shuffle(rows)
    for index, row in enumerate(rows):
        row["scene_index"] = index + 1
    return rows


def _manifest(root: Path, paths: list[Path], kind: str) -> dict[str, Any]:
    return {"schema_version": 1, "kind": kind, "protocol": "capability_gate_atomic_v2",
            "files": [{"path": path.relative_to(root).as_posix(),
                       "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                      for path in sorted(set(paths))]}


def generate_atomic_v2_data(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    path = root / "artifacts/data_v2/atomic_qualification/scenes.jsonl"
    manifest_path = root / "artifacts/atomic_v2/manifests/data_manifest.json"
    if path.exists() or manifest_path.exists():
        if not path.exists() or not manifest_path.exists():
            raise DataValidationError("partial data artifacts preserved; automatic retry prohibited")
        return validate_atomic_v2_data(root)
    _assert_pre_outcome(root)
    journal = root / "artifacts/atomic_v2/data/generation_attempt.json"
    if journal.exists():
        raise DataValidationError("existing generation attempt preserved; no automatic regeneration")
    _json(journal, {"schema_version": 1, "batch": 1, "seeds": SEEDS,
                    "namespace_uuid": str(NAMESPACE), "automatic_regenerations": 0,
                    "max_explicit_preoutcome_regenerations": 1})
    rows = _atomic_rows(root)
    _jsonl(path, rows)
    images = [root / row["image_path"] for row in rows]
    _json(manifest_path, _manifest(root, [path], "atomic_v2_data"))
    _json(root / "artifacts/atomic_v2/manifests/image_manifest.json",
          _manifest(root, images, "atomic_v2_images"))
    return validate_atomic_v2_data(root)


def _direction(query: dict[str, Any], reference: dict[str, Any]) -> str:
    dx, dy = query["x"] - reference["x"], query["y"] - reference["y"]
    if (dx == 0) == (dy == 0):
        raise ValueError("not a distinct cardinal relation")
    return ("east" if dx > 0 else "west") if dx else ("south" if dy > 0 else "north")


def _premise_fact(premise: str) -> tuple[str, str, str]:
    alias = r"([A-Za-z]+)"
    direction = r"(north|south|east|west)"
    patterns = [
        (rf"Starting at {alias}, {alias} lies toward the {direction}\.", (1, 0, 2)),
        (rf"Relative to {alias}, {alias} lies to the {direction}\.", (1, 0, 2)),
        (rf"From {alias}, move {direction} to reach {alias}\.", (2, 0, 1)),
        (rf"The position of {alias} is {direction} with respect to {alias}\.", (0, 2, 1)),
    ]
    for pattern, order in patterns:
        match = re.fullmatch(pattern, premise)
        if match:
            values = match.groups()
            return tuple(values[i] for i in order)  # type: ignore[return-value]
    raise ValueError("premise does not match a frozen text template")


def symbolic_atomic_oracle(row: dict[str, Any]) -> str:
    """Derive answer from coordinates / parsed fact, not any serialized answer field."""
    entities = row["entities"]
    by_name = {obj["name"]: obj for obj in entities}
    query, reference = row["query_names"]
    if query == reference:
        raise ValueError("self-relation")
    if row["task"] in {TASKS[1], TASKS[2]}:
        source, destination, relation = _premise_fact(row["premise"])
        if source == destination:
            raise ValueError("self-relation in premise")
        if (query, reference) == (source, destination):
            return relation
        if (reference, query) == (source, destination):
            return REVERSE[relation]
        raise ValueError("question not supported by the premise identities")
    if row["task"] == TASKS[3]:
        matches = [obj for obj in entities if obj["color"] == row["color"]
                   and obj["shape"] == row["shape"] and obj["drawn"]]
        if len(matches) != 1 or matches[0]["name"] != query:
            raise ValueError("bridge descriptor not unique or not the query identity")
        if any(obj["display_name"] for obj in entities):
            raise ValueError("bridge construct changed into rendered alias matching")
    return _direction(by_name[query], by_name[reference])


def _row_failures(row: dict[str, Any], root: Path | None = None) -> list[str]:
    failures = []
    entities = row["entities"]
    names = [obj["name"] for obj in entities]
    if len(names) != len(set(names)):
        failures.append("within_scene_alias_collision")
    if any(not re.fullmatch(r"[a-z]{8,12}", name) or
           any(direction in name for direction in CARDINAL) for name in names):
        failures.append("illegal_alias")
    distractors = [obj["name"] for obj in entities if obj["role"] == "distractor"]
    if len(distractors) != len(set(distractors)):
        failures.append("duplicate_distractor")
    query, reference = row["query_names"]
    if query == reference or row.get("query_name") == row.get("reference_name"):
        failures += ["query_reference_equality", "self_relation"]
    if row.get("query_name") != query or row.get("reference_name") != reference:
        failures.append("query_identity_inconsistent")
    if query not in names or reference not in names:
        failures.append("missing_query_or_reference")
    displayed = [obj["display_name"] for obj in entities if obj["display_name"]]
    if len(displayed) != len(set(displayed)):
        failures.append("duplicate_display_label")
    if row["task"] == TASKS[0] and any(
        not obj["drawn"] or obj["display_name"] != obj["name"]
        for obj in entities if obj["name"] in {query, reference}
    ):
        failures.append("query_or_reference_not_drawn")
    visual = row["task"] in {TASKS[0], TASKS[3]}
    if row["requires_image"] != visual:
        failures.append("image_presence_contract")
    if not visual and any(obj["drawn"] for obj in entities):
        failures.append("text_task_contains_evidence_image")
    drawn = [obj for obj in entities if obj["drawn"]]
    for i, first in enumerate(drawn):
        for second in drawn[i + 1:]:
            if math.hypot(first["x"] - second["x"], first["y"] - second["y"]) <= 95:
                failures.append("object_overlap_or_occlusion")
    try:
        oracle = symbolic_atomic_oracle(row)
        if oracle != row["target"] or oracle != row["symbolic_answer"]:
            failures.append("symbolic_target_mismatch")
        if row["task"] in {TASKS[1], TASKS[2]}:
            source, destination, _relation = _premise_fact(row["premise"])
            expected_pair = ((query, reference) if row["task"] == TASKS[1]
                             else (reference, query))
            if (source, destination) != expected_pair:
                failures.append("text_fact_direction_contract")
            if row["task"] == TASKS[2] and re.search(rf"\b{oracle}\b", row["premise"]):
                failures.append("inverse_target_word_leakage")
            if source == destination:
                failures.append("self_relation")
        expected_question = (
            BRIDGE_FORMS[row["template_id"]].format(name=query, color=row["color"],
                                                    shape=row["shape"])
            if row["task"] == TASKS[3]
            else QUESTION_FORMS[row["template_id"]].format(a=query, b=reference)
        )
        if row["task"] in {TASKS[1], TASKS[2]}:
            expected_question = ("The image is unrelated decoration; use the text statement.\n"
                                 + row["premise"] + "\n" + expected_question)
        if row["question"] != expected_question:
            failures.append("question_identity_or_template_mismatch")
    except (KeyError, ValueError, StopIteration):
        failures.append("symbolic_oracle_error")
    if sorted(row["options"]) != sorted(CARDINAL):
        failures.append("answer_candidates_changed")
    elif row["options"].index(row["target"]) != row["correct_option_position"]:
        failures.append("option_position_mismatch")
    if root is not None:
        image_path = root / row["image_path"]
        if not image_path.is_file() or sha256_file(image_path) != row["image_sha256"]:
            failures.append("image_hash_mismatch")
        elif visual:
            with Image.open(image_path) as image:
                if image.size != (512, 512):
                    failures.append("image_dimensions")
                elif any(image.getpixel((obj["x"], obj["y"])) != RGB[obj["color"]]
                         for obj in drawn):
                    failures.append("image_pixels_disagree_with_geometry")
    return sorted(set(failures))


def grouped_shortcut_probes(rows: list[dict[str, Any]], *, joint: bool = False) -> dict[str, Any]:
    """Actual out-of-fold predictions: train-only categorical conditional majority.

    Feature allowlists exclude legitimate evidence, answer fields, coordinates,
    seeds and target-bearing role annotations. Unseen categories use the training
    marginal majority. This is a finite family of probes, not proof against all
    possible learned shortcuts. Correct option position is tested diagnostically
    in addition to the model-visible option-order predictor.
    """
    labels = DIAGONAL if joint else CARDINAL
    group_key = "base_quartet_id" if joint else "nuisance_block_id"
    groups = sorted({row[group_key] for row in rows})
    random.Random(SEEDS["grouped_cross_validation"]).shuffle(groups)
    group_folds = {group: index % 4 for index, group in enumerate(groups)}
    if not joint:
        # Nuisance-only stratification: every fold includes one of each option
        # rotation, without inspecting labels. This makes option-order balance
        # hold in training and testing, not merely in the pooled table.
        order_groups: dict[str, list[str]] = defaultdict(list)
        for group in groups:
            example = next(row for row in rows if row[group_key] == group)
            order_groups[canonical_json(example["options"])].append(group)
        if len(order_groups) == 4 and all(len(value) == 4 for value in order_groups.values()):
            group_folds = {group: fold for value in order_groups.values()
                           for fold, group in enumerate(value)}
    features: dict[str, Callable[[dict[str, Any]], Any]] = {
        "scene_index": lambda row: row["scene_index"],
        "entity_name": lambda row: row["query_names"],
        "question_text": lambda row: row["question_without_premise"],
        "template": lambda row: row["template_id"],
        "option_position": lambda row: row["correct_option_position"],
        "option_order": lambda row: row["options"],
        "image_filename": lambda row: Path(row["image_path"]).name,
        "metadata": lambda row: [row["task"], row["template_id"], row["entity_count"],
                                  row["color"], row["shape"]],
    }
    if not joint and rows and rows[0]["task"] == TASKS[3]:
        features["descriptor_aware_text_only"] = lambda row: [
            row["template_id"], row["color"], row["shape"]]
        features["descriptor_color"] = lambda row: row["color"]
        features["descriptor_shape"] = lambda row: row["shape"]
    output = {}
    learned = {"scene_index", "entity_name", "question_text", "image_filename", "metadata"}
    for name, feature in features.items():
        predictions = []
        for fold in range(4):
            training = [row for row in rows if group_folds[row[group_key]] != fold]
            testing = [row for row in rows if group_folds[row[group_key]] == fold]
            conditional: dict[str, Counter[str]] = defaultdict(Counter)
            marginal = Counter(row["target"] for row in training)
            for row in training:
                conditional[canonical_json(feature(row))][row["target"]] += 1
            learned_predictions = None
            if name in learned:
                learned_predictions = _ridge_predict(training, testing, feature, name, labels)
            for test_index, row in enumerate(testing):
                counts = conditional.get(canonical_json(feature(row)), marginal)
                prediction = max(labels, key=lambda label: (counts[label], -labels.index(label)))
                if learned_predictions is not None:
                    prediction = learned_predictions[test_index]
                predictions.append({"scene_id": row.get("scene_id", row[group_key]),
                                    "fold": fold, "target": row["target"],
                                    "prediction": prediction})
        accuracy = sum(item["target"] == item["prediction"] for item in predictions) / len(rows)
        output[name] = {"cross_validated_accuracy": accuracy, "gate": accuracy <= 0.30,
                        "prediction_count": len(predictions), "predictions": predictions,
                        "classifier": "unconditional_train_only_ridge" if name in learned
                        else "train_only_categorical_conditional_majority"}
    return {"classifier": "frozen_categorical_and_unconditional_train_only_ridge",
            "folds": 4, "group_key": group_key, "seed": SEEDS["grouped_cross_validation"],
            "held_out_groups": [[group for group in groups if group_folds[group] == fold]
                                for fold in range(4)],
            "limitation": "Finite categorical and ridge probes do not exclude every "
            "possible nonlinear shortcut.",
            "probes": output, "overall_gate": all(probe["gate"] for probe in output.values())}


def _probe_vector(value: Any, name: str) -> np.ndarray:
    vector = np.zeros(256, dtype=float)
    if name == "scene_index":
        index = int(value)
        vector[0] = index / 256
        vector[1 + min(15, (index - 1) // 16)] = 1
        offset = 17
        for modulus in (2, 3, 4, 5, 7, 8, 16):
            vector[offset + index % modulus] = 1
            offset += modulus
    else:
        serialized = canonical_json(value).lower()
        for length in (1, 2, 3):
            for index in range(len(serialized) - length + 1):
                gram = serialized[index:index + length].encode()
                digest = hashlib.sha256(gram).digest()
                slot = int.from_bytes(digest[:2], "big") % 255
                vector[slot] += 1 if digest[2] % 2 else -1
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    vector[-1] = 1  # intercept: the training prior is available to ridge.
    return vector


def _ridge_predict(training: list[dict[str, Any]], testing: list[dict[str, Any]],
                   feature: Callable, name: str, labels: tuple[str, ...]) -> list[str]:
    x_train = np.stack([_probe_vector(feature(row), name) for row in training])
    x_test = np.stack([_probe_vector(feature(row), name) for row in testing])
    y_train = np.zeros((len(training), len(labels)))
    for index, row in enumerate(training):
        y_train[index, labels.index(row["target"])] = 1
    # Fixed regularization, no tuning on any held-out labels.
    dual = np.linalg.solve(x_train @ x_train.T + np.eye(len(training)), y_train)
    scores = x_test @ x_train.T @ dual
    return [labels[index] for index in np.argmax(scores, axis=1)]


def _manifest_errors(root: Path, manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        return ["missing_manifest:" + manifest_path.name]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ["manifest_mismatch:" + item["path"] for item in manifest["files"]
            if not (root / item["path"]).is_file()
            or sha256_file(root / item["path"]) != item["sha256"]]


def validate_atomic_v2_data(root: Path = ROOT, *, raise_on_invalid: bool = True) -> dict[str, Any]:
    root = Path(root)
    path = root / "artifacts/data_v2/atomic_qualification/scenes.jsonl"
    if not path.exists():
        raise DataValidationError("ATOMIC_V2_DATA_INVALID: v2 data missing")
    rows = read_jsonl(path)
    old_path = root / "artifacts/data/atomic_qualification/scenes.jsonl"
    old = read_jsonl(old_path) if old_path.exists() else []
    failures: dict[str, list[str]] = {}
    for row in rows:
        try:
            errors = _row_failures(row, root)
        except (KeyError, ValueError, TypeError) as exc:
            errors = ["malformed_row:" + type(exc).__name__]
        if row.get("schema_version") != 2 or row.get("protocol") != "capability_gate_atomic_v2":
            errors.append("v1_or_foreign_row_forbidden")
        if errors:
            failures[row.get("scene_id", "missing")] = errors
    counts = Counter(error for errors in failures.values() for error in errors)
    ids = [row["scene_id"] for row in rows]
    overlap = set(ids) & {row["scene_id"] for row in old}
    task_reports = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        answers = Counter(row["target"] for row in selected)
        positions = Counter(row["correct_option_position"] for row in selected)
        oracle_correct = 0
        for row in selected:
            try:
                oracle_correct += symbolic_atomic_oracle(row) == row["target"]
            except (KeyError, ValueError):
                pass
        shortcuts = grouped_shortcut_probes(selected) if selected else {"overall_gate": False}
        task_reports[task] = {
            "count": len(selected), "answer_balance": dict(answers),
            "option_position_balance": dict(positions),
            "symbolic_oracle_accuracy": oracle_correct / len(selected) if selected else 0.0,
            "shortcut_probes": shortcuts,
            "gate": (len(selected) == 64 and answers == Counter(dict.fromkeys(CARDINAL, 16))
                     and positions == Counter(dict.fromkeys(range(4), 16))
                     and oracle_correct == 64),
        }
    pooled_shortcuts = {}
    common_probes = ("scene_index", "entity_name", "question_text", "template",
                     "option_position", "option_order", "image_filename", "metadata")
    for probe_name in common_probes:
        predictions = [prediction for task in task_reports.values()
                       for prediction in task["shortcut_probes"].get("probes", {}).get(
                           probe_name, {}).get("predictions", [])]
        score = sum(item["target"] == item["prediction"] for item in predictions) / len(rows)
        pooled_shortcuts[probe_name] = {"cross_validated_accuracy": score,
                                       "prediction_count": len(predictions),
                                       "gate": score <= 0.30 and len(predictions) == 256}
    binding_probes = task_reports[TASKS[3]]["shortcut_probes"].get("probes", {})
    binding_shortcut_gate = all(binding_probes.get(name, {}).get("gate", False) for name in (
        "descriptor_aware_text_only", "descriptor_color", "descriptor_shape"))
    manifests = root / "artifacts/atomic_v2/manifests"
    manifest_errors = (_manifest_errors(root, manifests / "data_manifest.json")
                       + _manifest_errors(root, manifests / "image_manifest.json"))
    overall = (len(rows) == 256 and len(set(ids)) == 256 and not overlap and not failures
               and not manifest_errors and all(task["gate"] for task in task_reports.values())
               and all(probe["gate"] for probe in pooled_shortcuts.values())
               and binding_shortcut_gate
               and {row["task"] for row in rows} == set(TASKS)
               and [row["scene_index"] for row in rows] == list(range(1, 257)))
    report = {
        "schema_version": 1, "protocol": "capability_gate_atomic_v2",
        "status": "ATOMIC_V2_DATA_VALID" if overall else "ATOMIC_V2_DATA_INVALID",
        "overall_gate": bool(overall), "scene_count": len(rows), "task_count": len(task_reports),
        "unique_scene_uuid_count": len(set(ids)), "v1_uuid_overlap_count": len(overlap),
        "within_scene_alias_collision_count": counts["within_scene_alias_collision"],
        "self_relation_count": counts["self_relation"],
        "duplicate_distractor_count": counts["duplicate_distractor"],
        "query_reference_equality_count": counts["query_reference_equality"],
        "symbolic_oracle_accuracy": sum(
            item["symbolic_oracle_accuracy"] * item["count"] for item in task_reports.values()
        ) / len(rows) if rows else 0.0,
        "tasks": task_reports, "row_failures": failures, "manifest_errors": manifest_errors,
        "shortcut_gate_scope": "per-shortcut pooled 256 OOF predictions; per-task diagnostic",
        "pooled_shortcut_probes": pooled_shortcuts,
        "binding_descriptor_shortcut_gate": binding_shortcut_gate,
        "data_manifest_sha256": sha256_file(manifests / "data_manifest.json")
        if (manifests / "data_manifest.json").exists() else None,
        "image_manifest_sha256": sha256_file(manifests / "image_manifest.json")
        if (manifests / "image_manifest.json").exists() else None,
        "evidence_policy": {
            "text_tasks": "Fresh blank provenance image; no image passed to the model.",
            "binding": "Unique internal aliases; no labels rendered on descriptor-selected objects.",
            "question_only": "Premise/descriptor removed; entity query retained.",
            "metadata": "Only task, template, entity_count, color and shape; no answer/evidence fields.",
        },
    }
    _json(root / "artifacts/atomic_v2/data_validation.json", report)
    report_path = root / "reports/atomic_v2/data_validation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Atomic v2 data validation", "", report["status"], "",
             f"Scenes: {len(rows)}; tasks: {len(task_reports)}; old UUID overlap: {len(overlap)}.",
             (f"Collisions: {report['within_scene_alias_collision_count']}; "
             f"self-relations: {report['self_relation_count']}; "
              f"symbolic oracle: {report['symbolic_oracle_accuracy']:.4f}."), "",
             "All reported shortcut scores are held-out, four-fold group-CV predictions.",
             "Categorical probes use training-only conditional majority. Index, aliases, question,",
             "filename and metadata use unconditional fixed train-only ridge classifiers.",
             "Hard gates apply to each shortcut's pooled 256 OOF predictions; per-task values",
             "are diagnostics. Binding descriptor text-only probes have an additional n=64 gate.",
             "These finite probes cannot exclude every possible nonlinear shortcut.", "",
             "| Task | n | Answer counts | Position counts | Oracle | Maximum CV |",
             "|---|---:|---|---|---:|---:|"]
    for task, result in task_reports.items():
        maximum = max((probe["cross_validated_accuracy"]
                       for probe in result["shortcut_probes"].get("probes", {}).values()), default=0)
        lines.append(f"| {task} | {result['count']} | {result['answer_balance']} | "
                     f"{result['option_position_balance']} | "
                     f"{result['symbolic_oracle_accuracy']:.4f} | {maximum:.4f} |")
    lines.extend(["", "Pooled shortcut hard gates:", ""])
    for name, probe in pooled_shortcuts.items():
        lines.append(f"- {name}: {probe['cross_validated_accuracy']:.6f}; gate={probe['gate']}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    if not overall and raise_on_invalid:
        raise DataValidationError("ATOMIC_V2_DATA_INVALID: see artifacts/atomic_v2/data_validation.json")
    return report


def _joint_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    templates = [i % 4 for i in range(128)]
    random.Random(SEEDS["joint_template_assignment"]).shuffle(templates)
    for index in range(128):
        base_id = _uid("joint", index)
        a, b, c = _aliases(base_id, ["A", "B", "C"])
        color, shape = COLORS[(index // 4) % 4], SHAPES[(index // 16) % 4]
        template = templates[index]
        options = list(DIAGONAL)
        random.Random(_seed("joint_option_permutation", base_id)).shuffle(options)
        horizontal = index % 2 == 0
        image_relations = ("east", "west") if horizontal else ("north", "south")
        text_relations = ("north", "south") if horizontal else ("east", "west")
        images = []
        for image_bit, image_relation in enumerate(image_relations):
            image_seed = _seed("joint_image", base_id, image_bit)
            rng = random.Random(image_seed)
            center = (256 + rng.randint(-5, 5), 256 + rng.randint(-5, 5))
            objects = [_obj(a, "A", _position(center, image_relation, rng.randint(140, 150)),
                            color, shape), _obj(b, "B", center, "black", "circle")]
            image_path = root / "artifacts/data_v2/joint_composition_screen/images" / (
                _uid("joint-image", base_id, image_bit) + ".png")
            _render_entities(image_path, objects, "Composition image relation")
            images.append({"image_path": image_path.relative_to(root).as_posix(),
                           "image_sha256": sha256_file(image_path), "image_entities": objects,
                           "image_seed": image_seed})
        conditions = []
        query = QUESTION_FORMS[template].format(a=a, b=c)
        for image_bit, image_relation in enumerate(image_relations):
            for text_bit, text_relation in enumerate(text_relations):
                premise = TEXT_FORMS[template].format(b=b, c=c, relation=text_relation)
                vx = VECTORS[image_relation][0] + VECTORS[text_relation][0]
                vy = VECTORS[image_relation][1] + VECTORS[text_relation][1]
                target = ("north" if vy < 0 else "south") + ("east" if vx > 0 else "west")
                conditions.append({
                    "condition": f"I{image_bit}T{text_bit}", "image_bit": image_bit,
                    "text_bit": text_bit, "image_relation": image_relation,
                    "text_relation": text_relation, **images[image_bit], "premise": premise,
                    "question": premise + "\n" + query, "question_without_premise": query,
                    "target": target, "symbolic_answer": target,
                    "correct_option_position": options.index(target),
                })
        rows.append({
            "schema_version": 2, "protocol": "capability_gate_joint_v2",
            "base_quartet_id": base_id, "split": "joint_composition_screen",
            "task": "joint_composition", "entities": {"A": a, "B": b, "C": c,
                                                            "A_color": color, "A_shape": shape},
            "options": options, "template_id": template,
            "axis_map": "horizontal_image_vertical_text" if horizontal
            else "vertical_image_horizontal_text", "conditions": conditions,
            "psi_fixed_target_condition": "I1T1", "psi_fixed_target": conditions[-1]["target"],
            "image_seed": _seed("joint_image", base_id),
        })
    random.Random(SEEDS["joint_scene_order"]).shuffle(rows)
    return rows


def _pixel_image_direction(root: Path, quartet: dict[str, Any], condition: dict[str, Any]) -> str:
    """Independent rendered-pixel color centroids, including the historical Joint audit."""
    with Image.open(root / condition["image_path"]) as image:
        image = image.convert("RGB")
        colors = [RGB[quartet["entities"]["A_color"]], RGB["black"]]
        centroids = []
        pixels = image.load()
        for color in colors:
            points = [(x, y) for y in range(60, 470) for x in range(40, 475)
                      if pixels[x, y] == color]
            if not points:
                raise ValueError("image entity color absent")
            centroids.append((sum(x for x, _ in points) / len(points),
                              sum(y for _, y in points) / len(points)))
    dx, dy = centroids[0][0] - centroids[1][0], centroids[0][1] - centroids[1][1]
    if max(abs(dx), abs(dy)) < 90 or min(abs(dx), abs(dy)) > 8:
        raise ValueError("image centroids inconsistent with cardinal construct")
    return ("east" if dx > 0 else "west") if abs(dx) > abs(dy) else (
        "south" if dy > 0 else "north")


def validate_joint_rows(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    errors: dict[str, list[str]] = {}
    flattened = []
    for index, quartet in enumerate(rows):
        failed = []
        a, b, c = [quartet["entities"][key] for key in ("A", "B", "C")]
        if len({a, b, c}) != 3:
            failed.append("alias_collision")
        if a == c:
            failed.append("query_reference_equality")
        conditions = quartet["conditions"]
        if ({row["condition"] for row in conditions} != {"I0T0", "I0T1", "I1T0", "I1T1"}
                or len(conditions) != 4 or len({row["target"] for row in conditions}) != 4):
            failed.append("factorial_quartet_invalid")
        for condition in conditions:
            try:
                image_relation = _pixel_image_direction(root, quartet, condition)
                source, destination, text_relation = _premise_fact(condition["premise"])
                if (source, destination) != (b, c) or source == destination:
                    failed.append("text_identity_error")
                vx = VECTORS[image_relation][0] + VECTORS[text_relation][0]
                vy = VECTORS[image_relation][1] + VECTORS[text_relation][1]
                if vx == 0 or vy == 0:
                    raise ValueError("nonorthogonal composition")
                oracle = ("north" if vy < 0 else "south") + ("east" if vx > 0 else "west")
                if oracle != condition["target"] or oracle != condition["symbolic_answer"]:
                    failed.append("symbolic_oracle_error")
                if sha256_file(root / condition["image_path"]) != condition["image_sha256"]:
                    failed.append("image_hash_error")
                if quartet["options"][condition["correct_option_position"]] != oracle:
                    failed.append("option_position_error")
            except (ValueError, KeyError, OSError):
                failed.append("symbolic_oracle_error")
            flattened.append({**condition, "base_quartet_id": quartet["base_quartet_id"],
                              "scene_id": quartet["base_quartet_id"] + condition["condition"],
                              "scene_index": index + 1, "query_names": [a, c],
                              "template_id": quartet["template_id"], "task": "joint_composition",
                              "entity_count": 3, "color": quartet["entities"]["A_color"],
                              "shape": quartet["entities"]["A_shape"], "options": quartet["options"]})
        # Holding either modality constant leaves two distinct correct answers.
        for bit_key in ("image_bit", "text_bit"):
            for bit in range(2):
                matching = [row for row in conditions if row[bit_key] == bit]
                if len({row["target"] for row in matching}) != 2:
                    failed.append("unimodal_uniquely_identifying")
        if len({row["question_without_premise"] for row in conditions}) != 1:
            failed.append("question_only_leakage")
        if failed:
            errors[quartet["base_quartet_id"]] = sorted(set(failed))
    shortcuts = grouped_shortcut_probes(flattened, joint=True) if flattened else {
        "overall_gate": False}
    # Filename reveals the image relation, not the unavailable orthogonal factor;
    # grouped CV prevents a quartet's images appearing on both sides of a fold.
    answer_counts = Counter(row["target"] for row in flattened)
    position_counts = Counter(row["correct_option_position"] for row in flattened)
    valid = (len(rows) == 128 and len({row["base_quartet_id"] for row in rows}) == 128
             and not errors and shortcuts["overall_gate"]
             and answer_counts == Counter(dict.fromkeys(DIAGONAL, 128))
             and position_counts == Counter(dict.fromkeys(range(4), 128)))
    return {"overall_gate": bool(valid), "quartet_count": len(rows),
            "condition_count": len(flattened), "quartet_failures": errors,
            "answer_balance": dict(answer_counts), "option_position_balance": dict(position_counts),
            "shortcut_probes": shortcuts, "single_modality_nonidentifying": not any(
                "unimodal_uniquely_identifying" in failures for failures in errors.values()),
            "symbolic_oracle_accuracy": 1 - sum(
                "symbolic_oracle_error" in failures for failures in errors.values()
            ) / len(rows) if rows else 0.0}


def validate_joint_data_before_atomic_v2(root: Path = ROOT, *,
                                        raise_on_invalid: bool = True) -> dict[str, Any]:
    root = Path(root)
    old_path = root / "artifacts/data/joint_composition_screen/quartets.jsonl"
    if not old_path.exists():
        raise DataValidationError("historical Joint data missing; cannot establish preservation")
    old = validate_joint_rows(read_jsonl(old_path), root)
    old["sha256"] = sha256_file(old_path)
    path = root / "artifacts/data_v2/joint_composition_screen/quartets.jsonl"
    manifest_path = root / "artifacts/atomic_v2/manifests/joint_data_manifest.json"
    if not old["overall_gate"] and not path.exists():
        _assert_pre_outcome(root)
        journal = root / "artifacts/atomic_v2/data/joint_generation_attempt.json"
        if journal.exists():
            raise DataValidationError("partial Joint generation preserved; no automatic regeneration")
        _json(journal, {"batch": 1, "old_joint_sha256": old["sha256"],
                        "reason": "pre_outcome_construct_repair", "seeds": SEEDS})
        rows = _joint_rows(root)
        _jsonl(path, rows)
        images = [root / condition["image_path"] for row in rows for condition in row["conditions"]]
        _json(manifest_path, _manifest(root, [path, *images], "joint_v2_pre_atomic_freeze"))
    selected_path = path if path.exists() else old_path
    selected = validate_joint_rows(read_jsonl(selected_path), root)
    manifest_errors = _manifest_errors(root, manifest_path) if selected_path == path else []
    old_ids = {row["base_quartet_id"] for row in read_jsonl(old_path)}
    new_ids = {row["base_quartet_id"] for row in read_jsonl(selected_path)}
    overlap = len(old_ids & new_ids) if selected_path == path else None
    overall = selected["overall_gate"] and not manifest_errors and (overlap in {0, None})
    result = {"schema_version": 1, "status": "JOINT_DATA_VALID" if overall
              else "ATOMIC_V2_DATA_INVALID", "overall_gate": bool(overall), "historical": old,
              "selected": selected, "selected_data_path": selected_path.relative_to(root).as_posix(),
              "selected_data_sha256": sha256_file(selected_path),
              "pre_outcome_construct_repair": selected_path == path,
              "v1_quartet_uuid_overlap_count": overlap, "manifest_errors": manifest_errors,
              "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None}
    _json(root / "artifacts/atomic_v2/joint_data_validation.json", result)
    if not overall and raise_on_invalid:
        raise DataValidationError("ATOMIC_V2_DATA_INVALID: Joint pre-outcome validation failed")
    return result
