from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

from capability_gate.artifacts import build_manifest, sha256_file, write_json, write_jsonl
from capability_gate.paths import ARTIFACTS, CONFIGS, ROOT, ensure_layout

CARDINAL = ("north", "south", "east", "west")
DIAGONAL = ("northeast", "northwest", "southeast", "southwest")
COLORS = ("crimson", "azure", "emerald", "amber")
RGB = {
    "crimson": (198, 49, 73),
    "azure": (40, 116, 201),
    "emerald": (32, 146, 91),
    "amber": (225, 155, 34),
    "black": (35, 37, 40),
    "violet": (132, 76, 173),
}
SHAPES = ("circle", "square", "triangle", "diamond")
VECTORS = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
REVERSE = {"north": "south", "south": "north", "east": "west", "west": "east"}
QUESTION_FORMS = (
    "Where is {a} relative to {b}?",
    "What is {a}'s direction from {b}?",
    "Relative to {b}, which direction contains {a}?",
    "Choose the direction of {a} when {b} is the reference.",
)
TEXT_FORMS = (
    "{b} is {relation} of {c}.",
    "Relative to {c}, {b} lies to the {relation}.",
    "From {c}, move {relation} to reach {b}.",
    "The position of {b} is {relation} with respect to {c}.",
)
SYLLABLES = (
    "zor",
    "vek",
    "lumi",
    "prax",
    "navi",
    "tul",
    "seno",
    "kiri",
    "dax",
    "melo",
    "runi",
    "fex",
    "gavi",
    "haro",
    "juno",
    "wesi",
)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _uid(namespace: uuid.UUID, *parts: object) -> str:
    return str(uuid.uuid5(namespace, "capability-gate-v1/" + "/".join(map(str, parts))))


def _name(namespace: uuid.UUID, *parts: object) -> str:
    raw = uuid.uuid5(namespace, "entity/" + "/".join(map(str, parts))).int
    return (SYLLABLES[raw % 16] + SYLLABLES[(raw >> 8) % 16]).capitalize()


def _options(answer: str, position: int, candidates: tuple[str, ...], spin: int) -> list[str]:
    remaining = [candidate for candidate in candidates if candidate != answer]
    shift = spin % len(remaining)
    remaining = remaining[shift:] + remaining[:shift]
    remaining.insert(position, answer)
    return remaining


def _draw_shape(draw: ImageDraw.ImageDraw, xy: tuple[int, int], shape: str, color: str) -> None:
    x, y = xy
    radius = 27
    fill = RGB[color]
    outline = (20, 20, 20)
    if shape == "circle":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill, outline, width=3)
    elif shape == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill, outline, width=3)
    elif shape == "triangle":
        draw.polygon(((x, y - 32), (x - 31, y + 26), (x + 31, y + 26)), fill)
        draw.line(((x, y - 32), (x - 31, y + 26), (x + 31, y + 26), (x, y - 32)), outline, 3)
    else:
        draw.polygon(((x, y - 33), (x - 30, y), (x, y + 33), (x + 30, y)), fill)
        draw.line(((x, y - 33), (x - 30, y), (x, y + 33), (x + 30, y), (x, y - 33)), outline, 3)


def _render(
    path: Path,
    objects: list[dict[str, Any]],
    *,
    title: str,
    decorative: bool = False,
) -> None:
    image = Image.new("RGB", (512, 512), (249, 248, 243))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.rectangle((6, 6, 505, 505), outline=(80, 80, 80), width=2)
    draw.text((18, 14), title, fill=(25, 25, 25), font=font)
    if decorative:
        draw.text((18, 42), "DECORATIVE IMAGE — TEXT TASK", fill=(125, 55, 55), font=small)
    for obj in objects:
        _draw_shape(draw, (obj["x"], obj["y"]), obj["shape"], obj["color"])
        label = obj["name"]
        if not label:
            continue
        box = draw.textbbox((0, 0), label, font=small)
        width = box[2] - box[0]
        draw.rectangle(
            (obj["x"] - width // 2 - 3, obj["y"] + 35, obj["x"] + width // 2 + 3, obj["y"] + 55),
            fill=(249, 248, 243),
        )
        draw.text((obj["x"] - width // 2, obj["y"] + 36), label, fill=(15, 15, 15), font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def _position(origin: tuple[int, int], relation: str, distance: int = 150) -> tuple[int, int]:
    dx, dy = VECTORS[relation]
    return origin[0] + dx * distance, origin[1] + dy * distance


def _distractors(
    namespace: uuid.UUID,
    key: str,
    count: int,
    occupied: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    slots = [(90, 100), (422, 100), (90, 420), (422, 420), (256, 410), (405, 255)]
    result = []
    for j, xy in enumerate(point for point in slots if point not in occupied):
        if len(result) >= count:
            break
        result.append(
            {
                "name": _name(namespace, key, "distractor", j),
                "color": COLORS[(j + 2) % 4],
                "shape": SHAPES[(j + 1) % 4],
                "x": xy[0],
                "y": xy[1],
                "role": "distractor",
            }
        )
    return result


def _atomic_row(
    namespace: uuid.UUID,
    task: str,
    index: int,
    split: str,
    output_dir: Path,
) -> dict[str, Any]:
    relation = CARDINAL[index % 4]
    nuisance_block = index // 4
    block = nuisance_block % 4
    feature = (nuisance_block + nuisance_block // 4) % 4
    color = COLORS[feature]
    shape = SHAPES[(index // 4 + 2 * block) % 4]
    entity_count = 2 + (nuisance_block % 4)
    scene_id = _uid(namespace, split, task, index)
    a = _name(namespace, split, task, nuisance_block, "a")
    b = _name(namespace, split, task, nuisance_block, "b")
    c = _name(namespace, split, task, nuisance_block, "c")
    image_path = output_dir / "images" / f"{scene_id}.png"
    template_id = block
    premise = ""

    if task == "direct_visual_relation":
        center = (256, 260)
        query_xy = _position(center, relation)
        objects = [
            {
                "name": a,
                "color": color,
                "shape": shape,
                "x": query_xy[0],
                "y": query_xy[1],
                "role": "query",
            },
            {
                "name": b,
                "color": "black",
                "shape": "circle",
                "x": center[0],
                "y": center[1],
                "role": "reference",
            },
        ]
        distractor_key = f"{split}/{task}/block/{nuisance_block}"
        objects += _distractors(namespace, distractor_key, entity_count - 2, {center, query_xy})
        question = QUESTION_FORMS[template_id].format(a=a, b=b)
        _render(image_path, objects, title="Spatial relation scene")
    elif task in {"direct_text_relation", "direction_reversal"}:
        decorative_key = f"{split}/{task}/block/{nuisance_block}"
        decorative_objects = _distractors(namespace, decorative_key, entity_count, set())
        _render(image_path, decorative_objects, title="Matched decorative panel", decorative=True)
        premise_relation = relation if task == "direct_text_relation" else REVERSE[relation]
        premise = TEXT_FORMS[template_id].format(b=b, c=c, relation=premise_relation)
        query_a, query_b = (b, c) if task == "direct_text_relation" else (c, b)
        question = (
            "The image is unrelated decoration; use the text statement.\n"
            + premise
            + "\n"
            + QUESTION_FORMS[template_id].format(a=query_a, b=query_b)
        )
        objects = decorative_objects
    elif task == "cross_modal_bridge_binding":
        center = (256, 260)
        bridge = b
        descriptor_by_relation = {}
        objects = [
            {
                "name": "",
                "color": "black",
                "shape": "circle",
                "x": center[0],
                "y": center[1],
                "role": "anchor",
            }
        ]
        for direction_index, direction in enumerate(CARDINAL):
            object_color = COLORS[(feature + direction_index) % 4]
            object_shape = SHAPES[(SHAPES.index(shape) + 2 * direction_index) % 4]
            object_xy = _position(center, direction)
            descriptor_by_relation[direction] = (object_color, object_shape)
            objects.append(
                {
                    "name": "",
                    "color": object_color,
                    "shape": object_shape,
                    "x": object_xy[0],
                    "y": object_xy[1],
                    "role": "bridge" if direction == relation else "candidate",
                }
            )
        color, shape = descriptor_by_relation[relation]
        entity_count = 5
        forms = (
            "The bridge entity named {name} is the {color} {shape} shown in the image. Where is that bridge entity relative to the black anchor?",
            "In the image, bind the name {name} to its {color} {shape}. Which direction is it from the black anchor?",
            "Locate {name}, the {color} {shape}, in the image. Relative to the black anchor, where is it?",
            "The text refers to the {color} {shape} as {name}. Choose its direction from the black anchor.",
        )
        question = forms[template_id].format(name=bridge, color=color, shape=shape)
        _render(image_path, objects, title="Bridge-binding scene")
    else:
        raise ValueError(f"unknown task: {task}")

    options = list(CARDINAL[block:] + CARDINAL[:block])
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "split": split,
        "task": task,
        "image_path": image_path.relative_to(ROOT).as_posix(),
        "image_sha256": sha256_file(image_path),
        "question": question,
        "premise": premise,
        "options": options,
        "target": relation,
        "correct_option_position": options.index(relation),
        "symbolic_answer": relation,
        "template_id": template_id,
        "nuisance_block_id": nuisance_block,
        "question_form": template_id,
        "color": color,
        "shape": shape,
        "entity_count": entity_count,
        "answer_surface_token_length": 1,
        "entities": objects,
        "query_names": [a, b] if task == "direct_visual_relation" else [b, c],
        "image_seed": int(uuid.UUID(scene_id)) % (2**32),
    }


def _joint_answer(image_relation: str, text_relation: str) -> str:
    x = VECTORS[image_relation][0] + VECTORS[text_relation][0]
    y = VECTORS[image_relation][1] + VECTORS[text_relation][1]
    if x == 0 or y == 0:
        raise ValueError("joint relations must be orthogonal and produce a diagonal")
    return ("north" if y < 0 else "south") + ("east" if x > 0 else "west")


def _joint_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    namespace = uuid.UUID(config["namespace_uuid"])
    output_dir = ARTIFACTS / "data" / "joint_composition_screen"
    rows: list[dict[str, Any]] = []
    for index in range(config["base_quartets"]):
        base_id = _uid(namespace, "joint", index)
        a = _name(namespace, "joint", index, "a")
        b = _name(namespace, "joint", index, "b")
        c = _name(namespace, "joint", index, "c")
        color = COLORS[(index // 4) % 4]
        shape = SHAPES[(index // 16) % 4]
        template_id = (index // 32) % 4
        option_order = list(DIAGONAL)
        random.Random(config["seed"] + index).shuffle(option_order)
        axis_map = (
            "horizontal_image_vertical_text" if index % 2 == 0 else "vertical_image_horizontal_text"
        )
        image_relations = ("east", "west") if index % 2 == 0 else ("north", "south")
        text_relations = ("north", "south") if index % 2 == 0 else ("east", "west")
        images: dict[str, dict[str, str]] = {}
        for image_bit, relation in enumerate(image_relations):
            center = (256, 260)
            a_xy = _position(center, relation)
            image_path = output_dir / "images" / f"{base_id}_I{image_bit}.png"
            objects = [
                {
                    "name": a,
                    "color": color,
                    "shape": shape,
                    "x": a_xy[0],
                    "y": a_xy[1],
                    "role": "A",
                },
                {
                    "name": b,
                    "color": "black",
                    "shape": "circle",
                    "x": center[0],
                    "y": center[1],
                    "role": "B",
                },
            ]
            _render(image_path, objects, title="Composition image relation")
            images[f"I{image_bit}"] = {
                "path": image_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(image_path),
                "relation": relation,
            }
        conditions = []
        for image_bit, image_relation in enumerate(image_relations):
            for text_bit, text_relation in enumerate(text_relations):
                condition = f"I{image_bit}T{text_bit}"
                premise = TEXT_FORMS[template_id].format(b=b, c=c, relation=text_relation)
                question = premise + "\n" + QUESTION_FORMS[template_id].format(a=a, b=c)
                answer = _joint_answer(image_relation, text_relation)
                conditions.append(
                    {
                        "condition": condition,
                        "image_bit": image_bit,
                        "text_bit": text_bit,
                        "image_relation": image_relation,
                        "text_relation": text_relation,
                        "image_path": images[f"I{image_bit}"]["path"],
                        "image_sha256": images[f"I{image_bit}"]["sha256"],
                        "premise": premise,
                        "question": question,
                        "question_without_premise": QUESTION_FORMS[template_id].format(a=a, b=c),
                        "target": answer,
                        "correct_option_position": option_order.index(answer),
                        "symbolic_answer": answer,
                    }
                )
        rows.append(
            {
                "schema_version": 1,
                "base_quartet_id": base_id,
                "split": "joint_composition_screen",
                "task": "joint_composition",
                "entities": {"A": a, "B": b, "C": c, "A_color": color, "A_shape": shape},
                "options": option_order,
                "template_id": template_id,
                "axis_map": axis_map,
                "conditions": conditions,
                "psi_fixed_target_condition": "I1T1",
                "psi_fixed_target": next(
                    x["target"] for x in conditions if x["condition"] == "I1T1"
                ),
                "image_seed": int(uuid.UUID(base_id)) % (2**32),
            }
        )
    return rows


def generate_all() -> dict[str, Any]:
    ensure_layout()
    frozen_registry = ARTIFACTS / "models" / "frozen_registry.json"
    if not frozen_registry.exists():
        raise RuntimeError("freeze the three-model registry before generating task data")
    existing_manifest = ARTIFACTS / "manifests" / "data_manifest.json"
    if existing_manifest.exists():
        return json.loads(existing_manifest.read_text(encoding="utf-8"))
    atomic_config = _load(CONFIGS / "atomic_tasks.yaml")
    joint_config = _load(CONFIGS / "joint_screen.yaml")
    namespace = uuid.UUID(atomic_config["namespace_uuid"])
    written: list[Path] = []

    smoke_dir = ARTIFACTS / "data" / "engineering_smoke"
    tasks = tuple(atomic_config["splits"]["atomic_qualification"])
    smoke_rows = [
        _atomic_row(namespace, task, local_index, "engineering_smoke", smoke_dir)
        for task in tasks
        for local_index in range(3)
    ]
    smoke_path = smoke_dir / "scenes.jsonl"
    write_jsonl(smoke_path, smoke_rows)
    written.append(smoke_path)

    atomic_dir = ARTIFACTS / "data" / "atomic_qualification"
    atomic_rows = []
    for task, count in atomic_config["splits"]["atomic_qualification"].items():
        atomic_rows.extend(
            _atomic_row(namespace, task, index, "atomic_qualification", atomic_dir)
            for index in range(count)
        )
    atomic_path = atomic_dir / "scenes.jsonl"
    write_jsonl(atomic_path, atomic_rows)
    written.append(atomic_path)

    joint_path = ARTIFACTS / "data" / "joint_composition_screen" / "quartets.jsonl"
    write_jsonl(joint_path, _joint_rows(joint_config))
    written.append(joint_path)

    image_paths = sorted((ARTIFACTS / "data").glob("**/*.png"))
    manifest = build_manifest(ROOT, [*written, *image_paths], "generated_data")
    manifest["counts"] = {
        "engineering_smoke_scenes": len(smoke_rows),
        "atomic_qualification_scenes": len(atomic_rows),
        "joint_base_quartets": joint_config["base_quartets"],
        "joint_conditions": joint_config["base_quartets"] * 4,
        "images": len(image_paths),
    }
    manifest_path = ARTIFACTS / "manifests" / "data_manifest.json"
    write_json(manifest_path, manifest)
    return manifest
