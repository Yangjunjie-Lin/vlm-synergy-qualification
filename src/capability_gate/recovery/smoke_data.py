from __future__ import annotations

import json
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from capability_gate.artifacts import sha256_file, write_jsonl
from capability_gate.paths import ARTIFACTS, ROOT

DIRECTIONS = ("north", "south", "east", "west")


def _formal_ids() -> set[str]:
    result: set[str] = set()
    for path in (
        ROOT / "artifacts/data/atomic_qualification/scenes.jsonl",
        ROOT / "artifacts/data/joint_composition_screen/quartets.jsonl",
    ):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for key in ("scene_id", "base_quartet_id"):
                    if row.get(key):
                        result.add(row[key])
    return result


def generate_recovery_smoke_scenes() -> list[dict[str, Any]]:
    manifest_dir = ARTIFACTS / "engineering_recovery/manifests"
    image_dir = manifest_dir / "images"
    scene_path = manifest_dir / "engineering_only_scenes.jsonl"
    image_dir.mkdir(parents=True, exist_ok=True)
    formal_ids = _formal_ids()
    rows = []
    colors = ("royalblue", "darkorange", "seagreen")
    for index in range(12):
        direction = DIRECTIONS[index % len(DIRECTIONS)]
        scene_id = f"engineering-recovery-visual-{index:02d}"
        if scene_id in formal_ids:
            raise RuntimeError(f"engineering scene overlaps formal data: {scene_id}")
        image_path = image_dir / f"{scene_id}.png"
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        color = colors[index % len(colors)]
        draw.rounded_rectangle((42, 42, 470, 470), radius=36, outline=color, width=18)
        draw.rectangle((100, 196, 412, 316), fill=color)
        font = ImageFont.load_default(size=52)
        bbox = draw.textbbox((0, 0), direction, font=font)
        x = (512 - (bbox[2] - bbox[0])) // 2
        y = (512 - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), direction, fill="white", font=font)
        image.save(image_path, format="PNG", optimize=False)
        rows.append(
            {
                "schema_version": 1,
                "scene_id": scene_id,
                "engineering_only": True,
                "formal_overlap": False,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "image_sha256": sha256_file(image_path),
                "prompt": (
                    "Engineering visual transport check: read the single direction word printed "
                    "inside the colored panel. Return one allowed answer."
                ),
                "candidates": list(DIRECTIONS),
                "target": direction,
            }
        )
    write_jsonl(scene_path, rows)
    return rows
