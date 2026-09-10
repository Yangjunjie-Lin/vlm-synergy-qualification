# Atomic v2 data validation

ATOMIC_V2_DATA_VALID

Scenes: 256; tasks: 4; old UUID overlap: 0.
Collisions: 0; self-relations: 0; symbolic oracle: 1.0000.

All reported shortcut scores are held-out, four-fold group-CV predictions.
Categorical probes use training-only conditional majority. Index, aliases, question,
filename and metadata use unconditional fixed train-only ridge classifiers.
Hard gates apply to each shortcut's pooled 256 OOF predictions; per-task values
are diagnostics. Binding descriptor text-only probes have an additional n=64 gate.
These finite probes cannot exclude every possible nonlinear shortcut.

| Task | n | Answer counts | Position counts | Oracle | Maximum CV |
|---|---:|---|---|---:|---:|
| direct_visual_relation | 64 | {'north': 16, 'east': 16, 'west': 16, 'south': 16} | {1: 16, 0: 16, 3: 16, 2: 16} | 1.0000 | 0.2500 |
| direct_text_relation | 64 | {'north': 16, 'west': 16, 'east': 16, 'south': 16} | {2: 16, 0: 16, 1: 16, 3: 16} | 1.0000 | 0.3281 |
| direction_reversal | 64 | {'east': 16, 'north': 16, 'west': 16, 'south': 16} | {2: 16, 1: 16, 3: 16, 0: 16} | 1.0000 | 0.3281 |
| cross_modal_bridge_binding | 64 | {'north': 16, 'south': 16, 'east': 16, 'west': 16} | {2: 16, 3: 16, 0: 16, 1: 16} | 1.0000 | 0.2500 |

Pooled shortcut hard gates:

- scene_index: 0.230469; gate=True
- entity_name: 0.230469; gate=True
- question_text: 0.214844; gate=True
- template: 0.250000; gate=True
- option_position: 0.250000; gate=True
- option_order: 0.250000; gate=True
- image_filename: 0.250000; gate=True
- metadata: 0.250000; gate=True
