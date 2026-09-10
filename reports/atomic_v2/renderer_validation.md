# Official renderer validation

```json
{
  "qwen2_5_vl_7b": {
    "schema_version": 1,
    "model_key": "qwen2_5_vl_7b",
    "status": "RENDERER_VALIDATION_PASS",
    "overall_gate": true,
    "processor": {
      "model_key": "qwen2_5_vl_7b",
      "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
      "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
      "processor_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
      "tokenizer_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
      "processor_class": "Qwen2_5_VLProcessor",
      "tokenizer_class": "Qwen2TokenizerFast",
      "official_chat_template": "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
      "official_chat_template_sha256": "4df95d05455c2a6f762616492bba63d0f9bef904ef47daabad35f5715d6d2e4e",
      "native_model_class": "Qwen2_5_VLForConditionalGeneration",
      "renderer_adapter_sha256": "c35cdf9a0cdb061101aad2102f803868546efd33577059bc97123673324b91f7",
      "inherited_scoring_adapter_sha256": "1cd203b0b43862ce9c40dfff0706ba7f46c0c6a8ac3186eb768bdd790fbee1c5"
    },
    "snapshots": [
      {
        "scene_id": "3381c178-7f53-5d5c-8bc1-799fa98a6833",
        "task": "direct_visual_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "3b57ef0f89036aace080149fe005ad9cdb8107c26b9851e395d31c66fa623511",
        "input_ids_sha256": "6b4d005db092f655fbdf6d9d12533715aeb36a04fb87d38704e96410afa14aaf",
        "input_sequence_length": 408,
        "image_block_counts": {
          "<|vision_start|>": 1,
          "<|image_pad|>": 1,
          "<|vision_end|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      },
      {
        "scene_id": "eadf0543-57e5-5b6d-83c5-a37d23ab1be7",
        "task": "direct_visual_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "7bdadf2a1f87389d7ed9244a16aa4a92a38e2b2d5e0936608ff1c00542eb55ad",
        "input_ids_sha256": "4a741aa51c46cf9ee9b26ff3515a9f3aefd3c16a684a2e44e35a385566a081d3",
        "input_sequence_length": 409,
        "image_block_counts": {
          "<|vision_start|>": 1,
          "<|image_pad|>": 1,
          "<|vision_end|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      },
      {
        "scene_id": "3fd755b1-dd1b-5457-ad6b-95cc64412f6e",
        "task": "direct_text_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "825cd888a9d26b0b65fcbc14e0fdf4b8e908bcf8b446f40a42de9007bb193252",
        "input_ids_sha256": "127fc651b99c0ef7191ade766e9ec0592a7637215ff77da527db36117ad4aa7e",
        "input_sequence_length": 112,
        "image_block_counts": {
          "<|vision_start|>": 0,
          "<|image_pad|>": 0,
          "<|vision_end|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "99b36973-22a5-5066-a088-e42f6d53be88",
        "task": "direct_text_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "6a9e884d02671b07a34b17c0e594cc42f4b3856e285890833bc7f1324c4245bd",
        "input_ids_sha256": "6d4bf9d0f11cc91c0e796e105a0102259cf0dae3226f9c31657c4bc1b3a668a3",
        "input_sequence_length": 112,
        "image_block_counts": {
          "<|vision_start|>": 0,
          "<|image_pad|>": 0,
          "<|vision_end|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "54a0893c-6d73-598b-bf48-13068b97c43e",
        "task": "direction_reversal",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "b5f7a5533b44a11c2f9e5233142221e980c3228e42911f8a2fa688179b3b6da7",
        "input_ids_sha256": "95150b831c9559d93b5e83c4d3b6d2adb4a81b605d6131be1f64bc9f0abf5c69",
        "input_sequence_length": 112,
        "image_block_counts": {
          "<|vision_start|>": 0,
          "<|image_pad|>": 0,
          "<|vision_end|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "98b3d118-f942-5026-a34e-7551f55a2974",
        "task": "direction_reversal",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "b95c64d278b189b9852c3e6a483715a3487dd5c6702e238bc3a7bb65c93f8eaa",
        "input_ids_sha256": "7e66638dcf9561bccff40f78ad660581842710db9d8ece74535640dfa5deb57f",
        "input_sequence_length": 114,
        "image_block_counts": {
          "<|vision_start|>": 0,
          "<|image_pad|>": 0,
          "<|vision_end|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "9e127134-6a18-56ef-9dab-02f97cbb0d34",
        "task": "cross_modal_bridge_binding",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "8b6781abe3b8423666b6108710cba80a03a903b444d6d16069c0329c60621e19",
        "input_ids_sha256": "d161f8d57cb3c7731aa6f3f082c5d643a3fee63acf08168d71f6aca8d25aab52",
        "input_sequence_length": 415,
        "image_block_counts": {
          "<|vision_start|>": 1,
          "<|image_pad|>": 1,
          "<|vision_end|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      },
      {
        "scene_id": "bf1e9113-fc90-5839-b209-82427ab0bc85",
        "task": "cross_modal_bridge_binding",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "661984e7577140c102983c0df33b1ba2036bee2d9f0a91fc4433430235839e64",
        "input_ids_sha256": "b4382772093ba8b142fb752322af25b05d6ef798f867da8a32f54403edc83e1b",
        "input_sequence_length": 415,
        "image_block_counts": {
          "<|vision_start|>": 1,
          "<|image_pad|>": 1,
          "<|vision_end|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      }
    ],
    "snapshot_count": 8,
    "failure": null,
    "runtime_seconds": 7.738562342000023,
    "model_forward_passes": 0,
    "files": [
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/cross_modal_bridge_binding_01.json",
        "sha256": "6eca01231496695c2fe779c658407d6a9ce8675f15601d416b55e626b61ecbe2"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/cross_modal_bridge_binding_02.json",
        "sha256": "44e063e287631e0f00dedd90055495b72b80207673d894343ea6e97a891a855c"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/direct_text_relation_01.json",
        "sha256": "5c88c53bfe8508c933a66a379ac48a20214456b34a0b5e340d657d1e368d2a65"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/direct_text_relation_02.json",
        "sha256": "28be893fabd9dba1a4c8cba334a0b8d43a2c1f9dcd7760ccca468d204e5d6272"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/direct_visual_relation_01.json",
        "sha256": "51cd72b5d475950d209953719932662e9881a097603dad1c8db6adc15e0018ab"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/direct_visual_relation_02.json",
        "sha256": "2fa8e0375bfe0e28cacd404ded09f491890d85f454b1afd7a96521b009c565ef"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/direction_reversal_01.json",
        "sha256": "bbaf6698b42b687fcea125325111b86f8fa5e32b2d1a1d7df02952c6fdb05e4e"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/direction_reversal_02.json",
        "sha256": "04a7f8a7fe5ac41a16277ac6558a211a125b07adaf7523b8fe9f83770a60f8a2"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/qwen2_5_vl_7b/stderr.log",
        "sha256": "8c6e98ec4b34cb4f952e4c27da6527607c0d5e976dfe5c5aa3b9c1a323a644d0"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/cross_modal_bridge_binding_01.txt",
        "sha256": "8b6781abe3b8423666b6108710cba80a03a903b444d6d16069c0329c60621e19"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/cross_modal_bridge_binding_02.txt",
        "sha256": "661984e7577140c102983c0df33b1ba2036bee2d9f0a91fc4433430235839e64"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/direct_text_relation_01.txt",
        "sha256": "825cd888a9d26b0b65fcbc14e0fdf4b8e908bcf8b446f40a42de9007bb193252"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/direct_text_relation_02.txt",
        "sha256": "6a9e884d02671b07a34b17c0e594cc42f4b3856e285890833bc7f1324c4245bd"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/direct_visual_relation_01.txt",
        "sha256": "3b57ef0f89036aace080149fe005ad9cdb8107c26b9851e395d31c66fa623511"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/direct_visual_relation_02.txt",
        "sha256": "7bdadf2a1f87389d7ed9244a16aa4a92a38e2b2d5e0936608ff1c00542eb55ad"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/direction_reversal_01.txt",
        "sha256": "b5f7a5533b44a11c2f9e5233142221e980c3228e42911f8a2fa688179b3b6da7"
      },
      {
        "path": "tests/snapshots/qwen_atomic_prompts/direction_reversal_02.txt",
        "sha256": "b95c64d278b189b9852c3e6a483715a3487dd5c6702e238bc3a7bb65c93f8eaa"
      }
    ],
    "created_at": "2026-09-10T12:15:33.362586+00:00",
    "hostname": "10-60-165-203",
    "platform": "Linux-5.15.0-113-generic-x86_64-with-glibc2.39",
    "python": "3.11.16 (main, Sep  1 2026, 14:18:37) [Clang 22.1.3 ]",
    "executable": "/workspace/vlm-synergy-qualification/envs/qwen/.venv/bin/python",
    "pid": 366,
    "git_commit": "c0aee2c24b569874c4861d7495e91236b4f14782",
    "git_branch": "codex/atomic-protocol-v2-clean-rerun"
  },
  "glm4_1v_9b": {
    "schema_version": 1,
    "model_key": "glm4_1v_9b",
    "status": "RENDERER_VALIDATION_PASS",
    "overall_gate": true,
    "processor": {
      "model_key": "glm4_1v_9b",
      "model_id": "zai-org/GLM-4.1V-9B-Thinking",
      "model_revision": "3c1471e51dc811b589d4d12b1c1c7c1c941267c2",
      "processor_revision": "3c1471e51dc811b589d4d12b1c1c7c1c941267c2",
      "tokenizer_revision": "3c1471e51dc811b589d4d12b1c1c7c1c941267c2",
      "processor_class": "Glm4vProcessor",
      "tokenizer_class": "PreTrainedTokenizerFast",
      "official_chat_template": "[gMASK]<sop>\n{%- for msg in messages %}\n    {%- if msg.role == 'system' %}\n<|system|>\n{{ msg.content }}\n    {%- elif msg.role == 'user' %}\n<|user|>{{ '\\n' }}\n\n        {%- if msg.content is string %}\n{{ msg.content }}\n        {%- else %}\n            {%- for item in msg.content %}\n                {%- if item.type == 'video' or 'video' in item %}\n<|begin_of_video|><|video|><|end_of_video|>\n                {%- elif item.type == 'image' or 'image' in item %}\n<|begin_of_image|><|image|><|end_of_image|>\n                {%- elif item.type == 'text' %}\n{{ item.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n    {%- elif msg.role == 'assistant' %}\n        {%- if msg.metadata %}\n<|assistant|>{{ msg.metadata }}\n{{ msg.content }}\n        {%- else %}\n<|assistant|>\n{{ msg.content }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}<|assistant|>\n{% endif %}",
      "official_chat_template_sha256": "5232dbe7ee947df4d2009f2b262b70a22d4f2d2f0590e2af4ef2a78096171e90",
      "native_model_class": "Glm4vForConditionalGeneration",
      "renderer_adapter_sha256": "a580b636e33410bf8888e96afeb687c5856ae583d4305ddcff0759f56b7275fc",
      "inherited_scoring_adapter_sha256": "1cd203b0b43862ce9c40dfff0706ba7f46c0c6a8ac3186eb768bdd790fbee1c5"
    },
    "snapshots": [
      {
        "scene_id": "3381c178-7f53-5d5c-8bc1-799fa98a6833",
        "task": "direct_visual_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "2965591ff19607b946a2f901f4eb533b426943d1776f82e29aa690703c61ba7d",
        "input_ids_sha256": "d967ad80e6bf96f5be3e52236c2d7dd74a30f391405fd11d0e5df0ff5b9a0cc7",
        "input_sequence_length": 403,
        "image_block_counts": {
          "<|begin_of_image|>": 1,
          "<|image|>": 1,
          "<|end_of_image|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      },
      {
        "scene_id": "eadf0543-57e5-5b6d-83c5-a37d23ab1be7",
        "task": "direct_visual_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "a7ec8dba5c43c015b3a896a0c30707c96ba28397a02495a5f831bd8e93571179",
        "input_ids_sha256": "bb3a986d9999b0a5f5d0637a0918b272015a9c41b03705794d35d7f79f9ebf53",
        "input_sequence_length": 403,
        "image_block_counts": {
          "<|begin_of_image|>": 1,
          "<|image|>": 1,
          "<|end_of_image|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      },
      {
        "scene_id": "3fd755b1-dd1b-5457-ad6b-95cc64412f6e",
        "task": "direct_text_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "ba4f921fc55ab9404d9ed8a9ab0d7bb0fcd8853f970052214d128424efc47900",
        "input_ids_sha256": "e03538a1d11fbfe467fafcef7d8de341c2260d9623bf09f99f3f6341a23f251d",
        "input_sequence_length": 107,
        "image_block_counts": {
          "<|begin_of_image|>": 0,
          "<|image|>": 0,
          "<|end_of_image|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "99b36973-22a5-5066-a088-e42f6d53be88",
        "task": "direct_text_relation",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "8283c39b385a37d7a3c6b88bc70182a1c3506b42e82f0fb51639a6418aca2c5e",
        "input_ids_sha256": "a43fe1eeae7517bdc552ba16c5fae46e0d480b929c2cb7cefe79989d4fc9e070",
        "input_sequence_length": 107,
        "image_block_counts": {
          "<|begin_of_image|>": 0,
          "<|image|>": 0,
          "<|end_of_image|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "54a0893c-6d73-598b-bf48-13068b97c43e",
        "task": "direction_reversal",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "45336ce585e4235f6d39d4cb9228e50a575908fdc0fc987f3678b792f364039b",
        "input_ids_sha256": "e43c7499fb42e488c741f65e3b25f86d4446d7b28830c82338a8013bc5772677",
        "input_sequence_length": 107,
        "image_block_counts": {
          "<|begin_of_image|>": 0,
          "<|image|>": 0,
          "<|end_of_image|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "98b3d118-f942-5026-a34e-7551f55a2974",
        "task": "direction_reversal",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "f41aa9d7868042e89bf7e2a0bbeaf8c473ff6a601e88b079b11dfee52857a2b3",
        "input_ids_sha256": "f0c93eee5d6aeb63fad6fba66b15ac477298251f05998de87535c09f6ca8e436",
        "input_sequence_length": 109,
        "image_block_counts": {
          "<|begin_of_image|>": 0,
          "<|image|>": 0,
          "<|end_of_image|>": 0
        },
        "visual_input_keys": []
      },
      {
        "scene_id": "9e127134-6a18-56ef-9dab-02f97cbb0d34",
        "task": "cross_modal_bridge_binding",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "5dc0a6a0fdd3704d7751b8bc5f05e44de6ad97379b407a81591bbf1a6b23af92",
        "input_ids_sha256": "38988a6182a0c2ec8e892a43f9c75d8e45a6e45888cd60e0dd377117c0f652b5",
        "input_sequence_length": 410,
        "image_block_counts": {
          "<|begin_of_image|>": 1,
          "<|image|>": 1,
          "<|end_of_image|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      },
      {
        "scene_id": "bf1e9113-fc90-5839-b209-82427ab0bc85",
        "task": "cross_modal_bridge_binding",
        "overall_gate": true,
        "gates": {
          "system_instruction_once": true,
          "system_is_plain_text_not_python_repr": true,
          "correct_image_block_count": true,
          "visual_tensor_presence_matches_task": true,
          "query_reference_distinct": true,
          "question_matches_data_row": true,
          "candidate_list_once": true,
          "generation_prompt_at_end": true,
          "no_hidden_answer_or_added_content": true,
          "tokenized_prompt_nonempty": true
        },
        "rendered_prompt_sha256": "a9aeacc5f4f1e79292ec5f828a937c1b4859bbc4e665bb8d7cb6a092e3d215f7",
        "input_ids_sha256": "867801d1dc13908ab738af639843c3624c85831d03441679e8ba6ad98c1f8204",
        "input_sequence_length": 410,
        "image_block_counts": {
          "<|begin_of_image|>": 1,
          "<|image|>": 1,
          "<|end_of_image|>": 1
        },
        "visual_input_keys": [
          "image_grid_thw",
          "pixel_values"
        ]
      }
    ],
    "snapshot_count": 8,
    "failure": null,
    "runtime_seconds": 4.596079302999897,
    "model_forward_passes": 0,
    "files": [
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/cross_modal_bridge_binding_01.json",
        "sha256": "d9bc565ff3f03fdac0f26cf1df8d69bf1d8401a9418018dc23ffaf4fae94792a"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/cross_modal_bridge_binding_02.json",
        "sha256": "7f29939b2bcfdc385fc40ae6ee883028520f611a76657898e2ac3f592bda0cd5"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/direct_text_relation_01.json",
        "sha256": "1cf5cfd5d75f49e5cb25e55f52d82ddbb79b13ec175c150dba378af69635de87"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/direct_text_relation_02.json",
        "sha256": "4e421a8a3ed0aa2d74be89d2010c9293f02f9ebec2b01c1381e448d51774d667"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/direct_visual_relation_01.json",
        "sha256": "d996c3321ec8e0fd34a8dd541df5e4bcbfa3019e5074c47aef42178b572584a0"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/direct_visual_relation_02.json",
        "sha256": "44875f497c2416858ed2455dade0791c87225eefcb7db6e783005ecd7c54c266"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/direction_reversal_01.json",
        "sha256": "cab5c89050b330b9dba067fc950dcb0db84f7ef48eba51690f5c4e7315a3d6ce"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/direction_reversal_02.json",
        "sha256": "a41acaaf1466a325d3822d56a34f1bb8a20edef877ef43fbd5aded81af5934a0"
      },
      {
        "path": "artifacts/atomic_v2/renderer_snapshots/glm4_1v_9b/repair_01/stderr.log",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/cross_modal_bridge_binding_01.txt",
        "sha256": "5dc0a6a0fdd3704d7751b8bc5f05e44de6ad97379b407a81591bbf1a6b23af92"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/cross_modal_bridge_binding_02.txt",
        "sha256": "a9aeacc5f4f1e79292ec5f828a937c1b4859bbc4e665bb8d7cb6a092e3d215f7"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/direct_text_relation_01.txt",
        "sha256": "ba4f921fc55ab9404d9ed8a9ab0d7bb0fcd8853f970052214d128424efc47900"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/direct_text_relation_02.txt",
        "sha256": "8283c39b385a37d7a3c6b88bc70182a1c3506b42e82f0fb51639a6418aca2c5e"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/direct_visual_relation_01.txt",
        "sha256": "2965591ff19607b946a2f901f4eb533b426943d1776f82e29aa690703c61ba7d"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/direct_visual_relation_02.txt",
        "sha256": "a7ec8dba5c43c015b3a896a0c30707c96ba28397a02495a5f831bd8e93571179"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/direction_reversal_01.txt",
        "sha256": "45336ce585e4235f6d39d4cb9228e50a575908fdc0fc987f3678b792f364039b"
      },
      {
        "path": "tests/snapshots/glm_atomic_prompts/direction_reversal_02.txt",
        "sha256": "f41aa9d7868042e89bf7e2a0bbeaf8c473ff6a601e88b079b11dfee52857a2b3"
      }
    ],
    "created_at": "2026-09-10T12:29:01.672738+00:00",
    "hostname": "10-60-165-203",
    "platform": "Linux-5.15.0-113-generic-x86_64-with-glibc2.39",
    "python": "3.11.16 (main, Sep  1 2026, 14:18:37) [Clang 22.1.3 ]",
    "executable": "/workspace/vlm-synergy-qualification/envs/glm/.venv/bin/python",
    "pid": 727,
    "git_commit": "0ec86bbbe00f9bc5b579040fde49dbd527a0c000",
    "git_branch": "codex/atomic-protocol-v2-clean-rerun"
  }
}
```
