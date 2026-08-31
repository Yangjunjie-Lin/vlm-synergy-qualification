from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import torch

from capability_gate.recovery.adapters import QwenRecoveryAdapter


class DummyTokenizer:
    eos_token_id = 2
    mapping: ClassVar[dict[str, int]] = {
        " north": 10,
        " south": 11,
        " east": 12,
        " west": 13,
    }

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [self.mapping[value]]


class DummyModel:
    generation_config = SimpleNamespace(eos_token_id=2)

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        logits = torch.zeros((1, input_ids.shape[1], 32), dtype=torch.float32)
        logits[:, :, 10] = 1.0
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        prompt = kwargs["input_ids"]
        suffix = torch.tensor([[10, 2]], dtype=prompt.dtype)
        return torch.cat([prompt, suffix], dim=1)


def _dummy_adapter() -> QwenRecoveryAdapter:
    adapter = QwenRecoveryAdapter()
    adapter.torch = torch
    adapter.model = DummyModel()
    adapter.processor = SimpleNamespace(tokenizer=DummyTokenizer())
    adapter._to_model = lambda values: dict(values)  # type: ignore[method-assign]
    adapter._encode = lambda _system, _user, _image: (  # type: ignore[method-assign]
        "rendered",
        {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        },
    )
    return adapter


def test_candidate_prefix_is_preserved_and_candidate_tokens_are_nonempty() -> None:
    adapter = _dummy_adapter()
    prompt = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }
    full = adapter._append_candidate(prompt, adapter._candidate_ids(" ", "north"))
    assert full["input_ids"][0, :3].tolist() == [1, 2, 3]
    assert full["input_ids"][0, 3:].tolist() == [10]


def test_cll_is_finite_generation_is_constrained_and_rerun_is_deterministic() -> None:
    adapter = _dummy_adapter()
    kwargs = {
        "system": "system",
        "user": "user",
        "image_path": None,
        "candidates": ["north", "south", "east", "west"],
        "target": "north",
    }
    first = adapter.score_and_generate(**kwargs)
    second = adapter.score_and_generate(**kwargs)
    assert all(score["candidate_token_count"] > 0 for score in first["candidate_scores"])
    assert all(
        torch.isfinite(torch.tensor(score["normalized_log_likelihood"]))
        for score in first["candidate_scores"]
    )
    assert first["constrained_answer"] in kwargs["candidates"]
    assert first["candidate_ranking"] == second["candidate_ranking"]
    assert first["constrained_answer"] == second["constrained_answer"]
