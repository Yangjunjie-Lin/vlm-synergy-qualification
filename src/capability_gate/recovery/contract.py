from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

CONTRACT_VERSION = 1
OPERATIONS = frozenset(
    {"dependency_preflight", "processor_preflight", "load_preflight", "score_and_generate"}
)


@dataclass(frozen=True)
class WorkerRequest:
    model_key: str
    model_revision: str
    processor_revision: str
    image_path: str | None
    prompt: Mapping[str, str]
    candidates: Sequence[str]
    target: str
    operation: str
    request_id: str | None = None
    schema_version: int = CONTRACT_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerRequest:
        required = {
            "model_key",
            "model_revision",
            "processor_revision",
            "image_path",
            "prompt",
            "candidates",
            "target",
            "operation",
        }
        missing = required - value.keys()
        unknown = value.keys() - required - {"request_id", "schema_version"}
        if missing or unknown:
            raise ValueError(f"invalid worker request fields: missing={missing}, unknown={unknown}")
        request = cls(**value)
        request.validate()
        return request

    def validate(self) -> None:
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError(f"unsupported worker contract version: {self.schema_version}")
        if len(self.model_revision) != 40 or len(self.processor_revision) != 40:
            raise ValueError("model and processor revisions must be exact 40-character SHAs")
        if self.operation not in OPERATIONS:
            raise ValueError(f"unsupported worker operation: {self.operation}")
        if set(self.prompt) != {"system", "user"}:
            raise ValueError("prompt must contain exactly system and user")
        if len(self.candidates) < 2 or len(set(self.candidates)) != len(self.candidates):
            raise ValueError("candidates must contain at least two unique answers")
        if self.target not in self.candidates:
            raise ValueError("target must belong to candidates")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates"] = list(self.candidates)
        result["prompt"] = dict(self.prompt)
        return result


RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "status",
        "model_metadata",
        "candidate_scores",
        "constrained_answer",
        "runtime",
        "peak_vram",
        "resolved_device_map",
        "artifact_hash",
        "error_class",
        "traceback",
    }
)


def validate_worker_response(value: Mapping[str, Any]) -> None:
    if set(value) != RESPONSE_FIELDS:
        raise ValueError(f"invalid worker response fields: {set(value) ^ RESPONSE_FIELDS}")
    if value["schema_version"] != CONTRACT_VERSION:
        raise ValueError("unsupported worker response version")
