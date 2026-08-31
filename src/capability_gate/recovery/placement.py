from __future__ import annotations

from typing import Any


def single_gpu_4bit_model_kwargs(
    *,
    torch: Any,
    revision: str,
    trust_remote_code: bool,
    attn_implementation: str,
) -> dict[str, Any]:
    """Build the frozen Q1 placement: quantized model wholly on one CUDA GPU."""

    from transformers import BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    result = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "quantization_config": quantization,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
        "output_loading_info": True,
    }
    validate_no_auto_offload(result)
    return result


def validate_no_auto_offload(model_kwargs: dict[str, Any]) -> None:
    """Reject the unverified historical auto GPU/CPU/disk offload mechanism."""

    if model_kwargs.get("device_map") != {"": 0}:
        raise ValueError("recovery Q1 requires explicit whole-model placement on CUDA device 0")
    forbidden = {"max_memory", "offload_folder", "offload_state_dict"} & model_kwargs.keys()
    if forbidden:
        raise ValueError(f"recovery Q1 forbids auto-offload arguments: {sorted(forbidden)}")
    quantization = model_kwargs.get("quantization_config")
    if quantization is None or not getattr(quantization, "load_in_4bit", False):
        raise ValueError("recovery Q1 requires explicit 4-bit quantization")
    if getattr(quantization, "llm_int8_enable_fp32_cpu_offload", False):
        raise ValueError("recovery Q1 forbids the historical CPU-offload flag")

