#!/usr/bin/env python3
"""Deterministic contract test for the vendored SVD-LLM KV-cache attention path."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "third_party" / "SVD-LLM" / "component" / "svd_llama_kvcache.py"


def install_transformers_stub() -> None:
    """Provide the minimal transformers API that the vendored module imports."""
    if "transformers" in sys.modules:
        return

    transformers_mod = types.ModuleType("transformers")
    activations_mod = types.ModuleType("transformers.activations")
    activations_mod.ACT2FN = {
        "gelu": torch.nn.functional.gelu,
        "relu": torch.nn.functional.relu,
        "silu": torch.nn.functional.silu,
    }

    utils_mod = types.ModuleType("transformers.utils")
    utils_mod.logging = types.SimpleNamespace(get_logger=logging.getLogger)

    class LlamaConfig:
        def __init__(
            self,
            *,
            hidden_size: int,
            intermediate_size: int,
            num_hidden_layers: int,
            num_attention_heads: int,
            max_position_embeddings: int,
            hidden_act: str = "silu",
        ) -> None:
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.max_position_embeddings = max_position_embeddings
            self.hidden_act = hidden_act

    transformers_mod.LlamaConfig = LlamaConfig
    transformers_mod.activations = activations_mod
    transformers_mod.utils = utils_mod

    sys.modules["transformers"] = transformers_mod
    sys.modules["transformers.activations"] = activations_mod
    sys.modules["transformers.utils"] = utils_mod


def load_target_module():
    install_transformers_stub()
    spec = importlib.util.spec_from_file_location("svd_llama_kvcache_local", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module from {TARGET}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_causal_mask(q_len: int, kv_len: int, past_len: int = 0) -> torch.Tensor:
    mask = torch.zeros((1, 1, q_len, kv_len), dtype=torch.float32)
    blocked = torch.finfo(mask.dtype).min
    for query_index in range(q_len):
        max_visible = past_len + query_index + 1
        if max_visible < kv_len:
            mask[:, :, query_index, max_visible:] = blocked
    return mask


def initialize_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters(), start=1):
            torch.manual_seed(index)
            parameter.copy_(torch.randn_like(parameter))


def assert_cached_decode_matches_full_prefix(module, config, compression_ratio: float) -> None:
    torch.manual_seed(7)
    hidden_states = torch.randn(1, 4, config.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)

    full_output, _, _ = module(
        hidden_states,
        attention_mask=build_causal_mask(q_len=4, kv_len=4),
        position_ids=position_ids,
        use_cache=False,
    )

    cached_outputs = []
    past_key_value = None
    for step in range(hidden_states.shape[1]):
        step_output, _, past_key_value = module(
            hidden_states[:, step : step + 1, :],
            attention_mask=build_causal_mask(q_len=1, kv_len=step + 1, past_len=step),
            position_ids=position_ids[:, step : step + 1],
            past_key_value=past_key_value,
            use_cache=True,
        )
        cached_outputs.append(step_output)

        if compression_ratio != 1:
            low_rank = int(config.hidden_size * compression_ratio / 2)
            expected_shape = (1, step + 1, low_rank)
            assert past_key_value[0].shape == expected_shape, (
                f"compressed key cache shape mismatch at step {step}: "
                f"expected {expected_shape}, got {tuple(past_key_value[0].shape)}"
            )
            assert past_key_value[1].shape == expected_shape, (
                f"compressed value cache shape mismatch at step {step}: "
                f"expected {expected_shape}, got {tuple(past_key_value[1].shape)}"
            )
        else:
            expected_shape = (1, config.num_attention_heads, step + 1, config.hidden_size // config.num_attention_heads)
            assert past_key_value[0].shape == expected_shape, (
                f"full key cache shape mismatch at step {step}: "
                f"expected {expected_shape}, got {tuple(past_key_value[0].shape)}"
            )
            assert past_key_value[1].shape == expected_shape, (
                f"full value cache shape mismatch at step {step}: "
                f"expected {expected_shape}, got {tuple(past_key_value[1].shape)}"
            )

    cached_output = torch.cat(cached_outputs, dim=1)
    torch.testing.assert_close(
        cached_output,
        full_output,
        atol=1e-5,
        rtol=1e-5,
        msg=f"cached decode diverged from full-prefix causal attention for compression_ratio={compression_ratio}",
    )


def main() -> None:
    module = load_target_module()
    config = sys.modules["transformers"].LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
    )

    for compression_ratio in (1.0, 0.5):
        attention = module.SVD_LlamaAttention(config, compression_ratio=compression_ratio)
        attention.eval()
        initialize_module(attention)
        assert_cached_decode_matches_full_prefix(
            module=attention,
            config=config,
            compression_ratio=compression_ratio,
        )

    print(
        "PASS [43-svd-llm-kv-cache]: cached decode matches the causal full-prefix path "
        "and past_key_value shapes stay stable for compressed and uncompressed attention."
    )


if __name__ == "__main__":
    main()
