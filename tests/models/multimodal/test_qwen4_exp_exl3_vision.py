# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint layout routing with actual model mapping helpers."""

from types import SimpleNamespace

import torch

from vllm.models.qwen4_exp.nvidia.model import (
    _EXL3_VISION_WEIGHTS_MAPPER,
    _qwen4_exl3_vision_quant_config,
)


def test_fused_exl3_vision_keeps_its_original_mapping():
    config = SimpleNamespace(
        get_name=lambda: "exl3",
        non_routed_exl3={"layers": {"visual.blocks.0.attn.qkv": {"bits": 6}}},
    )
    assert _qwen4_exl3_vision_quant_config(config) is config


def test_split_exl3_vision_maps_spec_and_all_qkv_shards():
    config = SimpleNamespace(
        get_name=lambda: "exl3",
        non_routed_exl3={"layers": {"visual.blocks.0.attn.qkv_proj": {"bits": 6}}},
    )
    mapped = _qwen4_exl3_vision_quant_config(config)
    assert mapped.non_routed_exl3["layers"]["visual.blocks.0.attn.qkv"] == {"bits": 6}
    assert "visual.blocks.0.attn.qkv" not in config.non_routed_exl3["layers"]
    weights = [
        (f"attn.{part}_proj.bias", torch.tensor([i]))
        for i, part in enumerate(("q", "k", "v"))
    ]
    result = list(_EXL3_VISION_WEIGHTS_MAPPER.apply(weights))
    assert [name for name, _ in result] == ["attn.qkv.bias"] * 3
    assert [tensor.shard_id for _, tensor in result] == ["q", "k", "v"]
    assert [tensor.item() for _, tensor in result] == [0, 1, 2]
