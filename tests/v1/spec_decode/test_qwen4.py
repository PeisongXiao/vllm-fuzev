# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import torch

from vllm.v1.kv_cache_interface import CircularBufferSpec
from vllm.v1.spec_decode import step3p5
from vllm.v1.spec_decode.qwen4 import Qwen4MTPProposer


class Group:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = layer_names
        self.kv_cache_group_id = kv_cache_group_id
        self.builder = NS(
            kv_cache_spec=kv_cache_spec,
            build_for_drafting=lambda common_attn_metadata, draft_index: (
                common_attn_metadata
            ),
        )

    def create_metadata_builders(self, *args, **kwargs):
        pass

    def get_metadata_builder(self):
        return self.builder


def test_qwen4_hybrid_uses_owner_group_tables_and_attention_kernel_size(monkeypatch):
    names = {"a.indexer", "z.attention"}
    layers = {
        name: NS(
            get_attn_backend=lambda name=name: NS(
                full_cls_name=lambda: "backend",
                get_name=lambda: (
                    "QWEN4_EXP_QSA_TRITON" if name == "z.attention" else "STATE"
                ),
            )
        )
        for name in names
    }
    monkeypatch.setattr(step3p5, "get_layers_from_vllm_config", lambda *a, **k: layers)
    monkeypatch.setattr(step3p5, "AttentionGroup", Group)
    proposer = Qwen4MTPProposer.__new__(Qwen4MTPProposer)
    proposer.vllm_config, proposer.device = None, "cpu"
    proposer._draft_attn_layer_names = names
    proposer._per_group_block_tables = {}
    proposer._per_group_slot_mappings = {}
    config = NS(
        kv_cache_groups=[
            NS(layer_names=["z.attention"], kv_cache_spec=NS(block_size=256)),
            NS(
                layer_names=["a.indexer"],
                kv_cache_spec=CircularBufferSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=16,
                    head_size_v=0,
                    dtype=torch.bfloat16,
                ),
            ),
        ]
    )
    proposer.initialize_attn_backend(config, [64, 8])
    assert proposer.kv_cache_gid == 0
    assert proposer.block_size == 64
    assert len(proposer.draft_attn_groups) == 2
    tables = {
        0: torch.tensor([[11, 12], [13, 14]]),
        1: torch.tensor([[23, 24, 25], [26, 27, 28]]),
    }
    slots = {0: torch.tensor([1, 2]), 1: torch.tensor([3, 4])}
    for gid in tables:
        proposer.set_per_group_attn_metadata(gid, tables[gid], slots[gid])
    common = NS(
        num_reqs=1,
        num_actual_tokens=1,
        block_table_tensor=tables[0],
        slot_mapping=slots[0],
    )
    _, metadata = proposer.build_per_group_and_layer_attn_metadata(common, 2)
    assert torch.equal(metadata["z.attention"].block_table_tensor, tables[0][:1])
    assert torch.equal(metadata["a.indexer"].block_table_tensor, tables[1][:1])
    assert torch.equal(metadata["z.attention"].slot_mapping, slots[0][:1])
    assert torch.equal(metadata["a.indexer"].slot_mapping, slots[1][:1])
    assert common.block_table_tensor is tables[0]
    assert proposer.model_returns_tuple()
