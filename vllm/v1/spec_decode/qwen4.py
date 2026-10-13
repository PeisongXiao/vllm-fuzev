# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, MambaSpec
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer


class Qwen4MTPProposer(Step3p5MTPProposer):
    """Reuse engine-owned multi-group MTP metadata for Qwen's hybrid drafter."""

    def _create_draft_vllm_config(self) -> VllmConfig:
        return SpecDecodeBaseProposer._create_draft_vllm_config(self)

    def _maybe_share_lm_head(self, target_language_model: torch.nn.Module) -> None:
        SpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)

    def model_returns_tuple(self) -> bool:
        return True

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)
        # Position/slot arithmetic belongs to attention, not recurrent states.
        primary = next(
            (
                group
                for group in self.draft_attn_groups
                if group.backend.get_name() == "QWEN4_EXP_QSA_TRITON"
            ),
            None,
        )
        if primary is None:
            primary = next(
                (
                    group
                    for group in self.draft_attn_groups
                    if not isinstance(
                        group.get_metadata_builder().kv_cache_spec,
                        (MambaSpec, CircularBufferSpec),
                    )
                ),
                None,
            )
        if primary is None:
            return
        self.kv_cache_gid = primary.kv_cache_group_id
        self.block_size = (
            kernel_block_sizes[self.kv_cache_gid]
            if kernel_block_sizes is not None
            else primary.get_metadata_builder().kv_cache_spec.block_size
        )
