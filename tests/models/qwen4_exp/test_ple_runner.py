# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _CPUBuffer:
    """Separate CPU storage for the runner's pinned and device buffer roles."""

    def __init__(self, owner, *shape, dtype):
        self.owner = owner
        self.np = np.full(shape, -777, dtype=np.int32)
        self.gpu = torch.empty(shape, dtype=dtype, device="cpu")

    def copy_to_gpu(self):
        assert self.owner.input_prep_depth > 0
        self.gpu.copy_(torch.from_numpy(self.np))
        return self.gpu


def _runner(
    *,
    architecture="Qwen4ExpForConditionalGeneration",
    ple=(1,),
    asynchronous=False,
    ngram_size=4,
):
    owner = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[architecture] if architecture else [],
            hf_text_config=SimpleNamespace(
                ple_layer_ids=ple,
                ngram_size=ngram_size,
                eos_token_id=99,
            ),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=asynchronous),
        max_num_reqs=4,
        qwen4_ngram_context=None,
        qwen4_ple_query_start_loc=None,
        qwen4_ngram_eos_token_id=0,
        lookback_token_ids=None,
        is_pooling_model=False,
        input_prep_depth=0,
        allocations=[],
    )

    def make_buffer(*shape, dtype, numpy=True):
        assert numpy
        assert dtype == torch.int32
        buffer = _CPUBuffer(owner, *shape, dtype=dtype)
        owner.allocations.append(buffer)
        return buffer

    @contextmanager
    def synchronize_input_prep():
        owner.input_prep_depth += 1
        try:
            yield
        finally:
            owner.input_prep_depth -= 1

    owner._make_buffer = make_buffer
    owner.synchronize_input_prep = synchronize_input_prep
    owner._prepare_qwen4_ple_kwargs = MethodType(
        GPUModelRunner._prepare_qwen4_ple_kwargs,
        owner,
    )
    GPUModelRunner._init_qwen4_ple_buffers(owner)
    return owner


def _batch(owner, rows, starts, boundaries, *, prompt_lengths=None):
    num_reqs = len(starts)
    token_ids = np.asarray(rows, dtype=np.int32)
    owner.input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        token_ids_cpu=token_ids,
        num_computed_tokens_cpu=np.asarray(starts, dtype=np.int32),
        num_tokens_no_spec=np.full(num_reqs, token_ids.shape[1] if num_reqs else 0),
        num_prompt_tokens=np.asarray(
            prompt_lengths if prompt_lengths is not None else starts,
            dtype=np.int32,
        ),
        req_ids=[f"request-{i}" for i in range(num_reqs)],
    )
    owner.query_start_loc = SimpleNamespace(
        np=np.asarray(boundaries, dtype=np.int32),
    )


def _assert_kwargs(kwargs, context, boundaries):
    assert set(kwargs) == {"ngram_context", "query_start_loc"}
    assert kwargs["ngram_context"].device.type == "cpu"
    assert kwargs["query_start_loc"].device.type == "cpu"
    assert kwargs["ngram_context"].dtype == torch.int32
    assert kwargs["query_start_loc"].dtype == torch.int32
    assert kwargs["ngram_context"].tolist() == context
    assert kwargs["query_start_loc"].tolist() == boundaries


@pytest.mark.parametrize(
    ("architecture", "ple"),
    [
        ("OtherForCausalLM", (1,)),
        (None, (1,)),
        ("Qwen4ExpForConditionalGeneration", ()),
        ("Qwen4ExpForConditionalGeneration", None),
    ],
)
def test_disabled_configuration_allocates_nothing(architecture, ple):
    owner = _runner(architecture=architecture, ple=ple, asynchronous=True)
    assert owner.allocations == []
    assert owner._prepare_qwen4_ple_kwargs() == {}


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpForCausalLM",
    ],
)
def test_enabled_configuration_owns_fixed_int32_buffers(architecture):
    owner = _runner(architecture=architecture)
    assert owner.qwen4_ngram_context.np.shape == (4, 3)
    assert owner.qwen4_ple_query_start_loc.np.shape == (5,)
    assert owner.qwen4_ngram_eos_token_id == 99


def test_live_history_is_chronological_and_eos_left_padded():
    owner = _runner()
    _batch(owner, [[11, 12, 13, 14]] * 4, [0, 1, 2, 4], [0, 2, 3, 4, 5])
    _assert_kwargs(
        owner._prepare_qwen4_ple_kwargs(),
        [
            [99, 99, 99],
            [99, 99, 11],
            [99, 11, 12],
            [12, 13, 14],
        ],
        [0, 2, 3, 4, 5],
    )
    assert owner.input_prep_depth == 0


def test_live_history_includes_committed_decodes():
    owner = _runner()
    _batch(owner, [[11, 12, 21, 22]] * 2, [3, 4], [0, 1, 2], prompt_lengths=[2, 2])
    _assert_kwargs(
        owner._prepare_qwen4_ple_kwargs(),
        [
            [11, 12, 21],
            [12, 21, 22],
            [99] * 3,
            [99] * 3,
        ],
        [0, 1, 2, 2, 2],
    )


def test_new_cached_prefix_at_32k_uses_retained_token_history():
    owner = _runner()
    tokens = np.arange(32772, dtype=np.int32)[None, :]
    _batch(owner, tokens, [32768], [0, 2], prompt_lengths=[32772])
    _assert_kwargs(
        owner._prepare_qwen4_ple_kwargs(),
        [
            [32765, 32766, 32767],
            [99] * 3,
            [99] * 3,
            [99] * 3,
        ],
        [0, 2, 2, 2, 2],
    )


def test_current_batch_order_and_reused_slots_do_not_leak_history():
    owner = _runner()
    _batch(owner, [[21, 22, 23, 24], [11, 99, 13, 14]], [3, 3], [0, 2, 3])
    owner.input_batch.req_ids = ["B", "A"]
    _assert_kwargs(
        owner._prepare_qwen4_ple_kwargs(),
        [
            [21, 22, 23],
            [11, 99, 13],
            [99] * 3,
            [99] * 3,
        ],
        [0, 2, 3, 3, 3],
    )
    _batch(owner, [[31, 32, 33, 34]], [1], [0, 1])
    owner.input_batch.req_ids = ["new-request"]
    _assert_kwargs(
        owner._prepare_qwen4_ple_kwargs(),
        [
            [99, 99, 31],
            [99] * 3,
            [99] * 3,
            [99] * 3,
        ],
        [0, 1, 1, 1, 1],
    )


def test_live_trailing_zero_length_requests_keep_terminal_boundary():
    owner = _runner()
    _batch(owner, [[11, 12, 13], [21, 22, 23]], [3, 3], [0, 2, 2])
    _assert_kwargs(
        owner._prepare_qwen4_ple_kwargs(),
        [
            [11, 12, 13],
            [21, 22, 23],
            [99] * 3,
            [99] * 3,
        ],
        [0, 2, 2, 2, 2],
    )


@pytest.mark.parametrize(
    ("counts", "padded", "boundaries"),
    [
        ([1, 1, 3], 5, [0, 1, 2, 5, 5]),
        ([2, 0, 3], 8, [0, 2, 2, 8, 8]),
        ([1, 2, 1, 1], 7, [0, 1, 3, 4, 7]),
        ([2, 3], None, [0, 2, 5, 5, 5]),
    ],
)
def test_dummy_uses_supplied_counts_and_padding_last_active(counts, padded, boundaries):
    owner = _runner()
    kwargs = owner._prepare_qwen4_ple_kwargs(
        num_scheduled_tokens=np.asarray(counts, dtype=np.int32),
        num_tokens=padded,
    )
    _assert_kwargs(kwargs, [[99] * 3 for _ in range(4)], boundaries)


@pytest.mark.parametrize("dummy", [False, True])
def test_no_active_requests_clear_context_and_boundaries(dummy):
    owner = _runner()
    _batch(owner, [], [], [0])
    if dummy:
        kwargs = owner._prepare_qwen4_ple_kwargs(
            num_scheduled_tokens=np.array([], dtype=np.int32),
            num_tokens=0,
        )
    else:
        kwargs = owner._prepare_qwen4_ple_kwargs()
    _assert_kwargs(kwargs, [[99] * 3 for _ in range(4)], [0] * 5)


def test_buffer_addresses_and_shapes_are_stable_across_live_and_dummy():
    owner = _runner()
    _batch(owner, [[11, 12, 13, 14]], [4], [0, 1])
    live = owner._prepare_qwen4_ple_kwargs()
    addresses = {key: value.data_ptr() for key, value in live.items()}
    dummy = owner._prepare_qwen4_ple_kwargs(
        num_scheduled_tokens=np.array([1, 1, 3]),
        num_tokens=8,
    )
    _assert_kwargs(dummy, [[99] * 3 for _ in range(4)], [0, 1, 2, 8, 8])
    for key, value in dummy.items():
        assert value is live[key]
        assert value.data_ptr() == addresses[key]


def test_init_model_kwargs_prepares_live_ple_but_skips_dummy_state():
    owner = _runner()
    _batch(owner, [[11, 12, 13, 14]], [3], [0, 1])
    _assert_kwargs(
        GPUModelRunner._init_model_kwargs(owner),
        [
            [11, 12, 13],
            [99] * 3,
            [99] * 3,
            [99] * 3,
        ],
        [0, 1, 1, 1, 1],
    )
    assert GPUModelRunner._init_model_kwargs(owner, num_reqs=0) == {}


def test_multimodal_preparation_preserves_raw_ids_with_embeddings():
    owner = SimpleNamespace(
        model=SimpleNamespace(requires_raw_input_tokens=True),
        input_ids=SimpleNamespace(gpu=torch.tensor([11, 99, 13], dtype=torch.int32)),
        inputs_embeds=SimpleNamespace(gpu=torch.ones(3, 2, device="cpu")),
    )
    input_ids, inputs_embeds = GPUModelRunner._prepare_mm_inputs(owner, 2)
    assert input_ids.tolist() == [11, 99]
    assert input_ids.data_ptr() == owner.input_ids.gpu.data_ptr()
    assert inputs_embeds.shape == (2, 2)
    assert inputs_embeds.data_ptr() == owner.inputs_embeds.gpu.data_ptr()
