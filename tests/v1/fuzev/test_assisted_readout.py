# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual readout/transport methods, CPU tensors and a counted synthetic model.

No runner/model initialization, weights, datasets, kernels or live engine.
Run with --confcutdir=tests/v1/fuzev to exclude model/GPU fixtures.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu_input_batch as input_batch_module
import vllm.v1.worker.gpu_model_runner as runner_module
from vllm.config import CUDAGraphMode
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.fuzev import fuzev_enabled
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
)


class CountedModel:
    def __init__(self):
        self.calls = []

    def compute_logits(self, hidden):
        self.calls.append(hidden.clone())
        return (
            hidden[:, :1].float()
            + torch.arange(32, dtype=torch.float32, device=hidden.device)[None]
        )


@pytest.fixture(autouse=True)
def cpu_readout(monkeypatch):
    monkeypatch.setattr(
        runner_module, "current_platform", NS(is_cuda_alike=lambda: False)
    )


def test_fuzev_configuration_preserves_generation_settings():
    method = "mtp"
    config = NS(
        additional_config={"fuzev": {"enabled": True}},
        model_config=NS(runner_type="generate", hf_text_config=NS(hidden_size=8)),
        parallel_config=NS(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        scheduler_config=NS(async_scheduling=False),
        use_v2_model_runner=False,
        speculative_config=None if method is None else NS(method=method),
    )
    original = config.speculative_config
    assert fuzev_enabled(config)
    assert config.speculative_config is original
    config.additional_config["fuzev"]["enabled"] = False
    assert not fuzev_enabled(config)


def worker(labels, *, finished=True):
    params = [
        PoolingParams(
            task="fuzev_features",
            extra_kwargs=(None if ids is None else {"baseline_token_ids": ids}),
        )
        for ids in labels
    ]
    count = len(params)
    tokens = np.tile(np.array([4, 7, 9], dtype=np.int32), (count, 1))
    model = CountedModel()
    runner = NS(
        input_batch=NS(
            num_reqs=count,
            get_pooling_params=lambda: params,
            vocab_size=32,
            num_prompt_tokens=np.full(count, 3, dtype=np.int32),
            num_tokens_no_spec=np.full(count, 3, dtype=np.int32),
            token_ids_cpu=tokens,
            req_ids=[f"readout-{i}" for i in range(count)],
            req_id_to_index={f"readout-{i}": i for i in range(count)},
        ),
        query_start_loc=NS(gpu=torch.arange(count + 1) * 2),
        optimistic_seq_lens_cpu=torch.full((count,), 3 if finished else 2),
        model=model,
        _sync_device=lambda: None,
    )
    hidden = torch.arange(count * 2 * 8, dtype=torch.float32).reshape(-1, 8)
    return runner, hidden


@pytest.mark.parametrize("count", [2, 3, 17])
def test_original_model_projects_query_once_and_never_candidates(count):
    labels = list(range(count - 1, -1, -1))
    runner, hidden = worker([labels] + [None] * count)
    output = GPUModelRunner._fuzev_readout(runner, hidden, None)
    assert len(runner.model.calls) == 1
    assert torch.equal(runner.model.calls[0], hidden[1:2])
    assert output.sampled_token_ids == []
    query = output.pooler_output[0]
    assert query.shape == (8 + count,) and query.dtype == torch.float32
    assert torch.equal(query[:8], hidden[1])
    assert torch.equal(query[8:], hidden[1, 0] + torch.tensor(labels))
    for index, value in enumerate(output.pooler_output[1:], 1):
        assert value.shape == (8,) and torch.equal(value, hidden[index * 2 + 1])
    hidden.fill_(-999)
    runner.input_batch.token_ids_cpu.fill(999)
    labels[:] = [31, 30]
    assert query[0] != -999


def test_feature_only_batch_and_unfinished_query_have_zero_projections():
    for labels, finished in [([None, []], True), ([[3, 1]], False)]:
        runner, hidden = worker(labels, finished=finished)
        output = GPUModelRunner._fuzev_readout(runner, hidden, None)
        assert not runner.model.calls and not output.sampled_token_ids
        if finished:
            assert all(value.shape == (8,) for value in output.pooler_output)
        else:
            assert output.pooler_output == [None]


def test_finished_queries_share_projection_and_keep_label_order():
    labels = [[7, 2, 7], None, [1], [], [3, 0]]
    runner, hidden = worker(labels)
    runner.optimistic_seq_lens_cpu[-1] = 2
    output = GPUModelRunner._fuzev_readout(runner, hidden, None)
    assert len(runner.model.calls) == 1
    torch.testing.assert_close(runner.model.calls[0], hidden[[1, 5]])
    for index, ids in enumerate(labels[:-1]):
        expected = hidden[index * 2 + 1]
        if ids:
            expected = torch.cat((expected, expected[0] + torch.tensor(ids)))
        torch.testing.assert_close(output.pooler_output[index], expected)
    assert output.pooler_output[-1] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA async copy")
def test_async_readout_returns_finished_features_and_labels(monkeypatch):
    monkeypatch.setattr(
        runner_module, "current_platform", NS(is_cuda_alike=lambda: True)
    )
    runner, hidden = worker([[7, 2], None, [1]])
    runner.optimistic_seq_lens_cpu[-1] = 2
    runner.query_start_loc.gpu = runner.query_start_loc.gpu.cuda()
    stream = torch.cuda.Stream()
    runner._get_or_create_async_output_copy_stream = lambda: stream
    output = GPUModelRunner._fuzev_readout(runner, hidden.cuda(), None).get_output()
    torch.testing.assert_close(
        output.pooler_output[0],
        torch.cat((hidden[1], hidden[1, 0] + torch.tensor([7, 2]))),
    )
    torch.testing.assert_close(output.pooler_output[1], hidden[3])
    assert output.pooler_output[-1] is None


@pytest.mark.parametrize("enabled", [False, True])
def test_generation_does_not_enter_readout_validation(enabled):
    runner = NS(fuzev_enabled=enabled, input_batch=NS(pooling_params={}))
    assert not GPUModelRunner._prepare_fuzev_readout(runner, object())


def test_readout_does_not_publish_or_clear_pending_generation_drafts():
    drafts, probabilities = [[2, 7, 4]], torch.ones(1, 3, 32)
    runner = NS(
        _fuzev_readout_batch=True,
        num_spec_tokens=3,
        _draft_token_req_ids=["generation"],
        _draft_token_ids=drafts,
        _draft_probs=probabilities,
        _draft_prob_req_ids=["generation"],
        _get_draft_token_ids_cpu=lambda: (drafts, ["generation"]),
    )
    assert GPUModelRunner.take_draft_token_ids(runner) is None
    assert runner._draft_token_ids is drafts
    assert runner._draft_probs is probabilities
    assert runner._draft_prob_req_ids == ["generation"]
    runner._fuzev_readout_batch = False
    output = GPUModelRunner.take_draft_token_ids(runner)
    assert output.req_ids == ["generation"] and output.draft_token_ids == drafts


def test_real_input_batch_restores_generation_state_after_readout(monkeypatch):
    monkeypatch.setattr(runner_module, "get_pp_group", lambda: NS(is_last_rank=True))
    monkeypatch.setattr(input_batch_module, "PIN_MEMORY", False)
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.fuzev_enabled = True
    runner._fuzev_paused_accepted_tokens = {}
    runner.num_accepted_tokens_event = NS(synchronize=lambda: None)
    runner.input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        vocab_size=32,
        block_sizes=[4],
        kernel_block_sizes=[4],
        max_num_blocks_per_req=[4],
        num_spec_tokens=3,
    )
    generation = CachedRequestState(
        req_id="generation",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0),
        generator=None,
        block_ids=([1, 2],),
        num_computed_tokens=5,
        output_token_ids=[8, 9, 10],
    )
    runner.requests = {generation.req_id: generation}
    runner.input_batch.add_request(generation)
    runner.input_batch.refresh_metadata()
    runner.input_batch.num_accepted_tokens_cpu[0] = 3
    runner.num_prompt_logprobs = {}
    runner.encoder_cache = {}
    runner.late_interaction_runner = NS(
        on_requests_finished=lambda _: None, register_request=lambda *_: None
    )
    runner.speculative_config = NS(use_ngram_gpu=lambda: False)
    runner.use_async_spec_decode = runner.use_async_scheduling = False
    runner.is_pooling_model = runner.uses_mrope = False
    runner._may_reorder_batch = lambda _: None
    scheduled = NS(
        fuzev_readout_lane=True,
        finished_req_ids=set(),
        new_block_ids_to_zero=[],
        kv_cache_block_copies=None,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={"readout": 3},
        scheduled_spec_decode_tokens={},
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        scheduled_new_reqs=[
            NewRequestData(
                req_id="readout",
                prompt_token_ids=[4, 5, 6],
                mm_features=[],
                sampling_params=None,
                pooling_params=PoolingParams(task="fuzev_features"),
                block_ids=([3],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ],
    )
    runner._update_states(scheduled)
    assert runner.input_batch.req_ids == ["readout"]
    assert runner.input_batch.num_accepted_tokens_cpu[0] == 1
    assert runner._fuzev_paused_accepted_tokens == {"generation": 3}

    scheduled.finished_req_ids = {"readout"}
    scheduled.fuzev_readout_lane = False
    scheduled.num_scheduled_tokens = {"generation": 3}
    scheduled.scheduled_spec_decode_tokens = {"generation": [11, 12]}
    scheduled.scheduled_new_reqs = []
    scheduled.scheduled_cached_reqs = CachedRequestData(
        req_ids=["generation"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[5],
        num_output_tokens=[3],
    )
    runner._update_states(scheduled)
    assert runner.input_batch.req_ids == ["generation"]
    assert runner.input_batch.num_accepted_tokens_cpu[0] == 3
    assert runner.input_batch.spec_token_ids[0] == [11, 12]
    assert generation.output_token_ids == [8, 9, 10]
    assert generation.block_ids == ([1, 2],)
    assert not runner._fuzev_paused_accepted_tokens
    assert not runner.input_batch.pooling_params


def test_mtp3_readout_keeps_piecewise_graphs_without_speculative_shape_aliasing():
    compilation = NS(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        is_attention_compiled_piecewise=lambda: True,
        max_cudagraph_capture_size=16,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        compile_sizes=[],
        pass_config=NS(enable_sp=False),
    )
    config = NS(
        compilation_config=compilation,
        num_speculative_tokens=3,
        lora_config=None,
        scheduler_config=NS(max_num_seqs=4),
        parallel_config=NS(data_parallel_size=1),
        observability_config=NS(cudagraph_metrics=False),
    )
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.PIECEWISE, 4)
    uniform = []

    def dispatch(**kwargs):
        uniform.append(kwargs["uniform_decode"])
        return dispatcher.dispatch(**kwargs)

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = config
    runner.compilation_config = compilation
    runner.model_config = NS(is_encoder_decoder=False)
    runner.input_batch = NS(lora_id_to_lora_request={})
    runner.num_spec_tokens = 3
    runner.uniform_decode_query_len = 4
    runner._pad_for_sequence_parallelism = lambda tokens: tokens
    runner.cudagraph_dispatcher = NS(dispatch=dispatch)
    for readout in [True, False]:
        runner._fuzev_readout_batch = readout
        mode, _, _, _, _ = runner._determine_batch_execution_and_padding(
            num_tokens=4,
            num_reqs=1,
            num_scheduled_tokens_np=np.array([4]),
            max_num_scheduled_tokens=4,
            use_cascade_attn=False,
        )
        assert mode == CUDAGraphMode.PIECEWISE
    assert uniform == [False, True]

    runner, hidden = worker([None])
    graph_buffer = torch.cat([hidden, torch.full((6, 8), -999.0)])
    output = GPUModelRunner._fuzev_readout(runner, graph_buffer, None)
    assert torch.equal(output.pooler_output[0], hidden[1])


def scheduled_output(model_output):
    """Exercise the real scheduler output method, replacing only cache resources."""
    request = Request(
        "readout-0", [4, 7, 9], None, PoolingParams(task="fuzev_features")
    )
    request.status = RequestStatus.RUNNING
    request.num_in_flight_tokens = 3
    request.num_computed_tokens = 3
    scheduler = NS(
        defer_block_free=False,
        perf_metrics=None,
        requests={request.request_id: request},
        running=[request],
        enable_return_routed_experts=False,
        return_sampling_mask=False,
        is_mm_encoder_only=False,
        grammar_compile_error_reqs=set(),
        recompute_kv_load_failures=False,
        ec_connector=None,
        connector=None,
        finished_req_ids_dict={},
        kv_cache_manager=NS(
            take_events=lambda: None, estimate_cached_tokens=lambda _: 0
        ),
        make_stats=lambda *_: None,
        _handle_stopped_request=lambda _: True,
        _free_request=lambda _: (None, None),
    )
    scheduled = NS(
        num_scheduled_tokens={request.request_id: 3},
        scheduled_spec_decode_tokens={},
        total_num_scheduled_tokens=3,
    )
    result = Scheduler.update_from_output(scheduler, scheduled, model_output)
    assert not scheduler.running
    return result[0].outputs[0]


def frontend_output(core):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, "external-query")
    request = EngineCoreRequest(
        request_id="readout-0",
        external_req_id="external-query",
        prompt_token_ids=[4, 7, 9],
        mm_features=None,
        sampling_params=None,
        pooling_params=PoolingParams(task="fuzev_features"),
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    processor.add_request(request, prompt=None, queue=collector)
    result = processor.process_outputs([core])
    assert not result.request_outputs and not result.reqs_to_abort
    assert processor.get_num_unfinished_requests() == 0
    return collector.get_nowait()


@pytest.mark.parametrize("labels", [None, [7, 2]])
def test_features_survive_scheduler_serialization_and_frontend(labels):
    runner, hidden = worker([labels])
    output = GPUModelRunner._fuzev_readout(runner, hidden, None)
    core = scheduled_output(output)
    decoded = MsgpackDecoder(EngineCoreOutput).decode(MsgpackEncoder().encode(core))
    public = frontend_output(decoded)
    assert public.finished and public.request_id == "external-query"
    assert torch.equal(public.outputs.data, output.pooler_output[0])


def test_non_output_tp_rank_participates_in_projection_without_returned_logits():
    runner, hidden = worker([[1, 3]])
    runner.model.compute_logits = lambda _: None
    output = GPUModelRunner._fuzev_readout(runner, hidden, None)
    assert torch.equal(output.pooler_output[0], hidden[1])
