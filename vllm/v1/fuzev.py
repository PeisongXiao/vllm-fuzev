# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in final-position readout scheduled alongside generation."""

from collections.abc import Iterable
from itertools import chain
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request


FEATURE_TASK: Literal["fuzev_features"] = "fuzev_features"


def fuzev_enabled(config: "VllmConfig") -> bool:
    options = config.additional_config
    if not isinstance(options, dict):
        return False
    options = options.get("fuzev", {})
    if not isinstance(options, dict):
        raise ValueError("additional_config.fuzev must be a dictionary")
    if not options.get("enabled", False):
        return False
    if options.get("enabled") is not True:
        raise ValueError("FuZev enabled must be a boolean")
    if config.model_config.runner_type != "generate":
        raise ValueError(
            "FuZev feature readout needs a generation model (runner='generate')"
        )
    if config.parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError(
            "FuZev feature readout currently supports pipeline_parallel_size=1"
        )
    if config.scheduler_config.async_scheduling or config.use_v2_model_runner:
        raise NotImplementedError(
            "FuZev feature readout currently supports the V1 GPU model runner: "
            "set VLLM_USE_V2_MODEL_RUNNER=0 and --no-async-scheduling"
        )
    return True


def is_feature_request(request: "Request") -> bool:
    params = request.pooling_params
    return params is not None and params.task == FEATURE_TASK


def select_feature_batch(
    running: Iterable["Request"],
    waiting: Iterable["Request"],
    skipped_waiting: Iterable["Request"],
    available_slots: int,
    previous: bool,
) -> bool:
    """Alternate readout and generation batches without hiding live requests
    from the scheduler.
    """
    queues = (running, waiting, skipped_waiting) if available_slots > 0 else (running,)
    lanes = set()
    for request in chain.from_iterable(queues):
        lanes.add(is_feature_request(request))
        if len(lanes) == 2:
            return not previous
    return True in lanes
