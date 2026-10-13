# FuZev Engine Changes

Release `0.30.0+fuzev.v1.0` extends upstream vLLM `0.30.0`. See
[installation](../README.md#install-from-a-source-zip) and the
[readout interface](design/fuzev.md). The runtime, decision API plugin and
optional AGPL-3.0-only EXL3 quantization plugin are separate packages.

- **PLE:** per-layer n-gram embeddings.
- **QSA:** the sparse-attention layer of Qwen4Exp models such as Qwen3.8-Flash-Next.

## Implementation Map

| Paths | Function |
| --- | --- |
| `vllm/tasks.py`, `vllm/pooling_params.py` | Feature task and boundary validation of ordered vocabulary labels |
| `vllm/v1/fuzev.py` | Readout configuration and alternating task selection |
| `vllm/v1/core/sched/{scheduler,output,request_queue}.py` | Shared scheduling, live-feature count, unordered queue scan and target/draft cache isolation |
| `vllm/v1/engine/input_processor.py` | Feature requests on the generation runner |
| `vllm/v1/worker/{gpu_input_batch,gpu_model_runner}.py` | Final-token readout, optional query projection, paused accepted-state restoration and PLE history |
| `vllm/config/{vllm,speculative}.py` | Mmap loader/graph configuration and Qwen MTP dispatch |
| `vllm/v1/spec_decode/{qwen4,llm_base_proposer}.py` | Qwen hyper-connection widths, embedding/LM sharing and per-group draft cache metadata |
| `vllm/models/qwen4_exp/nvidia/model.py` | Packed EXL3 LM head, split vision packs and QSA scale-name mapping |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | PLE format adapter and lazy mmap row gather |
| `vllm/models/qwen4_exp/nvidia/qsa.py` | FP8 main-cache storage and scale handling |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py` | FP8 dequantization in the sparse attention kernel |
| `vllm/compilation/breakable_cudagraph.py` | Opt-in producer materialization for host n-gram embedding lookups (mmap PLE, EXL3 plugin disk-backed tables) |
| `VERSION`, `setup.py`, `MANIFEST.in`, `pyproject.toml` | Source-archive release version and packaging |

- The raw circular QSA cache declares `uses_slot_mapping=False`. The V1 runner
  respects it, including in wrapped cache specifications. Mamba and normal
  attention mapping keep their existing paths. Related upstream
  cache-capability work:
  [PR #53781](https://github.com/vllm-project/vllm/pull/53781),
  [PR #57317](https://github.com/vllm-project/vllm/pull/57317).
- The FP8 QSA path uses upstream's launch heuristic and supports graph capture,
  prefix caching and NVFP4 MTP.

## Verification

- CPU tests cover feature outputs and label ordering, engine transport,
  generation/readout scheduling, accepted-state restoration, PLE history and
  file-backed row lookup.
- GPU replay and FP8 attention tests require CUDA.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).

Model-quality and workload results are in the
[FuZev index](https://github.com/PeisongXiao/fuzev). Current readout and
storage support is in [limitations](design/fuzev.md#limitations).
