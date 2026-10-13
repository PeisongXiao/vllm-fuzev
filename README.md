<!-- markdownlint-disable MD001 MD041 -->
## vLLM FuZev

Release `0.30.0+fuzev.v1.0` adds optional decision-feature readout to vLLM
`0.30.0` for [FuZev](https://github.com/PeisongXiao/fuzev). Generation and
decisions share the loaded model, scheduler and prefix cache.
The separate [fuzev-vllm-plugin](https://github.com/PeisongXiao/fuzev-vllm-plugin)
serves the decision API and loads the
[decision head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev) with
[fuzev-runtime](https://github.com/PeisongXiao/fuzev-runtime).

**Contents**

- [Install From a Source Zip](#install-from-a-source-zip)
- [Limitations](#limitations)
- [Upstream vLLM](#upstream-vllm)

### Install From a Source Zip

Create one environment and install Torch with the canonical
[FuZev install procedure](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#create-an-environment).
In that active environment, the [vLLM step](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#vllm)
combines the stock **0.30.0** wheel's compiled extensions with this fork's
Python code. Run from the extracted fork root:

```bash
python -m pip download --no-deps --only-binary=:all: \
  --dest wheels "vllm==0.30.0"
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$(python -c \
  'from pathlib import Path; print(next(Path("wheels").glob("vllm-0.30.0-*.whl")).resolve())')"
python -m pip install .
```

- The wheel location selects the matching release binaries. Choose a wheel
  compatible with your platform and CUDA/PyTorch runtime.
- Before engine startup, install the matching
  [prebuilt FlashInfer cache](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#vllm)
  and keep `MAX_JOBS` set, as in that guide.
- This fork changes Python and Triton code. Its C/CUDA extension sources match
  vLLM 0.30.0.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).

Install the decision plugin separately. Its model selection, head bundle and
API settings are independent of the engine installation.
[Engine integration](docs/design/fuzev.md) covers the readout interface;
the [implementation map](docs/fuzev-fork.md) lists the changed engine components.

### Limitations

- NVFP4 MTP is supported; EXL3 with MTP is unsupported.
- The selected head bundle sets the option count. The feature width is the
  loaded model's hidden size.
- A small matched comparison of generation with and without the head loaded is
  inconclusive ([integration](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md#loaded-but-unused-head)).
- Fork overhead against stock vLLM 0.30.0 is unmeasured.
- Readout and storage: see the [design limitations](docs/design/fuzev.md#limitations).

---

## Upstream vLLM

The upstream overview below describes stock vLLM. Its package installation
commands install stock vLLM, which lacks the FuZev hooks.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
