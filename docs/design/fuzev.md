# Decision Features in vLLM

FuZev adds an opt-in `fuzev_features` pooling task to the existing generation
runner. It shares the engine's model, scheduler, encoder cache, hybrid state
and KV-cache manager.

## Enable Readout

Before constructing the engine, set:

- `additional_config={"fuzev": {"enabled": true}}`
- `async_scheduling=false`
- `VLLM_USE_V2_MODEL_RUNNER=0`

The decision plugin configures readout and scores the features with its
separately loaded shared head.

`PoolingParams(task="fuzev_features")` returns the last consumed token's final
hidden state before the vocabulary projection, as FP32. The feature width
follows the loaded model. Chunked prefill returns a result only after consuming
the entire submitted prompt.

A query may add `extra_kwargs={"baseline_token_ids": [...]}` with
in-vocabulary label IDs. The runner appends their target-model LM logits in the
requested order, and the plugin's head reads them with the features. Candidate
feature requests need no vocabulary projection. The head bundle sets the
maximum number of options.

## Scheduling and Caching

- Generation and feature requests alternate task-homogeneous steps through the
  same scheduler, within its normal sequence and token budgets.
- With no live feature requests, task selection is one counter check with no
  queue scan.
- Generation does no feature readout or head scoring.
- Generation keeps speculative decoding. Feature requests consume only submitted
  target tokens.
- The engine handles pending accepted tokens through ordinary prefill and
  restores paused generation's accepted Mamba-state positions.
- QSA sparse-attention layers and each side-cache group use their own block
  tables and addressing rules.
- Feature requests may reuse valid engine prefix-cache entries. Under MTP they
  do not publish target-only cache blocks as complete target-and-draft entries.
- Decision scores, probabilities and responses are never cached.

## Disk-backed PLE

For raw BF16 or FP8 Qwen4Exp PLE (per-layer n-gram embedding) tables, set
`additional_config={"ple_storage": "mmap"}`. This setting is independent of
FuZev readout.

- The standard model loader resolves Hugging Face IDs, revisions and local paths.
- Mmap mode selects lazy safetensors loading and keeps those CPU tensor views.
- Only requested rows are gathered and moved to the accelerator. Row coverage
  is checked once, when loading completes.
- Non-eager execution adds `vllm::qwen4_exp_ple_mmap_lookup_out` to the graph
  splitting operations. Full graph modes become PIECEWISE with a warning.
- With breakable graphs enabled, this lookup and the EXL3 plugin's disk-backed
  lookup replay their immediately preceding captured producer before the eager
  CPU gather. The gather writes into a stable output buffer for downstream
  graph replay.

Quantization plugins may supply `get_ple_embedding_method(prefix, embedding_dtype)`
for packed tables. EXL3's packed PLE adapter uses the optional AGPL-3.0-only
[quantization plugin](https://github.com/PeisongXiao/vllm-exl3-plugin). The fork
also maps split EXL3 vision Q/K/V packs and applies EXL3 to the existing LM head.

## Limitations

- Readout currently supports the V1 GPU model runner
  (`VLLM_USE_V2_MODEL_RUNNER=0`), synchronous scheduling
  (`--no-async-scheduling` or `"async_scheduling": false`) and pipeline-parallel
  size 1.
- Tensor/data-parallel readout and KV/encoder-cache transfer connectors have no
  live readout verification.
- Mmap PLE supports raw BF16/FP8 tables and eager or piecewise execution. The
  recorded setup uses ETP/TP/DP=1; other parallel configurations are untested.
- EXL3 plus MTP is unsupported.
- Qwen MTP index sharing across iterations is unverified.
- Image decision quality is experimental.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).
