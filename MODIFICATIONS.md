# Modified upstream files

Apache License 2.0 §4(b) notice: the `flash-next-fast` patch series (by aiueo52, 2026) modifies the
following files of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) at
`16e5682aad` (itself derived from SGLang, and for some files from vLLM, flash-linear-attention,
causal-conv1d and others; see the companion repository's NOTICE and `third_party_licenses/flash-next-fast/`). Every other file the series touches is new. The patch numbers
are the commits of this branch in order after `16e5682aad` (0001 = first), identical to
`patches/sglang/series/` in the companion repository; each commit is the exact record of the change. FlashInfer files are listed at the end.

| modified upstream file | patches |
|---|---|
| `python/sglang/kernels/jit/csrc/elementwise/hc_combine.cuh` | 0054, 0058 |
| `python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py` | 0001 |
| `python/sglang/kernels/ops/attention/fla/layernorm_gated.py` | 0071 (reverted by 0072; net unchanged) |
| `python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py` | 0048, 0070 |
| `python/sglang/kernels/ops/elementwise/elementwise.py` | 0055, 0071 |
| `python/sglang/kernels/ops/elementwise/hc_combine.py` | 0054, 0058 |
| `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py` | 0023, 0045 |
| `python/sglang/kernels/ops/speculative/topk1.py` | 0016, 0082 |
| `python/sglang/srt/arg_groups/speculative_hook.py` | 0002 |
| `python/sglang/srt/environ.py` | 0019, 0042, 0054, 0056, 0059, 0075, 0082 |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | 0012 |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | 0048 |
| `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` | 0018 |
| `python/sglang/srt/layers/attention/qsa/graph_metadata.py` | 0005, 0043 |
| `python/sglang/srt/layers/attention/qsa/metadata.py` | 0005 |
| `python/sglang/srt/layers/attention/qsa/qsa_indexer.py` | 0005 |
| `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` | 0005, 0015, 0016 |
| `python/sglang/srt/layers/hc_mix_triton.py` | 0010 |
| `python/sglang/srt/layers/hyperconnection.py` | 0003, 0010, 0017, 0025, 0031, 0034, 0040, 0054, 0056, 0059, 0075 |
| `python/sglang/srt/layers/linear.py` | 0003, 0099 |
| `python/sglang/srt/layers/logits_processor.py` | 0032 |
| `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py` | 0100, 0103 |
| `python/sglang/srt/layers/moe/topk.py` | 0019 |
| `python/sglang/srt/layers/quantization/fp8.py` | 0004, 0032, 0040, 0049, 0079, 0098 |
| `python/sglang/srt/layers/quantization/modelopt_quant.py` | 0003, 0022 |
| `python/sglang/srt/layers/quantization/nvfp4_online.py` | 0022 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | 0005 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 0019, 0020 |
| `python/sglang/srt/mem_cache/qsa_kv_pool.py` | 0005 |
| `python/sglang/srt/model_executor/input_buffers.py` | 0014 |
| `python/sglang/srt/model_executor/model_runner.py` | 0004, 0012, 0013 |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | 0012 |
| `python/sglang/srt/models/qwen2_moe.py` | 0003, 0019, 0024, 0040, 0041, 0055, 0057, 0060 |
| `python/sglang/srt/models/qwen3_5.py` | 0006, 0007, 0048, 0049, 0050, 0051, 0079 |
| `python/sglang/srt/models/qwen3_5_mtp.py` | 0011, 0022 |
| `python/sglang/srt/models/qwen3_next.py` | 0004 |
| `python/sglang/srt/models/qwen4_exp.py` | 0003, 0017, 0055, 0056, 0075 |
| `python/sglang/srt/models/qwen4_exp_mtp.py` | 0019, 0024, 0042, 0044 |
| `python/sglang/srt/server_args.py` | 0002, 0003 |
| `python/sglang/srt/speculative/adaptive_runtime_state.py` | 0012, 0061, 0082, 0104 |
| `python/sglang/srt/speculative/adaptive_spec_params.py` | 0012, 0061, 0062, 0063, 0066, 0082 |
| `python/sglang/srt/speculative/base_spec_worker.py` | 0082 |
| `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` | 0061 |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | 0004, 0006, 0012, 0014, 0016, 0065, 0082, 0095, 0096, 0104 |
| `python/sglang/srt/speculative/spec_info.py` | 0002 |
| `test/registered/unit/models/test_qwen4_exp_mtp.py` | 0044 |
| `test/registered/unit/spec/test_adaptive_spec_params.py` | 0063 |
| `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py` | 0012 |

The separate training-data dump hook in the companion repository's `patches/sglang/mtp-dump/` (nine patches, applied on top of `flash-next-fast` by its own `apply.sh`, not part of this series) also modifies `python/sglang/srt/models/qwen4_exp.py` and `python/sglang/srt/models/qwen4_exp_mtp.py` and adds `python/sglang/srt/models/mtp_dump.py`.

## FlashInfer (flashinfer-python 0.6.17 wheel)

Modified by the companion repository's `patches/flashinfer/` (production chain 01–07); on this branch,
`bench/glue_e/gdn_wy_T16_strided.patch` carries patch 07 (FlashInfer-derived code; FlashInfer's NOTICE is in
`third_party_licenses/flash-next-fast/NOTICE.flashinfer`):

- `flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` (01–06)
- `flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` (01–06)
- `flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py` (07)

Additionally, only in the companion repository's `patches/flashinfer/experimental/` (rejected variants):
`flashinfer/data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu`,
`flashinfer/fused_moe/core.py`.

On this branch, the final commit (after the 105 patches) adds a one-line "Modified by aiueo52 for
flash-next-fast (2026)" header comment to every file in the SGLang table above, and this list.
