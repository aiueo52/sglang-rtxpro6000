# Third-party notices

This release includes SGLang-derived code, FlashInfer source overlays, and other third-party code. The index below identifies their license texts, copyright notices, and source attributions. Per-file licenses and incorporated third-party notices remain applicable; an Apache identifier on a copied file does not override an incorporated notice. “Not stated” means the cited source attribution does not specify a license. Model-weight licenses are separate from source-code licenses.

Paths are relative to the release root. `S/` means `python/sglang/`, `K/` means `python/sglang/kernels/`, and `M/` means `python/sglang/multimodal_gen/`.

## SGLang source base and changes

The SGLang source base is jpezzulli/sglang-rtxpro6000 at 16e5682 (2026-08-27) plus the flash-next-fast changes by aiueo52 and contributors. Public base: [16e5682](https://github.com/jpezzulli/sglang-rtxpro6000/tree/16e5682aad6e3335f38ec8d5711176278f12a086). Build instructions: [BUILD.md](https://github.com/jpezzulli/sglang-rtxpro6000/blob/16e5682aad6e3335f38ec8d5711176278f12a086/BUILD.md).

Compared with the public base `16e5682`, `python/` has **61 modified, 31 added and 5 deleted files**, all under `python/sglang/`. The five licensing removals are listed below. `test/` has **3 modified, 28 added, and 1 removed file**. Counts include all file types, exclude generated Python caches, and compare symlinks by type and target. Of the **2,065 upstream test files**, 2,061 are unchanged, 3 modified and 1 removed, for **2,092 files under `test/`** in this release. The removed fake-worker fast-path test is recorded in `MODIFICATIONS.md`; its removal is not a licensing removal.

As recorded in the companion repository's `NOTICE`, the base fork's Flash-Next model code (QSA, hyper-connections, PLE, native MTP and memory pool) adapts the unmerged SGLang [PR #36497](https://github.com/sgl-project/sglang/pull/36497), by Qiaolin-Yu with co-authors ch-wan, ispobock, JustinTong0323, samuellees, YAMY1234, yhyang201, yizhang2077 and zijiexia. Its GDN MTP state adapts the unmerged [PR #30967 (RecoverSSM)](https://github.com/sgl-project/sglang/pull/30967). John Pezzulli contributed the SM120 integration, recovery graphs, HiCache/NIXL persistence and qualification; see the base fork's `CHANGES.md` and `PROVENANCE.md`. The base fork carries SGLang's Apache-2.0 `LICENSE` and has no `NOTICE` of its own.

## FlashInfer: shipped changes versus user-assembled dependencies

This distribution does **not** include complete `flashinfer-vt/` or `flashinfer-vtoea/` trees. It contains only 15 locally modified/added source copies under `flashinfer-overlays/{vt,vtoea}/flashinfer/`, two byte-identical wheel license copies at `flashinfer-overlays/{vt,vtoea}/LICENSE`, and `flashinfer-overlays/MANIFEST.json`. Here `FI/` refers only to retained overlay sources in both variants, except the vt-only optional policy loader. No TR policy JSON is shipped; `fused_moe/tactic_rows_policy.py` is loader code, not a policy artifact. The wheel licenses include an upstream bundled-component list; retaining that list does not mean those components are shipped in this overlay archive.

These seven paths in each variant are modified from the public `flashinfer-python==0.6.17` wheel:

- `data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu`
- `data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`
- `data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h`
- `gdn_kernels/gdn_decode_bf16_wy_output_only.py`
- `fused_moe/core.py`
- `jit/fused_moe.py`
- `jit/core.py`

### Modified FlashInfer files (Apache-2.0 section 4(b))

**aiueo52 and contributors changed the seven files above, in both variants, and added one file.** Each changed file keeps its original copyright and license header unchanged. The measured FlashInfer kernel/inference paths are unchanged; release maintenance updates notices/comments and the optional tactic-rows loader's approval label and error text. Tactic rows remain disabled and no policy is shipped. What we changed from the wheel:

- `flashinfer_cutlass_fused_moe_binding.cu`: an optional per-request length input (`vt_lens`) for the CUTLASS fused-MoE entry point.
- `cutlass_fused_moe_kernels.cuh` and `moe_kernels.h`: the MoE routing prologue fused into the grouped-GEMM setup, and a cheaper expert sort for small batches.
- `gdn_decode_bf16_wy_output_only.py`: strided q/k/v reads for the 16-token verify shape. Identical in both variants.
- `fused_moe/core.py`: passes `vt_lens` through. In vt it also adds the optional tactic-rows hook.
- `jit/fused_moe.py`: builds the MoE module under its own name (`fused_moe_<arch>_vt_prologue_v1`, or `…_vtoea_prologue_v1`).
- `jit/core.py`: with `FLASHINFER_P2_NO_NINJA=1`, it loads an existing compiled module as-is and raises an error instead of compiling when one is missing. Identical in both variants.
- Added in vt only: `fused_moe/tactic_rows_policy.py`, which loads and checks the optional tactic-rows policy. The update changes its accepted approval label to `trial` and its error text from private to local cache and workspace; no policy is shipped.
- The vtoea files differ from vt where its MoE path differs (the OEA variant, without tactic rows).

Only vt adds `fused_moe/tactic_rows_policy.py`. The C++ files retain NVIDIA Apache-2.0 notices; `fused_moe/core.py` and `jit/fused_moe.py` retain FlashInfer Apache-2.0 notices. The additions and modifications by aiueo52 and contributors are covered by the Apache-2.0 grant below unless a file states otherwise; retained third-party notices continue to apply.

Users install `flashinfer-python==0.6.17` separately and run `tools/assemble_flashinfer.py` (or select its installation with `--site-packages`). The standard-library-only tool discovers distribution metadata without importing FlashInfer, checks version and every retained RECORD file's SHA-256, ignores non-RECORD extras and `__pycache__`, validates the overlays, and restores the two expected backend directories beside `serve-fast.sh`. Replaced source files need not pass the wheel hash because their bytes are not copied. The manifest records each original wheel hash (or `added`), each shipped overlay hash, and the final tree hashes. Existing non-empty targets are accepted only if already identical; site-packages is never modified.

The **4,962 wheel-original package files per variant are not redistributed here**. This excludes the full vendored CUTLASS/CuTe, CCCL, spdlog/fmt, FlashAttention-derived Hopper files, and the wheel's proprietary-marked DSL and separately licensed TensorRT sources. User assembly restores those dependency files from the user's installation with their original notices intact; it does not remove restrictions or relicense them under Apache-2.0. The installed wheel's per-file notices and external license terms apply to those dependencies.

## Components and retained notice locations

| Component / name | License evidence | Existing text / notice / attribution location |
| --- | --- | --- |
| SGLang, including the bundled upstream-derived fork | Apache-2.0; per-file exceptions below | Root `LICENSE`; `python/pyproject.toml` identifies SGLang and the `https://github.com/sgl-project/sglang` homepage; SGLang copyright/license headers throughout `S/`, e.g. `S/srt/models/llama_eagle.py`. The fork source and build instructions are linked above; the copied metadata does not identify a different fork license. |
| SGLang AOT / sgl-kernel | Apache-2.0, with incorporated notices | `K/aot/LICENSE`; `K/aot/THIRDPARTYNOTICES.txt` (FlashInfer, TensorRT-LLM, DeepGEMM, FlashAttention sections). Those components are also indexed individually below. |
| FlashInfer modified-source overlays | Retained Apache-2.0 notices; additions and modifications by aiueo52 and contributors under Apache-2.0 unless otherwise stated | `flashinfer-overlays/vt/LICENSE`, `flashinfer-overlays/vtoea/LICENSE`; the retained `FI/` files enumerated above; FlashInfer section of `K/aot/THIRDPARTYNOTICES.txt`. Complete wheel sources are user-installed, not shipped. |
| NVIDIA CUTLASS-derived SGLang extension headers | BSD-3-Clause where stated | Retained full notice in `K/aot/csrc/cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp` and sibling headers. The full FlashInfer CUTLASS dependency is not shipped here. |
| FlashAttention, including the CuTe implementation | BSD-3-Clause | `K/ops/attention/flash_attn/cute/LICENSE` and `AUTHORS`; Dao-AILab/flash-attention section of `K/aot/THIRDPARTYNOTICES.txt`; retained FlashAttention attributions in `K/aot/csrc/infllm_v2/flash_attn/` and FlashInfer's sparse/Hopper files. |
| NVIDIA TensorRT-LLM and adapted kernels | Apache-2.0 where so marked | NVIDIA/TensorRT-LLM section of `K/aot/THIRDPARTYNOTICES.txt`; `K/jit/csrc/distributed/tp_qknorm.cuh`; retained Apache headers in the three modified C++ files under `FI/data/csrc/` listed above. This is distinct from the wheel's separately licensed TensorRT components, which are not shipped. |
| DeepSeek DeepGEMM | MIT | Full DeepGEMM notice in `K/aot/THIRDPARTYNOTICES.txt`. The FlashInfer DeepGEMM wrapper and vendored sources are not shipped. |
| gpt-fast-derived MoE | Not stated in the cited provenance header | `S/srt/layers/moe/fused_moe_native.py`, retaining its source link. The FlashInfer PyTorch-derived build helpers are not shipped. |
| NVIDIA DALI / PyTorch-inspired static-switch helper | Not stated in the cited attribution | `K/aot/csrc/infllm_v2/flash_attn/src/static_switch.h` (“Inspired by” links). |
| vLLM and copied/adapted runtime, attention, quantization, test, and communication code | Apache-2.0 where the copied header says so; incorporated original notices remain applicable | `S/benchmark/serving.py`; `S/srt/distributed/utils.py`; `K/ops/mamba/triton_ops/`; additional vLLM attribution/SPDX headers throughout `S/` and `K/`. |
| ROCm/vLLM, vLLM-Ascend, vLLM-Metal adaptations | Apache-2.0 in the cited copied files | `M/runtime/platforms/rocm.py`, `M/runtime/platforms/npu.py`, `S/srt/utils/tensor_bridge.py`. |
| Hugging Face Transformers and attributed model-code authors | Per-file headers only, commonly Apache-2.0; not a statement of model-weight or unmarked upstream-code licenses | `S/srt/models/`, `S/srt/configs/`, `S/srt/multimodal/processors/`; examples include `models/gpt2.py`, `models/voxtral.py`, `configs/granitemoehybrid.py`, and `processors/gemma3.py`. Retained author/source headers identify the individual adaptations. |
| Hugging Face Diffusers and UniPC-derived scheduler/model code | Apache-2.0 in cited copied headers | `M/runtime/layers/layernorm.py`; `M/runtime/models/schedulers/scheduling_unipc_multistep.py` and neighboring scheduler files; retained upstream links and author notices. |
| FastVideo | Apache-2.0 in marked copied files | `M/envs.py`; repeated “Copied and adapted from” headers throughout `M/`, including `runtime/pipelines_core/stages/input_validation.py`. |
| TorchTune / xDiT-derived diffusion loading and distribution code | Apache-2.0 in the cited copied files | `M/runtime/loader/fsdp_load.py`; `M/runtime/distributed/parallel_state.py`. |
| NVIDIA TensorRT Model Optimizer | Apache-2.0 in the copied file | `S/srt/layers/quantization/mxfp4_tensor.py`. |
| NVIDIA Megatron-LM-derived distributed utilities | Apache-2.0 in the copied adaptation | `S/srt/distributed/utils.py`, with vLLM and Megatron source/author notices. |
| LightSeek Foundation kernels and attention backend | MIT, full retained permission notices | `K/jit/csrc/moe/moe_finalize_fuse_shared.cu`; `K/ops/quantization/fp8_quantize.py`; `K/ops/quantization/nvfp4_gemm_swiglu_nvfp4_quant.py`; `S/srt/layers/attention/tokenspeed_mla_backend.py`. |
| Microsoft block-sparse attention utility | MIT statement | `K/aot/csrc/attention/vertical_slash_index.cu`. |
| Kimi-VL / MoonViT model-code adaptation | Retained Apache-2.0 and MIT allocation statements and texts | `S/srt/models/kimi_vl.py`; `S/srt/models/kimi_vl_moonvit.py`. These files explicitly distinguish derived portions from other portions. |
| DeepSeek-VL v2 processor | MIT, full retained permission notice | `S/srt/multimodal/processors/deepseek_vl_v2.py`. |
| DeepSeek-OCR / detectron2-derived model and sampling adaptations | Apache-2.0 in the cited model file; other provenance remains as recorded | `S/srt/models/deepseek_ocr.py`; `S/srt/sampling/custom_logit_processor.py`. |
| DeepEP-derived elementwise / concat utilities | Not stated in the cited attribution | `K/aot/csrc/elementwise/utils.cuh`; `K/jit/csrc/elementwise/concat_mla.cuh`. |
| DeepSeek EPLB and LPLB-derived solvers | Not stated in the cited attribution | `S/srt/eplb/eplb_algorithms/deepseek.py` and `deepseek_vec.py`; `K/jit/csrc/lplb/ipm.cuh`. |
| FlashMLA-derived dense FP8 utilities | Not stated in the cited provenance header; FlashAttention derivation is also attributed | `K/jit/csrc/sparse_mla_q8kv8_prefill_sm90/dense_fp8_utils.h`. |
| Flash Linear Attention | MIT for incorporated original code; other files retain their individual attribution / Apache markers where present | `K/ops/attention/fla/kda.py`; full MIT notice below; attribution headers in `K/ops/attention/fla/`, including the modified `fused_sigmoid_gating_recurrent.py`. `third_party_licenses/flash-next-fast/LICENSE.flash-linear-attention` retains Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. |
| Triton LayerNorm tutorial | MIT for incorporated original code | FLA-derived `K/ops/attention/fla/layernorm_gated.py`; `third_party_licenses/flash-next-fast/LICENSE.triton` retains Copyright 2018-2020 Philippe Tillet and Copyright 2020-2022 OpenAI. Historical series patches 0071/0072 have net zero change to this file. |
| Mamba / SSD adaptations | Apache-2.0 in marked vLLM adaptations; original author attributions retained | `K/ops/mamba/triton_ops/ssd_chunk_state.py` and sibling files. The FlashInfer SSD implementation is not shipped. |
| causal-conv1d adaptations | BSD-3-Clause for incorporated causal-conv1d code; Apache-2.0 where marked for adaptations; retained notices apply | Modified `K/ops/mamba/causal_conv1d_triton.py` attributes causal-conv1d and vLLM; `third_party_licenses/flash-next-fast/LICENSE.causal-conv1d` retains Copyright (c) 2022, the respective contributors (AUTHORS: Tri Dao). Also `S/srt/layers/attention/mamba/causal_conv1d.py`, `K/aot/csrc/mamba/causal_conv1d.h` and `K/jit/csrc/mamba/causal_conv1d.cuh`. |
| Marlin, including Neural Magic modifications | Apache-2.0 in retained headers | `K/jit/csrc/gemm/marlin/gptq_marlin.cuh` and related files. |
| ExLlama, ExLlamaV2, GPTQ-for-LLaMa-derived kernels | Not stated in the cited provenance headers | `K/aot/csrc/gemm/gptq/matrix_view.cuh`, `qdq_util.cuh`, `gptq_kernel.cu`, and adjacent attributed files. |
| llama.cpp-derived GGUF kernels | Not stated in the cited provenance header | `K/aot/csrc/quantization/gguf/vecdotq.cuh`, with original and vLLM source revisions. |
| fast-hadamard-transform | Copyright/source attribution retained; separate license not stated in cited headers | `K/jit/csrc/fast-hadamard-transform/`, e.g. `fast_hadamard_transform_common.h` and `fast_hadamard_transform_special.h`. |
| XGrammar token-bitmask kernel | Apache-2.0 in the copied file | `K/aot/csrc/grammar/apply_token_bitmask_inplace_cuda.cu`. |
| TileLang-derived top-k kernel | Not stated in the cited attribution | `K/aot/csrc/elementwise/topk.cu`. |
| LightLLM-derived attention kernels | Not stated in the cited provenance header | `K/ops/attention/decode_attention.py` and sibling LightLLM-attributed attention files. |
| LMDeploy-derived quantization / function-call code | Not stated in the cited provenance headers | `K/ops/quantization/int8_kernel.py`; `S/srt/function_call/internlm_detector.py`. |
| LightX2V-derived RMSNorm | Not stated in the cited attribution | `K/ops/diffusion/norm/rmsnorm_onepass_triton.py`. |
| PainlessInferenceAcceleration / Ant Financial SegLA | Copyright/source attribution retained; separate license not stated there | `K/ops/attention/linear/seg_la.py`. |
| Thinking Machines Lab batch-invariant operations | Not stated in the cited attribution | `S/srt/batch_invariant_ops/batch_invariant_ops.py`. |
| depyf-derived compilation code | Apache-2.0 in the copied file | `S/srt/compilation/backend.py`. |
| OpenAI simple-evals adaptations | Not stated in the cited attribution headers | `S/test/simple_eval_common.py`, `simple_eval_longbench_v2.py`, `simple_eval_humaneval.py`, and other `simple_eval_*.py` files. Benchmark/dataset links are not grants for model weights. |
| LLaVA-NeXT-derived multimodal utilities | Apache-2.0 in the copied file | `S/srt/multimodal/mm_utils.py`. |
| EAGLE and ChatGLM model-code adaptations | Apache-2.0 in the cited copied files, not inferred upstream/model licenses | `S/srt/models/llama_eagle.py`; `S/srt/models/chatglm.py`, including their source links. |
| InternVL / MiniCPM-o / ChatTTS-attributed model-code adaptations | Per-file copied headers only; no separate license is inferred from a model/repository name | `S/srt/configs/internvl.py`; `S/srt/models/internvl.py`; `S/srt/models/minicpmo.py` (including ChatTTS attributions inside the Apache-marked file). |
| MiMo Audio tokenizer adaptation | Not stated in the cited attribution | `S/srt/models/mimo_audio.py`. |
| NVIDIA LocateAnything / Nemotron VL model-code adaptations | Apache-2.0 in the cited copied files, not a model-weight license | `S/srt/models/locate_anything.py`; `S/srt/models/nano_nemotron_vl.py`; `S/srt/configs/nano_nemotron_vl.py`. |
| FastVideo / Helios adaptation | Apache-2.0 in the copied file | `M/runtime/models/dits/helios.py`. |
| LTX-2 upsampler adaptation | Apache-2.0 in the copied file | `M/runtime/models/upsampler/latent_upsampler.py`. |
| LongLive adaptation | Apache-2.0 in the copied files | `M/configs/pipeline_configs/longlive2.py`; `M/configs/models/dits/longlive2.py`. |
| MOVA scheduler adaptation | Apache-2.0 in the copied file | `M/runtime/models/schedulers/flow_match_pair.py`. |
| LingBot-Video / LingBot-World adaptations | Apache-2.0 in the cited copied files | `M/runtime/layers/moe.py`; `M/runtime/models/dits/lingbot_world.py`; `M/runtime/pipelines/lingbot_world_causal_dmd_pipeline.py`. |
| Wan2.2 utility adaptation | Per-file attribution only; no upstream/model license inferred | `M/utils.py`. |
| TeaCache coefficients | Not stated in the cited coefficient-source attribution | `M/configs/sample/wan.py`, with pinned TeaCache source links. |
| cache-dit integration/adaptation | Apache-2.0 in the copied integration file | `M/runtime/cache/cache_dit_integration.py`. |
| Sparse-VideoGen adaptation | Not stated in the cited attribution | `M/runtime/layers/attention/backends/sparse_video_gen_2_attn.py`. |
| SLA / TurboDiffusion sparse-linear attention | Apache-2.0 in the copied implementation | `M/runtime/layers/attention/backends/sparse_linear_attn.py`, including retained authorship and citation. |
| VMoBA attention source | Apache-2.0 SPDX marker and Apache package classifier in copied setup metadata | `M/csrc/attn/vmoba_attn/setup.py` and `README.md`. |
| long-context-attention / Yunchang reference | Not stated in the cited reference header | `M/runtime/distributed/parallel_groups.py`. |
| Hunyuan3D-2 Python adaptations | Source attribution retained; no separate license stated in the cited headers | `M/runtime/models/vaes/hunyuan3d_vae.py` and `M/runtime/utils/mesh3d_utils.py`. The rasterizer and mesh extensions are not shipped; see removed files below. |
| NVIDIA BigVGAN / HiFi-GAN / Snake-derived audio VAE code | MIT statements in the copied file | `M/runtime/models/vaes/minimax_h3_audio_vae/bigvgan.py`, including the individual source links. |
| alias-free-torch / Julius-derived filtering | Apache-2.0 marker plus MIT for incorporated Julius portions | `M/runtime/models/vaes/minimax_h3_audio_vae/alias_free.py`; full Julius MIT notice below. |
| Descript Audio Codec adaptation | MIT SPDX marker | `M/runtime/models/vaes/dac.py`. |
| Real-ESRGAN model code | BSD-3-Clause in retained upstream notice; enclosing wrapper has an Apache-2.0 marker | `M/runtime/postprocess/realesrgan_upscaler.py`. |
| RIFE / Practical-RIFE model code | MIT in retained upstream notice; enclosing wrapper has an Apache-2.0 marker | `M/runtime/postprocess/rife_interpolator.py`. |
| NVIDIA NVML Python bindings, copied through FastVideo | BSD-3-Clause full retained notice, alongside an Apache-2.0 marker | `M/third_party/pynvml.py`, identifying nvidia-ml-py version 12.570.86. |
| Codeplay / Intel CPU preprocessing contributions | BSD-3-Clause SPDX marker and full retained notice | `K/aot/csrc/cpu/preprocessor.cpp`. |

Third-party copyright and license notices retained in shipped files apply to their incorporated portions. Some source headers refer to upstream `LICENSE`, `NOTICE`, `licenses/`, or external terms whose full text is not at that relative location in this distribution. Import/build references to external packages (such as quack, CUDA libraries, PyTorch, or storage connectors) do not mean those entire packages are bundled.

## Removed upstream files

The following inherited files were removed for licensing reasons: the Moore Threads kernel carried a proprietary/confidential notice, and the four Hunyuan3D extension files carried Tencent Hunyuan non-commercial-license notices. Paths below identify removed files, not shipped components.

- `K/aot/csrc/elementwise/fused_add_rms_norm_kernel.mu`
- `K/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer.cpp`
- `K/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer_gpu.cu`
- `K/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer.h`
- `K/ops/diffusion/ext/mesh_processor/mesh_processor.cpp`

The MUSA fused-add-RMSNorm kernel and the Hunyuan3D rasterizer/mesh extensions are therefore unavailable in this release. Other Moore Threads files with Apache-2.0 notices are retained.

## Additions by aiueo52 and contributors; model assets

Unless a file states otherwise, the additions and modifications by aiueo52 and contributors are licensed under Apache-2.0 (see LICENSE), except the four public token-map data files below. Copyright 2026 aiueo52 and contributors.

Model weights are not included. The normal stock checkpoint and hyperpro ThinkingCap/original-head weight overlay are user-supplied assets with their own applicable terms. Qwen3.8-Flash-Next is under the Qwen Community License 1.0, not Apache-2.0. A checkpoint with a trained head written back is a derivative of the model: include its license and notices when sharing it and check the license conditions.

### Public token-map data (CC BY-SA 4.0)

The following four files are offered under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/), **not Apache-2.0**, matching the companion repository's `NOTICE`:

- `tokenmaps/public_49152.pt`
- `tokenmaps/public_32768.pt`
- `tokenmaps/public_49152.sources.json`
- `tokenmaps/public_32768.sources.json`

The 49,152-id map is a sorted list of tokenizer ids ranked by frequency in public text; the 32,768-id map is a subset. The maps contain no source text or counts. Whether a frequency-ranked id list adapts Wikipedia text under ShareAlike is unsettled; these four data files are offered under CC BY-SA 4.0 as a precaution. This does not relicense the source corpora or repositories, which are not shipped.

Changes made to every source: the listed files were read once (conversations: assistant / `gpt` / `ai` turns only, capped per source in file order), tokenized, and only token frequencies were used to rank ids. The two `.sources.json` files retain exact revisions, input file hashes, attribution and per-source totals.

Attributions and source-use caveats:

- English and Japanese Wikipedia, by Wikipedia contributors, via [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, dump `20231101`, shards `20231101.en/train-00007-of-00041` and `20231101.ja/train-00007-of-00015`: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and GFDL.
- [SWE-agent-trajectories by Nebius](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), revision `68195a1450865274106246d0d0296a1d6807b88e`, shard `data/train-00000-of-00012`: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its card also asks users to respect each instance repository's license and says use of its model outputs must comply with the Llama 3.1 Community License. Whether either condition applies to a frequency-ranked id list is unsettled; these caveats are retained for users to assess.
- Other sources: HuggingFaceH4/ultrachat_200k — MIT; llm-jp/oasst2-33k-ja, open-thoughts/OpenThoughts-114k and NousResearch/hermes-function-calling-v1 — Apache-2.0; CPython v3.13.0 — PSF-2.0; Go go1.23.0 — BSD-3-Clause; React v18.3.1, Vue v3.5.12 and Tokio tokio-1.40.0 — MIT; curl curl-8_10_1 — curl (MIT-style). Exact revisions and file hashes are recorded in the provenance JSON files.

The map-building scripts are not in this repository. They are `tokenmaps/build_hot_vocab.py` and `tokenmaps/public/prep_public.py` in [aiueo52/hyperburst-rtxpro6000](https://github.com/aiueo52/hyperburst-rtxpro6000) (Apache-2.0; see its `tokenmaps/README.md`), and they are not among these four data files.

## MIT notices for incorporated portions

### Flash Linear Attention portions in `K/ops/attention/fla/kda.py`

The copied Flash Linear Attention portions retain the following MIT notice. Source: https://github.com/fla-org/flash-linear-attention. The enclosing file retains its vLLM attribution and Apache-2.0 marker.

MIT License

Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

### Julius portions in `M/runtime/models/vaes/minimax_h3_audio_vae/alias_free.py`

The `sinc` and `LowPassFilters`-derived filtering portions are from Julius (https://github.com/adefossez/julius), incorporated through alias-free-torch. The retained source links are https://adefossez.github.io/julius/julius/core.html and https://adefossez.github.io/julius/julius/lowpass.html. The enclosing file retains its Apache-2.0 marker.

Copyright 2020 Alexandre Défossez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Inventory scope

This index covers the shipped FlashInfer overlay sources and licenses, the SGLang-derived sources including local additions and modifications, and the incorporated third-party code identified above. Generated `flashinfer-vt/` and `flashinfer-vtoea/` dependency trees are not included in the release archive. `flashinfer-overlays/MANIFEST.json` records hashes for the shipped overlays and assembled trees; the dependencies retain their own applicable terms.
