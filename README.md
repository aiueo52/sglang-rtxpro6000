<!-- Modified by aiueo52 for flash-next-fast (2026); see MODIFICATIONS.md. -->

# flash-next-fast

This branch contains jpezzulli/sglang-rtxpro6000 at 16e5682 plus the flash-next-fast series and the 2026-10-27 update by aiueo52 and contributors ([public base (2026-08-27)](https://github.com/jpezzulli/sglang-rtxpro6000/tree/16e5682aad6e3335f38ec8d5711176278f12a086)). It is tuned for **Qwen3.8-Flash-Next NVFP4 on one 96 GB RTX PRO 6000 (sm120)**, including the case where the same GPU also drives a desktop.

The method, measurement notes and history are in the companion repository [aiueo52/hyperburst-rtxpro6000](https://github.com/aiueo52/hyperburst-rtxpro6000).

It adds support for four concurrent requests with long context while retaining the single-request optimizations. Performance comparisons apply only to the measured configurations below. It ships in two editions:

| edition | weights | what is different |
|---|---|---|
| `normal` (default) | Qwen3.8-Flash-Next NVFP4 | — |
| `hyperpro` | ThinkingCap (Qwen3.8-Flash-Next NVFP4) | OEA on (an approximate MoE path; outputs change slightly, quality checked below) |

## What you get

- **Up to 4 concurrent requests, 262,144-token context each**, with 7 GB of VRAM left for the display (`DISPLAY_RESERVE_GB=7`).
- **Hierarchical prefix cache in host RAM (32 GB at the default `CONC=4`)**, so conversations that are not running keep their prefix and come back without a full re-prefill.
- **MTP speculative decoding with an adaptive draft length**. The default table is calibrated for 4 concurrent requests.
- Single-request optimizations including PLE embedding offload, fused verify and optimized MoE/draft kernels.

## Requirements

- An RTX PRO 6000 Blackwell (96 GB). Other sm120 cards are untested.
- The base environment from the upstream [`BUILD.md`](https://github.com/jpezzulli/sglang-rtxpro6000/blob/16e5682aad6e3335f38ec8d5711176278f12a086/BUILD.md): Python 3.12, PyTorch 2.13.0+cu130, FlashInfer 0.6.17 and a CUDA 13 toolkit with `nvcc`. The first start compiles kernels, so `nvcc` and a host C++ compiler are required.
- Measured on: Ubuntu 24.04 (Linux 7.0), NVIDIA driver 595.91.07, RTX PRO 6000 Blackwell Max-Q Workstation Edition, nvcc 13.3.73, g++ 13.3.0.
- **The first start compiles kernels with nvcc.** On the machine used by aiueo52 and contributors, with the build limited to 12 of its 48 CPU threads and MAX_JOBS=2, the first start compiles 14 modules: about 47 min, peak RAM about 20 GiB (the largest, one MoE module, about 27 min and 16 GiB). Part of this build shared its CPUs with other work, so treat the time as an upper bound. Later starts reuse the cache.
  - Hyper pro compiles its own MoE module in place of that one. We have not timed it: our 90-minute test budget ran out about 9 minutes into this module, at 11 of 97 steps.
  - SGLang's own JIT kernels (three small ones) are built on the first start, before the server reports ready: 13–17 s in total on the machine used by aiueo52 and contributors. They need `ninja` on `PATH` (it is installed in the venv; run with the venv active). Without it the server stops at its first request.
  - Not included: kernels that compile on first use at run time (CuTe-DSL NVFP4 quantize, Triton) and CUDA graph capture. These need the GPU and were not timed separately.
- Host RAM: about 100 GiB for the server process with the defaults (measured peak 100.5 GiB: about 67 GiB for the model and its offloads, plus the 32 GB HiCache host pool). Set `FLASH_NEXT_HICACHE_GB` lower to reduce the pool at `CONC=4`. Measured on a 256 GB machine.
- Weights (not included; see [Assets](#assets)):
  - normal: RadixArk/Qwen3.8-Flash-Next-NVFP4.
  - hyper pro: a local overlay of ThinkingCap on that model, built with `tools/thinkingcap_overlay/overlay.py` (16 GB of new files; the rest are links to the original).

## Install (source overlay)

Use a venv built from the upstream [`BUILD.md`](https://github.com/jpezzulli/sglang-rtxpro6000/blob/16e5682aad6e3335f38ec8d5711176278f12a086/BUILD.md). Only that venv plus `serve-fast.sh` is tested; **installing this branch with pip (for example `pip install ./python`) is not tested or supported**.

The measured runs used `serve-fast.sh` inside that venv, not a pip install of this branch. With `REPO` denoting the checkout root, the launcher sets `PYTHONPATH` to `$REPO/flashinfer-vt:$REPO/python` for normal or `$REPO/flashinfer-vtoea:$REPO/python` for hyperpro (OEA on). The assembled FlashInfer tree and this branch's Python source therefore take precedence over installed packages.

### Install FlashInfer (once)

This repository ships only the FlashInfer files we changed or added (`flashinfer-overlays/`). The rest comes from the public FlashInfer 0.6.17 package. A tool rebuilds the two trees the server uses (`flashinfer-vt/` for normal, `flashinfer-vtoea/` for hyper pro) beside `serve-fast.sh`:

```bash
python3 -m pip install flashinfer-python==0.6.17 ninja   # in the venv you serve from
python3 tools/assemble_flashinfer.py                      # or: --site-packages /path/to/venv/lib/python3.12/site-packages
```

- The tool uses only the Python standard library and needs no GPU or network.
- It requires version 0.6.17 and checks every file it keeps against the package's RECORD hashes. It also checks both final trees against `flashinfer-overlays/MANIFEST.json`. The manifest identifies the release's assembled trees, including modification notices, comment edits and the optional policy-loader label/text update; these hashes are not a new performance measurement.
- It never writes into site-packages, and it refuses to overwrite a non-empty target unless that target is already identical.
- If the trees are missing, `serve-fast.sh` prints the command to build them.

## Quick start

Run with the venv active, so that `ninja` is on `PATH`:

```bash
export TARGET_MODEL=/path/to/Qwen3.8-Flash-Next-NVFP4   # hyper pro: the model-overlay-orig directory
CONC=4 FLASH_NEXT_EDITION=normal bash serve-fast.sh wa  # FLASH_NEXT_EDITION=hyperpro for hyper pro
```

- The server binds to **127.0.0.1** on `PORT` (default `8001`) and serves the OpenAI-compatible API under `/v1` with the model name `flash-next`.
- The launcher configures the inherited admin and management routes, including weight updates and tensor-based LoRA loading, to require an admin key by default. It generates a random per-start key unless you set `FLASH_NEXT_ADMIN_API_KEY`. Tools calling admin routes (for example `/flush_cache`) need `FLASH_NEXT_ADMIN_API_KEY` set before launch and must send that value in an `Authorization: Bearer <key>` header. The shipped `serve-*` helpers and `oea` do not call admin routes; `oea` uses the control file.
- **The launcher does not enable LoRA.** If you enable it, inference requests can name loaded adapters. Only path-loaded adapters, supplied through `--lora-paths` or admin routes, can reload from their recorded path after eviction or an admin unload; unloading them is not access revocation. Tensor-loaded adapters record the placeholder path `__tensor__` and cannot reload this way. Restart the server without that adapter in `--lora-paths` to retire it. This is upstream behavior, not per-adapter access control added by the fork.
- **Several tokenizer workers are refused while a key is set.** With `--tokenizer-worker-num` greater than 1, upstream skips the authentication middleware, so this release stops at startup with `authentication with multiple tokenizer workers is not supported in this release` when `--api-key` or `--admin-api-key` is set. The launcher always sets an admin key, so launches through it must keep one tokenizer worker (the default).
- **Readback redaction is not a complete security boundary.** `/server_info`, deprecated `/get_server_info` and the startup log always redact `admin_api_key`, `api_key` and `ssl_keyfile_password` when set; unset values remain `None`. Request and crash dumps are off by default. When enabled, the crash-dump `resolved_config` snapshot, `server_args` and recorded launch command redact secrets. Request data, including prompts and outputs, remains sensitive and is not redacted; treat dump files as sensitive. The admin key remains visible in the live process argv to local processes, as described below.
- SGLang's `server_args.py` has no admin-key environment-variable support, so the launcher passes `--admin-api-key=<key>` in argv. Other local processes can read it with `ps`: on Linux, command lines are visible to all local users by default, not just the server's user.
- Normal inference endpoints (`/v1/*`, `/generate`) and `/health` and `/metrics` remain unauthenticated by default on 127.0.0.1. Upstream CORS allows any origin, so any web page you visit can send inference requests to them. **Do not expose the server to a network without an authenticating proxy.** To require a key for inference and server-info endpoints, append upstream arguments after the preset: `bash serve-fast.sh wa --api-key=<inference-key>`. For keys beginning with `-`, use `--api-key=<key>`; argparse may reject a separate value that starts with `-`. This does not replace the admin key; `/health` and `/metrics` stay unauthenticated.
- `bash serve-fast.sh wa --print` prints the resolved command and a filtered environment without starting anything. Argument names are lowercased, stripped of leading dashes, and normalized from `_` to `-`; except for known nonsensitive options, values are shown as `<redacted>` when the name ends in `key`, `token`, `secret`, `password`, `passwd`, `auth`, `cred`, `creds`, `credential` or `credentials`. Both `--flag value` and `--flag=value` are handled, including sensitive long-option abbreviations such as `--admin-api-k` and `--api-ke`, plus `--hf-token` abbreviations. The harmless options `--bucket-time-to-first-token` and `--kt-max-deferred-experts-per-token` stay visible despite their `token` suffix, as do known nonsensitive options such as `--ssl-keyfile`, `--speculative-num-draft-tokens` and `--speculative-token-map`. After a credential flag without `=`, the next token is masked whatever it starts with, unless it is itself a credential option such as `--admin-api-key` (then that option's value is masked); when the masked token is a bare `--` or looks like a credential flag without `=`, the token after it is masked too. Environment redaction is unchanged: values whose names contain `TOKEN`, `KEY`, `SECRET`, `PASS`, `AUTH` or `CRED` (case-insensitive) are hidden. Review the output before sharing it; diagnostic redaction does not hide runtime argv from local processes.

### Settings you may change

| env | default | meaning |
|---|---|---|
| `FLASH_NEXT_EDITION` | `normal` | `normal` or `hyperpro` |
| `CONC` | `4` | `wa` launch preset: `4` or single-stream preset `1` (see below); `6` is unvalidated and its long-context ladder does not fit in KV |
| `DISPLAY_RESERVE_GB` | `7` | VRAM kept free for the desktop |
| `FLASH_NEXT_HICACHE_GB` | `32` | host-RAM prefix cache at `CONC=4`; `0` turns it off |
| `FLASH_NEXT_CHUNK` | `8192` | normal edition, `CONC>1`: prefill chunk size; `4096`, `8192`, `16384` or `32768` |
| `FLASH_NEXT_PD_INTERVAL` | `0` | normal edition, `CONC>1`: nonnegative upstream `--prefill-decode-interval`; `0` leaves the server default |
| `FLASH_NEXT_RECUR_HEAD` | `tokenmaps/public_32768.pt` | recurrent MTP head map; empty disables it |
| `FLASH_NEXT_ADMIN_API_KEY` | random per start | key required by admin/management routes; visible to local processes through argv (see above) |
| `SGLANG_TACTIC_ROWS` | `0` | tactic rows are not available in this release (no policy is shipped); any value other than `0` stops at start |
| `ADAPTIVE_CONFIG` | `adaptive/w16_c4.json` | adaptive draft-length table, for both editions and every `CONC` unless explicitly overridden |

**`CONC=1` is not a one-request admission limit.** It leaves `serve-local.sh`'s `--max-running-requests 4` and `--chunked-prefill-size 8192` in place, defaults to 10 Mamba cache slots, and does not enable HiCache, even if `FLASH_NEXT_HICACHE_GB` is set. Send only one request at a time to reproduce a single-stream workload. This documents the measured launcher behavior; it has not been changed to enforce a limit of one.

The other retained adaptive tables are explicit alternatives or test fixtures; the launcher does not automatically select them by edition or `CONC`.

### Runtime files and disk use

By default, `serve-fast.sh` writes under `~/.cache/flash-next-fast/<edition>/`: `hf/` for Hugging Face caches, `workspace/` for FlashInfer builds, `jit/`, `triton/`, `nvfp4/` and other compiler caches, `tmp/`, `runtime/` and `oea/`. If unset, `TMPDIR` points to `tmp/`, `XDG_RUNTIME_DIR` to `runtime/`, and `XDG_CACHE_HOME` to `xdg/` there. A pre-existing `XDG_CACHE_HOME` supplies the base instead of `~/.cache`; explicit cache-variable overrides are preserved. Model weights remain in `TARGET_MODEL`, not copied into these caches.

Budget several GB per edition for compiled-kernel/build caches (a planning estimate, not a measured disk total); use varies with kernels and toolchain, and temporary build files need extra space. A packed 49,152 × 2,560 NVFP4 head is about 71 MB per cache entry; changing model or map can create another entry. The 32 GB HiCache pool is host RAM, not a disk allocation.

### Live OEA control (hyperpro)

After startup completes, use `./oea status`, `./oea on`, `./oea off` or `./oea rows <2..16>`. **Switch only between generations/requests, never during a generation.** `on` uses the server's launch value of `OEA_K0` (default `6`); `off` restores standard computation. `rows` changes the minimum number of live verify rows needed for OEA without changing on/off (default `16`).

The default control file is `~/.cache/flash-next-fast/hyperpro/oea/flash-next.ctl`; acknowledgements are in the adjacent `.status.jsonl` file. If you override `XDG_CACHE_HOME` or `SGLANG_OEA_CTL_FILE`, pass the same settings to the launcher and `oea`. The helper controls an already-running hyperpro server; it does not start one.

## Measured results

These 2026-10-27 figures are author-reported; the run files behind the comparisons, quality checks and BetterBench results are not published in this update.

The machine was one RTX PRO 6000 Blackwell (Max-Q), which also drives a 4K desktop; the screen was off during measurement. Each comparison was made against our previous best build of the same edition, with alternating server starts (A B B A blocks) and 95 % intervals over the start pairs. These measurements predate the release's licensing, packaging and launcher cleanups; the final release has not been re-measured on a GPU.

Since measurement, the Python changes are security hardening (the admin key on tensor-based LoRA loading, the refusal of authentication with several tokenizer workers, and redaction of readbacks, logs, dumps and argv) and docstring, log and error text. The inference path is unchanged, and the launcher command only adds the admin key. In the FlashInfer overlay, the optional tactic-rows policy loader changed only its accepted approval label and its error text; tactic rows stay disabled and no policy is shipped.

### Single request (c = 1): measured configurations only

| edition | server ms/step (new/old) | decode tok/s (new/old) | verdict |
|---|---|---|---|
| normal | 1.001 [0.993, 1.008] | 1.001 [0.987, 1.015] | no significant difference (12 starts) |
| hyperpro | 0.978 [0.966, 0.989] | 1.029 [1.003, 1.055] | faster (12 starts) |

These are averaged over 11 workload shapes: English and Japanese prose, agent loop, reasoning, code edit, greedy and sampling, and one 64k-token prompt.

Conditions: both arms of every comparison used a private MTP head and token map (the release ships the stock head and public maps instead, see Assets). Do not assume these ratios apply to the public assets; their absolute speed is in the BetterBench table. The normal row used tactic rows, which are not available in this release. With tactic rows off (the release setting), an earlier 8-start check found every shape within ±4 %, but its intervals were too wide for a conclusion.

### Quality

| check | normal | hyperpro |
|---|---|---|
| GSM8K 1,319 (thinking closed), c=1 → c=4 | 624 → 622 | 648 → 654 |
| 200 problems with thinking, c=1 → c=4 | 192 → 191 | 193 → 193 |
| 4 × 64k-token synthetic, cross-talk / wrong / mismatch vs c=1 | 0 / 0 / 0 | 0 / 0 / 0 |

For ThinkingCap against the stock weights on the same fast path: +0.6 points [−1.2, +2.7] over 308 items (GSM, MATH, CE), thinking 14 % shorter, time to answer 12 % shorter.

### BetterBench (BetterBench 0.6.0, the machine used by aiueo52 and contributors, screen off, one start per build)

| build | c=1 decode tok/s | 4 concurrent, total tok/s | TTFT p50 c=1 | prefill tok/s @ 32k |
|---|---|---|---|---|
| flash-next-fast normal | 369.9 | 710.2 | 101 ms | 12,796 |
| flash-next-fast hyperpro | 342.3 | 693.7 | 103 ms | 12,266 |
| previous best build, normal | 371.6 | 331.5 | 99 ms | 12,918 |
| previous best build, hyperpro | 353.3 | 323.6 | 101 ms | 12,401 |

Conditions:
- All four builds ran with the shipped assets: the public token maps and the stock MTP head. Hyper pro used the locally built ThinkingCap overlay.
- c=1 decode is BetterBench's weighted median over its single-stream categories.
- The previous builds serve one request at a time, so their 4-concurrent requests queue; flash-next-fast serves 4 at once (`CONC=4`).
- One start per build: treat differences of a few percent as run-to-run noise.
- Reference runs with a private token map and fine-tuned MTP head (not shipped) reached 420.9 (normal) and 423.7 (hyperpro) tok/s at c=1; the public assets were about 12 % and 19 % slower in those runs.

## Reference: tactic rows (not available in this release)

Tactic rows select MoE GEMM tactics by row count. **Tactic rows are not available in this release (no policy is shipped).** Every `SGLANG_TACTIC_ROWS` value other than `0` stops at start. Policy-building and approval tools are not included.

These normal-edition prefill numbers come from an earlier tree on the machine used by aiueo52 and contributors, before a later SGLang MoE fix. They are reference values, not measurements or features of this release:

| | without | with | change |
|---|---|---|---|
| c = 1 | 13,361 tok/s | 14,021 tok/s | +4.9 % |
| c = 4 | 10,335 tok/s | 10,968 tok/s | +6.1 % |

Earlier-tree quality checks found logit drift within A/A noise and retrieval 24/24 with tactic rows versus off for normal; hyperpro kept them off.

## Not in this version

- 6 concurrent requests as a default: at 6 the long ladder does not fit in KV.
- Per-request verify/draft-length control, an experimental prefill/decode interleaving scheduler and experimental additional kernels were measured but are not included. `FLASH_NEXT_PD_INTERVAL` only sets upstream's existing `--prefill-decode-interval`; it does not enable that experimental scheduler.

## Assets

- **Draft token map:** `tokenmaps/public_49152.pt` is the default. It was built from public data only; the sources and their licenses are in `tokenmaps/public_49152.sources.json`. The two public maps and their two `.sources.json` files are offered under **CC BY-SA 4.0, not Apache-2.0**; see the attributions and source-use caveats in `THIRD_PARTY_NOTICES.md`. To use your own map, set `TOKEN_MAP=/path/to/map.pt` (a sorted list of token ids saved with `torch.save`). The build and preparation scripts named in the provenance JSON are not in this repository; the maps can be rebuilt with `tokenmaps/build_hot_vocab.py` and `tokenmaps/public/prep_public.py` from [aiueo52/hyperburst-rtxpro6000](https://github.com/aiueo52/hyperburst-rtxpro6000) (see its `tokenmaps/README.md`).
- **Recurrent MTP head map:** `tokenmaps/public_32768.pt` (a subset of the above, same sources) is the default. Set `FLASH_NEXT_RECUR_HEAD=` (empty) to turn it off, or point it at your own map. At launch, `serve-fast.sh` aborts if a configured map file is missing. During server initialization, if loading or installing an existing recurrent map fails, the server logs a warning and keeps the original MTP head and token map. This fallback does not apply to the draft token map.
- **MTP head:** the stock head in the model checkpoint is used. A head you trained yourself can be used by putting it in your own checkpoint directory and pointing `TARGET_MODEL` at that directory. Our own fine-tuned head is not shipped.
- **ThinkingCap (hyper pro):** the weights are not redistributed here. Download the ThinkingCap NVFP4 checkpoint linked below and build the overlay locally (see `tools/thinkingcap_overlay/overlay.py`; `plan`, `build`, then `verify`).

### Model cards and licenses

Model weights have separate terms from this code. Read the model cards and linked licenses before use:

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) (revision `de4b8e4`): the source model. Its LICENSE file is the Qwen Community License 1.0, which sets conditions on some commercial uses.
- [bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4/tree/df3922b21fc28a8604cc9b31451495690d155194) (revision `df3922b`): its LICENSE file is the Qwen Community License 1.0, which sets conditions on some commercial uses.
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) (revision `7b71922`): its card lists the license as `other` and refers to the source model for terms.

## Changes from upstream

- SGLang base: jpezzulli/sglang-rtxpro6000 at 16e5682 plus the flash-next-fast series and the 2026-10-27 update by aiueo52 and contributors ([public base (2026-08-27)](https://github.com/jpezzulli/sglang-rtxpro6000/tree/16e5682aad6e3335f38ec8d5711176278f12a086)), including the published 121-patch series.
- Compared with the public base `16e5682`, `python/` has **61 modified, 31 added and 5 deleted files**, all under `python/sglang/`. `test/` has **3 modified, 28 added and 1 deleted file**. Counts include all file types, exclude generated Python caches, and compare symlinks by type and target.
- Of the **2,065 upstream test files**, 2,061 are unchanged, 3 modified and 1 removed, for **2,092 files under `test/`** in this release. The removed file is `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py`: its fake worker no longer matches this tree's worker, so it failed without testing the production path (see `MODIFICATIONS.md`). The additions include the CPU-only redaction and route-authentication tests.
- Added single-GPU Qwen3.8-Flash-Next NVFP4 serving optimizations, PLE offload, fused verify/draft kernels and adaptive MTP decoding; added four-request scheduling/cache settings, public token maps, launch helpers and ThinkingCap overlay tools.
- FlashInfer 0.6.17 changes are distributed as `flashinfer-overlays/`, assembled locally into normal and OEA variants rather than shipping full dependency trees.
- Removed for licensing reasons: the MUSA `fused_add_rms_norm_kernel.mu` and the Hunyuan3D `rasterizer.cpp`, `rasterizer_gpu.cu`, `rasterizer.h` and `mesh_processor.cpp`. The MUSA fused-add-RMSNorm kernel and Hunyuan3D rasterizer/mesh extensions are therefore unavailable in this release.

## Credits and license

Built on jpezzulli/sglang-rtxpro6000 at 16e5682 plus the flash-next-fast series and the 2026-10-27 update by aiueo52 and contributors ([public base (2026-08-27)](https://github.com/jpezzulli/sglang-rtxpro6000/tree/16e5682aad6e3335f38ec8d5711176278f12a086)), and on [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0, see `LICENSE`). The base fork's Flash-Next model code adapts the unmerged SGLang [PR #36497](https://github.com/sgl-project/sglang/pull/36497), and its GDN MTP state adapts [PR #30967 (RecoverSSM)](https://github.com/sgl-project/sglang/pull/30967); John Pezzulli contributed the SM120 integration, recovery graphs, HiCache/NIXL persistence and qualification. Retained third-party notices and per-file exceptions apply. Unless a file states otherwise, the additions and modifications by aiueo52 and contributors are licensed under Apache-2.0 (see `LICENSE`), **except the four public token-map data files, offered under CC BY-SA 4.0** (see `THIRD_PARTY_NOTICES.md`). Copyright 2026 aiueo52 and contributors.

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 0.6.17 (Apache-2.0): we changed 7 of its files in each variant and added 1 file (normal only).
- They ship in `flashinfer-overlays/` with their original copyright and license headers kept.
- Each file carries a notice stating that we changed it (or added it).
- They are listed, with what we changed, under "Modified FlashInfer files" in `THIRD_PARTY_NOTICES.md`.
- The rest of FlashInfer is not redistributed here; you install it from PyPI. Other retained third-party notices and licenses are indexed in `THIRD_PARTY_NOTICES.md`. Model-weight terms are linked above.

---

## The base fork's README

The text below is jpezzulli/sglang-rtxpro6000's README at 16e5682, unchanged.

# One SGLang runtime, two Qwen3.8 configurations, one RTX PRO 6000

> One optimized SGLang runtime for a single RTX PRO 6000, with qualified
> launch configurations for both Qwen3.8-27B/DFlash2 and Qwen3.8 Flash-Next.

This repository contains the complete SGLang-derived source used on one
NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, SM120, TP=1). It is
not two builds: both model configurations run from the same patched source.

- **Qwen3.8-27B** pairs an FP8 target with `incoai/Qwen3.8-27B-DFlash2`.
- **Qwen3.8 Flash-Next** pairs an NVFP4 target with its native NEXTN MTP layer.

Flash-Next is day-one engineering. It has extensive qualification on this exact
machine—including 524K context, multimodal input, reasoning, tools, agentic
workloads, CUDA-graph recovery, and persistent prefix restoration—but it may
still contain rough edges or hardware/model-specific assumptions.

## Why this runtime exists

The important work is architectural, not merely a collection of launch flags:

- DFlash2 integration for Qwen3.8-27B, with the selector inside the draft CUDA
  graph, fused draft KV materialization, and fixed-width TRTLLM-MHA/XQA target
  verification on SM120.
- Native NEXTN MTP for Flash-Next with QSA index sharing, recovery graphs, and
  `gdn_mtp_cache_mode=none`.
- FlashInfer GDN decode and prefill for Flash-Next, plus a deliberately narrow
  SM120 FlashInfer WY output-only RecoverSSM path.
- QSA sparse attention: Triton sparse-GQA prefill, the FlashInfer QSA wrapper
  resolving to XQA on SM120, and `sgl-kernel` top-k.
- FlashInfer CUTLASS MoE for the Flash-Next target and native-MTP layer.
- Correct three-axis multimodal mRoPE in the fused QK RMSNorm+RoPE kernel.
- Target, draft, verify, prefill, and accepted-state recovery CUDA graphs.
- FP8 KV and 524,288-token factor-2 YaRN for both launch configurations.
- HiCache/NIXL persistence for complete hybrid state—not KV alone—including
  DFlash2 side state or Flash-Next GDN, Qwen4 PLE, and compressed QSA state.
- Restart restoration, representation-specific FILE namespaces, and radix-
  prefix reuse followed by long continuation.
- Measured reasoning, tool-calling, vision, long-context needle, controlled
  decode, and real agentic behavior.

## Published identity

| Item | Value |
|---|---|
| Canonical branch | `pennyroyal-main-sm120-final` |
| Current executable source | `64ecd64924fee338e3bf846a32167cd604186827` |
| Current dated tag | `sglang-rtxpro6000-20260827` |
| Earlier 27B dated tag | `qwen38-dflash2-pro6000-20260824` |
| Upstream integration base | `e7e78940168f3ba65c762a6f82fd8bc5b6ee04e3` |
| Installed SGLang | `0.5.19.dev485+g64ecd6492` |
| Python / PyTorch | `3.12.13` / `2.13.0+cu130` |
| CUDA / compiler | CUDA `13.3` (NVCC `13.3.73`) / GCC `15.3.1` |
| FlashInfer / NIXL | `0.6.17` / `1.4.0` |
| GPU / driver | RTX PRO 6000 96 GB, SM120 / `610.57.04` |

The current documentation commit sits above the executable source. It does not
change `python/`, `rust/`, or the kernel trees. Exact source lineage and current
upstream status are in [CHANGES.md](CHANGES.md) and
[PROVENANCE.md](PROVENANCE.md).

## Qualified configuration matrix

Rows intentionally separate representation, datatype, backend, and capacity.
The BF16 runtime dtype is not presented as a claim that every quantized kernel
executes or accumulates entirely in BF16.

| Property | Qwen3.8-27B + DFlash2 | Qwen3.8 Flash-Next |
|---|---|---|
| Target checkpoint | `orcarouter/Qwen3.8-27B-Uncensored-FP8` family | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Target weight format | block FP8 E4M3, 128x128 blocks | ModelOpt NVFP4, group size 16 on selected Linear modules |
| Quantized-path activations | dynamic FP8 E4M3 | NVFP4 input activations on selected Linear modules |
| Runtime dtype for unquantized tensors | BF16; includes excluded layers and BF16 `lm_head` | BF16; includes ignored layers, native MTP, PLE, QSA/GDN state-facing tensors, and vision |
| Target KV datatype | FP8 E4M3 | FP8 E4M3 |
| Speculative KV datatype | DFlash2 draft: FP8 E4M3 | native-MTP: FP8 E4M3 |
| Recurrent/GDN SSM state | FP32 | BF16 |
| Convolution state | BF16 | BF16 |
| MoE backend | Triton FP8 MoE; auto resolves to Triton with A2A `none` | FlashInfer CUTLASS for target and native MTP |
| Target attention | FlashInfer prefill; TRTLLM-MHA/XQA decode and fixed-width verify | QSA Triton sparse prefill; FlashInfer QSA wrapper resolving to XQA for sparse decode; general attention FlashInfer |
| Linear/GDN attention | Triton decode, prefill, and state-writing verify | FlashInfer decode/prefill; WY output-only verify/recovery in `none` mode |
| Speculative backend | DFlash2, 8 draft tokens, 2,048-token window | native NEXTN, 3 steps, top-k 1, 4 draft tokens |
| Served context | 524,288, factor-2 YaRN target and draft | 524,288, factor-2 YaRN |
| KV page size | 64 | 64 |
| Current GPU KV capacity | 1,118,784 target and draft tokens | 824,384 target and native-MTP tokens |
| Mamba capacity | 24 slots; maximum 5 retained states/path | 24 slots; `extra_buffer`, tracking interval 64 |
| HiCache/NIXL | 96 GB configured host tier; target KV + Mamba/GDN + DFlash2 state | 32 GB configured host tier; packed target/native-MTP KV + GDN + PLE + QSA keys |
| Maximum active requests | 4 | 4 |

The current 27B launcher uses the 24-slot/five-state setting qualified on
2026-08-26. The earlier 2026-08-24 performance release used 16 slots, a
three-state path cap, and a 1,194,496-token KV pool. Both campaigns are retained
in [RESULTS.md](RESULTS.md) rather than silently merging their allocations.

## Resolved backends: Qwen3.8-27B/DFlash2

| Component or phase | Resolved implementation | How selected or enabled | Evidence |
|---|---|---|---|
| Target prefill attention | FlashInfer | explicit hybrid prefill backend | 64K and 490K controlled prefill |
| Target decode attention | TRTLLM-MHA with XQA | explicit decode backend; SM120 supported path | controlled and agentic decode |
| Fixed-width target verification | TRTLLM-MHA/XQA with packed causal mask | local `0f159cd545` | graph-metadata regression; 1x/4x decode |
| Linear/GDN decode, prefill, verify | Triton | resolved linear backend | startup configuration and long-context suite |
| Target FP8 MoE | Triton | `auto` + A2A `none` resolves `Fp8MoEMethod` to Triton | server args plus source resolver |
| DFlash2 draft attention | FlashInfer | explicit draft backend | draft-runner startup line |
| Draft local convolution | DFlash2 local-convolution path | merged upstream PR #35371 | DFlash2 startup and decode |
| Candidate selection | folded into draft CUDA graph | upstream DFlash2 integration | graph-capture startup line |
| DFlash fused KV materialization | enabled, 5 layers / 8 KV heads / 128 head dim | upstream DFlash2 integration | allocation/startup line |
| Multimodal attention | `triton_attn` | automatic | startup log; vision smoke |
| Multimodal rotary | fused three-axis mRoPE | local `64ecd64924`, PR #35744 | numerical kernel tests and vision |
| Sampling / grammar | FlashInfer / XGrammar | automatic | startup log; reasoning and tools |
| HiCache transfer | NIXL POSIX, io_uring, O_DIRECT | explicit storage backend | restart and concurrent restoration |
| Persistent representation | target KV + Mamba/GDN + DFlash2 sidecar | local NIXL integration | namespace and restore qualification |

The selected 27B target keeps `lm_head` in BF16. The quantized-head selector
from upstream PR #35496 is present in the base but is not the source of this
checkpoint's measured speed. No separate local DeepGEMM SM120 patch is claimed;
this FP8 configuration resolves MoE to Triton.

## Resolved backends: Qwen3.8 Flash-Next

| Component or phase | Resolved implementation | How selected or enabled | Evidence |
|---|---|---|---|
| Target GDN decode | `FlashInferGDNKernel` | narrow phase override | startup dispatcher; 1x/4x decode |
| Target GDN prefill | `FlashInferGDNKernel` | narrow phase override | 64K/490K prefill and needles |
| Native-MTP draft decode | `FlashInferGDNKernel` | shared GDN dispatch | draft CUDA graphs and live decode |
| Native-MTP draft extend | `FlashInferGDNKernel` | shared GDN dispatch | draft extend graph capture |
| Active GDN target verification | FlashInfer WY output-only | local `280825c3e2`, `none` mode only | direct state parity and graph replay |
| Ordinary/tree state-writing verification | `TritonGDNKernel` | architecture gate deliberately preserved | non-`none` source dispatch and focused tests |
| Accepted-state recovery | FlashInfer WY output-only | narrow SM120 RecoverSSM route | recovery graphs BS 1-4; long continuation |
| QSA sparse prefill | Triton sparse GQA | model path + FP8 tile fix `95da38fb3b` | 64K/490K exact qualification |
| QSA sparse decode | FlashInfer QSA wrapper resolving to XQA | SM120 wrapper dispatch `c1da0eef56`; FlashInfer selects XQA on SM12x | direct backend probe and live decode |
| QSA top-k | `sgl-kernel` | automatic | startup resolution |
| QSA MTP index sharing | enabled | model path | startup line and shared-index tests |
| Target MoE | FlashInfer CUTLASS | automatic ModelOpt-NVFP4 resolution | startup resolution and live tests |
| Native-MTP MoE | FlashInfer CUTLASS | automatic speculative resolution | startup resolution and live tests |
| HyperConnection Mix | persistent Triton Mix | automatic SM120 fallback | HC numerical tests; FlashInfer gate retained |
| HyperConnection Combine | fused SGLang CUDA | Qwen4 model path | kernel tests and full suite |
| Multimodal attention | `triton_attn` | automatic | full 1,024-token vision validation |
| Sampling / grammar | FlashInfer / XGrammar | automatic | startup log; reasoning and tool suite |
| HiCache transfer | NIXL POSIX, io_uring, O_DIRECT | explicit storage backend | 64K/490K restart restoration |
| Persistent representation | packed target/native-MTP KV + complete GDN/PLE siblings + compressed QSA keys | local hybrid-pool integration | restart restore, needles, continuation |

## SM120 enablement was scoped, not indiscriminate

Several upstream checks treated SM100 as the only eligible architecture even
when an underlying kernel already compiled for SM120. This runtime distinguishes
three cases:

1. **Explicit phase override.** Automatic GDN selection used an SM100 helper and
   therefore did not select FlashInfer on SM120. FlashInfer 0.6.17 already
   supported the BF16 decode and prefill operations used here. The launcher
   selects only those phases with `--linear-attn-decode-backend flashinfer` and
   `--linear-attn-prefill-backend flashinfer`. Long prefill, exact needles,
   decode, and vision/reasoning qualification cover the resulting runtime.

2. **Narrow source integration.** RecoverSSM needed FlashInfer WY output-only
   state output and accepted-state recovery. FlashInfer already contained an
   `sm_120a` implementation, but SGLang did not route SM120 into it. Commit
   `280825c3e2` enables only the output/recovery route used by
   `gdn_mtp_cache_mode=none` and preserves Qwen4 PLE accepted-state commit
   ordering. Direct tests cover all accepted lengths for four draft tokens,
   batch sizes 1-4, mixed acceptance, positions 63/64/65, `extra_buffer`, prefix
   restoration, retraction, long continuation, and captured-graph replay.

3. **QSA wrapper enabled, backend distinguished.** Commit `c1da0eef56` lets
   SM120 enter FlashInfer's page-aligned QSA decode wrapper. On SM120 that
   wrapper resolves to XQA, an existing supported implementation; it does not
   select TRTLLM-Gen. The upstream approximately 35% statement from PR #36497
   was recorded while this resolver was SM100-only and is not attributed to
   Penny's SM120 results.

4. **Incompatible gates deliberately retained.** Forced TRTLLM-Gen fails with
   `Unsupported architecture`. Bypassing the guard and loading the exact
   BF16-Q/FP8-KV/H256/P64 cubin directly returns
   `CUDA_ERROR_NO_BINARY_FOR_GPU (209)`: the binary targets SM100 and contains
   the `sm10x_tcgen05` instruction family unavailable on SM120. Supporting it
   would require kernel/compiler adaptation upstream of SGLang; see
   [TensorRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799)
   and [FlashInfer #3628](https://github.com/flashinfer-ai/flashinfer/issues/3628).
   Ordinary FlashInfer state-writing GDN verification also remains gated and
   uses Triton, while FlashInfer HyperConnection Mix remains SM100-only and
   falls back to the qualified persistent Triton Mix.

Other bounded corrections follow the same rule. `0f159cd545` supplies XQA's
packed mask for fixed-width DFlash2 verification rather than enabling a new
global attention backend. `64ecd64924` extends the fused rotary kernel to real
three-axis mRoPE rather than dropping the fused path or ignoring H/W positions.

## Memory recovery and automatic KV sizing

| Flash-Next configuration | Intermediate SSM | Mamba slots | GPU KV capacity |
|---|---:|---:|---:|
| BF16 before RecoverSSM | 1.05 GiB | 24 | 745,600 tokens |
| RecoverSSM `none` mode | 0 | 24 | 824,384 tokens |

The sequence matters:

1. BF16 reduced recurrent-state bytes per slot.
2. Automatic Mamba sizing then expanded from 21 to 49 slots and consumed much
   of that saving.
3. The workload needed far fewer slots; 24 remained above observed demand and
   returned the rest of the budget to KV.
4. RecoverSSM removed the separate 1.05 GiB intermediate speculative SSM pool.
5. With no forced KV-token count, the measured KV pool increased by 78,784
   tokens—from 745,600 to 824,384, or 10.6%.

The physical pool is removed in `280825c3e2`, but the current
`kv_cache_configurator.py::_handle_max_mamba_cache` estimator only omits that
reserve for ReplaySSM, not RecoverSSM `none`. There is no separate allocator
correction commit in this runtime. The qualified launcher increases
`--mem-fraction-static` from `.97` to `.981`, approximately the recovered GPU
fraction, so automatic KV sizing can consume the physically freed memory. The
824,384-token result was measured, not hard-coded, but the estimator debt is
real and documented.

## Performance and qualification

These rows are observations from different model configurations and are not a
single A/B benchmark.

### Flash-Next final campaign — source `64ecd64924`

| Test | Result |
|---|---:|
| 64K cold prefill | 10,103.70 tok/s |
| ~490K cold prefill | 7,872.15 tok/s; 3/3 exact needles |
| 1x 1,024-token decode | 171.09 tok/s |
| 4x 1,024-token synchronized batch | 427.54 tok/s aggregate |
| MTP mean accepted length / rate | 2.58 / 52.74% |
| Reasoning | 97.49/100 across 139,863 completion tokens |
| Tools | 30/30 exact calls and semantically correct after review |
| Vision | complete 1,024-token validation passed |
| Sealed agentic control | 148.80 tok/s |
| Natural 3,072-token decode | 162.05 tok/s |

The four individual post-first-token rates were 115.13, 127.56, 126.64, and
122.96 tok/s. They do **not** sum to 427.54 because they use each stream's own
post-first-token interval. The aggregate is the matched batch metric:
`4,096 output tokens / 9.580393 seconds`, including TTFT and the batch tail.
These results were measured with QSA sparse decode resolving to XQA. No matched
SM120 end-to-end A/B supports a percentage claim against another QSA backend.

### Qwen3.8-27B/DFlash2 dated performance campaign

| Test | Result |
|---|---:|
| 64K prefill | 6,163.07 tok/s |
| 489,921-token prefill | 1,618.31 tok/s; 3/3 exact needles |
| 1x 1,024-token decode | 108.75 tok/s after first token |
| 4x 1,024-token decode | 390.23 tok/s aggregate |
| Reasoning, xhigh / medium | 98.26 / 95.807 |
| Medium tool suite | 30/30 tool selections and arguments; 29/30 reviewed response discipline |

Against the cited public TP1 official-FP8/MTP3 community capture, the dated
DFlash2 campaign measured +39.8% at C1, +33.3% at C4, and +4.9% at 64K prefill.
This remains directional rather than a strict A/B because checkpoint, runtime,
speculation, power, harness, output duration, and cache configuration differ.

The current 24-slot/five-state confirmation measured 6,169.18 tok/s at 64K,
1,616.29 tok/s at ~490K with all needles exact, 108.93 tok/s at C1, and
375.81 tok/s aggregate at C4. It reached a 96.92 reasoning score across 50,986
completion tokens and used at most 10 of 24 Mamba entries.

### 27B real agentic context behavior

The dated 124-request sample contained 85,156 output tokens and inputs from 183
to 350,195 tokens. All observed requests measured 131.31 tok/s median and
102.57 tok/s token-weighted. Excluding seven overlapping request intervals,
the result was 133.21 median / 125.36 weighted. Short 0-2K requests reached
165.25 median / 148.35 weighted; the non-overlapping 340-360K band measured
105.45 median / 101.69 weighted. The fastest completed request was 244.24 tok/s,
while a favorable instantaneous DFlash2 telemetry window reached 300.16 tok/s
at 7.75 accepted tokens and 0.96 acceptance. Telemetry is not reported as
sustained completed-request throughput.

### Flash-Next real agentic context behavior

A separate decontaminated 96-request window produced 100,666 output tokens at
139.5 tok/s token-weighted, 153.5 median, and 155.1 arithmetic mean. The
sustained completed-request peak was 218.8 tok/s; instantaneous telemetry
reached 247.0 tok/s for one stream and briefly 543.1 tok/s at four concurrent
requests. The 90K-279K input lane measured 138.7 weighted / 148.4 median. The
first sample after a large prefill was excluded because its telemetry interval
mixed prefill or idle time with decode.

Detailed definitions, complete 27B context bands, both 27B campaigns, and
persistence evidence are in [RESULTS.md](RESULTS.md).

## HiCache/NIXL persistence

HiCache/NIXL does not cause GPU-resident decode speed. It provides disposable,
reusable prefix state across GPU/host eviction and service restart:

```text
GPU radix state
  -> page-first HiCache host RAM, kernel I/O, write-through
  -> NIXL POSIX FILE storage, io_uring + O_DIRECT
```

Flash-Next restart evidence:

- 63,808 of 63,864 input tokens restored; 56 recomputed.
- 489,856 of 489,879 restored; 23 recomputed.
- restored 490K effective input rate: 62,040.60 tok/s.
- all three needles exact after restart.

The 62,040.60 figure is restored-prefix throughput, not cold model prefill.

The earlier 27B work also demonstrated 518,528 restored tokens with a six-token
tail in 14.64 seconds, 60,032-token namespace reuse after restart, A→B→A page-
size namespace rollback, three concurrent ~60K restores, and a 775,168-token
three-request concurrent restore with only 320 tail/page-rounding tokens
computed.

Two portable NIXL fixes from this project remain open upstream: PR #36520
isolates overlapping path registrations, and PR #36524 bounds bounce-backed
hybrid transfers. Flash-Next adds complete PLE/GDN/QSA sibling-state persistence
and load/COW ordering. [MEMORY-AND-PERSISTENCE.md](MEMORY-AND-PERSISTENCE.md)
documents the representation namespace and actual failure boundaries.

## Build and launch

There is one native build procedure and exactly two canonical launch recipes:

```bash
uv python install 3.12.13
uv venv --python 3.12.13 .venv
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15
export MAX_JOBS=24 CMAKE_BUILD_PARALLEL_LEVEL=24
export FLASHINFER_NINJA_JOBS=24 FLASHINFER_NVCC_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=24

uv pip install --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ \
  --no-build-isolation -e python
```

Set `REPO_ROOT`, `CACHE_BASE`, `NIXL_STORAGE_BASE`, and the checkpoint paths,
then choose one recipe:

```bash
configs/pennyroyal/serve-qwen38-27b-dflash2.sh
configs/pennyroyal/serve-flash-next.sh
```

Do not run them simultaneously on one GPU. [BUILD.md](BUILD.md) records exact
dependencies and [RUN.md](RUN.md) provides startup assertions, ordinary
OpenAI-compatible smoke requests, and cold/radix/NIXL cache distinctions.

## Source changes and upstream work

The cumulative history starts with the 2026-08-24 27B release, then layers the
unified runtime and Flash-Next work on the same source line. The current active
stack contains 19 commits above its integration base. Major groups are:

- Qwen3.8-27B/DFlash2: independent target/draft overrides, fixed-width XQA mask,
  NVCC host-compiler identity, request-span observability, and NIXL correctness.
- Flash-Next: Qwen4 model support, QSA/HC/PLE/native-MTP integration, SM120 QSA
  decode, FP8 QSA prefill, RecoverSSM, complete hybrid persistence, and mRoPE.
- Open project PRs: #36520, #36524, and #35584.
- Closed project submissions retained in runtime history: #35583; transient
  ragged/DSpARK PR #35586 is documented but not in the active source.
- Related upstream work: #30967, #35371, #35496, #35744, #36497, and #36644.

[CHANGES.md](CHANGES.md) lists every material current commit, the earlier dated
release hashes, exact PR links/status/heads, affected execution paths, and test
coverage.

## Scope and limitations

- One RTX PRO 6000, TP=1, exact checkpoint families, and the recorded CUDA,
  PyTorch, FlashInfer, compiler, NIXL, and SGLang source.
- SM120 paths were qualified narrowly; unsupported-by-default is not treated as
  proof of incompatibility, but no architecture gate was globally deleted.
- Benchmark observations are not guarantees for another system.
- Flash-Next and 27B results remain separate; neither model's features or
  numbers are silently attributed to the other.
- Current Flash-Next qualification ran at `64ecd64924`. The later Flash-Next-
  specific commits were not all re-benchmarked on the 27B campaign, so the
  dated/current 27B evidence is labeled by its actual runtime line.
- NIXL cleaner thresholds are whole-filesystem occupancy percentages, not an
  absolute directory byte quota.

See [LIMITATIONS.md](LIMITATIONS.md) for measurement and reproducibility detail.

## Documentation map

- [BUILD.md](BUILD.md) — one native build and dependency identity.
- [RUN.md](RUN.md) — the two canonical launch configurations.
- [BACKENDS.md](BACKENDS.md) — source-backed resolved backend and SM120 tables.
- [RESULTS.md](RESULTS.md) — controlled, reasoning, tools, context, agentic, and
  persistence measurements for both models.
- [MEMORY-AND-PERSISTENCE.md](MEMORY-AND-PERSISTENCE.md) — GPU allocation and
  hybrid state representation.
- [CHANGES.md](CHANGES.md) — cumulative release and upstream history.
- [PROVENANCE.md](PROVENANCE.md) — source, package, checkpoint, and tag identity.
- [LIMITATIONS.md](LIMITATIONS.md) — what the evidence does and does not prove.
- [`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation)
  — maintained public validation harness and result catalog.

## Engineering history

The experimental path—including DFlash2 work and the HiCache/Mooncake/NIXL
detour—is described at
[msoexpert.com](https://msoexpert.com/articles/qwen38-dflash2-rtx-pro-6000/).
Mooncake is not a fallback in this runtime; it was rejected and removed.

## License

The SGLang-derived source remains under Apache-2.0. Model checkpoints and
third-party runtimes retain their own licenses.
