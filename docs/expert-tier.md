# Expert tier

The expert tier is an opt-in LRU store for routed MoE expert weights. It keeps
K experts per layer in VRAM and copies missing experts from a host-backed mmap
source. A hot set initializes residency; it does not pin those experts forever.
Requests with at most K route entries use the decide/gather path. Larger calls
use the arena path, so reducing K can increase transfer work and reduce speed.

## Usage

The default is off. `EXPERT_TIER_K=0` adds no tier environment settings and
preserves the existing launcher behavior, including inherited SGLang settings.
To enable it, launch from the repository root. A hot set built from public text
is included at `expert-tier/hot-set.json` (see "Building the hot set"):

```bash
TARGET_MODEL=models/model CONC=8 EXPERT_TIER_K=320 ./serve-fast.sh wa
```

To fit the hot set to your own workload, build a replacement from traces first:

```bash
python tools/build_hot_set.py traces/trace-000000.npz traces/trace-000001.npz \
  --output expert-tier/hot-set.json
```

This example uses the default context length of 262,144. With K > 0, an unset
or empty `MAX_TOTAL_TOKENS` defaults to `CONTEXT_LENGTH`, or 262,144 if that is
unset or empty, and is exported. A nonempty user-supplied `MAX_TOTAL_TOKENS` is
exported unchanged. `CONTEXT_LENGTH` itself is not changed by this tier default;
K=0 leaves the existing token-budget behavior unchanged.

Lower `MAX_TOTAL_TOKENS` to reduce VRAM use; raise it to allocate more available
VRAM to the KV cache. Set it in the launch environment alongside the other
settings. `WA_MEM_FRACTION` (default 0.96) also controls the memory budget; the
launcher subtracts the display reserve from the effective fraction.

When `CONTEXT_LENGTH` exceeds 262144, `serve-local.sh` sets
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` and adds
`--json-model-override-args` with `text_config.rope_parameters`: `rope_type` yarn,
`factor` 2.0, `original_max_position_embeddings` 262144, `mrope_interleaved` true,
`mrope_section` [11,11,10], `rope_theta` 10000000, and `partial_rotary_factor` 0.25.
There is no need to edit the model configuration.

| Setting | Default | Meaning |
| --- | --- | --- |
| `EXPERT_TIER_K` | `0` | Resident experts per layer, integer 0..512; 0 disables launcher tier settings |
| `EXPERT_TIER_HOT` | `expert-tier/hot-set.json` inside the repository | Initialization JSON; must exist when K > 0 |
| `EXPERT_TIER_KEEP` | `60` | Host page keeper interval in seconds; 0 disables it, otherwise 5..3600 |
| `ADAPTIVE_CONFIG` | Existing launcher selection | Actual adaptive table used for the warning and launch |

The existing table selection is preserved: CONC=8 defaults to
`adaptive/w16_c8.json`; other supported concurrency settings default to
`adaptive/w16_c4.json`. Override `ADAPTIVE_CONFIG` to select another table,
including `adaptive/w16_c6.json` for CONC=6.

With K > 0, the launcher exports the measured configuration: pruning outside the
prologue (`SGLANG_MOE_PRUNE_IN_PROLOGUE=0`), source mmap, gather v2 with chunk
8192 and 376 CTAs, pretouch 1, decide sort, prediction 0, timing 0, arena v2,
prefetch 0, fused 0, policy lru, and `SGLANG_RND_PLE_META=1`. K, HOT, and
KEEP map to their `SGLANG_EXPERT_TIER_*` counterparts.

Use `./serve-fast.sh wa --print` with the same environment to emit JSON containing
the resolved environment and arguments without starting the server or creating
cache directories. An absent hot-set file is an error even in this mode.

For the adaptive table warning, M is the maximum of
`batch_size * (candidate_steps + 1)` over supported launch batches, routing each
batch to the greatest table bucket not exceeding it (or the first bucket).
The extra row is the verification/extend row. Top-k is 10, so the fast path
requires `10 * M <= K`. The c8 table gives M=32 and required K=320. Insufficient
K prints a warning to stderr and continues. This table-based warning describes
adaptive verification; prefill and explicit argument overrides may produce
larger calls and take the arena path.

## Building the hot set

Enable recording with `SGLANG_EXPERT_TIER_TRACE` set to an absolute trace
directory during a representative tier-enabled run. Recording adds overhead;
leave it unset for normal serving. The recorder writes `trace-*.npz` snapshots.
Pass the desired files to `build_hot_set.py`; no model, GPU, or serving imports
are required by the converter (NumPy is required).

The recorder stores `hdr [N,4]` and `counts [N,512]` alongside its prediction
and demand fields. Counts preserve token-route multiplicity and exclude ID -1
and weight 0 independently, including routes whose ID was retained. Counts are
recorded only when tracing is enabled and only for the small decide/gather calls
that the recorder observes; arena/prefill calls are not included.

Legacy snapshots without `counts` use the `dem` presence bits instead: frequency
then means calls using an expert, not token routes. Such snapshots cannot recover
multiplicity or distinguish a retained ID whose weight alone was zero. Use
normal ID-removal pruning for legacy traces, or collect a fresh trace for exact
nonzero-route counts. The converter also accepts raw `layers [N]`, `ids [N,...]`,
and matching `weights`, excluding ID -1 and weight 0 independently.

Do not mix legacy call-presence counts with route counts when a uniform frequency
measure is needed. Multiple input files are summed; duplicate file paths are
rejected. Snapshot ring losses cannot be reconstructed, and separately collected
runs should be checked for overlapping records.

Output is `{"K": 512, "layers": [...]}` with 48 lists of all 512 expert IDs.
Each layer is sorted by decreasing frequency, breaking ties by ascending ID;
unseen experts are included. The runtime loads the first K entries per layer.
Use representative workloads: hot-set initialization is workload dependent.

The bundled `expert-tier/hot-set.json` was built this way from 109 trace files of
a K320, `CONC=8` run over public text only: the first user turn of 64 Japanese
OpenAssistant oasst2 conversations, 256 GSM8K test questions with reasoning off,
and the interference check over CPython 3.13.0 `Lib/` source text. Initial
residency changes transfer work only; it does not change which experts a token
is routed to.

## Requirements and limits

The integration supports the ModelOpt NVFP4 MoE path with the FlashInfer CUTLASS
backend, MoE TP=1 and EP=1, 48 routed layers, 512 experts, and top-k 10.
Interleaved w13, fused shared experts, the draft MoE GEMV path on tiered layers,
and an armed OEA prologue are unsupported. Use the normal edition configuration
rather than enabling incompatible prologues. CPU tests validate logic, not GPU
serving compatibility.

Host RAM and storage must support the mmap source and page residency; lowering
VRAM use shifts work and memory pressure to the host. K is a per-layer weight
residency budget, not a total VRAM limit. KV cache, context length, concurrency,
arenas, and other allocations still affect peak VRAM. Transfers and acceptance
rates affect performance; the measurements below are specific configurations,
not guarantees. Model weights are provided separately.

## Reported measurements

These are supplied measurements, not rerun by the CPU-only release tests.

* This launcher as shipped, `CONC=8 EXPERT_TIER_K=320 ./serve-fast.sh wa` with
  the bundled hot set: context 262,144, `MAX_TOTAL_TOKENS=262144`, HiCache
  32 GB (the default), prefill chunk 8192, effective memory fraction 0.886778.
  Peak VRAM **66,230 MiB**. Host MemAvailable fell by about **107 GB** during
  the run, including the 32 GB HiCache pool.

| Concurrency | Per-request tok/s | Aggregate tok/s |
| --- | ---: | ---: |
| c1 | 232 | 232 |
| c2 | 206 | 409 |
| c4 | 157 | 610 |
| c8 | 112 | 869 |

Against profile W4 (draft 4, no adaptive mode) with the tier at K320 and context
262,144 (earlier launch method), per-request speed was within 2% at c1 and c2,
2.4% lower at c4, and 10% higher at c8 (aggregate 869 versus 791). Code editing
at c1 ran at **615 tok/s** with a mean accepted length of about 10. Prompt
log-probabilities over 8 Japanese prompts (1,479 tokens) were identical to that
reference. GSM8K, 1,319 questions: **545 versus 570**; 121 questions were solved
only by the reference and 96 only by this launch (exact McNemar p ≈ 0.10). The two
524k configurations below scored 549 (W4) and 552 (w16_c8) against the same
reference. 63k interference check: **8/8**, **0 leaks**.

* K320, context 524,288 (YaRN factor 2), profile wa with table w16_c8:
  peak VRAM **72,922 MiB**. The 524k needle check passed **3/3**, with
  **78.6 seconds to the first character**. These measurements used an earlier
  launch method with running 8, graph batch sizes 1–8, mamba 40, prefill chunk
  16384, `MAX_TOTAL_TOKENS=557056`, memory fraction 0.886778, and no HiCache;
  they were not measured through this launcher's current `CONC=8` path.

| Concurrency | Per-request tok/s | Aggregate tok/s |
| --- | ---: | ---: |
| c1 | 227 | 227 |
| c2 | 206 | 402 |
| c4 | 158 | 623 |
| c8 | 97 | 763 |

No quality difference was observed against profile W4 (draft 4, no adaptive
mode) with the same tier configuration (K320, context 524,288):
GSM8K, 1,319 questions, **552 versus 549**; reasoning enabled, 200 questions ×
3 runs, **585 versus 586**; 63k interference check, **8/8**, **0 leaks**.
Japanese text at c1 was approximately **7% slower** than that same profile W4
(draft 4) configuration because longer drafts were less likely to be accepted.
These checks bound the quality claim to those workloads.

* Lower-VRAM configuration: K216, context 262,144, profile W4 (draft 4),
  running 4, `MAX_TOTAL_TOKENS=262144`:
  peak VRAM **48,172 MiB** and speed **86%** of the no-tier configuration,
  averaged over **5 measurements**.
