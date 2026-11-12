# Results

This document preserves the complete useful 27B/DFlash2 measurement record and
adds Flash-Next as a separate configuration of the same SGLang source. Metrics
with different timing windows are never added together or presented as one
interchangeable throughput number.

## Measurement definitions

- **Cold prefill:** prompt tokens divided by time to first token with no matching
  radix or persistent prefix.
- **Completed-request effective decode:** output tokens divided by SGLang's
  post-prefill elapsed span. This excludes initial prefill but includes waits,
  re-prefill after retraction, and interference after the prefill boundary.
- **Per-request median:** median of completed-request effective decode rates.
- **Token-weighted effective decode:** `sum(output_tokens) /
  sum(post_prefill_seconds)`.
- **Instantaneous server throughput:** SGLang periodic `gen throughput`; a short
  telemetry window, not completed-request throughput.
- **Concurrent aggregate:** all group output tokens divided by synchronized
  group makespan.
- **Restored-prefix effective prefill:** input tokens divided by TTFT after
  NIXL restoration. It is not cold model prefill.

## Flash-Next final campaign

Source: `64ecd64924fee338e3bf846a32167cd604186827`.

| Workload | Context / concurrency | Result | Correctness / note |
|---|---|---:|---|
| Cold prefill | ~64K, C1 | 10,103.70 tok/s | normal chat request |
| Cold prefill + needles | ~490K, C1 | 7,872.15 tok/s | 3/3 exact needles |
| Decode | 1 x 1,024 output | 171.09 tok/s | post-first-token |
| Decode | 4 x 1,024 output | 427.54 tok/s aggregate | synchronized batch makespan |
| Native NEXTN acceptance | final suite | 2.58 mean accepted / 52.74% | live speculative telemetry |
| Reasoning | full suite | 97.49/100 | 139,863 completion tokens |
| Tools | 30 invocations | 30/30 exact calls and semantic success | legacy literal checker was 27/30 |
| Vision | complete request | passed | 1,024-token validation |
| Sealed agentic control | controlled request | 148.80 tok/s | normal API path |
| Natural decode | 3,072 output | 162.05 tok/s | normal response |

Flash-Next QSA sparse decode resolved to XQA inside FlashInfer's wrapper for
this campaign. These measured results remain valid; they are not evidence for
TRTLLM-Gen on SM120, and no matched end-to-end backend percentage is claimed.

### Four-request timing reconciliation

The reported individual streams were 115.13, 127.56, 126.64, and 122.96
tok/s. Each uses 1,023 post-first-token tokens over its own 8.02-8.89 second
decode interval. The aggregate instead uses all 4,096 completion tokens over
the synchronized 9.580393-second batch, including TTFT and the slowest tail:

```text
4096 / 9.580393 = 427.539869 tok/s
```

The stream values are useful latency observations but are not additive. The
427.54 figure is the matched aggregate measurement.

### Flash-Next real agentic sample

| Measurement | Result |
|---|---:|
| Completed requests | 96 |
| Output tokens | 100,666 |
| Token-weighted effective decode | 139.54 tok/s |
| Per-request median | 153.48 tok/s |
| Arithmetic mean | 155.05 tok/s |
| Sustained completed-request peak | 218.84 tok/s |
| Instantaneous C1 peak | 246.99 tok/s |
| Brief instantaneous C4 peak | 543.07 tok/s |
| 90K-279K lane | 138.66 weighted / 148.41 median tok/s |

The first telemetry sample after a large prefill was excluded because its
window mixed prefill or idle time with decode. The sample showed no high-
context cliff; acceptance variation caused most visible throughput movement.

### Flash-Next persistent restoration

| Prefix | Restored | Recomputed | Effective restored-prefix rate | Result |
|---|---:|---:|---:|---|
| ~64K | 63,808 | 56 | not used as a cold-prefill comparison | coherent response |
| ~490K | 489,856 | 23 | 62,040.60 tok/s | 3/3 needles exact |

## Qwen3.8-27B/DFlash2 current-launcher confirmation

This 2026-08-26 campaign qualified the current 24-slot, five-state path setting
at runtime-line source `8e197ed3af`. It is separate from the earlier dated
performance release below.

| Property | Result |
|---|---:|
| Target/draft FP8 KV capacity | 1,118,784 tokens each |
| Mamba slots / state-path cap | 24 / 5 |
| Maximum observed Mamba entries | 10 |
| 64K prefill | 6,169.18 tok/s |
| ~490K prefill | 1,616.29 tok/s; 3/3 needles exact |
| 1x decode | 108.93 tok/s |
| 4x aggregate decode | 375.81 tok/s |
| 4x individual post-first-token | 98.69 / 103.00 / 96.51 / 105.61 tok/s |
| DFlash2 acceptance during C4 | 2.92 mean accepted / 27.67% |
| Reasoning | 96.92/100 across 50,986 completion tokens |
| Reasoning token-weighted effective decode | 156.70 tok/s |
| Tool selection and arguments | 30/30 |
| Reviewed response discipline | 29/30; no tool-use or security failure |

## Qwen3.8-27B/DFlash2 dated performance release

Tag: `qwen38-dflash2-pro6000-20260824`. This is the established 27B result set
and is retained exactly rather than overwritten by the later allocation.

### Controlled results

| Workload | Context | Concurrency | Effort | Output | Result | Notes |
|---|---:|---:|---|---:|---:|---|
| Prefill | 63,906 | 1 | xhigh request | 28 | 6,163.07 prompt tok/s | 10.369 s TTFT |
| Prefill + needles | 489,921 | 1 | xhigh request | 203 | 1,618.31 prompt tok/s | 3/3 exact |
| Decode ceiling | 136 prompt | 1 | xhigh | 1,024 | 108.75 tok/s | post-first-token; 107.52 makespan |
| Decode ceiling | 136 each | 4 | xhigh | 4 x 1,024 | 390.23 tok/s aggregate | group makespan |
| Reasoning active decode | case-dependent | 3 | xhigh | 70,770 group tokens | 396.78 server tok/s median | telemetry while C3 active |
| Reasoning active decode | case-dependent | 3 | medium | suite-dependent | 486.19 server tok/s median | acceptance 4.657 / 0.522 |

The four individual decode rates were 120.79, 99.78, 102.82, and 104.26 tok/s.
As with Flash-Next, these are post-first-token per-stream windows, while 390.23
uses the synchronized group makespan; they are not additive.

### Public TP1 directional baseline

The cited community baseline is the
[`local-inference-lab/rtx6kpro` Qwen3.8-27B catalog](https://github.com/local-inference-lab/rtx6kpro/blob/7ceba1df33bb9c76060a251bba2d851b4f37a485/models/qwen38-27b.md)
at commit `7ceba1df33bb9c76060a251bba2d851b4f37a485`. Its closest TP1 row used the
official FP8 checkpoint, vLLM Gilded Gnosis r31, MTP3, FP8 KV, prefix caching,
and one RTX PRO 6000.

| Metric | Community official-FP8/MTP3 | DFlash2 dated release | Directional change |
|---|---:|---:|---:|
| C1 near empty context | 77.8 tok/s | 108.75 tok/s | +39.8% |
| C4 near empty context | 292.7 tok/s | 390.23 tok/s | +33.3% |
| 64K prefill | 5,877 tok/s | 6,163 tok/s | +4.9% |

This is not a strict A/B. Checkpoint, runtime, speculation, power, client,
prompt, duration, and cache/offload configuration differ. No “fastest” claim is
made.

### Reasoning and tools

| Qualification | Result |
|---|---:|
| xhigh reasoning | 98.26/100 dimension-weighted |
| medium reasoning | 95.807/100 dimension-weighted |
| Medium completion-token reduction vs xhigh | 80.4% |
| Medium summed-request-time reduction vs xhigh | 84.74% |
| Medium completion tokens / summed request second | 160.21 tok/s |
| xhigh completion tokens / summed request second | 124.74 tok/s |
| Medium tool suite | 27/30 automatic; 30/30 exact calls; 29/30 reviewed semantic |

The two automatic tool misses caused by wording/checker mismatch were retained
as raw checker outcomes; the reviewed result distinguishes those from the one
substantive response-discipline miss.

### Real agentic completed requests

The sanitized sample spans 2026-08-24 15:56:11-18:18:51 EDT and contains 124
completed requests, 85,156 output tokens, and 183-350,195 input tokens. It
begins one hour after the reasoning/tool suite ended.

#### Non-overlapping request intervals

| Input context | Requests | Output tokens | Median effective decode | Token-weighted effective decode |
|---|---:|---:|---:|---:|
| 0-2K | 34 | 12,791 | 165.68 tok/s | 175.46 tok/s |
| 64-100K | 5 | 1,250 | 168.01 tok/s | 151.46 tok/s |
| 100-150K | 14 | 13,901 | 133.25 tok/s | 130.36 tok/s |
| 150-262K | 2 | 605 | 117.96 tok/s | 116.10 tok/s |
| 300-325K | 21 | 19,944 | 110.81 tok/s | 116.69 tok/s |
| 325-340K | 18 | 11,263 | 133.03 tok/s | 130.98 tok/s |
| 340-360K | 23 | 14,415 | 105.45 tok/s | 101.69 tok/s |
| **All non-overlapping** | **117** | **74,169** | **133.21 tok/s** | **125.36 tok/s** |

#### Full sample, including interference

| Input context | Requests | Output tokens | Median effective decode | Token-weighted effective decode |
|---|---:|---:|---:|---:|
| 0-2K | 36 | 13,118 | 165.25 tok/s | 148.35 tok/s |
| 2-64K | 1 | 4,123 | 43.02 tok/s | 43.02 tok/s |
| 64-100K | 5 | 1,250 | 168.01 tok/s | 151.46 tok/s |
| 100-150K | 15 | 16,922 | 130.25 tok/s | 124.36 tok/s |
| 150-262K | 3 | 676 | 112.29 tok/s | 111.06 tok/s |
| 300-325K | 21 | 19,944 | 110.81 tok/s | 116.69 tok/s |
| 325-340K | 18 | 11,263 | 133.03 tok/s | 130.98 tok/s |
| 340-360K | 25 | 17,860 | 103.81 tok/s | 74.83 tok/s |
| **All observed** | **124** | **85,156** | **131.31 tok/s** | **102.57 tok/s** |

Seven overlapping requests measured 68.37 tok/s median and 46.05 tok/s
weighted. That is valid user experience under interference, not an isolated
engine-speed estimate.

### DFlash2 acceptance

Across single-running-request telemetry, median instantaneous throughput was
106.22 tok/s and mean accepted length was 3.81. The strongest window reached
300.16 tok/s at 7.75 accepted tokens and 0.96 acceptance. The fastest completed
request reached 244.24 tok/s and contained successive 258.26, 276.34, and
278.40 tok/s telemetry windows. At 340-360K, single-request telemetry measured
90.10 tok/s median with 3.32 mean accepted length and 0.332 acceptance.

### 27B HiCache/NIXL qualification

- 518,528 tokens restored with a six-token tail in 14.64 seconds; all needles
  exact.
- Identical 60K configuration restored 60,032 tokens after restart in 3.16
  seconds.
- Changing page size selected another namespace; restoring the original setting
  selected and reused the original namespace.
- Three approximately 60K requests restored concurrently after restart in 6.92
  seconds.
- 375K/200K/200K concurrent restoration reused 775,168 tokens and computed only
  320 tail/page-rounding tokens.

These results establish prefix persistence. They do not explain or cause
DFlash2's GPU-resident decode rate.

The dated tag retains the sanitized raw 27B agentic log and machine-generated
summaries. The current maintained harness and published result catalog are in
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).

## Experimental QSA relative prefill

These measurements compare `SGLANG_QSA_RELATIVE_PREFILL=0` with `1`.
The knob is experimental and off by default; outputs are not bit-exact or
token-identical. See [README.md](README.md#experimental-qsa-relative-prefill)
for dispatch requirements and mechanism, and [PROVENANCE.md](PROVENANCE.md#qsa-relative-prefill-backport)
for the Apache-2.0 backport source.

### Where it was measured

- One RTX PRO 6000 Blackwell Max-Q (the repository's usual machine), one stream (c=1), the `wa` mode of `serve-fast.sh`,
  context length 208,896, memory fraction 0.90, thinking off.
- Measured on a pre-release tree whose two QSA files are the same as this change apart from the knob name and the comment
  header. The speed and quality numbers below have NOT been re-measured on this tree (only the operation check at the end
  of this section ran on it), and not on other modes, not at concurrency above 1.

### Speed (knob 0 -> knob 1)

- Server-side prefill time for 4,096 new tokens after 200,000 cached tokens (2026-11-11, 6 measurements per arm,
  alternating launches): median 0.431 s -> 0.343 s (-20.3%); ranges 0.421-0.440 s and 0.340-0.350 s.
  In that test the 16-token continuations were not identical in every pair.
- Client-side wall time for a request that appends about 4,000 new tokens after 200,000 cached tokens, including a
  5-token answer and the HTTP round trip (2026-11-12, 16 requests per arm over two sessions of 4 launches each):
  mean 1.115 s -> 0.984 s (about -12%). Per session: about -17% (6 per arm) and about -9% (10 per arm); launch-to-launch
  variation is about 0.1 s, the same size as the gain.
- Same client-side measurement at 100,000 cached tokens (6 per arm): mean 0.652 s -> 0.611 s; the two ranges overlap.
- Reading a 200,000-token prompt from scratch (one read per launch, 4 launches per arm over two sessions):
  17.0, 17.0, 16.9, 17.7 s -> 15.8, 16.2, 16.3, 16.3 s (about -6%).
- Short questions after the long prompt is cached: no measurable difference.

### Answer quality (2026-11-12)

- Task: find one numbered record in a synthetic haystack of about 200,000 tokens and answer with its number
  ("single lookup"), plus a variant where the haystack holds records with near-identical ids ("similar id").
- Main run: 4 launches in off/on/on/off order, 300 questions per launch (240 single lookup + 60 similar id), 1,200 in total.
  The pass rule was fixed before the run: on may be at most 15 questions below off out of 600.

| | all (600) | single lookup (480) | similar id (120) |
|---|---|---|---|
| knob 0 | 536 | 420 | 116 |
| knob 1 | 538 | 423 | 115 |

- Per launch: knob 0 gave 270 and 266; knob 1 gave 268 and 270. Questions whose result flipped between two launches of the
  same arm: 8 and 10; between the arms: 14 and 8.
- Two smaller earlier runs with different haystacks, 200,000-token single lookup: 23/32 vs 25/32, and 90/96 vs 90/96
  (knob 0 vs knob 1). At 32,000 and 100,000 tokens single lookup and similar id were at or near 100% in both arms.

### Operation check on this tree (2026-11-12)

- Run on this tree with the change applied. One launch per arm (knob 0, then knob 1), the same 60 questions per arm on one
  synthetic haystack of about 200,000 tokens: 48 single lookup + 12 similar id.
- Correct answers: knob 0 53/60 (42/48 single lookup, 11/12 similar id); knob 1 52/60 (43/48, 9/12).
  The rule fixed before the run: knob 1 may be at most 6 questions below knob 0 out of 60.
- The knob-1 launch logged `QSA_RELATIVE_PREFILL_ACTIVE`; the knob-0 launch did not.
- This is an operation check only: 60 questions and one launch per arm are too few to measure quality or speed.

### Not tested

- Real documents, code or conversations; thinking mode; multi-step reasoning over long contexts (with thinking off the
  model fails two-step lookups at 100,000 tokens and above with either setting); concurrency above 1; other serve modes.
