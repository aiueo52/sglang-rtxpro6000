# NGRAM_CHAIN

`NGRAM_CHAIN` is an EAGLE-compatible speculative provider for Qwen4-Exp MTP.
For each request it keeps a bounded host copy of the prompt and committed output,
then looks up the request's current suffix from the largest configured n-gram to
the smallest. A full match supplies the linear chain that followed the previous
occurrence. The existing EAGLE tree builder, target verification, sampling, KV
compaction, draft-extend, and GDN commit paths are unchanged.

The index records an occurrence only once all `speculative_num_steps` following
tokens exist. Consequently a partial continuation is a miss and falls back to
MTP; NGRAM_CHAIN never pads or verifies a short chain. Accepted tokens are staged
after verify and appended to the host index at the next draft boundary. Per-key
position deques retain the last in-window occurrence while supporting incremental
window eviction.

## Flags

- `--speculative-algorithm NGRAM_CHAIN`
- `--speculative-ngram-chain-min-size` (default `4`)
- `--speculative-ngram-chain-max-size` (default `12`)
- `--speculative-ngram-chain-window-size` (default `32768` tokens)
- `--speculative-num-steps` controls the full proposal width.

NGRAM_CHAIN requires EAGLE `topk=1`; therefore
`speculative_num_draft_tokens` is `speculative_num_steps + 1`. Its delta
`draft_probs` place probability one on each proposed token, which is the exact
proposal distribution used when rejection sampling is enabled at temperature
greater than zero.

## v0 limitations

- The optimized target is batch size 1.
- A batch bypasses MTP only when every request has a full-width hit. Mixed
  hit/miss batches fall back to MTP as a whole.
- Partial continuations are misses; there is no truncation or padding policy.
- Adaptive speculative step counts are unsupported because index eligibility is
  built for one fixed chain width.
- Request state is released when its request leaves the active decode batch.

## GPU validation checklist

- Needle test: repeat a unique prompt/output span, confirm the expected 15-token
  chain is drafted, and verify no MTP draft CUDA graph launches on the hit.
- Record the acceptance-length histogram and tokens-per-verify for NGRAM_CHAIN,
  then compare with a `NEXTN` baseline using the same requests and sampling args.
- Measure per-step time against the `NEXTN` baseline and confirm
  `_flush_pending_commits` does not stall the overlap scheduler with a fresh
  `.cpu()` D2H of staged commit tensors. If it does, consume the result
  processor's already-synchronized CPU token lists instead of issuing another
  D2H transfer.
- Repeat at temperature greater than zero with rejection sampling enabled;
  confirm each draft-probability row sums to one and is nonzero only at the
  proposed token, then compare output distributions against non-spec decoding.
- Exercise batch size 1 misses and mixed batches, confirming unchanged MTP
  fallback, KV compaction, and GDN cache commit behavior.
