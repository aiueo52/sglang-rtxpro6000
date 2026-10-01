"""Sparse target-only sampling verify (SGLANG_OPT_SPEC_SPARSE_VERIFY).

The dense verify in eagle_utils.eagle_sample builds full-vocabulary target
probabilities for every draft row: softmax over V, FlashInfer top-k renorm,
AIR top-p renorm (and min-p), a zeros_like draft buffer, then
`tree_speculative_sampling_target_only` reads one entry per draft token and
makes a full-row pass for the final draw. When every request has
top_k <= KP, only the KP largest logits of a row can end up non-zero, so this
path works on those:

1. The KP largest logits, sorted (KP >= max top_k + a margin for ties):
   torch.topk, or FlashInfer's radix top-k with SGLANG_OPT_SPEC_SPARSE_TOPK.
2. `_sparse_target_probs_kernel`, one program per row: temperature, softmax
   over the kept entries, top-k keeping ties (FlashInfer: p >= pivot), top-p
   keeping ties (AIR top-p: p >= the value where the cumulative mass reaches
   top_p), optional min-p, renormalize. Renormalizing once over the kept set
   is the same distribution as the dense chain up to rounding.
3. `_tree_sampling_target_only_sparse_kernel`, one program per request: the
   same tree walk as the CUDA kernel; the final draw takes its cumulative sum
   in token-id order like the dense kernel, so equal coins give equal tokens
   except where a coin lands within rounding of a bucket edge.

Known difference: ties at the top-k boundary that extend past KP entries are
cut (KP leaves a margin of at least 8 entries past max top_k).
"""

import torch
import triton
import triton.language as tl
from flashinfer import top_k as flashinfer_top_k

# Largest top_k the sparse verify accepts; larger (or TOP_K_ALL) -> dense path.
SPARSE_VERIFY_MAX_K = 248
# Slots past max top_k for ties at the k-th value; arbitrary, no tie past it seen in tests.
_TIE_MARGIN = 8


def sparse_verify_width(max_top_k: int) -> int:
    """Support width KP for a batch whose largest top_k is max_top_k; 0 = use
    the dense verify."""
    if max_top_k < 1 or max_top_k > SPARSE_VERIFY_MAX_K:
        return 0
    return max(16, triton.next_power_of_2(max_top_k + _TIE_MARGIN))


@triton.jit
def _sparse_target_probs_kernel(
    Vals,  # [N, KP] largest logits per row, sorted descending
    Temps,  # [bs]
    TopKs,  # [bs] int32
    TopPs,  # [bs]
    MinPs,  # [bs]
    Out,  # [N, KP] float32
    stride_v,
    stride_o,
    NUM_DRAFT: tl.constexpr,
    KP: tl.constexpr,
    APPLY_TOP_P: tl.constexpr,
    APPLY_MIN_P: tl.constexpr,
):
    row = tl.program_id(0)
    req = row // NUM_DRAFT
    offs = tl.arange(0, KP)
    x = tl.load(Vals + row * stride_v + offs).to(tl.float32)
    x = x / tl.load(Temps + req).to(tl.float32)
    k = tl.load(TopKs + req)
    k = tl.minimum(tl.maximum(k, 1), KP)
    # top-k: keep everything >= the k-th largest value, ties included.
    pivot = tl.min(tl.where(offs < k, x, float("inf")), axis=0)
    keep = x >= pivot
    m = tl.max(x, axis=0)
    p = tl.where(keep, tl.exp(x - m), 0.0)
    p = p / tl.sum(p, axis=0)
    if APPLY_TOP_P:
        # rows are sorted, so the first entry whose inclusive cumulative mass
        # reaches top_p has the largest p among those that do; keep p >= it.
        top_p = tl.load(TopPs + req).to(tl.float32)
        c = tl.cumsum(p, axis=0)
        thr = tl.max(tl.where(c >= top_p * tl.sum(p, axis=0), p, 0.0), axis=0)
        keep = keep & (p >= thr)
    if APPLY_MIN_P:
        min_p = tl.load(MinPs + req).to(tl.float32)
        p_max = tl.max(tl.where(keep, p, 0.0), axis=0)
        keep = keep & (p >= p_max * min_p)
    q = tl.where(keep, p, 0.0)
    q = q / tl.sum(q, axis=0)
    tl.store(Out + row * stride_o + offs, q)


def sparse_target_probs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    num_draft_tokens: int,
    kp: int,
    apply_top_p: bool,
    apply_min_p: bool,
    use_flashinfer_topk: bool,
):
    """Returns (probs [N, kp] float32, token ids [N, kp] int64) for the
    N = bs * num_draft_tokens rows of `logits`."""
    if use_flashinfer_topk:
        # deterministic=True sorts inside the top-k kernel (no torch.sort + gather).
        vals, idx = flashinfer_top_k(input=logits, k=kp, sorted=True, deterministic=True)
    else:
        vals, idx = torch.topk(input=logits, k=kp, dim=-1, largest=True, sorted=True)
    out = torch.empty((logits.shape[0], kp), dtype=torch.float32, device=logits.device)
    _sparse_target_probs_kernel[(logits.shape[0],)](
        vals,
        temperatures.reshape(-1),
        top_ks.reshape(-1),
        top_ps.reshape(-1),
        min_ps.reshape(-1),
        out,
        vals.stride(0),
        out.stride(0),
        NUM_DRAFT=num_draft_tokens,
        KP=kp,
        APPLY_TOP_P=apply_top_p,
        APPLY_MIN_P=apply_min_p,
        num_warps=1 if kp <= 64 else 2,
    )
    return out, idx


@triton.jit
def _tree_sampling_target_only_sparse_kernel(
    Predicts,  # [bs * NUM_DRAFT] int32, mutable
    AcceptIndex,  # [bs, NUM_SPEC] int32, mutable
    AcceptTokenNum,  # [bs] int32, mutable
    Candidates,  # [bs, NUM_DRAFT]
    RetriveIndex,  # [bs, NUM_DRAFT]
    RetriveNextToken,  # [bs, NUM_DRAFT]
    RetriveNextSibling,  # [bs, NUM_DRAFT]
    Coins,  # [bs, NUM_DRAFT]
    CoinsFinal,  # [bs]
    Q,  # [bs * NUM_DRAFT, KP] float32
    Idx,  # [bs * NUM_DRAFT, KP] int64
    threshold_single,
    threshold_acc,
    NUM_SPEC: tl.constexpr,
    NUM_DRAFT: tl.constexpr,
    KP: tl.constexpr,
    VOCAB: tl.constexpr,
):
    bx = tl.program_id(0)
    base = bx * NUM_DRAFT
    offs = tl.arange(0, KP)

    coin = tl.load(Coins + base).to(tl.float32)
    last_accept_index = tl.load(RetriveIndex + base).to(tl.int32)
    tl.store(AcceptIndex + bx * NUM_SPEC, last_accept_index)
    num_correct = 0
    cur_row = 0
    cur_index = 0
    prob_acc = 0.0
    # entries of the current row that belong to rejected siblings
    rej = tl.zeros([KP], dtype=tl.int32)

    j = 1
    walking = 1
    while (j < NUM_SPEC) and (walking == 1):
        cur_index = tl.load(RetriveNextToken + base + cur_index).to(tl.int32)
        found = 0
        while (cur_index != -1) and (found == 0):
            draft_index = tl.load(RetriveIndex + base + cur_index).to(tl.int32)
            tok = tl.load(Candidates + base + cur_index).to(tl.int64)
            row_off = (base + cur_row) * KP
            idx_row = tl.load(Idx + row_off + offs)
            q_row = tl.load(Q + row_off + offs)
            q_tok = tl.sum(tl.where(idx_row == tok, q_row, 0.0), axis=0)
            prob_acc += q_tok
            if (coin <= prob_acc / threshold_acc) or (q_tok >= threshold_single):
                prob_acc = 0.0
                cur_row = cur_index
                coin = tl.load(Coins + base + cur_index).to(tl.float32)
                tl.store(Predicts + last_accept_index, tok.to(tl.int32))
                num_correct += 1
                tl.store(AcceptIndex + bx * NUM_SPEC + num_correct, draft_index)
                last_accept_index = draft_index
                rej = tl.zeros([KP], dtype=tl.int32)
                found = 1
            else:
                rej = tl.where(idx_row == tok, 1, rej)
                cur_index = tl.load(RetriveNextSibling + base + cur_index).to(tl.int32)
        if found == 0:
            walking = 0
        j += 1
    tl.store(AcceptTokenNum + bx, num_correct)

    # final draw from relu(target - rejected siblings), cumulative sum in
    # token-id order, first entry whose cumulative mass exceeds u
    coin_f = tl.load(CoinsFinal + bx).to(tl.float32)
    row_off = (base + cur_row) * KP
    idx_row = tl.load(Idx + row_off + offs)
    w = tl.where(rej != 0, 0.0, tl.load(Q + row_off + offs))
    u = coin_f * tl.sum(w, axis=0)
    before = idx_row[None, :] <= idx_row[:, None]  # [i, j]: token j sorts at or before token i
    cum = tl.sum(tl.where(before, w[None, :], 0.0), axis=1)
    hit = (cum > u) & (w > 0)
    sampled = tl.min(tl.where(hit, idx_row, VOCAB), axis=0)
    last_valid = tl.max(tl.where(w > 0, idx_row, -1), axis=0)
    fallback = tl.where(last_valid == -1, VOCAB - 1, last_valid)
    sampled = tl.where(sampled == VOCAB, fallback, sampled)
    tl.store(Predicts + last_accept_index, sampled.to(tl.int32))


def tree_speculative_sampling_target_only_sparse(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,  # [bs, num_draft_tokens, kp] float32
    target_index: torch.Tensor,  # [bs, num_draft_tokens, kp] int64
    vocab_size: int,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
) -> None:
    bs, num_draft_tokens = candidates.shape
    kp = target_probs.shape[-1]
    assert target_probs.shape == target_index.shape == (bs, num_draft_tokens, kp)
    _tree_sampling_target_only_sparse_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates.contiguous(),
        retrive_index.contiguous(),
        retrive_next_token.contiguous(),
        retrive_next_sibling.contiguous(),
        uniform_samples.contiguous(),
        uniform_samples_for_final_sampling.contiguous(),
        target_probs.contiguous(),
        target_index.contiguous(),
        float(threshold_single),
        float(threshold_acc),
        NUM_SPEC=accept_index.shape[1],
        NUM_DRAFT=num_draft_tokens,
        KP=kp,
        VOCAB=vocab_size,
        num_warps=2 if kp <= 64 else 4,
    )
