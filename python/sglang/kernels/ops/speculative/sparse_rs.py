from __future__ import annotations

import torch
import triton
import triton.language as tl

_DRAFT_BLOCK = 2048


@triton.jit
def _logit_keys(vals, indices):
    vals = tl.where(condition=vals == vals, x=vals, y=-float("inf"))
    vals = tl.where(condition=vals == 0.0, x=0.0, y=vals)
    bits = vals.to(tl.uint32, bitcast=True)
    ordered = tl.where(condition=bits & 0x80000000 != 0, x=bits ^ 0xFFFFFFFF, y=bits ^ 0x80000000)
    return ((ordered.to(tl.int64) - 0x80000000) << 32) | (0xFFFFFFFF - indices)


@triton.jit
def _draft_partial_topk_kernel(
    Logits, Keys, stride_l, VOCAB: tl.constexpr, SPLITS: tl.constexpr,
    KB: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    indices = split * BLOCK + tl.arange(start=0, end=BLOCK)
    vals = tl.load(pointer=Logits + row * stride_l + indices, mask=indices < VOCAB,
                   other=-float("inf")).to(tl.float32)
    keys = _logit_keys(vals=vals, indices=indices.to(tl.int64))
    keys = tl.where(condition=indices < VOCAB, x=keys, y=-9223372036854775807 - 1)
    best = tl.topk(x=keys, k=KB)
    tl.store(pointer=Keys + (row * SPLITS + split) * KB + tl.arange(start=0, end=KB), value=best)


@triton.jit
def _sum_compensated(left_hi, left_lo, right_hi, right_lo):
    total = left_hi + right_hi
    right = total - left_hi
    error = (left_hi - (total - right)) + (right_hi - right)
    low = left_lo + right_lo + error
    high = total + low
    return high, low - (high - total)


@triton.jit
def _draft_finalize_kernel(
    Keys, Temps, TopKs, TopPs, MinPs, Uniforms, HotTokens,
    Probs, Tokens, TopkP, TopkIndex, Positions, DraftTokens,
    stride_q, stride_t, stride_u, stride_chain, column, temp_scale, onehot_above,
    CANDIDATES: tl.constexpr, K: tl.constexpr, KB: tl.constexpr,
    HAS_MIN_P: tl.constexpr, HAS_MAP: tl.constexpr,
    HAS_TEMP_SCALE: tl.constexpr, HAS_ONEHOT: tl.constexpr,
    WRITE_POSITION: tl.constexpr, WRITE_CHAIN: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(start=0, end=BLOCK)
    keys = tl.load(pointer=Keys + row * CANDIDATES + offs, mask=offs < CANDIDATES,
                   other=-9223372036854775807 - 1)
    best = tl.topk(x=keys, k=KB)
    indices = 0xFFFFFFFF - (best & 0xFFFFFFFF)
    ordered = ((best >> 32) + 0x80000000).to(tl.uint32)
    bits = tl.where(condition=ordered & 0x80000000 != 0, x=ordered ^ 0x80000000, y=ordered ^ 0xFFFFFFFF)
    ranks = tl.arange(start=0, end=KB)
    vals = tl.where(condition=ranks < K, x=bits.to(tl.float32, bitcast=True), y=-float("inf"))
    maximum = tl.max(vals, axis=0)
    temperature = tl.load(Temps + row).to(tl.float32)
    if HAS_TEMP_SCALE:
        temperature = tl.where(condition=temperature > 0, x=temperature * temp_scale, y=1.0)
    else:
        temperature = tl.where(condition=temperature > 0, x=temperature, y=1.0)
    weights = tl.where(condition=vals == maximum, x=1.0, y=tl.exp((vals - maximum) / temperature))
    top_k = tl.load(TopKs + row)
    weights = tl.where(condition=(ranks < K) & ((ranks < top_k) | (top_k <= 0)), x=weights, y=0.0)
    probs = weights / tl.sum(weights, axis=0)
    # Compensated prefixes avoid moving a top-p boundary on equal bf16 logits.
    prefix_hi, prefix_lo = tl.associative_scan(
        input=(probs, tl.full(shape=(KB,), value=0.0, dtype=tl.float32)), axis=0, combine_fn=_sum_compensated,
    )
    exclusive = (prefix_hi + prefix_lo) - probs
    probs = tl.where(condition=(exclusive < tl.load(TopPs + row)) | (ranks == 0), x=probs, y=0.0)
    if HAS_MIN_P:
        p0 = tl.sum(tl.where(condition=ranks == 0, x=probs, y=0.0), axis=0)
        probs = tl.where(condition=probs >= p0 * tl.load(MinPs + row), x=probs, y=0.0)
    probs = probs / tl.sum(probs, axis=0)
    if HAS_ONEHOT:
        p0 = tl.sum(tl.where(condition=ranks == 0, x=probs, y=0.0), axis=0)
        probs = tl.where(condition=p0 >= onehot_above, x=(ranks == 0).to(tl.float32), y=probs)
    cdf = tl.cumsum(probs, axis=0)
    u = tl.load(Uniforms + row * stride_u) * tl.sum(probs, axis=0)
    pick = tl.min(tl.where(condition=(cdf > u) & (probs > 0), x=ranks, y=KB), axis=0)
    last = tl.max(tl.where(condition=probs > 0, x=ranks, y=0), axis=0)
    pick = tl.where(condition=pick == KB, x=last, y=pick)
    tokens = indices
    if HAS_MAP:
        tokens = tl.load(pointer=HotTokens + indices, mask=ranks < K, other=0)
    draft_token = tl.sum(tl.where(condition=ranks == pick, x=indices, y=0), axis=0)
    target_token = tl.sum(tl.where(condition=ranks == pick, x=tokens, y=0), axis=0)
    tl.store(pointer=Probs + row * stride_q + ranks, value=probs, mask=ranks < K)
    tl.store(pointer=Tokens + row * stride_t + ranks, value=tokens, mask=ranks < K)
    tl.store(pointer=TopkP + row, value=tl.sum(tl.where(condition=ranks == pick, x=probs, y=0.0), axis=0))
    tl.store(pointer=TopkIndex + row, value=target_token if WRITE_CHAIN else draft_token)
    if WRITE_CHAIN:
        tl.store(pointer=DraftTokens + row * stride_chain + column, value=target_token)
    if WRITE_POSITION:
        tl.store(pointer=Positions + row, value=tl.load(Positions + row) + 1)


def _proposal_topk_keys(*, logits, k_block, topk_impl):
    n, vocab = logits.shape
    if topk_impl == "torch":
        vals = torch.nan_to_num(input=logits.float(), nan=-float("inf"),
                                posinf=float("inf"), neginf=-float("inf"))
        vals = torch.where(condition=vals == 0, input=torch.zeros_like(vals), other=vals)
        bits = vals.view(torch.int32).to(torch.int64)
        ordered = torch.where(condition=bits < 0, input=(~bits) & 0xFFFFFFFF,
                              other=bits ^ 0x80000000)
        indices = torch.arange(vocab, device=logits.device, dtype=torch.int64)
        keys = ((ordered - 0x80000000) << 32) | (0xFFFFFFFF - indices)
        return torch.topk(input=keys, k=min(k_block, vocab), dim=-1).values
    if topk_impl != "triton":
        raise ValueError(f"Unknown topk_impl: {topk_impl}")
    block = max(k_block, min(_DRAFT_BLOCK, triton.next_power_of_2(vocab)))
    splits = triton.cdiv(x=vocab, y=block)
    keys = torch.empty(size=(n, splits * k_block), dtype=torch.int64, device=logits.device)
    _draft_partial_topk_kernel[(n, splits)](
        Logits=logits, Keys=keys, stride_l=logits.stride(0), VOCAB=vocab,
        SPLITS=splits, KB=k_block, BLOCK=block, num_warps=4,
    )
    return keys


def rs_draft_proposal_sparse(
    *, next_token_logits: torch.Tensor, temperatures: torch.Tensor,
    top_ks: torch.Tensor, top_ps: torch.Tensor, uniforms: torch.Tensor, k: int,
    min_ps: torch.Tensor | None = None, hot_token_id: torch.Tensor | None = None,
    positions: torch.Tensor | None = None, draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0, draft_support_probs: torch.Tensor | None = None,
    draft_support_tokens: torch.Tensor | None = None, topk_impl: str = "triton",
    temp_scale: float = 1.0, onehot_above: float = 0.0,
):
    assert temp_scale > 0 and 0 <= onehot_above <= 1
    assert next_token_logits.ndim == 2 and next_token_logits.stride(1) == 1
    n, vocab = next_token_logits.shape
    assert k > 0 and vocab > 0
    k = min(k, vocab)
    k_block = triton.next_power_of_2(k)
    device = next_token_logits.device
    probs = draft_support_probs
    tokens = draft_support_tokens
    if probs is None:
        probs = torch.empty(size=(n, k), dtype=torch.float32, device=device)
    if tokens is None:
        tokens = torch.empty(size=(n, k), dtype=torch.int64, device=device)
    assert probs.shape == tokens.shape == (n, k)
    assert probs.dtype == torch.float32 and tokens.dtype == torch.int64
    assert probs.stride(1) == tokens.stride(1) == 1
    assert temperatures.numel() == top_ks.numel() == top_ps.numel() == n
    assert temperatures.is_contiguous() and top_ks.is_contiguous() and top_ps.is_contiguous()
    assert uniforms.ndim == 1 and uniforms.shape[0] == n
    if min_ps is not None:
        assert min_ps.numel() == n and min_ps.is_contiguous()
    if hot_token_id is not None:
        assert hot_token_id.shape == (vocab,) and hot_token_id.dtype == torch.int64
        assert hot_token_id.is_contiguous()
    if positions is not None:
        assert positions.shape == (n,) and positions.is_contiguous()
    if draft_tokens is not None:
        assert draft_tokens.shape[0] == n and draft_tokens.stride(1) == 1
        assert draft_tokens.dtype == torch.int64
        assert 0 <= draft_token_column < draft_tokens.shape[1]
    topk_p = torch.empty(size=(n, 1), dtype=torch.float32, device=device)
    topk_index = torch.empty(size=(n, 1), dtype=torch.int64, device=device)
    if n == 0:
        return probs, tokens, topk_p, topk_index
    keys = _proposal_topk_keys(logits=next_token_logits, k_block=k_block, topk_impl=topk_impl)
    _draft_finalize_kernel[(n,)](
        Keys=keys, Temps=temperatures, TopKs=top_ks, TopPs=top_ps,
        MinPs=min_ps if min_ps is not None else top_ps, Uniforms=uniforms,
        HotTokens=hot_token_id if hot_token_id is not None else tokens,
        Probs=probs, Tokens=tokens, TopkP=topk_p, TopkIndex=topk_index,
        Positions=positions if positions is not None else topk_index,
        DraftTokens=draft_tokens if draft_tokens is not None else topk_index,
        stride_q=probs.stride(0), stride_t=tokens.stride(0), stride_u=uniforms.stride(0),
        stride_chain=draft_tokens.stride(0) if draft_tokens is not None else 0,
        column=draft_token_column, temp_scale=temp_scale, onehot_above=onehot_above,
        CANDIDATES=keys.shape[1], K=k, KB=k_block,
        HAS_MIN_P=min_ps is not None, HAS_MAP=hot_token_id is not None,
        HAS_TEMP_SCALE=temp_scale != 1.0, HAS_ONEHOT=onehot_above > 0,
        WRITE_POSITION=positions is not None, WRITE_CHAIN=draft_tokens is not None,
        BLOCK=triton.next_power_of_2(keys.shape[1]), num_warps=4,
    )
    return probs, tokens, topk_p, topk_index


@triton.jit
def _chain_sampling_sparse_kernel(
    Predicts, AcceptIndex, AcceptTokenNum, Candidates, RetriveIndex,
    Coins, CoinsFinal, P, PI, Q, QI,
    NUM_SLOTS: tl.constexpr, K: tl.constexpr, KQ: tl.constexpr,
    KP: tl.constexpr, VOCAB: tl.constexpr,
):
    bx = tl.program_id(0).to(tl.int64)
    poffs = tl.arange(start=0, end=KP)
    qoffs = tl.arange(start=0, end=KQ)
    base = bx * NUM_SLOTS
    last = tl.load(RetriveIndex + base)
    tl.store(pointer=AcceptIndex + base, value=last)
    num_correct = 0
    walking = 1
    step = 1
    while (step < NUM_SLOTS) & (walking == 1):
        token = tl.load(Candidates + base + step)
        p = tl.load(P + (base + step - 1) * KP + poffs)
        pi = tl.load(PI + (base + step - 1) * KP + poffs)
        qbase = (bx * (NUM_SLOTS - 1) + step - 1) * K
        q = tl.load(pointer=Q + qbase + qoffs, mask=qoffs < K, other=0.0)
        qi = tl.load(pointer=QI + qbase + qoffs, mask=qoffs < K, other=-1)
        q = tl.where(condition=q == q, x=q, y=0.0)
        pt = tl.sum(tl.where(condition=pi == token, x=p, y=0.0), axis=0)
        qt = tl.sum(tl.where(condition=qi == token, x=q, y=0.0), axis=0)
        coin = tl.load(Coins + base + step - 1)
        if coin * qt < pt:
            tl.store(pointer=Predicts + last, value=token)
            last = tl.load(RetriveIndex + base + step)
            num_correct += 1
            tl.store(pointer=AcceptIndex + base + num_correct, value=last)
            step += 1
        else:
            walking = 0
    tl.store(pointer=AcceptTokenNum + bx, value=num_correct)
    p = tl.load(P + (base + num_correct) * KP + poffs)
    pi = tl.load(PI + (base + num_correct) * KP + poffs)
    weights = p
    if num_correct < NUM_SLOTS - 1:
        qbase = (bx * (NUM_SLOTS - 1) + num_correct) * K
        q = tl.load(pointer=Q + qbase + qoffs, mask=qoffs < K, other=0.0)
        qi = tl.load(pointer=QI + qbase + qoffs, mask=qoffs < K, other=-1)
        q = tl.where(condition=q == q, x=q, y=0.0)
        matched = tl.sum(tl.where(condition=pi[:, None] == qi[None, :], x=q[None, :], y=0.0), axis=1)
        weights = tl.maximum(x=p - matched, y=0.0)
        # Deliberately draw P on a zero residual; dense RS returns VOCAB - 1.
        weights = tl.where(condition=tl.sum(weights, axis=0) > 0, x=weights, y=p)
    order_keys = (pi << 32) | poffs.to(tl.int64)
    sorted_keys = tl.sort(x=order_keys, descending=False)
    order = (sorted_keys & 0xFFFFFFFF).to(tl.int32)
    tokens = sorted_keys >> 32
    weights = tl.gather(src=weights, index=order, axis=0)
    u = tl.load(CoinsFinal + bx) * tl.sum(weights, axis=0)
    hit = (tl.cumsum(weights, axis=0) > u) & (weights > 0)
    token = tl.min(tl.where(condition=hit, x=tokens, y=VOCAB), axis=0)
    last_valid = tl.max(tl.where(condition=weights > 0, x=tokens, y=-1), axis=0)
    token = tl.where(condition=token < VOCAB, x=token, y=tl.where(condition=last_valid >= 0, x=last_valid, y=VOCAB - 1))
    tl.store(pointer=Predicts + last, value=token)


def chain_speculative_sampling_sparse(
    *, predicts: torch.Tensor, accept_index: torch.Tensor, accept_token_num: torch.Tensor,
    candidates: torch.Tensor, retrive_index: torch.Tensor, uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor, target_probs: torch.Tensor,
    target_index: torch.Tensor, draft_support_probs: torch.Tensor,
    draft_support_tokens: torch.Tensor, vocab_size: int,
):
    bs, num_slots = candidates.shape
    kp = target_probs.shape[-1]
    k = draft_support_probs.shape[-1]
    assert target_probs.shape == target_index.shape == (bs, num_slots, kp)
    assert draft_support_probs.shape == draft_support_tokens.shape == (bs, num_slots - 1, k)
    assert accept_index.shape == retrive_index.shape == candidates.shape
    assert triton.next_power_of_2(kp) == kp
    if bs == 0:
        return
    _chain_sampling_sparse_kernel[(bs,)](
        Predicts=predicts, AcceptIndex=accept_index, AcceptTokenNum=accept_token_num,
        Candidates=candidates.contiguous(), RetriveIndex=retrive_index.contiguous(),
        Coins=uniform_samples.contiguous(), CoinsFinal=uniform_samples_for_final_sampling.contiguous(),
        P=target_probs.contiguous(), PI=target_index.contiguous(),
        Q=draft_support_probs.contiguous(), QI=draft_support_tokens.contiguous(),
        NUM_SLOTS=num_slots, K=k, KQ=triton.next_power_of_2(k), KP=kp, VOCAB=vocab_size,
        num_warps=4,
    )
