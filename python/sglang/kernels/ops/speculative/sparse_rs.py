# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_DRAFT_BLOCK = 2048
_DRAFT_CTX_VOCAB = int(os.environ.get("SGLANG_RND_DRAFT_CTX_VOCAB", "0")) > 0


@triton.jit
def _logit_keys(vals, indices):
    vals = tl.where(condition=vals == vals, x=vals, y=-float("inf"))
    vals = tl.where(condition=vals == 0.0, x=0.0, y=vals)
    bits = vals.to(tl.uint32, bitcast=True)
    ordered = tl.where(condition=bits & 0x80000000 != 0, x=bits ^ 0xFFFFFFFF, y=bits ^ 0x80000000)
    return ((ordered.to(tl.int64) - 0x80000000) << 32) | (0xFFFFFFFF - indices)


@triton.jit
def _draft_partial_topk_kernel(
    Logits, Keys, TopKs, stride_l, VOCAB: tl.constexpr, SPLITS: tl.constexpr,
    KB: tl.constexpr, GREEDY_FAST: tl.constexpr, BLOCK: tl.constexpr,
    AcceptLens, Hidden, HiddenOut, RowsOut, PartialMax, PartialSum, stride_h, tokens_per_req,
    HIDDEN_SIZE: tl.constexpr, HIDDEN_BLOCK: tl.constexpr, ROW_FROM_ACCEPT: tl.constexpr,
    COPY_HIDDEN: tl.constexpr, WRITE_ROWS: tl.constexpr, WRITE_PROB: tl.constexpr,
    MASK_EMPTY_SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    # Draft extend: read each request's last accepted row; split SPLITS copies its hidden row.
    src = row
    if ROW_FROM_ACCEPT:
        src = row * tokens_per_req + tl.load(pointer=AcceptLens + row).to(tl.int64) - 1
    if COPY_HIDDEN:
        if split == SPLITS:
            for h0 in tl.range(0, HIDDEN_SIZE, HIDDEN_BLOCK):
                offs = h0 + tl.arange(start=0, end=HIDDEN_BLOCK)
                h = tl.load(pointer=Hidden + src * stride_h + offs, mask=offs < HIDDEN_SIZE)
                tl.store(pointer=HiddenOut + row * HIDDEN_SIZE + offs, value=h, mask=offs < HIDDEN_SIZE)
            return
    indices = split * BLOCK + tl.arange(start=0, end=BLOCK)
    raw = tl.load(pointer=Logits + src * stride_l + indices, mask=indices < VOCAB,
                  other=-float("inf"))
    if WRITE_ROWS:
        tl.store(pointer=RowsOut + row * VOCAB + indices, value=raw, mask=indices < VOCAB)
    vals = raw.to(tl.float32)
    if WRITE_PROB:
        # Per-split max and sum(exp(x - max)) for the position-0 top-1 probability.
        cvals = tl.where(condition=vals == vals, x=vals, y=-1e30)
        cmax = tl.max(cvals, axis=0)
        tl.store(pointer=PartialMax + row * SPLITS + split, value=cmax)
        sum_max = cmax
        if MASK_EMPTY_SPLITS:
            sum_max = tl.where(condition=cmax == -float("inf"), x=0.0, y=cmax)
        tl.store(pointer=PartialSum + row * SPLITS + split, value=tl.sum(tl.exp(cvals - sum_max), axis=0))
    keys = _logit_keys(vals=vals, indices=indices.to(tl.int64))
    keys = tl.where(condition=indices < VOCAB, x=keys, y=-9223372036854775807 - 1)
    if GREEDY_FAST:
        if tl.load(pointer=TopKs + row) == 1:
            best = tl.full(shape=(KB,), value=0, dtype=tl.int64) + tl.max(input=keys, axis=0)
        else:
            best = tl.topk(x=keys, k=KB)
    else:
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
    HAS_TEMP_SCALE: tl.constexpr, HAS_ONEHOT: tl.constexpr, GREEDY_FAST: tl.constexpr,
    WRITE_POSITION: tl.constexpr, WRITE_CHAIN: tl.constexpr, BLOCK: tl.constexpr,
    Conf, PartialMax, PartialSum,
    WRITE_PROB: tl.constexpr, SPLITS: tl.constexpr, SPLIT_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if WRITE_PROB:
        # softmax max of the raw logits: 1 / sum_s(sum_s * exp(max_s - max)), fp32.
        soffs = tl.arange(start=0, end=SPLIT_BLOCK)
        pmax = tl.load(pointer=PartialMax + row * SPLITS + soffs, mask=soffs < SPLITS, other=-float("inf"))
        psum = tl.load(pointer=PartialSum + row * SPLITS + soffs, mask=soffs < SPLITS, other=0.0)
        tl.store(pointer=Conf + row, value=1.0 / tl.sum(psum * tl.exp(pmax - tl.max(pmax, axis=0)), axis=0))
    offs = tl.arange(start=0, end=BLOCK)
    keys = tl.load(pointer=Keys + row * CANDIDATES + offs, mask=offs < CANDIDATES,
                   other=-9223372036854775807 - 1)
    if GREEDY_FAST:
        if tl.load(pointer=TopKs + row) == 1:
            best = tl.full(shape=(KB,), value=0, dtype=tl.int64) + tl.max(input=keys, axis=0)
        else:
            best = tl.topk(x=keys, k=KB)
    else:
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


def _plain_row_args(*, dummy):
    # Disabled draft-extend operands get a valid dummy pointer; constexpr flags gate them off.
    return dict(
        AcceptLens=dummy, Hidden=dummy, HiddenOut=dummy, RowsOut=dummy, PartialMax=dummy,
        PartialSum=dummy, stride_h=0, tokens_per_req=0, HIDDEN_SIZE=0, HIDDEN_BLOCK=1,
        ROW_FROM_ACCEPT=False, COPY_HIDDEN=False, WRITE_ROWS=False, WRITE_PROB=False,
    )


def _proposal_split(*, vocab, k_block):
    block = max(k_block, min(_DRAFT_BLOCK, triton.next_power_of_2(vocab)))
    return block, triton.cdiv(x=vocab, y=block)


def _proposal_topk_keys(*, logits, top_ks, k_block, topk_impl, greedy_fast, n=None, row_args=None):
    vocab = logits.shape[1]
    if topk_impl == "torch":
        assert row_args is None, "draft-extend row selection needs topk_impl='triton'"
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
    if n is None:
        n = logits.shape[0]
    if row_args is None:
        row_args = _plain_row_args(dummy=logits)
    block, splits = _proposal_split(vocab=vocab, k_block=k_block)
    keys = torch.empty(size=(n, splits * k_block), dtype=torch.int64, device=logits.device)
    _draft_partial_topk_kernel[(n, splits + int(row_args["COPY_HIDDEN"]))](
        Logits=logits, Keys=keys, TopKs=top_ks, stride_l=logits.stride(0), VOCAB=vocab,
        SPLITS=splits, KB=k_block, GREEDY_FAST=greedy_fast, BLOCK=block, num_warps=4,
        MASK_EMPTY_SPLITS=_DRAFT_CTX_VOCAB, **row_args,
    )
    return keys


def _rs_proposal_sparse(
    *, next_token_logits, temperatures, top_ks, top_ps, uniforms, k, min_ps, hot_token_id,
    positions, draft_tokens, draft_token_column, draft_support_probs, draft_support_tokens,
    topk_impl, temp_scale, onehot_above, greedy_fast, accept_lens, num_tokens_per_req,
    hidden_states, write_rows, write_prob,
):
    assert temp_scale > 0 and 0 <= onehot_above <= 1
    assert next_token_logits.ndim == 2 and next_token_logits.stride(1) == 1
    vocab = next_token_logits.shape[1]
    n = next_token_logits.shape[0] if accept_lens is None else accept_lens.shape[0]
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
    row_args = None
    hidden_out = rows_out = conf = None
    partial_max = partial_sum = topk_p
    splits = 1
    if accept_lens is not None:
        assert accept_lens.ndim == 1 and accept_lens.is_contiguous()
        assert accept_lens.dtype in (torch.int32, torch.int64)
        assert num_tokens_per_req > 0 and next_token_logits.shape[0] >= n * num_tokens_per_req
        # Rows may carry trailing dims (e.g. HC streams); the kernel copies flat rows.
        hidden_rows = hidden_states.flatten(1) if hidden_states is not None else None
        if hidden_rows is not None:
            assert hidden_rows.stride(1) == 1 and hidden_rows.shape[0] == next_token_logits.shape[0]
            hidden_out = torch.empty(size=(n, *hidden_states.shape[1:]), dtype=hidden_states.dtype,
                                     device=device)
        if write_rows:
            rows_out = torch.empty(size=(n, vocab), dtype=next_token_logits.dtype, device=device)
        _, splits = _proposal_split(vocab=vocab, k_block=k_block)
        if write_prob:
            conf = torch.empty(size=(n,), dtype=torch.float32, device=device)
            partial_max = torch.empty(size=(n, splits), dtype=torch.float32, device=device)
            partial_sum = torch.empty(size=(n, splits), dtype=torch.float32, device=device)
        hidden_size = hidden_rows.shape[1] if hidden_rows is not None else 0
        row_args = dict(
            AcceptLens=accept_lens,
            Hidden=hidden_rows if hidden_rows is not None else next_token_logits,
            HiddenOut=hidden_out if hidden_out is not None else next_token_logits,
            RowsOut=rows_out if rows_out is not None else next_token_logits,
            PartialMax=partial_max, PartialSum=partial_sum,
            stride_h=hidden_rows.stride(0) if hidden_rows is not None else 0,
            tokens_per_req=num_tokens_per_req, HIDDEN_SIZE=hidden_size,
            HIDDEN_BLOCK=min(_DRAFT_BLOCK, triton.next_power_of_2(max(hidden_size, 1))),
            ROW_FROM_ACCEPT=True, COPY_HIDDEN=hidden_rows is not None,
            WRITE_ROWS=write_rows, WRITE_PROB=write_prob,
        )
    else:
        assert hidden_states is None and not write_rows and not write_prob
    if n == 0:
        return probs, tokens, topk_p, topk_index, hidden_out, rows_out, conf
    keys = _proposal_topk_keys(
        logits=next_token_logits, top_ks=top_ks, k_block=k_block,
        topk_impl=topk_impl, greedy_fast=greedy_fast, n=n, row_args=row_args,
    )
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
        HAS_TEMP_SCALE=temp_scale != 1.0, HAS_ONEHOT=onehot_above > 0, GREEDY_FAST=greedy_fast,
        WRITE_POSITION=positions is not None, WRITE_CHAIN=draft_tokens is not None,
        BLOCK=triton.next_power_of_2(keys.shape[1]),
        Conf=conf if conf is not None else topk_p, PartialMax=partial_max, PartialSum=partial_sum,
        WRITE_PROB=conf is not None, SPLITS=splits, SPLIT_BLOCK=triton.next_power_of_2(splits),
        num_warps=4,
    )
    return probs, tokens, topk_p, topk_index, hidden_out, rows_out, conf


def rs_draft_proposal_sparse(
    *, next_token_logits: torch.Tensor, temperatures: torch.Tensor,
    top_ks: torch.Tensor, top_ps: torch.Tensor, uniforms: torch.Tensor, k: int,
    min_ps: torch.Tensor | None = None, hot_token_id: torch.Tensor | None = None,
    positions: torch.Tensor | None = None, draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0, draft_support_probs: torch.Tensor | None = None,
    draft_support_tokens: torch.Tensor | None = None, topk_impl: str = "triton",
    temp_scale: float = 1.0, onehot_above: float = 0.0, greedy_fast: bool = False,
):
    probs, tokens, topk_p, topk_index, _, _, _ = _rs_proposal_sparse(
        next_token_logits=next_token_logits, temperatures=temperatures, top_ks=top_ks,
        top_ps=top_ps, uniforms=uniforms, k=k, min_ps=min_ps, hot_token_id=hot_token_id,
        positions=positions, draft_tokens=draft_tokens, draft_token_column=draft_token_column,
        draft_support_probs=draft_support_probs, draft_support_tokens=draft_support_tokens,
        topk_impl=topk_impl, temp_scale=temp_scale, onehot_above=onehot_above,
        greedy_fast=greedy_fast, accept_lens=None, num_tokens_per_req=0, hidden_states=None,
        write_rows=False, write_prob=False,
    )
    return probs, tokens, topk_p, topk_index


def rs_draft_extend_proposal_sparse(
    *, next_token_logits: torch.Tensor, accept_lens: torch.Tensor, num_tokens_per_req: int,
    temperatures: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor,
    uniforms: torch.Tensor, k: int, min_ps: torch.Tensor | None = None,
    hot_token_id: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None,
    write_rows: bool = False, write_prob: bool = False,
    temp_scale: float = 1.0, onehot_above: float = 0.0, greedy_fast: bool = False,
):
    """rs_draft_proposal_sparse on rows ``i * num_tokens_per_req + accept_lens[i] - 1``.

    Also copies those hidden rows; ``write_rows`` returns the selected logits rows
    (bit-exact input for top1_prob), ``write_prob`` returns their softmax max (fp32
    order differs from torch top1_prob: rounding-level). Returns
    (probs, tokens, topk_p, topk_index, hidden, rows, conf); unasked outputs are None.
    """
    return _rs_proposal_sparse(
        next_token_logits=next_token_logits, temperatures=temperatures, top_ks=top_ks,
        top_ps=top_ps, uniforms=uniforms, k=k, min_ps=min_ps, hot_token_id=hot_token_id,
        positions=None, draft_tokens=None, draft_token_column=0,
        draft_support_probs=None, draft_support_tokens=None, topk_impl="triton",
        temp_scale=temp_scale, onehot_above=onehot_above, greedy_fast=greedy_fast,
        accept_lens=accept_lens, num_tokens_per_req=num_tokens_per_req,
        hidden_states=hidden_states, write_rows=write_rows, write_prob=write_prob,
    )


@triton.jit
def _bv_combine(a1, c1, b1, a2, c2, b2):
    return a2 * a1, a2 * c1 + c2, tl.minimum(x=a2 * b1 + c2, y=b2)


@triton.jit
def _chain_sampling_sparse_kernel(
    Predicts, AcceptIndex, AcceptTokenNum, Candidates, RetriveIndex,
    Coins, CoinsFinal, P, PI, Q, QI, VTLens,
    NUM_SLOTS: tl.constexpr, K: tl.constexpr, KQ: tl.constexpr,
    KP: tl.constexpr, VOCAB: tl.constexpr, BLOCK_VERIFY: tl.constexpr, SP: tl.constexpr,
    HAS_VT: tl.constexpr,
):
    bx = tl.program_id(0).to(tl.int64)
    num_drafts = NUM_SLOTS - 1
    if HAS_VT:
        num_drafts = tl.minimum(tl.maximum(tl.load(VTLens + bx), 0), NUM_SLOTS - 1)
    poffs = tl.arange(start=0, end=KP)
    qoffs = tl.arange(start=0, end=KQ)
    base = bx * NUM_SLOTS
    last = tl.load(RetriveIndex + base)
    tl.store(pointer=AcceptIndex + base, value=last)
    num_correct = 0
    if BLOCK_VERIFY:
        u = tl.arange(start=0, end=SP)
        valid = (u >= 1) & (u < NUM_SLOTS)
        if HAS_VT:
            valid = valid & (u <= num_drafts)
        token = tl.load(pointer=Candidates + base + u, mask=valid, other=-1)
        coin = tl.load(pointer=Coins + base + u - 1, mask=valid, other=1.0)
        position_p = tl.load(pointer=P + (base + u[:, None] - 1) * KP + poffs[None, :],
                             mask=valid[:, None], other=0.0).to(dtype=tl.float32)
        position_ids = tl.load(pointer=PI + (base + u[:, None] - 1) * KP + poffs[None, :],
                               mask=valid[:, None], other=-1)
        position_qbase = (bx * (NUM_SLOTS - 1) + u - 1) * K
        position_q = tl.load(pointer=Q + position_qbase[:, None] + qoffs[None, :],
                             mask=valid[:, None] & (qoffs[None, :] < K), other=0.0).to(dtype=tl.float32)
        position_qids = tl.load(pointer=QI + position_qbase[:, None] + qoffs[None, :],
                                mask=valid[:, None] & (qoffs[None, :] < K), other=-1)
        position_q = tl.where(condition=position_q == position_q, x=position_q, y=0.0)
        pt = tl.sum(input=tl.where(condition=position_ids == token[:, None], x=position_p, y=0.0), axis=1)
        qt = tl.sum(input=tl.where(condition=position_qids == token[:, None], x=position_q, y=0.0), axis=1)
        a = tl.where(condition=valid & (pt > 0) & (qt != 0),
                     x=pt / tl.where(condition=qt != 0, x=qt, y=1.0), y=0.0)
        c = tl.where(condition=valid & (pt > 0) & (qt == 0), x=1.0, y=0.0)
        b = tl.where(condition=valid & (pt > 0), x=1.0, y=0.0)
        a = tl.where(condition=u == 0, x=1.0, y=a)
        b = tl.where(condition=u == 0, x=1.0, y=b)
        a, c, b = tl.associative_scan(input=(a, c, b), axis=0, combine_fn=_bv_combine)
        path_pi = tl.minimum(x=a + c, y=b)
        first_zero = tl.min(input=tl.where(condition=valid & (path_pi == 0), x=u, y=SP), axis=0)
        path_pi = tl.where(condition=u >= first_zero, x=0.0, y=path_pi)
        # Row L is a bonus row: neither its acceptance threshold nor final draw uses q_L.
        next_valid = valid & (u < num_drafts)
        position_p = tl.load(pointer=P + (base + u[:, None]) * KP + poffs[None, :],
                             mask=next_valid[:, None], other=0.0).to(dtype=tl.float32)
        position_ids = tl.load(pointer=PI + (base + u[:, None]) * KP + poffs[None, :],
                               mask=next_valid[:, None], other=-1)
        position_matched = tl.full(shape=(SP, KP), value=0.0, dtype=tl.float32)
        # Bound the match tile to [SP, KP, 8] instead of [SP, KP, KQ].
        chunks = tl.arange(start=0, end=8)
        position_qbase = (bx * (NUM_SLOTS - 1) + u) * K
        for chunk in range(triton.cdiv(x=KQ, y=8)):
            koffs = chunk * 8 + chunks
            chunk_q = tl.load(pointer=Q + position_qbase[:, None] + koffs[None, :],
                              mask=next_valid[:, None] & (koffs[None, :] < K), other=0.0).to(dtype=tl.float32)
            chunk_qi = tl.load(pointer=QI + position_qbase[:, None] + koffs[None, :],
                               mask=next_valid[:, None] & (koffs[None, :] < K), other=-1)
            chunk_q = tl.where(condition=chunk_q == chunk_q, x=chunk_q, y=0.0)
            position_matched += tl.sum(input=tl.where(condition=position_ids[:, :, None] == chunk_qi[:, None, :],
                                                      x=chunk_q[:, None, :], y=0.0), axis=2)
        n = tl.sum(input=tl.maximum(x=path_pi[:, None] * position_p - position_matched, y=0.0), axis=1)
        denominator = n + 1.0 - path_pi
        h = tl.where(condition=denominator > 0, x=n / denominator, y=1.0)
        h = tl.where(condition=u == num_drafts, x=path_pi, y=h)
        h = tl.where(condition=valid & (u < first_zero), x=h, y=0.0)
        num_correct = tl.max(input=tl.where(condition=coin < h, x=u, y=0), axis=0)
        final_pi = tl.sum(input=tl.where(condition=u == num_correct, x=path_pi, y=0.0), axis=0)
        accepted = valid & (u <= num_correct)
        previous = tl.load(pointer=RetriveIndex + base + u - 1, mask=accepted, other=0)
        current = tl.load(pointer=RetriveIndex + base + u, mask=accepted, other=0)
        tl.store(pointer=Predicts + previous, value=token, mask=accepted)
        tl.store(pointer=AcceptIndex + base + u, value=current, mask=accepted)
        last = tl.load(pointer=RetriveIndex + base + num_correct)
    else:
        walking = 1
        step = 1
        while (step < num_drafts + 1) & (walking == 1):
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
    if BLOCK_VERIFY:
        p = p.to(tl.float32)
    weights = p
    if num_correct < num_drafts:
        qbase = (bx * (NUM_SLOTS - 1) + num_correct) * K
        q = tl.load(pointer=Q + qbase + qoffs, mask=qoffs < K, other=0.0)
        qi = tl.load(pointer=QI + qbase + qoffs, mask=qoffs < K, other=-1)
        if BLOCK_VERIFY:
            q = q.to(tl.float32)
        q = tl.where(condition=q == q, x=q, y=0.0)
        matched = tl.sum(tl.where(condition=pi[:, None] == qi[None, :], x=q[None, :], y=0.0), axis=1)
        if BLOCK_VERIFY:
            weights = tl.maximum(x=final_pi * p - matched, y=0.0)
        else:
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
    draft_support_tokens: torch.Tensor, vocab_size: int, block_verify: bool = False,
    vt_lens: torch.Tensor | None = None,
):
    bs, num_slots = candidates.shape
    if vt_lens is not None:
        if (
            vt_lens.shape != (bs,)
            or vt_lens.dtype != torch.int32
            or not vt_lens.is_contiguous()
            or not vt_lens.is_cuda
            or vt_lens.device != candidates.device
        ):
            raise ValueError(
                "vt_lens must be contiguous CUDA int32 [batch_size] on the candidates device"
            )
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
        VTLens=vt_lens if vt_lens is not None else candidates,
        HAS_VT=vt_lens is not None,
        NUM_SLOTS=num_slots, K=k, KQ=triton.next_power_of_2(k), KP=kp, VOCAB=vocab_size, BLOCK_VERIFY=block_verify,
        SP=triton.next_power_of_2(n=num_slots),
        num_warps=8 if block_verify and num_slots > 8 else 4,
    )
