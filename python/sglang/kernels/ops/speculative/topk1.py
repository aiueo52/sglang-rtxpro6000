# Modified by aiueo52 for flash-next-fast (2026); see MODIFICATIONS.md.
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_DRAFT_TOPK1_BLOCK = 8192


@triton.jit
def _draft_topk1_partial_argmax_kernel(
    logits,
    partial_vals,
    partial_indices,
    partial_sums,
    logits_row_stride,
    vocab_size: tl.constexpr,
    num_splits: tl.constexpr,
    WRITE_PROB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 row base: row * stride overflows int32 once bs * vocab reaches 2^31.
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(
        logits + row * logits_row_stride + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    # Keep NaNs on valid lanes from selecting the masked tail.
    vals = tl.where(vals == vals, vals, -1e30)

    max_val = tl.max(vals, axis=0)
    local_index = tl.argmax(vals, axis=0)
    out_offset = row * num_splits + split
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + local_index)
    if WRITE_PROB:
        # Online-softmax partial: sum over this split of exp(v - m_split).
        # Masked lanes hold -inf (-> exp 0) so they contribute nothing.
        tl.store(partial_sums + out_offset, tl.sum(tl.exp(vals - max_val), axis=0))


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_vals,
    partial_indices,
    partial_sums,
    topk_p,
    topk_index,
    positions,
    hot_token_id,
    draft_tokens,
    draft_tokens_stride,
    draft_token_column,
    chain_probs,
    chain_probs_stride,
    num_splits: tl.constexpr,
    HAS_TOKEN_MAP: tl.constexpr,
    WRITE_DRAFT_TOKEN: tl.constexpr,
    WRITE_PROB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets,
        mask=mask,
        other=-float("inf"),
    )

    split = tl.argmax(vals, axis=0)
    if WRITE_PROB:
        # p_top1 = exp(M - logsumexp) = 1 / sum_j s_j * exp(m_j - M), with
        # M = max_j m_j the global max (== the top-1 logit).
        m = tl.max(vals, axis=0)
        sums = tl.load(
            partial_sums + row * num_splits + offsets, mask=mask, other=0.0
        )
        denom = tl.sum(sums * tl.exp(vals - m), axis=0)
        tl.store(
            chain_probs + row * chain_probs_stride + draft_token_column, 1.0 / denom
        )
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    token = tl.load(hot_token_id + index) if HAS_TOKEN_MAP else index
    tl.store(topk_index + row, token)
    tl.store(topk_p + row, 1.0)
    if WRITE_DRAFT_TOKEN:
        tl.store(draft_tokens + row * draft_tokens_stride + draft_token_column, token)

    position = tl.load(positions + row)
    tl.store(positions + row, position + 1)


def draft_topk1_postprocess(
    next_token_logits: torch.Tensor,
    positions: torch.Tensor,
    draft_tokens: torch.Tensor | None = None,
    draft_token_column: int = 0,
    hot_token_id: Optional[torch.Tensor] = None,
    chain_probs: torch.Tensor | None = None,
):
    """Argmax draft logits for topk=1 and advance positions.

    PyTorch eager argmax reduces each row with too little parallelism for the
    GLM/DSV4 vocab widths in CUDA graph replay. This split reduction exposes
    the vocab dimension across CTAs, then finalizes one token per row.

    If ``hot_token_id`` is given, the argmax is mapped to a full-vocabulary
    token ID in the finalize kernel. If ``draft_tokens`` is given, that token
    is also stored into ``draft_tokens[:, draft_token_column]``, mutating the
    caller-owned buffer in place. ``topk_p`` is returned as constant 1.0:
    topk=1 drafting is greedy and the chain probabilities are unused downstream.

    If ``chain_probs`` is given, the softmax probability of the selected token
    is *additionally* written to ``chain_probs[:, draft_token_column]``.  It is
    obtained from the same single pass over the vocabulary that the argmax
    already makes (online softmax: each split stores ``sum(exp(v - m_split))``
    alongside its max, and the finalize kernel combines them), so it costs one
    extra fp32 store per split and no extra global traffic.  ``topk_p`` is left
    at 1.0 so that every existing consumer is bit-identical; the confidence is
    a side channel for the adaptive speculative controller.  When
    ``chain_probs`` is ``None`` the kernels specialize back to the original
    code (``WRITE_PROB`` is a ``tl.constexpr``).
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert positions.ndim == 1
    assert positions.is_contiguous()
    assert positions.shape[0] == next_token_logits.shape[0]
    assert positions.device == next_token_logits.device
    has_token_map = hot_token_id is not None
    if has_token_map:
        assert hot_token_id.ndim == 1
        assert hot_token_id.dtype == torch.int64
        assert hot_token_id.is_contiguous()
        assert hot_token_id.device == next_token_logits.device
        assert hot_token_id.shape[0] == next_token_logits.shape[1]
    write_prob = chain_probs is not None
    if write_prob:
        assert chain_probs.ndim == 2
        assert chain_probs.dtype == torch.float32
        assert chain_probs.device == next_token_logits.device
        assert chain_probs.shape[0] >= next_token_logits.shape[0]
        assert chain_probs.stride(1) == 1
        assert 0 <= draft_token_column < chain_probs.shape[1]
    write_draft_token = draft_tokens is not None
    if write_draft_token:
        assert draft_tokens.ndim == 2
        assert draft_tokens.dtype == torch.long
        assert draft_tokens.device == next_token_logits.device
        assert draft_tokens.shape[0] == next_token_logits.shape[0]
        assert draft_tokens.stride(1) == 1
        assert 0 <= draft_token_column < draft_tokens.shape[1]

    bs, vocab_size = next_token_logits.shape
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=next_token_logits.device)
    topk_index = torch.empty(
        (bs, 1), dtype=torch.int64, device=next_token_logits.device
    )
    if bs == 0:
        return topk_p, topk_index

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty(
        (bs, num_splits), dtype=torch.float32, device=next_token_logits.device
    )
    partial_indices = torch.empty(
        (bs, num_splits), dtype=torch.int32, device=next_token_logits.device
    )
    partial_sums = (
        torch.empty((bs, num_splits), dtype=torch.float32, device=next_token_logits.device)
        if write_prob
        else partial_vals
    )

    _draft_topk1_partial_argmax_kernel[(bs, num_splits)](
        next_token_logits,
        partial_vals,
        partial_indices,
        partial_sums,
        next_token_logits.stride(0),
        vocab_size,
        num_splits,
        WRITE_PROB=write_prob,
        BLOCK=block,
        num_warps=8,
    )
    # Dummy operand for the disabled draft-token slot: the pointer must be
    # valid even though the kernel never dereferences it (gated off by
    # WRITE_DRAFT_TOKEN).
    _draft_topk1_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        partial_sums,
        topk_p,
        topk_index,
        positions,
        hot_token_id if has_token_map else topk_index,
        draft_tokens if write_draft_token else topk_index,
        draft_tokens.stride(0) if write_draft_token else 0,
        draft_token_column,
        chain_probs if write_prob else topk_p,
        chain_probs.stride(0) if write_prob else 0,
        num_splits,
        HAS_TOKEN_MAP=has_token_map,
        WRITE_DRAFT_TOKEN=write_draft_token,
        WRITE_PROB=write_prob,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index


@triton.jit
def _select_split_argmax(
    logits_row,
    rows_out_row,
    partial_vals,
    partial_indices,
    partial_sums,
    out_offset,
    split,
    vocab_size: tl.constexpr,
    WRITE_ROWS: tl.constexpr,
    WRITE_PROB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = split * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    vals = tl.load(logits_row + offsets, mask=mask, other=-float("inf"))
    if WRITE_ROWS:
        tl.store(rows_out_row + offsets, vals, mask=mask)
    vals = vals.to(tl.float32)
    vals = tl.where(vals == vals, vals, -1e30)
    max_val = tl.max(vals, axis=0)
    tl.store(partial_vals + out_offset, max_val)
    tl.store(partial_indices + out_offset, split * BLOCK + tl.argmax(vals, axis=0))
    if WRITE_PROB:
        tl.store(partial_sums + out_offset, tl.sum(tl.exp(vals - max_val), axis=0))


@triton.jit
def _draft_extend_select_partial_kernel(
    logits,
    row_src,
    partial_vals,
    partial_indices,
    partial_sums,
    rows_out,
    hidden,
    hidden_out,
    logits_row_stride,
    hidden_row_stride,
    num_rows,
    hidden_num_rows,
    num_tokens_per_req,
    vocab_size: tl.constexpr,
    hidden_size: tl.constexpr,
    num_splits: tl.constexpr,
    ROW_FROM_ACCEPT: tl.constexpr,
    WRITE_ROWS: tl.constexpr,
    WRITE_PROB: tl.constexpr,
    COPY_HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
):
    # One program per (output row, vocab split); with COPY_HIDDEN the extra
    # split index num_splits copies the selected hidden-state row instead.
    out_row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    row = tl.load(row_src + out_row).to(tl.int64)
    if ROW_FROM_ACCEPT:
        row = out_row * num_tokens_per_req + row - 1
    # Negative rows wrap like torch advanced indexing.
    lrow = tl.where(row < 0, row + num_rows, row)
    if COPY_HIDDEN:
        if split == num_splits:
            hrow = tl.where(row < 0, row + hidden_num_rows, row)
            offsets = tl.arange(0, HIDDEN_BLOCK)
            mask = offsets < hidden_size
            h = tl.load(hidden + hrow * hidden_row_stride + offsets, mask=mask)
            tl.store(hidden_out + out_row * hidden_size + offsets, h, mask=mask)
        else:
            _select_split_argmax(
                logits + lrow * logits_row_stride,
                rows_out + out_row * vocab_size,
                partial_vals,
                partial_indices,
                partial_sums,
                out_row * num_splits + split,
                split,
                vocab_size,
                WRITE_ROWS,
                WRITE_PROB,
                BLOCK,
            )
    else:
        _select_split_argmax(
            logits + lrow * logits_row_stride,
            rows_out + out_row * vocab_size,
            partial_vals,
            partial_indices,
            partial_sums,
            out_row * num_splits + split,
            split,
            vocab_size,
            WRITE_ROWS,
            WRITE_PROB,
            BLOCK,
        )


@triton.jit
def _draft_extend_select_finalize_kernel(
    partial_vals,
    partial_indices,
    partial_sums,
    topk_p,
    topk_index,
    conf_out,
    num_splits: tl.constexpr,
    WRITE_PROB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_splits
    vals = tl.load(
        partial_vals + row * num_splits + offsets, mask=mask, other=-float("inf")
    )
    split = tl.argmax(vals, axis=0)
    if WRITE_PROB:
        m = tl.max(vals, axis=0)
        sums = tl.load(
            partial_sums + row * num_splits + offsets, mask=mask, other=0.0
        )
        tl.store(conf_out + row, 1.0 / tl.sum(sums * tl.exp(vals - m), axis=0))
    index = tl.load(partial_indices + row * num_splits + split).to(tl.int64)
    tl.store(topk_index + row, index)
    tl.store(topk_p + row, 1.0)


def draft_extend_select_topk1(
    next_token_logits: torch.Tensor,
    select_index: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    write_rows: bool = False,
    write_prob: bool = False,
    accept_lens: Optional[torch.Tensor] = None,
    num_tokens_per_req: int = 0,
):
    """topk=1 draft-extend tail: argmax and hidden row of each selected row.

    Rows are ``select_index``, or ``i * num_tokens_per_req + accept_lens[i] - 1``.
    Returns (topk_p, topk_index, hidden, rows, conf); rows/conf are None unless asked.
    """
    assert next_token_logits.ndim == 2
    assert next_token_logits.stride(1) == 1
    assert (select_index is None) != (accept_lens is None)
    row_from_accept = accept_lens is not None
    row_src = accept_lens if row_from_accept else select_index
    assert row_src.ndim == 1
    assert row_src.is_contiguous()
    assert row_src.device == next_token_logits.device
    if row_from_accept:
        assert accept_lens.dtype in (torch.int32, torch.int64)
        assert num_tokens_per_req > 0
    else:
        assert select_index.dtype == torch.int64
    copy_hidden = hidden_states is not None
    # Rows may carry trailing dims (e.g. HC streams); the kernel copies flat rows.
    hidden_rows = hidden_states.flatten(1) if copy_hidden else None
    if copy_hidden:
        assert hidden_rows.stride(1) == 1
        assert hidden_states.device == next_token_logits.device

    device = next_token_logits.device
    bs = row_src.shape[0]
    vocab_size = next_token_logits.shape[1]
    topk_p = torch.empty((bs, 1), dtype=torch.float32, device=device)
    topk_index = torch.empty((bs, 1), dtype=torch.int64, device=device)
    hidden_out = (
        torch.empty(
            (bs, *hidden_states.shape[1:]), dtype=hidden_states.dtype, device=device
        )
        if copy_hidden
        else None
    )
    rows_out = (
        torch.empty((bs, vocab_size), dtype=next_token_logits.dtype, device=device)
        if write_rows
        else None
    )
    conf = (
        torch.empty((bs,), dtype=torch.float32, device=device) if write_prob else None
    )
    if bs == 0:
        return topk_p, topk_index, hidden_out, rows_out, conf

    block = _DRAFT_TOPK1_BLOCK
    num_splits = triton.cdiv(vocab_size, block)
    partial_vals = torch.empty((bs, num_splits), dtype=torch.float32, device=device)
    partial_indices = torch.empty((bs, num_splits), dtype=torch.int32, device=device)
    partial_sums = (
        torch.empty((bs, num_splits), dtype=torch.float32, device=device)
        if write_prob
        else partial_vals
    )
    hidden_size = hidden_rows.shape[1] if copy_hidden else 0
    # Disabled operands get a valid dummy pointer; constexpr flags gate them off.
    _draft_extend_select_partial_kernel[(bs, num_splits + int(copy_hidden))](
        next_token_logits,
        row_src,
        partial_vals,
        partial_indices,
        partial_sums,
        rows_out if write_rows else next_token_logits,
        hidden_rows if copy_hidden else next_token_logits,
        hidden_out if copy_hidden else next_token_logits,
        next_token_logits.stride(0),
        hidden_rows.stride(0) if copy_hidden else 0,
        next_token_logits.shape[0],
        hidden_rows.shape[0] if copy_hidden else 0,
        num_tokens_per_req,
        vocab_size,
        hidden_size,
        num_splits,
        ROW_FROM_ACCEPT=row_from_accept,
        WRITE_ROWS=write_rows,
        WRITE_PROB=write_prob,
        COPY_HIDDEN=copy_hidden,
        BLOCK=block,
        HIDDEN_BLOCK=triton.next_power_of_2(max(hidden_size, 1)),
        num_warps=8,
    )
    _draft_extend_select_finalize_kernel[(bs,)](
        partial_vals,
        partial_indices,
        partial_sums,
        topk_p,
        topk_index,
        conf if write_prob else topk_p,
        num_splits,
        WRITE_PROB=write_prob,
        BLOCK=triton.next_power_of_2(num_splits),
        num_warps=1,
    )
    return topk_p, topk_index, hidden_out, rows_out, conf
