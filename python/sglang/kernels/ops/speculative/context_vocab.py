# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Device-only FIFO planning and dirty-row copies for the draft context head."""

from __future__ import annotations

import operator

import torch
import triton
import triton.language as tl


@triton.jit
def _plan_context_vocab_kernel(
    Ids,
    Lengths,
    HotMask,
    Resident,
    Cursor,
    Dirty,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    VOCAB: tl.constexpr,
    CAPACITY: tl.constexpr,
    HAS_LENGTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A single CTA serializes insertion order across requests. Keep the FIFO
    # in registers, so hits do not refresh age and evicted IDs may reenter.
    offsets = tl.arange(0, BLOCK)
    slots = tl.load(Resident + offsets, mask=offsets < CAPACITY, other=-1)
    dirty = tl.full((BLOCK,), 0, tl.int32)
    cursor = tl.load(Cursor)
    for row in range(BATCH):
        length = tl.load(Lengths + row) if HAS_LENGTHS else WIDTH
        tl.device_assert((length >= 0) & (length <= WIDTH), "invalid accepted length")
        length = tl.minimum(tl.maximum(length, 0), WIDTH)
        for column in range(length):
            token = tl.load(Ids + row * WIDTH + column)
            valid = (token >= 0) & (token < VOCAB)
            tl.device_assert(valid, "accepted token outside vocabulary")
            # Guard every indirect load even when device assertions are disabled.
            if valid:
                hot = tl.load(HotMask + token)
                if CAPACITY > 0:
                    if not hot:
                        present = tl.sum(
                            ((slots == token) & (offsets < CAPACITY)).to(tl.int32), 0
                        )
                        if present == 0:
                            slots = tl.where(offsets == cursor, token, slots)
                            dirty = tl.where(offsets == cursor, 1, dirty)
                            cursor = (cursor + 1) % CAPACITY
    tl.store(Resident + offsets, slots, mask=offsets < CAPACITY)
    tl.store(Dirty + offsets, dirty, mask=offsets < CAPACITY)
    tl.store(Cursor, cursor)


@triton.jit
def _copy_context_vocab_kernel(
    Source,
    SourceScale,
    Resident,
    Dirty,
    Weight,
    Scale,
    TokenMap,
    source_stride_hidden,
    source_stride_vocab,
    weight_stride_hidden,
    weight_stride_vocab,
    HOT: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    if tl.load(Dirty + slot) != 0:
        token = tl.load(Resident + slot)
        columns = tl.arange(0, BLOCK)
        row = WEIGHT_OFFSET + slot
        # No `other`: Triton cannot cast the integer fill to fp8e4nv (GPU compile error); the masked
        # lanes are never stored because the store below uses the same mask.
        values = tl.load(
            Source + token * source_stride_vocab + columns * source_stride_hidden,
            mask=columns < HIDDEN,
        )
        tl.store(
            Weight + row * weight_stride_vocab + columns * weight_stride_hidden,
            values,
            mask=columns < HIDDEN,
        )
        tl.store(Scale + row, tl.load(SourceScale + token))
        tl.store(TokenMap + HOT + slot, token)


@triton.jit
def _update_context_map_kernel(
    Resident,
    Dirty,
    TokenMap,
    HOT: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dirty = tl.load(Dirty + offsets, mask=offsets < CAPACITY, other=0)
    tokens = tl.load(Resident + offsets, mask=offsets < CAPACITY, other=0)
    tl.store(TokenMap + HOT + offsets, tokens, mask=(offsets < CAPACITY) & (dirty != 0))


@triton.jit
def _mask_context_logits_kernel(
    Logits,
    Resident,
    PrefixMask,
    row_stride,
    HOT: tl.constexpr,
    CAPACITY: tl.constexpr,
    HAS_PREFIX_MASK: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tokens = tl.load(Resident + offsets, mask=offsets < CAPACITY, other=-1)
    invalid = tokens < 0
    if HAS_PREFIX_MASK:
        duplicate = tl.load(
            PrefixMask + tl.maximum(tokens, 0),
            mask=(offsets < CAPACITY) & (tokens >= 0) & (tokens < VOCAB),
            other=False,
        )
        invalid = invalid | duplicate
    tl.store(
        Logits + row * row_stride + HOT + offsets,
        -float("inf"),
        mask=(offsets < CAPACITY) & invalid,
    )


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def update_context_vocab(
    ids,
    accept_lens,
    width,
    hot_mask,
    resident,
    cursor,
    dirty,
    source_head,
    source_scale,
    weight,
    weight_scale,
    token_map,
    hot_count,
    weight_offset=None,
):
    """Plan accepted prefixes and copy only dirty tail rows, on the current stream.

    Metadata checks use shapes/dtypes only. Accepted lengths and token values
    are checked on-device; rejected padding is never read. ``None`` lengths
    accept all columns and need no temporary device tensor.
    """
    width = _integer(width, "width")
    if width <= 0:
        raise ValueError("width must be positive")
    if (
        ids.ndim != 1
        or ids.dtype not in (torch.int32, torch.int64)
        or ids.device != source_head.device
        or not ids.is_contiguous()
    ):
        raise ValueError("ids must be contiguous device int32/int64 [batch * width]")
    if accept_lens is None:
        if ids.numel() % width:
            raise ValueError("ids must contain width entries per request")
        batch_size = ids.numel() // width
    else:
        if (
            accept_lens.ndim != 1
            or accept_lens.dtype not in (torch.int32, torch.int64)
            or accept_lens.device != source_head.device
            or not accept_lens.is_contiguous()
        ):
            raise ValueError("accept_lens must be contiguous device int32/int64 [batch]")
        batch_size = accept_lens.numel()
        if ids.numel() != batch_size * width:
            raise ValueError("ids must contain width entries per request")
    capacity = resident.numel()
    hot_count = _integer(hot_count, "hot_count")
    weight_offset = hot_count if weight_offset is None else _integer(weight_offset, "weight_offset")
    if (
        hot_count <= 0
        or weight_offset < 0
        or tuple(weight.shape) != (source_head.shape[0], weight_offset + capacity)
        or weight_scale.numel() != weight_offset + capacity
        or token_map.ndim != 1
        or token_map.numel() != hot_count + capacity
        or token_map.dtype != torch.int64
        or not token_map.is_contiguous()
        or any(t.device != source_head.device for t in (weight, weight_scale, token_map))
    ):
        raise ValueError("context weight and map offsets must match their fixed storage")
    _plan_context_vocab_kernel[(1,)](
        ids,
        accept_lens if accept_lens is not None else ids,
        hot_mask,
        resident,
        cursor,
        dirty,
        BATCH=batch_size,
        WIDTH=width,
        VOCAB=source_head.shape[1],
        CAPACITY=capacity,
        HAS_LENGTHS=accept_lens is not None,
        BLOCK=triton.next_power_of_2(max(1, capacity)),
        num_warps=4,
        debug=True,
    )
    if capacity:
        _copy_context_vocab_kernel[(capacity,)](
            source_head,
            source_scale,
            resident,
            dirty,
            weight,
            weight_scale,
            token_map,
            source_head.stride(0),
            source_head.stride(1),
            weight.stride(0),
            weight.stride(1),
            HOT=hot_count,
            WEIGHT_OFFSET=weight_offset,
            HIDDEN=source_head.shape[0],
            BLOCK=triton.next_power_of_2(source_head.shape[0]),
            num_warps=4,
        )


def update_context_map(resident, dirty, token_map, hot_count):
    """Mirror the shared FIFO into another stable prefix+tail token map."""
    if (
        token_map.ndim != 1
        or token_map.numel() != hot_count + resident.numel()
        or token_map.dtype != torch.int64
        or token_map.device != resident.device
        or not token_map.is_contiguous()
    ):
        raise ValueError("token_map must be contiguous device int64 [prefix + capacity]")
    if resident.numel():
        _update_context_map_kernel[(triton.cdiv(resident.numel(), 256),)](
            resident,
            dirty,
            token_map,
            HOT=hot_count,
            CAPACITY=resident.numel(),
            BLOCK=256,
            num_warps=4,
        )


def mask_context_logits(logits, resident, hot_count, prefix_mask=None):
    """Mask empty slots and resident tokens already represented by this prefix."""
    if (
        logits.ndim != 2
        or logits.shape[1] != hot_count + resident.numel()
        or logits.stride(1) != 1
        or logits.device != resident.device
        or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    ):
        raise ValueError("logits must be device floating [batch, hot + capacity]")
    if prefix_mask is not None and (
        prefix_mask.ndim != 1
        or prefix_mask.dtype != torch.bool
        or prefix_mask.device != resident.device
        or not prefix_mask.is_contiguous()
    ):
        raise ValueError("prefix_mask must be contiguous device bool [vocab]")
    capacity = resident.numel()
    if logits.shape[0] and capacity:
        _mask_context_logits_kernel[(logits.shape[0], triton.cdiv(capacity, 256))](
            logits,
            resident,
            prefix_mask if prefix_mask is not None else resident,
            logits.stride(0),
            HOT=hot_count,
            CAPACITY=capacity,
            HAS_PREFIX_MASK=prefix_mask is not None,
            VOCAB=prefix_mask.numel() if prefix_mask is not None else 0,
            BLOCK=256,
            num_warps=4,
        )
    return logits
