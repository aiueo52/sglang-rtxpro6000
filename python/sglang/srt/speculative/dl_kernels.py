# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""DL stopping state, shared with VT's device length buffer."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dl_policy import dl_hostw_enabled


@triton.jit
def _advance_kernel(confidence, product, lengths, predicate, forwards,
                    K: tl.constexpr, STEPS: tl.constexpr, TAU: tl.constexpr):
    c = tl.load(confidence + K)
    p = tl.where(TAU == 0.0, 1.0, tl.load(product) * c)
    keep = (TAU == 0.0) | ((c >= 0.0) & (c <= 1.0) & (p >= TAU))
    tl.store(product, tl.where(keep, p, 0.0))
    if keep:
        tl.store(lengths, K + 1)
    tl.store(predicate, keep & (K + 1 < STEPS))
    if K > 0:
        tl.store(forwards, tl.load(forwards) + 1)


@triton.jit
def _advance_batch_kernel(confidence, product, lengths, predicate, forwards,
                          STRIDE: tl.constexpr, BS: tl.constexpr,
                          K: tl.constexpr, STEPS: tl.constexpr,
                          TAU: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    live = (row < BS) & (tl.load(lengths + row, mask=row < BS, other=-1) == K)
    c = tl.load(confidence + row * STRIDE + K, mask=row < BS, other=0.0)
    previous = tl.load(product + row, mask=row < BS, other=0.0)
    p = tl.where(TAU == 0.0, 1.0, previous * c)
    keep = live & ((TAU == 0.0) | ((c >= 0.0) & (c <= 1.0) & (p >= TAU)))
    tl.store(product + row, tl.where(keep, p, 0.0), mask=row < BS)
    tl.store(lengths + row, K + 1, mask=keep)
    active = keep & (K + 1 < STEPS)
    tl.store(predicate, tl.sum(active.to(tl.int32), axis=0) > 0)
    if K > 0:
        tl.store(forwards, tl.load(forwards) + 1)


def advance(confidence, product, lengths, predicate, forwards, k, steps, tau):
    if dl_hostw_enabled() and confidence.shape[0] > 1:
        if (
            confidence.ndim != 2
            or confidence.dtype != torch.float32
            or not confidence.is_cuda
            or confidence.stride(1) != 1
            or not 0 <= k < steps <= confidence.shape[1]
            or not 0 <= tau <= 1
        ):
            raise ValueError("HOSTW DL requires complete CUDA float32 confidence rows")
        bs = confidence.shape[0]
        for tensor, dtype, capacity in (
            (product, torch.float32, bs),
            (lengths, torch.int32, bs),
            (predicate, torch.bool, 1),
            (forwards, torch.int32, 1),
        ):
            if (
                tensor.ndim != 1
                or tensor.dtype != dtype
                or tensor.device != confidence.device
                or not tensor.is_contiguous()
                or tensor.numel() < capacity
                or (capacity == 1 and tensor.numel() != 1)
            ):
                raise ValueError("HOSTW DL state must have matching contiguous CUDA storage")
        _advance_batch_kernel[(1,)](
            confidence, product, lengths, predicate, forwards,
            STRIDE=confidence.stride(0), BS=bs, K=k, STEPS=steps, TAU=tau,
            BLOCK=triton.next_power_of_2(bs), num_warps=4,
        )
        return
    _advance_kernel[(1,)](
        confidence, product, lengths, predicate, forwards,
        K=k, STEPS=steps, TAU=tau, num_warps=1,
    )
