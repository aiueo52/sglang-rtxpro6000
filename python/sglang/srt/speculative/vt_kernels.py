# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Graph-safe VT prefix lengths and -1/0 MoE tail masking."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import triton
import triton.language as tl

from sglang.srt.speculative.dl_policy import dl_hostw_enabled
from sglang.srt.speculative.vt_policy import vt_tau

# Shared by all captured adaptive-width graphs. Never replace these tensors:
# target graphs may be captured before the draft worker exists.
_LENGTHS = {}
_ACTIVE_LENGTHS = None
_ACTIVE_BATCH_SIZE = 1


@triton.jit
def _prefix_length_kernel(confidence, lengths, STRIDE: tl.constexpr,
                          STEPS: tl.constexpr, TAU: tl.constexpr):
    row = tl.program_id(0)
    product = 1.0
    length = 0
    alive = True
    for k in tl.static_range(STEPS):
        c = tl.load(confidence + row * STRIDE + k)
        alive = alive & (c >= 0.0) & (c <= 1.0)
        product = product * c
        alive = alive & (product >= TAU)
        length = tl.where(alive, k + 1, length)
    tl.store(lengths + row, length)


@triton.jit
def _mask_tail_kernel(ids, weights, lengths, ROWS: tl.constexpr,
                      K: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(lengths)
    mask = (offset < ROWS * K) & (offset // K > length)
    tl.store(ids + offset, -1, mask=mask)
    tl.store(weights + offset, 0.0, mask=mask)


@triton.jit
def _mask_batch_tail_kernel(ids, weights, lengths, ROWS: tl.constexpr,
                            ROWS_PER_REQUEST: tl.constexpr, K: tl.constexpr,
                            BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offset // K
    length = tl.load(lengths + row // ROWS_PER_REQUEST,
                     mask=offset < ROWS * K, other=0)
    mask = (offset < ROWS * K) & (row % ROWS_PER_REQUEST > length)
    tl.store(ids + offset, -1, mask=mask)
    tl.store(weights + offset, 0.0, mask=mask)


def length_buffer(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    capacity = 1
    if dl_hostw_enabled():
        from sglang.srt.runtime_context import get_exec, get_schedule

        graph_config = get_exec().graph.cuda_graph_config
        capacity = max(
            get_schedule().max_running_requests or 0,
            (graph_config.decode.max_bs or 0) if graph_config is not None else 0,
            1,
        )
    if device not in _LENGTHS:
        if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise ValueError("VT length storage must be initialized on CUDA before capture")
        _LENGTHS[device] = torch.full((capacity,), 2**30, dtype=torch.int32, device=device)
    elif _LENGTHS[device].numel() < capacity:
        raise ValueError("VT length storage cannot exceed its pre-capture capacity")
    return _LENGTHS[device]


def assert_shared_length_storage(*tensors):
    """Fail before capture if any producer/consumer uses another L allocation."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DL length storage must be checked before capture")
    if not tensors:
        raise ValueError("DL length storage check requires producer and consumer references")
    expected = length_buffer(tensors[0].device)
    for tensor in tensors:
        if (
            tensor.device != expected.device
            or tensor.dtype != torch.int32
            or tensor.shape != expected.shape
            or tensor.untyped_storage().data_ptr() != expected.untyped_storage().data_ptr()
            or tensor.data_ptr() != expected.data_ptr()
        ):
            raise RuntimeError("DL worker, target and sampler must share length storage")


def compute_lengths(confidence, steps):
    if confidence.ndim != 2 or confidence.shape[0] < 1 or not 0 < steps <= confidence.shape[1]:
        raise ValueError("VT confidence must contain the complete draft chain")
    if confidence.stride(1) != 1 or confidence.dtype != torch.float32 or not confidence.is_cuda:
        raise ValueError("VT requires contiguous CUDA float32 confidence columns")
    lengths = length_buffer(confidence.device)
    bs = confidence.shape[0] if dl_hostw_enabled() else 1
    if bs > lengths.numel():
        raise ValueError("VT confidence exceeds pre-capture length capacity")
    _prefix_length_kernel[(bs,)](
        confidence, lengths, confidence.stride(0), steps, vt_tau(steps=steps), num_warps=1
    )
    return lengths


@contextmanager
def verify_moe_context(forward_batch, rows, device):
    global _ACTIVE_LENGTHS, _ACTIVE_BATCH_SIZE
    bs = forward_batch.batch_size
    if rows < 1 or bs != 1:
        raise ValueError("VT/HOSTW requires one nonempty unpadded request")
    lengths = length_buffer(device)
    if rows % bs != 0 or bs > lengths.numel():
        raise ValueError("VT requires equal verify rows within pre-capture batch capacity")
    previous = _ACTIVE_LENGTHS, _ACTIVE_BATCH_SIZE
    _ACTIVE_LENGTHS, _ACTIVE_BATCH_SIZE = lengths[:bs], bs
    try:
        yield
    finally:
        _ACTIVE_LENGTHS, _ACTIVE_BATCH_SIZE = previous


def active_lengths():
    """Stable device cap for this verify MoE, or None for ordinary forwards."""
    return _ACTIVE_LENGTHS


def mask_verify_tail_(ids, weights):
    if _ACTIVE_LENGTHS is None:
        return
    if ids.ndim != 2 or ids.shape != weights.shape or not ids.is_contiguous() or not weights.is_contiguous():
        raise ValueError("VT requires contiguous matching [rows, experts] routes")
    if ids.device != _ACTIVE_LENGTHS.device or weights.device != ids.device:
        raise ValueError("VT routes and lengths must use the same device")
    rows, k = ids.shape
    if _ACTIVE_BATCH_SIZE > 1:
        if rows < _ACTIVE_BATCH_SIZE or rows % _ACTIVE_BATCH_SIZE != 0 or k < 1:
            raise ValueError("VT requires nonempty equal-sized per-request routes")
        _mask_batch_tail_kernel[(triton.cdiv(rows * k, 128),)](
            ids, weights, _ACTIVE_LENGTHS, rows, rows // _ACTIVE_BATCH_SIZE, k,
            BLOCK=128, num_warps=4,
        )
        return
    _mask_tail_kernel[(triton.cdiv(rows * k, 128),)](
        ids, weights, _ACTIVE_LENGTHS, rows, k, BLOCK=128, num_warps=4
    )
