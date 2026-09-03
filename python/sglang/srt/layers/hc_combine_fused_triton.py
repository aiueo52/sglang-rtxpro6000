"""Single-launch HyperConnection combine (gate + apply in one Triton kernel).

``GatedResidual.combine`` currently runs the sgl-kernel split pair
``hc_combine_gate`` + ``hc_combine_apply`` (``jit/csrc/elementwise/hc_combine.cuh``).
At the decode/verify widths of Qwen3.8-Flash-Next that is 2 launches x 96
boundaries per step, and each launch costs ~1 us of fixed time on top of a kernel
that only moves ~1 MB.  This module does the same math in one launch:

    a[m, c] = 2 * sigmoid(dot(normed_residual[m, :], inject_weight[c, :]) / HC)
    out[m, c*HS + i] = residual[m, c*HS + i] + a[m, c] * block_output[m, i]

Geometry.  The row (HC*HS = 10240 elements for hc_count 4 / hidden 2560) is cut
into ``NPROG`` contiguous power-of-two chunks; program ``(m, p)`` owns chunk ``p``
of row ``m`` for both phases.  Phase 1 accumulates that chunk's contribution to all
HC gate dots into a persistent fp32 ``partials`` buffer, a grid barrier publishes
them, phase 2 reduces the NPROG partials per branch, forms the gates and writes the
chunk of the output row.  A chunk may straddle a branch boundary, so the gate is
selected per element rather than per CTA.

Grid barrier safety.  ``rows * NPROG`` is kept <= the SM count, so every CTA of the
launch is resident before the barrier is reached (a CUDA-graph replay serialises the
stream, and PDL only lets a successor start once this grid is fully launched).  The
wrapper falls back to the two-kernel path whenever that does not hold.  The two
counters are restored to 0 by the last CTA out, so a graph replay starts clean.

Numerics.  Everything is fp32 with a bf16 round at the store, exactly as the CUDA
pair does, but the fp32 dot is reduced in a different order (per-chunk tl.sum tree
plus a sequential fold over the chunks, vs. the reference's per-thread sequential
sum, warp butterfly and sequential fold over its 8 splits).  The gate therefore
differs by a few fp32 ulp, i.e. ~1e-7 relative; the bf16 output only moves when the
fp32 result sits within that of a bf16 rounding boundary, and then by exactly one
bf16 ulp.  See ``test/srt/layers/test_hc_combine_fused.py`` for the measured rate.

Enable with ``SGLANG_HC_COMBINE_FUSED=1``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

try:  # __nv_expf, matching the reference's math::exp -> expf
    from triton.language.extra.libdevice import exp as _exp_f32
except Exception:  # pragma: no cover - older Triton
    _exp_f32 = tl.exp

# Chunk counts tried from the most parallel down; the first one whose grid fits the
# machine wins. ROW // NPROG must be a power of two >= 256 for a clean 16 B access.
_NPROG_CANDIDATES = (40, 20, 10, 5)
_MAX_ROWS = 32
_MAX_CTAS = 256  # partials/counters are sized for this many programs


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Bench hooks: pin the chunk count / warp count instead of letting _plan choose.
_FORCE_NPROG = _env_int("SGLANG_HC_COMBINE_FUSED_NPROG", 0)
_FORCE_WARPS = _env_int("SGLANG_HC_COMBINE_FUSED_WARPS", 0)


@triton.jit
def _hc_combine_fused_kernel(
    y_ptr,  # block_output      [M, HS]      bf16/fp16
    r_ptr,  # residual          [M, HC*HS]
    n_ptr,  # normed_residual   [M, HC*HS]
    w_ptr,  # inject_weight     [HC, HC*HS]
    o_ptr,  # out               [M, HC*HS]
    part_ptr,  # fp32           [>= M*NPROG, HC]
    cnt_ptr,  # int32           [2]
    n_ctas,
    HC: tl.constexpr,
    HS: tl.constexpr,
    NPROG: tl.constexpr,
    BLK: tl.constexpr,
):
    ROW: tl.constexpr = HC * HS
    pid = tl.program_id(0)
    m = pid // NPROG
    p = pid % NPROG

    offs = p * BLK + tl.arange(0, BLK)
    n = tl.load(n_ptr + m * ROW + offs).to(tl.float32)

    # ---- phase 1: this chunk's contribution to each branch's gate dot ----
    for c in tl.static_range(HC):
        w = tl.load(w_ptr + c * ROW + offs).to(tl.float32)
        tl.store(
            part_ptr + pid * HC + c,
            tl.sum(n * w, axis=0),
            cache_modifier=".cg",
        )

    # ---- grid barrier ----
    tl.debug_barrier()
    tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(cnt_ptr, 0, sem="acq_rel", scope="gpu") < n_ctas:
        pass

    # ---- phase 2: reduce the gate dots and stream the chunk ----
    branch = offs // HS
    col = offs - branch * HS
    r = tl.load(r_ptr + m * ROW + offs).to(tl.float32)
    yv = tl.load(y_ptr + m * HS + col).to(tl.float32)

    a_vec = tl.zeros([BLK], tl.float32)
    base = m * NPROG
    for c in tl.static_range(HC):
        total = tl.load(part_ptr + base * HC + c, cache_modifier=".cg")
        for s in tl.static_range(1, NPROG):
            total += tl.load(part_ptr + (base + s) * HC + c, cache_modifier=".cg")
        # 2 * sigmoid(dot / HC), the reference's `2.0f / (1.0f + expf(-total / HC))`.
        a_c = 2.0 / (1.0 + _exp_f32(-total / HC))
        a_vec = tl.where(branch == c, a_c, a_vec)

    tl.store(o_ptr + m * ROW + offs, (r + a_vec * yv).to(o_ptr.dtype.element_ty))

    # ---- restore the counters for the next launch / graph replay ----
    ticket = tl.atomic_add(cnt_ptr + 1, 1, sem="acq_rel", scope="gpu")
    if ticket == n_ctas - 1:
        tl.store(cnt_ptr, 0)
        tl.store(cnt_ptr + 1, 0)


_STATE: dict = {}


def _num_sms(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _state(device, hc_count: int):
    key = (str(device), hc_count)
    st = _STATE.get(key)
    if st is None:
        # Outside inference_mode: the kernel mutates both buffers in place and
        # CUDA-graph capture rejects in-place writes to inference tensors.
        with torch.inference_mode(False):
            partials = torch.zeros(
                (_MAX_CTAS, hc_count), dtype=torch.float32, device=device
            )
            counters = torch.zeros(2, dtype=torch.int32, device=device)
        st = (partials, counters)
        _STATE[key] = st
    return st


def _plan(rows: int, row_size: int, sms: int):
    """(NPROG, BLK, num_warps) or None when no chunking fits the grid barrier."""
    for nprog in _NPROG_CANDIDATES:
        if _FORCE_NPROG and nprog != _FORCE_NPROG:
            continue
        if row_size % nprog:
            continue
        blk = row_size // nprog
        if blk < 256 or (blk & (blk - 1)):  # power of two, >= 8 elems/thread
            continue
        if rows * nprog > sms:
            continue
        warps = _FORCE_WARPS or max(1, min(8, blk // 256))
        return nprog, blk, warps
    return None


def hc_combine_fused_supported(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> bool:
    """Whether `hc_combine_fused` can serve this call (else use the two-kernel path).

    The kernel addresses every tensor with compact strides and needs its whole grid
    resident at the barrier, so anything non-contiguous or too wide falls back.
    """
    row_size = hc_count * hidden_size
    tensors = (block_output, residual, normed_residual, inject_weight)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        return False
    if residual.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not all(t.dtype == residual.dtype for t in tensors):
        return False
    if tuple(inject_weight.shape) != (hc_count, row_size):
        return False
    rows = residual.numel() // row_size
    if rows < 1 or rows > _MAX_ROWS or rows * row_size != residual.numel():
        return False
    if block_output.numel() != rows * hidden_size:
        return False
    if normed_residual.numel() != residual.numel():
        return False
    return _plan(rows, row_size, _num_sms(residual.device)) is not None


def hc_combine_fused(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-launch equivalent of ``hc_combine`` / ``hc_combine_split``."""
    row_size = hc_count * hidden_size
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, row_size)
    n = normed_residual.reshape(-1, row_size)
    o = torch.empty_like(r) if out is None else out.reshape(-1, row_size)
    rows = r.shape[0]

    plan = _plan(rows, row_size, _num_sms(r.device))
    if plan is None:
        raise ValueError(
            f"hc_combine_fused: no chunking fits rows={rows} row_size={row_size}"
        )
    nprog, blk, warps = plan
    n_ctas = rows * nprog
    partials, counters = _state(r.device, hc_count)

    _hc_combine_fused_kernel[(n_ctas,)](
        y,
        r,
        n,
        inject_weight,
        o,
        partials,
        counters,
        n_ctas,
        HC=hc_count,
        HS=hidden_size,
        NPROG=nprog,
        BLK=blk,
        num_warps=warps,
        num_stages=1,
    )
    return o.reshape(residual.shape)
