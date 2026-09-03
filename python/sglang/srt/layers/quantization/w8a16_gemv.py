"""Triton W8A16 skinny GEMM: y[M,N] = x[M,K](bf16) @ W[N,K](fp8 e4m3)^T * scale[N] (fp32).
Targets decode/verify shapes (M <= 16). Weight-only FP8: no activation quantization.

Two properties of the decode shapes decide throughput here:

- Weight layout. Every caller hands over an [N, K]-CONTIGUOUS weight, stride (K, 1).
  Fp8LinearMethod quantizes to [N, K], stores the [K, N] transposed view on the layer,
  and transposes it back at the call site; bf16_gemv passes linear.weight directly.
  So K is the contiguous axis, the natural weight tile is [BLOCK_N, BLOCK_K], and
  tl.dot needs it transposed (W_KN False). The W_KN True path is for a genuinely
  [K, N]-contiguous weight, which no caller produces today.
- Grid size. One CTA per N block owning the whole K reduction launches 3..80 CTAs for
  the small-N linears (in_proj_ba, shared-expert gate_up/down, GDN out_proj, o_proj) on
  a 188-SM part, so those calls run at a few hundred GB/s no matter how well the inner
  loop is written. Splitting K across CTAs fixes that, and the reduction is folded into
  the same launch: every CTA writes an fp32 partial and bumps a per-N-block counter, and
  whoever observes the last increment sums the partials, scales, and writes bf16 y. The
  counter is reset to 0 by that same CTA, so no memset is needed and the kernel is safe
  to capture in a CUDA graph and replay.

M == 1 additionally wants USE_DOT False on every shape measured (2-23% faster): the
broadcast-multiply path keeps M_PAD at 1 instead of padding to a 16-row mma, which
shrinks both the accumulator and the split-K partials 16-fold.

The split-K scratch is one buffer per device, so two split-K calls must not overlap;
that holds for the decode path, which issues every linear on one stream.
"""

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_gemv_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    y_ptr,
    ws_ptr,
    cnt_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    PER_CHANNEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_PAD: tl.constexpr,
    SPLITS: tl.constexpr,
    EVEN_K: tl.constexpr,
    W_KN: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_k * BLOCK_K, K, SPLITS * BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        # W_KN: the weight is [K, N] contiguous, so the [BLOCK_K, BLOCK_N] tile is the
        # contiguous one and tl.dot takes it directly instead of transposing in smem.
        # Every current caller is [N, K] contiguous and takes the else branch.
        if W_KN:
            wp = w_ptr + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn
            if EVEN_K:
                w = tl.load(wp, mask=n_mask[None, :], other=0.0)
            else:
                w = tl.load(wp, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        else:
            wp = w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk
            if EVEN_K:
                w = tl.load(wp, mask=n_mask[:, None], other=0.0)
            else:
                w = tl.load(wp, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        if USE_DOT:
            xp = x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk
            if EVEN_K:
                x = tl.load(xp, mask=m_mask[:, None], other=0.0)
            else:
                x = tl.load(xp, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            if W_KN:
                acc += tl.dot(x, w.to(tl.bfloat16), out_dtype=tl.float32)
            else:
                acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
        else:
            # M == 1: broadcast-multiply and reduce instead of padding to a 16-row mma.
            if EVEN_K:
                xv = tl.load(x_ptr + kk * stride_xk).to(tl.float32)
            else:
                xv = tl.load(x_ptr + kk * stride_xk, mask=k_mask, other=0.0).to(tl.float32)
            if W_KN:
                acc += tl.sum(w.to(tl.float32) * xv[:, None], axis=0)[None, :]
            else:
                acc += tl.sum(w.to(tl.float32) * xv[None, :], axis=1)[None, :]

    if SPLITS == 1:
        if PER_CHANNEL:
            acc = acc * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        else:
            acc = acc * tl.load(s_ptr)
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])
        return

    # Split-K fixup, in this same launch. `.cg` keeps the partials out of the
    # non-coherent L1 so the CTA that runs the reduction sees the other CTAs' writes.
    slot = offs_m[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    base = ws_ptr + (pid_n * SPLITS) * (M_PAD * BLOCK_N)
    tl.store(base + pid_k * (M_PAD * BLOCK_N) + slot, acc, cache_modifier=".cg")
    tl.debug_barrier()
    done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
    if done == SPLITS - 1:
        tot = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            tot += tl.load(base + s * (M_PAD * BLOCK_N) + slot, cache_modifier=".cg")
        if PER_CHANNEL:
            tot = tot * tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        else:
            tot = tot * tl.load(s_ptr)
        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, tot.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])
        # Every increment for this N block has happened, so a plain store is enough to
        # leave the counter at 0 for the next launch / graph replay.
        tl.store(cnt_ptr + pid_n, 0)


# Split-K scratch, allocated once per device and reused: the fixup CTA leaves the
# counters at 0, so nothing has to be cleared between calls. Sized for the largest
# plan _plan can emit (MAX_N_BLOCKS n blocks x MAX_SPLITS splits x 16 rows x 128 wide
# would be 8 M floats; the planner caps the product at _WS_FLOATS instead).
_WS_FLOATS = 1 << 21
_WS_COUNTERS = 4096
_MAX_SPLITS = 32
_WS = {}


def _workspace(device):
    ws = _WS.get(device)
    if ws is None:
        ws = (
            torch.empty(_WS_FLOATS, dtype=torch.float32, device=device),
            torch.zeros(_WS_COUNTERS, dtype=torch.int32, device=device),
        )
        _WS[device] = ws
    return ws


_NUM_SMS = None


def _num_sms(device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


def _m_bucket(M: int) -> int:
    return 1 if M == 1 else (4 if M <= 4 else 16)


# Measured on RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs) with
# bench/w8a16v2/tune_v3.py + the head-to-head re-check in bench/w8a16v2/final_v3.py;
# the metric is the CUPTI kernel-duration median over a CUDA-graph replay whose
# weight working set is several times the 128 MB L2, i.e. the number the serving
# torch profile reports. Re-run it after a kernel or hardware change.
#
# key:   (weight element bytes, N is the contiguous weight axis, N, K)
# value: (BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, num_warps, num_stages)
# W_KN None derives the tile orientation from the weight strides.
#
# `N is the contiguous weight axis` is False on every row: both callers pass an
# [N, K]-contiguous weight (see the module docstring). The pre-v3 table keyed the
# fp8 rows True, which no call ever matched, so the server silently ran _plan's
# fallback for all of them; the "vs" ratios below are against that fallback.
_BY_SHAPE = {
    # fp8, w8a16_gemv
    # shared-expert down: the single-split (32,128,1,...,4,4) bench winner measured 5.2us
    # in the W4 server (M=4) vs 4.0us for the split-5 fallback; M=16 keeps the winner below.
    (1, False, 2560, 640): (32, 128, 5, True, None, 4, 3),
    (1, False, 1280, 2560): (16, 128, 10, True, None, 2, 2),  # shared gate_up, 1.08x
    (1, False, 32768, 2560): (128, 256, 1, True, None, 8, 3),  # draft lm_head, 1.14x
    (1, False, 248320, 2560): (128, 256, 1, True, None, 8, 3),  # lm_head, 1.02x
    # bf16, bf16_gemv
    # GDN in_proj_ba: the bench's split-K winner (16,256,10,...,2,4) measured 9.2us in the
    # server vs 7.9us for this single-launch tile (2026-09-03 gemv3 trace), so keep it.
    (2, False, 96, 2560): (32, 512, 1, True, None, 4, 3),
    (2, False, 512, 2560): (16, 128, 10, True, None, 2, 4),  # MoE router gate, 1.04x
    # The 2560x6144 out_proj/o_proj and the 13312x2560 attention qkv are absent at
    # M > 1 because _plan's fallback already emits the fastest tile measured for them.
}
_BY_SHAPE_M16 = {
    # The _BY_SHAPE gate_up tile loses 12% at M=16; the qkvz tile below loses 1% at M=4.
    (1, False, 1280, 2560): (16, 64, 10, True, None, 4, 3),  # shared gate_up, 1.03x
    (1, False, 16384, 2560): (64, 128, 1, True, None, 4, 3),  # GDN in_proj_qkvz, 1.04x
    (1, False, 2560, 640): (32, 128, 1, True, None, 4, 4),  # shared-expert down (W16 server 4.1->3.8us)
    # attention qkv at M=16: both the wide (128,256,8w) tile and the (64,128) qkvz tile
    # measured 27.5us in the server vs 25.6us for the narrow single-split tile; pin it.
    (1, False, 13312, 2560): (32, 128, 1, True, None, 4, 3),
}
_BY_SHAPE_M1 = {
    # USE_DOT False (the broadcast path, M_PAD 1) is only legal at M == 1.
    (1, False, 2560, 640): (16, 256, 3, False, None, 4, 3),  # shared-expert down, 1.20x
    (1, False, 1280, 2560): (16, 128, 10, False, None, 2, 3),  # shared gate_up, 1.23x
    (1, False, 2560, 6144): (16, 256, 6, False, None, 4, 3),  # out_proj/o_proj, 1.05x
    (1, False, 13312, 2560): (32, 256, 1, False, None, 4, 3),  # attention qkv, 1.02x
    (1, False, 32768, 2560): (32, 256, 1, False, None, 4, 3),  # draft lm_head, 1.17x
    (2, False, 512, 2560): (16, 256, 10, False, None, 4, 3),  # MoE router gate, 1.15x
    (2, False, 2560, 2560): (16, 256, 5, False, None, 4, 2),  # MTP fc_*, 1.06x
}


def _expand(table, buckets):
    return {
        (wb, cn, m, N, K): cfg
        for (wb, cn, N, K), cfg in table.items()
        for m in buckets
    }


_TUNED = _expand(_BY_SHAPE, (1, 4, 16))
_TUNED.update(_expand(_BY_SHAPE_M16, (16,)))
_TUNED.update(_expand(_BY_SHAPE_M1, (1,)))


def _prev_pow2(v: int) -> int:
    return 1 << max(0, v.bit_length() - 1)


@functools.lru_cache(maxsize=512)
def _plan(M: int, N: int, K: int, contig_n: bool, w_bytes: int, sms: int):
    """(BLOCK_N, BLOCK_K, SPLITS, USE_DOT, W_KN, num_warps, num_stages) for one call."""
    tuned = _TUNED.get((w_bytes, contig_n, _m_bucket(M), N, K))
    if tuned is not None:
        return _fit(M, N, tuned)
    if contig_n:
        # A [K, N]-contiguous weight wants the widest N block it can keep the grid
        # full with, since 32 fp8 columns is only one 32 B sector per k. Untuned:
        # no caller produces this layout.
        block_n = 128 if N >= 2048 else (64 if N >= 512 else 32)
        return _fit(M, N, (block_n, 128, 1, True, None, 8, 3))
    # [N, K] contiguous. When N alone fills the machine, a long BLOCK_K keeps more
    # bytes in flight: 1.15x on the 32768x2560 draft head, 1.01-1.04x on the other
    # three big shapes. The M == 4 bucket is the exception -- there the wide tile
    # measured 1-2% slower than the [32, 128] one on both 2560-K shapes.
    if triton.cdiv(N, 128) * 2 >= sms and _m_bucket(M) != 4:
        if M == 1:
            return (32, 256, 1, False, None, 4, 3)
        # A [128, 256] bf16 tile needs 144 KB of smem, over the 99 KB sm_120 limit.
        return (128 if w_bytes == 1 else 64, 256, 1, True, None, 8, 3)
    use_dot = M > 1
    block_n = 16 if M == 1 else 32
    block_k, warps = 128, 4
    if triton.cdiv(N, 32) >= sms:
        return (block_n, block_k, 1, use_dot, None, warps, 3)
    # Aim for ~2 CTAs per SM, ~5 at M == 1 where an M_PAD 1 partial is 16x cheaper;
    # shrink BLOCK_K if K does not hold enough blocks for that.
    per_sm = 5 if M == 1 else 2
    want = min(_MAX_SPLITS, max(1, (per_sm * sms) // triton.cdiv(N, block_n)))
    block_k = min(block_k, max(64, _prev_pow2(max(1, K // want))))
    n_kb = triton.cdiv(K, block_k)
    splits = min(want, n_kb)
    # Prefer a split count that divides the k-block count so every CTA does equal work.
    for cand in range(min(n_kb, splits + 2), max(1, splits - 3), -1):
        if n_kb % cand == 0 and cand <= _MAX_SPLITS:
            splits = cand
            break
    return _fit(M, N, (block_n, block_k, splits, use_dot, None, warps, 3))


def _fit(M: int, N: int, cfg):
    """Drop to SPLITS=1 if the plan would not fit the preallocated scratch."""
    block_n, block_k, splits, use_dot, w_kn, warps, stages = cfg
    if splits == 1:
        return cfg
    m_pad = 16 if (use_dot or M > 1) else 1
    n_blocks = triton.cdiv(N, block_n)
    if (
        splits > _MAX_SPLITS
        or n_blocks > _WS_COUNTERS
        or n_blocks * splits * m_pad * block_n > _WS_FLOATS
    ):
        return (block_n, block_k, 1, use_dot, w_kn, warps, stages)
    return cfg


def _launch(x, w, s, y, M, N, K, per_channel, cfg):
    block_n, block_k, splits, use_dot, w_kn, num_warps, num_stages = cfg
    use_dot = use_dot or M > 1
    m_pad = 16 if use_dot else 1
    ws, cnt = _workspace(x.device)
    _w8a16_gemv_kernel[(triton.cdiv(N, block_n), splits)](
        x,
        w,
        s,
        y,
        ws,
        cnt,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        PER_CHANNEL=per_channel,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_PAD=m_pad,
        SPLITS=splits,
        EVEN_K=K % block_k == 0,
        W_KN=(w.stride(0) == 1) if w_kn is None else w_kn,
        USE_DOT=use_dot,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def w8a16_gemv(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, cfg=None):
    """x: [M,K] bf16; w: [N,K] fp8_e4m3 (any strides); scale: [N] or [N,1] or scalar fp32."""
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    s = scale.reshape(-1)
    if s.dtype != torch.float32 or not s.is_contiguous():
        s = s.contiguous().float()
    if cfg is None:
        cfg = _plan(M, N, K, w.stride(0) == 1, w.element_size(), _num_sms(x.device))
    return _launch(x, w, s, y, M, N, K, s.numel() > 1, cfg)


_ONES = {}


def bf16_gemv(x: torch.Tensor, w: torch.Tensor, cfg=None):
    """Skinny BF16 GEMM y = x @ w^T for tiny-N linears where cuBLAS picks a poor kernel.
    x: [M,K] bf16 (M<=16); w: [N,K] bf16."""
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    one = _ONES.get(x.device)
    if one is None:
        one = torch.ones(1, dtype=torch.float32, device=x.device)
        _ONES[x.device] = one
    if cfg is None:
        cfg = _plan(M, N, K, w.stride(0) == 1, w.element_size(), _num_sms(x.device))
    return _launch(x, w, one, y, M, N, K, False, cfg)
