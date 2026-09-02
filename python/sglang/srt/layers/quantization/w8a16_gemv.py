"""Triton W8A16 skinny GEMM: y[M,N] = x[M,K](bf16) @ W[N,K](fp8 e4m3)^T * scale[N] (fp32).
Targets decode/verify shapes (M <= 16). Weight-only FP8: no activation quantization.
"""
import torch, triton, triton.language as tl

@triton.jit
def _w8a16_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                       stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                       PER_CHANNEL: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, M_PAD: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, M_PAD)
    offs_k = tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0)          # [BLOCK_N, BLOCK_K] fp8
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)          # [M_PAD, BLOCK_K] bf16
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)), out_dtype=tl.float32)
    if PER_CHANNEL:
        s = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc * s[None, :]
    else:
        s = tl.load(s_ptr)
        acc = acc * s
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def w8a16_gemv(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, block_n=32, block_k=256, num_warps=4):
    """x: [M,K] bf16; w: [N,K] fp8_e4m3 (any strides); scale: [N] or [N,1] or scalar fp32."""
    M, K = x.shape
    N = w.shape[0]
    assert w.shape[1] == K and M <= 16
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    per_channel = scale.numel() > 1
    s = scale.reshape(-1).contiguous().float()
    grid = (triton.cdiv(N, block_n),)
    _w8a16_gemv_kernel[grid](x, w, s, y, M, N, K,
                             x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
                             PER_CHANNEL=per_channel, BLOCK_N=block_n, BLOCK_K=block_k, M_PAD=16, num_warps=num_warps)
    return y


_ONES = {}


def bf16_gemv(x: torch.Tensor, w: torch.Tensor, block_n=32, block_k=512, num_warps=4):
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
    grid = (triton.cdiv(N, block_n),)
    _w8a16_gemv_kernel[grid](x, w, one, y, M, N, K,
                             x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
                             PER_CHANNEL=False, BLOCK_N=block_n, BLOCK_K=block_k, M_PAD=16, num_warps=num_warps)
    return y


if __name__ == "__main__":
    import sys, time
    torch.manual_seed(0)
    dev = "cuda"
    shapes = [(4, 2560, 10240), (4, 2560, 16384), (4, 6144, 2560), (1, 2560, 248320), (4, 2560, 248320)]
    print("props:", torch.cuda.get_device_name())
    for (M, K, N) in shapes:
        wb = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        # per-channel fp8 quant
        amax = wb.abs().amax(dim=1).float().clamp(min=1e-8)
        scale = (amax / 448.0)
        w8 = (wb.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        x = torch.randn(M, K, device=dev).to(torch.bfloat16)
        ref = (x.float() @ (w8.float() * scale[:, None]).t()).to(torch.bfloat16)
        best = None
        for bn in (16, 32, 64, 128):
            for bk in (128, 256, 512):
                for nw in (2, 4, 8):
                    if bn * bk > 65536: continue
                    try:
                        y = w8a16_gemv(x, w8, scale, bn, bk, nw)
                    except Exception as e:
                        continue
                    err = (y.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-6)
                    if err > 2e-2: print("  bad err", bn, bk, nw, err); continue
                    torch.cuda.synchronize()
                    for _ in range(3): w8a16_gemv(x, w8, scale, bn, bk, nw)
                    torch.cuda.synchronize(); t0 = time.perf_counter(); iters = 50
                    for _ in range(iters): w8a16_gemv(x, w8, scale, bn, bk, nw)
                    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / iters
                    if best is None or dt < best[0]: best = (dt, bn, bk, nw, err)
        # bf16 reference timing (torch matmul, cuBLAS)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(50): torch.matmul(x, wb.t())
        torch.cuda.synchronize(); dt_bf16 = (time.perf_counter() - t0) / 50
        gb = N * K / 1e9
        print(f"M={M} K={K} N={N}: triton best {best[0]*1e6:7.1f}us (bn={best[1]} bk={best[2]} nw={best[3]} relerr={best[4]:.2e}) -> {gb/best[0]/1e3:.2f} TB/s eff (fp8 bytes) | bf16 cuBLAS {dt_bf16*1e6:7.1f}us -> {2*gb/dt_bf16/1e3:.2f} TB/s")
