"""Numerics check for w4a16_nvfp4_gemv against a torch reference.

Two separate claims are checked, because they can fail independently:

1. **The quantiser and the kernel agree on the format.** `dequantize_nvfp4` (a plain
   torch LUT gather, written from `modelopt_quant.py:810-828`) is the reference decoder.
   The kernel's fp16 bit-trick decode must reproduce it *bit-exactly* for all 16 codes at
   every block scale -- this is a discrete map, so anything but exactness is a bug, not a
   tolerance question.
2. **The GEMV matches a reference matmul on the dequantised weight.** Here the fp32
   accumulation order differs (tiled/split-K vs cuBLAS), so the comparison is a relative
   error against an fp64 reference, reported alongside the same figure for the FP8 kernel
   at the same shape -- the FP8 number is the bar, since that path is what ships today.
"""

import os
import sys

import torch

sys.path.insert(0, "/home/user/tools/sglang-n1/python")

from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (  # noqa: E402
    _E2M1_LEVELS,
    dequantize_nvfp4,
    quantize_nvfp4,
    w4a16_nvfp4_gemv,
)
from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv  # noqa: E402

DEV = "cuda"
torch.manual_seed(0)


def check_code_coverage():
    """Every one of the 16 codes must decode exactly, at a spread of block scales."""
    # Build a weight that forces all 8 magnitudes x both signs into one 16-block.
    lvl = torch.tensor(_E2M1_LEVELS, dtype=torch.float32)
    row = torch.cat([lvl, -lvl])  # 16 values, amax 6 -> block scale lands on 1
    scales = [1.0, 0.5, 3.0, 1e-3, 17.0]
    w = torch.stack([row * s for s in scales]).to(DEV).bfloat16()
    wq, bs, gs = quantize_nvfp4(w)
    deq = dequantize_nvfp4(wq, bs, gs)
    # exact by construction: each row's values are exactly representable
    rel = ((deq - w.float()).abs().max() / w.float().abs().max()).item()
    codes_seen = set(int(c) for c in torch.cat([wq & 0x0F, wq >> 4]).flatten().tolist())
    return rel, sorted(codes_seen)


def check_kernel_vs_reference(M, N, K, per_channel=False):
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.02).bfloat16()

    wq, bs, gs = quantize_nvfp4(w)
    deq = dequantize_nvfp4(wq, bs, gs)  # fp32, the exact value the kernel must use

    y = w4a16_nvfp4_gemv(x, wq, bs, gs)
    # Reference: same dequantised weight, fp64 accumulation.
    ref = (x.double() @ deq.double().t())
    err = ((y.float().double() - ref).abs() / ref.abs().clamp(min=1e-9)).max().item()
    rms = ((y.float().double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()

    # Same figure for the FP8 kernel, as the bar.
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    s8 = amax / 448.0
    w8 = (w.float() / s8).clamp(-448, 448).to(torch.float8_e4m3fn)
    y8 = w8a16_gemv(x, w8, s8.reshape(-1))
    ref8 = (x.double() @ (w8.float().double() * s8.double()).t())
    err8 = ((y8.float().double() - ref8).abs() / ref8.abs().clamp(min=1e-9)).max().item()
    rms8 = ((y8.float().double() - ref8).pow(2).mean().sqrt() / ref8.pow(2).mean().sqrt()).item()

    # And the quantisation error itself (nvfp4 vs fp8 vs bf16), the quality-relevant number.
    qerr4 = ((deq - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()).item()
    qerr8 = (((w8.float() * s8) - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()).item()
    return dict(kernel_max_rel=err, kernel_rms=rms, fp8_kernel_max_rel=err8,
                fp8_kernel_rms=rms8, weight_rms_nvfp4=qerr4, weight_rms_fp8=qerr8)


def check_bitexact_decode(N=256, K=2560):
    """The kernel's decode must equal the reference decode bit-for-bit.

    Isolated from the dot by using a one-hot activation: y[0, n] = deq[n, k0] exactly
    (a single product, no accumulation, no reassociation), for every k0 in a stride
    that covers all 16 lanes of a block and several blocks.
    """
    w = (torch.randn(N, K, device=DEV, dtype=torch.float32) * 0.05).bfloat16()
    wq, bs, gs = quantize_nvfp4(w)
    deq = dequantize_nvfp4(wq, bs, gs)
    bad = 0
    tested = 0
    for k0 in list(range(0, 16)) + [16, 17, 31, 100, 1279, 2559]:
        x = torch.zeros(1, K, device=DEV, dtype=torch.bfloat16)
        x[0, k0] = 1.0
        y = w4a16_nvfp4_gemv(x, wq, bs, gs)[0].float()
        want = deq[:, k0].bfloat16().float()  # kernel rounds the fp32 acc to bf16
        bad += int((y != want).sum())
        tested += N
    return bad, tested


if __name__ == "__main__":
    print(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)}")
    rel, codes = check_code_coverage()
    print(f"[codes] all-16-code round trip: max rel err {rel:.3e}, codes seen {codes}")
    assert codes == list(range(16)), "not all 16 e2m1 codes exercised"
    assert rel == 0.0, f"exactly-representable weights must round-trip exactly, got {rel}"

    bad, tested = check_bitexact_decode()
    print(f"[decode] one-hot bit-exactness vs reference dequant: {bad}/{tested} mismatched")

    for (M, N, K) in [(1, 49152, 2560), (4, 49152, 2560), (16, 49152, 2560),
                      (1, 2560, 2560), (16, 13312, 2560)]:
        r = check_kernel_vs_reference(M, N, K)
        print(f"[gemv] M={M:2d} N={N:6d} K={K}: "
              f"nvfp4 kernel rms {r['kernel_rms']:.2e} (fp8 {r['fp8_kernel_rms']:.2e}) | "
              f"weight rms nvfp4 {r['weight_rms_nvfp4']:.4f} fp8 {r['weight_rms_fp8']:.4f}")
    print("OK" if bad == 0 else f"DECODE MISMATCH: {bad}")
