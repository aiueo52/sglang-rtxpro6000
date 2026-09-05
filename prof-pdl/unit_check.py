"""Per-kernel unit check: run the patched kernels on fixed inputs and dump outputs.
Run once with SGLANG_TRITON_PDL=0 and once with =1, then compare the two dumps."""
import os, sys, torch
sys.path.insert(0, '/home/user/tools/sglang-pdl/python')
from sglang.kernels.triton_pdl import PDL
print("PDL =", PDL, "(SGLANG_TRITON_PDL=%s)" % os.environ.get("SGLANG_TRITON_PDL"))

from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv, bf16_gemv, w8a16_gemv_silu_mul
from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2, quantize_hc_mix2_weights_fp8
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)

out = {}
dev = 'cuda'
torch.manual_seed(1234)

# --- w8a16_gemv over the decode shapes (M in {1,4,16}) ---------------------
for M in (1, 4, 16):
    for (N, K) in ((4096, 2560), (1280, 2560), (2560, 2560), (512, 2560), (2560, 1280)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = (torch.randn(N, K, device=dev) / 8).to(torch.float8_e4m3fn)
        s = torch.rand(N, device=dev, dtype=torch.float32) + 0.5
        out[f"gemv/{M}x{N}x{K}"] = w8a16_gemv(x, w, s).float().cpu()
        wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / 8
        out[f"bf16gemv/{M}x{N}x{K}"] = bf16_gemv(x, wb).float().cpu()

# --- fused gate_up silu ---------------------------------------------------
for M in (1, 4, 16):
    K, H = 2560, 1280
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(2 * H, K, device=dev) / 8).to(torch.float8_e4m3fn)
    s = torch.rand(2 * H, device=dev, dtype=torch.float32) + 0.5
    out[f"silu/{M}"] = w8a16_gemv_silu_mul(x, w, s).float().cpu()

# --- hc_mix2 (bf16 and fp8 weights) ---------------------------------------
hc, hs, lowrank = 4, 2560, 320
for M in (1, 4, 16):
    xin = torch.randn(M, hc * hs, device=dev, dtype=torch.bfloat16)
    nw = torch.randn(hc * hs, device=dev, dtype=torch.bfloat16) / 8
    wd = torch.randn(lowrank, hc * hs, device=dev, dtype=torch.bfloat16) / 32
    wu = torch.randn(hc * hs, lowrank, device=dev, dtype=torch.bfloat16) / 32
    mixed, normed = hc_norm_mix2(xin, nw, 1e-6, wd, wu, hc, hs)
    out[f"mix2bf16/{M}/mixed"] = mixed.float().cpu()
    out[f"mix2bf16/{M}/normed"] = normed.float().cpu()
    qd, sd, qu, su = quantize_hc_mix2_weights_fp8(wd, wu)
    mixed8, normed8 = hc_norm_mix2(xin, nw, 1e-6, qd, qu, hc, hs, s_down=sd, s_up=su)
    out[f"mix2fp8/{M}/mixed"] = mixed8.float().cpu()
    out[f"mix2fp8/{M}/normed"] = normed8.float().cpu()

# --- fused qkvzba split ---------------------------------------------------
nq, nv, hq, hv = 16, 32, 128, 128
for T in (1, 4, 16):
    qkvz = torch.randn(T, nq * hq * 2 + nv * hv * 2, device=dev, dtype=torch.bfloat16)
    ba = torch.randn(T, nv * 2, device=dev, dtype=torch.bfloat16)
    mq, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, nq, nv, hq, hv)
    for n, t in (("mq", mq), ("z", z), ("b", b), ("a", a)):
        out[f"qkvzba/{T}/{n}"] = t.float().cpu()

torch.save(out, sys.argv[1])
print("wrote", sys.argv[1], len(out), "tensors")
