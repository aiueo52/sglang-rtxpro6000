"""Fused shared-expert gate_up GEMV + SiLU-and-mul (SGLANG_SHARED_GATEUP_FUSED=1)."""

import pytest
import torch

from sglang.srt.layers.quantization.w8a16_gemv import (
    w8a16_gemv,
    w8a16_gemv_silu_mul,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# The flash-next shared expert: gate_up is [2 * 640, 2560] fp8, per-channel scale.
SHAPES = [(1280, 2560), (2048, 4096)]


def _fp8_weight(N: int, K: int, seed: int):
    torch.manual_seed(seed)
    w = torch.randn(N, K, device="cuda", dtype=torch.float32) * 0.02
    scale = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (w / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.contiguous().float()


def _two_kernel_reference(x, w, scale):
    """What the unfused path computes: bf16 gate_up, then act_and_mul in fp32."""
    gate_up = w8a16_gemv(x, w, scale)
    h = gate_up.shape[1] // 2
    g = gate_up[:, :h].float()
    u = gate_up[:, h:].float()
    return (torch.nn.functional.silu(g) * u).to(torch.bfloat16)


def _fp32_golden(x, w, scale):
    """Exact-ish reference: fp32 GEMM on the dequantized weights, fp32 activation."""
    wf = w.float() * scale[:, None]
    y = x.float() @ wf.t()
    h = y.shape[1] // 2
    return (torch.nn.functional.silu(y[:, :h]) * y[:, h:]).to(torch.bfloat16)


def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Difference in units of the last place of `b` (bf16 has 8 mantissa bits)."""
    scale = b.float().abs().clamp(min=1e-30)
    exp = torch.floor(torch.log2(scale))
    ulp = torch.pow(2.0, exp - 7.0)
    return (a.float() - b.float()).abs() / ulp


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", [1, 4, 16])
def test_fused_matches_two_kernel_path(M: int, N: int, K: int):
    w, scale = _fp8_weight(N, K, 20260903 + M + N)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    fused = w8a16_gemv_silu_mul(x, w, scale)
    ref = _two_kernel_reference(x, w, scale)
    golden = _fp32_golden(x, w, scale)

    assert fused.shape == (M, N // 2)
    assert fused.dtype == torch.bfloat16

    # The fused kernel keeps gate/up in fp32 through the activation, so it is at
    # least as close to the fp32 golden as the two-kernel path is.
    err_fused = (fused.float() - golden.float()).abs()
    err_ref = (ref.float() - golden.float()).abs()
    assert err_fused.max().item() <= err_ref.max().item() * 1.5 + 1e-6, (
        err_fused.max().item(),
        err_ref.max().item(),
    )
    # ... and it stays within a couple of bf16 ulps of the two-kernel result.
    # The two paths differ only in that the unfused one rounds gate and up to
    # bf16 before the activation, so ~1/3 of outputs land one ulp apart.
    ulps = _ulp_diff(fused, ref)
    assert ulps.mean().item() <= 1.0, ulps.mean().item()
    assert ulps.max().item() <= 4.0, ulps.max().item()


@pytest.mark.parametrize("M", [1, 16])
def test_fused_cuda_graph_replay(M: int):
    N, K = 1280, 2560
    calls = 3
    w, scale = _fp8_weight(N, K, 4242)
    static_x = [
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16) for _ in range(calls)
    ]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for x in static_x:
            w8a16_gemv_silu_mul(x, w, scale)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [w8a16_gemv_silu_mul(x, w, scale) for x in static_x]

    for trial in range(3):
        for i, x in enumerate(static_x):
            x.copy_(
                torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * (1.0 + i + trial)
            )
        graph.replay()
        torch.cuda.synchronize()
        for i, out in enumerate(outs):
            ref = _two_kernel_reference(static_x[i], w, scale)
            assert _ulp_diff(out, ref).mean().item() <= 1.0


@pytest.mark.parametrize("M", [1, 4, 16])
def test_fused_split_k_configs_agree(M: int):
    """Every split count must reproduce the SPLITS=1 result (the fixup is in-launch)."""
    N, K = 1280, 2560
    w, scale = _fp8_weight(N, K, 909)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    use_dot = M > 1
    base = w8a16_gemv_silu_mul(x, w, scale, cfg=(16, 128, 1, use_dot, None, 4, 3))
    for splits in (2, 5, 10, 20):
        got = w8a16_gemv_silu_mul(x, w, scale, cfg=(16, 128, splits, use_dot, None, 4, 3))
        rel = (got.float() - base.float()).abs() / base.float().abs().clamp(min=1e-4)
        assert rel.max().item() <= 2e-2, (splits, rel.max().item())
