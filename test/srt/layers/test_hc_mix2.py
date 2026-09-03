import pytest
import torch

from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2
from sglang.srt.layers.hc_mix_triton import fused_hc_mix

HC = 4
HS = 2560
K = HC * HS
LR = 320
EPS = 1e-6

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_weights(seed: int):
    torch.manual_seed(seed)
    norm_w = torch.randn(K, device="cuda", dtype=torch.bfloat16) * 0.02
    w_down = torch.randn(LR, K, device="cuda", dtype=torch.bfloat16) * 0.02
    w_up = torch.randn(K, LR, device="cuda", dtype=torch.bfloat16) * 0.02
    return norm_w, w_down, w_up


def _reference(x, norm_w, w_down, w_up):
    normed = grouped_gemma_rmsnorm(x, norm_w, HS, EPS)
    return fused_hc_mix(normed, w_down, w_up, HC, HS), normed


def _assert_close(mixed, normed, ref_mixed, ref_normed):
    mixed_error = (mixed.float() - ref_mixed.float()).abs()
    assert mixed_error.max().item() <= 3e-2
    assert mixed_error.mean().item() <= 3e-3
    normed_error = (normed.float() - ref_normed.float()).abs()
    assert normed_error.max().item() <= 2e-2


@pytest.mark.parametrize("rows", [1, 5, 16])
def test_hc_mix2_matches_reference_chain(rows: int):
    norm_w, w_down, w_up = _make_weights(1234 + rows)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    ref_mixed, ref_normed = _reference(x, norm_w, w_down, w_up)
    mixed, normed = hc_norm_mix2(x, norm_w, EPS, w_down, w_up, HC, HS)
    _assert_close(mixed, normed, ref_mixed, ref_normed)


def test_hc_mix2_cuda_graph_replay():
    rows = 16
    calls = 3
    norm_w, w_down, w_up = _make_weights(99)
    static_x = [
        torch.randn(rows, K, device="cuda", dtype=torch.bfloat16) for _ in range(calls)
    ]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for x in static_x:
            hc_norm_mix2(x, norm_w, EPS, w_down, w_up, HC, HS)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [
            hc_norm_mix2(x, norm_w, EPS, w_down, w_up, HC, HS) for x in static_x
        ]

    for trial in range(2):
        for index, x in enumerate(static_x):
            x.copy_(
                torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
                * (1.0 + index + trial)
            )
        graph.replay()
        torch.cuda.synchronize()
        for index, (mixed, normed) in enumerate(outputs):
            ref_mixed, ref_normed = _reference(static_x[index], norm_w, w_down, w_up)
            _assert_close(mixed, normed, ref_mixed, ref_normed)


# --------------------------------------------------------------------------
# FP8 (e4m3) weight-only variant: SGLANG_HC_MIX2_FP8=1
# --------------------------------------------------------------------------


def _fp8_weights(w_down, w_up):
    from sglang.srt.layers.hc_mix2_triton import quantize_hc_mix2_weights_fp8

    return quantize_hc_mix2_weights_fp8(w_down, w_up)


def _dequant(w_fp8, scale):
    from sglang.srt.layers.hc_mix2_triton import dequantize_hc_mix2_weight

    return dequantize_hc_mix2_weight(w_fp8, scale)


def _err_ulps(actual, expected):
    """Error in bf16 ulps of the reference RMS.

    Per-element relative error is useless here: `mixed` is a gated mean whose
    outputs pass through zero, so a handful of near-zero elements dominate any
    ratio. The output is bf16, so one ulp of its RMS is the smallest difference
    that can exist at all -- measuring in those units says directly how many
    representable steps the fp8 weights moved the gate.
    """
    ulp = expected.float().pow(2).mean().sqrt() / 128.0
    return (actual.float() - expected.float()).abs() / ulp


def _stats(e):
    f = e.flatten()
    return {
        "max": f.max().item(),
        "mean": f.mean().item(),
        "p99": torch.quantile(f, 0.99).item(),
    }


@pytest.mark.parametrize("rows", [1, 5, 16])
def test_hc_mix2_fp8_matches_dequantized_bf16_kernel(rows: int):
    """The fp8 kernel must reproduce the bf16 kernel fed the SAME (dequantized)
    weights: any difference beyond fp32-accumulation noise is a kernel bug, not
    a quantization effect."""
    norm_w, w_down, w_up = _make_weights(4321 + rows)
    wd8, sd, wu8, su = _fp8_weights(w_down, w_up)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)

    ref_mixed, ref_normed = hc_norm_mix2(
        x, norm_w, EPS, _dequant(wd8, sd), _dequant(wu8, su), HC, HS
    )
    mixed, normed = hc_norm_mix2(
        x, norm_w, EPS, wd8, wu8, HC, HS, None, sd, su
    )
    # normed does not touch the mix weights at all.
    assert torch.equal(normed, ref_normed)
    # Not zero: the reference dequantizes to bf16 while the kernel keeps the
    # fp32 scale on the fp32 accumulator, so the kernel is the more accurate of
    # the two. Measured: >=93% of elements bit-identical, max 2 ulp of RMS.
    stats = _stats(_err_ulps(mixed, ref_mixed))
    assert stats["max"] <= 8.0, stats
    assert stats["mean"] <= 0.5, stats


@pytest.mark.parametrize("rows", [1, 5, 16])
def test_hc_mix2_fp8_vs_bf16_weight_reference(rows: int):
    """Error vs the bf16-weight reference chain, i.e. the quantization cost."""
    norm_w, w_down, w_up = _make_weights(777 + rows)
    wd8, sd, wu8, su = _fp8_weights(w_down, w_up)
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)

    ref_mixed, ref_normed = _reference(x, norm_w, w_down, w_up)
    mixed, normed = hc_norm_mix2(
        x, norm_w, EPS, wd8, wu8, HC, HS, None, sd, su
    )
    normed_error = (normed.float() - ref_normed.float()).abs()
    assert normed_error.max().item() <= 2e-2
    # Measured on this data: max 2.0, p99 1.0, mean 0.17 ulp of the output RMS.
    stats = _stats(_err_ulps(mixed, ref_mixed))
    assert stats["max"] <= 8.0, stats
    assert stats["p99"] <= 2.0, stats
    assert stats["mean"] <= 0.5, stats


def test_hc_mix2_fp8_cuda_graph_replay():
    rows = 16
    calls = 3
    norm_w, w_down, w_up = _make_weights(1357)
    wd8, sd, wu8, su = _fp8_weights(w_down, w_up)
    static_x = [
        torch.randn(rows, K, device="cuda", dtype=torch.bfloat16) for _ in range(calls)
    ]

    def run(x):
        return hc_norm_mix2(x, norm_w, EPS, wd8, wu8, HC, HS, None, sd, su)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for x in static_x:
            run(x)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [run(x) for x in static_x]

    for trial in range(2):
        for index, x in enumerate(static_x):
            x.copy_(
                torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
                * (1.0 + index + trial)
            )
        graph.replay()
        torch.cuda.synchronize()
        for index, (mixed, normed) in enumerate(outputs):
            ref_mixed, ref_normed = _reference(static_x[index], norm_w, w_down, w_up)
            assert (normed.float() - ref_normed.float()).abs().max().item() <= 2e-2
            assert _stats(_err_ulps(mixed, ref_mixed))["mean"] <= 0.5
