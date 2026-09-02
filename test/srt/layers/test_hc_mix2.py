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
