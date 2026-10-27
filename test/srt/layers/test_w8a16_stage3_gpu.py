# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Stage 3A real-shape tests with both knobs=32."""

import pytest
import torch

from test_w8a16_stage3_utils import (
    BF16_SHAPES, FP8_SHAPES, assert_bf16_close, assert_one_ulp,
    load_kernels, make_case, run_case,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
g, v = load_kernels()
CASES = [(1, n, k, kind) for n, k in FP8_SHAPES
         for kind in ("plain", "out2", "norm", "silu")]
CASES += [(2, n, k, kind) for n, k in BF16_SHAPES for kind in ("plain", "out2")]


@pytest.fixture(autouse=True)
def opt_in(monkeypatch):
    monkeypatch.setattr(g, "W8A16_GEMV_MAX_M", 32)
    monkeypatch.setattr(v, "NVFP4_GEMV_MAX_M", 32)


# These are also the GPU counterparts of checks blocked by the interpreter's
# bf16 dot limitation; hold tile/split order fixed to isolate M_PAD.
@pytest.mark.parametrize("M", [17, 24, 32])
@pytest.mark.parametrize("kind", ["plain", "out2", "norm", "silu"])
@pytest.mark.parametrize("splits", [1, 2])
def test_small_shape_reference(M, kind, splits):
    case = make_case(M, 256, 512, 1, "cuda")
    cfg = (32, 128, splits, True, None, 4, 2)
    got = run_case(g, case, kind, cfg=cfg)
    again = run_case(g, case, kind, cfg=cfg)
    torch.testing.assert_close(got, again, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["plain", "out2", "norm", "silu"])
def test_small_shape_first16(kind):
    case = make_case(32, 256, 512, 1, "cuda")
    cfg = (32, 128, 1, True, None, 4, 2)
    got = run_case(g, case, kind, cfg=cfg)
    old = run_case(g, case, kind, cfg=cfg, rows=16)
    print(f"{kind}: {assert_one_ulp(got[:16], old)}")


@pytest.mark.parametrize("M", [17, 24, 32])
@pytest.mark.parametrize("wb,N,K,kind", CASES)
def test_real_shape_reference(M, wb, N, K, kind):
    case = make_case(M, N, K, wb, "cuda")
    run_case(g, case, kind)


@pytest.mark.parametrize("wb,N,K,kind", CASES)
def test_real_shape_first16(wb, N, K, kind):
    case = make_case(32, N, K, wb, "cuda")
    # Hold tile/split order fixed to isolate the change in row padding.
    plan = (g._fused_plan(32, N, K, 188) if kind == "silu"
            else g._norm_plan(32, N, K, 128, wb, 188) if kind == "norm"
            else g._plan(32, N, K, False, wb, 188))
    got = run_case(g, case, kind, cfg=plan)
    old = run_case(g, case, kind, cfg=plan, rows=16)
    print(f"{wb,N,K,kind}: {assert_one_ulp(got[:16], old)}")


@pytest.mark.parametrize("M", [17, 32])
@pytest.mark.parametrize("N", [248320, 32768, 49152])
def test_nvfp4_head_vs_slice_loop(M, N):
    K = 2560
    torch.manual_seed(10104)
    weight = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
    wq, bs, gs = v.quantize_nvfp4(weight)
    del weight
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    got = v.w4a16_nvfp4_gemv(x, wq, bs, gs)
    ref = torch.cat([v.w4a16_nvfp4_gemv(x[m:m+16], wq, bs, gs)
                     for m in range(0, M, 16)])
    assert_bf16_close(got, ref.float())
    # Match the 32-row tile to measure row-padding parity independently of tuning.
    cfg = v._plan(32, N, K, 188)
    same_tile = torch.cat([v.w4a16_nvfp4_gemv(x[m:m+16], wq, bs, gs, cfg=cfg)
                          for m in range(0, M, 16)])
    assert_one_ulp(got, same_tile)


@pytest.mark.parametrize("kind", ["plain", "out2", "norm", "silu", "nvfp4"])
def test_cuda_graph_replay(kind):
    case = make_case(32, 1280, 2560, 1, "cuda")
    g.prealloc(torch.device("cuda"))
    if kind == "nvfp4":
        packed = v.quantize_nvfp4(case[1].to(torch.bfloat16))
        call = lambda: v.w4a16_nvfp4_gemv(case[0], *packed)
    else:
        # References and assertions must stay outside capture.
        x, w, s, z, nw = case
        if kind == "norm":
            call = lambda: g.w8a16_gemv_norm_gated(x, w, s, z, nw, 128, 1e-5)
        elif kind == "silu":
            call = lambda: g.w8a16_gemv_silu_mul(x, w, s)
        elif kind == "out2":
            lo = torch.empty(32, 641, device="cuda", dtype=torch.bfloat16)
            hi = torch.empty(32, 639, device="cuda", dtype=torch.bfloat16)
            call = lambda: g.w8a16_gemv(x, w, s, out=lo, out2=hi, split_n=641)
        else:
            call = lambda: g.w8a16_gemv(x, w, s)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for trial in range(3):
        case[0].mul_(0.9)
        graph.replay()
        saved = tuple(t.clone() for t in
                      (captured if isinstance(captured, tuple) else (captured,)))
        expected = call()
        for got, ref in zip(saved,
                            expected if isinstance(expected, tuple) else (expected,)):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
