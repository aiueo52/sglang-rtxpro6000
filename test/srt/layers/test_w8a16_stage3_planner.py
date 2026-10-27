# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Stage 3A CPU planner, golden compatibility, knobs, and launch constexprs."""

import sys
from types import SimpleNamespace

import pytest
import torch
import triton

from sglang.srt.model_executor.forward_context import (
    ForwardContext, forward_context, get_gemv_m_limit,
)

from test_w8a16_stage3_golden import GOLDEN
from test_w8a16_stage3_utils import (
    BF16_SHAPES, FP8_SHAPES, QUANT, ROOT, CudaView, env_block, load_kernels, load_source, method,
)

g, v = load_kernels()
SHAPES = [(1, n, k) for n, k in FP8_SHAPES] + [(2, n, k) for n, k in BF16_SHAPES]


@pytest.fixture(autouse=True)
def decode_phase():
    mode = SimpleNamespace(is_decode=lambda: True, is_target_verify=lambda: False)
    with forward_context(ForwardContext(attn_backend=None, forward_mode=mode)):
        yield


def plans(M, wb, N, K):
    yield "plain", g._plan(M, N, K, False, wb, 188), N, 1
    yield "norm", g._norm_plan(M, N, K, 128, wb, 188), N, 1
    if wb == 1:
        yield "silu", g._fused_plan(M, N, K, 188), N // 2, 2


@pytest.mark.parametrize("M", range(1, 33))
@pytest.mark.parametrize("wb,N,K", SHAPES)
def test_plans_and_golden(M, wb, N, K):
    for kind, cfg, columns, factor in plans(M, wb, N, K):
        bn, bk, splits, dot, _, warps, stages = cfg
        mp = g._m_pad(M, dot)
        assert mp == (32 if M > 16 else (16 if M > 1 or dot else 1))
        nb = triton.cdiv(columns, bn)
        assert nb <= g._WS_COUNTERS
        # SPLITS=1 never reads or writes partial slots or counters. The old
        # 248320-row head also exceeds _WS_FLOATS if counted as a partial.
        workspace_floats = factor * nb * splits * mp * bn if splits > 1 else 0
        assert workspace_floats <= g._WS_FLOATS
        if M <= 16:
            assert (mp, cfg) == GOLDEN[(kind, wb, N, K, M)]


@pytest.mark.parametrize("M", range(1, 33))
@pytest.mark.parametrize("N", [248320, 32768, 49152])
def test_nvfp4_plans(M, N):
    cfg = v._plan(M, N, 2560, 188)
    bn, bk, sp, dot, warps, stages = cfg
    mp = g._m_pad(M, dot)
    nb = triton.cdiv(N, bn)
    assert (nb if sp > 1 else 0) <= v._WS_COUNTERS
    assert (nb * sp * mp * bn if sp > 1 else 0) <= v._WS_FLOATS
    if M <= 16:
        assert (mp, cfg) == GOLDEN[("nvfp4", 1, N, 2560, M)]


M32_TUNED = {
    (1, False, 13312, 2560): (64, 128, 1, True, None, 4, 3),
    (1, False, 16384, 2560): (128, 256, 1, True, None, 8, 3),
    (1, False, 32768, 2560): (128, 256, 1, True, None, 4, 2),
    (1, False, 2560, 6144): (64, 128, 4, True, None, 4, 3),
    (1, False, 2560, 640): (16, 128, 1, True, None, 4, 3),
    (1, False, 1280, 2560): (16, 128, 4, True, None, 4, 3),
}


def test_bucket32_is_independent():
    assert g._BY_SHAPE_M32 == M32_TUNED
    assert {key: cfg for key, cfg in g._TUNED.items() if key[2] == 32} == {
        (wb, cn, 32, N, K): cfg for (wb, cn, N, K), cfg in M32_TUNED.items()}
    # The G1 tiles are used verbatim at every 17..32-row M (no _fit downgrade).
    for M in (17, 24, 32):
        for (wb, cn, N, K), cfg in M32_TUNED.items():
            assert g._plan(M, N, K, cn, wb, 188) == cfg
        assert v._plan(M, 248320, 2560, 188) == (128, 256, 1, True, 8, 2)
        assert v._plan(M, 49152, 2560, 188) == (64, 256, 1, True, 4, 2)
    assert not any(key[0] == 32 for key in g._FUSED_NORM_BY_SHAPE)
    assert [g._m_bucket(m) for m in (1, 4, 5, 16, 17, 32)] == [1, 4, 16, 16, 32, 32]


@pytest.mark.parametrize("M", [17, 24, 32])
def test_fits_and_smem(M):
    # A config fitting 16 rows must be rejected when 32-row partials overflow.
    cfg = (32, 128, 16, True, None, 4, 3)
    assert g._fit(16, 8192, cfg) == cfg
    assert g._fit(M, 8192, cfg)[2] == 1
    assert g._fit_fused(M, 4096, cfg)[2] == 1
    assert v._fit(M, 8192, (32, 128, 16, True, 4, 3))[2] == 1
    assert g._fit(M, 32, cfg, n_blocks=g._WS_COUNTERS + 1)[2] == 1
    assert g._fit_fused(M, (g._WS_COUNTERS + 1) * 32, cfg)[2] == 1
    assert v._fit(M, (v._WS_COUNTERS + 1) * 32, (32, 128, 16, True, 4, 3))[2] == 1
    # The staged-bf16 bound guards untuned fallbacks. A shape in the bucket-32 table
    # was compiled and error-checked on the GPU by the G1 sweep; the norm/silu plans
    # derived from its big-N tiles are never launched (only out_proj takes the norm
    # kernel and only shared gate_up the silu one), so those two are checked below.
    for wb, N, K in SHAPES:
        tuned = (wb, False, 32, N, K) in g._TUNED
        for kind, plan, _, factor in plans(M, wb, N, K):
            if tuned:
                continue
            bn, bk, sp, _, _, warps, stages = plan
            staged_bf16 = stages * (32 + factor * bn) * bk * 2
            assert staged_bf16 < 99 * 1024
        for contig in (False, True):
            if not contig and tuned:
                continue
            plan = g._plan(M, N, K, contig, wb, 188)
            assert plan[-1] * (32 + plan[0]) * plan[1] * 2 < 99 * 1024
    for kind, plan, _, factor in plans(M, 1, 2560, 6144):
        if kind == "norm":
            assert plan[-1] * (32 + factor * plan[0]) * plan[1] * 2 < 99 * 1024
    for kind, plan, _, factor in plans(M, 1, 1280, 2560):
        if kind == "silu":
            assert plan[-1] * (32 + factor * plan[0]) * plan[1] * 2 < 99 * 1024


@pytest.mark.parametrize("limit", [8, 16, 17, 24, 32, 33])
def test_knob_parsing_and_gates(monkeypatch, limit):
    monkeypatch.setenv("SGLANG_FP8_W8A16_GEMV_MAX_M", str(limit))
    monkeypatch.setenv("SGLANG_NVFP4_GEMV_MAX_M", str(limit))
    settings = [
        (QUANT / "fp8.py", "_W8A16_GEMV_MAX_M", "SGLANG_FP8_W8A16_GEMV_MAX_M"),
        (ROOT / "python/sglang/srt/models/qwen2_moe.py",
         "_SHARED_GATEUP_FUSED_MAX_M", "SGLANG_FP8_W8A16_GEMV_MAX_M"),
        (QUANT / "w8a16_gemv.py", "W8A16_GEMV_MAX_M", "SGLANG_FP8_W8A16_GEMV_MAX_M"),
        (QUANT / "w4a16_nvfp4_gemv.py", "NVFP4_GEMV_MAX_M", "SGLANG_NVFP4_GEMV_MAX_M"),
    ]
    for path, name, env in settings:
        if limit > 32:
            with pytest.raises(ValueError, match=env + " must be <= 32"):
                env_block(path, name)
        else:
            assert env_block(path, name) == limit
    if limit > 32:
        return
    gg, vv = load_kernels()
    fp_gate = method(QUANT / "fp8.py", "Fp8LinearMethod", "_w8a16_gemv_ok",
                     {"_W8A16_GEMV_ENABLED": True, "_W8A16_GEMV_MAX_M": limit})
    nv_gate = method(QUANT / "nvfp4_dense.py", "NVFP4DenseLinearMethod", "_ok")
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization.w4a16_nvfp4_gemv", vv)
    monkeypatch.setattr(gg, "NORM_INTO_GEMV", True)
    layer = SimpleNamespace(weight=torch.empty(512, 256, dtype=torch.float8_e4m3fn),
                            weight_scale=torch.ones(256), nvfp4_wq=torch.empty(1))
    linear = SimpleNamespace(use_marlin=False, block_quant=False, use_mxfp8=False)
    w = torch.empty(256, 512, dtype=torch.float8_e4m3fn)
    s = torch.ones(256)
    for M in (8, 16, 17, 24, 32, 33):
        x = CudaView(torch.empty(M, 512, dtype=torch.bfloat16))
        expect = M <= limit
        assert fp_gate(linear, layer, x) == expect
        assert nv_gate(layer, x) == expect
        # The kernels themselves keep accepting 16 rows below a knob of 16; the
        # callers' own gates (fp8, nvfp4, the qwen2_moe max_m) still apply the knob.
        kernel = M <= max(16, limit)
        assert gg.w8a16_gemv_silu_mul_supported(x, w, s) == kernel
        assert gg.w8a16_gemv_silu_mul_supported(x, w, s, limit) == expect
        assert gg.w8a16_gemv_norm_gated_supported(x, w, s, x, torch.ones(128), 128) == kernel


@pytest.mark.parametrize("M", [1, 16, 17, 24, 32])
def test_launch_m_pad(monkeypatch, M):
    monkeypatch.setattr(g, "W8A16_GEMV_MAX_M", 32)
    monkeypatch.setattr(v, "NVFP4_GEMV_MAX_M", 32)
    monkeypatch.setattr(g, "_num_sms", lambda device: 188)
    monkeypatch.setattr(v, "_num_sms", lambda device: 188)
    seen = []
    class Recorder:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: seen.append((grid, kwargs))
    monkeypatch.setattr(g, "_w8a16_gemv_kernel", Recorder())
    monkeypatch.setattr(g, "_w8a16_gemv_silu_kernel", Recorder())
    monkeypatch.setattr(v, "_w4a16_nvfp4_gemv_kernel", Recorder())
    x = torch.empty(M, 512, dtype=torch.bfloat16)
    w = torch.empty(256, 512, dtype=torch.float8_e4m3fn)
    s = torch.ones(256)
    # Includes segmented out2 (which rechecks the scratch product in _launch).
    monkeypatch.setattr(g, "BA_SPLIT_GRID", True)
    g.w8a16_gemv(x, w, s, out=torch.empty(M, 97, dtype=torch.bfloat16),
                  out2=torch.empty(M, 159, dtype=torch.bfloat16), split_n=97)
    g.w8a16_gemv_norm_gated(x, w, s, x, torch.ones(128), 128, 1e-5)
    g.w8a16_gemv_silu_mul(x, w, s)
    v.w4a16_nvfp4_gemv(x, torch.zeros(256, 256, dtype=torch.uint8),
                       torch.zeros(256, 32, dtype=torch.uint8), torch.ones(1))
    for grid, kw in seen:
        assert kw["M_PAD"] == g._m_pad(M, kw["USE_DOT"])
    assert len(seen) == 4


def test_nvfp4_slice_dispatch(monkeypatch):
    vv = SimpleNamespace(NVFP4_GEMV_MAX_M=32)
    vv._m_limit = lambda: get_gemv_m_limit(vv.NVFP4_GEMV_MAX_M)
    calls = []
    vv.w4a16_nvfp4_gemv = lambda x, *args, out: calls.append(x.shape[0])
    vv.dequantize_nvfp4 = lambda *args: pytest.fail("unexpected large-M dispatch")
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization.w4a16_nvfp4_gemv", vv)
    wide = method(QUANT / "nvfp4_dense.py", "NVFP4DenseLinearMethod", "_wide_m",
                  {"_CHUNK_M_MAX": 256, "_FALLBACK_ROW_CHUNK": 16384})
    class View(CudaView):
        def __getitem__(self, s):
            return View(self.tensor[s])
    layer = SimpleNamespace(nvfp4_wq=torch.empty(256, 256),
                            nvfp4_bs=torch.empty(256, 32), nvfp4_gscale=torch.ones(1))
    for limit, M, expected in [(16, 33, [16, 16, 1]), (32, 65, [32, 32, 1]),
                                (24, 49, [24, 24, 1]), (8, 17, [8, 8, 1])]:
        vv.NVFP4_GEMV_MAX_M = limit
        calls.clear()
        wide(None, layer, View(torch.empty(M, 512, dtype=torch.bfloat16)))
        assert calls == expected


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
def test_invalid_limit_fails_at_kernel_import(monkeypatch, kind):
    monkeypatch.setenv("SGLANG_FP8_W8A16_GEMV_MAX_M", "16")
    monkeypatch.setenv("SGLANG_NVFP4_GEMV_MAX_M", "16")
    if kind == "fp8":
        monkeypatch.setenv("SGLANG_FP8_W8A16_GEMV_MAX_M", "33")
        with pytest.raises(ValueError, match="SGLANG_FP8_W8A16_GEMV_MAX_M"):
            load_source("_stage3_invalid_fp8", QUANT / "w8a16_gemv.py")
    else:
        monkeypatch.setenv("SGLANG_NVFP4_GEMV_MAX_M", "33")
        monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization.w8a16_gemv", g)
        with pytest.raises(ValueError, match="SGLANG_NVFP4_GEMV_MAX_M"):
            load_source("_stage3_invalid_nvfp4", QUANT / "w4a16_nvfp4_gemv.py")


def test_prealloc_covers_all_slots():
    device = torch.device("cpu")
    g.prealloc(device)
    for slot in range(g._N_SLOTS):
        ws, cnt = g._workspace_slot(device, slot)
        assert ws.numel() == g._WS_FLOATS
        assert cnt.numel() == g._WS_COUNTERS
        assert not cnt.any()
    assert g._ONES[device].item() == 1
