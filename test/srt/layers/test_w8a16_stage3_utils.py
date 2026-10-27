# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Isolated source loading for CPU tests, avoiding quantization's CUDA imports."""

import ast
import importlib.util
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
QUANT = ROOT / "python/sglang/srt/layers/quantization"
FP8_SHAPES = [
    (16384, 2560), (13312, 2560), (2560, 6144), (1280, 2560),
    (2560, 640), (248320, 2560), (32768, 2560),
]
BF16_SHAPES = [(96, 2560), (512, 2560), (2560, 2560)]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Triton resolves globals through the defining module's name.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_kernels():
    g = load_source("_stage3_w8a16", QUANT / "w8a16_gemv.py")
    canonical = "sglang.srt.layers.quantization.w8a16_gemv"
    old = sys.modules.get(canonical)
    sys.modules[canonical] = g
    try:
        v = load_source("_stage3_nvfp4", QUANT / "w4a16_nvfp4_gemv.py")
    finally:
        if old is None:
            sys.modules.pop(canonical, None)
        else:
            sys.modules[canonical] = old
    return g, v


def env_block(path, name):
    """Execute the actual env assignment and its adjacent upper-bound check."""
    body = ast.parse(path.read_text()).body
    i = next(i for i, n in enumerate(body)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    nodes = [body[i]]
    if i + 1 < len(body) and isinstance(body[i + 1], ast.If):
        nodes.append(body[i + 1])
    ns = {"os": os, "_os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


def method(path, cls, name, namespace=None):
    tree = ast.parse(path.read_text())
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {"torch": torch}
    if namespace:
        ns.update(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


class CudaView:
    """CPU tensor metadata presented to CUDA-only predicates; never launches CUDA."""
    is_cuda = True

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def assert_bf16_close(got, ref):
    # Bound by two bf16 ulps at the reference tensor's peak, allowing
    # cancellation near zero without a meaningless relative-error explosion.
    peak = ref.abs().max().clamp_min(1e-30)
    ulp = 2.0 ** (torch.floor(torch.log2(peak)).item() - 7)
    assert torch.isfinite(got).all()
    assert (got.float() - ref).abs().max().item() <= 2 * ulp + 1e-6


def assert_one_ulp(got, ref):
    # Integer code distance handles sign and exponent boundaries exactly.
    def ordered(t):
        code = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(code < 0, -32768 - code, code)
    distance = (ordered(got) - ordered(ref)).abs()
    assert distance.max().item() <= 1
    return "equal" if torch.equal(got, ref) else "within 1 ulp"


def make_case(M, N, K, wb, device):
    torch.manual_seed(10103)
    x = (torch.randn(M, K, device=device) * 0.25).to(torch.bfloat16)
    dtype = torch.float8_e4m3fn if wb == 1 else torch.bfloat16
    w = (torch.randn(N, K, device=device) * 0.0625).to(dtype)
    scale = torch.rand(N, device=device, dtype=torch.float32) * 0.25 + 0.01
    z = torch.randn_like(x)
    nw = torch.rand(128, device=device, dtype=torch.float32) + 0.5
    return x, w, scale, z, nw


def run_case(g, case, kind, cfg=None, rows=None):
    x, w, scale, z, nw = case
    if rows is not None:
        x, z = x[:rows], z[:rows]
    M, K = x.shape
    N = w.shape[0]
    if kind == "norm":
        # Match the fused prologue's bf16 rounding before the matrix multiply.
        grouped = x.float().reshape(M, -1, 128)
        norm = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-5)
        norm = norm * nw * torch.nn.functional.silu(z.float().reshape(M, -1, 128))
        a = norm.reshape(M, K).to(torch.bfloat16)
        if cfg is None:
            got = g.w8a16_gemv_norm_gated(x, w, scale, z, nw, 128, 1e-5)
        else:
            assert cfg[1] == 128
            got = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
            g._launch(x, w, scale, got, M, N, K, True, cfg,
                      norm=(z, nw, 128, 1e-5, False))
    else:
        a = x
        if kind == "silu":
            got = g.w8a16_gemv_silu_mul(x, w, scale, cfg=cfg)
        elif kind == "out2":
            # Deliberately straddle a BLOCK_N boundary.
            split = N // 2 + 1
            out = torch.empty(M, split, device=x.device, dtype=torch.bfloat16)
            out2 = torch.empty(M, N - split, device=x.device, dtype=torch.bfloat16)
            fn = g.w8a16_gemv if w.dtype == torch.float8_e4m3fn else g.bf16_gemv
            args = (x, w, scale) if fn is g.w8a16_gemv else (x, w)
            lo, hi = fn(*args, cfg=cfg, out=out, out2=out2, split_n=split)
            got = torch.cat((lo, hi), dim=1)
        elif w.dtype == torch.float8_e4m3fn:
            got = g.w8a16_gemv(x, w, scale, cfg=cfg)
        else:
            got = g.bf16_gemv(x, w, cfg=cfg)
    ref = a.float() @ w.float().t()
    if w.dtype == torch.float8_e4m3fn:
        ref = ref * scale
    if kind == "silu":
        H = N // 2
        ref = torch.nn.functional.silu(ref[:, :H]) * ref[:, H:]
    assert_bf16_close(got, ref)
    return got
