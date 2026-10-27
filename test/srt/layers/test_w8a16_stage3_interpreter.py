# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""CPU Triton interpreter coverage. Run with TRITON_INTERPRET=1 and PDL off."""

import os

import pytest
import torch
import triton
import triton.language as tl

from test_w8a16_stage3_utils import assert_one_ulp, load_kernels, make_case, run_case

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="requires TRITON_INTERPRET=1")
g, v = load_kernels()


@triton.jit
def _bf16_dot_probe(a, out):
    i = tl.arange(0, 16)
    x = tl.load(a + i[:, None] * 16 + i[None, :])
    result = tl.dot(x, x, out_dtype=tl.float32)
    tl.store(out + i[:, None] * 16 + i[None, :], result)


@pytest.fixture(scope="module")
def bf16_dot_support():
    # Triton stores bf16 as NumPy uint16, but some releases' create_dot
    # convert only fp8 operands. Probe this independently of the GEMV code
    # so a kernel regression cannot silently turn into an interpreter skip.
    a = torch.ones(16, 16, dtype=torch.bfloat16)
    out = torch.empty(16, 16, dtype=torch.float32)
    _bf16_dot_probe[(1,)](a, out)
    if not torch.equal(out, torch.full_like(out, 16)):
        reason = (
            f"Triton {triton.__version__} interpreter bf16 dot is unsupported: "
            f"ones[16,16] @ ones yields {out[0,0].item()} instead of 16.0; "
            "create_dot multiplies raw uint16 bf16 encodings. "
            "These plain/out2/norm/SiLU and row-parity checks are GPU-only."
        )
        print(reason)
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch, bf16_dot_support):
    monkeypatch.setattr(g, "W8A16_GEMV_MAX_M", 32)
    monkeypatch.setattr(g, "PDL", False)
    monkeypatch.setattr(g, "_num_sms", lambda device: 188)
    monkeypatch.setattr(g, "BA_SPLIT_GRID", False)


@pytest.mark.parametrize("M", [17, 24, 32])
@pytest.mark.parametrize("kind", ["plain", "out2", "norm", "silu"])
@pytest.mark.parametrize("splits", [1, 2])
def test_interpreter_reference(M, kind, splits):
    case = make_case(M, 256, 512, 1, "cpu")
    cfg = (32, 128, splits, True, None, 4, 2)
    got = run_case(g, case, kind, cfg=cfg)
    assert got.dtype == torch.bfloat16
    assert not g._workspace(torch.device("cpu"))[1].any()
    # Replay checks the self-resetting split-K counters.
    again = run_case(g, case, kind, cfg=cfg)
    torch.testing.assert_close(got, again, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["plain", "out2", "norm", "silu"])
def test_interpreter_first16(kind):
    case = make_case(32, 256, 512, 1, "cpu")
    cfg = (32, 128, 1, True, None, 4, 2)
    got = run_case(g, case, kind, cfg=cfg)
    old = run_case(g, case, kind, cfg=cfg, rows=16)
    result = assert_one_ulp(got[:16], old)
    print(f"{kind}: M32 rows 0-15 vs M16: {result}")
