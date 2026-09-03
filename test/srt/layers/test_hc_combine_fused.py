"""One-launch HyperConnection combine vs. the sgl-kernel gate+apply pair.

Reference: ``sglang.kernels.ops.elementwise.hc_combine.hc_combine_split`` (the two
kernels ``combine()`` launches today) and ``hc_combine`` (the single-CTA-per-row
kernel it is bit-identical to).

The fused kernel reduces the fp32 gate dot in a different order, so the contract is
"within 1 bf16 ulp" rather than bit-identical; this test measures the actual gap and
also checks the result against an fp64 reference so a systematic error cannot hide
inside the 1-ulp allowance.

Run:  python test/srt/layers/test_hc_combine_fused.py
"""

from __future__ import annotations

import unittest

import torch

HC = 4
HS = 2560
ROW = HC * HS


def _mono(x: torch.Tensor) -> torch.Tensor:
    """bf16 bit patterns mapped to a monotonically increasing integer key."""
    u = x.view(torch.int16).to(torch.int64) & 0xFFFF
    return torch.where(u >= 0x8000, 0x10000 - u, u + 0x8000)


def bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_mono(a) - _mono(b)).abs()


def ref_fp64(y, r, n, w, hc=HC, hs=HS):
    """The documented semantics, in fp64."""
    a = 2.0 / (1.0 + torch.exp(-(n.double() @ w.double().T) / hc))  # [M, HC]
    out = r.double().unflatten(-1, (hc, hs)) + a[:, :, None] * y.double()[:, None, :]
    return out.flatten(1)


def make(rows: int, seed: int = 0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    y = torch.randn(rows, HS, dtype=torch.bfloat16, device=device, generator=g)
    r = torch.randn(rows, ROW, dtype=torch.bfloat16, device=device, generator=g)
    n = torch.randn(rows, ROW, dtype=torch.bfloat16, device=device, generator=g)
    # inject_weight rows are small so the gate lands in sigmoid's steep region,
    # where the gate error has the best chance of moving a bf16 output bit.
    w = (
        torch.randn(HC, ROW, dtype=torch.float32, device=device, generator=g)
        .mul_(0.02)
        .to(torch.bfloat16)
    )
    return y, r, n, w


class TestHcCombineFusedPlan(unittest.TestCase):
    """CPU-side: the grid-barrier chunking rule."""

    def test_plan_fits_the_sm_count(self):
        from sglang.srt.layers.hc_combine_fused_triton import _plan

        for rows in range(1, 33):
            plan = _plan(rows, ROW, 188)
            if plan is None:
                continue
            nprog, blk, warps = plan
            self.assertEqual(nprog * blk, ROW)
            self.assertEqual(blk & (blk - 1), 0, "chunk must be a power of two")
            self.assertGreaterEqual(blk, 256)
            self.assertLessEqual(rows * nprog, 188, "grid barrier would not be resident")
            self.assertEqual(blk // (32 * warps), 8, "want 8 elements per thread")

    def test_plan_rejects_grids_that_cannot_barrier(self):
        from sglang.srt.layers.hc_combine_fused_triton import _plan

        # 5 chunks is the coarsest option, so a machine with < 5*rows SMs has none.
        self.assertIsNone(_plan(rows=16, row_size=ROW, sms=8))


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestHcCombineFused(unittest.TestCase):
    def _run_all(self, rows, seed=0):
        from sglang.kernels.ops.elementwise.hc_combine import (
            hc_combine,
            hc_combine_split,
        )
        from sglang.srt.layers.hc_combine_fused_triton import (
            hc_combine_fused,
            hc_combine_fused_supported,
        )

        y, r, n, w = make(rows, seed)
        self.assertTrue(hc_combine_fused_supported(y, r, n, w, HC, HS))
        single = hc_combine(y, r, n, w, HC, HS)
        split = hc_combine_split(y, r, n, w, HC, HS)
        fused = hc_combine_fused(y, r, n, w, HC, HS)
        return (y, r, n, w), single, split, fused

    def test_split_matches_single(self):
        """The documented bit-identity the fused kernel is measured against."""
        for rows in (1, 16):
            _, single, split, _ = self._run_all(rows)
            self.assertTrue(torch.equal(single, split), f"rows={rows}")

    def test_fused_within_one_bf16_ulp(self):
        worst = 0
        worst_frac = 0.0
        for rows in (1, 2, 3, 4, 8, 16):
            for seed in (0, 1, 2):
                _, _, split, fused = self._run_all(rows, seed)
                ulp = bf16_ulp_diff(split, fused)
                mx = int(ulp.max().item())
                frac = float((ulp > 0).float().mean().item())
                worst = max(worst, mx)
                worst_frac = max(worst_frac, frac)
                self.assertLessEqual(
                    mx, 1, f"rows={rows} seed={seed}: max ulp {mx} > 1"
                )
        print(
            f"\n[hc_combine_fused] max bf16 ulp vs hc_combine_split = {worst}, "
            f"worst differing-element fraction = {worst_frac:.3e}"
        )

    def test_fused_matches_fp64_reference(self):
        for rows in (1, 16):
            (y, r, n, w), _, split, fused = self._run_all(rows)
            exact = ref_fp64(y, r, n, w)
            e_split = (split.double() - exact).abs().max().item()
            e_fused = (fused.double() - exact).abs().max().item()
            scale = exact.abs().max().item()
            # The fused kernel must not be measurably worse than the reference.
            self.assertLess(e_fused, max(4 * e_split, 1e-2 * scale), f"rows={rows}")

    def test_out_argument(self):
        from sglang.srt.layers.hc_combine_fused_triton import hc_combine_fused

        y, r, n, w = make(16)
        buf = torch.empty_like(r)
        got = hc_combine_fused(y, r, n, w, HC, HS, out=buf)
        self.assertEqual(got.data_ptr(), buf.data_ptr())
        self.assertTrue(torch.equal(got, hc_combine_fused(y, r, n, w, HC, HS)))

    def test_counters_are_restored(self):
        """A stuck barrier counter would deadlock the next launch / graph replay."""
        from sglang.srt.layers.hc_combine_fused_triton import _state, hc_combine_fused

        y, r, n, w = make(16)
        hc_combine_fused(y, r, n, w, HC, HS)
        torch.cuda.synchronize()
        _, counters = _state(r.device, HC)
        self.assertEqual(counters.tolist(), [0, 0])

    def test_cuda_graph_replay(self):
        from sglang.kernels.ops.elementwise.hc_combine import hc_combine_split
        from sglang.srt.layers.hc_combine_fused_triton import hc_combine_fused

        y, r, n, w = make(16)
        out = torch.empty_like(r)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                hc_combine_fused(y, r, n, w, HC, HS, out=out)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            hc_combine_fused(y, r, n, w, HC, HS, out=out)
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        ulp = bf16_ulp_diff(hc_combine_split(y, r, n, w, HC, HS), out)
        self.assertLessEqual(int(ulp.max().item()), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
