"""The draft lm_head GEMV writing straight into the next-token logits buffer.

Removes the per-draft-step ``next_token_logits_buffer.copy_(logits)`` kernel
(14 launches / ~38 us per W16 decode step) by handing the fp32 buffer to the
W8A16 Triton GEMV as ``out=``.  The GEMV still rounds its fp32 accumulator to
bf16 before widening, so the buffer contents are bit-identical to the old
"bf16 result, then copy" sequence.

Run:  python test/srt/layers/test_draft_logits_out.py
"""

from __future__ import annotations

import unittest

import torch

# The served draft head: 32768 hot rows out of a 248320 vocab, hidden 2560.
N_HOT = 32768
K = 2560


class _MD:
    """Only field `_direct_logits_buffer` reads."""

    def __init__(self, buf):
        self.next_token_logits_buffer = buf


class TestDirectLogitsBufferGate(unittest.TestCase):
    """CPU-side: when is it legal to hand the buffer to the lm_head?"""

    def _proc(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        p = LogitsProcessor.__new__(LogitsProcessor)
        p.logit_scale = None
        p.do_tensor_parallel_all_gather = False
        p.do_tensor_parallel_all_gather_dp_attn = False
        p.vocab_size = 248320
        return p

    def setUp(self):
        import sglang.srt.layers.logits_processor as lp

        self.lp = lp
        self._saved = lp._DRAFT_LOGITS_OUT
        lp._DRAFT_LOGITS_OUT = True
        self.buf = torch.zeros(1, N_HOT, dtype=torch.float32)

    def tearDown(self):
        self.lp._DRAFT_LOGITS_OUT = self._saved

    def test_accepts_the_served_shape(self):
        p = self._proc()
        self.assertIs(p._direct_logits_buffer(_MD(self.buf), None, True), self.buf)

    def test_rejects_when_flag_is_off(self):
        self.lp._DRAFT_LOGITS_OUT = False
        p = self._proc()
        self.assertIsNone(p._direct_logits_buffer(_MD(self.buf), None, True))

    def test_rejects_the_cases_that_would_change_numerics(self):
        p = self._proc()
        md = _MD(self.buf)
        self.assertIsNone(p._direct_logits_buffer(md, None, False))  # no buffer use
        self.assertIsNone(p._direct_logits_buffer(_MD(None), None, True))
        self.assertIsNone(  # embedding_bias is added by the caller, not the GEMV
            p._direct_logits_buffer(md, torch.zeros(N_HOT), True)
        )
        for attr, val in (
            ("logit_scale", 2.0),
            ("do_tensor_parallel_all_gather", True),
            ("do_tensor_parallel_all_gather_dp_attn", True),
        ):
            q = self._proc()
            setattr(q, attr, val)
            self.assertIsNone(q._direct_logits_buffer(md, None, True), attr)
        q = self._proc()  # bf16 buffer: the copy also did a dtype change
        self.assertIsNone(
            q._direct_logits_buffer(_MD(self.buf.bfloat16()), None, True)
        )
        q = self._proc()  # head wider than the vocab: the copy path narrows it
        q.vocab_size = 1024
        self.assertIsNone(q._direct_logits_buffer(md, None, True))

    def test_copy_to_buffer_short_circuits_on_identity(self):
        p = self._proc()
        p.final_logit_softcapping = None
        md = _MD(self.buf)
        got = p._copy_logits_to_buffer(self.buf, md, use_buffer=True)
        self.assertIs(got, self.buf)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestGemvOutArgument(unittest.TestCase):
    def _fp8_case(self, M):
        g = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g)
        wf = torch.randn(N_HOT, K, dtype=torch.float32, device="cuda", generator=g)
        w = (wf / wf.abs().amax(dim=1, keepdim=True) * 400).to(torch.float8_e4m3fn)
        scale = torch.rand(N_HOT, 1, device="cuda", generator=g).add_(0.5).float()
        return x, w, scale

    def test_w8a16_gemv_out_is_bit_identical(self):
        from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv

        for M in (1, 2, 4, 16):
            x, w, scale = self._fp8_case(M)
            base = w8a16_gemv(x, w, scale)
            bf = torch.empty(M, N_HOT, dtype=torch.bfloat16, device="cuda")
            f32 = torch.empty(M, N_HOT, dtype=torch.float32, device="cuda")
            self.assertTrue(torch.equal(base, w8a16_gemv(x, w, scale, out=bf)))
            self.assertTrue(
                torch.equal(base.float(), w8a16_gemv(x, w, scale, out=f32)),
                f"M={M}: fp32 out= is not the widened bf16 result",
            )

    def test_bf16_gemv_out_is_bit_identical(self):
        from sglang.srt.layers.quantization.w8a16_gemv import bf16_gemv

        g = torch.Generator(device="cuda").manual_seed(1)
        for M in (1, 4, 16):
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g)
            w = torch.randn(512, K, dtype=torch.bfloat16, device="cuda", generator=g)
            base = bf16_gemv(x, w)
            f32 = torch.empty(M, 512, dtype=torch.float32, device="cuda")
            self.assertTrue(torch.equal(base.float(), bf16_gemv(x, w, out=f32)))

    def test_out_shape_is_validated(self):
        from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv

        x, w, scale = self._fp8_case(1)
        with self.assertRaises(ValueError):
            w8a16_gemv(
                x, w, scale, out=torch.empty(2, N_HOT, dtype=torch.float32, device="cuda")
            )
        with self.assertRaises(ValueError):
            w8a16_gemv(
                x, w, scale, out=torch.empty(1, N_HOT, dtype=torch.float16, device="cuda")
            )

    def test_apply_into_matches_apply(self):
        """Fp8LinearMethod.apply_into vs. apply + copy, through the served layout."""
        import sglang.srt.layers.quantization.fp8 as fp8

        if not fp8._W8A16_GEMV_ENABLED:
            self.skipTest("SGLANG_FP8_W8A16_GEMV is not enabled in this process")

        x, w, scale = self._fp8_case(1)

        class Layer:
            pass

        layer = Layer()
        layer.weight = w.t()  # served layout: [K, N]
        layer.weight_scale = scale

        method = fp8.Fp8LinearMethod.__new__(fp8.Fp8LinearMethod)
        method.use_marlin = False
        method.block_quant = False
        method.use_mxfp8 = False

        buf = torch.empty(1, N_HOT, dtype=torch.float32, device="cuda")
        got = method.apply_into(layer, x, buf)
        self.assertIsNotNone(got, "apply_into refused the served shape")
        self.assertIs(got, buf)
        want = method.apply(layer, x, None)
        self.assertTrue(torch.equal(want.float(), buf))
        # bias and a mismatched destination must decline rather than mis-write
        self.assertIsNone(method.apply_into(layer, x, buf, torch.zeros(N_HOT)))
        self.assertIsNone(
            method.apply_into(
                layer, x, torch.empty(1, 8, dtype=torch.float32, device="cuda")
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
