"""Offline proof that the T16-strided patch to the vendored FlashInfer GDN WY
wrapper is bit-identical to the unpatched wrapper.

The wrapper lives in the venv
(``flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py``) and a server may be
re-importing it at any time, so this test never touches it: it copies the file to a
temp directory, applies ``bench/glue_e/gdn_wy_T16_strided.patch`` there, imports the
copy under a private module name and compares against the installed module.

Shapes are the ones the Qwen3.8-Flash-Next verify path actually passes
(``srt/layers/attention/linear/gdn_backend.py`` -> ``kernels/gdn_flashinfer.py``):
H = HK = 16 heads x 128, HV = 48 heads x 128, so the fused post-conv row is
conv_dim = 2048 + 2048 + 6144 = 10240 elements and q/k/v are column slices of it
with token stride 10240.

Run:  python test/srt/layers/test_gdn_wy_t16_strided.py
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

REPO = pathlib.Path(__file__).resolve().parents[3]
PATCH = REPO / "bench" / "glue_e" / "gdn_wy_T16_strided.patch"

# Real Qwen3.8-Flash-Next GDN shapes.
H = HK = 16
HV = 48
K_DIM = V_DIM = 128
Q_DIM = H * K_DIM  # 2048
K_OFF = Q_DIM
V_OFF = Q_DIM + HK * K_DIM  # 4096
CONV_DIM = Q_DIM + HK * K_DIM + HV * V_DIM  # 10240
POOL = 8

ENV = "FLASHINFER_GDN_WY_STRIDED_QKV"


def _vendored_path() -> pathlib.Path:
    spec = importlib.util.find_spec(
        "flashinfer.gdn_kernels.gdn_decode_bf16_wy_output_only"
    )
    assert spec is not None and spec.origin is not None, "flashinfer not importable"
    return pathlib.Path(spec.origin)


def _import_from(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_modules():
    """(reference_module, patched_module). Reference has strided-qkv OFF."""
    src = _vendored_path()

    os.environ.pop(ENV, None)
    ref = _import_from(src, "_glue_e_gdn_wy_ref")
    assert ref._STRIDED_QKV is False

    tmp = tempfile.mkdtemp(prefix="glue_e_gdn_")
    dst = pathlib.Path(tmp) / "gdn_decode_bf16_wy_output_only_patched.py"
    shutil.copy2(src, dst)
    subprocess.run(
        ["patch", "-p0", "-s", str(dst)],
        stdin=open(PATCH, "rb"),
        check=True,
        cwd=tmp,
    )
    os.environ[ENV] = "1"
    try:
        pat = _import_from(dst, "_glue_e_gdn_wy_patched")
    finally:
        os.environ.pop(ENV, None)
    assert pat._STRIDED_QKV is True
    return ref, pat


def _make_inputs(B: int, T: int, seed: int = 0, pool: int = POOL):
    """Post-conv fused row plus the strided q/k/v column slices the backend passes."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    conv_out = torch.randn(
        B * T, CONV_DIM, dtype=torch.bfloat16, device="cuda", generator=g
    )
    rows = conv_out.view(B, T, CONV_DIM)
    q = rows[:, :, :K_OFF].view(B, T, H, K_DIM)
    k = rows[:, :, K_OFF:V_OFF].view(B, T, HK, K_DIM)
    v = rows[:, :, V_OFF:].view(B, T, HV, V_DIM)
    assert not q.is_contiguous() and q.stride(1) == CONV_DIM
    assert not k.is_contiguous() and k.stride(1) == CONV_DIM
    assert not v.is_contiguous() and v.stride(1) == CONV_DIM

    a = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda", generator=g)
    b = torch.randn(B, T, HV, dtype=torch.bfloat16, device="cuda", generator=g)
    A_log = torch.randn(HV, dtype=torch.float32, device="cuda", generator=g)
    dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda", generator=g)
    h0 = (
        torch.randn(
            pool, HV, V_DIM, K_DIM, dtype=torch.float32, device="cuda", generator=g
        )
        .mul_(0.05)
        .to(torch.bfloat16)
    )
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    return dict(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=h0,
        initial_state_indices=idx,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=None,
        output=None,
        conv_out=conv_out,
    )


def _run(mod, args):
    kw = {k: v for k, v in args.items() if k != "conv_out"}
    return mod.gated_delta_rule_mtp(**kw).clone()


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestGdnWyT16Strided(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref, cls.pat = _load_modules()

    def _check(self, B, T):
        args = _make_inputs(B, T)
        out_ref = _run(self.ref, args)
        out_pat = _run(self.pat, args)
        self.assertEqual(out_ref.shape, out_pat.shape)
        self.assertTrue(torch.isfinite(out_ref.float()).all(), "reference is not finite")
        same = torch.equal(out_ref, out_pat)
        if not same:
            d = (out_ref.float() - out_pat.float()).abs()
            self.fail(
                f"T={T} B={B}: not bit-identical, max|d|={d.max().item():.3e}, "
                f"n_diff={(d > 0).sum().item()}/{d.numel()}"
            )

    def test_t16_bit_identical(self):
        """T == T_KERNEL: the path this patch adds (W16 NEXTN verify)."""
        self._check(B=1, T=16)

    def test_t16_batch(self):
        self._check(B=4, T=16)

    def test_t4_bit_identical(self):
        """T == 4 (W4 verify): the pre-existing native strided path, unchanged."""
        self._check(B=1, T=4)

    def test_t4_batch(self):
        self._check(B=3, T=4)

    def test_guard_rejects_non_canonical_layouts(self):
        ok = self.pat._strided_qkv_ok
        args = _make_inputs(1, 16)
        q, k, v = args["q"], args["k"], args["v"]
        self.assertTrue(ok(q, k, v, 16, H, HK, HV, K_DIM, V_DIM))
        # a contiguous q (different token stride from k/v) must be rejected
        self.assertFalse(ok(q.contiguous(), k, v, 16, H, HK, HV, K_DIM, V_DIM))
        # a feature-minor (head-strided) q must be rejected: stride(-1) != 1
        rows = args["conv_out"].view(1, 16, CONV_DIM)
        q_bad = rows[:, :, :K_OFF].view(1, 16, K_DIM, H).transpose(2, 3)
        self.assertEqual(tuple(q_bad.shape), tuple(q.shape))
        self.assertFalse(ok(q_bad, k, v, 16, H, HK, HV, K_DIM, V_DIM))
        # wrong n_valid => batch stride mismatch (only visible at B > 1)
        args4 = _make_inputs(4, 16)
        self.assertFalse(
            ok(args4["q"], args4["k"], args4["v"], 8, H, HK, HV, K_DIM, V_DIM)
        )

    def test_patched_module_takes_the_strided_path(self):
        """Guard against a silent fallback making the bit-identity vacuous.

        ``cache_key[5]`` is ``_qkv_rs``: 0 on the copy path, the token row stride
        (conv_dim) on the strided one, and the strided key additionally carries
        ``(B, pool)``.
        """
        self._check(B=1, T=16)  # populates both caches
        self.assertTrue(self.pat._CACHE, "patched module compiled nothing")
        self.assertTrue(self.ref._CACHE, "reference module compiled nothing")
        self.assertTrue(
            any(key[5] == CONV_DIM and len(key) == 12 for key in self.pat._CACHE),
            f"patched module never took the strided path: {list(self.pat._CACHE)}",
        )
        self.assertTrue(
            all(key[5] == 0 and len(key) == 10 for key in self.ref._CACHE),
            f"reference module took a strided path: {list(self.ref._CACHE)}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
