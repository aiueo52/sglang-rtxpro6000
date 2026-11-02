"""CPU checks for SGLANG_QSA_PAGED_PREFILL (QSA sparse chunk prefill over the pool).

Runs under the Triton interpreter only (no GPU):

    CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 PYTHONPATH=python \
        python -m unittest -v test/registered/kernels/test_qsa_paged_prefill_cpu.py

The paged kernel (k/v = the KV pool, logical top-k positions resolved through
req_to_token) must match the packed kernel fed with the gathered + concatenated
per-request K/V bit for bit, and the torch reference within 2e-2. Two requests
(pool rows 3 and 1, 700 and 300 tokens, 64 extend rows each) live on 40 shuffled
64-token pages; top-k rows are random compressed blocks expanded to width 35
with -1 holes pushed back and the partial tail block.
"""

import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

_INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

if _INTERPRET:
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as triton_interpreter

    from sglang.srt.environ import envs
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_mod
    from sglang.srt.layers.attention.qsa import sparse_attn
    from sglang.srt.layers.attention.qsa.kernel import (
        qsa_sparse_attention_reference,
        torch_expand_qsa_block_indices,
    )
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

PAGE = 64
NUM_PAGES = 40
Q_HEADS = 24
KV_HEADS = 2
RATIO = 4
TOKEN_TOPK = 32
BLOCK_TOPK = TOKEN_TOPK // RATIO  # 8 compressed blocks -> 35 token columns
REQ_POOL_ROWS = (3, 1)
SEQ_LENS = (700, 300)
EXTEND = 64
REQ_TO_TOKEN_SHAPE = (5, 1024)
# Smallest slot whose fp8 row offset (512 elements per slot at 2 x 256) no
# longer fits int32: 2**31 / 512.
INT32_SLOT_LIMIT = 2**31 // (KV_HEADS * 256)


_float_arg_patch = None


def setUpModule():
    # On the GPU a Python float kernel argument (the softmax scale) is an fp32
    # runtime scalar, so `bf16 q * scale` computes in fp32. This interpreter
    # keeps it a Python constant, turns the product into bf16 arithmetic and
    # fails (no bf16 constants). Pass floats as fp32 scalars, like the GPU.
    global _float_arg_patch
    if not _INTERPRET:
        return
    original = triton_interpreter._implicit_cvt

    def implicit_cvt(arg):
        if isinstance(arg, float):
            handle = triton_interpreter.TensorHandle(
                np.array([arg], dtype=np.float32), tl.float32
            )
            return tl.tensor(handle, tl.float32)
        return original(arg)

    _float_arg_patch = mock.patch.object(
        triton_interpreter, "_implicit_cvt", implicit_cvt
    )
    _float_arg_patch.start()


def tearDownModule():
    if _float_arg_patch is not None:
        _float_arg_patch.stop()


def _poison(t: torch.Tensor) -> None:
    """NaN everywhere, so any stray pool read shows up in the output."""
    if t.dtype == torch.float8_e4m3fn:
        t.view(torch.uint8).fill_(0x7F)
    else:
        t.fill_(float("nan"))


def _make_case(head_dim, kv_dtype, seed, *, interleaved=False, slot_base=0):
    g = torch.Generator().manual_seed(seed)
    total_tokens = sum(SEQ_LENS)

    # Page 0 is the padding page; requests take shuffled pages 1..40.
    pages = torch.randperm(NUM_PAGES, generator=g) + 1
    req_to_token = torch.randint(
        1, (NUM_PAGES + 1) * PAGE, REQ_TO_TOKEN_SHAPE, generator=g, dtype=torch.int32
    )
    req_to_token += slot_base
    slots_per_req = []
    cursor = 0
    for req, length in zip(REQ_POOL_ROWS, SEQ_LENS):
        n_pages = math.ceil(length / PAGE)
        own = pages[cursor : cursor + n_pages]
        cursor += n_pages
        slots = (own[:, None] * PAGE + torch.arange(PAGE)).reshape(-1)[:length]
        slots = (slots + slot_base).to(torch.int32)
        req_to_token[req, :length] = slots
        slots_per_req.append(slots.long())
    all_slots = torch.cat(slots_per_req)

    pool_rows = slot_base + (NUM_PAGES + 1) * PAGE
    shape = (pool_rows, KV_HEADS, head_dim)
    if slot_base:
        # Slots past 2**31 / row stride: only touched pages get memory.
        k_pool = torch.empty(shape, dtype=kv_dtype)
        v_pool = torch.empty(shape, dtype=kv_dtype)
    elif interleaved:
        # Non-contiguous views of one [rows, 2, H, D] buffer; the paged path
        # must follow the strides instead of copying the pool.
        kv = torch.empty((pool_rows, 2, KV_HEADS, head_dim), dtype=kv_dtype)
        _poison(kv)
        k_pool, v_pool = kv[:, 0], kv[:, 1]
    else:
        k_pool = torch.empty(shape, dtype=kv_dtype)
        v_pool = torch.empty(shape, dtype=kv_dtype)
        _poison(k_pool)
        _poison(v_pool)

    k_tokens = torch.randn(total_tokens, KV_HEADS, head_dim, generator=g).to(kv_dtype)
    v_tokens = torch.randn(total_tokens, KV_HEADS, head_dim, generator=g).to(kv_dtype)
    k_pool[all_slots] = k_tokens
    v_pool[all_slots] = v_tokens

    q = torch.randn(len(SEQ_LENS) * EXTEND, Q_HEADS, head_dim, generator=g).to(
        torch.bfloat16
    )

    positions, row_req = [], []
    for b, length in enumerate(SEQ_LENS):
        positions.append(torch.arange(length - EXTEND, length))
        row_req.append(torch.full((EXTEND,), REQ_POOL_ROWS[b]))
    positions = torch.cat(positions)
    row_req = torch.cat(row_req)
    visible = positions + 1

    blocks = torch.full((positions.numel(), BLOCK_TOPK), -1, dtype=torch.int32)
    for row, vis in enumerate(visible.tolist()):
        full_blocks = vis // RATIO
        chosen = torch.randperm(full_blocks, generator=g)[:BLOCK_TOPK]
        drop = torch.rand(BLOCK_TOPK, generator=g) < 0.3
        if vis % RATIO and row % 7 == 0:
            drop[:] = True  # only the partial tail block remains
        elif drop.all():
            drop[0] = False
        blocks[row] = torch.where(drop, torch.full_like(chosen, -1), chosen).to(
            torch.int32
        )
    topk = torch_expand_qsa_block_indices(
        blocks, positions, visible, RATIO, TOKEN_TOPK
    ).contiguous()
    assert topk.shape[1] == TOKEN_TOPK + RATIO - 1 == 35
    assert (topk < visible[:, None]).all()
    assert ((topk >= 0).sum(1) > 0).all()

    physical = req_to_token[row_req[:, None], topk.clamp(min=0).long()]
    physical = torch.where(topk >= 0, physical, torch.full_like(physical, -1))

    cu_q = torch.arange(len(SEQ_LENS) + 1, dtype=torch.int32) * EXTEND
    kv_lens = torch.tensor(SEQ_LENS, dtype=torch.int32)
    return SimpleNamespace(
        head_dim=head_dim,
        kv_dtype=kv_dtype,
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        k_tokens=k_tokens,
        v_tokens=v_tokens,
        req_to_token=req_to_token,
        req_pool_indices=torch.tensor(REQ_POOL_ROWS, dtype=torch.int64),
        slots_per_req=slots_per_req,
        topk=topk,
        physical=physical.to(torch.int32),
        cu_q=cu_q,
        cu_k=torch.nn.functional.pad(kv_lens.cumsum(0), (1, 0)).to(torch.int32),
        kv_lens=kv_lens,
        scale=head_dim**-0.5,
    )


def _run_packed(case):
    """The current (gather + cat) path."""
    k = torch.cat([case.k_pool.index_select(0, s) for s in case.slots_per_req])
    v = torch.cat([case.v_pool.index_select(0, s) for s in case.slots_per_req])
    return sparse_attn.sparse_gqa_fwd_interface_triton_ck(
        case.q, k, v, case.topk, case.cu_q, case.cu_k, case.kv_lens, case.scale
    )


def _run_paged(case):
    return sparse_attn.sparse_gqa_fwd_interface_triton_ck(
        case.q,
        case.k_pool,
        case.v_pool,
        case.topk,
        case.cu_q,
        None,
        case.kv_lens,
        case.scale,
        req_to_token=case.req_to_token,
        req_pool_indices=case.req_pool_indices,
    )


def _reference(case):
    return qsa_sparse_attention_reference(
        case.q, case.k_pool, case.v_pool, case.physical, case.scale
    )


@unittest.skipUnless(_INTERPRET, "needs TRITON_INTERPRET=1 (CPU Triton interpreter)")
class TestPagedChunkPrefillKernel(unittest.TestCase):
    def _check(self, case):
        packed = _run_packed(case)
        paged = _run_paged(case)
        self.assertFalse(packed.isnan().any().item(), "packed output has NaN")
        self.assertTrue(torch.equal(paged, packed), "paged != packed (bitwise)")
        torch.testing.assert_close(
            paged.float(), _reference(case).float(), atol=2e-2, rtol=2e-2
        )
        return paged

    def test_fp8_d64(self):
        self._check(_make_case(64, torch.float8_e4m3fn, seed=1))

    def test_bf16_d64(self):
        self._check(_make_case(64, torch.bfloat16, seed=2))

    def test_bf16_d64_interleaved_pool_view(self):
        self._check(_make_case(64, torch.bfloat16, seed=3, interleaved=True))

    def test_fp8_d64_multi_block_loop(self):
        # BLOCK_N 16 < 35 columns: three loop iterations, online softmax
        # rescaling and the masked last block.
        with mock.patch.object(sparse_attn, "_get_best_config", lambda _: (16, 4, 2)):
            self._check(_make_case(64, torch.float8_e4m3fn, seed=4))

    def test_fp8_d256(self):
        self._check(_make_case(256, torch.float8_e4m3fn, seed=5))

    def test_fp8_d256_slots_past_int32_offsets(self):
        # Every request slot sits above 2**31 / 512, so slot * row stride needs
        # int64; the pool is ~2.2 GB of untouched virtual memory.
        case = _make_case(
            256, torch.float8_e4m3fn, seed=6, slot_base=INT32_SLOT_LIMIT + 1000
        )
        self.assertGreater(int(torch.cat(case.slots_per_req).min()), INT32_SLOT_LIMIT)
        self._check(case)

    def test_paged_rejects_missing_req_pool_indices(self):
        case = _make_case(64, torch.bfloat16, seed=7)
        with self.assertRaises(ValueError):
            sparse_attn.sparse_gqa_fwd_interface_triton_ck(
                case.q,
                case.k_pool,
                case.v_pool,
                case.topk,
                case.cu_q,
                None,
                case.kv_lens,
                case.scale,
                req_to_token=case.req_to_token,
            )


class _CudaFlagged(torch.Tensor):
    """CPU tensor reporting is_cuda, so forward_extend takes its CUDA branch."""

    @property
    def is_cuda(self):
        return True


def _plain(x):
    return x.as_subclass(torch.Tensor) if isinstance(x, torch.Tensor) else x


@unittest.skipUnless(_INTERPRET, "needs TRITON_INTERPRET=1 (CPU Triton interpreter)")
class TestPagedChunkPrefillBackend(unittest.TestCase):
    """forward_extend's prefix branch, old (gather + cat) vs SGLANG_QSA_PAGED_PREFILL."""

    def _forward(self, case, paged):
        k_pool = case.k_pool.clone()
        v_pool = case.v_pool.clone()
        out_cache_loc = torch.cat(
            [s[-EXTEND:] for s in case.slots_per_req]
        ).to(torch.int64)
        # The extend rows are not in the pool yet: forward_extend commits them.
        stale = torch.empty_like(k_pool[out_cache_loc])
        _poison(stale)
        k_pool[out_cache_loc] = stale
        v_pool[out_cache_loc] = stale

        class Pool:
            def set_kv_buffer(self, layer, loc, k, v):
                k_pool[loc.long()] = _plain(k).to(k_pool.dtype)
                v_pool[loc.long()] = _plain(v).to(v_pool.dtype)

            def get_key_buffer(self, layer_id):
                return k_pool

            def get_value_buffer(self, layer_id):
                return v_pool

        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.token_to_kv_pool = Pool()
        backend.req_to_token_pool = SimpleNamespace(req_to_token=case.req_to_token)
        with mock.patch.dict(os.environ):
            if paged:
                os.environ["SGLANG_QSA_PAGED_PREFILL"] = "1"
            else:
                os.environ.pop("SGLANG_QSA_PAGED_PREFILL", None)
            # What QwenSparseAttnBackend.__init__ reads.
            backend._qsa_paged_prefill = envs.SGLANG_QSA_PAGED_PREFILL.get()
        self.assertEqual(backend._qsa_paged_prefill, paged)

        extend_rows = torch.cat(
            [
                torch.arange(length - EXTEND, length) + offset
                for length, offset in zip(SEQ_LENS, (0, SEQ_LENS[0]))
            ]
        )
        k_new = case.k_tokens[extend_rows].to(torch.bfloat16)
        v_new = case.v_tokens[extend_rows].to(torch.bfloat16)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=out_cache_loc,
            extend_seq_lens=torch.full((len(SEQ_LENS),), EXTEND, dtype=torch.int32),
            extend_seq_lens_cpu=[EXTEND] * len(SEQ_LENS),
            seq_lens_cpu=list(SEQ_LENS),
            req_pool_indices=case.req_pool_indices,
        )
        layer = SimpleNamespace(
            tp_q_head_num=Q_HEADS,
            head_dim=case.head_dim,
            layer_id=0,
            scaling=case.scale,
        )
        calls = []
        real = sparse_attn.sparse_gqa_fwd_interface_triton_ck

        def spy(*args, **kwargs):
            args = [_plain(a) for a in args]
            kwargs = {key: _plain(value) for key, value in kwargs.items()}
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        q = case.q.reshape(case.q.shape[0], -1).as_subclass(_CudaFlagged)
        with mock.patch.object(backend_mod, "sparse_gqa_fwd_interface_triton_ck", spy):
            out = backend.forward_extend(
                q,
                k_new,
                v_new,
                layer,
                batch,
                save_kv_cache=True,
                topk_indices=case.topk,
            )
        self.assertEqual(len(calls), 1)
        return _plain(out), calls[0], k_pool, v_pool

    def _check(self, case):
        old, (old_args, old_kwargs), _, _ = self._forward(case, paged=False)
        new, (new_args, new_kwargs), k_pool, v_pool = self._forward(case, paged=True)
        total = sum(SEQ_LENS)
        # Old path: packed full-context copies; new path: the pool itself.
        self.assertEqual(old_args[1].shape[0], total)
        self.assertEqual(old_kwargs, {})
        self.assertEqual(new_args[1].data_ptr(), k_pool.data_ptr())
        self.assertEqual(new_args[2].data_ptr(), v_pool.data_ptr())
        self.assertEqual(new_args[1].shape, k_pool.shape)
        self.assertEqual(
            new_kwargs["req_to_token"].data_ptr(), case.req_to_token.data_ptr()
        )
        self.assertFalse(old.isnan().any().item())
        self.assertTrue(torch.equal(new, old), "paged forward_extend != gather path")
        # Same as the kernel run on the fully written pool, and the reference.
        kernel = _run_paged(case).reshape(old.shape)
        self.assertTrue(torch.equal(new, kernel))
        torch.testing.assert_close(
            new.float(),
            _reference(case).reshape(old.shape).float(),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_forward_extend_fp8_d64(self):
        self._check(_make_case(64, torch.float8_e4m3fn, seed=11))

    def test_forward_extend_bf16_d64(self):
        self._check(_make_case(64, torch.bfloat16, seed=12))

    def test_env_switch_defaults_off(self):
        self.assertFalse(QwenSparseAttnBackend._qsa_paged_prefill)
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_QSA_PAGED_PREFILL", None)
            self.assertFalse(envs.SGLANG_QSA_PAGED_PREFILL.get())
        with mock.patch.dict(os.environ, {"SGLANG_QSA_PAGED_PREFILL": "1"}):
            self.assertTrue(envs.SGLANG_QSA_PAGED_PREFILL.get())
        with mock.patch.dict(os.environ, {"SGLANG_QSA_PAGED_PREFILL": "0"}):
            self.assertFalse(envs.SGLANG_QSA_PAGED_PREFILL.get())


if __name__ == "__main__":
    unittest.main(verbosity=2)
