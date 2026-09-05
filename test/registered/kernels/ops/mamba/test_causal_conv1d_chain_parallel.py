"""Bit-exactness of the chain-parallel verify conv1d (R4).

`SGLANG_GDN_CONV_CHAIN_PARALLEL=1` swaps the serial-over-tokens
`_causal_conv1d_update_kernel` for `_causal_conv1d_update_chain_kernel` on the
topk=1 speculative-verify path. The replacement must be byte-identical: same
outputs, same rolled-forward conv state, same intermediate conv window.

Run:
    PYTHONPATH=<worktree>/python .venv/bin/python -m pytest -q \
        test/registered/kernels/ops/mamba/test_causal_conv1d_chain_parallel.py
"""

import itertools
import os

import pytest
import torch

import sglang.kernels.ops.mamba.causal_conv1d_triton as ccv


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _make_inputs(bs, dim, seqlen, width, state_len, num_cache_lines, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # x laid out exactly as the verify call site builds it:
    # mixed_qkv.view(bs, seqlen, dim).transpose(1, 2)
    x = (
        torch.randn(
            (bs, seqlen, dim), generator=g, device="cuda", dtype=torch.float32
        )
        .to(dtype)
        .transpose(1, 2)
    )
    conv_state = torch.randn(
        (num_cache_lines, dim, state_len),
        generator=g,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)
    weight = torch.randn(
        (dim, width), generator=g, device="cuda", dtype=torch.float32
    ).to(dtype)
    bias = torch.randn((dim,), generator=g, device="cuda", dtype=torch.float32).to(
        dtype
    )
    cache_indices = torch.randperm(num_cache_lines, generator=g, device="cuda")[
        :bs
    ].to(torch.int32)
    inter_indices = torch.randperm(num_cache_lines, generator=g, device="cuda")[
        :bs
    ].to(torch.int32)
    return x, conv_state, weight, bias, cache_indices, inter_indices


def _run(chain, x, conv_state, weight, bias, cache_indices, inter_indices, has_bias,
         activation, save_intermediate, seqlen, width, dtype):
    bs, dim, _ = x.shape
    state = conv_state.clone()
    inter = None
    if save_intermediate:
        inter = torch.full(
            (conv_state.shape[0], seqlen, dim, width - 1),
            float("nan"),
            device="cuda",
            dtype=dtype,
        )
    out = torch.empty_like(x)
    old = ccv._CHAIN_PARALLEL_ENABLED
    ccv._CHAIN_PARALLEL_ENABLED = chain
    try:
        ccv.causal_conv1d_update(
            x,
            state,
            weight,
            bias if has_bias else None,
            activation,
            conv_state_indices=cache_indices,
            intermediate_conv_window=inter,
            intermediate_state_indices=inter_indices if save_intermediate else None,
            out=out,
        )
    finally:
        ccv._CHAIN_PARALLEL_ENABLED = old
    torch.cuda.synchronize()
    return out, state, inter


# dim 10240 / width 4 are the real Qwen3.8-Flash-Next GDN conv shapes
# (linear_conv_kernel_dim=4, q 16*128 + k 16*128 + v 48*128); seqlen 4 / 16 are
# --speculative-num-draft-tokens for the W4 / W16 profiles.
REAL_SHAPES = [
    # (bs, dim, seqlen, width)
    (1, 10240, 4, 4),
    (1, 10240, 16, 4),
]
RANDOM_SHAPES = [
    (1, 64, 4, 4),
    (1, 129, 16, 4),
    (2, 512, 8, 4),
    (3, 1024, 16, 3),
    (4, 333, 5, 2),
    (1, 2048, 2, 4),
    (2, 10240, 16, 4),
    (5, 96, 7, 3),
]


@pytest.mark.parametrize("shape", REAL_SHAPES + RANDOM_SHAPES)
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("activation", ["silu", None])
@pytest.mark.parametrize("save_intermediate", [True, False])
def test_chain_parallel_bit_exact(shape, has_bias, activation, save_intermediate):
    _skip_if_no_cuda()
    bs, dim, seqlen, width = shape
    dtype = torch.bfloat16
    state_len = width - 1
    num_cache_lines = bs + 3
    args = _make_inputs(
        bs, dim, seqlen, width, state_len, num_cache_lines, dtype, seed=hash(shape) % 10007
    )
    ref = _run(False, *args, has_bias, activation, save_intermediate, seqlen, width, dtype)
    new = _run(True, *args, has_bias, activation, save_intermediate, seqlen, width, dtype)
    assert torch.equal(ref[0], new[0]), "output mismatch"
    assert torch.equal(ref[1], new[1]), "conv state mismatch"
    if save_intermediate:
        a, b = ref[2], new[2]
        assert torch.equal(
            torch.nan_to_num(a, nan=1234.0), torch.nan_to_num(b, nan=1234.0)
        ), "intermediate conv window mismatch"


@pytest.mark.parametrize("block_t", [4, 8, 16])
def test_chain_parallel_block_t_invariance(block_t):
    """Tile size must not change the result (single state writer stays tile 0)."""
    _skip_if_no_cuda()
    bs, dim, seqlen, width = 1, 10240, 16, 4
    dtype = torch.bfloat16
    args = _make_inputs(bs, dim, seqlen, width, width - 1, bs + 2, dtype, seed=7)
    ref = _run(False, *args, True, "silu", True, seqlen, width, dtype)
    old = ccv._CHAIN_BLOCK_T_ENV
    ccv._CHAIN_BLOCK_T_ENV = str(block_t)
    try:
        new = _run(True, *args, True, "silu", True, seqlen, width, dtype)
    finally:
        ccv._CHAIN_BLOCK_T_ENV = old
    assert torch.equal(ref[0], new[0])
    assert torch.equal(ref[1], new[1])
    assert torch.equal(
        torch.nan_to_num(ref[2], nan=1.0), torch.nan_to_num(new[2], nan=1.0)
    )


def test_decode_and_tree_paths_untouched():
    """seqlen==1 decode and the eagle-tree (topk>1) call must keep the old kernel."""
    _skip_if_no_cuda()
    dim, width = 512, 4
    dtype = torch.bfloat16
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn((2, dim), generator=g, device="cuda", dtype=torch.float32).to(dtype)
    conv_state = torch.randn(
        (4, dim, width - 1), generator=g, device="cuda", dtype=torch.float32
    ).to(dtype)
    weight = torch.randn(
        (dim, width), generator=g, device="cuda", dtype=torch.float32
    ).to(dtype)
    bias = torch.randn((dim,), generator=g, device="cuda", dtype=torch.float32).to(dtype)
    idx = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    def go(chain):
        st = conv_state.clone()
        old = ccv._CHAIN_PARALLEL_ENABLED
        ccv._CHAIN_PARALLEL_ENABLED = chain
        try:
            o = ccv.causal_conv1d_update(
                x, st, weight, bias, "silu", conv_state_indices=idx
            )
        finally:
            ccv._CHAIN_PARALLEL_ENABLED = old
        torch.cuda.synchronize()
        return o, st

    a = go(False)
    b = go(True)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
