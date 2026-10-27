# Modified by aiueo52 for flash-next-fast (2026); see MODIFICATIONS.md.

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu, next_power_of_2

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import fill_accept_out_cache_loc_cpu, fill_bonus_tokens_cpu


@triton.jit
def fill_bonus_tokens(
    accept_tokens,
    accept_lens,
    bonus_tokens_ptr,
    accept_stride: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    # `accept_lens` includes the bonus token; the last accepted slot is at -1.
    accept_len = tl.load(accept_lens + pid)

    # accept_stride = per-req width of accept_tokens (= accept_index.shape[1]).
    bonus_token_idx = accept_stride * pid + accept_len - 1
    bonus_token = tl.load(accept_tokens + bonus_token_idx)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


def fill_bonus_tokens_func(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,  # mutable
    accept_stride: int,
    batch_size: int,
):
    if _is_cpu:
        fill_bonus_tokens_cpu(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_stride,
        )
        return
    fill_bonus_tokens[(batch_size,)](
        accept_tokens,
        accept_lens,
        bonus_tokens,
        accept_stride,
    )


@triton.jit
def fill_accept_out_cache_loc(
    accept_index,
    out_cache_loc,
    accept_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accept_out_cache_loc + dst, value)


def fill_accept_out_cache_loc_func(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_out_cache_loc: torch.Tensor,  # mutable
    size: int,
):
    if _is_cpu:
        fill_accept_out_cache_loc_cpu(
            accept_index,
            out_cache_loc,
            accept_out_cache_loc,
        )
        return
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )


@triton.jit
def _draft_extend_lens_kernel(
    seq_lens_ptr,
    prefix_lens_ptr,
    extend_lens_ptr,
    seq_lens_after_ptr,
    bs,
    front_offset,
    num_window_tokens,
    num_draft_tokens,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < bs
    seq_lens = tl.load(seq_lens_ptr + offs, mask=mask, other=0)
    prefix_lens = tl.maximum(seq_lens - front_offset, 0)
    tl.store(prefix_lens_ptr + offs, prefix_lens.to(tl.int32), mask=mask)
    extend_lens = tl.full([BLOCK], num_window_tokens, tl.int32)
    tl.store(extend_lens_ptr + offs, extend_lens, mask=mask)
    tl.store(seq_lens_after_ptr + offs, seq_lens + num_draft_tokens, mask=mask)


def draft_extend_lens(
    seq_lens: torch.Tensor,
    *,
    front_offset: int,
    num_window_tokens: int,
    num_draft_tokens: int,
):
    """One launch for the gpu_only draft-extend lengths: int32 prefix_lens
    (seq_lens - front_offset clamped at 0), int32 extend_lens and seq_lens +
    num_draft_tokens; bit-exact vs the torch ops in prepare_for_draft_extend."""
    bs = seq_lens.shape[0]
    prefix_lens = torch.empty((bs,), dtype=torch.int32, device=seq_lens.device)
    extend_lens = torch.empty_like(prefix_lens)
    seq_lens_after = torch.empty_like(seq_lens)
    if bs > 0:
        block = min(next_power_of_2(bs), 1024)
        _draft_extend_lens_kernel[(triton.cdiv(bs, block),)](
            seq_lens,
            prefix_lens,
            extend_lens,
            seq_lens_after,
            bs,
            front_offset,
            num_window_tokens,
            num_draft_tokens,
            BLOCK=block,
        )
    return prefix_lens, extend_lens, seq_lens_after
