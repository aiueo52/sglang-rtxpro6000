"""GPU builders for QSA CUDA-graph replay metadata.

Compressed addressing is pure arithmetic over the page-aligned full-KV
cache (the DSV4 scheme): a group's compressed slot is any of its raw slots
floor-divided by the compress ratio, so the kernels rebuild the per-row
graph buffers from request lengths plus ``req_to_token`` alone — no
allocation, no ownership state, and accept-dependent speculative lengths
never need the host.

* ``_qsa_graph_layout_kernel`` (one program per request + a tail program) —
  request row layout: decode rows or speculative verify/draft-extend rows
  plus static dummy-tail rows.
* ``_qsa_graph_row_metadata_kernel`` (one program per row) — compressed
  lengths, the boundary write slot (last raw slot // ratio; non-boundary
  rows keep the inert reserved slot 0), the row's page table of full-KV
  page ids, and the layer-independent indexer inputs (logical position,
  pending-ring state slot, trailing-group member ring slots).

Both are launched once eagerly at capture warmup (JIT compile + dummy
layout) and then recorded into the main CUDA graph through
``init_forward_metadata_in_graph``. All inputs are stable-address runner
buffers.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl


# Page-table entries handled by one program (serial path: per loop trip).
_PAGE_BLOCK = 128


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


@functools.lru_cache(maxsize=None)
def _page_parallel_enabled() -> bool:
    """``SGLANG_QSA_META_PAGE_PARALLEL``: spread the page-table build over CTAs.

    The row-metadata kernel is launched one program per row with ``num_warps=1``
    and then walks the whole ``max_pages`` page table in a serial
    ``PAGE_BLOCK``-strided loop (32 iterations at the 262144-token / page-64
    serving config).  One warp cannot hide the latency of that strided gather,
    so the kernel costs ~10 us warm / ~21 us cold *regardless of row count* --
    at W16 it runs 14x per draft phase plus twice more per step in
    verify/draft-extend, i.e. ~0.20 ms of the 20.5 ms decode step (~0.10 ms at
    W4; see specs/DRAFT_GLUE_SPEC.md item DG-1).

    With the flag on, the page dimension becomes ``program_id(1)`` and each
    program writes exactly one ``PAGE_BLOCK`` slice.  Every store is the same
    value at the same address as before, so the flag is bit-exact.
    """
    return _env_flag("SGLANG_QSA_META_PAGE_PARALLEL")


@functools.lru_cache(maxsize=None)
def _page_bound_enabled() -> bool:
    """``SGLANG_QSA_META_PAGE_BOUND``: only refresh the pages the row can read.

    A row's scoring kernel reads page ids below ``cdiv(seq_len, FULL_PAGE)``;
    the entries above that are rebuilt every call for nothing (4096 of them at
    32k context, i.e. 8x the useful work).  Bounding the loop leaves stale
    values in the unread tail, so this is NOT bit-exact on the buffer -- only
    on everything downstream reads.  Keep it off until the indexer's page bound
    has been re-confirmed (DG-1b).
    """
    return _env_flag("SGLANG_QSA_META_PAGE_BOUND")


def qsa_row_metadata_grid(num_rows: int, max_pages: int, page_block: int,
                          page_parallel: bool):
    """Launch grid for ``_qsa_graph_row_metadata_kernel`` (pure python, testable).

    Serial form keeps the historical 1-D ``(num_rows,)`` grid; the parallel form
    adds one program per ``page_block`` slice of the page table.
    """
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}")
    if max_pages <= 0:
        raise ValueError(f"max_pages must be positive, got {max_pages}")
    if page_block <= 0:
        raise ValueError(f"page_block must be positive, got {page_block}")
    if not page_parallel:
        return (num_rows,)
    return (num_rows, triton.cdiv(max_pages, page_block))


@triton.jit
def _qsa_graph_layout_kernel(
    # Request-level inputs.
    seq_lens_ptr,  # [bs] base lengths
    req_pool_ptr,  # [bs] request pool slots
    extend_lens_ptr,  # [bs] per-request extend lengths (draft extend)
    # Row layout buffers (graph persistent state).
    row_seq_lens_ptr,
    row_prefix_lens_ptr,
    row_req_pool_ptr,
    bs,
    num_tokens,
    num_padding,
    extend_len,  # uniform extend length (target verify); 0 -> extend_lens_ptr
    MODE: tl.constexpr,  # 0 = decode, 1 = target verify, 2 = draft extend
):
    pid = tl.program_id(0)

    if MODE == 0:
        if pid < bs:
            real = pid < bs - num_padding
            seq_len = tl.load(seq_lens_ptr + pid).to(tl.int32)
            req = tl.load(req_pool_ptr + pid).to(tl.int64)
            # Padding rows alias request slot 0: it is never allocated, so
            # its pending-ring rows are the inert dump for their state
            # stores, and its req_to_token row reads stay in-bounds.
            req = tl.where(real, req, 0)
            seq_len = tl.where(real, seq_len, 1)
            prefix = tl.maximum(seq_len - 1, 0)
            tl.store(row_seq_lens_ptr + pid, seq_len)
            tl.store(row_req_pool_ptr + pid, req.to(tl.int32))
            tl.store(row_prefix_lens_ptr + pid, prefix)
        return

    real_reqs = bs - num_padding
    if pid == bs:
        # Tail program: dummy-fill the static-capacity rows past the real
        # layout (length 1, prefix 0, aliased to request slot 0, matching
        # the legacy padding contract).
        if MODE == 1:
            row_start = real_reqs * extend_len
        else:
            row_start = 0
            for j in range(real_reqs):
                row_start += tl.load(extend_lens_ptr + j)
        for row in range(row_start, num_tokens):
            tl.store(row_seq_lens_ptr + row, 1)
            tl.store(row_prefix_lens_ptr + row, 0)
            # Request slot 0 is never allocated: inert for ring stores and
            # in-bounds for every row read.
            tl.store(row_req_pool_ptr + row, 0)
        return

    if MODE == 1:
        eff = tl.where(pid < real_reqs, extend_len, 0)
        offset = tl.minimum(pid, real_reqs) * extend_len
    else:
        eff = 0
        offset = 0
        for j in range(bs):
            e_j = tl.where(j < real_reqs, tl.load(extend_lens_ptr + j), 0)
            offset += tl.where(j < pid, e_j, 0)
            eff = tl.where(j == pid, e_j, eff)
    base = tl.load(seq_lens_ptr + pid).to(tl.int32)
    req = tl.load(req_pool_ptr + pid).to(tl.int64)
    if MODE == 1:
        prefix = base
        limit = base + eff
    else:
        prefix = tl.maximum(base - eff, 0)
        limit = base
    for j in range(eff):
        row = offset + j
        seq_len = tl.minimum(prefix + 1 + j, limit)
        tl.store(row_seq_lens_ptr + row, seq_len)
        tl.store(row_prefix_lens_ptr + row, prefix)
        tl.store(row_req_pool_ptr + row, req.to(tl.int32))


@triton.jit
def _qsa_graph_row_metadata_kernel(
    # Row layout buffers (filled by the layout kernel).
    row_seq_lens_ptr,
    row_req_pool_ptr,
    # Graph output buffers.
    compressed_lens_ptr,
    write_locs_ptr,
    page_table_ptr,
    logical_positions_ptr,
    state_slots_ptr,
    ring_locs_ptr,
    # Pool state.
    req_to_token_ptr,
    req_to_token_row_stride,
    max_pages,
    RATIO: tl.constexpr,
    RING: tl.constexpr,
    FULL_PAGE: tl.constexpr,  # full-KV tokens per page
    PAGE_BLOCK: tl.constexpr,
    PAGE_PARALLEL: tl.constexpr = False,  # page dim on program_id(1)
    PAGE_BOUND: tl.constexpr = False,  # stop at the row's last readable page
):
    row = tl.program_id(0)
    # Serial launch keeps a 1-D grid, so `PAGE_PARALLEL` folds the page id and
    # the guard below away at trace time -- flag-off lowers to the same IR it
    # did before this variant existed.
    if PAGE_PARALLEL:
        page_pid = tl.program_id(1)
        write_scalars = page_pid == 0
    else:
        page_pid = 0
        write_scalars = True
    seq_len = tl.load(row_seq_lens_ptr + row).to(tl.int32)
    req = tl.load(row_req_pool_ptr + row).to(tl.int64)
    token_row = req * req_to_token_row_stride
    current = tl.maximum(seq_len - 1, 0)
    last_loc = tl.load(req_to_token_ptr + token_row + current).to(tl.int32)

    compressed = seq_len // RATIO
    if write_scalars:
        tl.store(compressed_lens_ptr + row, compressed)

    # DSV4-style compressed addressing: the page-aligned full-KV allocator
    # keeps every compression group contiguous inside one page, so the
    # group's compressed slot is any of its raw slots floor-divided by the
    # ratio. Non-boundary rows keep the inert reserved slot 0 (full slot 0
    # is the pools' padding slot).
    boundary = (seq_len > 0) & (seq_len % RATIO == 0)
    write_loc = tl.where(boundary, last_loc // RATIO, 0)
    if write_scalars:
        tl.store(write_locs_ptr + row, write_loc)

        tl.store(logical_positions_ptr + row, current)
        tl.store(state_slots_ptr + row, req * RING + (current % RING).to(tl.int64))
        ring_base = row.to(tl.int64) * RATIO
        for k in tl.static_range(RATIO):
            member = tl.maximum(current - (RATIO - 1 - k), 0)
            slot = req * RING + (member % RING).to(tl.int64)
            tl.store(ring_locs_ptr + ring_base + k, slot.to(tl.int32))

    # Page-table entries are the request's FULL-KV page ids, read from the
    # page-aligned req_to_token row; the scoring kernels turn them into
    # compressed slots as page_id * (FULL_PAGE // RATIO) + block_in_page.
    table_row = page_table_ptr + row.to(tl.int64) * max_pages
    offs = tl.arange(0, PAGE_BLOCK)
    row_width_pages = req_to_token_row_stride // FULL_PAGE
    page_limit = tl.minimum(max_pages, row_width_pages)
    if PAGE_BOUND:
        # The scoring kernel walks pages while page * FULL_PAGE < seq_len, so
        # every entry at or above this bound is dead.  Not bit-exact on the
        # buffer: the tail keeps whatever the previous refresh left there.
        page_limit = tl.minimum(page_limit, (seq_len + FULL_PAGE - 1) // FULL_PAGE)
    if PAGE_PARALLEL:
        idx = page_pid * PAGE_BLOCK + offs
        valid = idx < page_limit
        loc = tl.load(
            req_to_token_ptr + token_row + idx * FULL_PAGE, mask=valid, other=0
        )
        tl.store(table_row + idx, tl.maximum(loc // FULL_PAGE, 0), mask=valid)
    else:
        for p0 in range(0, max_pages, PAGE_BLOCK):
            idx = p0 + offs
            valid = idx < page_limit
            loc = tl.load(
                req_to_token_ptr + token_row + idx * FULL_PAGE, mask=valid, other=0
            )
            tl.store(table_row + idx, tl.maximum(loc // FULL_PAGE, 0), mask=valid)


def supports_graph_metadata_kernels(pool, device) -> bool:
    """Whether the CUDA fast path can serve this pool/device pair."""

    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    return torch.device(device).type == "cuda" and isinstance(pool, QSATokenToKVPool)


def launch_graph_metadata(
    *,
    mode,
    bs,
    num_rows,
    seq_lens,
    req_pool_indices,
    extend_lens,
    extend_len,
    num_padding,
    metadata,
    req_to_token,
    pool,
) -> None:
    """Launch the two metadata kernels for one graph bucket.

    Used both for the eager capture-warmup launch and for recording the
    kernels into the main CUDA graph (``init_forward_metadata_in_graph``).
    """

    indexer = metadata.indexer_metadata
    max_pages = indexer.graph_compressed_page_table.shape[1]
    row_seq_lens = metadata.sequence_lengths
    row_req_pool = metadata.row_req_pool_indices
    row_prefix_lens = indexer.graph_prefix_lengths

    _qsa_graph_layout_kernel[(bs + 1,)](
        seq_lens,
        req_pool_indices,
        (
            extend_lens
            if extend_lens is not None
            else row_seq_lens  # unused dummy pointer
        ),
        row_seq_lens,
        row_prefix_lens,
        row_req_pool,
        bs,
        num_rows,
        num_padding,
        extend_len,
        MODE=mode,
        num_warps=1,
    )
    page_parallel = _page_parallel_enabled()
    _qsa_graph_row_metadata_kernel[
        qsa_row_metadata_grid(num_rows, max_pages, _PAGE_BLOCK, page_parallel)
    ](
        row_seq_lens,
        row_req_pool,
        indexer.graph_compressed_lengths,
        indexer.graph_write_locs,
        indexer.graph_compressed_page_table,
        indexer.decode_logical_positions,
        indexer.pending_ring_slots,
        indexer.graph_ring_group_locs,
        req_to_token,
        req_to_token.stride(0),
        max_pages,
        RATIO=indexer.compress_ratio,
        RING=indexer.ring_size or indexer.compress_ratio,
        FULL_PAGE=pool.qsa_compressed_page_size * indexer.compress_ratio,
        PAGE_BLOCK=_PAGE_BLOCK,
        PAGE_PARALLEL=page_parallel,
        PAGE_BOUND=_page_bound_enabled(),
        num_warps=1,
    )
