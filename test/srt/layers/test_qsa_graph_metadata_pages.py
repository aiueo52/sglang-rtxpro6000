"""Page-parallel QSA graph metadata (``SGLANG_QSA_META_PAGE_PARALLEL``).

``_qsa_graph_row_metadata_kernel`` is launched one program per *row* with
``num_warps=1`` and then walks the whole ``max_pages`` page table in a serial
``PAGE_BLOCK``-strided loop.  At the served config (context 262144, page 64)
``max_pages == 262144 // 64 == 4096``, so that is 32 loop trips of a strided
gather issued by a single warp -- ~10 us warm, ~21 us cold, and *independent of
the row count* (measured: grid [1,1,1] 10.0 us in the draft, grid [4,1,1]
21.2 us and grid [16,1,1] 20.4 us in verify).  At W16 it runs 14x per draft
phase plus twice per step elsewhere: ~0.20 ms of a 20.5 ms decode step
(~0.10 ms of 11.95 ms at W4).

With the flag on, the page dimension moves to ``program_id(1)``: one program
per ``PAGE_BLOCK`` slice, the scalar section guarded to ``program_id(1) == 0``.
Every store is the same value at the same address, so the buffers must come out
bit-identical.

Run (CPU part only):  python test/srt/layers/test_qsa_graph_metadata_pages.py
The GPU parity case skips itself when no CUDA device is visible.
"""

from __future__ import annotations

import os
import unittest

import torch

from sglang.srt.layers.attention.qsa.graph_metadata import (
    _PAGE_BLOCK,
    _qsa_graph_row_metadata_kernel,
    qsa_row_metadata_grid,
)

# Served QSA geometry (serve-local.sh: --page-size 64 --context-length 262144).
FULL_PAGE = 64
MAX_PAGES = 262144 // FULL_PAGE  # 4096
RATIO = 4
RING = 4


class TestRowMetadataGrid(unittest.TestCase):
    """Pure-python launch planner -- runs anywhere, no torch CUDA, no triton run."""

    def test_serial_grid_is_unchanged(self):
        self.assertEqual(qsa_row_metadata_grid(1, MAX_PAGES, _PAGE_BLOCK, False), (1,))
        self.assertEqual(qsa_row_metadata_grid(16, MAX_PAGES, _PAGE_BLOCK, False), (16,))

    def test_parallel_grid_covers_every_page(self):
        for num_rows in (1, 4, 16, 129):
            grid = qsa_row_metadata_grid(num_rows, MAX_PAGES, _PAGE_BLOCK, True)
            self.assertEqual(grid[0], num_rows)
            # No page may be left out and no program may run past the table.
            self.assertGreaterEqual(grid[1] * _PAGE_BLOCK, MAX_PAGES)
            self.assertLess((grid[1] - 1) * _PAGE_BLOCK, MAX_PAGES)

    def test_served_shape_is_32_page_programs(self):
        # The number this whole item is about: 32 serial loop trips today.
        self.assertEqual(qsa_row_metadata_grid(1, MAX_PAGES, _PAGE_BLOCK, True), (1, 32))

    def test_ragged_page_count_rounds_up(self):
        self.assertEqual(qsa_row_metadata_grid(2, 129, 128, True), (2, 2))
        self.assertEqual(qsa_row_metadata_grid(2, 128, 128, True), (2, 1))

    def test_rejects_degenerate_shapes(self):
        for args in ((0, 4096, 128, True), (1, 0, 128, True), (1, 4096, 0, True)):
            with self.assertRaises(ValueError):
                qsa_row_metadata_grid(*args)


def _reference_row_metadata(row_seq_lens, row_req_pool, req_to_token, max_pages):
    """Torch reference for what the kernel stores (CPU, no triton).

    Mirrors the kernel line for line so the GPU parity case has an independent
    check and not just "new kernel == old kernel".
    """
    stride = req_to_token.shape[1]
    rows = row_seq_lens.shape[0]
    seq = row_seq_lens.to(torch.int64)
    req = row_req_pool.to(torch.int64)
    current = torch.clamp(seq - 1, min=0)
    last_loc = req_to_token[req, current].to(torch.int64)

    compressed = (seq // RATIO).to(torch.int32)
    boundary = (seq > 0) & (seq % RATIO == 0)
    write_locs = torch.where(boundary, last_loc // RATIO, torch.zeros_like(last_loc))
    logical = current.to(torch.int32)
    state_slots = req * RING + (current % RING)
    ring = torch.empty((rows, RATIO), dtype=torch.int32)
    for k in range(RATIO):
        member = torch.clamp(current - (RATIO - 1 - k), min=0)
        ring[:, k] = (req * RING + (member % RING)).to(torch.int32)

    idx = torch.arange(max_pages, dtype=torch.int64)
    limit = min(max_pages, stride // FULL_PAGE)
    table = torch.zeros((rows, max_pages), dtype=torch.int32)
    valid = idx < limit
    gathered = req_to_token[req][:, (idx * FULL_PAGE).clamp(max=stride - 1)]
    table[:, :] = torch.where(
        valid.unsqueeze(0),
        torch.clamp(gathered.to(torch.int64) // FULL_PAGE, min=0),
        torch.zeros(1, dtype=torch.int64),
    ).to(torch.int32)
    return {
        "compressed_lens": compressed,
        "write_locs": write_locs.to(torch.int32),
        "page_table": table,
        "logical_positions": logical,
        "state_slots": state_slots,
        "ring_locs": ring,
    }


def _run_kernel(page_parallel, row_seq_lens, row_req_pool, req_to_token, max_pages):
    dev = row_seq_lens.device
    rows = row_seq_lens.shape[0]
    out = {
        "compressed_lens": torch.full((rows,), -7, dtype=torch.int32, device=dev),
        "write_locs": torch.full((rows,), -7, dtype=torch.int32, device=dev),
        "page_table": torch.full(
            (rows, max_pages), -7, dtype=torch.int32, device=dev
        ),
        "logical_positions": torch.full((rows,), -7, dtype=torch.int32, device=dev),
        "state_slots": torch.full((rows,), -7, dtype=torch.int64, device=dev),
        "ring_locs": torch.full((rows, RATIO), -7, dtype=torch.int32, device=dev),
    }
    _qsa_graph_row_metadata_kernel[
        qsa_row_metadata_grid(rows, max_pages, _PAGE_BLOCK, page_parallel)
    ](
        row_seq_lens,
        row_req_pool,
        out["compressed_lens"],
        out["write_locs"],
        out["page_table"],
        out["logical_positions"],
        out["state_slots"],
        out["ring_locs"],
        req_to_token,
        req_to_token.stride(0),
        max_pages,
        RATIO=RATIO,
        RING=RING,
        FULL_PAGE=FULL_PAGE,
        PAGE_BLOCK=_PAGE_BLOCK,
        PAGE_PARALLEL=page_parallel,
        PAGE_BOUND=False,
        num_warps=1,
    )
    return out


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class TestRowMetadataParity(unittest.TestCase):
    """Flag on vs flag off must be bit-identical on every output buffer."""

    def _inputs(self, rows, max_pages, num_reqs=8):
        dev = "cuda"
        torch.manual_seed(1234)
        stride = max_pages * FULL_PAGE
        req_to_token = torch.randint(
            0, 1 << 20, (num_reqs, stride), dtype=torch.int32, device=dev
        )
        # Draft-loop shape: one request, lengths growing by one per draft step.
        seq = torch.arange(30000, 30000 + rows, dtype=torch.int32, device=dev)
        req = torch.zeros(rows, dtype=torch.int32, device=dev)
        return seq, req, req_to_token

    def test_bit_exact_against_serial(self):
        for rows, max_pages in ((1, MAX_PAGES), (4, MAX_PAGES), (16, MAX_PAGES), (3, 129)):
            seq, req, r2t = self._inputs(rows, max_pages)
            a = _run_kernel(False, seq, req, r2t, max_pages)
            b = _run_kernel(True, seq, req, r2t, max_pages)
            for key in a:
                self.assertTrue(
                    torch.equal(a[key], b[key]),
                    f"{key} differs at rows={rows} max_pages={max_pages}",
                )

    def test_matches_torch_reference(self):
        rows, max_pages = 16, 1024
        seq, req, r2t = self._inputs(rows, max_pages)
        got = _run_kernel(True, seq, req, r2t, max_pages)
        want = _reference_row_metadata(seq.cpu(), req.cpu(), r2t.cpu(), max_pages)
        for key in want:
            self.assertTrue(
                torch.equal(got[key].cpu(), want[key]), f"{key} differs from reference"
            )


if __name__ == "__main__":
    if os.environ.get("SGLANG_QSA_META_PAGE_BOUND") == "1":
        raise SystemExit("unset SGLANG_QSA_META_PAGE_BOUND: it is not bit-exact")
    unittest.main(verbosity=2)
