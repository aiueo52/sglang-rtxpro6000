"""CPU reference and dispatch checks for SGLANG_QSA_RELATIVE_PREFILL.

Uses unittest like test_qsa_paged_prefill_cpu.py. No CUDA kernel is invoked;
the dispatch check uses a CPU tensor that only reports the CUDA flag.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention.qsa import mqa, qsa_indexer


class _CudaFlagged(torch.Tensor):
    is_cuda = property(lambda self: True)


class TestRelativePrefillCPU(unittest.TestCase):
    def test_reference_matches_absolute_row_windows(self):
        generator = torch.Generator().manual_seed(0)
        q = torch.randn(4, 4, 16, generator=generator)
        k = torch.randn(32, 1, 16, generator=generator)
        starts = torch.tensor([0, 0, 16, 16], dtype=torch.int32)
        ends = torch.tensor([0, 4, 32, 16], dtype=torch.int32)
        width = 16
        for scale in (None, 1):
            with self.subTest(scale=scale):
                full = mqa.torch_qsa_mqa_prefill(q, k, starts, ends, scale)
                relative = mqa.torch_qsa_mqa_prefill_relative(
                    q, k, starts, ends, width, scale
                )
                expected = torch.full((len(starts), width), -float("inf"))
                for row, (start, end) in enumerate(zip(starts, ends)):
                    start, end = int(start), int(end)
                    expected[row, : end - start] = full[row, start:end]
                torch.testing.assert_close(relative, expected, rtol=0, atol=0)
        empty = mqa.torch_qsa_mqa_prefill_relative(
            q, k[:0], torch.zeros_like(starts), torch.zeros_like(ends), width
        )
        self.assertEqual(empty.shape, expected.shape)
        self.assertTrue(torch.isneginf(empty).all().item())

    def test_key_split_bounds(self):
        levels = mqa._PREFILL_KEY_SPLIT_LEVELS
        block_n = 16
        device = torch.device("cpu")
        with mock.patch.object(mqa, "get_device_core_count", return_value=16):
            for rows in (0, 1, 4, 32, 100000):
                for tiles in (0, 1, 4 - 1, 4, 16, 32, 100000):
                    with self.subTest(rows=rows, tiles=tiles):
                        splits = mqa._prefill_key_splits(
                            rows, tiles * block_n, block_n, device
                        )
                        self.assertIn(splits, levels)
                        self.assertGreaterEqual(splits, min(levels))
                        self.assertLessEqual(splits, max(levels))
                        useful = max(1, tiles)
                        rounded_useful = next(
                            (level for level in levels if level >= useful),
                            max(levels),
                        )
                        self.assertLessEqual(splits, rounded_useful)
                        row_blocks = max(rows, 1)
                        wanted = (2 * 16 + row_blocks - 1) // row_blocks
                        self.assertGreaterEqual(
                            splits, min(wanted, useful, max(levels))
                        )
            self.assertEqual(mqa._prefill_key_splits(1, block_n, block_n, device), 1)
            self.assertEqual(
                mqa._prefill_key_splits(100000, 32 * block_n, block_n, device), 1
            )
        with mock.patch.object(mqa, "get_device_core_count", return_value=100000):
            self.assertEqual(
                mqa._prefill_key_splits(1, 100000 * block_n, block_n, device),
                max(levels),
            )

    def test_max_row_keys_host_sources(self):
        metadata = SimpleNamespace(
            compress_ratio=4, prefill_sequence_lengths_cpu=(16, 32)
        )
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor([4, 16]))
        self.assertEqual(qsa_indexer._prefill_max_row_keys(metadata, batch), 8)
        metadata.prefill_sequence_lengths_cpu = None
        self.assertEqual(qsa_indexer._prefill_max_row_keys(metadata, batch), 4)
        self.assertIsNone(
            qsa_indexer._prefill_max_row_keys(metadata, SimpleNamespace())
        )
        metadata.prefill_sequence_lengths_cpu = ()
        self.assertIsNone(qsa_indexer._prefill_max_row_keys(metadata, batch))

    def test_zero_switch_uses_existing_path(self):
        indexer = qsa_indexer.QSAIndexer.__new__(qsa_indexer.QSAIndexer)
        indexer.token_topk = 4
        indexer.compress_ratio = 1
        indexer.block_topk = 4
        q = torch.zeros(1, 4, 16).as_subclass(_CudaFlagged)
        keys = torch.zeros(16, 1, 16)
        starts = torch.tensor([4], dtype=torch.int32)
        ends = torch.tensor([16], dtype=torch.int32)
        positions = torch.tensor([16], dtype=torch.int32)
        lengths = positions + 1
        logits = torch.zeros(1, 16)
        blocks = torch.zeros(1, 4, dtype=torch.int32)
        selected = torch.ones_like(blocks)
        with (
            mock.patch.dict(os.environ, {"SGLANG_QSA_RELATIVE_PREFILL": "0"}),
            mock.patch.object(qsa_indexer, "HAS_TILELANG", True),
            mock.patch.object(qsa_indexer, "qsa_mqa_prefill", return_value=logits) as old,
            mock.patch.object(
                qsa_indexer, "qsa_mqa_prefill_relative",
                side_effect=AssertionError("relative path taken with switch off"),
            ) as relative,
            mock.patch.object(qsa_indexer, "qsa_fast_topk", return_value=blocks) as topk,
            mock.patch.object(
                qsa_indexer, "expand_qsa_block_indices", return_value=selected
            ),
        ):
            result = indexer.select_prefill_tokens(
                q, keys, starts, ends, positions, lengths, max_row_keys=16
            )
        old.assert_called_once()
        relative.assert_not_called()
        called_q, called_keys, called_starts, called_ends = old.call_args.args
        torch.testing.assert_close(called_q, q)
        self.assertIs(called_keys, keys)
        torch.testing.assert_close(called_starts, starts)
        torch.testing.assert_close(called_ends, ends)
        topk.assert_called_once()
        called_logits, topk_starts, topk_ends = topk.call_args.args
        self.assertIs(called_logits, logits)
        torch.testing.assert_close(topk_starts, starts)
        torch.testing.assert_close(topk_ends, ends)
        self.assertEqual(topk.call_args.kwargs, {"topk": indexer.block_topk})
        torch.testing.assert_close(result, selected)
        self.assertFalse(getattr(indexer, "_relative_prefill_reported", False))


if __name__ == "__main__":
    unittest.main(verbosity=1)
