"""CPU invariants for the QSA per-request pending index-key ring."""

import random
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.metadata import (
    build_group_ring_slots,
    build_pending_ring_slots,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
)
from sglang.srt.mem_cache.qsa_kv_pool import (
    QSATokenToKVPool,
    get_qsa_pending_ring_size,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RATIO = 4


def _write_slots(state, slots, requests, positions):
    for slot, request, position in zip(
        slots.tolist(), requests.tolist(), positions.tolist()
    ):
        state[int(slot)] = (int(request), int(position))


def _seed_pending_tail(state, request, length, ring_size):
    positions = torch.arange(length - length % RATIO, length, dtype=torch.long)
    if positions.numel() == 0:
        return
    requests = torch.full_like(positions, request)
    slots = build_pending_ring_slots(
        token_to_batch_idx=torch.arange(positions.numel()),
        req_pool_indices=requests,
        sequence_lengths=positions + 1,
        logical_positions=positions,
        compress_ratio=RATIO,
        ring_size=ring_size,
        is_extend=False,
    )
    _write_slots(state, slots, requests, positions)


def _verify_forward(state, request_lengths, request_slots, window, ring_size):
    positions = torch.cat(
        [
            torch.arange(length, length + window, dtype=torch.long)
            for length in request_lengths
        ]
    )
    requests = torch.repeat_interleave(
        torch.tensor(request_slots, dtype=torch.long), window
    )
    rows = torch.arange(positions.numel(), dtype=torch.long)
    state_slots = build_pending_ring_slots(
        token_to_batch_idx=rows,
        req_pool_indices=requests,
        sequence_lengths=positions + 1,
        logical_positions=positions,
        compress_ratio=RATIO,
        ring_size=ring_size,
        is_extend=False,
    )
    _write_slots(state, state_slots, requests, positions)

    group_rows = rows[(positions + 1) % RATIO == 0]
    group_ends = positions.index_select(0, group_rows)
    group_slots = build_group_ring_slots(
        req_pool_indices=requests,
        group_end_positions=group_ends,
        sequence_ids=group_rows,
        compress_ratio=RATIO,
        ring_size=ring_size,
    )
    for group_idx, row in enumerate(group_rows.tolist()):
        request = int(requests[row])
        group_end = int(positions[row])
        actual = [state[int(slot)] for slot in group_slots[group_idx].tolist()]
        expected = [
            (request, position)
            for position in range(group_end - RATIO + 1, group_end + 1)
        ]
        if actual != expected:
            raise AssertionError(
                f"group ending at {group_end} read {actual}, expected {expected}"
            )


def _draft_extend(state, old_lengths, accepted, request_slots, ring_size):
    positions = torch.cat(
        [
            torch.arange(length, length + count, dtype=torch.long)
            for length, count in zip(old_lengths, accepted)
        ]
    )
    if positions.numel() == 0:
        return
    token_to_batch = torch.repeat_interleave(
        torch.arange(len(request_slots), dtype=torch.long),
        torch.tensor(accepted, dtype=torch.long),
    )
    lengths = torch.tensor(
        [length + count for length, count in zip(old_lengths, accepted)],
        dtype=torch.long,
    )
    request_tensor = torch.tensor(request_slots, dtype=torch.long)
    slots = build_pending_ring_slots(
        token_to_batch_idx=token_to_batch,
        req_pool_indices=request_tensor,
        sequence_lengths=lengths,
        logical_positions=positions,
        compress_ratio=RATIO,
        ring_size=ring_size,
        is_extend=True,
    )
    requests = request_tensor.index_select(0, token_to_batch)
    _write_slots(state, slots, requests, positions)


class TestQsaPendingRing(CustomTestCase):
    def test_ring_size_formula(self):
        for window in (0, 1, 2, 4, 8, 16):
            ring_size = get_qsa_pending_ring_size(RATIO, window)
            self.assertGreaterEqual(ring_size, RATIO + max(1, window) - 1)
            self.assertEqual(ring_size % RATIO, 0)
            if window <= 1:
                self.assertEqual(ring_size, RATIO)

    def test_verify_window_slots_are_distinct_at_every_ring_offset(self):
        for window in (1, 2, 4, 8, 16):
            ring_size = get_qsa_pending_ring_size(RATIO, window)
            for start in range(ring_size):
                positions = torch.arange(start, start + window)
                slots = build_pending_ring_slots(
                    token_to_batch_idx=torch.arange(window),
                    req_pool_indices=torch.full((window,), 3),
                    sequence_lengths=positions + 1,
                    logical_positions=positions,
                    compress_ratio=RATIO,
                    ring_size=ring_size,
                    is_extend=False,
                )
                self.assertEqual(torch.unique(slots).numel(), window)

    def test_randomized_verify_accept_extend_round_trip(self):
        rng = random.Random(20260902)
        for batch_size in (1, 3):
            request_slots = [2, 5, 9][:batch_size]
            for window in (2, 4, 8, 16):
                ring_size = get_qsa_pending_ring_size(RATIO, window)
                lengths = [rng.randrange(1, 64) for _ in range(batch_size)]
                state = {}
                for request, length in zip(request_slots, lengths):
                    _seed_pending_tail(state, request, length, ring_size)
                for _ in range(2000):
                    _verify_forward(state, lengths, request_slots, window, ring_size)
                    accepted = [rng.randint(1, window) for _ in range(batch_size)]
                    old_lengths = lengths
                    _draft_extend(
                        state,
                        old_lengths,
                        accepted,
                        request_slots,
                        ring_size,
                    )
                    lengths = [
                        length + count for length, count in zip(old_lengths, accepted)
                    ]

    def test_one_row_below_the_bound_aliases_a_group_member(self):
        window = 8
        ring_size = RATIO + window - 2
        start = 7  # The first verify token completes a compression group.
        state = {}
        _seed_pending_tail(state, 2, start, ring_size)
        with self.assertRaises(AssertionError):
            _verify_forward(state, [start], [2], window, ring_size)

    def test_chain_speculation_limit_matches_ring_capacity(self):
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.compress_ratio = RATIO
        backend.token_to_kv_pool = SimpleNamespace(qsa_ring_size=20)
        mode = SimpleNamespace(is_target_verify=lambda: True)
        backend._require_chain_speculation(
            mode, SimpleNamespace(topk=1, draft_token_num=17)
        )
        with self.assertRaisesRegex(NotImplementedError, "18.*17"):
            backend._require_chain_speculation(
                mode, SimpleNamespace(topk=1, draft_token_num=18)
            )

    def test_host_speculative_padding_uses_reserved_request_slot(self):
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([7, 8], dtype=torch.int32),
            extend_seq_lens=torch.tensor([1, 0], dtype=torch.int32),
        )
        row_requests = QwenSparseAttnBackend._speculative_row_to_request(
            forward_batch, 2
        )
        self.assertEqual(row_requests.tolist(), [7, 0])

        mode = SimpleNamespace(is_target_verify=lambda: False)
        spec_info = SimpleNamespace(extend_seq_lens_cpu=torch.tensor([1, 0]))
        _, graph_requests, _ = QwenSparseAttnBackend._graph_speculative_layout(
            bs=2,
            num_tokens=2,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens_cpu=torch.tensor([9, 12]),
            forward_mode=mode,
            spec_info=spec_info,
        )
        self.assertEqual(graph_requests.tolist(), [7, 0])

    def test_extend_dump_rows_stay_in_reserved_ring(self):
        ring_size = 12
        positions = torch.arange(8)
        slots = build_pending_ring_slots(
            token_to_batch_idx=torch.zeros(8, dtype=torch.long),
            req_pool_indices=torch.tensor([5]),
            sequence_lengths=torch.tensor([8]),
            logical_positions=positions,
            compress_ratio=RATIO,
            ring_size=ring_size,
            is_extend=True,
        )
        self.assertTrue(bool(((0 <= slots) & (slots < ring_size)).all()))

    def test_pool_allocates_ring_size_per_request_slot(self):
        pool = QSATokenToKVPool(
            size=8,
            dtype=torch.bfloat16,
            page_size=4,
            head_num=1,
            head_dim=8,
            full_attention_layer_ids=[0],
            device="cpu",
            mamba_pool=None,
            qsa_index_kv_heads=1,
            qsa_index_head_dim=8,
            qsa_compress_ratio=RATIO,
            qsa_token_topk=4,
            num_request_slots=3,
            qsa_ring_size=12,
        )
        self.assertEqual(pool.qsa_ring_size, 12)
        self.assertEqual(pool.qsa_key_state_buffer_pool[0].shape[0], 36)
        self.assertEqual(pool.qsa_rope_position_buffer.shape[0], 36)

        with self.assertRaisesRegex(ValueError, "multiple.*at least"):
            QSATokenToKVPool(
                size=8,
                dtype=torch.bfloat16,
                page_size=4,
                head_num=1,
                head_dim=8,
                full_attention_layer_ids=[0],
                device="cpu",
                mamba_pool=None,
                qsa_index_kv_heads=1,
                qsa_index_head_dim=8,
                qsa_compress_ratio=RATIO,
                qsa_token_topk=4,
                num_request_slots=3,
                qsa_ring_size=10,
            )


if __name__ == "__main__":
    unittest.main()
