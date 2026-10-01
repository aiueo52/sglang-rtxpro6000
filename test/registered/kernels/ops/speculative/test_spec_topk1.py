from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.spec_tree import (
    sgl_build_tree_kernel_efficient_triton,
)
from sglang.kernels.ops.speculative.topk1 import (
    build_chain_tree_topk1,
    draft_extend_select_topk1,
    draft_topk1_postprocess,
)
from sglang.test.test_utils import CustomTestCase


def _make_logits_with_unique_argmax(
    batch_size: int,
    vocab_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        (batch_size, vocab_size), dtype=dtype, device=device, generator=g
    )
    expected_index = (
        torch.arange(batch_size, dtype=torch.long, device=device) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, expected_index[:, None], 1000.0)
    return logits, expected_index[:, None]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestSpecTopk1Triton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_draft_topk1_postprocess_matches_argmax_and_position_add(self):
        configs = [
            (1, 127, torch.float32),
            (4, 8192, torch.float16),
            (7, 8193, torch.bfloat16),
            (3, 50000, torch.float32),
        ]
        for batch_size, vocab_size, dtype in configs:
            with self.subTest(
                batch_size=batch_size, vocab_size=vocab_size, dtype=dtype
            ):
                logits, expected_index = _make_logits_with_unique_argmax(
                    batch_size,
                    vocab_size,
                    dtype=dtype,
                    device=self.device,
                    seed=vocab_size,
                )
                positions = torch.arange(
                    batch_size, dtype=torch.long, device=self.device
                )
                expected_positions = positions + 1

                topk_p, topk_index = draft_topk1_postprocess(logits, positions)

                torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
                torch.testing.assert_close(
                    topk_p,
                    torch.ones(
                        (batch_size, 1), dtype=torch.float32, device=self.device
                    ),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    positions, expected_positions, rtol=0, atol=0
                )

    def test_draft_topk1_postprocess_can_write_draft_token_column(self):
        batch_size = 17
        # Multi-split vocab so the fused write composes with the split reduction.
        vocab_size = 50000
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=0,
        )
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        backing = torch.full((batch_size, 5), -1, dtype=torch.long, device=self.device)
        draft_tokens = backing[:, 1:4]

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=2
        )

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        # Exactly one backing column is written; both neighbors stay untouched.
        expected_backing = torch.full_like(backing, -1)
        expected_backing[:, 3] = expected_index[:, 0]
        torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )

    def test_row_strided_logits_view_matches_argmax(self):
        batch_size = 5
        vocab_size = 8193
        # Poison the padding columns: if the kernel used the dense vocab width
        # as the row stride it would read them and pick the wrong index.
        backing = torch.full(
            (batch_size, vocab_size + 64),
            2000.0,
            dtype=torch.float32,
            device=self.device,
        )
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=1,
        )
        backing[:, :vocab_size] = logits
        strided_logits = backing[:, :vocab_size]
        self.assertFalse(strided_logits.is_contiguous())
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(strided_logits, positions)

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )

    def test_empty_batch(self):
        logits = torch.empty((0, 1024), dtype=torch.float32, device=self.device)
        positions = torch.empty((0,), dtype=torch.long, device=self.device)
        draft_tokens = torch.empty((0, 3), dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=1
        )

        self.assertEqual(topk_p.shape, (0, 1))
        self.assertEqual(topk_index.shape, (0, 1))
        self.assertEqual(draft_tokens.numel(), 0)

    def test_non_contiguous_inputs_raise(self):
        logits = torch.empty((16, 4), dtype=torch.float32, device=self.device).t()
        positions = torch.arange(8, dtype=torch.long, device=self.device)[::2]

        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(
                logits, torch.empty(4, dtype=torch.long, device=self.device)
            )
        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(torch.empty((4, 16), device=self.device), positions)

    def test_draft_extend_select_matches_torch_tail(self):
        # Bit-identical to logits[sel].argmax / hidden[sel] (the torch tail it
        # replaces), incl. first-max ties that span vocab splits.
        bs, width, vocab_size, hidden_size = 3, 8, 49152, 640
        g = torch.Generator(device=self.device).manual_seed(0)
        logits = torch.randn(
            (bs * width, vocab_size), device=self.device, generator=g
        ).bfloat16()
        logits = logits.float()
        logits[:, [100, 9000, 40000]] = 10.0
        hidden = torch.randn(
            (bs * width, hidden_size), device=self.device, generator=g
        ).bfloat16()
        select_index = torch.tensor([5, 8, 23], device=self.device)

        topk_p, topk_index, hidden_out, rows, _ = draft_extend_select_topk1(
            next_token_logits=logits,
            select_index=select_index,
            hidden_states=hidden,
            write_rows=True,
        )
        _, _, _, _, conf = draft_extend_select_topk1(
            next_token_logits=logits, select_index=select_index, write_prob=True
        )

        expected_rows = logits[select_index]
        torch.testing.assert_close(
            topk_index,
            torch.argmax(expected_rows, dim=-1, keepdim=True),
            rtol=0,
            atol=0,
        )
        self.assertEqual(topk_index.view(-1).tolist(), [100] * bs)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        torch.testing.assert_close(hidden_out, hidden[select_index], rtol=0, atol=0)
        torch.testing.assert_close(rows, expected_rows, rtol=0, atol=0)
        expected_conf = (
            expected_rows.amax(dim=-1) - torch.logsumexp(expected_rows, dim=-1)
        ).exp()
        torch.testing.assert_close(conf, expected_conf, rtol=1e-5, atol=0)

        # Same rows from accept_lens: i * width + accept_lens[i] - 1.
        accept_lens = torch.tensor([6, 1, 8], dtype=torch.int32, device=self.device)
        _, index_from_accept, hidden_from_accept, _, _ = draft_extend_select_topk1(
            next_token_logits=logits,
            hidden_states=hidden,
            accept_lens=accept_lens,
            num_tokens_per_req=width,
        )
        torch.testing.assert_close(index_from_accept, topk_index, rtol=0, atol=0)
        torch.testing.assert_close(hidden_from_accept, hidden_out, rtol=0, atol=0)

    def test_chain_tree_matches_reference_tree_build(self):
        g = torch.Generator(device=self.device).manual_seed(5)
        for bs, steps, full_mask in ((1, 3, True), (3, 7, True), (2, 15, False)):
            T = steps + 1
            seq_lens = torch.randint(
                3, 400, (bs,), generator=g, device=self.device, dtype=torch.int64
            )
            numel = T * T * bs + (int(seq_lens.sum()) * T if full_mask else 0)
            init = torch.randint(0, 2, (numel,), generator=g, device=self.device).bool()
            parents = torch.arange(
                -1, steps - 1, dtype=torch.long, device=self.device
            ).repeat(bs, 1)
            scores = torch.arange(steps, dtype=torch.long, device=self.device).repeat(
                bs, 1
            )
            bonus = torch.randint(
                0, 1000, (bs,), generator=g, device=self.device, dtype=torch.int32
            )
            draft = torch.randint(
                0, 1000, (bs, steps), generator=g, device=self.device, dtype=torch.int64
            )
            mask_ref = init.clone()
            retrieve_ref = torch.full(
                (3, bs, T), -1, dtype=torch.long, device=self.device
            )
            positions_ref = torch.empty((bs * T,), dtype=torch.long, device=self.device)
            sgl_build_tree_kernel_efficient_triton[(bs,)](
                parents,
                scores,
                seq_lens,
                torch.cumsum(seq_lens, dim=0) - seq_lens,
                mask_ref,
                positions_ref,
                retrieve_ref[0],
                retrieve_ref[1],
                retrieve_ref[2],
                topk=1,
                depth=steps,
                draft_token_num=T,
                tree_mask_mode=0 if full_mask else 1,
                batch_size=bs,
                parent_list_stride=parents.stride(0),
                selected_index_stride=scores.stride(0),
            )
            mask = init.clone()
            positions, index, next_token, next_sibling, tokens = build_chain_tree_topk1(
                bonus_tokens=bonus,
                draft_tokens=draft,
                seq_lens=seq_lens,
                tree_mask=mask,
                full_mask=full_mask,
            )
            self.assertTrue(torch.equal(mask, mask_ref))
            self.assertTrue(torch.equal(positions, positions_ref))
            self.assertTrue(torch.equal(index, retrieve_ref[0]))
            self.assertTrue(torch.equal(next_token, retrieve_ref[1]))
            self.assertTrue(torch.equal(next_sibling, retrieve_ref[2]))
            self.assertTrue(
                torch.equal(tokens, torch.cat((bonus[:, None], draft), dim=1).flatten())
            )


if __name__ == "__main__":
    unittest.main()
