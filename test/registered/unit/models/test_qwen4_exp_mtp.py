import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import qwen4_exp_mtp
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen4ExpMTPRuntimeContext(CustomTestCase):
    def test_lm_head_reads_configured_dp_flag(self):
        config = SimpleNamespace(
            hc_count=4,
            hidden_size=8,
            rms_norm_eps=1e-6,
            vocab_size=16,
        )
        parallel = SimpleNamespace(
            tp_size=1,
            config=SimpleNamespace(enable_dp_lm_head=True),
        )

        with (
            patch(
                "sglang.srt.models.qwen4_exp_mtp.get_parallel",
                return_value=parallel,
            ),
            patch("sglang.srt.models.qwen4_exp_mtp.get_pp_group"),
            patch("sglang.srt.models.qwen4_exp_mtp.Qwen4ExpModel"),
            patch("sglang.srt.models.qwen4_exp_mtp.ParallelLMHead") as lm_head,
            patch("sglang.srt.models.qwen4_exp_mtp.LogitsProcessor"),
            patch.object(
                Qwen4ExpForCausalLMMTP,
                "_init_mtp_input_fusion",
                return_value=None,
            ),
        ):
            Qwen4ExpForCausalLMMTP(config)

        self.assertTrue(lm_head.call_args.kwargs["use_attn_tp_group"])


class _Embedder(torch.nn.Module):
    """Stand-in for the draft's VocabParallelEmbedding at tp_size == 1."""

    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.num_embeddings = vocab
        self.weight = torch.nn.Parameter(torch.randn(vocab, hidden))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids.long(), self.weight)


class TestQwen4ExpMTPEmbedTable(CustomTestCase):
    """SGLANG_MTP_EMBED_TABLE: the token-only half of the entry fusion."""

    VOCAB = 48
    HIDDEN = 8
    HC = 4

    def _make_head(self) -> Qwen4ExpForCausalLMMTP:
        torch.manual_seed(0)
        config = SimpleNamespace(
            hc_count=self.HC,
            hidden_size=self.HIDDEN,
            rms_norm_eps=1e-6,
            vocab_size=self.VOCAB,
        )
        parallel = SimpleNamespace(
            tp_size=1,
            config=SimpleNamespace(enable_dp_lm_head=False),
        )
        with (
            patch(
                "sglang.srt.models.qwen4_exp_mtp.get_parallel",
                return_value=parallel,
            ),
            patch("sglang.srt.models.qwen4_exp_mtp.get_pp_group"),
            patch("sglang.srt.models.qwen4_exp_mtp.Qwen4ExpModel"),
            patch("sglang.srt.models.qwen4_exp_mtp.ParallelLMHead"),
            patch("sglang.srt.models.qwen4_exp_mtp.LogitsProcessor"),
        ):
            head = Qwen4ExpForCausalLMMTP(config)

        head.model = SimpleNamespace(embed_tokens=_Embedder(self.VOCAB, self.HIDDEN))
        for norm in (head.pre_fc_norm_embedding, head.pre_fc_norm_hidden):
            # Pin the pure-torch reference: this test runs on CPU tensors even
            # when the box happens to have a GPU.
            norm._forward_method = norm.forward_native
            with torch.no_grad():
                norm.weight.normal_(0, 0.1)
                norm.gemma_weight.copy_(norm.weight + 1.0)
        with torch.no_grad():
            head.fc_embedding.weight.normal_(0, 0.1)
            head.fc_hidden.weight.normal_(0, 0.1)
        return head

    @staticmethod
    def _build(head: Qwen4ExpForCausalLMMTP) -> None:
        with patch.dict(os.environ, {"SGLANG_MTP_EMBED_TABLE": "1"}):
            head._build_fc_embed_table()

    def _live_row(self, head: Qwen4ExpForCausalLMMTP, token_id: int) -> torch.Tensor:
        ids = torch.tensor([token_id])
        embeds = head.model.embed_tokens(ids)
        return head.fc_embedding(head.pre_fc_norm_embedding(embeds))[0]

    def test_table_rows_are_bit_exact(self):
        head = self._make_head()
        self._build(head)
        self.assertIsNotNone(head.fc_embed_table)
        self.assertEqual(tuple(head.fc_embed_table.shape), (self.VOCAB, self.HIDDEN))
        for token_id in range(self.VOCAB):
            self.assertTrue(
                torch.equal(
                    head.fc_embed_table[token_id], self._live_row(head, token_id)
                ),
                msg=f"table row {token_id} differs from the live computation",
            )

    def test_table_is_off_by_default(self):
        head = self._make_head()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_MTP_EMBED_TABLE", None)
            head._build_fc_embed_table()
        self.assertIsNone(head.fc_embed_table)

    def _fuse_both_ways(self, head, ids, hidden_states):
        live = head._fuse_residual_linear_shared(
            head.model.embed_tokens(ids), hidden_states
        )
        embed_proj = head._table_embed_proj(
            ids, SimpleNamespace(mm_input_embeds=None), None
        )
        tabled = head._fuse_residual_linear_shared(None, hidden_states, embed_proj)
        self.assertEqual(live.shape, tabled.shape)
        return live, tabled

    def test_fusion_matches_the_live_path_at_decode_width(self):
        """bs=1 draft steps: the table is built at that same M, so it is exact."""
        head = self._make_head()
        self._build(head)
        for token_id in (0, 5, self.VOCAB - 1):
            ids = torch.tensor([token_id])
            hidden_states = torch.randn(1, self.HC * self.HIDDEN)
            live, tabled = self._fuse_both_ways(head, ids, hidden_states)
            torch.testing.assert_close(live, tabled, rtol=0, atol=0)

    def test_fusion_matches_the_live_path_when_batched(self):
        """Wider batches (draft_extend, prefill) only match to GEMM tolerance.

        The live path's own result already moves with M — BLAS picks a
        different kernel — so the table cannot be bit-identical to every
        width at once; it is pinned to the M=1 decode width.
        """
        head = self._make_head()
        self._build(head)
        ids = torch.tensor([3, 17, 41])
        hidden_states = torch.randn(ids.numel(), self.HC * self.HIDDEN)
        live, tabled = self._fuse_both_ways(head, ids, hidden_states)
        torch.testing.assert_close(live, tabled, rtol=1e-5, atol=1e-6)

    def test_gather_is_shape_agnostic_and_yields_to_explicit_embeds(self):
        head = self._make_head()
        self._build(head)
        forward_batch = SimpleNamespace(mm_input_embeds=None)

        for shape in [(1,), (7,), (2, 5)]:
            ids = torch.randint(0, self.VOCAB, shape)
            proj = head._table_embed_proj(ids, forward_batch, None)
            self.assertEqual(tuple(proj.shape), (*shape, self.HIDDEN))

        ids = torch.tensor([1, 2])
        explicit = torch.zeros(2, self.HIDDEN)
        self.assertIsNone(head._table_embed_proj(ids, forward_batch, explicit))
        self.assertIsNone(
            head._table_embed_proj(
                ids, SimpleNamespace(mm_input_embeds=explicit), None
            )
        )

        head.fc_embed_table = None
        self.assertIsNone(head._table_embed_proj(ids, forward_batch, None))

    def test_build_is_skipped_when_the_fusion_has_no_token_only_term(self):
        head = self._make_head()
        head._mtp_input_fusion = head._fuse_standard
        self._build(head)
        self.assertIsNone(head.fc_embed_table)


if __name__ == "__main__":
    unittest.main()
