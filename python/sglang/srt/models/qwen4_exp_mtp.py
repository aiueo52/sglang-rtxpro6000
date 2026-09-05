"""Inference-only Qwen4-Exp MTP speculative decoding."""

import copy
import os
import logging
import time
from contextlib import ExitStack
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP, _mtp_quant_config
from sglang.srt.models.qwen4_exp import Qwen4ExpModel
from sglang.srt.runtime_context import get_model, get_parallel, get_spec
from sglang.srt.utils import add_prefix, is_npu, set_weight_attrs

logger = logging.getLogger(__name__)

# Route the MTP entry projections through the tuned skinny BF16 GEMV at
# decode sizes (SGLANG_MTP_FC_GEMV=1).
_MTP_FC_GEMV = os.environ.get("SGLANG_MTP_FC_GEMV", "0") == "1"


def _embed_table_enabled() -> bool:
    """SGLANG_MTP_EMBED_TABLE=1: precompute the token-only half of the entry.

    Read at table-build time rather than import time so a test (or a caller
    that sets the variable after import) sees its own value.
    """
    return os.environ.get("SGLANG_MTP_EMBED_TABLE", "0") == "1"


# Rows built per gather+norm batch. The projection itself is still issued one
# row at a time (see `_fc_embedding_row`), so this only bounds the staging
# buffer.
_EMBED_TABLE_CHUNK = int(os.environ.get("SGLANG_MTP_EMBED_TABLE_CHUNK", "8192"))
# Ids sampled after the build to prove table == live computation.
_EMBED_TABLE_CHECK = int(os.environ.get("SGLANG_MTP_EMBED_TABLE_CHECK", "256"))
# Device memory left free after the table; below this the build is skipped and
# the live path stays in place.
_EMBED_TABLE_RESERVE_GB = float(
    os.environ.get("SGLANG_MTP_EMBED_TABLE_RESERVE_GB", "2.0")
)


def _draft_vocab_weights_are_shared() -> bool:
    """Whether this draft will be handed the target's embedding / lm_head.

    Under EAGLE/NEXTN the worker replaces both tensors before the first forward
    (eagle_worker_v2.init_lm_head -> set_embed_and_head), and the Qwen4-Exp
    checkpoint ships no `mtp.embed_tokens` / `mtp.shared_head.head` rows, so
    the draft's own [vocab, hidden] pair is start-up peak for nothing. The
    speculative-algorithm check keeps a standalone load of this architecture
    (no draft worker to fill the tensors in) on the full-size path.
    """
    if not envs.SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS.get():
        return False
    return get_spec().speculative_algorithm is not None


def _build_with_placeholder_vocab_weight(build: Callable[[], nn.Module]) -> nn.Module:
    """Build a vocab-sized module without materialising its [vocab, hidden] table.

    Construction runs on the meta device, so the layout metadata (shard
    indices, num_embeddings, quant method) is real while the table costs
    nothing; the module then gets a 1-row placeholder on the ambient device
    for `set_embed_and_head` to `del` and replace.
    """
    with torch.device("meta"):
        module = build()
    meta_weight = module.weight
    placeholder = nn.Parameter(
        torch.empty(1, meta_weight.shape[1], dtype=meta_weight.dtype),
        requires_grad=False,
    )
    set_weight_attrs(
        placeholder,
        {"input_dim": 1, "output_dim": 0, "weight_loader": module.weight_loader},
    )
    module.register_parameter("weight", placeholder)
    return module


def _is_draft_vocab_weight(name: str) -> bool:
    """Checkpoint tensors that would target a placeholder (see above)."""
    if "mtp" not in name:
        return False
    return (
        name.endswith("embed_tokens.weight")
        or name.endswith("lm_head.weight")
        or "shared_head.head" in name
    )


class _Qwen4ExpDraftModel(Qwen4ExpModel):
    """Draft backbone whose input embedding is a placeholder.

    Only used when the target's table will be shared in; see
    `_draft_vocab_weights_are_shared`.
    """

    def _build_embed_tokens(self, config) -> nn.Module:
        return _build_with_placeholder_vocab_weight(
            lambda: super(_Qwen4ExpDraftModel, self)._build_embed_tokens(config)
        )


class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # Deepcopy so MTP-only mutations below don't leak into the main model.
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.full_attention_interval = 1
        config.ple_layer_ids = []

        quant_config = _mtp_quant_config(quant_config)

        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self._mtp_input_fusion = self._init_mtp_input_fusion(config)
        # [vocab, hidden] fc_embedding(pre_fc_norm_embedding(embed_tokens(id))),
        # built once the embedding table is in place (SGLANG_MTP_EMBED_TABLE=1).
        self.fc_embed_table = None

        self.skip_vocab_weights = _draft_vocab_weights_are_shared()
        model_cls = _Qwen4ExpDraftModel if self.skip_vocab_weights else Qwen4ExpModel
        self.model = model_cls(
            config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )

        def build_lm_head() -> nn.Module:
            return ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("model.shared_head.head", prefix),
                use_attn_tp_group=get_parallel().config.enable_dp_lm_head,
            )

        if self.skip_vocab_weights:
            self.lm_head = _build_with_placeholder_vocab_weight(build_lm_head)
            logger.info(
                "MTP draft embed_tokens / lm_head are placeholders; the target's "
                "tensors are shared in later (saved %.2f GB at load)",
                2 * config.vocab_size * config.hidden_size * 2 / (1 << 30),
            )
        else:
            self.lm_head = build_lm_head()
        self.logits_processor = LogitsProcessor(config)

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        if self.skip_vocab_weights:
            weights = (
                (name, weight)
                for name, weight in weights
                if not _is_draft_vocab_weight(name)
            )
            return super().load_weights(weights, is_mtp)
        result = super().load_weights(weights, is_mtp)
        # Standalone load: embed_tokens is final here (no `set_embed_and_head`).
        self._build_fc_embed_table()
        return result

    def _init_pre_fc_norms(self, config: PretrainedConfig) -> None:
        self.pre_fc_norm_embedding = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        hidden_norm_size = (
            self.hc_count * config.hidden_size
            if self.hc_count > 1
            else config.hidden_size
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            hidden_norm_size, eps=config.rms_norm_eps
        )

    def _init_linear_projections(self, config: PretrainedConfig) -> None:
        self.fc_embedding = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _init_standard_fusion(self, config: PretrainedConfig):
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self._init_pre_fc_norms(config)
        return self._fuse_standard

    def _init_mtp_input_fusion(self, config: PretrainedConfig):
        if self.hc_count <= 1:
            return self._init_standard_fusion(config)

        self._init_linear_projections(config)
        self._init_pre_fc_norms(config)
        return self._fuse_residual_linear_shared

    def _fuse_residual_linear_shared(
        self,
        input_embeds: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        embed_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Entry fusion: fc_embedding(norm(embed)) + fc_hidden(norm(hidden)).

        `embed_proj` is the first term already materialised (read straight out
        of the precomputed per-token table); when it is given, `input_embeds`
        is unused and both the embedding-side norm and GEMM are skipped.
        """
        if embed_proj is None:
            fused_tensors = (
                input_embeds,
                hidden_states,
                self.pre_fc_norm_embedding.weight,
                self.pre_fc_norm_hidden.weight,
                self.fc_embedding.weight,
                self.fc_hidden.weight,
            )
            if (
                envs.SGLANG_MTP_ENTRY_FUSED.get()
                and self.hc_count == 4
                and self.hidden_size == 2560
                and input_embeds.dim() == 2
                and hidden_states.shape == (input_embeds.shape[0], 4 * self.hidden_size)
                and 1 <= input_embeds.shape[0] <= 16
                and all(tensor.dtype == torch.bfloat16 for tensor in fused_tensors)
                and all(tensor.is_contiguous() for tensor in fused_tensors)
            ):
                from sglang.srt.layers.mtp_entry import mtp_entry_fused

                return mtp_entry_fused(
                    input_embeds,
                    hidden_states,
                    self.pre_fc_norm_embedding.weight,
                    self.pre_fc_norm_hidden.weight,
                    self.fc_embedding.weight,
                    self.fc_hidden.weight,
                    self.pre_fc_norm_embedding.variance_epsilon,
                )
            normed_embeds = self.pre_fc_norm_embedding(input_embeds)
        else:
            normed_embeds = None
        orig_shape = hidden_states.shape
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        decoder_view = hidden_states.view(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate_rows = embed_proj if normed_embeds is None else normed_embeds
        if (
            _MTP_FC_GEMV
            and gate_rows.dim() == 2
            and gate_rows.shape[0] * self.hc_count <= 16
            and gate_rows.dtype == torch.bfloat16
            and self.fc_embedding.weight.dtype == torch.bfloat16
            and self.fc_embedding.bias is None
            and self.fc_hidden.bias is None
        ):
            # Decode-size draft steps: cuBLAS picks slow kernels for these
            # [<=16, 2560] x [2560, 2560] BF16 GEMMs; use the tuned skinny GEMV.
            from sglang.srt.layers.quantization.w8a16_gemv import bf16_gemv

            if normed_embeds is not None:
                embed_proj = bf16_gemv(normed_embeds, self.fc_embedding.weight)
            rows = decoder_view.reshape(-1, self.hidden_size).contiguous()
            encoder_inputs = bf16_gemv(rows, self.fc_hidden.weight).view(
                decoder_view.shape
            )
        else:
            if normed_embeds is not None:
                embed_proj = self.fc_embedding(normed_embeds)
            encoder_inputs = self.fc_hidden(decoder_view)
        return (embed_proj.unsqueeze(-2) + encoder_inputs).view(orig_shape)

    def _fuse_standard(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        embed_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The single `fc` mixes both halves, so there is no token-only term to
        # precompute; `_build_fc_embed_table` never arms the table here.
        assert embed_proj is None
        input_embeds = self.pre_fc_norm_embedding(input_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        return self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

    # ------------------------------------------------------------------
    # R3: per-token table for the embedding side of the entry fusion
    # ------------------------------------------------------------------

    def _fc_embedding_row(
        self, normed_row: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """`fc_embedding` on a single normalised row, decode's kernel exactly.

        bs=1 draft steps hand `fc_embedding` an M=1 matrix and cuBLAS switches
        kernel (and therefore low-order bits) with M, so the table has to be
        filled one row at a time to come out bit-identical to the live path.
        """
        if (
            _MTP_FC_GEMV
            and normed_row.dtype == torch.bfloat16
            and self.fc_embedding.weight.dtype == torch.bfloat16
            and self.fc_embedding.bias is None
        ):
            from sglang.srt.layers.quantization.w8a16_gemv import bf16_gemv

            result = bf16_gemv(normed_row, self.fc_embedding.weight)
            return result if out is None else out.copy_(result)
        if out is None:
            return self.fc_embedding(normed_row)
        # torch.mm(x, W.t()) is the kernel F.linear dispatches to for a 2-D
        # input with no bias; the post-build check below proves it.
        return torch.mm(normed_row, self.fc_embedding.weight.t(), out=out)

    @torch.no_grad()
    def _build_fc_embed_table(self) -> None:
        """Materialise fc_embedding(pre_fc_norm_embedding(embed_tokens(id))).

        Called once the draft's embedding tensor is real (shared in from the
        target, or loaded from a standalone checkpoint). The term depends only
        on the token id, yet the live path recomputes it on every draft
        iteration; the table turns it into one gather.
        """
        self.fc_embed_table = None
        if not _embed_table_enabled():
            return
        # Bound methods are rebuilt per access, so compare the underlying
        # function rather than the bound object.
        fusion = getattr(self._mtp_input_fusion, "__func__", self._mtp_input_fusion)
        if fusion is not self.__class__._fuse_residual_linear_shared:
            logger.info("MTP embed table: entry fusion has no token-only term; off")
            return
        if self.tp_size != 1:
            logger.info("MTP embed table: unsupported at tp_size=%d; off", self.tp_size)
            return
        embed_tokens = getattr(self.model, "embed_tokens", None)
        weight = getattr(embed_tokens, "weight", None)
        if weight is None or weight.dim() != 2:
            logger.info("MTP embed table: no usable embedding tensor; off")
            return
        rows = min(
            int(weight.shape[0]),
            int(getattr(embed_tokens, "num_embeddings", weight.shape[0])),
        )
        hidden = self.hidden_size
        dtype = self.fc_embedding.weight.dtype
        device = weight.device
        nbytes = rows * hidden * dtype.itemsize
        if device.type == "cuda":
            free = torch.cuda.mem_get_info(device)[0]
            reserve = int(_EMBED_TABLE_RESERVE_GB * (1 << 30))
            if nbytes + reserve > free:
                logger.warning(
                    "MTP embed table: %.2f GB table + %.2f GB reserve exceeds "
                    "%.2f GB free; keeping the live path",
                    nbytes / (1 << 30),
                    reserve / (1 << 30),
                    free / (1 << 30),
                )
                return

        start_time = time.perf_counter()
        try:
            table = torch.empty((rows, hidden), dtype=dtype, device=device)
            chunk = max(1, _EMBED_TABLE_CHUNK)
            for lo in range(0, rows, chunk):
                hi = min(lo + chunk, rows)
                ids = torch.arange(lo, hi, device=device, dtype=torch.int64)
                # The norm is a per-row reduction, so it is bit-exact batched;
                # only the projection is sensitive to M.
                normed = self.pre_fc_norm_embedding(embed_tokens(ids))
                for i in range(hi - lo):
                    self._fc_embedding_row(
                        normed[i : i + 1], out=table[lo + i : lo + i + 1]
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            logger.warning("MTP embed table: OOM while building; keeping the live path")
            table = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return
        build_s = time.perf_counter() - start_time

        checked, exact, max_abs = self._check_fc_embed_table(table, embed_tokens)
        if checked and exact != checked:
            logger.warning(
                "MTP embed table: %d/%d sampled rows are not bit-identical to the "
                "live path (max|diff| %.3e); keeping the live path",
                checked - exact,
                checked,
                max_abs,
            )
            del table
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return
        self.fc_embed_table = table
        logger.info(
            "MTP embed table: %d x %d %s (%.2f GB) built in %.1fs via %s; "
            "sampled rows bit-exact %s, max|diff| %s; %.2f GB device memory free",
            rows,
            hidden,
            str(dtype).replace("torch.", ""),
            nbytes / (1 << 30),
            build_s,
            "bf16_gemv" if _MTP_FC_GEMV else "cuBLAS (M=1)",
            "n/a" if checked == 0 else f"{exact}/{checked}",
            "n/a" if max_abs is None else f"{max_abs:.3e}",
            (
                torch.cuda.mem_get_info(device)[0] / (1 << 30)
                if device.type == "cuda"
                else float("nan")
            ),
        )

    @torch.no_grad()
    def _check_fc_embed_table(
        self, table: torch.Tensor, embed_tokens: nn.Module
    ) -> Tuple[int, int, Optional[float]]:
        """Compare a sample of table rows against the live M=1 computation.

        Returns (rows checked, rows bit-identical, max abs difference).
        """
        n = min(_EMBED_TABLE_CHECK, table.shape[0])
        if n <= 0:
            return 0, 0, None
        device = table.device
        ids = torch.randint(0, table.shape[0], (n,), device=device, dtype=torch.int64)
        live = torch.empty((n, table.shape[1]), dtype=table.dtype, device=device)
        for i in range(n):
            row_ids = ids[i : i + 1]
            normed = self.pre_fc_norm_embedding(embed_tokens(row_ids))
            live[i] = self._fc_embedding_row(normed)[0]
        got = table.index_select(0, ids)
        exact = int((got == live).all(dim=1).sum().item())
        max_abs = (got.float() - live.float()).abs().max().item()
        return n, exact, max_abs

    def _table_embed_proj(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """The entry's embedding-side term straight out of the table.

        Shape-agnostic (decode, draft_extend and prefill all just gather), and
        CUDA-graph safe: one `index_select` with no host sync and no branch on
        device data.
        """
        table = self.fc_embed_table
        if table is None:
            return None
        if input_embeds is not None or forward_batch.mm_input_embeds is not None:
            return None
        ids = input_ids
        if ids.dtype not in (torch.int32, torch.int64):
            ids = ids.long()
        out = torch.index_select(table, 0, ids.reshape(-1))
        if ids.dim() != 1:
            out = out.view(*ids.shape, table.shape[1])
        return out

    def set_embed_and_head(self, embed, head):
        super().set_embed_and_head(embed, head)
        self._build_fc_embed_table()

    def _npu_quant_context(self):
        exit_stack = ExitStack()
        if (
            is_npu()
            and self.quant_config is None
            and get_model().quantization is not None
        ):
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )
        return exit_stack

    def _prepare_input_embeds(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert input_embeds is None
        input_embeds = forward_batch.mm_input_embeds
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            assert input_embeds is not None
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            input_embeds[last_indices] = self.model.embed_tokens(
                input_ids[last_indices]
            )
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)
        return input_embeds

    def _set_hc_logits_hidden_states(
        self,
        logits_output,
        hc_hidden_states: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> None:
        if hc_hidden_states is None:
            return

        # EAGLE v2 stores one hidden state per request in the future map.
        # When draft extend emits a token-shaped HC tensor, keep only the
        # last token per request so the overlap cache sees [bs, hidden].
        if (
            not forward_batch.forward_mode.is_draft_extend_v2()
            and forward_batch.extend_seq_lens is not None
            and hc_hidden_states.shape[0] != forward_batch.extend_seq_lens.shape[0]
        ):
            last_index = (
                torch.cumsum(forward_batch.extend_seq_lens.to(torch.int64), dim=0) - 1
            )
            hc_hidden_states = hc_hidden_states[last_index]

        assert hc_hidden_states.shape[-1] == self.hc_count * self.hidden_size
        logits_output.hidden_states = hc_hidden_states

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        with self._npu_quant_context():
            embed_proj = self._table_embed_proj(input_ids, forward_batch, input_embeds)
            if embed_proj is None:
                input_embeds = self._prepare_input_embeds(
                    input_ids, forward_batch, input_embeds
                )
            hidden_states = forward_batch.spec_info.hidden_states
            if not forward_batch.forward_mode.is_idle():
                hidden_states = self._mtp_input_fusion(
                    input_embeds, hidden_states, embed_proj
                )

            with get_global_expert_distribution_recorder().disable_this_region():
                model_output = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )

            hc_hidden_states = None
            if isinstance(model_output, tuple):
                hidden_states, hc_hidden_states = model_output
            else:
                hidden_states = model_output

        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )
        self._set_hc_logits_hidden_states(
            logits_output, hc_hidden_states, forward_batch
        )
        return logits_output


EntryClass = [Qwen4ExpForCausalLMMTP]
