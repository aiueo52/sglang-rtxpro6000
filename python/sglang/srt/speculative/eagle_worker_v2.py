# Modified by aiueo52 for flash-next-fast (2026); see MODIFICATIONS.md.

import contextlib
import logging
import os
import time
from dataclasses import replace
from typing import List, Optional

import torch

from sglang.kernels.ops.speculative.sparse_rs import (
    rs_draft_extend_proposal_sparse,
    rs_draft_proposal_sparse,
)
from sglang.kernels.ops.speculative.topk1 import (
    draft_extend_select_topk1,
    draft_topk1_postprocess,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.attention.qsa.config import is_qwen_qsa
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_confidence import (
    ChainTracer,
    chain_trace_enabled,
    top1_prob,
)
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
    adaptive_target_graph_warmup,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    default_tree_mask_mode,
    get_draft_recurrent_hidden_state_spec,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_for_draft,
    prepare_for_draft_extend,
    run_eagle_verify,
)
from sglang.srt.speculative.mtp_recur_head import is_recur_head_mode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.vt_policy import vt_grammar_enabled, vt_tau
from sglang.srt.speculative.dl_policy import dl_hostw_enabled, dl_tau
from sglang.srt.speculative.hostw_policy import (
    read_max_length,
    select_verify_steps,
    trim_draft_payload,
    validate_hostw_config,
)
from sglang.srt.speculative.spec_utils import (
    draft_tp_context,
    fast_sample,
    get_plan_stream,
    load_token_map,
    RS_DRAFT_TOPK,
    RS_DRAFT_TEMP_SCALE,
    RS_DRAFT_ONEHOT_ABOVE,
    RS_GREEDY_FAST,
    RS_BLOCK_VERIFY,
    SPEC_MIN_P,
    SPEC_SPARSE_RS,
    renorm_draft_probs,
    sample_draft_proposal,
    sample_draft_proposal_truncated,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    log_info_on_rank0,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_cpu = is_cpu()
_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()
_is_xpu = is_xpu()


logger = logging.getLogger(__name__)

#: N1 stage A -- serve the draft's hot-vocab lm_head as NVFP4 instead of the FP8 row
#: slice of the target head. Off by default; see _install_nvfp4_draft_head.
_MTP_LMHEAD_NVFP4 = os.environ.get("SGLANG_MTP_LMHEAD_NVFP4", "0").lower() in (
    "1", "true", "yes", "on"
)
#: N1 stage B -- serve the TARGET lm_head as NVFP4. Unlike stage A this changes the
#: model's own output distribution, so it carries the strict quality gate.
_LMHEAD_NVFP4 = os.environ.get("SGLANG_LMHEAD_NVFP4", "0").lower() in (
    "1", "true", "yes", "on"
)


def _qsa_index_share_requested(hf_config) -> bool:
    """--json-model-override-args writes top-level hf_config attributes, while
    checkpoint configs carry the flag on the nested text_config; read both."""
    text_config = getattr(hf_config, "text_config", hf_config)
    return bool(
        getattr(
            text_config,
            "index_share_for_mtp_iteration",
            getattr(hf_config, "index_share_for_mtp_iteration", False),
        )
    )


def _validate_hostw_startup(server_args, *hf_configs):
    if not dl_hostw_enabled():
        return
    decode = get_exec().graph.cuda_graph_config.decode
    validate_hostw_config(
        max_running_requests=get_schedule().max_running_requests,
        decode_max_bs=decode.max_bs,
        capture_bs=decode.bs,
        disable_padding=server_args.disable_cuda_graph_padding,
        index_sharing=any(
            _qsa_index_share_requested(config)
            or bool(getattr(config, "index_share_for_mtp_iteration", False))
            for config in hf_configs
        ),
    )


def _validate_vt_startup():
    max_bs = get_schedule().max_running_requests
    if max_bs == 1:
        return
    decode = get_exec().graph.cuda_graph_config.decode
    if (
        not max_bs
        or decode.backend != Backend.FULL
        or not set(range(1, max_bs + 1)).issubset(decode.bs or ())
    ):
        raise ValueError(
            "VT with concurrent requests requires full CUDA graphs capturing every "
            "batch size from 1 through --max-running-requests"
        )


class EagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self.target_worker = target_worker

        # Args for easy access
        self.device = get_device().device
        self.sparse_rs = SPEC_SPARSE_RS
        if self.sparse_rs:
            if not get_spec().speculative_use_rejection_sampling or RS_DRAFT_TOPK <= 0:
                raise ValueError("SGLANG_OPT_SPEC_SPARSE_RS requires rejection sampling and SGLANG_RS_DRAFT_TOPK > 0")
            if server_args.enable_multi_layer_eagle:
                raise ValueError("SGLANG_OPT_SPEC_SPARSE_RS does not support multi-layer EAGLE")
            logger.info(f"SGLANG_OPT_SPEC_SPARSE_RS on: sparse chain RS, draft support K={RS_DRAFT_TOPK}, temp_scale={RS_DRAFT_TEMP_SCALE}, onehot_above={RS_DRAFT_ONEHOT_ABOVE}, greedy_fast={RS_GREEDY_FAST}, block_verify={RS_BLOCK_VERIFY}")
        self.topk = get_spec().speculative_eagle_topk
        self._vt_enabled = vt_tau() > 0
        # Grammar-constrained requests run on the VT path unless SGLANG_RND_VT_GRAMMAR=0 (0/1 only, checked here at start).
        self._vt_grammar = vt_grammar_enabled() if self._vt_enabled else True
        self._dl_tau = dl_tau()
        self._dl_enabled = self._dl_tau > 0
        _validate_hostw_startup(server_args, target_worker.model_runner.model_config.hf_config)
        if self._vt_enabled and (self.topk != 1 or not _is_cuda):
            raise ValueError("VT requires CUDA and speculative_eagle_topk=1")
        if self._vt_enabled:
            _validate_vt_startup()
        if get_spec().speculative_use_rejection_sampling:
            assert self.topk == 1, "Chain speculative sampling supports only topk=1"
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        if self._vt_enabled:
            hf_config = target_worker.model_runner.model_config.hf_config
            architectures = (
                getattr(hf_config, "architectures", None)
                or getattr(getattr(hf_config, "text_config", None), "architectures", None)
                or ()
            )
            if (
                "Qwen4ExpForConditionalGeneration" not in architectures
                or self.speculative_algorithm != SpeculativeAlgorithm.EAGLE
                or server_args.enable_multi_layer_eagle
                or getattr(server_args, "enable_lora", False)
                or get_parallel().tp_size != 1
                or get_parallel().pp_size != 1
                or get_parallel().attn_dp_size != 1
                or server_args.dp_size != 1
            ):
                raise ValueError(
                    "VT requires single-GPU Qwen4ExpForConditionalGeneration, "
                    "single-layer EAGLE/MTP, and no LoRA"
                )

        self._rebuild_topk1_chain_buffers()
        if self._vt_enabled:
            # Sized from the scheduler's shared request pool before capture.
            self._hostw_position0 = None
        if self._dl_enabled:
            from sglang.srt.speculative.dl_graph import require_conditional_support
            from sglang.srt.speculative.vt_kernels import length_buffer

            require_conditional_support()
            if (
                get_exec().graph.cuda_graph_config.decode.backend != Backend.FULL
                or get_exec().features.enable_memory_saver
            ):
                raise ValueError("DL requires full CUDA graphs without memory saver")
            self._dl_lengths = length_buffer(self._vt_conf_buf.device)
            self._dl_product = torch.ones_like(self._dl_lengths, dtype=torch.float32)
            self._dl_predicate = torch.zeros(1, dtype=torch.bool, device=self.device)
            self._dl_forwards = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Load draft model weights only.
        if (
            get_parallel().config.enable_dp_attention
            and self.speculative_algorithm.is_eagle3()
        ):
            ctx = draft_tp_context(get_parallel().attn_tp_group)
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context(), draft_model_build_scope():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                # spec workers don't support pipeline parallelism
                ps=replace(ps, pp_rank=0),
                nccl_port=nccl_port,
                is_draft_worker=True,
                # The draft runs at absolute target positions.
                context_length=target_worker.model_runner.model_config.context_len,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self._init_dsa_index_share_state()
        # Eager draft-extend seed buffer (graph paths use their own static ones).
        self.dsa_extend_topk_buf: Optional[torch.Tensor] = None
        self.draft_tp_context = (
            draft_tp_context
            if get_parallel().config.enable_dp_attention
            else empty_context
        )
        self.tree_mask_mode = default_tree_mask_mode()

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)
        # SGLANG_OPT_DRAFT_TAIL: fused row select + argmax for the topk=1 tail of
        # _draft_extend_for_decode; FUSED_CONF also takes p0 from that pass.
        self._draft_tail_select = (
            envs.SGLANG_OPT_DRAFT_TAIL.get()
            and self.topk == 1
            and _is_cuda
            and not get_spec().speculative_use_rejection_sampling
        )
        self._draft_tail_fused_conf = (
            self._draft_tail_select and envs.SGLANG_OPT_DRAFT_TAIL_FUSED_CONF.get()
        )
        # Chain tree build in the draft phase epilogue (build_tree_kernel_efficient).
        self._draft_tail_chain = envs.SGLANG_OPT_DRAFT_TAIL.get() and self.topk == 1
        # SGLANG_OPT_RS_EXTEND_TAIL: the sparse-RS draft-extend tail reads its rows
        # in-kernel; FUSED_CONF also takes p0 from that pass.
        self._rs_extend_tail = self.sparse_rs and envs.SGLANG_OPT_RS_EXTEND_TAIL.get()
        self._draft_extend_lens_fused = envs.SGLANG_OPT_DRAFT_EXTEND_LENS.get()
        if self._draft_extend_lens_fused:
            logger.info("SGLANG_OPT_DRAFT_EXTEND_LENS on: draft-extend lengths fused")
        self._rs_extend_fused_conf = (
            self._rs_extend_tail and envs.SGLANG_OPT_RS_EXTEND_FUSED_CONF.get()
        )
        if envs.SGLANG_OPT_RS_EXTEND_TAIL.get():
            logger.info(
                "SGLANG_OPT_RS_EXTEND_TAIL on: active %s, fused conf %s",
                self._rs_extend_tail,
                self._rs_extend_fused_conf,
            )
        if envs.SGLANG_OPT_DRAFT_TAIL.get():
            logger.info(
                "SGLANG_OPT_DRAFT_TAIL on: draft-extend select %s, fused conf %s, "
                "chain tree %s",
                self._draft_tail_select,
                self._draft_tail_fused_conf,
                self._draft_tail_chain,
            )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Allocate draft KV cache pools (called by scheduler)."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        if self._vt_enabled and get_schedule().max_running_requests > 1:
            self._hostw_position0 = torch.zeros(
                req_to_token_pool.req_to_token.shape[0],
                dtype=torch.float32, device=self.device,
            )
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_token_map()
        self.init_lm_head()

        if get_spec().speculative_use_rejection_sampling:
            # Only dense RS graph buffers require matching configured vocab sizes.
            self._rs_vocab_size = self.target_worker.model_config.vocab_size
            draft_config_vocab = self.draft_runner.model_config.vocab_size
            if not self.sparse_rs and draft_config_vocab != self._rs_vocab_size:
                raise ValueError(
                    "--speculative-use-rejection-sampling needs the draft config "
                    f"vocab ({draft_config_vocab}) to equal the target vocab "
                    f"({self._rs_vocab_size})."
                )

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_attention_backends()
            self.init_attention_backend()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
            self._capture_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

    def _init_dsa_index_share_state(self) -> None:
        # Populate DSA index-share fields from the draft runner's hf_config.
        # Reused by the attention unit-test harnesses, which skip __init__.
        hf_config = self.draft_runner.model_config.hf_config
        if dl_hostw_enabled():
            _validate_hostw_startup(self.server_args, hf_config)
        # Reuse the first draft step's DSA indexer topk across the rest;
        # topk == 1 only (select_top_k_tokens reorders rows, desyncing indices).
        self.index_share_for_mtp_iteration = (
            getattr(hf_config, "index_share_for_mtp_iteration", False)
            and self.topk == 1
        )
        # GLM-5.2 MTP IndexShare: seed reused indexer top-k from draft-extend
        # (last verified token), not draft-decode step 0.
        self.dsa_index_topk = getattr(hf_config, "index_topk", None)
        self.seed_dsa_topk_from_draft_extend = (
            self.index_share_for_mtp_iteration and self.dsa_index_topk is not None
        )

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if get_spec().speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif get_spec().speculative_token_map is not None:
            self.hot_token_id = load_token_map(get_spec().speculative_token_map)
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        from sglang.srt.lora.layers import unwrap_lora_layer

        self.recur_hot_token_id = getattr(self, "recur_hot_token_id", None)
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        target_lm_head = unwrap_lora_layer(
            getattr(self.target_worker.model_runner.model, "lm_head", None)
        )
        self._draft_ctx_vocab = None
        ctxv_rows = int(os.environ.get("SGLANG_RND_DRAFT_CTX_VOCAB", "0"))
        if ctxv_rows < 0:
            raise ValueError("SGLANG_RND_DRAFT_CTX_VOCAB must be nonnegative")
        if ctxv_rows and (
            self.speculative_algorithm != SpeculativeAlgorithm.EAGLE
            or self.hot_token_id is None
            or head.dtype != torch.float8_e4m3fn
            or not head.is_cuda
            or get_parallel().tp_size != 1
            or get_parallel().pp_size != 1
            or get_parallel().attn_dp_size != 1
            or getattr(self.server_args, "enable_lora", False)
            or any(
                getattr(getattr(target_lm_head, "quant_method", None), name, False)
                for name in ("use_marlin", "block_quant", "use_mxfp8")
            )
        ):
            raise ValueError(
                "SGLANG_RND_DRAFT_CTX_VOCAB requires single-layer EAGLE/MTP, "
                "single-GPU CUDA per-row FP8 target rows, a speculative token map, and no LoRA"
            )

        def maybe_share_target_lm_head():
            if (
                target_lm_head is not None
                and self.hot_token_id is None
                and getattr(self.draft_runner.model, "hot_token_id", None) is None
                and hasattr(self.draft_runner.model, "set_lm_head_from_target")
            ):
                self.draft_runner.model.set_lm_head_from_target(target_lm_head)

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
                maybe_share_target_lm_head()
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )
            self._install_recur_head(target_lm_head)

        else:
            if self.hot_token_id is not None:
                self.hot_token_id = self.hot_token_id.to(head.device)
                if head.dtype == torch.float8_e4m3fn:
                    # Target lm_head is FP8 (per-channel, stored transposed [K, V]).
                    # Gather the hot rows and keep them FP8 in the same layout so the
                    # draft head runs through the target's Fp8LinearMethod (W8A16 GEMV).
                    weight_scale = target_lm_head.weight_scale.reshape(-1).float()
                    if ctxv_rows:
                        from sglang.srt.speculative.draft_context_vocab import (
                            DraftContextVocab,
                        )

                        self._draft_ctx_vocab = DraftContextVocab(
                            head, weight_scale, self.hot_token_id, ctxv_rows,
                            separate_tail=_MTP_LMHEAD_NVFP4,
                        )
                        logger.info("Draft context vocabulary: %d fixed FIFO rows", ctxv_rows)
                        if self._draft_ctx_vocab.target_topk:
                            logger.info(
                                "Draft context target candidates: top-%d after committed tokens",
                                self._draft_ctx_vocab.target_topk,
                            )
                    if ctxv_rows and not _MTP_LMHEAD_NVFP4:
                        self.hot_token_id = self._draft_ctx_vocab.hot_token_id
                        hot_head = self._draft_ctx_vocab.weight
                        hot_scale = self._draft_ctx_vocab.weight_scale
                    else:
                        hot_rows = head.t()[self.hot_token_id].contiguous()  # [H, K] fp8
                        hot_head = hot_rows.t()
                        hot_scale = weight_scale[self.hot_token_id].unsqueeze(1).contiguous()
                    head = torch.nn.Parameter(hot_head, requires_grad=False)  # [K, H+R]
                    self._draft_fp8_head_scale = torch.nn.Parameter(
                        hot_scale, requires_grad=False
                    )
                    self._draft_fp8_head_method = target_lm_head.quant_method
                    logger.info(
                        "Draft lm_head: %d hot FP8 rows sliced from the FP8 target head",
                        self.hot_token_id.numel(),
                    )
                else:
                    head = head.clone()
                    head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)
            maybe_share_target_lm_head()
            if getattr(self, "_draft_fp8_head_method", None) is not None:
                draft_lm_head = self.draft_runner.model.lm_head
                draft_lm_head.weight_scale = self._draft_fp8_head_scale
                draft_lm_head.input_scale = None
                draft_lm_head.quant_method = self._draft_fp8_head_method
                if self._draft_ctx_vocab is not None and not _MTP_LMHEAD_NVFP4:
                    draft_lm_head._rnd_ctx_vocab = self._draft_ctx_vocab
                    self.draft_runner.model.logits_processor.vocab_size = (
                        self.hot_token_id.numel()
                    )
            if _MTP_LMHEAD_NVFP4 and self.hot_token_id is not None:
                self._install_nvfp4_draft_head()
                if self._draft_ctx_vocab is not None:
                    # Pack only the immutable hot prefix. The FIFO keeps the
                    # target FP8 bytes/scales, even if NVFP4 packing fell back.
                    self.hot_token_id = self._draft_ctx_vocab.attach_head(
                        self.draft_runner.model.lm_head, self.hot_token_id
                    )
                    self.draft_runner.model.logits_processor.vocab_size = (
                        self.hot_token_id.numel()
                    )
                    logger.info(
                        "Draft mixed lm_head: %d static rows + %d FP8 FIFO rows",
                        len(self._draft_ctx_vocab.hot_ids), ctxv_rows,
                    )
            self._install_recur_head(target_lm_head)
            # Stage B runs *after* the draft head has been sliced, deliberately: the
            # draft slice is taken from the target's FP8 head, exactly as in production,
            # so the control arm is the shipped build and stage B is isolated to the
            # target's own logits. Doing it earlier would change both at once.
            if _LMHEAD_NVFP4 and target_lm_head is not None:
                self._install_nvfp4_target_head(target_lm_head)

    def _install_recur_head(self, target_lm_head):
        path = os.environ.get("SGLANG_RND_MTP_RECUR_HEAD")
        if not path:
            return
        from sglang.srt.speculative.mtp_recur_head import (
            RecurrentHeadLogitsProcessor,
            build_fp8_recur_head,
            build_nvfp4_recur_head,
            load_recur_token_map,
        )

        try:
            model = self.draft_runner.model
            if (
                type(model).__name__ != "Qwen4ExpForCausalLMMTP"
                or self.speculative_algorithm != SpeculativeAlgorithm.EAGLE
                or self.topk != 1
                or getattr(self.server_args, "enable_multi_layer_eagle", False)
                or getattr(self.server_args, "enable_lora", False)
                or not _is_cuda
                or get_parallel().tp_size != 1
                or get_parallel().pp_size != 1
                or get_parallel().attn_dp_size != 1
            ):
                raise ValueError("supported only for single-GPU, single-layer Qwen4 MTP EAGLE topk=1 without LoRA (not FrozenKV)")
            if isinstance(model.logits_processor, RecurrentHeadLogitsProcessor):
                raise ValueError("recurrent head is already installed; hot reload is unsupported")
            vocab_size = self.target_worker.model_config.vocab_size
            ctxv = getattr(self, "_draft_ctx_vocab", None)
            first_rows = (
                len(ctxv.hot_ids) if ctxv is not None
                else self.hot_token_id.numel() if self.hot_token_id is not None else vocab_size
            )
            min_rows = RS_DRAFT_TOPK if self.sparse_rs else 1
            ids = load_recur_token_map(
                path, vocab_size=vocab_size, first_head_rows=first_rows, min_rows=min_rows,
            )
            device = model.lm_head.weight.device
            if _MTP_LMHEAD_NVFP4:
                method = model.lm_head.quant_method
                method = getattr(method, "prefix_method", method)
                if type(method).__name__ != "NVFP4DenseLinearMethod":
                    raise ValueError("first MTP head fell back from NVFP4; keeping the same head for DECODE")
                cache_dir = os.environ.get("SGLANG_NVFP4_CACHE_DIR")
                if not cache_dir:
                    raise ValueError("set SGLANG_NVFP4_CACHE_DIR to a private cache for recurrent NVFP4 packing")
                recurrent_head = build_nvfp4_recur_head(
                    self.target_worker.model_runner.model_config.model_path,
                    ids, vocab_size=vocab_size, hidden_size=model.hidden_size, device=device,
                )
            else:
                recurrent_head = build_fp8_recur_head(
                    target_lm_head, ids, vocab_size=vocab_size, hidden_size=model.hidden_size,
                )
            from sglang.srt.layers.quantization.w8a16_gemv import prealloc

            prealloc(device)
            ids = ids.to(device)
            if ctxv is not None:
                ids = ctxv.attach_head(recurrent_head, ids)
            processor = RecurrentHeadLogitsProcessor(model.logits_processor, recurrent_head)
        except Exception as exc:
            logger.warning(
                "FLASH_NEXT_RECUR_HEAD (SGLANG_RND_MTP_RECUR_HEAD)=%s unavailable: %s; keeping the original MTP head and token map",
                path, exc,
            )
            return
        # Publish together, before either decode or extend graphs are captured.
        model.logits_processor = processor
        self.recur_hot_token_id = ids
        logger.info(
            "MTP recurrent head: %d rows in %s%s for DECODE only; first/extend head unchanged",
            ids.numel(), "NVFP4" if _MTP_LMHEAD_NVFP4 else "FP8",
            " + shared FP8 FIFO" if ctxv is not None else "",
        )

    def _install_nvfp4_draft_head(self):
        """Replace the draft's hot-vocab FP8 head with an NVFP4 one (N1 stage A).

        The rows are re-read from the checkpoint shard rather than taken from the FP8
        head this method just built: `head` is already FP8 by the time we get here, and
        quantising FP8 -> NVFP4 compounds two lossy steps for ~2 points of argmax
        agreement (N1_LOG.md 4). The packed result is cached on disk keyed by shard
        identity + row set, so only the first start pays the ~2 s.

        Measured: 1.487x on the draft-head GEMV (52.6 vs 78.3 us) for -0.64 % held-out
        acceptance at k=7. Off by default; SGLANG_MTP_LMHEAD_NVFP4=1 turns it on.
        """
        from sglang.srt.layers.quantization.nvfp4_dense import (
            NVFP4DenseLinearMethod,
            build_or_load,
        )

        model_dir = self.target_worker.model_runner.model_config.model_path
        draft_lm_head = self.draft_runner.model.lm_head
        try:
            wq, bs, gscale = build_or_load(
                model_dir, "lm_head.weight", self.hot_token_id,
                device=self.hot_token_id.device, label="draft lm_head",
            )
        except Exception:
            logger.exception(
                "SGLANG_MTP_LMHEAD_NVFP4=1 but the draft head could not be packed; "
                "keeping the FP8 head"
            )
            return
        # The NVFP4 GEMV shares w8a16_gemv's split-K scratch, and that scratch must
        # exist before any CUDA graph is captured or the first split-K call allocates
        # inside a capture and lands in that graph's private pool. Fp8LinearMethod's
        # process_weights_after_loading already preallocates it for every FP8 dense
        # category, but stage B/C drop categories from that list, so do not depend on
        # someone else having done it.
        from sglang.srt.layers.quantization.w8a16_gemv import prealloc

        prealloc(wq.device)
        NVFP4DenseLinearMethod.attach(draft_lm_head, wq, bs, gscale)
        # `weight` is a registered nn.Parameter, so it has to be deleted before a plain
        # uint8 tensor can take its place -- the same dance set_embed_and_head does.
        # It must still exist afterwards: should_apply_lm_head_quant_method refuses any
        # head without a `weight` attribute and would silently fall back to a dense
        # matmul against the packed codes.
        try:
            del draft_lm_head.weight
        except AttributeError:
            pass
        draft_lm_head.weight = wq
        draft_lm_head.weight_scale = None
        draft_lm_head.input_scale = None
        draft_lm_head.quant_method = NVFP4DenseLinearMethod("draft lm_head")
        logger.info(
            "Draft lm_head: %d hot rows in NVFP4 (%.1f MB vs %.1f MB FP8)",
            wq.shape[0],
            (wq.numel() + bs.numel()) / 1e6,
            wq.shape[0] * wq.shape[1] * 2 / 1e6,
        )

    def _install_nvfp4_target_head(self, target_lm_head):
        """Replace the target model's lm_head with NVFP4 (N1 stage B).

        All 248320 rows, packed from the checkpoint's BF16 rather than from the FP8
        weight `process_weights_after_loading` has already written, so the two lossy
        steps are not compounded. Measured 1.570x on this GEMV (250.2 vs 393.0 us at
        M=1, 88.5 % of the read roof) -- the largest single speedup in the N1 set,
        because 248320 rows give the widest grid.

        This changes what the model outputs, so it is gated on the full quality battery,
        not on acceptance.
        """
        from sglang.srt.layers.quantization.nvfp4_dense import (
            NVFP4DenseLinearMethod,
            build_or_load,
        )
        from sglang.srt.layers.quantization.w8a16_gemv import prealloc

        model_dir = self.target_worker.model_runner.model_config.model_path
        try:
            wq, bs, gscale = build_or_load(
                model_dir, "lm_head.weight", None,
                device=target_lm_head.weight.device, label="target lm_head",
            )
        except Exception:
            logger.exception(
                "SGLANG_LMHEAD_NVFP4=1 but the target head could not be packed; "
                "keeping the FP8 head"
            )
            return
        prealloc(wq.device)
        NVFP4DenseLinearMethod.attach(target_lm_head, wq, bs, gscale)
        # `weight` is a registered nn.Parameter, so it has to be deleted before a plain
        # uint8 tensor can take its place -- the same dance set_embed_and_head does.
        # It must still exist afterwards: should_apply_lm_head_quant_method refuses any
        # head without a `weight` attribute and would silently fall back to a dense
        # matmul against the packed codes.
        try:
            del target_lm_head.weight
        except AttributeError:
            pass
        target_lm_head.weight = wq
        target_lm_head.weight_scale = None
        target_lm_head.input_scale = None
        target_lm_head.quant_method = NVFP4DenseLinearMethod("target lm_head")
        torch.cuda.empty_cache()
        logger.info(
            "Target lm_head: %d rows in NVFP4 (%.1f MB vs %.1f MB FP8)",
            wq.shape[0],
            (wq.numel() + bs.numel()) / 1e6,
            wq.shape[0] * wq.shape[1] * 2 / 1e6,
        )

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=self.seed_dsa_topk_from_draft_extend,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        self.qsa_mtp_shared_sparse_indices = self._configure_qsa_mtp_index_share()
        self.tree_mask_mode = default_tree_mask_mode()

    def _configure_qsa_mtp_index_share(self):
        """Share the draft-extend's target-aligned QSA selection across the
        MTP decode steps (config-gated: index_share_for_mtp_iteration).

        Chain speculation only: with topk > 1 the decode rows are not
        request-major, so the captured per-request row has no unique reader.
        """
        from sglang.srt.layers.attention.qsa.config import is_qwen_qsa
        from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
        from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
            QSAMTPSharedSparseIndices,
        )

        hf_config = self.draft_runner.model_config.hf_config
        requested = _qsa_index_share_requested(hf_config)
        if (
            not requested
            or self.topk != 1
            or self.speculative_num_steps <= 1
            or not is_qwen_qsa(hf_config)
            or self.draft_attn_backend is None
            or self.draft_extend_attn_backend is None
        ):
            return
        layer_ids = sorted(
            {
                module.layer_id
                for module in self.draft_runner.model.modules()
                if isinstance(module, QSAIndexer)
            }
        )
        if not layer_ids:
            return
        from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend

        extend_backend = resolve_qsa_sparse_backend(self.draft_extend_attn_backend)
        state = getattr(extend_backend, "_mtp_shared_sparse_indices", None)
        if state is None or get_spec().speculative_adaptive:
            pool = self.draft_runner.token_to_kv_pool
            # The expansion emits token_topk + ratio - 1 columns (top-k blocks
            # plus the uncompressed tail of the capture position).
            expanded_width = pool.qsa_token_topk + pool.qsa_compress_ratio - 1
            state = QSAMTPSharedSparseIndices(
                layer_ids=layer_ids,
                num_requests=self.draft_runner.req_to_token_pool.req_to_token.shape[0],
                token_topk=expanded_width,
                tail_width=self.speculative_num_steps + 1,
                device=self.draft_runner.device,
            )
        self._install_qsa_mtp_index_share(
            state=state,
            draft_attn_backend=self.draft_attn_backend,
            draft_extend_attn_backend=self.draft_extend_attn_backend,
        )
        logger.info(
            "QSA MTP index sharing enabled: draft decode steps reuse the "
            f"draft-extend selection for layers {layer_ids}"
        )
        return state

    @staticmethod
    def _install_qsa_mtp_index_share(
        *, state, draft_attn_backend, draft_extend_attn_backend
    ) -> None:
        from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend

        for backend in (draft_attn_backend, draft_extend_attn_backend):
            if backend is None:
                continue
            resolved = resolve_qsa_sparse_backend(backend)
            resolved.set_mtp_shared_sparse_indices(state)

    def _capture_cuda_graphs(self):
        """Capture the draft worker's own cuda graphs (decode + draft-extend)."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if _is_cpu or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if get_model().model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "xpu": EAGLEDraftCudaGraphRunner,
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        decode_backend = get_exec().graph.cuda_graph_config.decode.backend
        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft decode CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.topk}, bs={capture_bs}, "
                f"avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_decode"] = (
                self._specialized_graph_memory_usage.get("draft_decode", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_decode"] = (
                self._specialized_graph_time_usage.get("draft_decode", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft decode CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "xpu": EAGLEDraftExtendCudaGraphRunner,
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_draft_extend_graph = False
        if _is_hip:
            # Keep imports local so non-HIP environments do not require these.
            # aiter packs draft-extend support into the decode (multi-step)
            # backend; DSV4 exposes it on the draft-extend backend itself.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )
            from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
                DeepseekV4HipRadixBackend,
            )
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            supports_hip_draft_extend_graph = (
                isinstance(self.draft_attn_backend, AiterMultiStepDraftBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekV4HipRadixBackend)
                or isinstance(self.draft_extend_attn_backend, DeepseekSparseAttnBackend)
            )

        graph_supported_backend_types = [
            TritonAttnBackend,
            TRTLLMMLABackend,
            TRTLLMHAAttnBackend,
            TokenspeedMLABackend,
            FlashInferAttnBackend,
        ]
        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)
            from sglang.srt.layers.attention.deepseek_v4_backend import (
                DeepseekV4AttnBackend,
            )

            graph_supported_backend_types.append(DeepseekV4AttnBackend)
        if _is_cuda:
            # FlashMLA is CUDA-only; import lazily so CPU builds don't pull
            # sgl_kernel.flash_mla at import time.
            from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

            graph_supported_backend_types.append(FlashMLABackend)

        graph_supported_backend = isinstance(
            self.draft_extend_attn_backend,
            tuple(graph_supported_backend_types),
        )
        if not graph_supported_backend and self.draft_extend_attn_backend is not None:
            # Qwen QSA keeps draft-extend metadata graph-stable (variable
            # accepted rows padded to the captured width), so its hybrid
            # backend is graph-capable even though it is not in the list.
            graph_supported_backend = is_qwen_qsa(
                self.draft_runner.model_config.hf_config
            )
        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if (
            self.draft_extend_attn_backend
            and not envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
            and (
                _is_npu
                or _is_xpu
                or supports_cuda_draft_extend_graph
                or supports_hip_draft_extend_graph
            )
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_req={self.speculative_num_draft_tokens}, "
                f"bs={capture_bs}, avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            # draft_extend is the step's last shared-buffer-reading phase; its
            # read-done event is what the scheduler's WAR barrier waits on.
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            capture_time = time.perf_counter() - tic
            self._specialized_graph_memory_usage["draft_extend"] = (
                self._specialized_graph_memory_usage.get("draft_extend", 0.0)
                + before_mem
                - after_mem
            )
            self._specialized_graph_time_usage["draft_extend"] = (
                self._specialized_graph_time_usage.get("draft_extend", 0.0)
                + capture_time
            )
            log_info_on_rank0(
                logger,
                "Capture draft extend CUDA graph end. "
                f"elapsed={capture_time:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

    def draft(self, batch: ScheduleBatch, hostw_states=None):
        if hostw_states is not None and batch.seq_lens.shape[0] > 1:
            raise ValueError("HOSTW requires a single unpadded request")
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_run_decode_cuda_graph = prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        if (
            can_run_decode_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_run_decode_cuda_graph = False

        if (
            self._dl_enabled
            and forward_batch.batch_size == 1
            and self.speculative_num_steps > 1
            and not forward_batch.forward_mode.is_idle()
            and not can_run_decode_cuda_graph
        ):
            raise RuntimeError("DL requires a captured draft graph; eager runtime fallback is forbidden")

        n_inner = self.speculative_num_steps - 1
        canary_outside_ctx = (
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            if can_run_decode_cuda_graph:
                draft_result = self.cuda_graph_runner.execute(forward_batch)
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                draft_result = self.draft_forward(forward_batch)

        draft_support_probs = draft_support_tokens = None
        if self.sparse_rs:
            parent_list, top_scores_index, draft_tokens, draft_probs, draft_support_probs, draft_support_tokens = draft_result
        else:
            parent_list, top_scores_index, draft_tokens, draft_probs = draft_result

        if (
            self._conf_channel is not None
            and self._chain_conf_buf is not None
            and not forward_batch.forward_mode.is_idle()
        ):
            # Copy before draft-extend overwrites column 0 for the next chain.
            self._conf_channel.record_chain(
                self._chain_conf_buf,
                bs=batch.seq_lens.shape[0],
                steps=self.speculative_num_steps,
            )

        verify_steps = self.speculative_num_steps
        target_attn_backend = None
        if hostw_states is not None and not batch.forward_mode.is_idle():
            # All width-independent work (including strided batch copies) is
            # queued before the sole host read. Never recapture in the hot path.
            prepared = {
                steps: (
                    trim_draft_payload(steps, draft_tokens, draft_probs,
                                       draft_support_probs, draft_support_tokens),
                    parent_list[:, :steps].contiguous(),
                    top_scores_index[:, :steps].contiguous(),
                )
                for steps, state in hostw_states.items()
                if 0 < steps <= self.speculative_num_steps
                and state.target_graph_runner is not None
            }
            length = read_max_length(self._dl_lengths, batch.seq_lens.shape[0])
            verify_steps = select_verify_steps(
                length, prepared, self.speculative_num_steps
            )
            payload, parent_list, top_scores_index = prepared[verify_steps]
            draft_tokens = payload["draft_tokens"]
            draft_probs = payload["draft_probs"]
            draft_support_probs = payload["draft_support_probs"]
            draft_support_tokens = payload["draft_support_tokens"]
            target_attn_backend = hostw_states[verify_steps].target_attn_backend

        verify_input = build_eagle_verify_input(
            batch=batch,
            draft_input=draft_input,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            draft_support_probs=draft_support_probs,
            draft_support_tokens=draft_support_tokens,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=verify_steps,
            num_draft_tokens=verify_steps + 1 if hostw_states is not None else self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
            chain_topk1=self._draft_tail_chain,
            target_attn_backend=target_attn_backend,
            **(
                {"vt_confidence": self._vt_conf_buf}
                if self._vt_enabled and batch.seq_lens.shape[0] == 1 else {}
            ),
        )
        if hostw_states is not None:
            verify_input.draft_reserved_token_num = self.speculative_num_draft_tokens
        return verify_input

    def _record_position0_confidence(self, next_token_logits: torch.Tensor):
        # C1: the confidence the adaptive controller decides on. This is the
        # only chain position whose probability exists before the next step's
        # state swap, because the draft loop itself is one captured CUDA graph.
        if self._conf_channel is None and self._vt_position0 is None:
            return
        self._store_position0_confidence(top1_prob(next_token_logits))

    def _store_position0_confidence(self, p0: torch.Tensor):
        if self._vt_position0 is not None:
            if p0.shape[0] > self._vt_position0.shape[0]:
                raise ValueError("VT position-0 batch exceeds launch dimensions")
            # Keep the singleton launch row-ordered; concurrent launches also
            # key p0 by request so singleton drafts survive batch filter/merge.
            self._vt_position0[: p0.shape[0]].copy_(p0)
            if self._hostw_position0 is not None:
                self._hostw_position0.index_copy_(
                    0, self._hostw_req_indices.to(torch.int64), p0
                )
        if self._conf_channel is not None:
            self._conf_channel.record_position0(p0)
            if self._chain_conf_buf is not None:
                n = min(p0.shape[0], self._chain_conf_buf.shape[0])
                self._chain_conf_buf[:n, 0].copy_(p0[:n])

    def _rs_draft_proposal(
        self,
        next_token_logits: torch.Tensor,
        sampling_info,
        out: Optional[torch.Tensor] = None,
        hot_token_id: Optional[torch.Tensor] = None,
    ):
        """Rejection-sampling draft proposal on the target vocab.

        Returns (q, q(X), X): q is (bs, target vocab) and zero off the draft's
        support, X is a draft-vocab id (the callers map it through
        hot_token_id). `out`, when given, must be zero-filled (bs, vocab) and
        receives q in place.
        """
        if hot_token_id is None:
            hot_token_id = self.hot_token_id
        bs = next_token_logits.shape[0]
        if RS_DRAFT_TOPK > 0:
            probs, ids, topk_p, topk_index = sample_draft_proposal_truncated(
                next_token_logits,
                sampling_info.temperatures,
                sampling_info.top_ks,
                sampling_info.top_ps,
                RS_DRAFT_TOPK,
                min_ps=sampling_info.min_ps if SPEC_MIN_P else None,
            )
            if hot_token_id is not None:
                ids = hot_token_id[ids]
        else:
            probs, topk_p, topk_index = sample_draft_proposal(
                next_token_logits, sampling_info.temperatures
            )
            ids = hot_token_id
            if ids is None:
                if out is None:
                    return probs, topk_p, topk_index
                out.copy_(probs)
                return out, topk_p, topk_index
        if out is None:
            out = torch.zeros(
                (bs, self._rs_vocab_size),
                dtype=torch.float32,
                device=next_token_logits.device,
            )
        if getattr(self, "_draft_ctx_vocab", None) is not None:
            # Empty tail slots map to hot[0] but have zero probability. A copy
            # would overwrite that token's real mass with a duplicate zero.
            out.scatter_add_(1, ids.expand(bs, -1), probs.float())
        elif ids.dim() == 1:
            out.index_copy_(1, ids, probs.float())
        else:
            out.scatter_(1, ids, probs)
        return out, topk_p, topk_index

    def _rs_sparse_proposal(self, *, next_token_logits, sampling_info):
        return rs_draft_proposal_sparse(
            next_token_logits=next_token_logits,
            temperatures=sampling_info.temperatures,
            top_ks=sampling_info.top_ks,
            top_ps=sampling_info.top_ps,
            min_ps=sampling_info.min_ps if SPEC_MIN_P else None,
            uniforms=torch.rand((next_token_logits.shape[0],), device=next_token_logits.device),
            hot_token_id=self.hot_token_id,
            k=RS_DRAFT_TOPK,
            temp_scale=RS_DRAFT_TEMP_SCALE, onehot_above=RS_DRAFT_ONEHOT_ABOVE,
            **({"greedy_fast": True} if RS_GREEDY_FAST else {}),
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        if forward_batch.forward_mode.is_idle():
            return self._draft_forward_idle(forward_batch, spec_info)

        dl_enabled = self._dl_enabled and forward_batch.batch_size == 1
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None and getattr(self, "_draft_ctx_vocab", None) is None:
            topk_index = self.hot_token_id[topk_index]

        # The seed still came from the unchanged draft-extend head. Only logits
        # produced by an own-HC DECODE below use the recurrent vocabulary.
        decode_token_map = (
            getattr(self, "recur_hot_token_id", None)
            if is_recur_head_mode(forward_batch.forward_mode)
            else None
        )
        if decode_token_map is None:
            decode_token_map = self.hot_token_id

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        # Rejection sampling: q per chain position on the target vocab. Row 0
        # came from the previous draft-extend; each draft step fills the next
        # row in place, so no per-step stack of vocab-wide rows is needed.
        draft_probs = None
        draft_support_probs = draft_support_tokens = uniforms = None
        if self.sparse_rs:
            bs, support_size = spec_info.draft_support_probs.shape
            draft_support_probs = torch.empty(
                (bs, self.speculative_num_steps, support_size), dtype=torch.float32, device=topk_index.device,
            )
            draft_support_tokens = torch.empty(
                (bs, self.speculative_num_steps, support_size), dtype=torch.int64, device=topk_index.device,
            )
            draft_support_probs[:, 0].copy_(spec_info.draft_support_probs)
            draft_support_tokens[:, 0].copy_(spec_info.draft_support_tokens)
            uniforms = torch.rand((bs, self.speculative_num_steps - 1), device=topk_index.device)
        elif get_spec().speculative_use_rejection_sampling:
            draft_probs = torch.zeros(
                (
                    topk_index.shape[0],
                    self.speculative_num_steps,
                    spec_info.draft_probs.shape[-1],
                ),
                dtype=torch.float32,
                device=spec_info.draft_probs.device,
            )
            draft_probs[:, 0].copy_(spec_info.draft_probs)

        topk1_chain_fits = (
            self.topk == 1
            and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
        )
        # Materialize the chain directly only when the CUDA kernel can write
        # every subsequent column. Other topk=1 paths retain the token list and
        # assemble it with one final cat instead of launching a copy per step.
        draft_tokens_topk1 = None
        if (
            topk1_chain_fits
            and _is_cuda
            and (dl_enabled or self.sparse_rs or not get_spec().speculative_use_rejection_sampling)
        ):
            draft_tokens_topk1 = torch.empty(
                (topk_index.shape[0], self.speculative_num_steps),
                dtype=topk_index.dtype,
                device=topk_index.device,
            )
            draft_tokens_topk1[:, :1].copy_(topk_index)

        if self.sparse_rs and draft_tokens_topk1 is None:
            draft_tokens_topk1 = torch.empty(
                (topk_index.shape[0], self.speculative_num_steps), dtype=topk_index.dtype, device=topk_index.device,
            )
            draft_tokens_topk1[:, :1].copy_(topk_index)
            if not topk1_chain_fits:
                raise ValueError("RS2 batch exceeds the topk1 chain preallocations")

        # C1: per-position draft confidence (trace only; None in production).
        # Column 0 was written by the previous iteration's draft-extend, which
        # is where this chain's first token came from.
        chain_conf_buf = (
            self._chain_conf_buf
            if (
                draft_tokens_topk1 is not None
                and self._chain_conf_buf is not None
                and topk_index.shape[0] <= self._chain_conf_buf.shape[0]
                and self.speculative_num_steps <= self._chain_conf_buf.shape[1]
            )
            else None
        )

        vt_conf_buf = self._vt_conf_buf if forward_batch.batch_size == 1 else None
        if vt_conf_buf is not None:
            bs = topk_index.shape[0]
            if (
                bs > vt_conf_buf.shape[0]
                or self.speculative_num_steps > vt_conf_buf.shape[1]
            ):
                raise ValueError("VT draft exceeds launch confidence dimensions")
            # This copy is captured too: every replay consumes the latest p0.
            p0 = (
                self._hostw_position0[forward_batch.req_pool_indices.to(torch.int64)]
                if self._hostw_position0 is not None else self._vt_position0[:bs]
            )
            vt_conf_buf[:bs, 0].copy_(p0)
        topk1_conf_buf = vt_conf_buf if vt_conf_buf is not None else chain_conf_buf

        if dl_enabled:
            from sglang.srt.speculative.dl_graph import conditional_if
            from sglang.srt.speculative.dl_kernels import advance

            # Capture each existing width with its own threshold, not the
            # launch width or C1's pre-stepup choice. Zero means no stopping.
            dl_threshold = dl_tau(steps=self.speculative_num_steps)
            if draft_tokens_topk1 is None or bs != 1:
                raise ValueError("DL requires materialized topk=1 chains")
            # Reset every replay, including after a longer adaptive-width stage.
            # Zero is an in-range token ID, not a semantic padding token.
            draft_tokens_topk1[:, 1:].zero_()
            vt_conf_buf[:, 1:].zero_()
            if chain_conf_buf is not None:
                chain_conf_buf[:, 1:].zero_()
            if draft_support_probs is not None:
                draft_support_probs[:, 1:].zero_()
                draft_support_tokens[:, 1:].zero_()
            self._dl_product.fill_(1.0)
            self._dl_lengths.zero_()
            self._dl_forwards.zero_()
            advance(vt_conf_buf[:bs], self._dl_product, self._dl_lengths,
                    self._dl_predicate, self._dl_forwards, 0,
                    self.speculative_num_steps, dl_threshold)

        # Forward multiple steps
        scores = None
        with IndexTopKShareState.mtp_iteration(
            forward_batch,
            enabled=self.index_share_for_mtp_iteration,
            keep_carry_seed=self.seed_dsa_topk_from_draft_extend,
        ):
            for i in range(self.speculative_num_steps):
                if draft_tokens_topk1 is not None:
                    input_ids = topk_index.flatten()
                else:
                    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                        i, topk_p, topk_index, hidden_states, scores, self.topk
                    )
                    score_list.append(tree_info[0])
                    token_list.append(tree_info[1])
                    parents_list.append(tree_info[2])

                if i == self.speculative_num_steps - 1:
                    break

                with (
                    conditional_if(self._dl_predicate)
                    if dl_enabled else contextlib.nullcontext()
                ):
                    forward_batch.input_ids = input_ids
                    # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
                    # argument must be contiguous.
                    if (
                        self.draft_runner.model_config.hf_config.architectures[0]
                        == "Qwen3MoeForCausalLMMTP"
                    ):
                        out_cache_loc = out_cache_loc.contiguous()
                    forward_batch.out_cache_loc = out_cache_loc[i]
                    spec_info.hidden_states = hidden_states

                    canary_index_ctx = (
                        c.with_active_single_forward_manager(i)
                        if (c := self.draft_runner.canary_manager) is not None
                        else contextlib.nullcontext()
                    )
                    with (
                        forward_context(
                            ForwardContext(
                                attn_backend=self.draft_attn_backend.attn_backends[i]
                            )
                        ),
                        canary_index_ctx,
                    ):
                        logits_output = self.draft_runner.forward(
                            forward_batch
                        ).logits_output
                    maybe_detect_nan(
                        logits_output.next_token_logits, f"draft_forward step {i}"
                    )
                    if getattr(self, "_draft_ctx_vocab", None) is None:
                        maybe_detect_inf(
                            logits_output.next_token_logits, f"draft_forward step {i}"
                        )
                    if (
                        vt_conf_buf is not None
                        and get_spec().speculative_use_rejection_sampling
                    ):
                        # Raw, untempered top-1 mass over every emitted logit, before
                        # RS truncation, temperature scaling, or one-hot conversion.
                        vt_conf_buf[:bs, i + 1].copy_(
                            top1_prob(logits_output.next_token_logits)
                        )
                    if self.sparse_rs:
                        _, _, topk_p, topk_index = rs_draft_proposal_sparse(
                            next_token_logits=logits_output.next_token_logits,
                            temperatures=forward_batch.sampling_info.temperatures,
                            top_ks=forward_batch.sampling_info.top_ks,
                            top_ps=forward_batch.sampling_info.top_ps,
                            min_ps=forward_batch.sampling_info.min_ps if SPEC_MIN_P else None,
                            uniforms=uniforms[:, i], k=RS_DRAFT_TOPK,
                            temp_scale=RS_DRAFT_TEMP_SCALE, onehot_above=RS_DRAFT_ONEHOT_ABOVE,
                            **({"greedy_fast": True} if RS_GREEDY_FAST else {}),
                            hot_token_id=decode_token_map, positions=forward_batch.positions,
                            draft_tokens=draft_tokens_topk1, draft_token_column=i + 1,
                            draft_support_probs=draft_support_probs[:, i + 1],
                            draft_support_tokens=draft_support_tokens[:, i + 1],
                        )
                    elif get_spec().speculative_use_rejection_sampling:
                        _, topk_p, topk_index = self._rs_draft_proposal(
                            logits_output.next_token_logits,
                            forward_batch.sampling_info,
                            out=draft_probs[:, i + 1],
                            hot_token_id=decode_token_map,
                        )
                        forward_batch.positions.add_(1)
                        if dl_enabled:
                            if decode_token_map is not None:
                                topk_index = decode_token_map[topk_index]
                            draft_tokens_topk1[:, i + 1].copy_(topk_index.flatten())
                    elif self.topk == 1 and not _is_hip:
                        if _is_cuda:
                            topk_p, topk_index = draft_topk1_postprocess(
                                logits_output.next_token_logits,
                                forward_batch.positions,
                                draft_tokens_topk1,
                                i + 1,
                                hot_token_id=(
                                    decode_token_map
                                    if draft_tokens_topk1 is not None
                                    else None
                                ),
                                # VT/C1: reduce all logits, including contextual
                                # vocabulary rows, without changing greedy topk_p.
                                # Both off: WRITE_PROB specialises away.
                                chain_probs=topk1_conf_buf,
                            )
                            if vt_conf_buf is not None and chain_conf_buf is not None:
                                chain_conf_buf[:bs, i + 1].copy_(
                                    vt_conf_buf[:bs, i + 1]
                                )
                        else:
                            topk_index = torch.argmax(
                                logits_output.next_token_logits, dim=-1, keepdim=True
                            )
                            topk_p = torch.ones_like(topk_index, dtype=torch.float32)
                            forward_batch.positions.add_(1)
                    else:
                        probs = renorm_draft_probs(
                            logits_output.next_token_logits,
                            forward_batch.sampling_info,
                            get_spec().speculative_use_rejection_sampling,
                        )
                        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
                        forward_batch.positions.add_(1)
                    topk_index_vocab_size = (
                        self.target_worker.model_config.vocab_size
                        if draft_tokens_topk1 is not None and decode_token_map is not None
                        else logits_output.next_token_logits.shape[-1]
                    )
                    maybe_detect_oob(
                        topk_index,
                        0,
                        topk_index_vocab_size,
                        f"draft_forward step {i}: topk_index OOB vs vocab_size={topk_index_vocab_size}",
                    )
                    if decode_token_map is not None and draft_tokens_topk1 is None:
                        topk_index = decode_token_map[topk_index]
                    hidden_states = logits_output.hidden_states
                    if dl_enabled:
                        advance(vt_conf_buf[:bs], self._dl_product, self._dl_lengths,
                                self._dl_predicate, self._dl_forwards, i + 1,
                                self.speculative_num_steps, dl_threshold)

        # Organize the results
        if draft_tokens_topk1 is not None:
            bs = draft_tokens_topk1.shape[0]
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            if self.sparse_rs:
                return parent_list, top_scores_index, draft_tokens_topk1, None, draft_support_probs, draft_support_tokens
            return parent_list, top_scores_index, draft_tokens_topk1, draft_probs

        if topk1_chain_fits:
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            return parent_list, top_scores_index, draft_tokens, draft_probs

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens, draft_probs

    def _draft_forward_idle(
        self, forward_batch: ForwardBatch, spec_info: EagleDraftInput
    ):
        """Run eager idle-rank collectives without materializing draft state."""
        input_ids = forward_batch.input_ids
        out_cache_loc = forward_batch.out_cache_loc
        hidden_states = spec_info.hidden_states

        # ModelRunner pads and unpads the empty batch on every call. Avoid the
        # normal tree/cache-layout path: idle outputs are discarded when the
        # verify input is built, but every rank must still enter each forward.
        for i in range(self.speculative_num_steps - 1):
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc
            spec_info.hidden_states = hidden_states
            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                self.draft_runner.forward(forward_batch)

        return (None,) * (6 if self.sparse_rs else 4)

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        if self._vt_position0 is not None:
            self._hostw_req_indices = batch.req_pool_indices
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            # Chunked-prefill-aware tail tokens (see PR #26329).
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
            ctxv = getattr(self, "_draft_ctx_vocab", None)
            if ctxv is not None:
                for req in batch.reqs:
                    if getattr(req, "_rnd_ctxv_prefill_state", None) is not ctxv:
                        # Include radix-cached prompt tokens, not just this chunk.
                        ctxv.update_prefill(req.origin_input_ids)
                        ctxv.update_prefill(req.output_ids)
                        req._rnd_ctxv_prefill_state = ctxv
                ctxv.update_committed(tail_tokens, None, 1)
            new_input_ids = torch.empty_like(batch.input_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                new_input_ids[pt : pt + extend_len].copy_(
                    torch.cat((input_ids[1:], tail_tokens[i].reshape(1)))
                )
                pt += extend_len
            assert pt == batch.input_ids.numel()
            batch.input_ids = new_input_ids

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            self.draft_runner,
            capture_hidden_mode=capture_hidden_mode,
            return_hidden_states_before_norm=False,
        )
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        # Seed the first draft-decode loop from each request's last prefill
        # position. Gather last-per-req before the copy (prefill can be long).
        seed_from_extend = (
            self.seed_dsa_topk_from_draft_extend
            and not forward_batch.forward_mode.is_idle()
        )
        if seed_from_extend:
            bs = forward_batch.batch_size
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(bs)
            )
            forward_batch.spec_info.dsa_seed_topk_select = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            ).long()

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        if getattr(self, "_draft_ctx_vocab", None) is None:
            maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")

        if self._vt_position0 is not None:
            self._record_position0_confidence(logits_output.next_token_logits)

        prefill_dsa_topk = None
        if seed_from_extend:
            prefill_dsa_topk = self.dsa_extend_topk_buf[:bs].clone()

        # Assemble the next-iter draft spec_info from the extend output.
        draft_probs = None
        draft_support_probs = draft_support_tokens = None
        if self.sparse_rs:
            draft_support_probs, draft_support_tokens, topk_p, topk_index = self._rs_sparse_proposal(
                next_token_logits=logits_output.next_token_logits, sampling_info=batch.sampling_info,
            )
        elif get_spec().speculative_use_rejection_sampling:
            draft_probs, topk_p, topk_index = self._rs_draft_proposal(
                logits_output.next_token_logits, batch.sampling_info
            )
        else:
            probs = renorm_draft_probs(
                logits_output.next_token_logits, batch.sampling_info, False
            )
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        if getattr(self, "_draft_ctx_vocab", None) is not None:
            # Snapshot the token now: another batch's prefill may evict its
            # tail row before this cached first proposal is consumed.
            topk_index = self.hot_token_id[topk_index]
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=draft_probs,
            draft_support_probs=draft_support_probs,
            draft_support_tokens=draft_support_tokens,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            dsa_topk_indices=prefill_dsa_topk,
        )

    def _get_dsa_extend_topk_buf(self, num_tokens: int) -> torch.Tensor:
        """Lazily-grown int32 [num_tokens, index_topk] eager draft-extend seed buffer."""
        buf = self.dsa_extend_topk_buf
        if buf is None or buf.shape[0] < num_tokens:
            buf = torch.full(
                (num_tokens, self.dsa_index_topk),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            self.dsa_extend_topk_buf = buf
        return buf[:num_tokens]

    def _draft_extend_select_tail(
        self,
        next_draft_input: EagleDraftInput,
        draft_logits_output: LogitsProcessorOutput,
        accept_lens: torch.Tensor,
    ):
        # The topk == 1 branch of _draft_extend_for_decode in two launches; p0 is
        # top1_prob of the gathered rows (bit-exact) unless FUSED_CONF is on.
        conf_on = self._conf_channel is not None or self._vt_position0 is not None
        topk_p, topk_index, hidden_states, rows, conf = draft_extend_select_topk1(
            next_token_logits=draft_logits_output.next_token_logits,
            hidden_states=draft_logits_output.hidden_states,
            write_rows=conf_on and not self._draft_tail_fused_conf,
            write_prob=conf_on and self._draft_tail_fused_conf,
            accept_lens=accept_lens,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        if conf_on:
            p0 = conf if self._draft_tail_fused_conf else top1_prob(rows)
            self._store_position0_confidence(p0)
        next_draft_input.topk_p = topk_p
        next_draft_input.topk_index = (
            self.hot_token_id[topk_index]
            if getattr(self, "_draft_ctx_vocab", None) is not None
            else topk_index
        )
        next_draft_input.hidden_states = hidden_states

    def _rs_draft_extend_tail(
        self,
        next_draft_input: EagleDraftInput,
        draft_logits_output: LogitsProcessorOutput,
        accept_lens: torch.Tensor,
        sampling_info,
    ):
        # The sparse_rs branch of _draft_extend_for_decode without the select_index
        # gathers; p0 is top1_prob of the copied rows (bit-exact) unless FUSED_CONF.
        conf_on = self._conf_channel is not None or self._vt_position0 is not None
        logits = draft_logits_output.next_token_logits
        support_probs, support_tokens, topk_p, topk_index, hidden_states, rows, conf = (
            rs_draft_extend_proposal_sparse(
                next_token_logits=logits,
                accept_lens=accept_lens,
                num_tokens_per_req=self.speculative_num_draft_tokens,
                temperatures=sampling_info.temperatures,
                top_ks=sampling_info.top_ks,
                top_ps=sampling_info.top_ps,
                min_ps=sampling_info.min_ps if SPEC_MIN_P else None,
                uniforms=torch.rand((accept_lens.shape[0],), device=logits.device),
                hot_token_id=self.hot_token_id,
                k=RS_DRAFT_TOPK,
                hidden_states=draft_logits_output.hidden_states,
                write_rows=conf_on and not self._rs_extend_fused_conf,
                write_prob=conf_on and self._rs_extend_fused_conf,
                temp_scale=RS_DRAFT_TEMP_SCALE,
                onehot_above=RS_DRAFT_ONEHOT_ABOVE,
                greedy_fast=RS_GREEDY_FAST,
            )
        )
        if conf_on:
            p0 = conf if self._rs_extend_fused_conf else top1_prob(rows)
            self._store_position0_confidence(p0)
        next_draft_input.topk_p = topk_p
        next_draft_input.topk_index = (
            self.hot_token_id[topk_index]
            if getattr(self, "_draft_ctx_vocab", None) is not None
            else topk_index
        )
        next_draft_input.hidden_states = hidden_states
        next_draft_input.draft_support_probs = support_probs
        next_draft_input.draft_support_tokens = support_tokens
        next_draft_input.draft_probs = None

    def _update_context_from_verify(self, batch, batch_result):
        ctxv = getattr(self, "_draft_ctx_vocab", None)
        if ctxv is None or batch.forward_mode.is_idle():
            return
        # Preserve committed-token update order. Candidates are
        # a second FIFO transaction, never a replacement for committed tokens.
        ctxv.update_committed(
            batch_result.next_token_ids,
            batch_result.accept_lens,
            self.speculative_num_draft_tokens,
        )
        topk_ids = getattr(batch_result, "draft_context_topk_ids", None)
        if getattr(ctxv, "target_topk", 0) and topk_ids is not None:
            ctxv.update_topk(topk_ids, batch_result.accept_lens)

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        if self._vt_position0 is not None:
            self._hostw_req_indices = batch.req_pool_indices
        self._update_context_from_verify(batch, batch_result)
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # accept_lens includes the bonus token; correct drafts exclude it.
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        # The fused draft tail derives these rows from accept_lens in-kernel.
        select_index = None
        fused_select = self._draft_tail_select or self._rs_extend_tail
        if not fused_select or self.seed_dsa_topk_from_draft_extend:
            select_index = (
                torch.arange(
                    0,
                    len(batch.seq_lens) * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    device=self.device,
                )
                + batch_result.accept_lens
                - 1
            )

        # Cast to int64 before entering plan stream to avoid cross-stream
        # synchronization issues with .to() inside the plan stream context.
        next_token_ids = batch_result.next_token_ids.to(torch.int64)

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = prepare_for_draft_extend(
                draft_extend_input,
                batch,
                next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
                return_hidden_states_before_norm=False,
                fused_lens=self._draft_extend_lens_fused,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_run_decode_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )

        # Eager path publishes the indexer top-k into a worker buffer (the graph
        # path uses the runner's static buffer). Gathered at select_index below.
        if self.seed_dsa_topk_from_draft_extend and not can_run_decode_cuda_graph:
            forward_batch.spec_info.dsa_seed_topk_capture = (
                self._get_dsa_extend_topk_buf(forward_batch.input_ids.shape[0])
            )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_run_decode_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                    forward_batch
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
        )
        if getattr(self, "_draft_ctx_vocab", None) is None:
            maybe_detect_inf(
                draft_logits_output.next_token_logits,
                f"draft_extend_for_decode (cuda_graph={can_run_decode_cuda_graph})",
            )

        # Gather the per-request last-position indexer top-k as the next loop's
        # seed (select_index already picks the last accepted position per req).
        dsa_seed_topk_indices = None
        if self.seed_dsa_topk_from_draft_extend:
            if can_run_decode_cuda_graph:
                dsa_extend_topk_capture = (
                    self.cuda_graph_runner_for_draft_extend.buffers.dsa_seed_topk_capture
                )
            else:
                dsa_extend_topk_capture = forward_batch.spec_info.dsa_seed_topk_capture
            # Fancy indexing returns a fresh tensor (detached from the buffer).
            dsa_seed_topk_indices = dsa_extend_topk_capture[select_index]

        # Reorganize the spec info for the next batch
        if self._draft_tail_select:
            self._draft_extend_select_tail(
                next_draft_input=batch_result.next_draft_input,
                draft_logits_output=draft_logits_output,
                accept_lens=batch_result.accept_lens,
            )
            if self.seed_dsa_topk_from_draft_extend:
                batch_result.next_draft_input.dsa_topk_indices = dsa_seed_topk_indices
            return
        if self._rs_extend_tail:
            self._rs_draft_extend_tail(
                next_draft_input=batch_result.next_draft_input,
                draft_logits_output=draft_logits_output,
                accept_lens=batch_result.accept_lens,
                sampling_info=batch.sampling_info,
            )
            if self.seed_dsa_topk_from_draft_extend:
                batch_result.next_draft_input.dsa_topk_indices = dsa_seed_topk_indices
            return
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
        # The draft-extend graph only anchors full logits; selected-row topk is
        # owned by the worker for both graph and eager paths.
        if self.sparse_rs:
            ret_support_probs, ret_support_tokens, ret_topk_p, ret_topk_index = self._rs_sparse_proposal(
                next_token_logits=draft_logits_output.next_token_logits, sampling_info=batch.sampling_info,
            )
            ret_draft_probs = None
            self._record_position0_confidence(draft_logits_output.next_token_logits)
        elif get_spec().speculative_use_rejection_sampling:
            ret_draft_probs, ret_topk_p, ret_topk_index = self._rs_draft_proposal(
                draft_logits_output.next_token_logits, batch.sampling_info
            )
            self._record_position0_confidence(draft_logits_output.next_token_logits)
        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None
            # ret_topk_p stays 1.0 so every existing consumer is unchanged.
            self._record_position0_confidence(draft_logits_output.next_token_logits)
        else:
            probs = renorm_draft_probs(
                draft_logits_output.next_token_logits,
                batch.sampling_info,
                get_spec().speculative_use_rejection_sampling,
            )
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_draft_probs = None
        ret_hidden_states = draft_logits_output.hidden_states
        if getattr(self, "_draft_ctx_vocab", None) is not None:
            ret_topk_index = self.hot_token_id[ret_topk_index]

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )
        if self.sparse_rs:
            next_draft_input.draft_support_probs = ret_support_probs
            next_draft_input.draft_support_tokens = ret_support_tokens
            next_draft_input.draft_probs = None
        elif get_spec().speculative_use_rejection_sampling:
            next_draft_input.draft_probs = ret_draft_probs
        if self.seed_dsa_topk_from_draft_extend:
            next_draft_input.dsa_topk_indices = dsa_seed_topk_indices


def _adaptive_split() -> frozenset:
    """Debug-only construction-step switches, see build_adaptive_runtime_state."""
    raw = os.environ.get("SGLANG_ADAPTIVE_SPLIT", "")
    if not raw:
        return frozenset()
    parts = frozenset(p.strip() for p in raw.split(",") if p.strip())
    logger.warning("SGLANG_ADAPTIVE_SPLIT=%s: extra spec states are DEBUG ONLY", raw)
    return parts


class EAGLEWorkerV2(BaseSpecWorker):
    # Off by default for subclasses that skip this __init__ (NgramChainWorkerV2, StandaloneWorkerV2).
    _hostw_enabled: bool = False

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()

        # Parse arguments
        self.server_args = server_args
        self.topk = get_spec().speculative_eagle_topk
        self.speculative_num_steps = get_spec().speculative_num_steps
        self.speculative_num_draft_tokens = get_spec().speculative_num_draft_tokens
        self.ps = ps
        self.gpu_id = gpu_id
        self.device = get_device().device
        self._target_worker = target_worker
        self.page_size = get_schedule().page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            ps,
            nccl_port,
            target_worker,
        )

        # C1 offline trace (SGLANG_ADAPTIVE_TRACE=<path>), debug only.
        self._chain_tracer: Optional[ChainTracer] = None
        if chain_trace_enabled():
            import os as _os

            self._chain_tracer = ChainTracer(
                _os.environ["SGLANG_ADAPTIVE_TRACE"]
                + f".rank{getattr(self, 'tp_rank', 0)}"
            )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if get_spec().speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=get_spec().speculative_adaptive_config,
            )

        self._hostw_enabled = dl_hostw_enabled()
        if self._hostw_enabled and self.adaptive_controller is None:
            raise ValueError("HOSTW requires --speculative-adaptive runtime states")

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    @property
    def last_shared_read_runner(self):
        # Per the base contract: the step's last shared-buffer-reading phase is
        # draft_extend, which runs on the draft runner.
        return self._draft_worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def validate_dl_length_storage(self):
        dw = self._draft_worker
        if not dw._dl_enabled:
            return
        from sglang.srt.speculative.vt_kernels import (
            assert_shared_length_storage,
            length_buffer,
        )

        target_lengths = [
            module._vt_lengths
            for module in self._target_worker.model_runner.model.modules()
            if getattr(module, "_vt_lengths", None) is not None
        ]
        if not target_lengths:
            raise RuntimeError("DL target has no registered length storage")
        sampler_lengths = length_buffer(dw._vt_conf_buf.device)
        assert_shared_length_storage(dw._dl_lengths, sampler_lengths, *target_lengths)

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        # Build adaptive runtime states after target and draft backends exist.
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                target_model_runner = self._target_worker.model_runner
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=target_model_runner.attn_backend,
                        target_graph_runner=(
                            target_model_runner.decode_cuda_graph_runner
                        ),
                        draft_extend_attn_backend=(
                            self._draft_worker.draft_extend_attn_backend
                        ),
                        cuda_graph_runner_for_draft_extend=(
                            self._draft_worker.cuda_graph_runner_for_draft_extend
                        ),
                        qsa_mtp_shared_sparse_indices=(
                            self._draft_worker.qsa_mtp_shared_sparse_indices
                        ),
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
                        else get_batch_sizes_to_capture(
                            target_model_runner, self.speculative_num_draft_tokens
                        )[0]
                    ),
                    max_batch_size=target_model_runner.max_running_requests,
                )
        # One startup line for the 2-D draft path, only when one of its parts is on, so a production log shows
        # which path the entry took (the verification harness reads the same attributes).
        stepup = os.environ.get("SGLANG_RND_DL_STEPUP", "0") == "1"
        dl = bool(getattr(self._draft_worker, "_dl_enabled", False))
        vt = bool(getattr(self._draft_worker, "_vt_enabled", False))
        hostw = bool(getattr(self, "_hostw_enabled", False))
        if dl or vt or stepup or hostw:
            widths = sorted(self.adaptive_controller._states) if self.adaptive_controller is not None else []
            logger.info(
                f"2-D tree: dl={int(dl)} vt={int(vt)} stepup={int(stepup)} hostw={int(hostw)} widths={widths}"
            )

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None, grammar_barrier=None
    ):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Target prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch_output = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=target_capture_mode
            )

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # Extend processed L prompt tokens; next verify iter expects same L.
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft_extend so the fence is at target-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Draft prefill
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
                return batch_output
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                    self.draft_worker.draft_runner
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=hidden_size,
                    dtype=hidden_dtype,
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                    vocab_size=self.target_worker.model_config.vocab_size,
                )
            if self.speculative_num_steps == 0:
                # Drafting disabled (high batch size). _draft_extend below still
                # runs, keeping draft KV warm for when the batch shrinks.
                verify_input = self._build_trivial_verify_input(batch)
            else:
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    if self._hostw_enabled:
                        verify_input = self.draft_worker.draft(
                            batch, hostw_states=self.adaptive_controller._states
                        )
                    else:
                        verify_input = self.draft_worker.draft(batch)
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            with self._hostw_runtime_state(verify_input.spec_steps):
                batch_output = self.verify(batch, grammar_barrier=grammar_barrier)
                # Publish before draft_extend so the fence is at verify-end.
                if on_publish is not None:
                    on_publish(batch_output.new_seq_lens)
                if (
                    self.speculative_num_steps == 0
                    and envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.get()
                ):
                    self.draft_worker._update_context_from_verify(batch, batch_output)
                    self._stub_skipped_draft_extend(batch, batch_output)
                else:
                    with (
                        self.draft_worker.draft_tp_context(
                            self.draft_worker.draft_runner.tp_group
                        ),
                        speculative_moe_backend_context(),
                        speculative_moe_a2a_backend_context(),
                        spec_stage_span("draft_extend"),
                    ):
                        self.draft_worker._draft_extend_for_decode(batch, batch_output)

            return batch_output

    @contextlib.contextmanager
    def _hostw_runtime_state(self, verify_steps):
        if not self._hostw_enabled or verify_steps == self.speculative_num_steps:
            yield
            return
        execution_state = self.adaptive_controller._states[self.speculative_num_steps]
        verify_state = self.adaptive_controller._states[verify_steps]
        # Extend must consume the same row stride as verify. Chain buffers are
        # unused in these phases; keep their pointers for the next S-wide draft.
        try:
            self.apply_runtime_state(verify_state, rebuild_chain=False)
            yield
        finally:
            self.apply_runtime_state(execution_state, rebuild_chain=False)

    def _build_trivial_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a 1-node EagleVerifyInput rooted at the previous bonus token.

        Used when ``speculative_num_steps == 0`` to skip drafting while still
        routing through the existing TARGET_VERIFY graph captured at
        ``draft_token_num=1``: the kernel always accepts the root and samples
        one new bonus token from target logits -- functionally a plain decode.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=self.topk, spec_steps=0, num_verify_tokens=1, device=self.device
            )

        draft_input: EagleDraftInput = batch.spec_info
        bs = batch.seq_lens.shape[0]
        device = self.device

        retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

        attn_backend = self._target_worker.model_runner.attn_backend
        verify_mask = attn_backend.verify_mask
        # Every position in a 1-node tree is visible, so an all-True fill is
        # correct under either layout.
        if verify_mask is not None and verify_mask.fits(bs):
            custom_mask = verify_mask.buffer
            custom_mask.fill_(True)
        else:
            if batch.seq_lens_sum is not None:
                seq_lens_sum = batch.seq_lens_sum
            elif batch.seq_lens_cpu is not None:
                seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                seq_lens_sum = bs * attn_backend.max_context_len
            custom_mask = torch.ones(seq_lens_sum + bs, dtype=torch.bool, device=device)

        positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_input.bonus_tokens,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def _stub_skipped_draft_extend(
        self, batch: ScheduleBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Fill shape-valid stubs on next_draft_input when draft_extend is skipped.

        ``verify`` already set ``bonus_tokens`` (the only field the next steps=0
        verify reads). The overlap FutureMap still stashes topk_p/topk_index/
        hidden_states, so provide zeroed tensors of the right shape. They are never
        consumed while at steps=0; an upshift to steps>0 would draft from this stale
        state (cold recovery), which is the documented cost of this experimental flag.
        """
        next_draft_input: EagleDraftInput = batch_output.next_draft_input
        bs = batch.seq_lens.shape[0]
        device = self.device
        next_draft_input.topk_p = torch.zeros(
            (bs, self.topk), dtype=torch.float32, device=device
        )
        next_draft_input.topk_index = torch.zeros(
            (bs, self.topk), dtype=torch.int64, device=device
        )
        hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
            self.draft_worker.draft_runner
        )
        if hidden_size is not None:
            next_draft_input.hidden_states = torch.zeros(
                (bs, hidden_size),
                dtype=hidden_dtype,
                device=device,
            )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        if self._chain_tracer is not None:
            channel = self.draft_worker._conf_channel
            popped = channel.pop_chain() if channel is not None else None
            chain, steps = popped if popped is not None else (None, 0)
            self._chain_tracer.write(
                steps=steps,
                bs=batch_size,
                chain=chain,
                accepted=num_correct_drafts_per_req,
            )
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is None:
            return
        # The confidence of the chain about to be drafted was staged during the
        # previous iteration's draft-extend, so this event is already complete.
        channel = self.draft_worker._conf_channel
        if channel is not None:
            confidences = channel.latest_position0()
            if confidences:
                self.adaptive_controller.observe_confidence(confidences, batch_size)
        self.adaptive_controller.activate_step_by_batch(batch_size)

    # -- Adaptive speculative decoding protocol --

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            # Extra runtime states must not share the draft runner's global
            # FlashInfer/attention workspace: plans captured into a shared
            # workspace by this state's graphs would overwrite the plan data
            # the initial state's graphs read at replay.
            draft_runner = self._draft_worker.draft_runner
            backup_draft_ws = getattr(draft_runner, "init_new_workspace", False)
            draft_runner.init_new_workspace = True
            from sglang.srt.model_executor.input_buffers import (
                set_private_input_buffers,
            )

            backup_private = set_private_input_buffers(True)
            # Debug-only bisection switches (SGLANG_ADAPTIVE_SPLIT, comma
            # separated). Building an extra runtime state measurably degrades
            # the *base* state's long-chain acceptance on qwen4_exp even when
            # the extra state is never activated; these let one server session
            # attribute that to a single construction step. Never set in
            # production -- each one leaves the extra state unusable.
            split = _adaptive_split()
            try:
                if "no_draft" not in split:
                    if "shared_extend" not in split:
                        # Every state needs its OWN draft-extend backend.
                        # DraftBackendFactory.create_draft_extend_backend()
                        # returns draft_model_runner.attn_backend for
                        # compressed-QSA draft models, so without this every
                        # state's draft-extend runner would alias one backend
                        # -- and QwenSparseAttnBackend caches its captured
                        # graph metadata in a dict keyed only by
                        # (forward_mode, bs). This state's capture would then
                        # replace the launch state's DRAFT_EXTEND_V2 entry with
                        # one sliced to *this* state's narrower token width, so
                        # the launch state would replay its 16-wide graph after
                        # filling only 4 rows of metadata: no crash, but the
                        # sparse selection seeding draft positions past the
                        # short state's width is stale, and long-chain
                        # acceptance collapses (measured: code-edit acc 10.8 ->
                        # 5.7 with the steps=3 state merely built).
                        # _override_worker_state restores the base backend.
                        draft_runner.attn_backend = (
                            draft_runner._get_attention_backend(
                                init_new_workspace=True
                            )
                        )
                    self._draft_worker.init_attention_backend()
                    if "no_draft_graphs" not in split:
                        self._draft_worker._capture_cuda_graphs()
            finally:
                draft_runner.init_new_workspace = backup_draft_ws
                set_private_input_buffers(backup_private)

            # Build target attention backend and CUDA graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(
                Phase.DECODE, Backend.DISABLED
            ) and "no_target_graphs" not in split:
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_before_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_tic = time.perf_counter()
                from sglang.srt.model_executor.input_buffers import (
                    set_private_input_buffers as _set_private,
                )

                _backup_private_t = _set_private(True)
                with adaptive_target_graph_warmup(
                    target_model_runner, target_attn_backend
                ):
                    target_graph_runner = TargetGraphRunnerCls(
                        target_model_runner,
                        attn_backend=target_attn_backend,
                        speculative_num_steps=speculative_num_steps,
                        speculative_num_draft_tokens=speculative_num_draft_tokens,
                    )
                if "no_gdn_recovery" not in split:
                    target_model_runner.maybe_capture_gdn_recovery_graphs(
                        attn_backend=target_attn_backend,
                        capture_bs=target_graph_runner.capture_bs,
                    )
                _set_private(_backup_private_t)
                target_graph_after_mem = get_available_gpu_memory(
                    self.device, self.gpu_id
                )
                target_graph_time = time.perf_counter() - target_graph_tic
                self._additional_graph_memory_usage["target_verify"] = (
                    self._additional_graph_memory_usage.get("target_verify", 0.0)
                    + target_graph_before_mem
                    - target_graph_after_mem
                )
                self._additional_graph_time_usage["target_verify"] = (
                    self._additional_graph_time_usage.get("target_verify", 0.0)
                    + target_graph_time
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=(
                    self._draft_worker.cuda_graph_runner_for_draft_extend
                ),
                qsa_mtp_shared_sparse_indices=(
                    self._draft_worker.qsa_mtp_shared_sparse_indices
                ),
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState, *, rebuild_chain=True) -> None:
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        dw = self._draft_worker
        outgoing_backends = (
            self._target_worker.model_runner.attn_backend,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.attn_backend,
        )
        drained_backend_ids = set()
        for backend in outgoing_backends:
            if (
                isinstance(backend, HybridLinearAttnBackend)
                and id(backend) not in drained_backend_ids
            ):
                backend.drain_pending_recovery()
                drained_backend_ids.add(id(backend))

        if rebuild_chain:
            log_info_on_rank0(
                logger,
                "Switch adaptive runtime state: "
                f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
                f"draft_tokens {self.speculative_num_draft_tokens} -> "
                f"{state.speculative_num_draft_tokens}",
            )

        # Top-level
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        # Keep the runner's attn_backend in step with the active draft-extend
        # backend (the draft-extend forward reads draft_runner.attn_backend);
        # mirrors init_attention_backend. When None, the runner keeps its
        # initialized backend (consistent across step configs).
        if state.draft_extend_attn_backend is not None:
            dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw.qsa_mtp_shared_sparse_indices = state.qsa_mtp_shared_sparse_indices
        if state.qsa_mtp_shared_sparse_indices is not None:
            dw._install_qsa_mtp_index_share(
                state=state.qsa_mtp_shared_sparse_indices,
                draft_attn_backend=state.draft_attn_backend,
                draft_extend_attn_backend=state.draft_extend_attn_backend,
            )
        if rebuild_chain:
            dw._rebuild_topk1_chain_buffers()

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        # Sync server_args
        get_context().override(
            "adaptive_spec.restore",
            speculative_num_steps=state.speculative_num_steps,
            speculative_num_draft_tokens=state.speculative_num_draft_tokens,
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            dw.qsa_mtp_shared_sparse_indices,
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_config,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        get_context().override(
            "adaptive_spec.capture_override",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        if cuda_graph_bs is not None:
            graph_config = get_exec().graph.cuda_graph_config
            decode_config = (
                replace(graph_config.decode, backend=Backend.DISABLED)
                if not cuda_graph_bs
                else replace(graph_config.decode, bs=cuda_graph_bs)
            )
            get_context().override(
                "adaptive_spec.capture_override",
                cuda_graph_config=replace(
                    graph_config,
                    decode=decode_config,
                ),
            )
        dw._rebuild_topk1_chain_buffers()

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.draft_runner.attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
                dw.qsa_mtp_shared_sparse_indices,
            ) = backup[:11]
            if dw.qsa_mtp_shared_sparse_indices is not None:
                dw._install_qsa_mtp_index_share(
                    state=dw.qsa_mtp_shared_sparse_indices,
                    draft_attn_backend=dw.draft_attn_backend,
                    draft_extend_attn_backend=dw.draft_extend_attn_backend,
                )
            get_context().override(
                "adaptive_spec.capture_restore",
                speculative_num_steps=backup[11],
                speculative_num_draft_tokens=backup[12],
                cuda_graph_config=backup[13],
            )
            dw._rebuild_topk1_chain_buffers()

    def verify(self, batch: ScheduleBatch, grammar_barrier=None):
        batch.spec_info.draft_context_vocab = (
            getattr(self.draft_worker, "_draft_ctx_vocab", None) is not None
        )
        return run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
        )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.ps.tp_rank]
        )
        success, message = (
            self.draft_worker.draft_runner.weight_updater.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
        )
        if not success:
            return success, message

        success, message = (
            self.target_worker.model_runner.weight_updater.update_weights_from_tensor(
                named_tensors=named_tensors,
                load_format=recv_req.load_format,
            )
        )
        return success, message
