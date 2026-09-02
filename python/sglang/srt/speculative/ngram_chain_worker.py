from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence

import msgspec
import torch
from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.runtime_context import get_device, get_schedule, get_spec
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    prepare_for_draft,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.srt.speculative.spec_utils import get_plan_stream


class NgramChainMatch(msgspec.Struct, frozen=True):
    tokens: tuple[int, ...]
    ngram_size: int
    position: int


class RequestNgramChainIndex:
    """Bounded request-local suffix index with full-width continuations only."""

    def __init__(
        self,
        *,
        min_size: int,
        max_size: int,
        window_size: int,
        num_steps: int,
    ) -> None:
        if min_size <= 0 or max_size < min_size:
            raise ValueError("ngram sizes must satisfy 0 < min_size <= max_size")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        self.min_size = min_size
        self.max_size = max_size
        self.window_size = window_size
        self.num_steps = num_steps
        self._tokens: list[int] = []
        self._storage_base = 0
        self._active_base = 0
        self._positions: dict[int, dict[tuple[int, ...], deque[int]]] = {
            n: defaultdict(deque) for n in range(min_size, max_size + 1)
        }

    def __len__(self) -> int:
        return self._end_position - self._active_base

    @property
    def tokens(self) -> tuple[int, ...]:
        offset = self._active_base - self._storage_base
        return tuple(self._tokens[offset:])

    @property
    def _end_position(self) -> int:
        return self._storage_base + len(self._tokens)

    def _slice(self, position: int, length: int) -> tuple[int, ...]:
        offset = position - self._storage_base
        return tuple(self._tokens[offset : offset + length])

    def seed(self, tokens: Iterable[int]) -> None:
        self.append(tokens)

    def append(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            self._tokens.append(int(token))
            end = self._end_position

            # An occurrence enters the lookup only after a complete continuation
            # exists. Thus the newest suffix never shadows an older usable match.
            for ngram_size in range(self.min_size, self.max_size + 1):
                position = end - self.num_steps - ngram_size
                if position < self._active_base:
                    continue
                key = self._slice(position, ngram_size)
                self._positions[ngram_size][key].append(position)

            self._evict_to_window()

    def _evict_to_window(self) -> None:
        while len(self) > self.window_size:
            position = self._active_base
            end = self._end_position
            for ngram_size in range(self.min_size, self.max_size + 1):
                if position + ngram_size + self.num_steps > end:
                    continue
                key = self._slice(position, ngram_size)
                positions = self._positions[ngram_size].get(key)
                if positions and positions[0] == position:
                    positions.popleft()
                    if not positions:
                        del self._positions[ngram_size][key]
            self._active_base += 1

        dead_prefix = self._active_base - self._storage_base
        if dead_prefix >= self.window_size:
            self._tokens = self._tokens[dead_prefix:]
            self._storage_base = self._active_base

    def propose(self) -> NgramChainMatch | None:
        end = self._end_position
        for ngram_size in range(self.max_size, self.min_size - 1, -1):
            if len(self) < ngram_size:
                continue
            key = self._slice(end - ngram_size, ngram_size)
            positions = self._positions[ngram_size].get(key)
            if not positions:
                continue
            position = positions[-1]
            continuation = self._slice(position + ngram_size, self.num_steps)
            if len(continuation) == self.num_steps:
                return NgramChainMatch(continuation, ngram_size, position)
        return None


class _PendingCommits(msgspec.Struct):
    request_ids: tuple[object, ...]
    tokens: torch.Tensor
    accept_lens: torch.Tensor
    stride: int


class NgramChainDraftWorker(EagleDraftWorker):
    """EAGLE draft worker that bypasses MTP on full request-local matches."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        spec = get_spec()
        self.ngram_chain_min_size = spec.speculative_ngram_chain_min_size
        self.ngram_chain_max_size = spec.speculative_ngram_chain_max_size
        self.ngram_chain_window_size = spec.speculative_ngram_chain_window_size
        self._request_indexes: dict[object, RequestNgramChainIndex] = {}
        self._pending_commits: deque[_PendingCommits] = deque()

    def clear_request_indexes(self) -> None:
        self._request_indexes.clear()
        self._pending_commits.clear()

    def seed_requests(self, requests: Sequence[object]) -> None:
        for request in requests:
            if request.rid in self._request_indexes:
                continue
            index = self._new_index()
            index.seed(request.origin_input_ids)
            index.append(request.output_ids)
            self._request_indexes[request.rid] = index

    def stage_commits(
        self,
        requests: Sequence[object],
        tokens: torch.Tensor,
        accept_lens: torch.Tensor,
        stride: int,
    ) -> None:
        self._pending_commits.append(
            _PendingCommits(
                request_ids=tuple(request.rid for request in requests),
                tokens=tokens.detach().clone(),
                accept_lens=accept_lens.detach().clone(),
                stride=stride,
            )
        )

    def _new_index(self) -> RequestNgramChainIndex:
        return RequestNgramChainIndex(
            min_size=self.ngram_chain_min_size,
            max_size=self.ngram_chain_max_size,
            window_size=self.ngram_chain_window_size,
            num_steps=self.speculative_num_steps,
        )

    def _flush_pending_commits(self) -> None:
        while self._pending_commits:
            pending = self._pending_commits.popleft()
            tokens = pending.tokens.cpu().tolist()
            accept_lens = pending.accept_lens.cpu().tolist()
            for row, (request_id, num_accept_tokens) in enumerate(
                zip(pending.request_ids, accept_lens)
            ):
                index = self._request_indexes.get(request_id)
                if index is None:
                    continue
                start = row * pending.stride
                index.append(tokens[start : start + num_accept_tokens])

    def _drop_departed_requests(self, requests: Sequence[object]) -> None:
        active_ids = {request.rid for request in requests}
        for request_id in self._request_indexes.keys() - active_ids:
            del self._request_indexes[request_id]

    def _find_batch_matches(
        self, requests: Sequence[object]
    ) -> list[NgramChainMatch] | None:
        matches = []
        for request in requests:
            match = self._request_indexes[request.rid].propose()
            if match is None:
                return None
            matches.append(match)
        return matches

    def _build_ngram_contract(
        self,
        proposals: Sequence[Sequence[int]],
        *,
        vocab_size: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device if device is None else device
        draft_tokens = torch.tensor(proposals, dtype=torch.long, device=device)
        if (
            draft_tokens.ndim != 2
            or draft_tokens.shape[1] != self.speculative_num_steps
        ):
            raise ValueError(
                "NGRAM_CHAIN proposals must be a full [batch, speculative_num_steps] chain"
            )
        bs = draft_tokens.shape[0]
        if bs > self._topk1_parents_prealloc.shape[0]:
            raise ValueError("NGRAM_CHAIN batch exceeds topk=1 preallocation")
        draft_probs = torch.zeros(
            (bs, self.speculative_num_steps, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        draft_probs.scatter_(2, draft_tokens.unsqueeze(-1), 1.0)
        return (
            self._topk1_parents_prealloc[:bs],
            self._topk1_score_indices_prealloc[:bs],
            draft_tokens,
            draft_probs,
        )

    def _draft_mtp_chain(self, batch):
        return super().draft(batch)

    def draft(self, batch):
        if batch.forward_mode.is_idle():
            return self._draft_mtp_chain(batch)

        self._flush_pending_commits()
        self.seed_requests(batch.reqs)
        self._drop_departed_requests(batch.reqs)
        matches = self._find_batch_matches(batch.reqs)
        if matches is None:
            # v0 keeps one tree shape per batch: any miss (including a partial
            # continuation) sends the whole mixed batch through unchanged MTP.
            return self._draft_mtp_chain(batch)

        draft_input = batch.spec_info
        # Preserve the normal draft-cache location bookkeeping. No draft model
        # forward or draft CUDA graph is executed on this path.
        prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        parent_list, top_scores_index, draft_tokens, draft_probs = (
            self._build_ngram_contract(
                [match.tokens for match in matches],
                vocab_size=self.target_worker.model_config.vocab_size,
            )
        )
        return build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            draft_tokens,
            draft_probs,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )


class NgramChainWorkerV2(EAGLEWorkerV2):
    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker) -> None:
        # Mirror EAGLEWorkerV2 setup so only one draft model is constructed.
        BaseSpecWorker.__init__(self)
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
        self._draft_worker = NgramChainDraftWorker(
            server_args, gpu_id, ps, nccl_port, target_worker
        )
        self.adaptive_controller = None
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)

    def clear_cache_pool(self):
        super().clear_cache_pool()
        self._draft_worker.clear_request_indexes()

    def forward_batch_generation(self, batch, on_publish=None, grammar_barrier=None):
        was_extend = batch.forward_mode.is_extend() or batch.is_extend_in_batch
        was_idle = batch.forward_mode.is_idle()
        if was_extend and not was_idle:
            self._draft_worker.seed_requests(batch.reqs)

        batch_output = super().forward_batch_generation(
            batch, on_publish=on_publish, grammar_barrier=grammar_barrier
        )
        if was_idle:
            return batch_output

        if was_extend:
            # Middle chunk outputs are not committed by the result processor.
            accept_lens = torch.tensor(
                [
                    0 if request.inflight_middle_chunks > 0 else 1
                    for request in batch.reqs
                ],
                dtype=torch.int32,
                device=batch_output.next_token_ids.device,
            )
            stride = 1
        else:
            accept_lens = batch_output.accept_lens
            stride = batch_output.speculative_num_draft_tokens
        self._draft_worker.stage_commits(
            batch.reqs, batch_output.next_token_ids, accept_lens, stride
        )
        return batch_output


class NgramChainAlgorithm(CustomSpecAlgo):
    def is_eagle(self) -> bool:
        return True

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device,
            self,
            req_to_token_pool,
            needs_cpu_seq_lens,
            needs_confidence_relay,
        )

    def __getattr__(self, name: str):
        # The SpeculativeAlgorithm enum grows helper methods over time; NGRAM_CHAIN
        # is EAGLE-shaped, so fall back to EAGLE's implementation for anything the
        # custom-algorithm base class does not define.
        if name.startswith("__"):
            raise AttributeError(name)
        eagle_attr = getattr(SpeculativeAlgorithm.EAGLE, name)
        if callable(eagle_attr):
            return eagle_attr
        return eagle_attr

    def supports_grammar_overlap(self) -> bool:
        return True

    def carries_draft_hidden_states(self) -> bool:
        return True

    def handle_server_args(self, server_args) -> None:
        from sglang.srt.arg_groups.speculative_hook import _handle_eagle_family

        cfg = resolving_view(server_args)
        if cfg.speculative_adaptive:
            raise ValueError("NGRAM_CHAIN does not support adaptive speculative steps")
        if cfg.speculative_eagle_topk is None:
            declare_resolution(
                server_args,
                "NgramChainAlgorithm.handle_server_args",
                speculative_eagle_topk=1,
            )
        elif cfg.speculative_eagle_topk != 1:
            raise ValueError("NGRAM_CHAIN requires --speculative-eagle-topk 1")

        _handle_eagle_family(server_args)
        cfg = resolving_view(server_args)
        if cfg.speculative_ngram_chain_min_size <= 0:
            raise ValueError("--speculative-ngram-chain-min-size must be positive")
        if cfg.speculative_ngram_chain_max_size < cfg.speculative_ngram_chain_min_size:
            raise ValueError(
                "--speculative-ngram-chain-max-size must be >= "
                "--speculative-ngram-chain-min-size"
            )
        if cfg.speculative_ngram_chain_window_size <= 0:
            raise ValueError("--speculative-ngram-chain-window-size must be positive")


@SpeculativeAlgorithm.register(
    "NGRAM_CHAIN", supports_overlap=True, spec_class=NgramChainAlgorithm
)
def _ngram_chain_worker_factory(server_args):
    return NgramChainWorkerV2
