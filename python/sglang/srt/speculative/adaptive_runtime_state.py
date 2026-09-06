from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sglang.srt.speculative.adaptive_spec_params import AdaptiveSpeculativeParams

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QSAMTPSharedSparseIndices,
    )
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )


def init_cuda_graph_state_no_shrink(
    attn_backend, max_bs: int, max_num_token: int
) -> None:
    """``init_cuda_graph_state`` that never shrinks an already-sized backend.

    Adaptive speculative decoding (``--speculative-adaptive``) builds one
    runtime state per candidate step count, each with its own draft-extend
    graph runner.  For compressed-QSA draft models
    ``DraftBackendFactory.create_draft_extend_backend()`` returns the draft
    runner's *own* attention backend (draft_utils.py), so **every** state's
    draft-extend runner shares one backend object.  ``init_cuda_graph_state``
    reallocates that backend's metadata tensors (``_graph_seq_lens``,
    ``_graph_compressed_page_table``, ...).  Re-running it for a smaller
    candidate frees exactly the tensors the already-captured graphs of the
    larger state baked in, so their next replay touches freed memory ->
    ``illegal memory access``.

    Sharing the (larger) tensors between states is safe: the metadata is
    written immediately before each capture/replay and only one state is
    active at a time -- the same argument ``share_input_buffer`` makes for the
    forward input buffers.  So allocate only when the request does not already
    fit.  For a non-adaptive server this runs exactly once and is a no-op
    change.

    A candidate *larger* than what is already allocated would still have to
    reallocate (and would break the earlier graphs); ``AdaptiveController``
    rejects such configs up front, and the assertion here documents the
    invariant.
    """
    from sglang.srt.runtime_context import get_spec

    if not get_spec().speculative_adaptive:
        # Single runtime state: the backend is sized exactly once, so keep the
        # upstream call (and do not stamp anything onto the backend).
        attn_backend.init_cuda_graph_state(max_bs, max_num_token)
        return
    prev = getattr(attn_backend, "_spec_graph_state_extent", None)
    if prev is not None and prev[0] >= max_bs and prev[1] >= max_num_token:
        return
    assert prev is None, (
        "draft-extend attention backend is shared across speculative runtime "
        f"states and is being re-sized upward ({prev} -> {(max_bs, max_num_token)}); "
        "this frees buffers already captured into earlier CUDA graphs"
    )
    attn_backend.init_cuda_graph_state(max_bs, max_num_token)
    attn_backend._spec_graph_state_extent = (max_bs, max_num_token)


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "DecodeCudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"
    qsa_mtp_shared_sparse_indices: "QSAMTPSharedSparseIndices | None"


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements AdaptiveSpecWorker protocol:
      - build_adaptive_runtime_state(steps, draft_tokens) → runtime state
      - apply_runtime_state(state) → apply it to the worker

    The worker only needs to:
      1. Call register() for the initial state, then init_states()
         once during startup.
      2. Call on_verify_complete(num_correct_drafts_per_req) after each decode verify.
    """

    def __init__(self, worker: AdaptiveSpecWorker, config_path: str | None = None):
        self.worker = worker
        self.initial_steps = worker.speculative_num_steps
        self.params = AdaptiveSpeculativeParams(
            initial_steps=worker.speculative_num_steps,
            cfg_path=config_path,
        )
        self._states: dict[int, SpecRuntimeState] = {}
        self._pending_steps: int | None = None

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to state.speculative_num_steps when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(
        self,
        cuda_graph_bs: list[int] | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        """Build and register runtime states for all candidate steps."""
        self.validate_candidates()
        self.params.set_cuda_graph_bs(cuda_graph_bs)

        for steps in self.candidate_steps:
            if steps in self._states:
                continue

            pruned_bs = self.params.cuda_graph_bs_for_step(steps)
            if (
                pruned_bs == []
                and max_batch_size is not None
                and not self.params.can_reach_step(
                    step=steps, max_batch_size=max_batch_size
                )
            ):
                continue
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

    def validate_candidates(self) -> None:
        """No candidate may exceed the step count the server started with.

        Everything sized once at start-up from ``--speculative-num-steps`` --
        the per-request KV / mamba reservation, the chain buffers, and (for
        draft models whose draft-extend backend is the draft runner's own
        backend, i.e. compressed QSA) that backend's CUDA-graph metadata
        tensors -- is dimensioned for the launch value.  A larger candidate
        would have to grow those buffers *after* the initial state's graphs
        were captured against them, which frees what those graphs point at
        (illegal memory access at the next replay).  Raise at start-up instead.
        """
        over = [s for s in self.candidate_steps if s > self.initial_steps]
        if over:
            raise ValueError(
                f"speculative_adaptive_config candidate_steps {sorted(over)} exceed "
                f"--speculative-num-steps ({self.initial_steps}). Launch the server "
                "with the largest candidate as --speculative-num-steps and list the "
                "smaller ones as candidates."
            )

    def observe_confidence(self, confidences: list[float], batch_size: int) -> None:
        """Draft confidence for the chain that is about to be drafted."""
        self.params.observe_confidence(confidences, batch_size)

    def activate_step_by_batch(self, batch_size: int) -> None:
        target = (
            self._pending_steps
            if self._pending_steps is not None
            else self.params.get_steps_for_batch(batch_size)
        )
        self._pending_steps = None
        if target != self.worker.speculative_num_steps:
            self._activate(target)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """Feed verify results and record any EMA step decision."""
        new_step = self.params.on_verify_complete(
            num_correct_drafts_per_req, batch_size
        )
        if new_step is not None:
            self._pending_steps = new_step

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
