# Modified by aiueo52 for flash-next-fast (2026); see MODIFICATIONS.md.

"""Per-forward-call control context.

Owns ForwardContext — a frozen dataclass holding control configs the model
layer reads at depth via get_forward_context(). The only mandatory field
today is attn_backend; pool refs are derived from attn_backend.*
(every backend caches them at __init__), so a published ForwardContext
is enough to resolve the active pools without a separate global.

ModelRunner._forward_raw publishes a fresh ForwardContext for the
duration of each forward; callers that need a per-call override (PDmux
per-stream backend, frozen-KV MTP draft loop, TBO per-child dispatch) use
dataclasses.replace and wrap the override scope with forward_context().

Distinct from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.TcPiecewiseForwardContext,
which collects compilation-time refs for the piecewise CUDA graph backend.

Concurrency: _current is a plain module-level global, not thread-local.
This matches the global_server_args precedent and is safe because each
forward runs synchronously on a single Python thread per worker process. If
worker threads ever share a process, migrate to contextvars.ContextVar.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs. Read via get_forward_context();
    extend by adding fields here. Frozen so accidental mutation raises at
    write time — use dataclasses.replace for per-call overrides."""

    attn_backend: AttentionBackend
    forward_mode: Optional[ForwardMode] = None


_current: Optional[ForwardContext] = None


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set the active context; return the previous one for explicit
    save/restore. Prefer the forward_context() context manager."""
    global _current
    prev, _current = _current, ctx
    return prev


def has_forward_context() -> bool:
    return _current is not None


def get_forward_context() -> ForwardContext:
    assert _current is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return _current


def get_attn_backend() -> AttentionBackend:
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> KVCache:
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool


def get_gemv_m_limit(max_m: int) -> int:
    """Only decode/target verify may opt into GEMV rows above the legacy 16."""
    mode = _current.forward_mode if _current is not None else None
    if mode is not None and (mode.is_decode() or mode.is_target_verify()):
        return max_m
    return min(16, max_m)


@contextmanager
def forward_batch_context(forward_batch):
    """Publish phase without changing the active (possibly draft) backend."""
    if _current is None:
        # Standalone callers without a runtime context keep the safe 16-row cap.
        yield
        return
    mode = getattr(forward_batch, "_original_forward_mode", None)
    if mode is None:
        mode = getattr(forward_batch, "forward_mode", None)
    ctx = replace(_current, forward_mode=mode)
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)


@contextmanager
def forward_context(ctx: ForwardContext):
    if ctx.forward_mode is None and _current is not None:
        ctx = replace(ctx, forward_mode=_current.forward_mode)
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
