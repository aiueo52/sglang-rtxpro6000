# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Host-selected DL verify width. No accelerator imports."""
from __future__ import annotations


def validate_hostw_config(*, max_running_requests, decode_max_bs, capture_bs,
                          disable_padding, index_sharing):
    """HOSTW has no active-request mask or cross-width QSA seed transfer."""
    if (
        max_running_requests != 1 or decode_max_bs != 1
        or (capture_bs is not None and list(capture_bs) != [1])
        or not disable_padding
    ):
        raise ValueError("HOSTW requires bs=1, capture bs=[1] and --disable-cuda-graph-padding")
    if index_sharing:
        raise ValueError("HOSTW does not support index_share_for_mtp_iteration")


def select_verify_steps(length: int, available_steps, execution_steps: int) -> int:
    if (
        isinstance(length, bool) or not isinstance(length, int)
        or not 0 <= length <= execution_steps
    ):
        raise ValueError("DL length must be in [0, execution steps]")
    candidates = sorted(s for s in available_steps if 0 < s <= execution_steps)
    for steps in candidates:
        if steps >= length:
            return steps
    raise ValueError("HOSTW requires a registered verify state covering DL length")


def trim_draft_payload(steps, draft_tokens, draft_probs=None,
                       draft_support_probs=None, draft_support_tokens=None):
    """Request-major columns, excluding the root; own strided batch slices."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("HOSTW verify steps must be positive")
    if draft_tokens.ndim != 2 or steps > draft_tokens.shape[1]:
        raise ValueError("HOSTW draft tokens must contain the requested chain")
    payload = {"draft_tokens": draft_tokens[:, :steps].contiguous()}
    for name, tensor in (
        ("draft_probs", draft_probs),
        ("draft_support_probs", draft_support_probs),
        ("draft_support_tokens", draft_support_tokens),
    ):
        if tensor is not None:
            if (tensor.ndim != 3 or tensor.shape[:2] != draft_tokens.shape):
                raise ValueError(f"HOSTW {name} must match the draft chain")
            tensor = tensor[:, :steps, :].contiguous()
        payload[name] = tensor
    return payload


def read_max_length(lengths, batch_size):
    """One D2H scalar read, also for batches (never one read per request)."""
    if batch_size < 1 or lengths.ndim != 1 or lengths.shape[0] < batch_size:
        raise ValueError("HOSTW lengths must contain the nonempty batch")
    return int(lengths[:batch_size].max().item())
