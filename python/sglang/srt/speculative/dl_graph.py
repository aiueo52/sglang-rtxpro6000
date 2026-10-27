# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Native torch CUDA-graph IF capture for the dynamic draft chain.

The caller owns the persistent one-element bool predicate and resets its chain
state on the captured root stream. Each sibling IF has a native device setter
that reads this predicate on replay and calls cudaGraphSetConditional. Torch
redirects capture to the IF body stream, retains its graph-private allocator
pool, and restores the root stream with a dependency on the completed IF.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


_NATIVE_METHODS = (
    "get_currently_capturing_graph",
    "begin_capture_to_if_node",
    "end_capture_to_conditional_node",
)


def require_conditional_support():
    """Check the native binding without initializing CUDA or building code."""
    if torch.version.hip or not torch.version.cuda or any(
        not callable(getattr(torch.cuda.CUDAGraph, name, None))
        for name in _NATIVE_METHODS
    ):
        raise RuntimeError(
            "DL requires CUDA and torch's native CUDA graph IF capture API; "
            "full-width fallback is not allowed"
        )


@contextmanager
def conditional_if(predicate: torch.Tensor):
    """Capture the enclosed forward and postprocessing as a CUDA graph IF.

    Outside capture, execute the body for startup warmup only. The production
    caller must reject graphless execution when DL is enabled: a Python context
    manager cannot skip a with-body without synchronizing the device. Keep the
    predicate alive with the graph and never replace it between replays.
    """
    require_conditional_support()
    if (
        not isinstance(predicate, torch.Tensor)
        or not predicate.is_cuda
        or predicate.dtype != torch.bool
        or predicate.numel() != 1
        or not predicate.is_contiguous()
    ):
        raise ValueError("DL IF predicate must be a contiguous one-element CUDA bool tensor")
    if not torch.cuda.is_current_stream_capturing():
        yield
        return
    if predicate.device != torch.cuda.current_stream().device:
        raise ValueError("DL IF predicate must be on the capture stream's device")
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    if graph is None:
        raise RuntimeError("DL requires a torch-owned CUDA graph capture")
    # Mirrors torch._higher_order_ops.cudagraph_conditional_nodes._if_body.
    # The native setter reads bool device storage, not a host capture-time value.
    # Native capture handles child streams, allocator filters, and dependencies;
    # raw cudaStreamBeginCaptureToGraph here would bypass those allocator hooks.
    graph.begin_capture_to_if_node(predicate)
    try:
        yield
    finally:
        graph.end_capture_to_conditional_node()
