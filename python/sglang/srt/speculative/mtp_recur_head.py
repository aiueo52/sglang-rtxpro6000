# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Load-time, draft-only recurrent MTP head and strict DECODE dispatch."""

import copy
import os

import torch
from torch import nn


def is_recur_head_mode(mode) -> bool:
    # ForwardMode.is_decode is equality with DECODE, not decode-or-idle.
    return mode.is_decode()


def validate_recur_token_map(token_map, *, vocab_size, first_head_rows, min_rows=1):
    """Return CPU int64 real-token IDs without silently converting bad IDs."""
    if isinstance(token_map, (list, tuple)):
        if not all(type(token) is int for token in token_map):
            raise ValueError("recurrent token map must contain integer IDs, not bools/floats")
        token_map = torch.tensor(token_map, dtype=torch.int64, device="cpu")
    if (
        not isinstance(token_map, torch.Tensor)
        or token_map.layout != torch.strided
        or token_map.device.type != "cpu"
        or token_map.dtype not in (torch.int32, torch.int64)
        or token_map.dim() != 1
    ):
        raise ValueError("recurrent token map must be a 1-D CPU int32/int64 tensor or integer list")
    if not 1 <= min_rows <= token_map.numel() < first_head_rows <= vocab_size:
        raise ValueError("recurrent token map must be smaller than the first head and large enough for proposal support")
    if token_map.min().item() < 0 or token_map.max().item() >= vocab_size:
        raise ValueError("recurrent token map ID outside target vocabulary")
    if token_map.unique().numel() != token_map.numel():
        raise ValueError("recurrent token map IDs must be unique")
    return token_map.detach().to(dtype=torch.int64).contiguous().clone()


def load_recur_token_map(path, *, vocab_size, first_head_rows, min_rows=1):
    return validate_recur_token_map(
        torch.load(path, weights_only=True, map_location="cpu"),
        vocab_size=vocab_size,
        first_head_rows=first_head_rows,
        min_rows=min_rows,
    )


def slice_fp8_head_rows(weight, weight_scale, token_ids, *, vocab_size, hidden_size):
    """Slice the target's transposed per-row FP8 head, preserving encoded bytes."""
    token_ids = validate_recur_token_map(
        token_ids, vocab_size=vocab_size, first_head_rows=vocab_size,
    )
    if (
        weight.layout != torch.strided
        or weight.dtype != torch.float8_e4m3fn
        or tuple(weight.shape) != (hidden_size, vocab_size)
        or not isinstance(weight_scale, torch.Tensor)
        or weight_scale.layout != torch.strided
        or weight_scale.dtype != torch.float32
        or tuple(weight_scale.shape) not in ((vocab_size,), (vocab_size, 1))
        or weight_scale.device != weight.device
    ):
        raise ValueError("recurrent FP8 head requires target [hidden, vocab] FP8 weights and FP32 row scales")
    ids = token_ids.to(device=weight.device, dtype=torch.int64)
    # uint8 indexing also works on CPU builds lacking FP8 index kernels; no QDQ.
    rows = weight.t().view(torch.uint8)[ids].contiguous().view(weight.dtype)
    scales = weight_scale.reshape(-1)[ids].unsqueeze(1).contiguous()
    return rows.t(), scales


def build_fp8_recur_head(target_head, token_ids, *, vocab_size, hidden_size):
    method = getattr(target_head, "quant_method", None)
    if not callable(getattr(method, "apply", None)) or any(
        getattr(method, flag, False) for flag in ("use_marlin", "block_quant", "use_mxfp8")
    ):
        raise ValueError("recurrent head requires the target's per-row FP8 linear method")
    weight, scales = slice_fp8_head_rows(
        target_head.weight,
        getattr(target_head, "weight_scale", None),
        token_ids,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    head = nn.Module()
    head.weight = nn.Parameter(weight, requires_grad=False)
    head.weight_scale = nn.Parameter(scales, requires_grad=False)
    head.input_scale = None
    head.use_flashinfer_bmm = False
    head.quant_method = method
    return head


def build_nvfp4_recur_head(model_dir, token_ids, *, vocab_size, hidden_size, device):
    from safetensors import safe_open

    from sglang.srt.layers.quantization.nvfp4_dense import (
        CACHE_DIR,
        NVFP4DenseLinearMethod,
        build_or_load,
        find_shard,
    )

    cache_dir = os.environ.get("SGLANG_NVFP4_CACHE_DIR")
    if not cache_dir or os.path.realpath(cache_dir) != os.path.realpath(CACHE_DIR):
        raise ValueError("recurrent NVFP4 requires a private SGLANG_NVFP4_CACHE_DIR set before quantization imports")
    token_ids = validate_recur_token_map(
        token_ids, vocab_size=vocab_size, first_head_rows=vocab_size,
    )
    if hidden_size % 128:
        raise ValueError("recurrent NVFP4 requires hidden size divisible by 128")
    # Check the source even on a cache hit: never repack the target's FP8 rows.
    shard = find_shard(model_dir, "lm_head.weight")
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        source = checkpoint.get_slice("lm_head.weight")
        if source.get_dtype() != "BF16" or tuple(source.get_shape()) != (vocab_size, hidden_size):
            raise ValueError("recurrent NVFP4 head requires a BF16 checkpoint lm_head")
    wq, bs, gscale = build_or_load(
        model_dir,
        "lm_head.weight",
        token_ids,
        device=device,
        label="MTP recurrent lm_head",
    )
    rows = token_ids.numel()
    if (
        wq.dtype != torch.uint8
        or tuple(wq.shape) != (rows, hidden_size // 2)
        or bs.dtype != torch.uint8
        or tuple(bs.shape) != (rows, hidden_size // 16)
        or bs.amax().item() > 126  # Nonnegative finite E4M3FN scale codes only.
        or gscale.dtype != torch.float32
        or tuple(gscale.shape) != (1,)
        or not torch.isfinite(gscale).all().item()
        or not (gscale > 0).all().item()
        or any(t.device != torch.device(device) or not t.is_contiguous() for t in (wq, bs, gscale))
    ):
        raise ValueError("invalid packed recurrent NVFP4 head")
    head = nn.Module()
    NVFP4DenseLinearMethod.attach(head, wq, bs, gscale)
    head.weight = wq
    head.weight_scale = None
    head.input_scale = None
    head.quant_method = NVFP4DenseLinearMethod("MTP recurrent lm_head")
    return head


class RecurrentHeadLogitsProcessor(nn.Module):
    """Keep the original processor/head for every non-DECODE forward.

    Mode, head pointers, and each processor's physical vocabulary width are
    fixed before capture. A differently sized graph buffer is declined without
    truncating recurrent logits or changing the draft-extend processor.
    """

    def __init__(self, base, recurrent_head):
        super().__init__()
        self.base = base
        self.recurrent_head = recurrent_head
        self.recurrent_processor = base
        if hasattr(base, "vocab_size"):
            self.recurrent_processor = copy.copy(base)
            ctxv = getattr(recurrent_head, "_rnd_ctx_vocab", None)
            if ctxv is not None:
                rows = ctxv.hot_token_id.numel()
            elif hasattr(recurrent_head, "nvfp4_wq"):
                rows = recurrent_head.nvfp4_wq.shape[0]
            elif recurrent_head.weight.dtype in (
                torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                torch.float8_e5m2, torch.float8_e5m2fnuz,
            ):
                rows = recurrent_head.weight.shape[1]
            else:
                rows = recurrent_head.weight.shape[0]
            self.recurrent_processor.vocab_size = rows

    def forward(self, input_ids, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        processor = self.base
        if is_recur_head_mode(logits_metadata.forward_mode):
            lm_head = self.recurrent_head
            processor = self.recurrent_processor
        return processor(input_ids, hidden_states, lm_head, logits_metadata, *args, **kwargs)
