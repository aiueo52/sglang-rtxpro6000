# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Fixed-size draft vocabulary with a static hot prefix and a context FIFO.

The reference implementation deliberately has no torch/sglang/triton imports.
GPU dependencies are imported only when constructing the device implementation.
"""

from __future__ import annotations

import operator
import os


def get_draft_context_topk(vocab_size):
    """Opt-in verify candidates; an inactive context FIFO ignores this flag."""
    if int(os.environ.get("SGLANG_RND_DRAFT_CTX_VOCAB", "0")) <= 0:
        return 0
    k = int(os.environ.get("SGLANG_RND_DRAFT_CTX_TOPK", "0"))
    if not 0 <= k <= _integer(vocab_size, "vocab_size"):
        raise ValueError("SGLANG_RND_DRAFT_CTX_TOPK must be in [0, vocab_size]")
    return k


def select_context_topk_rows(token_ids, accept_index, accept_lens):
    """Snapshot accepted node rows, including the bonus, without a host sync."""
    import torch

    if (
        token_ids.ndim != 2
        or token_ids.dtype not in (torch.int32, torch.int64)
        or token_ids.shape[1] <= 0
        or accept_index.ndim != 2
        or accept_index.dtype not in (torch.int32, torch.int64)
        or accept_lens.ndim != 1
        or accept_lens.dtype not in (torch.int32, torch.int64)
        or accept_lens.numel() != accept_index.shape[0]
        or accept_index.device != token_ids.device
        or accept_lens.device != token_ids.device
    ):
        raise ValueError("context top-k needs integer node rows, path indices and lengths")
    bs, depth = accept_index.shape
    if not bs:
        return token_ids.new_empty((0, depth, token_ids.shape[1]))
    if not token_ids.shape[0] or not depth:
        raise ValueError("nonempty accepted paths need target rows")
    torch._assert_async(
        ((accept_lens >= 0) & (accept_lens <= depth)).all(),
        "context top-k accepted length outside path width",
    )
    active = torch.arange(depth, device=token_ids.device)[None, :] < accept_lens[:, None]
    valid = (accept_index >= 0) & (accept_index < token_ids.shape[0])
    torch._assert_async((~active | valid).all(), "invalid accepted context top-k row")
    safe = accept_index.to(torch.int64).clamp(0, token_ids.shape[0] - 1)
    # Fancy indexing owns its storage: a later target graph replay cannot
    # overwrite the snapshot before draft-extend consumes it.
    selected = token_ids[safe]
    selected.masked_fill_(~active[:, :, None], -1)
    return selected


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _configuration(hot_ids, capacity, vocab_size):
    capacity = _integer(capacity, "capacity")
    vocab_size = _integer(vocab_size, "vocab_size")
    if capacity < 0 or vocab_size <= 0:
        raise ValueError("capacity must be nonnegative and vocab_size positive")
    hot_ids = tuple(_integer(token, "hot token") for token in hot_ids)
    if not hot_ids or len(set(hot_ids)) != len(hot_ids):
        raise ValueError("hot_ids must be nonempty and unique")
    if any(token < 0 or token >= vocab_size for token in hot_ids):
        raise ValueError("hot token outside vocabulary")
    if len(hot_ids) + capacity > vocab_size:
        raise ValueError("hot_ids length + capacity must not exceed vocab_size")
    return hot_ids, capacity, vocab_size


def _tokens(token_ids, vocab_size):
    tokens = tuple(_integer(token, "token") for token in token_ids)
    if any(token < 0 or token >= vocab_size for token in tokens):
        raise ValueError("token outside vocabulary")
    return tokens


def _prompt_tokens(token_ids, vocab_size):
    """Prompt ids without the ids outside the vocabulary.

    A multimodal prompt carries image/video placeholder (pad) values there. They
    are not tokens the draft can propose, so they are skipped, not rejected (the
    VLM warmup's image request stopped the scheduler on them). In-vocabulary
    input gives exactly the tokens ``_tokens`` gives.
    """
    tokens = tuple(_integer(token, "token") for token in token_ids)
    return tuple(token for token in tokens if 0 <= token < vocab_size)


class ContextTokenSlots:
    """CPU oracle: FIFO insertion order, not LRU; resident hits do not refresh.

    Updates are sequential in input/request order. An evicted token can be
    reinserted later in the same update. Empty tail map entries equal hot0;
    ``resident_tokens`` alone uses -1 to distinguish them from occupied slots.
    """

    def __init__(self, hot_ids, capacity, vocab_size):
        self.hot_ids, self.capacity, self.vocab_size = _configuration(
            hot_ids, capacity, vocab_size
        )
        self._hot_set = frozenset(self.hot_ids)
        self.resident_tokens = [-1] * self.capacity
        self.hot_token_id = list(self.hot_ids) + [self.hot_ids[0]] * self.capacity
        self.cursor = 0
        self.dirty = [False] * self.capacity

    def _update(self, tokens):
        self.dirty[:] = [False] * self.capacity
        for token in tokens:
            if (
                not self.capacity
                or token in self._hot_set
                or token in self.resident_tokens
            ):
                continue
            slot = self.cursor
            self.resident_tokens[slot] = token
            self.hot_token_id[len(self.hot_ids) + slot] = token
            self.dirty[slot] = True
            self.cursor = (slot + 1) % self.capacity

    def update_prefill(self, token_ids):
        self._update(_prompt_tokens(token_ids, self.vocab_size))

    def update_committed(self, ids, accept_lens=None, width=1):
        width = _integer(width, "width")
        if width <= 0:
            raise ValueError("width must be positive")
        ids = tuple(ids)
        if accept_lens is None:
            if len(ids) % width:
                raise ValueError("ids must contain width entries per request")
            lengths = (width,) * (len(ids) // width)
        else:
            lengths = tuple(
                _integer(length, "accepted length") for length in accept_lens
            )
        if len(ids) != len(lengths) * width:
            raise ValueError("ids must contain width entries per request")
        if any(length < 0 or length > width for length in lengths):
            raise ValueError("accepted length outside [0, width]")
        # Validate only accepted prefixes: rejected/padding entries may be -1.
        tokens = _tokens(
            (
                ids[row * width + column]
                for row, length in enumerate(lengths)
                for column in range(length)
            ),
            self.vocab_size,
        )
        self._update(tokens)

    def update_topk(self, token_ids, accept_lens):
        """CPU oracle for request/accepted-row/rank ordered candidate insertion."""
        batches = tuple(tuple(tuple(row) for row in req) for req in token_ids)
        lengths = tuple(_integer(length, "accepted length") for length in accept_lens)
        if len(batches) != len(lengths):
            raise ValueError("top-k rows must match accepted lengths")
        if batches:
            width = len(batches[0])
            k = len(batches[0][0]) if width else 0
            if not k or any(len(req) != width or any(len(row) != k for row in req) for req in batches):
                raise ValueError("top-k rows must have fixed positive row and rank widths")
        if any(length < 0 or length > len(req) for req, length in zip(batches, lengths)):
            raise ValueError("accepted length outside top-k row width")
        self._update(_tokens(
            (token for req, length in zip(batches, lengths) for row in req[:length] for token in row),
            self.vocab_size,
        ))

    def mask_logits(self, logits):
        if any(len(row) != len(self.hot_token_id) for row in logits):
            raise ValueError("logits width must match the fixed vocabulary")
        for row in logits:
            for slot, token in enumerate(self.resident_tokens):
                if token < 0:
                    row[len(self.hot_ids) + slot] = -float("inf")
        return logits


class DraftContextVocab:
    """Stable FP8 FIFO rows and token maps, optionally behind mixed-precision heads.

    Constructor validation may synchronize. Committed updates never transfer
    device data to the host, and all kernels use the caller's current stream.
    The source head/scales remain owned by the caller and must not be mutated.
    """

    def __init__(self, source_head, source_scale, hot_ids, capacity, *, separate_tail=False):
        import torch

        from sglang.kernels.ops.speculative.context_vocab import (
            mask_context_logits,
            update_context_map,
            update_context_vocab,
        )

        fp8_dtypes = tuple(
            getattr(torch, name)
            for name in (
                "float8_e4m3fn",
                "float8_e4m3fnuz",
                "float8_e5m2",
                "float8_e5m2fnuz",
            )
            if hasattr(torch, name)
        )
        if (
            source_head.ndim != 2
            or min(source_head.shape) <= 0
            or source_head.dtype not in fp8_dtypes
            or not source_head.is_cuda
            or source_head.stride(0) != 1
            or source_head.stride(1) < source_head.shape[0]
        ):
            raise ValueError("source_head must be transposed CUDA FP8 [hidden, vocab]")
        if (
            source_scale.ndim != 1
            or source_scale.numel() != source_head.shape[1]
            or source_scale.dtype != torch.float32
            or source_scale.device != source_head.device
            or not source_scale.is_contiguous()
        ):
            raise ValueError("source_scale must be contiguous CUDA float32 [vocab]")
        if (
            hot_ids.ndim != 1
            or hot_ids.dtype != torch.int64
            or hot_ids.device != source_head.device
            or not hot_ids.is_contiguous()
        ):
            raise ValueError("hot_ids must be contiguous CUDA int64 [hot]")
        self.hot_ids, self.capacity, self.vocab_size = _configuration(
            hot_ids.detach().cpu().tolist(), capacity, source_head.shape[1]
        )
        if not isinstance(separate_tail, bool):
            raise ValueError("separate_tail must be bool")
        self.separate_tail = separate_tail
        self.target_topk = get_draft_context_topk(self.vocab_size)
        self._hot_set = frozenset(self.hot_ids)
        self.source_head = source_head
        self.source_scale = source_scale
        self._update_device = update_context_vocab
        self._mask_device = mask_context_logits
        self._update_map_device = update_context_map
        self._extra_maps = []
        self._torch = torch
        self._fp8_dtypes = fp8_dtypes
        device = source_head.device
        hot_count = len(self.hot_ids)
        self.hot_token_id = torch.full(
            (hot_count + self.capacity,), self.hot_ids[0], dtype=torch.int64, device=device
        )
        self.hot_token_id[:hot_count].copy_(hot_ids)
        self._weight_offset = 0 if separate_tail else hot_count
        weight_ids = self.hot_token_id[hot_count:] if separate_tail else self.hot_token_id
        self.weight = source_head.t()[weight_ids].contiguous().t()
        self.weight_scale = source_scale[weight_ids].reshape(-1, 1).contiguous()
        self.tail_weight = self.weight[:, self._weight_offset:]
        self.tail_scale = self.weight_scale[self._weight_offset:]
        self._hot_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        self._hot_mask[hot_ids] = True
        self.resident_tokens = torch.full(
            (self.capacity,), -1, dtype=torch.int64, device=device
        )
        self.cursor = torch.zeros(1, dtype=torch.int64, device=device)
        self.dirty = torch.zeros(self.capacity, dtype=torch.int32, device=device)

    def _update(self, ids, accept_lens, width):
        self._update_device(
            ids,
            accept_lens,
            width,
            self._hot_mask,
            self.resident_tokens,
            self.cursor,
            self.dirty,
            self.source_head,
            self.source_scale,
            self.weight,
            self.weight_scale,
            self.hot_token_id,
            len(self.hot_ids),
            weight_offset=self._weight_offset,
        )
        for token_map, prefix_count in self._extra_maps:
            self._update_map_device(
                self.resident_tokens, self.dirty, token_map, prefix_count
            )

    def attach_head(self, head, token_ids):
        """Attach an immutable FP8/NVFP4 prefix and return its stable device map.

        The original method remains at ``head.quant_method.prefix_method`` on
        mixed heads. Their weight/scale attributes still describe only the
        prefix; every attached head shares this object's FP8 FIFO storage.
        Attach before capture, never to a head shared with the target.
        """
        torch = self._torch
        if isinstance(token_ids, torch.Tensor):
            if (
                token_ids.layout != torch.strided
                or token_ids.ndim != 1
                or token_ids.dtype not in (torch.int32, torch.int64)
                or token_ids.device.type not in ("cpu", "cuda")
                or (token_ids.is_cuda and token_ids.device != self.source_head.device)
            ):
                raise ValueError("prefix IDs must be 1-D CPU or same-device int32/int64")
            token_ids = token_ids.detach().cpu().tolist()
        prefix_ids, _, _ = _configuration(token_ids, self.capacity, self.vocab_size)
        existing = getattr(head, "_rnd_ctx_vocab", None)
        if existing is not None:
            if existing is self and prefix_ids == self.hot_ids:
                return self.hot_token_id
            if isinstance(existing, _ContextHeadView) and existing.owner is self and existing.prefix_ids == prefix_ids:
                return existing.hot_token_id
            raise ValueError("head is already attached to a context vocabulary")
        method = getattr(head, "quant_method", None)
        weight = getattr(head, "weight", None)
        if (
            not callable(getattr(method, "apply", None))
            or isinstance(method, _ContextVocabHeadMethod)
            or any(getattr(method, name, False) for name in ("use_marlin", "block_quant", "use_mxfp8"))
            or hasattr(head, "set_lora")
            or not isinstance(weight, torch.Tensor)
            or weight.layout != torch.strided
            or weight.ndim != 2
            or weight.device != self.source_head.device
        ):
            raise ValueError("context prefix requires an independent per-row FP8 or NVFP4 head")
        # Preserve the original single-GEMV extended FP8 path without wrapping.
        if (
            not self.separate_tail
            and prefix_ids == self.hot_ids
            and weight.dtype == self.weight.dtype
            and tuple(weight.shape) == tuple(self.weight.shape)
            and weight.data_ptr() == self.weight.data_ptr()
            and weight.stride() == self.weight.stride()
            and getattr(head, "weight_scale", None) is not None
            and head.weight_scale.data_ptr() == self.weight_scale.data_ptr()
        ):
            head._rnd_ctx_vocab = self
            return self.hot_token_id
        rows = len(prefix_ids)
        hidden = self.source_head.shape[0]
        if weight.dtype in self._fp8_dtypes:
            scale = getattr(head, "weight_scale", None)
            if (
                tuple(weight.shape) != (hidden, rows)
                or weight.stride(0) != 1
                or weight.stride(1) < hidden
                or not isinstance(scale, torch.Tensor)
                or scale.dtype != torch.float32
                or scale.numel() != rows
                or scale.device != weight.device
                or not scale.is_contiguous()
            ):
                raise ValueError("FP8 context prefix requires [hidden, prefix] and per-row FP32 scales")
        elif weight.dtype == torch.uint8:
            packed = getattr(head, "nvfp4_wq", None)
            scales = getattr(head, "nvfp4_bs", None)
            global_scale = getattr(head, "nvfp4_gscale", None)
            if (
                hidden % 16
                or tuple(weight.shape) != (rows, hidden // 2)
                or not isinstance(packed, torch.Tensor)
                or packed.data_ptr() != weight.data_ptr()
                or not isinstance(scales, torch.Tensor)
                or tuple(scales.shape) != (rows, hidden // 16)
                or scales.dtype != torch.uint8
                or not isinstance(global_scale, torch.Tensor)
                or global_scale.numel() != 1
                or global_scale.dtype != torch.float32
                or any(t.device != weight.device or not t.is_contiguous() for t in (packed, scales, global_scale))
            ):
                raise ValueError("NVFP4 context prefix requires matching packed weights and scales")
        else:
            raise ValueError("context prefix must retain FP8 or NVFP4 precision")
        if prefix_ids == self.hot_ids:
            token_map = self.hot_token_id
            prefix_mask = self._hot_mask
        else:
            token_map = torch.empty(rows + self.capacity, dtype=torch.int64, device=weight.device)
            ids = torch.tensor(prefix_ids, dtype=torch.int64, device=weight.device)
            token_map[:rows].copy_(ids)
            token_map[rows:].copy_(self.hot_token_id[len(self.hot_ids):])
            prefix_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=weight.device)
            prefix_mask[ids] = True
        view = _ContextHeadView(self, prefix_ids, token_map, prefix_mask)
        mixed_method = _ContextVocabHeadMethod(method, view)
        if self.capacity and self.tail_weight.dtype == torch.float8_e4m3fn:
            from sglang.srt.layers.quantization.w8a16_gemv import prealloc

            prealloc(weight.device)
        # Publish only after validation/allocation succeeds. Maps never change identity.
        head.quant_method = mixed_method
        head._rnd_ctx_vocab = view
        if token_map is not self.hot_token_id:
            self._extra_maps.append((token_map, rows))
        return token_map

    def update_prefill(self, token_ids):
        # Do not globally deduplicate: a token can reappear after FIFO eviction.
        tokens = [
            token
            for token in _prompt_tokens(token_ids, self.vocab_size)
            if token not in self._hot_set
        ]
        ids = self._torch.tensor(
            tokens, dtype=self._torch.int64, device=self.source_head.device
        )
        self._update(ids, None, max(1, len(tokens)))

    def update_committed(self, ids, accept_lens=None, width=1):
        self._update(ids, accept_lens, width)

    def update_topk(self, token_ids, accept_lens):
        """Insert accepted target candidates after committed tokens, on-device."""
        torch = self._torch
        if (
            token_ids.ndim != 3
            or token_ids.dtype not in (torch.int32, torch.int64)
            or token_ids.device != self.source_head.device
            or not token_ids.is_contiguous()
            or token_ids.shape[1] <= 0
            or token_ids.shape[2] <= 0
            or accept_lens.ndim != 1
            or accept_lens.dtype not in (torch.int32, torch.int64)
            or accept_lens.device != token_ids.device
            or accept_lens.numel() != token_ids.shape[0]
        ):
            raise ValueError("target top-k must be device integer [batch, rows, k] with accepted lengths")
        _, width, k = token_ids.shape
        self._update(token_ids.reshape(-1), (accept_lens * k).contiguous(), width * k)

    def mask_logits(self, logits):
        self._mask_device(logits, self.resident_tokens, len(self.hot_ids))
        return logits


class _ContextHeadView:
    """One immutable prefix's map/mask over the owner's shared FIFO."""

    def __init__(self, owner, prefix_ids, token_map, prefix_mask):
        self.owner = owner
        self.prefix_ids = prefix_ids
        self.prefix_count = len(prefix_ids)
        self.hot_token_id = token_map
        self.prefix_mask = prefix_mask

    def mask_logits(self, logits):
        self.owner._mask_device(
            logits, self.owner.resident_tokens, self.prefix_count, self.prefix_mask
        )
        return logits


class _ContextVocabHeadMethod:
    """Keep the prefix method; write prefix and shared FP8 tail into one output."""

    def __init__(self, prefix_method, view):
        self.prefix_method = prefix_method
        self.view = view

    def _validate_input(self, x):
        owner = self.view.owner
        torch = owner._torch
        if (
            x.ndim != 2
            or x.shape[1] != owner.source_head.shape[0]
            or x.device != owner.source_head.device
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            raise ValueError("mixed context head input must be device floating [batch, hidden]")

    def _compute(self, layer, x, out, bias):
        owner = self.view.owner
        torch = owner._torch
        prefix = self.view.prefix_count
        if bias is not None and (
            bias.ndim != 1
            or bias.numel() != self.view.hot_token_id.numel()
            or bias.device != x.device
            or not bias.is_floating_point()
        ):
            raise ValueError("mixed context head bias must cover prefix + capacity")
        if not x.shape[0]:
            return out
        prefix_out = out[:, :prefix]
        prefix_bias = bias[:prefix] if bias is not None else None
        apply_into = getattr(self.prefix_method, "apply_into", None)
        written = apply_into(layer, x, prefix_out, prefix_bias) if apply_into is not None else None
        if written is None:
            prefix_out.copy_(self.prefix_method.apply(layer, x, prefix_bias))
        if owner.capacity:
            tail_out = out[:, prefix:]
            if (
                x.is_cuda
                and x.dtype == torch.bfloat16
                and x.stride(1) == 1
                and x.shape[0] <= 256
                and owner.tail_weight.dtype == torch.float8_e4m3fn
                and out.dtype in (torch.bfloat16, torch.float32)
            ):
                from sglang.srt.layers.quantization.w8a16_gemv import w8a16_gemv

                for start in range(0, x.shape[0], 16):
                    w8a16_gemv(
                        x[start:start + 16], owner.tail_weight.t(), owner.tail_scale,
                        out=tail_out[start:start + 16],
                    )
            else:
                # Wide prefill/other activation dtypes keep the same source FP8
                # rows and scales; only the small FIFO is dequantized here.
                values = torch.mm(x.float(), owner.tail_weight.float())
                values.mul_(owner.tail_scale.reshape(1, -1))
                tail_out.copy_(values.to(dtype=x.dtype))
            if bias is not None:
                tail_out.add_(bias[prefix:])
        return out

    def apply(self, layer, x, bias=None):
        self._validate_input(x)
        torch = self.view.owner._torch
        dtype = torch.bfloat16 if layer.weight.dtype == torch.uint8 else x.dtype
        out = torch.empty(
            (x.shape[0], self.view.hot_token_id.numel()), dtype=dtype, device=x.device
        )
        return self._compute(layer, x, out, bias)

    def apply_into(self, layer, x, out, bias=None):
        self._validate_input(x)
        torch = self.view.owner._torch
        if (
            bias is not None
            or tuple(out.shape) != (x.shape[0], self.view.hot_token_id.numel())
            or out.dtype not in (torch.bfloat16, torch.float32)
            or out.device != x.device
            or out.stride(1) != 1
        ):
            return None
        return self._compute(layer, x, out, None)
