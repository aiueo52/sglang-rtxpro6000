from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_hc_combine_module(
    hc_count: int, hidden_size: int, dtype: torch.dtype
) -> Module:
    """Compile and cache the JIT HC combine module for a given shape/dtype."""
    # Checks on the compile key live here, not in `hc_combine`: `cache_once`
    # keys on (hc_count, hidden_size, dtype), so this runs once per
    # specialisation instead of once per call.
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {dtype}. Supported: bfloat16, float16")
    if hidden_size <= 0 or hidden_size % 8 != 0:
        raise RuntimeError(
            f"Unsupported hidden_size {hidden_size}. Must be a multiple of 8."
        )
    if hc_count <= 0 or (hc_count * hidden_size) % 2048 != 0:
        raise RuntimeError(
            f"Unsupported hc_count * hidden_size {hc_count * hidden_size}. "
            "Must be a multiple of 2048."
        )
    args = make_cpp_args(hc_count, hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "hc_combine",
        *args,
        cuda_files=["elementwise/hc_combine.cuh"],
        cuda_wrappers=[
            ("hc_combine", f"HcCombineKernel<{args}>::run"),
            ("hc_combine_split", f"HcCombineSplitKernel<{args}>::run"),
            ("hc_combine_gate", f"HcCombineSplitKernel<{args}>::run_gate"),
            # One export per (shared fold, partial-slot count): the slot count
            # is 8 from the split gate kernel and hc_count from the HC-mix K0
            # epilogue, and unrolling that loop is worth ~0.4 us a call.
            *(
                (
                    f"hc_combine_apply_p{n}",
                    f"HcCombineApplyKernel<{args}, false, {n}>::run_plain",
                )
                for n in (hc_count, 8)
                if n == 8 or hc_count != 8
            ),
            *(
                (
                    f"hc_combine_apply_shared_p{n}",
                    f"HcCombineApplyKernel<{args}, true, {n}>::run",
                )
                for n in (hc_count, 8)
                if n == 8 or hc_count != 8
            ),
        ],
    )


def hc_combine(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused HyperConnection (gated residual) combine.

        a[m, c] = 2 * sigmoid(dot(normed_residual[m], inject_weight[c]) / hc_count)
        out[m, c*H + i] = residual[m, c*H + i] + a[m, c] * block_output[m, i]

    Mirrors ``GatedResidual._combine_compute`` in
    ``sglang.srt.layers.hyperconnection``. All math is accumulated in fp32.

    Supported dtypes: torch.bfloat16, torch.float16.

    Parameters
    ----------
    block_output    : CUDA tensor [..., hidden_size]
    residual        : CUDA tensor [..., hc_count * hidden_size]
    normed_residual : CUDA tensor, same shape/dtype as residual
    inject_weight   : CUDA tensor [hc_count, hc_count * hidden_size]
    hc_count        : number of hyper-connection branches
    hidden_size     : per-branch hidden size
    out             : optional pre-allocated output tensor (same shape/dtype as residual)

    Returns
    -------
    Combined tensor, same shape/dtype as residual.
    """
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty_like(r)
    else:
        out = out.reshape(-1, hc_count * hidden_size)

    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine(y, r, n, inject_weight, out)
    return out.reshape(residual.shape)


_SPLIT = 8
_MAX_ROWS = 32
_partials_cache = {}


def _get_partials(hc_count: int, device: torch.device, rows: int) -> torch.Tensor:
    key = (device, hc_count)
    buf = _partials_cache.get(key)
    if buf is None or buf.shape[0] < rows:
        # The gate kernel writes one slot per (row, split, hc); a buffer shorter
        # than `rows` is written past its end rather than truncated.
        buf = torch.empty(
            (max(rows, _MAX_ROWS), _SPLIT, hc_count),
            dtype=torch.float32,
            device=device,
        )
        _partials_cache[key] = buf
    return buf


def hc_combine_split(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty_like(r)
    else:
        out = out.reshape(-1, hc_count * hidden_size)
    rows = r.shape[0]
    partials = _get_partials(hc_count, r.device, rows)[:rows]
    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    module.hc_combine_split(y, r, n, inject_weight, out, partials)
    return out.reshape(residual.shape)


def hc_combine_apply(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    partials: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    out: Optional[torch.Tensor] = None,
    shared_output: Optional[torch.Tensor] = None,
    shared_gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply stage only, for a gate whose partial dots are already in `partials`.

    `partials` is fp32 [rows, splits, hc_count]; `splits` is a runtime value so
    both the standalone gate kernel (8 splits) and the HC-mix K0 epilogue
    (one slot per branch CTA, i.e. hc_count splits) can feed this.

    With `shared_output`/`shared_gate` the shared-expert join is folded in:
    ``block_output`` is then the routed-expert output only and the kernel forms
    ``bf16(routed + gate * shared)`` before the residual update, matching the
    order of the separate ``fused_gate_sigmoid_mul_add`` + combine pair.
    """
    y = block_output.reshape(-1, hidden_size)
    r = residual.reshape(-1, hc_count * hidden_size)
    if out is None:
        out = torch.empty_like(r)
    else:
        out = out.reshape(-1, hc_count * hidden_size)
    module = _jit_hc_combine_module(hc_count, hidden_size, residual.dtype)
    n = partials.shape[1]
    if shared_output is None:
        getattr(module, f"hc_combine_apply_p{n}")(y, r, partials, out)
    else:
        getattr(module, f"hc_combine_apply_shared_p{n}")(
            y, r, shared_output.reshape(-1, hidden_size), shared_gate, partials, out
        )
    return out.reshape(residual.shape)


def hc_combine_gate(
    normed_residual: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    partials: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gate stage only: the split partial dots, without touching block output.

    Returns the fp32 [rows, 8, hc_count] partials that `hc_combine_apply`
    reduces. Only the gate kernel of the split pair is launched, so the values
    are bit-identical to what `hc_combine_split` computes internally.
    """
    n = normed_residual.reshape(-1, hc_count * hidden_size)
    rows = n.shape[0]
    if partials is None:
        partials = _get_partials(hc_count, n.device, rows)[:rows]
    module = _jit_hc_combine_module(hc_count, hidden_size, normed_residual.dtype)
    module.hc_combine_gate(n, inject_weight, partials)
    return partials
