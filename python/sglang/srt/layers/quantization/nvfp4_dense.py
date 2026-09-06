"""Load-time NVFP4 packing for dense weights, with an on-disk cache.

Used by the N1 stages to move a dense FP8 weight-only path to the NVFP4 format the
experts already use (see `w4a16_nvfp4_gemv`).

Two things this module exists to get right:

**Quantise from BF16, not from FP8.** By the time the EAGLE worker reaches the draft
head, `Fp8LinearMethod.process_weights_after_loading` has already replaced the target's
`lm_head.weight` in place, so the obvious implementation would quantise the *already
FP8* rows and compound two lossy steps. Measured on the real head, that costs ~2 points
of argmax agreement against quantising from BF16 (N1_LOG.md 4). So the rows are re-read
straight from the checkpoint shard instead -- `safetensors` slices lazily, so only the
rows that are wanted are ever materialised and there is no 1.27 GB transient.

**Cache the packed result.** Packing 49152x2560 costs a few seconds and is entirely
deterministic, so it is written next to the token map under a key derived from the
model shard's identity, the tensor name, and the exact row set. Any change to the model
directory, the tensor, or the token map yields a different key, so a stale cache cannot
be picked up silently.
"""

import hashlib
import json
import logging
import os
from typing import Optional, Sequence

import torch

from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (
    FP4_SCALE_BLOCK,
    quantize_nvfp4,
)

logger = logging.getLogger(__name__)


def _default_cache_dir() -> str:
    from sglang.srt.environ import envs

    return os.path.join(os.path.expanduser(envs.SGLANG_CACHE_DIR.get()), "nvfp4")


#: Where packed weights are cached: SGLANG_NVFP4_CACHE_DIR, else <SGLANG_CACHE_DIR>/nvfp4
#: (by default ~/.cache/sglang/nvfp4).
CACHE_DIR = os.environ.get("SGLANG_NVFP4_CACHE_DIR") or _default_cache_dir()


def _index_path(model_dir: str) -> Optional[str]:
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            return p
    return None


def find_shard(model_dir: str, tensor_name: str) -> str:
    """Absolute path of the safetensors shard holding `tensor_name`.

    Prefers the index json; falls back to opening each shard's header, which is cheap
    because `safe_open` reads only the metadata block.
    """
    idx = _index_path(model_dir)
    if idx is not None:
        with open(idx) as f:
            wm = json.load(f).get("weight_map", {})
        if tensor_name in wm:
            return os.path.join(model_dir, wm[tensor_name])
    from safetensors import safe_open

    shards = sorted(
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".safetensors")
    )
    for p in shards:
        with safe_open(p, framework="pt", device="cpu") as fh:
            if tensor_name in fh.keys():
                return p
    raise FileNotFoundError(f"{tensor_name} not found in any shard under {model_dir}")


def _shard_sig(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"


def _rows_sig(rows: Optional[torch.Tensor]) -> str:
    if rows is None:
        return "all"
    r = rows.detach().to("cpu", torch.int64).contiguous()
    h = hashlib.sha256(r.numpy().tobytes()).hexdigest()[:16]
    return f"{r.numel()}-{h}"


def cache_key(model_dir: str, tensor_name: str, rows: Optional[torch.Tensor]) -> str:
    shard = find_shard(model_dir, tensor_name)
    raw = "|".join(
        [os.path.realpath(model_dir), tensor_name, _shard_sig(shard), _rows_sig(rows)]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def read_bf16_rows(
    model_dir: str, tensor_name: str, rows: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """`tensor_name`'s rows straight from the checkpoint, in its stored dtype.

    `get_slice` keeps this lazy: for a row subset only those rows are read off disk, so
    the peak transient is the subset, not the whole [248320, 2560] tensor.
    """
    from safetensors import safe_open

    path = find_shard(model_dir, tensor_name)
    with safe_open(path, framework="pt", device="cpu") as fh:
        if rows is None:
            return fh.get_tensor(tensor_name)
        sl = fh.get_slice(tensor_name)
        idx = rows.detach().to("cpu", torch.int64)
        # One contiguous read per row; the hot map is ~49k rows of 5 KB, ~250 MB total.
        return torch.cat([sl[int(i) : int(i) + 1, :] for i in idx.tolist()], dim=0)


def build_or_load(
    model_dir: str,
    tensor_name: str,
    rows: Optional[torch.Tensor] = None,
    device: str = "cuda",
    label: str = "",
):
    """-> (wq [N, K/2] uint8, bs [N, K/16] uint8 e4m3, gscale [1] fp32), all on `device`.

    Deterministic: same shard + same row set -> byte-identical output, which is what
    makes the cache safe.
    """
    import time

    key = cache_key(model_dir, tensor_name, rows)
    path = os.path.join(CACHE_DIR, f"{tensor_name.replace('/', '_')}.{key}.pt")
    tag = label or tensor_name
    if os.path.exists(path):
        t0 = time.time()
        blob = torch.load(path, map_location=device, weights_only=True)
        logger.info(
            "NVFP4 %s: loaded packed weight from cache in %.2fs (%s)",
            tag, time.time() - t0, os.path.basename(path),
        )
        return blob["wq"], blob["bs"], blob["gscale"]

    t0 = time.time()
    w = read_bf16_rows(model_dir, tensor_name, rows)
    t_read = time.time() - t0
    t0 = time.time()
    wq, bs, gscale = quantize_nvfp4(w.to(device))
    del w
    t_pack = time.time() - t0
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    torch.save(
        {"wq": wq.cpu(), "bs": bs.cpu(), "gscale": gscale.cpu()}, tmp
    )
    os.replace(tmp, path)  # atomic, so two concurrent starts cannot read a partial file
    logger.info(
        "NVFP4 %s: packed %s from BF16 in %.2fs (read %.2fs) -> %s",
        tag, tuple(wq.shape), t_pack, t_read, os.path.basename(path),
    )
    return wq, bs, gscale


# ---------------------------------------------------------------------------
# Serving-side linear method
# ---------------------------------------------------------------------------

#: Rows dequantised at a time in the large-M fallback, bounding its transient.
_FALLBACK_ROW_CHUNK = 16384
#: Up to this many rows, the fallback loops the GEMV instead of dequantising; this is
#: the CUDA-graph-captured draft-extend regime, where a dequant is not capture-safe.
_CHUNK_M_MAX = 128


class NVFP4DenseLinearMethod:
    """Weight-only NVFP4 for an lm_head, matching the dispatch in `logits_processor`.

    `_compute_lm_head` needs exactly three things of a quant method: the layer must have
    a `.weight` attribute, `apply(layer, x, bias)` must return logits, and the optional
    `apply_into(layer, x, out, bias)` may write straight into the shared fp32 next-token
    buffer (returning None to decline). The packed tensors live on the *layer*
    (`nvfp4_wq` / `nvfp4_bs` / `nvfp4_gscale`), following the convention the FP8 method
    uses for `weight_scale`, so one method instance can serve several layers.

    `layer.weight` is set to the packed codes. Nothing downstream interprets the draft
    head's `.weight` shape -- the FP8 path already stores it transposed as [K, H] -- but
    the attribute has to exist for `should_apply_lm_head_quant_method` to say yes.
    """

    def __init__(self, name: str = "lm_head"):
        self.name = name

    @staticmethod
    def attach(layer, wq, bs, gscale) -> None:
        layer.nvfp4_wq = wq
        layer.nvfp4_bs = bs
        layer.nvfp4_gscale = gscale

    @staticmethod
    def _ok(layer, x) -> bool:
        return (
            getattr(layer, "nvfp4_wq", None) is not None
            and x.dim() == 2
            and x.shape[0] <= 16
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.stride(1) == 1
        )

    def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import w4a16_nvfp4_gemv

        if self._ok(layer, x):
            y = w4a16_nvfp4_gemv(x, layer.nvfp4_wq, layer.nvfp4_bs, layer.nvfp4_gscale)
        else:
            y = self._wide_m(layer, x)
        if bias is not None:
            y = y + bias
        return y

    def apply_into(
        self, layer, x: torch.Tensor, out: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import w4a16_nvfp4_gemv

        if bias is not None or not self._ok(layer, x):
            return None
        N = layer.nvfp4_wq.shape[0]
        if (
            tuple(out.shape) != (x.shape[0], N)
            or out.dtype not in (torch.bfloat16, torch.float32)
            or out.device != x.device
            or out.stride(1) != 1
        ):
            return None
        return w4a16_nvfp4_gemv(
            x, layer.nvfp4_wq, layer.nvfp4_bs, layer.nvfp4_gscale, out=out
        )

    def _wide_m(self, layer, x: torch.Tensor) -> torch.Tensor:
        """The M > 16 path the decode GEMV cannot serve.

        Two regimes, because they have different constraints:

        - **Modest M (<= _CHUNK_M_MAX): loop the GEMV over 16-row slices.** This is the
          draft-extend path, and draft-extend *is* CUDA-graph captured, so it has to be
          capture-safe -- no host->device copies, no syncs, no data-dependent shapes.
          Re-streaming the weight per slice costs bandwidth, but at these M it is a
          handful of slices.
        - **Large M: dequantise in row chunks and GEMM.** Real prefill, never captured.
          Looping the GEMV here would re-read the whole weight ~M/16 times; a GEMM
          amortises the weight read across all M rows instead. Chunking the dequant by
          output rows keeps the transient near 80 MB rather than the full BF16 weight.
        """
        from sglang.srt.layers.quantization.w4a16_nvfp4_gemv import (
            dequantize_nvfp4,
            w4a16_nvfp4_gemv,
        )

        wq, bs, gs = layer.nvfp4_wq, layer.nvfp4_bs, layer.nvfp4_gscale
        M, N = x.shape[0], wq.shape[0]
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        if M <= _CHUNK_M_MAX and x.dtype == torch.bfloat16 and x.is_cuda:
            for m0 in range(0, M, 16):
                m1 = min(M, m0 + 16)
                w4a16_nvfp4_gemv(x[m0:m1], wq, bs, gs, out=y[m0:m1])
            return y
        for r0 in range(0, N, _FALLBACK_ROW_CHUNK):
            r1 = min(N, r0 + _FALLBACK_ROW_CHUNK)
            w = dequantize_nvfp4(wq[r0:r1], bs[r0:r1], gs).to(torch.bfloat16)
            torch.mm(x.to(torch.bfloat16), w.t(), out=y[:, r0:r1])
            del w
        return y


# ---------------------------------------------------------------------------
# N1 stage C: dense projections
# ---------------------------------------------------------------------------
#
# These are packed inside `Fp8LinearMethod.process_weights_after_loading`, *before* it
# quantises, rather than by a post-load walk. Two reasons, and the first is decisive:
#
# - **The BF16 weight is right there.** A post-load pass would have to re-read the
#   checkpoint, and these layers do not exist in it under their module names --
#   `qkv_proj` is fused from q/k/v_proj and `in_proj_qkvz` from in_proj_qkv + in_proj_z
#   by the loader (`qwen4_exp.py:1923-1932`). Packing at this point gets the true BF16
#   fused tensor for free and skips replicating that mapping.
# - It is the same hook that already preallocates the GEMV scratch, so ordering against
#   CUDA-graph capture is settled.

def _cat_flags():
    def on(name):
        return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")

    cats = []
    if on("SGLANG_LINEAR_ATTN_NVFP4"):
        cats.append("in_proj_qkvz")
    if on("SGLANG_ATTN_NVFP4"):
        cats.append("qkv_proj")
    return tuple(cats)


def maybe_pack_dense_nvfp4(layer) -> bool:
    """Pack `layer` to NVFP4 in place and install the GEMV method. True if handled.

    Called at the top of `Fp8LinearMethod.process_weights_after_loading`, where
    `layer.weight` is still the loaded BF16 tensor. Deliberately NOT applied to the GDN
    `out_proj`: measured 1.08-1.13x there for 0.22 % of the W16 step, and it is the one
    shape where the 4-bit GEMV falls to 47-52 % of roof -- below break-even (N1_LOG.md 7).
    """
    cats = _cat_flags()
    if not cats:
        return False
    prefix = getattr(layer, "prefix", "") or ""
    if not any(prefix.endswith(c) or f".{c}" in prefix for c in cats):
        return False
    w = getattr(layer, "weight", None)
    if w is None or w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
        return False
    if w.shape[1] % FP4_SCALE_BLOCK:
        logger.warning("NVFP4 %s: K=%d not a multiple of 16, leaving FP8", prefix, w.shape[1])
        return False

    from sglang.srt.layers.quantization.w8a16_gemv import prealloc

    wq, bs, gscale = quantize_nvfp4(w.data)
    prealloc(wq.device)
    NVFP4DenseLinearMethod.attach(layer, wq, bs, gscale)
    try:
        del layer.weight
    except AttributeError:
        pass
    layer.weight = wq
    layer.weight_scale = None
    layer.input_scale = None
    layer.quant_method = NVFP4DenseLinearMethod(prefix)
    logger.info("NVFP4 dense: %s %s packed", prefix, tuple(wq.shape))
    return True

