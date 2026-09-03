from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.hc_mix_triton import fused_hc_mix, fused_hc_mix_supported
from sglang.srt.layers.linear import ReplicatedLinear

_HC_FUSED = os.environ.get("SGLANG_HC_FUSED", "0") == "1"
# Single-launch combine (gate + apply in one Triton kernel), see
# sglang.srt.layers.hc_combine_fused_triton. Replaces the sgl-kernel
# hc_combine_gate + hc_combine_apply pair (2 launches x 96 boundaries per step).
_HC_COMBINE_FUSED = os.environ.get("SGLANG_HC_COMBINE_FUSED", "0") == "1"
_HC_MIX2 = os.environ.get("SGLANG_HC_MIX2", "0") == "1"
_HC_MIX_FP8 = os.environ.get("SGLANG_HC_MIX_FP8", "0") == "1"

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class HyperConnectionConfig(msgspec.Struct, frozen=True):
    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, group_size: Optional[int] = None
    ):
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight.weight_loader = self._weight_loader
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(
                x, self.weight, self._jit_group_size, self.variance_epsilon
            )
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (
                x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnectionBase(nn.Module):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.hc_count = config.hc_count
        if config.mtp_hc and role is not None and "mtp" in role:
            self.hc_count = self.hc_count + 1
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        mixed_input = hyper_input.view(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        ).mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.view(
            *residual.shape[:-1], self.hc_count, self.hidden_size
        )
        combined_output = residual_reshaped + block_output.unsqueeze(-2)
        combined_output = combined_output.view(
            *residual.shape[:-1], self.hc_count * self.hidden_size
        )
        return combined_output


class GatedResidual(HyperConnectionBase):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, use_mix, use_combine, role)
        self._quantized_mix = quant_config is not None

        norm_dim = (
            self.config.hidden_size * self.hc_count
            if self.config.hc_per_branch_norm
            else self.config.hidden_size
        )
        norm_group_size = (
            self.config.hidden_size if self.config.hc_per_branch_norm else None
        )
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_dim, eps=self.config.rms_norm_eps, group_size=norm_group_size
        )

        if use_mix:
            if quant_config is None:
                self.input_mix_weight_down = nn.Linear(
                    self.hidden_size * self.hc_count,
                    self.config.hc_lowrank,
                    bias=False,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
                self.input_mix_weight_up = nn.Linear(
                    self.config.hc_lowrank,
                    self.hc_count * self.hidden_size,
                    bias=False,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            else:
                self.input_mix_weight_down = ReplicatedLinear(
                    self.hidden_size * self.hc_count,
                    self.config.hc_lowrank,
                    bias=False,
                    quant_config=quant_config,
                    prefix=(
                        f"{prefix}.input_mix_weight_down"
                        if prefix
                        else "input_mix_weight_down"
                    ),
                )
                self.input_mix_weight_up = ReplicatedLinear(
                    self.config.hc_lowrank,
                    self.hc_count * self.hidden_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=(
                        f"{prefix}.input_mix_weight_up"
                        if prefix
                        else "input_mix_weight_up"
                    ),
                )
            from sglang.srt.environ import envs

            lowrank = self.config.hc_lowrank
            self._jit_mix_ok = (
                envs.SGLANG_HC_MIX_CUDA.get()
                and torch.cuda.is_available()
                # The CuTe split-K pair is tcgen05 (sm_100 family) only; the
                # default-on env must not route Hopper/Ada to it.
                and torch.cuda.get_device_capability()[0] == 10
                and (self.hc_count * self.hidden_size) % 2048 == 0
                and self.hidden_size % 8 == 0
                and lowrank > 0
                and lowrank % 8 == 0
                and not self._quantized_mix
            )
            self._mix_up_weight_padded = None

        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hidden_size * self.hc_count,
                self.hc_count,
                bias=False,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
            # The JIT combine kernel requires hidden_size % 8 == 0 and
            # hc_count * hidden_size % 2048 == 0; this is init-static, so
            # resolve it once here (device/dtype stay per-call).
            self._jit_combine_ok = (
                self.hidden_size % 8 == 0
                and (self.hc_count * self.hidden_size) % 2048 == 0
            )
            from sglang.srt.environ import envs

            vecs = self.hc_count * self.hidden_size // 8
            self._split_combine_ok = (
                envs.SGLANG_HC_COMBINE_SPLIT.get()
                and self._jit_combine_ok
                and vecs % (8 * 160) == 0
                and (self.hidden_size // 8) % (vecs // 8) == 0
            )
            # The one-launch Triton combine only needs a chunking of the row that
            # fits its grid barrier; the per-call row count decides that, so the
            # rest of the check lives in hc_combine_fused_supported.
            self._fused_combine_ok = _HC_COMBINE_FUSED and torch.cuda.is_available()

        def _mix_compute(
            hyper_input_normed: torch.Tensor,
            input_mix_weight_down: torch.Tensor,
            input_mix_weight_up: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            input_mix_weight = F.silu(
                F.linear(hyper_input_normed, input_mix_weight_down) / hc
            )
            input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
            input_mix_weight = torch.sigmoid(input_mix_weight)
            input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
            output = (
                input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
            ).mean(dim=-2)
            return output

        def _combine_compute(
            block_output: torch.Tensor,
            residual: torch.Tensor,
            normed_residual: torch.Tensor,
            block_inject_weight: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            R = residual.unflatten(-1, (hc, hs))
            block_inject_weight_out = 2 * torch.sigmoid(
                F.linear(normed_residual, block_inject_weight) / hc
            )
            injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
                -1
            )
            return (R + injection).flatten(-2)

        self._mix_compute = torch.compile(_mix_compute)
        self._combine_compute = torch.compile(_combine_compute)

    def _mix_with_quantized_linears(
        self, hyper_input_normed: torch.Tensor
    ) -> torch.Tensor:
        input_mix_weight, _ = self.input_mix_weight_down(hyper_input_normed)
        input_mix_weight = F.silu(input_mix_weight / self.hc_count)
        input_mix_weight, _ = self.input_mix_weight_up(input_mix_weight)
        input_mix_weight = torch.sigmoid(input_mix_weight)
        input_mix_weight = input_mix_weight.unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        return (
            input_mix_weight
            * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)

    def _mix2_supported(self, hyper_input: torch.Tensor) -> bool:
        if not (
            _HC_MIX2
            and self.config.hc_per_branch_norm
            and not self._quantized_mix
            and hyper_input.dim() == 2
            and 1 <= hyper_input.shape[0] <= 16
            and hyper_input.shape[-1] == self.hc_count * self.hidden_size
        ):
            return False
        from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2_supported

        return hc_norm_mix2_supported(
            hyper_input,
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
            self.hc_count,
            self.hidden_size,
        )

    def _fused_mix_supported(self, hyper_input: torch.Tensor) -> bool:
        if not (
            _HC_FUSED
            and self.config.hc_per_branch_norm
            and not self._quantized_mix
            and hyper_input.is_cuda
            and hyper_input.dtype == torch.bfloat16
            and hyper_input.dim() == 2
            and 1 <= hyper_input.shape[0] <= 16
            and hyper_input.shape[1] == self.hc_count * self.hidden_size
            and hyper_input.shape[1] % 2048 == 0
            and self.config.hc_lowrank % 64 == 0
            and hyper_input.is_contiguous()
        ):
            return False
        weights = (
            self.hc_norm.weight,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        )
        return all(
            w.is_cuda
            and w.device == hyper_input.device
            and w.dtype == torch.bfloat16
            and w.is_contiguous()
            for w in weights
        )

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input)

        if self._mix2_supported(hyper_input):
            from sglang.srt.layers.hc_mix2_triton import hc_norm_mix2

            mixed_input, hyper_input_normed = hc_norm_mix2(
                hyper_input,
                self.hc_norm.weight,
                self.hc_norm.variance_epsilon,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            )
            return mixed_input, (hyper_input, hyper_input_normed)

        if self._fused_mix_supported(hyper_input):
            from sglang.srt.layers.hc_fused_triton import hc_fused_norm_mix

            mixed_input, hyper_input_normed = hc_fused_norm_mix(
                hyper_input,
                self.hc_norm.weight,
                self.hc_norm.variance_epsilon,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            )
            return mixed_input, (hyper_input, hyper_input_normed)

        if self.config.hc_per_branch_norm:
            hyper_input_normed = self.hc_norm(hyper_input)
        else:
            hyper_input_normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        if self._quantized_mix:
            mixed_input = self._mix_with_quantized_linears(hyper_input_normed).to(
                self.params_dtype
            )
        elif (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= 24
        ):
            from sglang.kernels.ops.elementwise.hc_mix import (
                hc_mix,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif (
            _HC_MIX_FP8
            and getattr(self, "_mix_fp8", None) is not None
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= 16
        ):
            wd, sd, wu, su = self._mix_fp8
            mixed_input = fused_hc_mix(
                hyper_input_normed, wd, wu, self.hc_count, self.hidden_size, sd, su
            ).to(self.params_dtype)
        elif fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        ):
            if _HC_MIX_FP8 and getattr(self, "_mix_fp8", None) is None:
                # Lazily build the FP8 mix weights once (first eager forward)
                # and route later decode-width calls through the FP8 kernel.
                from sglang.srt.layers.hc_mix_triton import (
                    quantize_hc_mix_weights_fp8,
                )

                w_up_padded = self._mix_up_weight_padded
                if w_up_padded is None:
                    from sglang.kernels.ops.elementwise.hc_mix import (
                        permute_pad_up_weight,
                    )

                    w_up_padded = permute_pad_up_weight(
                        self.input_mix_weight_up.weight, self.hc_count
                    )
                    self._mix_up_weight_padded = w_up_padded
                self._mix_fp8 = quantize_hc_mix_weights_fp8(
                    self.input_mix_weight_down.weight.data, w_up_padded
                )
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        else:
            mixed_input = self._mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        return mixed_input, (hyper_input, hyper_input_normed)

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        hyper_input, hyper_input_normed = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        if block_output.shape[0] == 0:
            return hyper_input.to(self.params_dtype)

        if (
            self._jit_combine_ok
            and block_output.is_cuda
            and block_output.dtype in (torch.bfloat16, torch.float16)
            and hyper_input.dtype == block_output.dtype
            and hyper_input_normed.dtype == block_output.dtype
            and self.block_inject_weight.weight.dtype == block_output.dtype
        ):
            if self._fused_combine_ok:
                from sglang.srt.layers.hc_combine_fused_triton import (
                    hc_combine_fused,
                    hc_combine_fused_supported,
                )

                if hc_combine_fused_supported(
                    block_output,
                    hyper_input,
                    hyper_input_normed,
                    self.block_inject_weight.weight.data,
                    self.hc_count,
                    self.hidden_size,
                ):
                    return hc_combine_fused(
                        block_output,
                        hyper_input,
                        hyper_input_normed,
                        self.block_inject_weight.weight.data,
                        self.hc_count,
                        self.hidden_size,
                    )
            if self._split_combine_ok and block_output.shape[0] <= 32:
                from sglang.kernels.ops.elementwise.hc_combine import (
                    hc_combine_split,
                )

                return hc_combine_split(
                    block_output,
                    hyper_input,
                    hyper_input_normed,
                    self.block_inject_weight.weight.data,
                    self.hc_count,
                    self.hidden_size,
                )
            from sglang.kernels.ops.elementwise.hc_combine import hc_combine

            return hc_combine(
                block_output,
                hyper_input,
                hyper_input_normed,
                self.block_inject_weight.weight,
                self.hc_count,
                self.hidden_size,
            )

        updated_residuals = self._combine_compute(
            block_output,
            hyper_input,
            hyper_input_normed,
            self.block_inject_weight.weight,
            self.hc_count,
            self.hidden_size,
        ).to(self.params_dtype)
        return updated_residuals

    def combine_then_mix(
        self,
        block_output: torch.Tensor,
        residuals,
        next_hc: "GatedResidual",
    ):
        """Combine this boundary and norm/mix the next one in one launch."""
        hyper_input, hyper_input_normed = residuals
        inject_w = self.block_inject_weight.weight
        if (
            next_hc._fused_mix_supported(hyper_input)
            and self.hc_count == next_hc.hc_count
            and self.hidden_size == next_hc.hidden_size
            and block_output.shape == (hyper_input.shape[0], self.hidden_size)
            and hyper_input_normed.shape == hyper_input.shape
            and block_output.is_cuda
            and block_output.device == hyper_input.device
            and hyper_input_normed.device == hyper_input.device
            and inject_w.device == hyper_input.device
            and block_output.dtype == torch.bfloat16
            and hyper_input_normed.dtype == torch.bfloat16
            and inject_w.dtype == torch.bfloat16
            and block_output.is_contiguous()
            and hyper_input_normed.is_contiguous()
            and inject_w.is_contiguous()
        ):
            from sglang.srt.layers.hc_fused_triton import hc_fused_combine_norm_mix

            mixed, new_residual, normed = hc_fused_combine_norm_mix(
                block_output,
                hyper_input,
                hyper_input_normed,
                inject_w,
                next_hc.hc_norm.weight,
                next_hc.hc_norm.variance_epsilon,
                next_hc.input_mix_weight_down.weight,
                next_hc.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            )
            return mixed, (new_residual, normed)
        return next_hc.mix(self.combine(block_output, residuals))


HYPERCONNECTION_CLASS_DICT = {
    "hyperconnection_average": HyperConnectionBase,
    "gated_residual_simple": GatedResidual,
}
