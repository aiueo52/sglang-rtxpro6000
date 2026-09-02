import pytest
import torch

from sglang.kernels.ops.elementwise.hc_combine import hc_combine
from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm as rmsnorm_ops
from sglang.srt.layers.hc_fused_triton import (
    hc_fused_combine_norm_mix,
    hc_fused_norm_mix,
)
from sglang.srt.layers.hc_mix_triton import fused_hc_mix

HC = 4
HS = 2560
LR = 320
EPS = 1e-6

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _assert_mix_error(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.float() - expected.float()).abs()
    assert error.max().item() <= 3e-2
    assert error.mean().item() <= 3e-3


def _assert_norm_error(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.float() - expected.float()).abs()
    assert error.max().item() <= 2e-2


def _assert_one_bf16_ulp(actual: torch.Tensor, expected: torch.Tensor) -> None:
    error = (actual.float() - expected.float()).abs()
    magnitude = expected.float().abs()
    # Normal bf16 spacing is 2**(floor(log2(abs(x))) - 7); subnormals have
    # fixed spacing 2**-133.  This is the requested ULP of |reference value|.
    normal_ulp = torch.exp2(torch.floor(torch.log2(magnitude)) - 7)
    ulp = torch.where(magnitude < 2.0**-126, 2.0**-133, normal_ulp)
    assert torch.all(error <= ulp).item()


def _make_inputs(rows: int):
    torch.manual_seed(1234 + rows)
    residual = torch.randn(rows, HC * HS, device="cuda", dtype=torch.bfloat16)
    normed_prev = torch.randn_like(residual)
    block_output = torch.randn(rows, HS, device="cuda", dtype=torch.bfloat16)
    inject_w = torch.randn(HC, HC * HS, device="cuda", dtype=torch.bfloat16) * 0.02
    norm_w = torch.randn(HC * HS, device="cuda", dtype=torch.bfloat16) * 0.02
    w_down = torch.randn(LR, HC * HS, device="cuda", dtype=torch.bfloat16) * 0.02
    w_up = torch.randn(HC * HS, LR, device="cuda", dtype=torch.bfloat16) * 0.02
    return residual, normed_prev, block_output, inject_w, norm_w, w_down, w_up


@pytest.mark.parametrize("rows", [1, 5, 16])
def test_hc_fused_entry_points_match_reference_chain(rows: int):
    residual, normed_prev, block_output, inject_w, norm_w, w_down, w_up = _make_inputs(
        rows
    )

    ref_normed = rmsnorm_ops.grouped_gemma_rmsnorm(residual, norm_w, HS, EPS)
    ref_mixed = fused_hc_mix(ref_normed, w_down, w_up, HC, HS)
    mixed, normed = hc_fused_norm_mix(residual, norm_w, EPS, w_down, w_up, HC, HS)
    _assert_norm_error(normed, ref_normed)
    _assert_mix_error(mixed, ref_mixed)

    ref_new_residual = hc_combine(block_output, residual, normed_prev, inject_w, HC, HS)
    ref_combined_normed = rmsnorm_ops.grouped_gemma_rmsnorm(
        ref_new_residual, norm_w, HS, EPS
    )
    ref_combined_mixed = fused_hc_mix(ref_combined_normed, w_down, w_up, HC, HS)
    combined_mixed, new_residual, combined_normed = hc_fused_combine_norm_mix(
        block_output,
        residual,
        normed_prev,
        inject_w,
        norm_w,
        EPS,
        w_down,
        w_up,
        HC,
        HS,
    )
    _assert_one_bf16_ulp(new_residual, ref_new_residual)
    _assert_norm_error(combined_normed, ref_combined_normed)
    _assert_mix_error(combined_mixed, ref_combined_mixed)
