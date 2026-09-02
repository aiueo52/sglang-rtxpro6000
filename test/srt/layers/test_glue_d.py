import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.router_gemv import router_gemv
from sglang.srt.layers.moe.topk import _get_zero_bias
from sglang.srt.layers.mtp_entry import mtp_entry_fused
from sglang.srt.mem_cache.fp8_kv_store import fp8_kv_store


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HIDDEN_SIZE = 2560
HC = 4
EXPERTS = 512
EPS = 1e-6


@pytest.mark.parametrize("rows", [1, 5, 16])
def test_router_gemv_matches_linear(rows: int):
    torch.manual_seed(1000 + rows)
    x = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(EXPERTS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        * 0.02
    )
    actual = router_gemv(x, weight)
    expected = F.linear(x, weight).float()
    assert actual.dtype == torch.float32
    assert (actual - expected).abs().max().item() <= 2e-2


def test_router_gemv_optional_bias_and_zero_bias_cache():
    torch.manual_seed(1017)
    x = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(EXPERTS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        * 0.02
    )
    bias = torch.randn(EXPERTS, device="cuda", dtype=torch.bfloat16) * 0.02
    actual = router_gemv(x, weight, bias)
    expected = F.linear(x, weight, bias).float()
    assert (actual - expected).abs().max().item() <= 2e-2

    first = _get_zero_bias(EXPERTS, x.device)
    second = _get_zero_bias(EXPERTS, x.device)
    assert first.data_ptr() == second.data_ptr()
    assert torch.count_nonzero(first).item() == 0


@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("scale_kind", ["none", "float", "tensor"])
@pytest.mark.parametrize("strided", [False, True])
def test_fp8_kv_store_matches_reference(tokens: int, scale_kind: str, strided: bool):
    torch.manual_seed(2000 + tokens)
    shape = (tokens, 4, 128)
    if strided:
        # k/v as token-strided views of one fused [N, 3*H*D] buffer
        fused = torch.randn((tokens, 3 * 4 * 128), device="cuda", dtype=torch.bfloat16)
        cache_k = fused[:, : 4 * 128].view(tokens, 4, 128)
        cache_v = fused[:, 4 * 128 : 2 * 4 * 128].view(tokens, 4, 128)
    else:
        cache_k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        cache_v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    loc = torch.randperm(64, device="cuda", dtype=torch.int64)[:tokens].contiguous()
    k_buffer = torch.zeros(
        (64, 4, 128), device="cuda", dtype=torch.float8_e4m3fn
    )
    v_buffer = torch.zeros_like(k_buffer)
    ref_k = torch.zeros_like(k_buffer)
    ref_v = torch.zeros_like(v_buffer)

    if scale_kind == "none":
        k_scale = v_scale = None
    elif scale_kind == "float":
        k_scale = v_scale = 0.7
    else:
        k_scale = torch.tensor(0.7, device="cuda")
        v_scale = torch.tensor(0.7, device="cuda")

    ref_k_values = cache_k.clone()
    ref_v_values = cache_v.clone()
    if k_scale is not None:
        ref_k_values.div_(k_scale)
        ref_v_values.div_(v_scale)
    ref_k[loc] = ref_k_values.to(torch.float8_e4m3fn)
    ref_v[loc] = ref_v_values.to(torch.float8_e4m3fn)

    fp8_kv_store(cache_k, cache_v, k_buffer, v_buffer, loc, k_scale, v_scale)
    assert torch.equal(k_buffer.view(torch.uint8), ref_k.view(torch.uint8))
    assert torch.equal(v_buffer.view(torch.uint8), ref_v.view(torch.uint8))


def _gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    return (
        x_fp32 * torch.rsqrt(variance + eps) * (1.0 + weight.float())
    ).to(torch.bfloat16)


def _mtp_reference(
    input_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    embedding_norm_weight: torch.Tensor,
    hidden_norm_weight: torch.Tensor,
    embedding_proj_weight: torch.Tensor,
    hidden_proj_weight: torch.Tensor,
) -> torch.Tensor:
    rows = input_embeds.shape[0]
    embedding = F.linear(
        _gemma_rms_norm(input_embeds, embedding_norm_weight, EPS),
        embedding_proj_weight,
    )
    hidden = _gemma_rms_norm(hidden_states, hidden_norm_weight, EPS)
    hidden = F.linear(hidden.view(rows, HC, HIDDEN_SIZE), hidden_proj_weight)
    return (embedding.unsqueeze(1) + hidden).view_as(hidden_states)


@pytest.mark.parametrize("rows", [1, 5, 16])
def test_mtp_entry_fused_matches_reference(rows: int):
    torch.manual_seed(3000 + rows)
    input_embeds = torch.randn(
        rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    hidden_states = torch.randn(
        rows, HC * HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    embedding_norm_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    hidden_norm_weight = (
        torch.randn(HC * HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    embedding_proj_weight = (
        torch.randn(
            HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    hidden_proj_weight = (
        torch.randn(
            HIDDEN_SIZE, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    expected = _mtp_reference(
        input_embeds,
        hidden_states,
        embedding_norm_weight,
        hidden_norm_weight,
        embedding_proj_weight,
        hidden_proj_weight,
    )
    actual = mtp_entry_fused(
        input_embeds,
        hidden_states,
        embedding_norm_weight,
        hidden_norm_weight,
        embedding_proj_weight,
        hidden_proj_weight,
        EPS,
    )
    assert (actual.float() - expected.float()).abs().max().item() <= 6.5e-2  # 2 bf16 ulps at |value|~4-8
