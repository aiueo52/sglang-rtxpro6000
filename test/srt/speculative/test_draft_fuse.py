import pytest
import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("use_token_map", [False, True])
def test_draft_topk1_finalize_matches_torch(batch_size, use_token_map):
    device = torch.device("cuda")
    hot_vocab_size = 32768
    generator = torch.Generator(device=device).manual_seed(1234 + batch_size)
    logits = torch.randn(
        (batch_size, hot_vocab_size),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    # Make the maximum unique so the reference does not depend on tie-breaking.
    expected_hot_index = torch.randint(
        hot_vocab_size,
        (batch_size, 1),
        device=device,
        generator=generator,
    )
    logits.scatter_(1, expected_hot_index, 1000.0)
    expected_hot_index = torch.argmax(logits, dim=-1, keepdim=True)

    hot_token_id = None
    expected_token = expected_hot_index
    if use_token_map:
        hot_token_id = (
            torch.randperm(
                hot_vocab_size, device=device, generator=generator, dtype=torch.int64
            )
            + 100_000
        )
        expected_token = hot_token_id[expected_hot_index]

    positions = torch.arange(batch_size, device=device, dtype=torch.int64)
    expected_positions = positions + 1
    draft_tokens = torch.full((batch_size, 3), -1, device=device, dtype=torch.int64)

    topk_p, topk_index = draft_topk1_postprocess(
        logits,
        positions,
        draft_tokens,
        draft_token_column=1,
        hot_token_id=hot_token_id,
    )

    torch.testing.assert_close(topk_index, expected_token, rtol=0, atol=0)
    torch.testing.assert_close(draft_tokens[:, 1:2], expected_token, rtol=0, atol=0)
    torch.testing.assert_close(draft_tokens[:, 0], torch.full_like(positions, -1))
    torch.testing.assert_close(draft_tokens[:, 2], torch.full_like(positions, -1))
    torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
    torch.testing.assert_close(positions, expected_positions, rtol=0, atol=0)


def _lookup_torch_reference(shared, req_pool_indices, current_positions, layer_id):
    slot = shared.layer_slots[int(layer_id)]
    rows = req_pool_indices.to(torch.long)
    out = shared.indices[slot, rows]
    base = shared.captured_len[slot, rows].to(torch.int64)
    tail_offsets = torch.arange(shared.tail_width, device=out.device)
    tail = base.unsqueeze(1) + tail_offsets.unsqueeze(0)
    valid = tail <= current_positions.to(torch.int64).unsqueeze(1)
    out[:, -shared.tail_width :] = torch.where(valid, tail, -1).to(out.dtype)
    return out


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_mtp_shared_sparse_indices_lookup_matches_torch(batch_size, index_dtype):
    device = torch.device("cuda")
    num_requests = 10
    token_topk = 2048
    tail_width = 8
    layer_id = 7
    shared = QSAMTPSharedSparseIndices(
        layer_ids=[layer_id],
        num_requests=num_requests,
        token_topk=token_topk,
        tail_width=tail_width,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(4321 + batch_size)
    shared.indices.random_(-1, 4096, generator=generator)
    shared.captured_len.random_(32, 256, generator=generator)
    req_pool_indices = torch.randperm(
        num_requests, device=device, generator=generator, dtype=torch.int64
    )[:batch_size].to(index_dtype)
    slot = shared.layer_slots[layer_id]
    bases = shared.captured_len[slot, req_pool_indices.to(torch.long)].to(index_dtype)
    # Include truncated tails: row 0 admits none; remaining rows admit only a prefix.
    admitted = torch.arange(batch_size, device=device, dtype=index_dtype) % tail_width
    current_positions = bases + admitted - 1

    expected = _lookup_torch_reference(
        shared, req_pool_indices, current_positions, layer_id
    )
    actual = shared.lookup(req_pool_indices, current_positions, layer_id)

    assert actual.dtype == torch.int32
    assert actual.shape == (batch_size, token_topk + tail_width)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
