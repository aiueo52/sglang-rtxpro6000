from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.speculative.ngram_chain_worker import (
    NgramChainAlgorithm,
    NgramChainDraftWorker,
    NgramChainMatch,
    NgramChainWorkerV2,
    RequestNgramChainIndex,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    _assert_custom_spec_algo_conforms,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _index(*, min_size=3, max_size=3, window_size=128, num_steps=3):
    return RequestNgramChainIndex(
        min_size=min_size,
        max_size=max_size,
        window_size=window_size,
        num_steps=num_steps,
    )


def test_prompt_seed_and_repeated_span_proposal():
    index = _index()
    index.seed([1, 2, 3, 10, 11, 12, 1, 2, 3])

    match = index.propose()

    assert match == NgramChainMatch(tokens=(10, 11, 12), ngram_size=3, position=0)


def test_incremental_append_enables_match():
    index = _index()
    index.seed([1, 2, 3, 10, 11, 12])
    assert index.propose() is None

    index.append([1, 2, 3])

    assert index.propose().tokens == (10, 11, 12)


def test_largest_ngram_is_preferred():
    index = _index(min_size=2, max_size=4, num_steps=2)
    index.seed([1, 2, 3, 4, 8, 9, 1, 2, 3, 4])

    match = index.propose()

    assert match.ngram_size == 4
    assert match.tokens == (8, 9)


def test_window_evicts_old_occurrence():
    index = _index(min_size=2, max_size=2, window_size=6, num_steps=2)
    index.seed([1, 2, 3, 4, 9, 9, 1, 2])

    assert index.tokens == (3, 4, 9, 9, 1, 2)
    assert index.propose() is None


def test_no_match_below_min_size():
    index = _index(min_size=3, max_size=4, num_steps=2)
    index.seed([5, 6, 8, 9, 5, 6])

    assert index.propose() is None


def test_partial_continuation_is_a_miss():
    index = _index(min_size=2, max_size=2, num_steps=4)
    # The occurrence at zero has only three following tokens. The current
    # suffix itself has no continuation, so neither is a full-width candidate.
    index.seed([1, 2, 7, 1, 2])

    assert index.propose() is None


def test_topk1_contract_shapes_dtypes_and_delta_probs():
    worker = object.__new__(NgramChainDraftWorker)
    worker.device = torch.device("cpu")
    worker.speculative_num_steps = 3
    worker._topk1_parents_prealloc = torch.tensor(
        [[-1, 0, 1], [-1, 0, 1]], dtype=torch.long
    )
    worker._topk1_score_indices_prealloc = torch.tensor(
        [[0, 1, 2], [0, 1, 2]], dtype=torch.long
    )

    parents, score_indices, draft_tokens, draft_probs = worker._build_ngram_contract(
        [[2, 4, 6], [1, 3, 5]], vocab_size=8, device="cpu"
    )

    assert parents.shape == score_indices.shape == draft_tokens.shape == (2, 3)
    assert draft_probs.shape == (2, 3, 8)
    assert parents.dtype == score_indices.dtype == draft_tokens.dtype == torch.long
    assert draft_probs.dtype == torch.float32
    assert torch.equal(draft_probs.sum(dim=-1), torch.ones((2, 3)))
    assert torch.equal(
        draft_probs.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1),
        torch.ones((2, 3)),
    )


def test_verify_commits_append_only_each_accept_prefix():
    worker = object.__new__(NgramChainDraftWorker)
    first = _index(min_size=2, max_size=2, num_steps=2)
    second = _index(min_size=2, max_size=2, num_steps=2)
    first.seed([1, 2])
    second.seed([3, 4])
    worker._request_indexes = {"a": first, "b": second}
    worker._pending_commits = deque()
    requests = [SimpleNamespace(rid="a"), SimpleNamespace(rid="b")]

    worker.stage_commits(
        requests,
        torch.tensor([10, 11, 99, 99, 20, 99, 99, 99]),
        torch.tensor([2, 1]),
        stride=4,
    )
    worker._flush_pending_commits()

    assert first.tokens == (1, 2, 10, 11)
    assert second.tokens == (3, 4, 20)


def test_departed_request_index_is_freed():
    worker = object.__new__(NgramChainDraftWorker)
    worker._request_indexes = {"active": object(), "done": object()}

    worker._drop_departed_requests([SimpleNamespace(rid="active")])

    assert set(worker._request_indexes) == {"active"}


def _decision_worker(match):
    worker = object.__new__(NgramChainDraftWorker)
    worker._flush_pending_commits = MagicMock()
    worker.seed_requests = MagicMock()
    worker._drop_departed_requests = MagicMock()
    worker._find_batch_matches = MagicMock(return_value=match)
    worker._draft_mtp_chain = MagicMock(return_value="mtp")
    return worker


def test_miss_invokes_mtp_path():
    worker = _decision_worker(None)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False), reqs=[SimpleNamespace()]
    )

    assert worker.draft(batch) == "mtp"
    worker._draft_mtp_chain.assert_called_once_with(batch)


def test_full_hit_skips_mtp_path():
    match = NgramChainMatch(tokens=(7, 8, 9), ngram_size=3, position=0)
    worker = _decision_worker([match])
    worker.req_to_token_pool = object()
    worker.cuda_graph_runner = object()
    worker.draft_runner = object()
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4
    worker.target_worker = SimpleNamespace(model_config=SimpleNamespace(vocab_size=16))
    worker.tree_mask_mode = object()
    worker.device = "cpu"
    worker._build_ngram_contract = MagicMock(
        return_value=("parents", "scores", "tokens", "probs")
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        reqs=[SimpleNamespace()],
        spec_info=object(),
    )

    with (
        patch("sglang.srt.speculative.ngram_chain_worker.prepare_for_draft") as prepare,
        patch(
            "sglang.srt.speculative.ngram_chain_worker.build_eagle_verify_input",
            return_value="verify",
        ),
    ):
        assert worker.draft(batch) == "verify"

    prepare.assert_called_once()
    worker._draft_mtp_chain.assert_not_called()


def test_registration_routes_as_eagle_but_not_ngram():
    algo = SpeculativeAlgorithm.from_string("NGRAM_CHAIN")

    assert isinstance(algo, CustomSpecAlgo)
    assert isinstance(algo, NgramChainAlgorithm)
    assert algo.is_eagle()
    assert not algo.is_ngram()
    assert algo.supports_grammar_overlap()
    assert algo.carries_draft_hidden_states()
    assert (
        algo.create_worker(SimpleNamespace(disable_overlap_schedule=False))
        is NgramChainWorkerV2
    )
    _assert_custom_spec_algo_conforms(NgramChainAlgorithm)
