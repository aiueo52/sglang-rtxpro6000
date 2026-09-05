"""Regression tests for adaptive speculative runtime-state sizing.

Covers the two start-up invariants that make several runtime states able to
coexist on one server:

* a candidate step count may never exceed ``--speculative-num-steps``; and
* the draft-extend attention backend, which compressed-QSA draft models share
  across every runtime state, is sized once and never re-sized downward.

The second one is what crashed Qwen3.8-Flash-Next with
``--speculative-adaptive``: ``DraftBackendFactory.create_draft_extend_backend``
returns ``draft_runner.attn_backend`` for those models, so building the
steps=3 / steps=7 states re-ran ``init_cuda_graph_state`` on the very backend
whose old metadata tensors the already-captured steps=15 draft-extend graphs
point at, freeing them (illegal memory access at the next replay).
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
    init_cuda_graph_state_no_shrink,
)
from sglang.srt.speculative.adaptive_spec_params import AdaptiveStepSlot
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MODULE = "sglang.srt.runtime_context"


class _StubWorker:
    def __init__(self, steps=15):
        self.speculative_num_steps = steps
        self.built = []

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        self.built.append(speculative_num_steps)
        return SpecRuntimeState(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            draft_attn_backend=None,
            cuda_graph_runner=None,
            target_attn_backend=object(),
            target_graph_runner=None,
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=None,
            qsa_mtp_shared_sparse_indices=None,
        )

    def apply_runtime_state(self, state):
        self.speculative_num_steps = state.speculative_num_steps


class _StubBackend:
    """Stands in for the shared compressed-QSA draft-extend backend."""

    def __init__(self):
        self.calls = []

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        self.calls.append((max_bs, max_num_tokens))


def _controller(candidates, initial_steps=15):
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump({"1": {"candidate_steps": candidates}}, f)
        f.flush()
        return AdaptiveController(_StubWorker(initial_steps), config_path=f.name)


class TestAdaptiveCandidateValidation(CustomTestCase):
    def test_candidate_above_launch_steps_is_rejected(self):
        controller = _controller([3, 15, 31], initial_steps=15)
        with self.assertRaises(ValueError) as ctx:
            controller.init_states(cuda_graph_bs=[1, 2], max_batch_size=2)
        self.assertIn("31", str(ctx.exception))
        self.assertIn("speculative-num-steps", str(ctx.exception))

    def test_candidates_within_launch_steps_are_built(self):
        controller = _controller([3, 15], initial_steps=15)
        controller.register(
            controller.worker.build_adaptive_runtime_state(15, 16), steps=15
        )
        controller.worker.built.clear()
        controller.init_states(cuda_graph_bs=[1, 2], max_batch_size=2)
        # 15 was pre-registered, so only the smaller candidate is built.
        self.assertEqual(controller.worker.built, [3])


class TestDraftExtendGraphStateNoShrink(CustomTestCase):
    def test_non_adaptive_call_is_passed_through_unstamped(self):
        backend = _StubBackend()
        with patch(f"{_MODULE}.get_spec") as get_spec:
            get_spec.return_value.speculative_adaptive = False
            init_cuda_graph_state_no_shrink(backend, 2, 32)
            init_cuda_graph_state_no_shrink(backend, 2, 8)
        self.assertEqual(backend.calls, [(2, 32), (2, 8)])
        self.assertFalse(hasattr(backend, "_spec_graph_state_extent"))

    def test_smaller_candidate_does_not_reallocate_shared_backend(self):
        backend = _StubBackend()
        with patch(f"{_MODULE}.get_spec") as get_spec:
            get_spec.return_value.speculative_adaptive = True
            # steps=15 state (num_draft_tokens=16, capture bs [1, 2]).
            init_cuda_graph_state_no_shrink(backend, 2, 32)
            # steps=3 state (num_draft_tokens=4) and a bs=1-only variant.
            init_cuda_graph_state_no_shrink(backend, 2, 8)
            init_cuda_graph_state_no_shrink(backend, 1, 4)
        self.assertEqual(backend.calls, [(2, 32)])
        self.assertEqual(backend._spec_graph_state_extent, (2, 32))

    def test_growing_a_shared_backend_is_refused(self):
        backend = _StubBackend()
        with patch(f"{_MODULE}.get_spec") as get_spec:
            get_spec.return_value.speculative_adaptive = True
            init_cuda_graph_state_no_shrink(backend, 2, 8)
            with self.assertRaises(AssertionError):
                init_cuda_graph_state_no_shrink(backend, 2, 32)
        self.assertEqual(backend.calls, [(2, 8)])


class TestControllerSettles(CustomTestCase):
    """The shipped w16_3_15 controller must settle, not oscillate."""

    CFG = {
        "candidate_steps": [3, 15],
        "ema_alpha": 0.1,
        "update_interval": 20,
        "warmup_batches": 15,
        "switch_grace_batches": 40,
        "down_hysteresis": 3.5,
        "up_hysteresis": 0.0,
        "reset_ema_on_switch": False,
    }

    LAG = 2  # verify results reach the controller this many batches late

    def _run(self, accept_at_steps, batches=200, lag=None):
        """Feed a workload whose accepted-draft count depends on the step count.

        *lag* models the real pipeline: the sample handed to ``update`` was
        produced by the step count that was live ``lag`` batches ago, so the
        first samples after a switch still describe the old configuration.
        """
        lag = self.LAG if lag is None else lag
        slot = AdaptiveStepSlot(initial_steps=15, cfg=dict(self.CFG))
        inflight = [slot.current_steps] * lag
        trace = []
        for _ in range(batches):
            produced_by = inflight.pop(0) if lag else slot.current_steps
            slot.update([accept_at_steps[produced_by]])
            inflight.append(slot.current_steps)
            trace.append(slot.current_steps)
        return slot, trace

    def test_prose_settles_at_three(self):
        # measured num_correct_drafts (acc - 1): prose-en 1.96 @15, 1.54 @3
        slot, trace = self._run({15: 1.96, 3: 1.54})
        self.assertEqual(slot.current_steps, 3)
        self.assertEqual(trace[-50:], [3] * 50)

    def test_code_settles_at_fifteen(self):
        # code-edit 8.85 @15, 2.81 @3
        slot, trace = self._run({15: 8.85, 3: 2.81})
        self.assertEqual(slot.current_steps, 15)
        self.assertEqual(trace[-50:], [15] * 50)

    def test_agent_settles_at_three(self):
        # agent-loop 3.76 @15, 2.24 @3 -- W4 is the faster profile there
        slot, trace = self._run({15: 3.76, 3: 2.24})
        self.assertEqual(slot.current_steps, 3)
        self.assertEqual(trace[-50:], [3] * 50)

    def test_stale_samples_are_dropped(self):
        """A leftover steps=15 reading cannot move a steps=3 EMA."""
        slot = AdaptiveStepSlot(initial_steps=3, cfg=dict(self.CFG))
        slot.ema_accept_len = 2.0
        slot.update([9])  # impossible at steps=3 -> stale, ignored
        self.assertEqual(slot.ema_accept_len, 2.0)
        slot.update([3])  # possible at steps=3 -> counted
        self.assertGreater(slot.ema_accept_len, 2.0)

    def test_switch_count_stays_small(self):
        """The 2026-09-05 run made 489 switches in 90s; bound the churn."""
        for name, accept in (
            ("prose-en", {15: 1.96, 3: 1.54}),
            ("code-edit", {15: 8.85, 3: 2.81}),
            ("agent-loop", {15: 3.76, 3: 2.24}),
        ):
            with self.subTest(name):
                _, trace = self._run(accept, batches=300)
                switches = sum(a != b for a, b in zip(trace, trace[1:]))
                self.assertLessEqual(switches, 4, f"{name}: {switches} switches")

    def test_grace_window_holds_decisions_after_a_switch(self):
        """No second decision until the post-switch transient has passed."""
        cfg = dict(self.CFG)
        slot = AdaptiveStepSlot(initial_steps=15, cfg=cfg)
        # Force a switch, then feed values that would otherwise decide again.
        while slot.current_steps == 15:
            slot.update([1.0])
        switched_at = slot._batch_count
        for _ in range(cfg["switch_grace_batches"] - 1):
            self.assertFalse(slot.update([3.0]), "decided inside the grace window")
        self.assertEqual(slot.current_steps, 3)
        self.assertGreaterEqual(
            slot._grace_until - switched_at, cfg["switch_grace_batches"] - 1
        )


class TestPerStateDraftExtendBackend(CustomTestCase):
    """Each runtime state must build its own draft-extend attention backend.

    For compressed-QSA draft models DraftBackendFactory returns the draft
    runner's own backend, and QwenSparseAttnBackend caches captured graph
    metadata in a dict keyed only by (forward_mode, bs). Two states sharing
    one backend therefore collide on the DRAFT_EXTEND_V2 entry: the last one
    captured wins, and the other replays its wider graph over metadata filled
    for the narrower token width. Measured cost of that collision with the
    steps=3 state merely built and never activated: code-edit acceptance
    10.8 -> 5.7, 564 -> 311 t/s.
    """

    def test_sharing_is_off_by_default(self):
        from sglang.srt.speculative import eagle_worker_v2

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_ADAPTIVE_SPLIT", None)
            self.assertEqual(eagle_worker_v2._adaptive_split(), frozenset())
        self.assertNotIn("shared_extend", eagle_worker_v2._adaptive_split())

    def test_split_switches_parse(self):
        from sglang.srt.speculative import eagle_worker_v2

        with patch.dict(
            os.environ, {"SGLANG_ADAPTIVE_SPLIT": "shared_extend, no_draft"}
        ):
            self.assertEqual(
                eagle_worker_v2._adaptive_split(),
                frozenset({"shared_extend", "no_draft"}),
            )


if __name__ == "__main__":
    unittest.main()
