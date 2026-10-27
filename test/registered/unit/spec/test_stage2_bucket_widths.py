# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    adaptive_capture_bs_for_step,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
PRODUCTION = REPO / "adaptive/w16_c1.json"
MC = REPO / "adaptive/w16_mc.json"


class _Worker:
    def __init__(self):
        self.speculative_num_steps = 15

    def apply_runtime_state(self, state):
        self.speculative_num_steps = state.speculative_num_steps

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
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


def _controller():
    controller = AdaptiveController(_Worker(), config_path=str(MC))
    controller.init_states(cuda_graph_bs=list(range(1, 9)), max_batch_size=8)
    return controller


class TestStage2BucketWidths(unittest.TestCase):
    def test_pending_decisions_do_not_cross_buckets(self):
        """A bs=1 decision must never select a forbidden width at bs=4 or bs=2."""
        with patch.dict("os.environ", {"SGLANG_ADAPTIVE_POLICY": "ema"}):
            controller = _controller()
            for bs, expected in [(1, 15), (4, 3), (2, 7), (1, 15)]:
                controller.activate_step_by_batch(bs)
                self.assertEqual(controller.worker.speculative_num_steps, expected)
                self.assertIn(
                    expected, controller.params.get_candidate_steps_for_batch(bs)
                )
                # Hold a pending decision in the bucket that just produced a batch.
                with patch.object(
                    controller.params, "on_verify_complete", return_value=expected
                ):
                    controller.on_verify_complete([expected] * bs, batch_size=bs)

    def test_delayed_feedback_updates_producing_bucket(self):
        """Delayed bs=1 feedback cannot change the active bs=4 bucket's history."""
        with patch.dict("os.environ", {"SGLANG_ADAPTIVE_POLICY": "ema"}):
            controller = _controller()
            controller.activate_step_by_batch(1)
            controller.activate_step_by_batch(4)
            before = controller.params._slots[4]._batch_count
            controller.on_verify_complete([1], batch_size=1)
            self.assertEqual(controller.params._slots[1]._batch_count, 1)
            self.assertEqual(controller.params._slots[4]._batch_count, before)
            controller.activate_step_by_batch(2)
            self.assertIn(controller.worker.speculative_num_steps, [3, 7])

    def test_pending_decision_survives_within_same_padded_bucket(self):
        with patch.dict("os.environ", {"SGLANG_ADAPTIVE_POLICY": "ema"}):
            controller = _controller()
            controller.params._slots[2].current_steps = 3
            with patch.object(controller.params, "on_verify_complete", return_value=7):
                controller.on_verify_complete([3, 3], batch_size=2)
            controller.activate_step_by_batch(3)
            self.assertEqual(controller.worker.speculative_num_steps, 7)

    def test_bs1_matches_pre_edit_recorded_decisions(self):
        transitions = {
            "ema": [(0, 15), (25, 3), (215, 7), (255, 15), (405, 7), (445, 3)],
            "confidence": [(0, 15), (25, 3), (215, 7), (295, 15), (445, 3)],
        }
        for policy, changes in transitions.items():
            with self.subTest(policy=policy), patch.dict(
                "os.environ", {"SGLANG_ADAPTIVE_POLICY": policy}
            ):
                controller = _controller()
                trace = []
                expected = []
                change_map = dict(changes)
                steps = 15
                for i in range(600):
                    low = i < 200 or i >= 400
                    controller.observe_confidence([0.7 if low else 0.999], 1)
                    controller.activate_step_by_batch(1)
                    current = controller.worker.speculative_num_steps
                    trace.append(current)
                    controller.on_verify_complete([min(current, 1 if low else 14)], 1)
                    steps = change_map.get(i, steps)
                    expected.append(steps)
                self.assertEqual(trace, expected)


class TestStage2CapturePlan(unittest.TestCase):
    def test_multi_bucket_plan(self):
        bs = list(range(1, 9))
        expected = {15: [1], 7: [1, 2, 3], 3: bs}
        params = AdaptiveSpeculativeParams(initial_steps=15, cfg_path=str(MC))
        params.set_cuda_graph_bs(bs)
        for steps, buckets in expected.items():
            self.assertEqual(
                adaptive_capture_bs_for_step(
                    capture_bs=bs, step=steps, cfg_path=str(MC)
                ),
                buckets,
            )
            self.assertEqual(params.cuda_graph_bs_for_step(steps), buckets)

    def test_production_plan_is_unchanged(self):
        for steps, expected in [(3, [1]), (7, [1]), (15, [1, 2])]:
            self.assertEqual(
                adaptive_capture_bs_for_step(
                    capture_bs=[1, 2], step=steps, cfg_path=str(PRODUCTION)
                ),
                expected,
            )

    def test_sparse_capture_grid_routes_by_padded_bucket(self):
        params = AdaptiveSpeculativeParams(initial_steps=15, cfg_path=str(MC))
        params.set_cuda_graph_bs([1, 4, 8])
        self.assertEqual(params.get_candidate_steps_for_batch(2), [3])
        self.assertEqual(params.get_bucket_for_batch(3), 4)
        self.assertEqual(params.cuda_graph_bs_for_step(7), [1])

    def test_startup_rejects_missing_bucket_graphs(self):
        controller = _controller()
        state = controller._states[3]

        def assert_coverage(required):
            assert set(required).issubset([1, 2, 3, 4, 5, 6, 7])

        state.cuda_graph_runner = SimpleNamespace(
            speculative_num_steps=3, assert_adaptive_graph_coverage=assert_coverage
        )
        with self.assertRaises(AssertionError):
            controller.init_states(cuda_graph_bs=list(range(1, 9)), max_batch_size=8)

    def test_profiles_preserve_production_slot_bytes(self):
        production = json.loads(PRODUCTION.read_text())

        def slot1_bytes(path):
            text = path.read_bytes()
            return text[text.index(b'  "1": {') : text.index(b'  "2": {')]

        for name, candidates in [("w16_mc.json", [3]), ("w16_mc7.json", [3, 7])]:
            path = REPO / "adaptive" / name
            actual = json.loads(path.read_text())
            self.assertEqual(slot1_bytes(path), slot1_bytes(PRODUCTION))
            self.assertEqual(actual["1"], production["1"])
            self.assertEqual(actual["2"], dict(production["2"], candidate_steps=[3, 7]))
            self.assertEqual(
                actual["4"], dict(production["2"], candidate_steps=candidates)
            )
            params = AdaptiveSpeculativeParams(initial_steps=15, cfg_path=str(path))
            self.assertEqual(params.get_candidate_steps_for_batch(4), candidates)


if __name__ == "__main__":
    unittest.main()
