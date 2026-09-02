"""Regression tests for the adaptive speculative state-switch boundary."""

import json
import tempfile
import unittest
from unittest.mock import MagicMock

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StubWorker:
    def __init__(self):
        self.speculative_num_steps = 3
        self.applied_steps = []

    def apply_runtime_state(self, state):
        self.applied_steps.append(state.speculative_num_steps)
        self.speculative_num_steps = state.speculative_num_steps


def _state(steps):
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=object(),
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
        qsa_mtp_shared_sparse_indices=None,
    )


class TestAdaptiveSwitchPoint(CustomTestCase):
    def test_verify_decision_is_applied_only_at_next_batch_boundary(self):
        """Result processing must not replace runtime resources mid-round."""
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump({"1": {"candidate_steps": [3, 15]}}, config_file)
            config_file.flush()
            worker = _StubWorker()
            controller = AdaptiveController(worker, config_path=config_file.name)

        controller.register(_state(3))
        controller.register(_state(15))
        controller.params.on_verify_complete = MagicMock(return_value=15)

        controller.on_verify_complete([3], batch_size=1)
        self.assertEqual(worker.applied_steps, [])

        controller.activate_step_by_batch(1)
        self.assertEqual(worker.applied_steps, [15])

        controller.on_verify_complete([15], batch_size=1)
        self.assertEqual(worker.applied_steps, [15])
        controller.activate_step_by_batch(1)
        self.assertEqual(worker.applied_steps, [15])


if __name__ == "__main__":
    unittest.main()
