"""CPU-only regression for candidate target warmup before graph capture."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.speculative.adaptive_runtime_state import adaptive_target_graph_warmup

FLAG = 'SGLANG_ADAPTIVE_TARGET_AUTOTUNE'


class TestAdaptiveTargetAutotune(unittest.TestCase):
    def test_disabled_preserves_model_and_constructor_changes(self):
        for value in ('', '0', 'true'):
            with self.subTest(value=value), patch.dict(os.environ, {FLAG: value}):
                base, candidate = object(), object()
                mr = SimpleNamespace(attn_backend=base, _kernel_warmed_up=True)
                with adaptive_target_graph_warmup(mr, candidate):
                    self.assertIs(mr.attn_backend, base)
                    self.assertTrue(mr._kernel_warmed_up)
                    mr._kernel_warmed_up = 'constructor-owned'
                self.assertEqual(mr._kernel_warmed_up, 'constructor-owned')

    def test_enabled_uses_candidate_then_restores(self):
        with patch.dict(os.environ, {FLAG: '1'}):
            base, candidate = object(), object()
            mr = SimpleNamespace(attn_backend=base, _kernel_warmed_up=True)
            with adaptive_target_graph_warmup(mr, candidate):
                self.assertIs(mr.attn_backend, candidate)
                self.assertFalse(mr._kernel_warmed_up)
                mr._kernel_warmed_up = True
            self.assertIs(mr.attn_backend, base)
            self.assertTrue(mr._kernel_warmed_up)

    def test_restores_on_capture_failure(self):
        with patch.dict(os.environ, {FLAG: '1'}):
            base = object()
            mr = SimpleNamespace(attn_backend=base, _kernel_warmed_up=True)
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                with adaptive_target_graph_warmup(mr, object()):
                    raise RuntimeError('capture failed')
            self.assertIs(mr.attn_backend, base)
            self.assertTrue(mr._kernel_warmed_up)

    def test_absent_warmup_attribute_is_restored(self):
        with patch.dict(os.environ, {FLAG: '1'}):
            mr = SimpleNamespace(attn_backend=object())
            with adaptive_target_graph_warmup(mr, object()):
                self.assertFalse(mr._kernel_warmed_up)
            self.assertFalse(hasattr(mr, '_kernel_warmed_up'))

    def test_real_base_warmup_retunes_and_respects_existing_gate(self):
        from sglang.srt.model_executor.runner.base_runner import BaseRunner
        module = 'sglang.srt.model_executor.runner.base_runner'
        for enabled, allowed in ((False, True), (True, False), (True, True)):
            with self.subTest(enabled=enabled, allowed=allowed):
                base, candidate = object(), object()
                mr = SimpleNamespace(attn_backend=base, _kernel_warmed_up=True,
                                     device='cuda', ps=SimpleNamespace(pp_size=1))
                seen = []
                runner = SimpleNamespace(
                    model_runner=mr,
                    _pre_initialize_flashinfer_allreduce_workspace=Mock(),
                    _pre_initialize_fi_a2a_workspace=Mock(),
                    _autotune_buffers=lambda: (object(), 1),
                    _flashinfer_autotune=lambda **kw: seen.append(mr.attn_backend),
                )
                with patch.dict(os.environ, {FLAG: '1' if enabled else '0'}), \
                     patch(module+'.should_run_flashinfer_autotune', return_value=allowed), \
                     patch(module+'.maybe_flashinfer_autotune_extend'):
                    with adaptive_target_graph_warmup(mr, candidate):
                        BaseRunner.warmup(runner)
                        BaseRunner.warmup(runner)
                self.assertEqual(seen, [candidate] if enabled and allowed else [])
                self.assertIs(mr.attn_backend, base)
                self.assertTrue(mr._kernel_warmed_up)


if __name__ == '__main__':
    unittest.main()
