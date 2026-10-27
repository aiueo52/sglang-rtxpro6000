# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
MODULE = "sglang.srt.mem_cache.kv_cache_configurator"
STAGE_BYTES = 36 * (10240 * 3 * 2 + 48 * 128 * 128 * 2)


def _configurator():
    configurator = object.__new__(KVCacheConfigurator)
    params = SimpleNamespace(
        layers=list(range(36)),
        shape=SimpleNamespace(
            conv=[(10240, 3)],
            conv_kernel=4,
            temporal=(48, 128, 128),
            disable_conv_window_dedup=False,
        ),
        dtype=SimpleNamespace(conv=torch.bfloat16),
        is_kda=False,
        mamba_cache_per_req=STAGE_BYTES,
    )
    configurator.mambaish_config = SimpleNamespace(mamba2_cache_params=params)
    configurator.hybrid_gdn_config = configurator.mambaish_config
    configurator.spec_algorithm = SpeculativeAlgorithm.from_string("EAGLE")
    configurator.device = "cuda"
    configurator.model_dtype = torch.bfloat16
    configurator.ps = SimpleNamespace(attn_dp_size=1, pp_size=1)
    return configurator


class TestMambaNoneModeSizing(unittest.TestCase):
    def _run(
        self,
        *,
        enabled,
        mode="none",
        disable_radix=False,
        slots=20,
        adaptive=True,
        launch_tokens=16,
        topk=1,
        layers=36,
    ):
        configurator = _configurator()
        configurator.mambaish_config.mamba2_cache_params.layers = list(range(layers))
        configurator.mambaish_config.mamba2_cache_params.mamba_cache_per_req = (
            STAGE_BYTES * layers // 36
        )
        with (
            envs.SGLANG_OPT_MAMBA_NONE_MODE_SIZING.override(enabled),
            get_context().override_server_args(
                speculative_algorithm="EAGLE",
                speculative_num_steps=15,
                speculative_num_draft_tokens=launch_tokens,
                speculative_eagle_topk=topk,
                speculative_adaptive=adaptive,
                speculative_adaptive_config=str(REPO / "adaptive/w16_mc.json"),
                max_running_requests=4,
                max_mamba_cache_size=slots,
                disable_radix_cache=disable_radix,
                disable_overlap_schedule=False,
                mamba_radix_cache_strategy="extra_buffer",
                gdn_mtp_cache_mode=mode,
                enable_linear_replayssm_spec=False,
                mamba_full_memory_ratio=0.5,
            ),
            patch(f"{MODULE}.logger.info") as log,
        ):
            remaining = configurator._handle_max_mamba_cache(20.0)
            actual_slots = get_schedule().max_mamba_cache_size
        return remaining, actual_slots, log

    def test_flag_off_matches_legacy_reservation(self):
        remaining, slots, log = self._run(enabled=False)
        expected = 20.0 - (STAGE_BYTES * 5 * 16) / (1 << 30)
        expected -= (slots + 1) * STAGE_BYTES / (1 << 30)
        self.assertEqual(remaining, expected)
        log.assert_not_called()

    def test_none_mode_charges_conv_and_recovery_shapes(self):
        """None mode has padded conv windows and unpadded FlashInfer recovery inputs."""
        remaining, slots, log = self._run(enabled=True)
        conv = 36 * 5 * 10240 * 18 * 2
        recovery = 36 * 4 * 16 * (10240 + 96) * 2
        expected = 20.0 - (conv + recovery) / (1 << 30)
        expected -= (slots + 1) * STAGE_BYTES / (1 << 30)
        self.assertEqual(remaining, expected)
        log.assert_called_once_with(
            "Mamba none-mode intermediate reservation: old=%d bytes, new=%d bytes",
            STAGE_BYTES * 5 * 16,
            conv + recovery,
        )

    def test_other_modes_keep_legacy_sizing(self):
        for mode in ["full", "last"]:
            with self.subTest(mode=mode):
                off = self._run(enabled=False, mode=mode)
                on = self._run(enabled=True, mode=mode)
                self.assertEqual(off[:2], on[:2])
                on[2].assert_not_called()

    def test_adaptive_maximum_is_used_instead_of_live_width(self):
        self.assertEqual(
            self._run(enabled=True, launch_tokens=4)[0],
            self._run(enabled=True, launch_tokens=16)[0],
        )

    def test_request_cap_and_padding_slot(self):
        remaining, slots, log = self._run(enabled=True, slots=10)
        conv = 36 * 3 * 10240 * 18 * 2
        recovery = 36 * 2 * 16 * (10240 + 96) * 2
        expected = 20.0 - (conv + recovery) / (1 << 30)
        expected -= (slots + 1) * STAGE_BYTES / (1 << 30)
        self.assertEqual(remaining, expected)
        self.assertEqual(log.call_args.args[2], conv + recovery)

    def test_radix_disabled_uses_same_allocation_shapes(self):
        remaining, slots, log = self._run(enabled=True, slots=None, disable_radix=True)
        self.assertEqual(slots, 4)
        conv = 36 * 5 * 10240 * 18 * 2
        recovery = 36 * 4 * 16 * (10240 + 96) * 2
        self.assertEqual(
            remaining,
            20.0 - (conv + recovery) / (1 << 30) - 5 * STAGE_BYTES / (1 << 30),
        )
        log.assert_called_once()

    def test_dense_tree_windows_keep_full_physical_size(self):
        remaining, slots, log = self._run(enabled=True, adaptive=False, topk=2)
        conv = 36 * 5 * 16 * 10240 * 3 * 2
        recovery = 36 * 4 * 16 * (10240 + 96) * 2
        self.assertEqual(
            remaining,
            20.0
            - (conv + recovery) / (1 << 30)
            - (slots + 1) * STAGE_BYTES / (1 << 30),
        )
        self.assertEqual(log.call_args.args[2], conv + recovery)

    def test_auto_fit_joint_budget_includes_none_mode_scratch(self):
        remaining, slots, log = self._run(enabled=True, slots=None)
        conv_per_req = 36 * 10240 * 18 * 2
        per_req = conv_per_req + 36 * 16 * (10240 + 96) * 2
        budget = 20.0 * 0.5 / 1.5 * (1 << 30)
        self.assertEqual(
            slots,
            int((budget - STAGE_BYTES - conv_per_req) // (STAGE_BYTES + per_req / 5)),
        )
        reservation = 4 * per_req + conv_per_req
        self.assertEqual(
            remaining,
            20.0 - reservation / (1 << 30) - (slots + 1) * STAGE_BYTES / (1 << 30),
        )
        log.assert_called_once()

    def test_flag_off_auto_fit_preserves_legacy_joint_solve(self):
        remaining, slots, log = self._run(enabled=False, slots=None)
        budget = 20.0 * 0.5 / 1.5 * (1 << 30)
        self.assertEqual(
            slots, int((budget - STAGE_BYTES * 17) // (STAGE_BYTES * (1 + 16 / 5)))
        )
        reservation = STAGE_BYTES * 5 * 16
        self.assertEqual(
            remaining,
            20.0 - reservation / (1 << 30) - (slots + 1) * STAGE_BYTES / (1 << 30),
        )
        log.assert_not_called()

    def test_recovery_dtype_is_independent_of_conv_cache_dtype(self):
        configurator = _configurator()
        configurator.mambaish_config.mamba2_cache_params.dtype.conv = torch.float32
        with (
            envs.SGLANG_OPT_MAMBA_NONE_MODE_SIZING.override(True),
            get_context().override_server_args(
                gdn_mtp_cache_mode="none",
                speculative_adaptive=False,
                speculative_num_draft_tokens=16,
                speculative_eagle_topk=1,
            ),
        ):
            per_req, padding = configurator._none_mode_intermediate_sizes(36)
        conv = 36 * 10240 * 18 * 4
        recovery = 36 * 16 * (10240 + 96) * 2
        self.assertEqual(padding, conv)
        self.assertEqual(per_req, conv + recovery)


if __name__ == "__main__":
    unittest.main()
