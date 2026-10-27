# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""CPU guards for opt-in confidence-policy transition configuration."""

import random
import unittest
from unittest.mock import patch

import sglang.srt.speculative.adaptive_confidence as policy
from sglang.srt.speculative.adaptive_confidence import ConfidenceStepSlot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class LegacyConfidenceStepSlot(ConfidenceStepSlot):
    """Frozen pre-WA3 selection arithmetic for exact default parity checks."""

    def best_steps(self):
        best, best_tps = self.current_steps, -1.0
        for s in self.candidate_steps:
            tps = self.value(s)
            if s == self.current_steps:
                tps *= 1.0 + self.switch_margin
            elif s > self.current_steps:
                tps /= 1.0 + self.up_margin
            if tps > best_tps:
                best, best_tps = s, tps
        return best, best_tps


class TestConfidenceTransitionConfig(unittest.TestCase):
    def slot(self, current=7, values=None, **extra):
        cfg = dict(candidate_steps=[3, 7, 15], switch_margin=.12, up_margin=.30,
                   switch_grace_batches=40, grace_backoff=2.0,
                   warmup_batches=15, update_interval=10)
        cfg.update(extra)
        slot = ConfidenceStepSlot(current, cfg)
        if values is not None:
            slot.value = values.__getitem__
        return slot

    def test_absent_keys_preserve_exact_legacy_scores(self):
        rng = random.Random(31415)
        for candidates in ([3, 15], [3, 7, 15], [15]):
            for current in candidates:
                cfg = dict(candidate_steps=candidates, switch_margin=.12, up_margin=.30)
                new = ConfidenceStepSlot(current, cfg)
                old = LegacyConfidenceStepSlot(current, cfg)
                for _ in range(200):
                    values = {s: rng.random() for s in candidates}
                    new.value = old.value = values.__getitem__
                    self.assertEqual(new.best_steps(), old.best_steps())

    def test_default_trajectory_matches_legacy(self):
        rng = random.Random(2718)
        cfg = dict(candidate_steps=[3, 15], switch_margin=.12, up_margin=.30,
                   switch_grace_batches=40, grace_backoff=2.0, tail_bias=.75,
                   warmup_batches=15, update_interval=10, weight_alpha=.05,
                   rate_alpha=.1, min_bucket_samples=20)
        new = ConfidenceStepSlot(15, cfg)
        old = LegacyConfidenceStepSlot(15, cfg)
        for i in range(2000):
            confidence = [.9999 if i % 400 < 200 else .5]
            accepted = [new.current_steps if i % 400 < 200 else rng.randrange(3)]
            new.observe_confidence(confidence)
            old.observe_confidence(confidence)
            self.assertEqual(new.update(accepted), old.update(accepted))
            self.assertEqual(vars(new), vars(old))
            self.assertEqual(new.best_steps(), old.best_steps())

    def test_down_margin_does_not_change_upward_threshold(self):
        slot = self.slot(values={3: 1.029, 7: 1.0, 15: 1.455}, down_margin=.03)
        self.assertEqual(slot.best_steps()[0], 7)
        slot.value = {3: 1.031, 7: 1.0, 15: 1.0}.__getitem__
        self.assertEqual(slot.best_steps()[0], 3)
        slot.value = {3: .5, 7: 1.0, 15: 1.457}.__getitem__
        self.assertEqual(slot.best_steps()[0], 15)

    def test_destination_down_margins_separate_code_and_prose(self):
        margins = {"3": .03, "7": .12}
        slot = self.slot(15, {3: .5, 7: 1.11, 15: 1.0}, down_margin=margins)
        self.assertEqual(slot.best_steps()[0], 15)
        slot.value = {3: .5, 7: 1.13, 15: 1.0}.__getitem__
        self.assertEqual(slot.best_steps()[0], 7)
        slot.current_steps = 7
        slot.value = {3: 1.04, 7: 1.0, 15: .5}.__getitem__
        self.assertEqual(slot.best_steps()[0], 3)

    def test_unmapped_down_destination_uses_legacy_margin(self):
        slot = self.slot(15, {3: .5, 7: 1.11, 15: 1.0}, down_margin={"3": .03})
        self.assertEqual(slot.best_steps()[0], 15)

    def test_adjacent_promotion_prevents_skipping_middle(self):
        values = {3: 1.0, 7: 1.2, 15: 2.0}
        slot = self.slot(3, values, adjacent_only_promotion=True, up_margin=0)
        self.assertEqual(slot.best_steps()[0], 7)
        slot.current_steps = 7
        self.assertEqual(slot.best_steps()[0], 15)
        slot.current_steps = 15
        slot.value = {3: 3.0, 7: 1.0, 15: 1.0}.__getitem__
        self.assertEqual(slot.best_steps()[0], 3)  # downshifts may still skip

    def test_adjacent_promotion_must_clear_margin(self):
        slot = self.slot(3, {3: 1.0, 7: 1.1, 15: 3.0},
                         adjacent_only_promotion=True, up_margin=0)
        self.assertEqual(slot.best_steps()[0], 3)

    def test_adjacent_false_is_inert_and_two_candidates_still_promote(self):
        values = {3: 1.0, 7: 1.1, 15: 3.0}
        self.assertEqual(
            self.slot(3, values, adjacent_only_promotion=False).best_steps()[0], 15
        )
        self.assertEqual(self.slot(3, {3: 1.0, 15: 3.0}, candidate_steps=[3, 15],
                                   adjacent_only_promotion=True).best_steps()[0], 15)

    def test_reversal_grace_cap_and_decision_hold(self):
        slot = self.slot(15, max_grace_batches=80)
        for target, expected in [(7, 40), (15, 80), (7, 80), (15, 80)]:
            slot._pmn[slot.current_steps] = 1
            slot.value = lambda s, target=target: 10.0 if s == target else 1.0
            self.assertTrue(slot._recompute())
            self.assertEqual(slot._grace, expected)
            self.assertEqual(slot._grace_until, slot._batch_count + expected)
        slot.value = {3: 1.0, 7: 10.0, 15: 1.0}.__getitem__
        for _ in range(84):
            self.assertFalse(slot.update([0]))
        self.assertTrue(slot.update([0]))  # first interval after the grace ends
        self.assertEqual(slot.current_steps, 7)
        slot.value = {3: 10.0, 7: 1.0, 15: 1.0}.__getitem__
        self.assertTrue(slot._recompute())
        self.assertEqual(slot._grace, 40)  # continuing downward resets backoff
        self.assertEqual(slot.max_grace_batches, 80)
        self.assertEqual(self.slot().max_grace_batches, 2000)

    def test_tail_bias_mapping_follows_live_steps(self):
        mapped = self.slot(3, tail_bias={"3": .95, "7": .75})
        for current, scalar in ((3, .95), (7, .75), (15, .9)):
            mapped.current_steps = current
            expect = self.slot(current, tail_bias=scalar)._extrapolate(2.0, .5, .6, 4)
            self.assertEqual(mapped._extrapolate(2.0, .5, .6, 4), expect)
        self.assertEqual(self.slot(7)._extrapolate(2.0, .5, .6, 4), self.slot(7, tail_bias=.9)._extrapolate(2.0, .5, .6, 4))

    def test_invalid_new_values_rejected(self):
        for bad in (-1, float('nan'), {"3": float('inf')}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.slot(down_margin=bad)
        with self.assertRaises(ValueError):
            self.slot(adjacent_only_promotion="false")


class TestConfidenceStepTimeTable(unittest.TestCase):
    def test_absent_table_matches_frozen_golden_decisions(self):
        # Compare every decision in the frozen Multi trajectory to the read-only
        # reference policy (the same no-table global model).
        import ast
        from pathlib import Path
        path = Path(__file__).resolve().parents[4] / "python/sglang/srt/speculative/adaptive_confidence.py"
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom)
                     and (n.module or "").startswith("sglang"))]
        reference = {"__name__": "_reference_policy", "dl_hostw_enabled": policy.dl_hostw_enabled}
        exec(compile(tree, str(path), "exec"), reference)
        reference.update(STEP_A=9.74, STEP_B=.70)
        cfg = dict(candidate_steps=[3, 7, 15], warmup_batches=5, update_interval=4,
                   switch_grace_batches=8, rate_alpha=.2, weight_alpha=.1,
                   min_bucket_samples=4)
        phases = ((.5, ((0,), (1,), (2,))),
                  (.9999, ((3,),)),
                  (.97, ((7,), (3,), (6,))),
                  (.85, ((1,), (0,))),
                  (.9999, ((3,),)),
                  (.5, ((0,), (2,))))
        with patch.object(policy, "STEP_A", 9.74), patch.object(policy, "STEP_B", .70):
            slot = ConfidenceStepSlot(7, cfg)
            old = reference["ConfidenceStepSlot"](7, cfg)
            trace, expected = [], []
            for confidence, pattern in phases:
                for i in range(40):
                    slot.observe_confidence([confidence])
                    old.observe_confidence([confidence])
                    accepts = list(pattern[i % len(pattern)])
                    changed = slot.update(accepts)
                    before = old.update(accepts)
                    trace.append((changed, slot.current_steps, slot.best_steps()[0]))
                    expected.append((before, old.current_steps, old.best_steps()[0]))
            self.assertEqual(trace, expected)

    def test_measured_ratio_keeps_or_moves_to_three(self):
        cfg = dict(candidate_steps=[3, 7], confidence_weight=0, warmup_batches=4,
                   update_interval=1, up_margin=0, switch_grace_batches=0)
        table = {"3": 12.0, "7": 19.2}
        self.assertAlmostEqual(table["7"] / table["3"], 1.6)
        with patch.object(policy, "STEP_A", 9.74), patch.object(policy, "STEP_B", .70):
            self.assertAlmostEqual(policy.step_time_ms(7) / policy.step_time_ms(3),
                                   1.2364864864864864)
            for initial in (3, 7):
                with self.subTest(initial=initial):
                    global_slot = ConfidenceStepSlot(initial, cfg)
                    measured_slot = ConfidenceStepSlot(initial, dict(cfg, step_time_ms=table))
                    for _ in range(12):
                        for slot in (global_slot, measured_slot):
                            slot.observe_confidence([.97] * 8)
                            slot.update([min(n, slot.current_steps) for n in range(8)])
                    self.assertEqual(global_slot.current_steps, 7)
                    self.assertEqual(measured_slot.current_steps, 3)
                    for s in (3, 7):
                        self.assertEqual(measured_slot.step_time_ms(s), table[str(s)])
                        self.assertEqual(measured_slot.value(s),
                                         (1 + measured_slot.expected_accept_for(s)) / table[str(s)])

    def test_missing_candidate_rejected(self):
        for table in ({}, {"3": 12.0}, {"7": 19.2}):
            with self.subTest(table=table), self.assertRaises(ValueError):
                ConfidenceStepSlot(3, dict(candidate_steps=[3, 7], step_time_ms=table))

    def test_invalid_times_rejected_at_construction(self):
        for bad in (0, -1, float("nan"), float("inf"), float("-inf"),
                    None, "invalid", "19.2", True):
            for candidate in ("3", "7"):
                table = {"3": 12.0, "7": 19.2, candidate: bad}
                with self.subTest(table=table), self.assertRaises(ValueError):
                    ConfidenceStepSlot(3, dict(candidate_steps=[3, 7], step_time_ms=table))
        for table in (None, [], 12.0, {"invalid": 12.0}, {3.5: 12.0, "7": 19.2}):
            with self.subTest(table=table), self.assertRaises(ValueError):
                ConfidenceStepSlot(3, dict(candidate_steps=[3, 7], step_time_ms=table))


if __name__ == "__main__":
    unittest.main()
