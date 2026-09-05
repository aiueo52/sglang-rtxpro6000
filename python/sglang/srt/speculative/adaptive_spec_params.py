"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import os
from functools import cached_property
from typing import TYPE_CHECKING

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 3, 7],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [0, 1, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [0, 1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "64": {
        "candidate_steps": [0],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    from sglang.srt.arg_groups.overrides import resolved_view

    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={cfg.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"
        )
    if cfg.speculative_eagle_topk is not None and cfg.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={cfg.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if resolved_view(server_args).enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    if resolved_view(server_args).enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorkerV2 does not implement adaptive)"
        )
    if cfg.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    if cfg.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


def _load_adaptive_config(
    cfg_path: str | None,
) -> tuple[dict, dict[int, dict]]:
    """Load and validate adaptive config.

    Uses ``DEFAULT_ADAPTIVE_CONFIG`` when *cfg_path* is ``None``.
    """
    if cfg_path is not None:
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_ADAPTIVE_CONFIG

    bs_entries: dict[int, dict] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue

        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s >= 0 for s in steps)
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be a list of non-negative ints, "
                f"got {steps!r}"
            )
        bs_entries[int(key)] = entry

    if not bs_entries:
        raise ValueError(
            "speculative_adaptive_config must contain at least one integer-string "
            'BS key, e.g. {"1": {"candidate_steps": [1,3,7]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return cfg, bs_entries


def resolve_candidate_steps_from_config(
    cfg_path: str | None = None,
) -> list[int]:
    """Union of every BS slot's candidate steps; sizes the runtime buffers."""
    _, bs_entries = _load_adaptive_config(cfg_path)
    all_steps: set[int] = set()
    for entry in bs_entries.values():
        all_steps.update(entry["candidate_steps"])
    return sorted(all_steps)


class AdaptiveStepSlot:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    - num_steps can be selected from different candidate sets on different batch_sizes
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)
        # One EMA is shared by every candidate, but the quantity it tracks
        # (accepted drafts per verify) is bounded by the *current* step count,
        # so its scale changes with every switch: an EMA of 5.9 that justified
        # leaving steps=15 is an impossibly high reading at steps=3 and would
        # bounce straight back up. Re-seeding it at the new step's neutral
        # value (target - 1, the same convention __init__ and the steps=0 probe
        # use) makes each candidate's threshold a statement about that
        # candidate, and turns a switch into a probe that the next update
        # confirms or undoes. Off by default (upstream behaviour).
        self.reset_ema_on_switch = cfg.get("reset_ema_on_switch", False)
        # Batches after a switch during which no new decision may be taken.
        # A swap leaves the draft model cold (its per-step KV branches and the
        # state's QSA shared-index buffers were captured for the other step
        # count), so the first verifies after it under-report acceptance. With
        # a short decision interval the controller reads its own transient,
        # decides the new setting is bad, switches back -- and never observes
        # either setting's steady state. The EMA keeps updating through the
        # grace window; only the decision is held.
        self.switch_grace_batches = cfg.get("switch_grace_batches", 0)
        self._grace_until = 0

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0
        # Debug accounting (SGLANG_ADAPTIVE_DEBUG=1): what the EMA actually saw
        # since the last decision.
        self._dbg = os.environ.get("SGLANG_ADAPTIVE_DEBUG", "") == "1"
        self._dbg_reset()

    def _dbg_reset(self) -> None:
        self._dbg_n = 0
        self._dbg_kept = 0
        self._dbg_sum = 0.0
        self._dbg_dropped_sum = 0.0
        self._dbg_hist: dict[int, int] = {}

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        if self.current_steps > 0:
            # Verify results arrive one or more batches behind the decision
            # that produced them (the scheduler processes them off the overlap
            # queue, and a decided switch is only applied at the next batch
            # boundary), so right after a switch the first samples still carry
            # the *previous* step count. A sample larger than the current chain
            # cannot have been produced by it, which makes those stale samples
            # identifiable and droppable -- without them a single leftover
            # steps=15 reading (up to 15 accepted drafts) drags a freshly
            # seeded steps=3 EMA straight back over the step-up threshold, and
            # the server ping-pongs every decision interval.
            fresh = [
                n for n in num_correct_drafts_per_req if n <= self.current_steps
            ]
            if self._dbg:
                self._dbg_n += len(num_correct_drafts_per_req)
                self._dbg_kept += len(fresh)
                self._dbg_sum += sum(fresh)
                self._dbg_dropped_sum += sum(num_correct_drafts_per_req) - sum(fresh)
                for n in num_correct_drafts_per_req:
                    self._dbg_hist[n] = self._dbg_hist.get(n, 0) + 1
            if fresh:
                batch_avg = sum(fresh) / len(fresh)
                self.ema_accept_len = (
                    1 - self.ema_alpha
                ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if self._batch_count < self._grace_until:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        if self._dbg:
            kept_avg = self._dbg_sum / self._dbg_kept if self._dbg_kept else float("nan")
            logger.info(
                "[adaptive-dbg] steps=%d batch=%d ema=%.2f seen=%d kept=%d "
                "kept_avg=%.2f hist=%s",
                self.current_steps,
                self._batch_count,
                self.ema_accept_len,
                self._dbg_n,
                self._dbg_kept,
                kept_avg,
                sorted(self._dbg_hist.items()),
            )
            self._dbg_reset()
        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        # EMA ceiling: only caps downward — never blocks step-ups, so the
        # system can explore higher steps and let the EMA catch up.
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            decided_at = self.ema_accept_len
            self.current_steps = target
            self._grace_until = self._batch_count + self.switch_grace_batches
            if self.reset_ema_on_switch and target > 0:
                self.ema_accept_len = float(target - 1)
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={decided_at:.2f}"
                + (
                    f", ema reseeded to {self.ema_accept_len:.2f})"
                    if self.ema_accept_len != decided_at
                    else ")"
                ),
            )
            return True
        return False


class AdaptiveSpeculativeParams:
    """Routes ``batch_size`` to the correct per-BS slot.

    A slot is a per-BS configuration of adaptive step selection.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
    ):
        cfg, bs_entries = _load_adaptive_config(cfg_path)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._cuda_graph_bs: list[int] | None = None

        for bs, entry in sorted(bs_entries.items()):
            self._slots[bs] = AdaptiveStepSlot(
                initial_steps=initial_steps,
                cfg={**cfg, **entry},
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """Union of all BS slots' candidate steps."""
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int) -> int:
        return self._route(batch_size).current_steps

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        """Feed verify results to the matching BS slot's EMA.

        Returns the new step if a switch is warranted, else ``None``.
        """
        params = self._route(batch_size)
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Return cuda_graph_bs values that can reach *step* at runtime.

        Returns ``None`` when CUDA graphs are disabled (``set_cuda_graph_bs``
        was never called or was called with ``None``).
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidate_steps
        ]

    def can_reach_step(self, step: int, max_batch_size: int) -> bool:
        """Whether any runnable batch size can select *step*."""
        return any(
            step in self._route(batch_size).candidate_steps
            for batch_size in range(1, max_batch_size + 1)
        )

    def _route(self, batch_size: int) -> AdaptiveStepSlot:
        """Map *batch_size* → pad to CUDA-graph BS → closest slot."""
        return self._slots[
            self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        ]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]
