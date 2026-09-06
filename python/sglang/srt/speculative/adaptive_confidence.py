"""Draft-confidence side channel and the confidence/throughput step policy (C1).

Why a side channel
------------------
The shipped adaptive controller (``adaptive_spec_params.AdaptiveStepSlot``)
picks the chain length from one number: an EMA of ``accept_lens - 1``.  That
statistic is (a) purely historical and (b) not on the same scale as the
quantity the decision actually needs, which is *throughput*:

    tokens/s(S) = (1 + E[accepted | S]) / step_time(S)

This module supplies the two missing ingredients.

1. ``ConfidenceChannel`` makes the draft model's own top-1 probability
   available on the host.  For topk=1 drafting neither producer computed it
   (both wrote a constant 1.0), so this is new information, not a new
   smoothing of the old information.  The probability of chain position 0 --
   the token ``_draft_extend_for_decode`` selected at the end of the previous
   iteration -- is the only confidence available *before* the step-count
   decision is taken, because the whole draft loop runs inside one captured
   CUDA graph (see specs/C1_LOG.md section 1).

2. ``ConfidenceStepSlot`` turns observations into a throughput comparison
   instead of a threshold test.  Modelling per-position acceptance as flat after
   position 2 (a modelling assumption), a chain of length S behaves
   like S iid Bernoulli(r) trials in series:

       E[accepted | S] = sum_{k=1..S} r^k = r (1 - r^S) / (1 - r)

   A *single* per-position acceptance rate ``r`` therefore explains the
   observed mean at whichever S is currently live, and -- crucially --
   predicts the mean at the S that is *not* live.  The EMA has no such
   transfer: it measures a quantity whose scale changes with every switch,
   which is why the shipped config needs the asymmetric "re-seed on step-down
   only" hack.  Checked against the 2026-09-05 fixed-profile measurements:

       workload    E@15   -> r     -> predicted E@3   measured E@3
       code-edit    9.87    0.930      2.60            2.86
       agent-loop   3.82    0.797      1.95            2.29
       prose-en     1.83    0.645      1.33            1.51
       prose-ja     1.55    0.605      1.21            1.31

   (systematically ~15% low, because acceptance is *not* iid at positions 0-1;
   ``position_bias`` absorbs that.)

``r`` is tracked per confidence bucket, so a request whose next token the
draft is sure about is allowed a long chain even while the running average
says otherwise -- which is exactly the code-edit failure mode: its accept
distribution at S=15 is bimodal (full chains interleaved with 0/1), so the
*mean* dips under a fixed threshold even though most steps still want 15.
"""

from __future__ import annotations

import collections
import logging
import math
import os
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _env_policy() -> str:
    return os.environ.get("SGLANG_ADAPTIVE_POLICY", "ema") or "ema"


def _env_trace() -> str:
    return os.environ.get("SGLANG_ADAPTIVE_TRACE", "")


def confidence_policy_enabled() -> bool:
    """The runtime policy needs the position-0 probability (eager, cheap)."""
    return _env_policy() == "confidence"


def chain_trace_enabled() -> bool:
    """Offline analysis needs every chain position (writes inside the draft graph)."""
    return bool(_env_trace())


def confidence_enabled() -> bool:
    return confidence_policy_enabled() or chain_trace_enabled()


def top1_prob(logits: torch.Tensor) -> torch.Tensor:
    """softmax(logits).max(-1) without materialising the softmax.

    ``logsumexp`` + ``amax`` are two reductions over the draft's hot vocabulary
    (49152 wide); at the batch sizes this server runs they cost a few
    microseconds against a 12-20 ms step.
    """
    f = logits.float()
    return (f.amax(dim=-1) - torch.logsumexp(f, dim=-1)).exp()


class ConfidenceChannel:
    """Async device->host staging for draft confidences.

    Two producers, both optional:
      * ``record_position0`` -- eager, at draft-extend, one value per request.
        This is what the policy reads, and it is ready a whole iteration before
        it is needed.
      * ``record_chain`` -- the full ``(bs, steps)`` matrix the Triton
        postprocess kernel filled during the draft.  Trace-only.

    Copies go out non-blocking on the current stream into a ring of pinned
    buffers; each slot carries an event so a reader never has to guess.  Nothing
    here ever synchronises the producing stream.
    """

    RING = 16

    def __init__(self, device: str, max_bs: int, max_steps: int):
        self.device = device
        self.max_bs = max_bs
        self.max_steps = max_steps
        self._p0_host = [
            torch.empty(max_bs, dtype=torch.float32, pin_memory=True)
            for _ in range(self.RING)
        ]
        self._p0_event = [torch.cuda.Event() for _ in range(self.RING)]
        self._p0_bs = [0] * self.RING
        self._p0_write = 0
        self._p0_read = -1  # index of the newest slot with a copy in flight
        self._p0_cached: Optional[list[float]] = None

        self._chain_host = None
        self._chain_event = None
        self._chain_write = 0
        self._chain_pending: collections.deque = collections.deque()
        if chain_trace_enabled():
            self._chain_host = [
                torch.empty((max_bs, max_steps), dtype=torch.float32, pin_memory=True)
                for _ in range(self.RING)
            ]
            self._chain_event = [torch.cuda.Event() for _ in range(self.RING)]

    # -- producers ---------------------------------------------------------
    def record_position0(self, p0: torch.Tensor) -> None:
        bs = p0.shape[0]
        if bs == 0 or bs > self.max_bs or torch.cuda.is_current_stream_capturing():
            return
        slot = self._p0_write
        self._p0_host[slot][:bs].copy_(p0.view(-1), non_blocking=True)
        self._p0_event[slot].record()
        self._p0_bs[slot] = bs
        self._p0_read = slot
        self._p0_write = (slot + 1) % self.RING

    def record_chain(self, chain: torch.Tensor, bs: int, steps: int) -> None:
        if (
            self._chain_host is None
            or bs == 0
            or bs > self.max_bs
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        slot = self._chain_write
        self._chain_host[slot][:bs, :steps].copy_(
            chain[:bs, :steps], non_blocking=True
        )
        self._chain_event[slot].record()
        self._chain_write = (slot + 1) % self.RING
        self._chain_pending.append((slot, bs, steps))

    # -- consumers ---------------------------------------------------------
    def latest_position0(self) -> Optional[list[float]]:
        """Newest position-0 confidences whose copy has ALREADY landed.

        Never synchronises.  The scheduler deliberately runs the CPU ahead of
        the GPU (that is why the adaptive controller is fed from
        ``batch_result_processor`` after ``accept_lens`` is already on the
        host, rather than from the worker hot path), so waiting on an event
        here would collapse the run-ahead and cost far more than the decision
        is worth.  Instead: walk back from the newest slot to the first one
        whose event has completed, and cache the value.  In practice that is
        one or two iterations old, which is inside the lag the controller
        already tolerates -- ``update()`` samples arrive late by the same
        mechanism.  Returns the last known value when nothing new has landed.
        """
        newest = self._p0_read
        if newest < 0:
            return self._p0_cached
        for k in range(self.RING):
            slot = (newest - k) % self.RING
            if self._p0_bs[slot] == 0:
                continue
            if self._p0_event[slot].query():
                self._p0_cached = self._p0_host[slot][: self._p0_bs[slot]].tolist()
                return self._p0_cached
        return self._p0_cached

    def pop_chain(self) -> Optional[tuple[list[list[float]], int]]:
        """Oldest recorded chain, paired FIFO with the verify results."""
        if not self._chain_pending:
            return None
        slot, bs, steps = self._chain_pending.popleft()
        self._chain_event[slot].synchronize()
        return self._chain_host[slot][:bs, :steps].tolist(), steps


class ChainTracer:
    """One line per verified decode step: the input to the offline simulator."""

    def __init__(self, path: str):
        self._f = open(path, "a", buffering=1 << 16)
        self._n = 0

    def write(self, steps: int, bs: int, chain, accepted: list[int]) -> None:
        import json

        self._f.write(
            json.dumps(
                {
                    "i": self._n,
                    "steps": steps,
                    "bs": bs,
                    "p": [[round(x, 5) for x in row] for row in chain]
                    if chain is not None
                    else None,
                    "a": accepted,
                }
            )
            + "\n"
        )
        self._n += 1

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# ms/iteration = STEP_A + STEP_B * S.  Fitted to the 2026-09-05 fnbench
# end-to-end step times (tokens/acc), which is the quantity being maximised:
#   W4  (S=3):  11.7-12.2 ms      W16 (S=15): 19.9-21.9 ms
# The pure-decode profiler numbers quoted in the C1 brief (10.2 / 17.3 ms at
# T=4 / T=16) have the same slope but a smaller intercept; they omit the
# scheduler/detokenizer overhead that fnbench throughput does contain.  Both
# put the S=15 step at ~1.7x the S=3 step, which is all the decision uses.
STEP_A = float(os.environ.get("SGLANG_ADAPTIVE_STEP_A", "9.74"))
STEP_B = float(os.environ.get("SGLANG_ADAPTIVE_STEP_B", "0.70"))


def step_time_ms(steps: int) -> float:
    return STEP_A + STEP_B * steps


def expected_accept(r: float, steps: int) -> float:
    """E[accepted drafts] for a length-*steps* chain at per-position rate *r*."""
    if steps <= 0:
        return 0.0
    r = min(max(r, 1e-4), 0.999999)
    return r * (1.0 - r**steps) / (1.0 - r)


def invert_accept(mean_accept: float, steps: int) -> float:
    """The *r* whose length-*steps* chain accepts *mean_accept* drafts.

    Monotone in r, so a fixed-count bisection is exact enough and has no
    convergence branch to get wrong in a hot path (30 iterations of bisection
    on [0,1) is ~1e-9).
    """
    if steps <= 0 or mean_accept <= 0.0:
        return 0.0
    if mean_accept >= steps - 1e-6:
        return 0.999999
    lo, hi = 0.0, 0.999999
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if expected_accept(mid, steps) < mean_accept:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class ConfidenceStepSlot:
    """Throughput-maximising step choice from a confidence-conditioned rate.

    Drop-in for ``AdaptiveStepSlot``: same ``current_steps`` / ``candidate_steps``
    / ``update()`` contract, plus ``observe_confidence()``.

    Per decision it:
      1. converts the batch's mean accepted-draft count into a per-position
         acceptance ``r`` (which is comparable across step counts, unlike the
         raw mean),
      2. folds it into the EMA of the confidence bucket the *producing* chain
         started from, and into that bucket's occupancy EMA,
      3. picks the candidate with the highest
         ``sum_b w_b (1 + E[accepted | r_b, S]) / step_time(S)``,
         i.e. the expected throughput over the confidence mixture the workload
         is currently producing, not over its mean.

    ``switch_margin`` is the only hysteresis: a candidate must beat the
    incumbent by that relative margin to displace it, which both damps
    oscillation and encodes the un-modelled cost of a switch (the draft state
    is cold for a batch or two afterwards).
    """

    # Position-0 confidence bucket edges.  Coarse on purpose: each bucket has
    # to accumulate its own rate estimate, and a decode server sees a few
    # thousand steps per workload.  The default edges are packed against 1.0
    # because that is where the mass is: on the 2026-09-06 code-edit trace the
    # position-0 probability has a 10th percentile of 0.966 and a median of
    # 1.000, so uniform edges would put 80% of steps in one bucket and
    # discriminate nothing.  Overridable from the config ("buckets").
    BUCKETS = (0.8, 0.95, 0.99, 0.999)

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert candidates, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates
        self.current_steps = (
            initial_steps
            if initial_steps in candidates
            else candidates[len(candidates) // 2]
        )

        self.rate_alpha = float(cfg.get("rate_alpha", 0.15))
        self.update_interval = int(cfg.get("update_interval", 4))
        self.warmup_batches = int(cfg.get("warmup_batches", 15))
        self.switch_grace_batches = int(cfg.get("switch_grace_batches", 20))
        self.switch_margin = float(cfg.get("switch_margin", 0.04))
        # E[accepted] from an iid model is biased low because positions 0-1
        # accept better than the tail; one multiplicative correction, measured
        # at 1.15 across all four workloads (see module docstring).
        self.position_bias = float(cfg.get("position_bias", 1.15))
        # Weight of the confidence-conditioned rate against the pooled rate.
        # 0 reproduces a pure "better statistic" policy (option 3); 1 trusts
        # the buckets outright.  Buckets are blended in proportion to their
        # own sample count either way, so this only sets the ceiling.
        self.confidence_weight = float(cfg.get("confidence_weight", 1.0))
        self.min_bucket_samples = int(cfg.get("min_bucket_samples", 12))
        self.weight_alpha = float(cfg.get("weight_alpha", 0.02))

        self.buckets = tuple(cfg.get("buckets", self.BUCKETS))
        nb = len(self.buckets) + 1
        self._nb = nb
        self._rate = [0.0] * nb
        self._rate_n = [0] * nb
        # Occupancy EMA: how often the incoming chains start from each bucket.
        self._w = [1.0 / nb] * nb
        self._pooled_rate = 0.0
        self._pooled_n = 0
        self._batch_count = 0
        self._grace_until = 0
        self._next_bucket = nb // 2  # until the first confidence arrives
        self._last_conf: Optional[float] = None
        # Bucket the chain currently being verified started from, so the
        # measurement lands in the right bucket when it comes back.
        self._inflight_bucket: collections.deque = collections.deque(maxlen=8)
        self._dbg = os.environ.get("SGLANG_ADAPTIVE_DEBUG", "") == "1"

    # -- inputs ------------------------------------------------------------
    def _bucket(self, conf: float) -> int:
        i = 0
        for edge in self.buckets:
            if conf < edge:
                return i
            i += 1
        return i

    def observe_confidence(self, confidences: list[float]) -> None:
        """Position-0 top-1 probability for the chain about to be drafted."""
        if not confidences:
            return
        # Batch-level statistic: the whole batch shares one chain length, and
        # the *weakest* request bounds how far the shared verify is useful.
        conf = sum(confidences) / len(confidences)
        self._last_conf = conf
        self._next_bucket = self._bucket(conf)
        self._inflight_bucket.append(self._next_bucket)

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        if not num_correct_drafts_per_req:
            return False
        steps = self.current_steps
        if steps > 0:
            # Same staleness guard as the EMA slot: a sample longer than the
            # live chain was produced by the previous, longer state.
            fresh = [n for n in num_correct_drafts_per_req if n <= steps]
            if fresh:
                mean = sum(fresh) / len(fresh)
                r = invert_accept(mean / self.position_bias, steps)
                b = (
                    self._inflight_bucket.popleft()
                    if self._inflight_bucket
                    else self._next_bucket
                )
                a = self.rate_alpha
                self._rate[b] = (
                    r if self._rate_n[b] == 0 else (1 - a) * self._rate[b] + a * r
                )
                self._rate_n[b] += 1
                self._pooled_rate = (
                    r if self._pooled_n == 0 else (1 - a) * self._pooled_rate + a * r
                )
                self._pooled_n += 1
                aw = self.weight_alpha
                for k in range(self._nb):
                    self._w[k] = (1 - aw) * self._w[k] + (aw if k == b else 0.0)

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False
        if self._batch_count < self._grace_until:
            return False
        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False
        return self._recompute()

    # -- decision ----------------------------------------------------------
    def _bucket_rate(self, b: int) -> float:
        """Bucket *b*'s acceptance rate, shrunk toward the pooled rate."""
        n = self._rate_n[b]
        if n == 0 or self.confidence_weight <= 0.0:
            return self._pooled_rate
        w = self.confidence_weight * min(1.0, n / max(1, self.min_bucket_samples))
        return w * self._rate[b] + (1 - w) * self._pooled_rate

    def mixture(self) -> list[tuple[float, float]]:
        """[(weight, rate)] over the confidence buckets currently in play."""
        if self.confidence_weight <= 0.0:
            return [(1.0, self._pooled_rate)]
        tot = sum(self._w) or 1.0
        return [
            (self._w[b] / tot, self._bucket_rate(b))
            for b in range(self._nb)
            if self._w[b] > 1e-4
        ]

    def value(self, steps: int, mix: list[tuple[float, float]]) -> float:
        """Expected tokens per ms at *steps*, integrated over the mixture."""
        num = sum(
            w * (1.0 + self.position_bias * expected_accept(r, steps))
            for w, r in mix
        )
        return num / step_time_ms(steps)

    def best_steps(self, mix: list[tuple[float, float]]) -> tuple[int, float]:
        best, best_tps = self.current_steps, -1.0
        for s in self.candidate_steps:
            tps = self.value(s, mix)
            if s == self.current_steps:
                # The only hysteresis: a challenger must clear the incumbent by
                # this margin, which pays for the cold draft state a switch
                # leaves behind.
                tps *= 1.0 + self.switch_margin
            if tps > best_tps:
                best, best_tps = s, tps
        return best, best_tps

    def _recompute(self) -> bool:
        if self._pooled_n == 0:
            return False
        mix = self.mixture()
        r = self._rate_for_next_dbg = sum(w * r for w, r in mix)
        target, _ = self.best_steps(mix)
        if self._dbg:
            logger.info(
                "[adaptive-conf] steps=%d batch=%d conf=%s rbar=%.4f "
                "w=%s r=%s n=%s -> %d",
                self.current_steps,
                self._batch_count,
                f"{self._last_conf:.3f}" if self._last_conf is not None else "-",
                r,
                [round(x, 3) for x in self._w],
                [round(self._bucket_rate(b), 3) for b in range(self._nb)],
                self._rate_n,
                target,
            )
        if target == self.current_steps:
            return False
        old = self.current_steps
        self.current_steps = target
        self._grace_until = self._batch_count + self.switch_grace_batches
        logger.info(
            "Adaptive spec params updated (confidence): steps %d -> %d "
            "(r=%.3f, conf=%s, E@%d=%.2f, E@%d=%.2f)",
            old,
            target,
            r,
            f"{self._last_conf:.3f}" if self._last_conf is not None else "-",
            old,
            self.value(old, mix) * step_time_ms(old) - 1.0,
            target,
            self.value(target, mix) * step_time_ms(target) - 1.0,
        )
        return True
