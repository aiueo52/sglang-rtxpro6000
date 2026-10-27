# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""DL configuration and sequential CPU reference; no torch/CUDA imports."""
from __future__ import annotations

import math
import os
from typing import NamedTuple


def _parse_tau(raw, name: str) -> float:
    try:
        tau = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return tau


def _parse_tau_by_s(raw) -> dict[int, float]:
    name = "SGLANG_RND_DL_TAU_BY_S"
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{name} must be a nonempty comma-separated S:tau mapping")
    thresholds = {}
    for entry in raw.split(","):
        parts = entry.split(":")
        if len(parts) != 2:
            raise ValueError(f"{name} must contain S:tau pairs")
        key, value = (part.strip() for part in parts)
        if not key.isascii() or not key.isdecimal() or int(key) < 1:
            raise ValueError(f"{name} S must be a positive integer")
        steps = int(key)
        if steps in thresholds:
            raise ValueError(f"{name} contains duplicate S={steps}")
        thresholds[steps] = _parse_tau(value, name)
    return thresholds


def dl_tau(environ=None, *, steps=None) -> float:
    env = os.environ if environ is None else environ
    tau = _parse_tau(env.get("SGLANG_RND_DL_TAU", "0"), "SGLANG_RND_DL_TAU")
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or steps < 1
    ):
        raise ValueError("DL steps must be a positive integer")
    if "SGLANG_RND_DL_TAU_BY_S" not in env:
        return tau
    thresholds = _parse_tau_by_s(env["SGLANG_RND_DL_TAU_BY_S"])
    if steps is None:
        return max(tau, *thresholds.values())
    return thresholds.get(steps, tau)


def _flag_enabled(env, name: str) -> bool:
    flag = env.get(name, "0")
    if flag not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return flag == "1"


def dl_hostw_enabled(environ=None) -> bool:
    """Host-selected verify widths require an enabled DL threshold."""
    env = os.environ if environ is None else environ
    return _flag_enabled(env, "SGLANG_RND_DL_HOSTW") and dl_tau(env) > 0


def dl_stepup31_enabled(environ=None) -> bool:
    """Allow the optional 31-step runtime state only with HOSTW."""
    env = os.environ if environ is None else environ
    return _flag_enabled(env, "SGLANG_RND_DL_STEPUP31") and dl_hostw_enabled(env)


def dl_hostw_steps(environ=None) -> tuple[int, ...]:
    """Execution-only states to reserve before any adaptive graph capture."""
    if not dl_hostw_enabled(environ):
        return ()
    return (3, 7, 15, 31) if dl_stepup31_enabled(environ) else (3, 7, 15)


def dl_stepup_steps(steps, available_steps, environ=None) -> int:
    env = os.environ if environ is None else environ
    enabled = _flag_enabled(env, "SGLANG_RND_DL_STEPUP")
    if steps not in (3, 7, 15):
        return steps
    if steps == 15 and not dl_stepup31_enabled(env):
        return steps
    tau = dl_tau(env, steps=steps)
    if not enabled or tau == 0:
        return steps
    destination = {3: 7, 7: 15, 15: 31}[steps]
    return destination if destination in available_steps else steps


class DraftState(NamedTuple):
    product: float
    length: int
    active: bool


class DraftResult(NamedTuple):
    length: int
    generated_positions: int
    recursive_forwards: int


def update_state(state: DraftState, confidence, k: int, steps: int, tau: float) -> DraftState:
    """Consume confidence k, then decide whether to generate position k+1."""
    tau = _parse_tau(tau, "SGLANG_RND_DL_TAU")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("DL steps must be a positive integer")
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < steps:
        raise ValueError("DL confidence position must be in [0, steps)")
    if not state.active:
        return state
    if tau == 0:
        return DraftState(state.product, k + 1, k + 1 < steps)
    try:
        valid = (
            math.isfinite(confidence)
            and 0 <= confidence <= 1
            and math.isfinite(state.product)
            and 0 <= state.product <= 1
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        return DraftState(0.0, state.length, False)
    product = state.product * confidence
    passed = product >= tau
    return DraftState(
        product if passed else 0.0,
        k + 1 if passed else state.length,
        passed and k + 1 < steps,
    )


def reference(confidences, tau: float, steps=None) -> DraftResult:
    """c0 is already known; the first failed position is generated but not in L."""
    tau = _parse_tau(tau, "SGLANG_RND_DL_TAU")
    if steps is None:
        steps = len(confidences)
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= len(confidences)
    ):
        raise ValueError("DL confidences must contain the positive draft width")
    if tau == 0:
        return DraftResult(steps, steps, steps - 1)
    state = DraftState(1.0, 0, True)
    for k in range(steps):
        state = update_state(state, confidences[k], k, steps, tau)
        if not state.active:
            break
    length = state.length
    return DraftResult(
        length, min(steps, length + 1), max(0, min(steps - 1, length))
    )
