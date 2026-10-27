# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""VT configuration and CPU reference; no torch/CUDA imports."""
from __future__ import annotations

import math
import os

from sglang.srt.speculative.dl_policy import _parse_tau, dl_tau


def vt_tau(environ=None, *, steps=None) -> float:
    env = os.environ if environ is None else environ
    tau = _parse_tau(env.get("SGLANG_RND_VT_TAU", "0"), "SGLANG_RND_VT_TAU")
    draft_tau = dl_tau(env, steps=steps)
    if "SGLANG_RND_DL_TAU_BY_S" in env:
        return draft_tau
    if draft_tau > 0:
        if tau > 0 and tau != draft_tau:
            raise ValueError("positive SGLANG_RND_VT_TAU must equal SGLANG_RND_DL_TAU")
        return draft_tau  # An explicit VT zero also allows DL to imply VT.
    return tau


def validate_environment(environ=None) -> float:
    env = os.environ if environ is None else environ
    tau = vt_tau(env)
    if tau == 0:
        return tau
    if float(env.get("SGLANG_SIMULATE_ACC_LEN", "-1")) > 0:
        raise ValueError("VT cannot be combined with simulated acceptance")
    return tau


def vt_grammar_enabled(environ=None) -> bool:
    """Grammar-constrained requests run on the VT path (1, the default) or are refused at admission (0)."""
    env = os.environ if environ is None else environ
    flag = env.get("SGLANG_RND_VT_GRAMMAR", "1")
    if flag not in ("0", "1"):
        raise ValueError("SGLANG_RND_VT_GRAMMAR must be 0 or 1")
    return flag == "1"


def prefix_length(confidences, tau: float) -> int:
    """Largest prefix whose untempered top-1 probability product is >= tau."""
    tau = vt_tau({"SGLANG_RND_VT_TAU": tau})
    if tau == 0:
        return len(confidences)
    product = 1.0
    length = 0
    for k, confidence in enumerate(confidences):
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            break  # Fail closed, like the device kernel.
        product *= confidence
        if product < tau:
            break
        length = k + 1
    return length
