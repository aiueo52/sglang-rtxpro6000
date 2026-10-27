# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
"""Digest-pinned N policy with native fallback outside approved prefill intervals."""

import atexit
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time


MIN_PREFILL_M = 2049
_SOURCE_TREE = Path(__file__).resolve().parents[3]
_VARIANT = Path(__file__).resolve().parents[2].name
EDITION = {"flashinfer-vt": "normal", "flashinfer-vtoea": "hyperpro"}[_VARIANT]
_PRIVATE_ROOT = next((p for p in _SOURCE_TREE.parents if p.name == ".cache"),
                     _SOURCE_TREE.parents[1] / ".cache")
_LOCK = threading.RLock()
_CONFIG = None
_POLICY = None
_VALIDATED = {}
_COUNTERS = None
_COUNTER_PID = None
_LAST_FLUSH = None
_LOGGER = logging.getLogger(__name__)
_MODES = (
    "EXTEND", "SPLIT_PREFILL", "DECODE", "TARGET_VERIFY", "IDLE",
    "DRAFT_EXTEND_V2", "PREBUILT", "MIXED", "DLLM_EXTEND", "MISSING", "UNKNOWN",
)
_TENSORS = (
    "output", "input", "token_selected_experts", "token_final_scales",
    "fc1_expert_weights", "fc2_expert_weights", "fc1_expert_biases",
    "fc2_expert_biases", "input_sf", "swiglu_alpha", "swiglu_beta", "swiglu_limit",
)
_OPTIONS = (
    "tp_size", "tp_rank", "ep_size", "ep_rank", "cluster_size", "cluster_rank",
    "enable_alltoall", "use_deepseek_fp8_block_scale", "use_w4_group_scaling",
    "use_mxfp8_act_scaling", "min_latency_mode", "enable_pdl", "activation_type",
    "use_packed_weights", "use_fused_finalize", "swizzled_input_sf",
)
_DYNAMIC = {"output", "input", "token_selected_experts", "token_final_scales"}


class PolicyError(RuntimeError):
    pass


def policy_enabled():
    global _CONFIG
    switch = os.environ.get("SGLANG_TACTIC_ROWS", "0")
    if switch not in ("0", "1"):
        raise PolicyError("SGLANG_TACTIC_ROWS must be 0 or 1")
    config = (switch, os.environ.get("SGLANG_CACHE_DIR"),
              os.environ.get("SGLANG_TACTIC_ROWS_APPROVED_SHA256"))
    with _LOCK:
        if _CONFIG is None:
            _CONFIG = config
        elif _CONFIG != config:
            raise PolicyError("tactic_rows configuration changed; restart required")
    return switch == "1"


def _pair(ids, *, allow_default=False):
    if (not isinstance(ids, (list, tuple)) or len(ids) != 2
            or any(type(i) is not int or i < (-1 if allow_default else 0) for i in ids)):
        raise PolicyError("expected two absolute GEMM tactic IDs, no implicit fallback")
    return tuple(ids)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PolicyError("duplicate policy key")
        result[key] = value
    return result


def load_policy(*, capturing=False):
    global _POLICY
    if not policy_enabled():
        raise PolicyError("tactic_rows is disabled")
    edition = os.environ.get("FLASH_NEXT_EDITION")
    if edition is not None and edition != EDITION:
        raise PolicyError("server edition differs from imported FlashInfer variant")
    with _LOCK:
        if _POLICY is not None:
            return _POLICY
        if capturing:
            raise PolicyError("load approved policy before CUDA graph capture")
        _, cache_dir, digest = _CONFIG
        if (not cache_dir or not Path(cache_dir).is_absolute()
                or not Path(cache_dir).resolve().is_relative_to(_PRIVATE_ROOT.resolve()) or not digest
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise PolicyError("absolute cache path and approved SHA256 required (N)")
        path = Path(cache_dir) / "tactic_rows" / "policy.json"
        try:
            with path.open("rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
                raise PolicyError("policy digest/size mismatch")
            policy = json.loads(raw, object_pairs_hook=_unique_object)
        except (OSError, ValueError) as error:
            raise PolicyError("cannot read approved tactic_rows policy") from error
        if (not isinstance(policy, dict)
                or set(policy) != {"schema", "edition", "category", "approval", "runtime", "tactic_counts", "buckets"}
                or type(policy["schema"]) is not int or policy["schema"] != 1
                or policy["edition"] != EDITION
                or policy["category"] != "N" or policy["approval"] not in ("trial", "owner")
                or not isinstance(policy["tactic_counts"], list) or len(policy["tactic_counts"]) != 2
                or any(type(n) is not int or n <= 0 for n in policy["tactic_counts"])
                or not isinstance(policy["runtime"], dict)
                or not isinstance(policy["buckets"], dict) or not policy["buckets"]):
            raise PolicyError("invalid/unapproved N policy")
        for key, row in policy["buckets"].items():
            if key not in ("4096", "8192", "16384") or not isinstance(row, dict):
                raise PolicyError("unregistered M bucket")
            bucket = int(key)
            if (set(row) != {"min_m", "max_m", "profile_ids", "signature"}
                    or type(row["min_m"]) is not int or type(row["max_m"]) is not int
                    or not bucket // 2 < row["min_m"] <= row["max_m"] <= bucket
                    or not isinstance(row["signature"], dict)):
                raise PolicyError("invalid approved bucket interval/signature")
            ids = _pair(row["profile_ids"])
            n1, n2 = policy["tactic_counts"]
            if not (0 <= ids[0] < n1 and n1 <= ids[1] < n1 + n2):
                raise PolicyError("approved row has wrong-stage tactic IDs")
        _POLICY = policy
        _LOGGER.info("tactic_rows edition=%s approval=%s (N)", EDITION, policy["approval"])
        atexit.register(flush_counters)
        return policy


def _counter_directory():
    cache = Path(_CONFIG[1])
    directory = cache / "tactic_rows"
    if (not cache.is_absolute()
            or not cache.resolve().is_relative_to(_PRIVATE_ROOT.resolve())
            or not directory.resolve().is_relative_to(_PRIVATE_ROOT.resolve())):
        raise PolicyError("counter directory must remain inside local cache")
    return directory


def flush_counters(*, force=True):
    """Atomic numeric snapshots, throttled during calls and forced at shutdown."""
    global _LAST_FLUSH
    with _LOCK:
        if (_COUNTERS is None or _COUNTER_PID != os.getpid()
                or _CONFIG is None or _CONFIG[0] != "1"):
            return
        now = time.monotonic()
        if not force and _LAST_FLUSH is not None and now - _LAST_FLUSH < 1:
            return
        directory = _counter_directory()
        directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=".counters-", dir=directory)
            with os.fdopen(fd, "w") as stream:
                json.dump(_COUNTERS, stream, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, directory / f"counters-{os.getpid()}.json")
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        _LAST_FLUSH = now


def _record(*, mode=None, bucket=None, fallback=None):
    global _COUNTERS, _COUNTER_PID, _LAST_FLUSH
    with _LOCK:
        if _COUNTERS is None or _COUNTER_PID != os.getpid():
            _COUNTER_PID = os.getpid()
            _LAST_FLUSH = None
            _COUNTERS = {
                "bucket_hits": {key: 0 for key in _POLICY["buckets"]},
                "native": 0, "small": 0, "large": 0, "unregistered": 0, "out_of_range": 0,
                "modes": {key: 0 for key in _MODES},
            }
        if mode is not None:
            _COUNTERS["modes"][mode] += 1
        if bucket is not None:
            _COUNTERS["bucket_hits"][str(bucket)] += 1
        if fallback is not None:
            _COUNTERS[fallback] += 1


def _row(m, *, capturing=False):
    policy = load_policy(capturing=capturing)
    if type(m) is not int or m < 0:
        raise PolicyError("invalid prefill M")
    if m < MIN_PREFILL_M:
        return None, "small"
    if m > max(map(int, policy["buckets"])):
        return None, "large"
    bucket = 1 << (m - 1).bit_length()
    row = policy["buckets"].get(str(bucket))
    if row is None:
        return None, "unregistered"
    if not row["min_m"] <= m <= row["max_m"]:
        return None, "out_of_range"
    return row, None


def requested_profile_ids(m, *, capturing=False):
    row, _ = _row(m, capturing=capturing)
    return list(_pair(row["profile_ids"])) if row is not None else None


def prefill_scope(mode_name):
    # is_prefill()/is_extend() in this tree also include TARGET_VERIFY.
    return mode_name in ("EXTEND", "SPLIT_PREFILL")


def record_forward(mode_name, m, *, capturing=False):
    """Count successful server forwards, not rejected policy/shape requests."""
    load_policy(capturing=capturing)
    mode = "MISSING" if mode_name is None else mode_name
    with _LOCK:
        _record(mode=mode if mode in _MODES else "UNKNOWN")
        if prefill_scope(mode_name):
            row, fallback = _row(m, capturing=capturing)
            _record(bucket=1 << (m - 1).bit_length() if row is not None else None,
                    fallback=fallback)
            if row is None:
                _record(fallback="native")
        else:
            _record(fallback="native")
        if not capturing:
            flush_counters(force=False)


def _tensor_meta(tensor, *, dynamic=False):
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if dynamic:
        shape[0] = "M"
    return {"shape": shape, "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def call_signature(arguments):
    tensors = {name: _tensor_meta(arguments[name], dynamic=name in _DYNAMIC)
               for name in _TENSORS}
    tensors["quant_scales"] = [_tensor_meta(t) for t in arguments["quant_scales"]]
    options = {name: arguments[name] for name in _OPTIONS}
    options["activation_type"] = int(options["activation_type"])
    options["output_dtype"] = str(arguments["output_dtype"])
    return {"tensors": tensors, "options": options}


def approved_profile_ids(arguments, *, capturing=False):
    row, _ = _row(arguments["input"].shape[0], capturing=capturing)
    if row is None:
        return None
    if (any(arguments[name] is not None for name in ("vt_lens", "input_sf"))
            or any(arguments.get(name) is not None for name in ("oea_k0", "oea_stats"))):
        raise PolicyError("VT/OEA/prequantized inputs are outside this prefill policy")
    if call_signature(arguments) != row["signature"]:
        raise PolicyError("unapproved shape/dtype/scales/topology/options")
    ids = _pair(row["profile_ids"])
    supplied = arguments["profile_ids"]
    if supplied is not None and _pair(supplied) != ids:
        raise PolicyError("profile_ids disagree with approved policy")
    return list(ids)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime(library_path, *, sm, device_name, torch_version, cuda_version,
                   capturing=False):
    policy = load_policy(capturing=capturing)
    if capturing:
        raise PolicyError("initialize module/runtime before CUDA graph capture")
    if tuple(sm) != (12, 0) or os.environ.get("FLASHINFER_DISABLE_JIT") != "1":
        raise PolicyError("SM120 and prebuilt local workspace (DISABLE_JIT=1) required")
    workspace = os.environ.get("FLASHINFER_WORKSPACE_BASE", "")
    if (not workspace or not Path(workspace).resolve().is_relative_to(_PRIVATE_ROOT.resolve())
            or not Path(library_path).resolve().is_relative_to(_PRIVATE_ROOT.resolve())):
        raise PolicyError("workspace and loaded library must be inside the cache directory")
    tree = _SOURCE_TREE
    paths = (
        f"{_VARIANT}/flashinfer/fused_moe/core.py",
        f"{_VARIANT}/flashinfer/fused_moe/tactic_rows_policy.py",
        "python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py",
    )
    try:
        runtime = {"sm": list(sm), "device_name": device_name,
                   "torch_version": str(torch_version), "cuda_version": cuda_version,
                   "sources": {p: _sha256(tree / p) for p in paths},
                   "library_sha256": _sha256(library_path)}
    except OSError as error:
        raise PolicyError("missing copied source/prebuilt library") from error
    if runtime != policy["runtime"]:
        raise PolicyError("GPU/source/build identity differs from approved policy")


def validate_profile_ids(runner, ids, *, capturing=False, allow_default=False,
                         expected_counts=None, warmup_key=None):
    """Stage membership + positive occupancy, once before capture, never tune."""
    pair = _pair(ids, allow_default=allow_default)
    if expected_counts is not None and (
            not isinstance(expected_counts, (list, tuple)) or len(expected_counts) != 2
            or any(type(n) is not int or n <= 0 for n in expected_counts)):
        raise PolicyError("expected two positive stage tactic counts")
    key = (id(runner), pair, tuple(expected_counts or ()), warmup_key)
    with _LOCK:
        if key in _VALIDATED:
            return pair
        if capturing:
            raise PolicyError("warm fixed tactics before CUDA graph capture")
        try:
            n1 = runner.get_gemm1_tactic_count()
            n2 = runner.get_gemm2_tactic_count()
            if any(type(n) is not int or n <= 0 for n in (n1, n2)):
                raise PolicyError("loaded runner must expose positive stage tactic counts")
            if expected_counts is not None and (n1, n2) != tuple(expected_counts):
                raise PolicyError("tactic counts differ from approved edition build")
            for stage, tactic in enumerate(pair):
                if tactic == -1:
                    continue
                low, high = (0, n1) if stage == 0 else (n1, n1 + n2)
                if not low <= tactic < high or runner.get_tactic_occupancy(tactic) <= 0:
                    raise PolicyError("wrong-stage/unavailable tactic; refusing fallback")
        except (AttributeError, TypeError) as error:
            raise PolicyError("build lacks strict tactic validation") from error
        _VALIDATED[key] = runner  # Keep a reference: object IDs cannot be reused.
    return pair
