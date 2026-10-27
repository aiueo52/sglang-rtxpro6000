# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Per-call pointers for OEA in FlashInfer's P2 prologue.

oea-sd (OEA on the 2-D tree): the device cell is int32[2] = {k0, min_rows}. FlashInfer's merged prologue
(flashinfer-vtoea, fused_moe_120_vtoea_prologue_v1) applies OEA only when at least min_rows rows take part in the
call: the VT live rows 0..L when VT passes vt_lens, every row otherwise. min_rows starts at SGLANG_OEA_MIN_ROWS
(16: OEA only on full 16-row verifies, the regime the spare entry was measured in) and the control file may switch
it at run time (2..WIDTH), together with mode and k0. The "rows" counter is the live rows.

The control thread writes the cell's two words with one 8-byte store. It reads the control file only after
mark_serving(), which run_scheduler_process (managers/scheduler.py) calls once Scheduler(...) has returned, i.e. after
every CUDA graph of this process is captured: applying a switch synchronizes the device, which would break a capture.
It says so in one "serving" line of the status file; `oea` writes nothing before that line.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import os
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
STAT_NAMES = (
    "calls", "rows", "valid_routes", "kept_routes", "d_before", "d_after",
    "mass_frac_sum", "oea_calls",
)
K0 = envs.SGLANG_OEA_K0.get() or 0
MODE = envs.SGLANG_OEA_MODE.get()
WIDTH = envs.SGLANG_OEA_WIDTH.get()
MAX_BS = envs.SGLANG_OEA_MAX_BS.get()
MIN_ROWS = envs.SGLANG_OEA_MIN_ROWS.get()
STATS = envs.SGLANG_OEA_STATS.get()
CTL_FILE = envs.SGLANG_OEA_CTL_FILE.get()
if not 0 <= K0 <= 9 or MODE not in {"oea", "prod"}:
    raise ValueError("OEA requires k0 in [0, 9] and mode in {oea, prod}")
if WIDTH < 1 or MAX_BS < 1 or not 2 <= MIN_ROWS <= WIDTH:
    raise ValueError("OEA requires positive width/batch size and 2 <= min_rows <= width")
ARMED = K0 > 0
_CONTEXT = ContextVar("oea_target_forward", default=None)
_STATE = {}
_LOCK = threading.RLock()
_SERVING = threading.Event()  # set by mark_serving(); the control threads wait for it
_warned_capture = False


@contextmanager
def target_forward_context(forward_batch):
    if not ARMED or forward_batch is None:
        yield
        return
    target_verify = forward_batch.forward_mode.is_target_verify()
    width = forward_batch.spec_info.draft_token_num if target_verify else 0
    token = _CONTEXT.set((target_verify, width, forward_batch.batch_size))
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def initial_config():
    # prune_singleton exports these before the first MoE call; explicit FlashInfer values win.
    tau = float(os.environ.get("FLASHINFER_MOE_PRUNE_SINGLETON_TAU", "0"))
    return dict(
        mode=MODE, k0=K0, min_rows=MIN_ROWS, tau=0.0, renorm=0, fallback=1,
        prod_tau=tau if tau > 0 else 0.0,
        prod_min_rank=max(1, int(os.environ.get("FLASHINFER_MOE_PRUNE_MIN_RANK", "1"))),
        prod_min_rows=int(os.environ.get("FLASHINFER_MOE_PRUNE_MIN_ROWS", "2")),
    )


def _state(device):
    global _warned_capture
    device = torch.device(device)
    st = _STATE.get(device)
    if st is not None:
        return st
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        if not _warned_capture:
            _warned_capture = True
            logger.warning("OEA state was not allocated before CUDA graph capture; omitting pointers")
        return None
    with _LOCK:
        if device not in _STATE:
            cfg = initial_config()
            _STATE[device] = dict(
                cell=torch.tensor(_cell_values(cfg), dtype=torch.int32, device=device),
                stats=torch.zeros(8, dtype=torch.float64, device=device),
                cfg_host=cfg, seq=None,
            )
            if CTL_FILE:
                _start_ctl_thread(device, _STATE[device])
    return _STATE[device]


def _cell_values(cfg):
    """{k0, min_rows} as the kernel reads them; mode prod writes k0 = 0 (production prune)."""
    return [cfg["k0"] if cfg["mode"] == "oea" else 0, cfg["min_rows"]]


def _write_cell(cell, cfg):
    """Both words with one store: the int32[2] cell viewed as one int64 and filled once. The kernel reads the two
    words with two 32-bit loads, so a store between them can still mix a pair, but the window shrinks from the gap
    between two kernel launches to the gap between two adjacent loads (every mixed pair is a valid setting)."""
    packed = torch.tensor(_cell_values(cfg), dtype=torch.int32).view(torch.int64).item()
    cell.view(torch.int64).fill_(packed)


def mark_serving():
    """The process has captured all its CUDA graphs (run_scheduler_process, after Scheduler(...) returned): the control
    threads may read the control file from now on. Idempotent; no allocation, no CUDA call."""
    _SERVING.set()


def call_kwargs(num_rows, *, device=None):
    if not ARMED:
        return {}
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
    st = _state(device)
    if st is None:
        return {}
    out = {"oea_stats": st["stats"]} if STATS else {}
    ctx = _CONTEXT.get()
    if ctx is not None:
        verify, width, bs = ctx
        # The row gate itself is in the kernel (live rows >= cell[1], switchable at run time). This host check must not
        # read the run-time value: CUDA graphs keep whatever pointer the capture passed.
        if verify and width == WIDTH and bs <= MAX_BS and num_rows >= 2:
            out["oea_k0"] = st["cell"]
    return out


def read_stats(dev=None):
    if not _STATE:
        return None
    dev = torch.device(dev) if dev is not None else next(iter(_STATE))
    stats = _STATE[dev]["stats"]
    return dict(zip(STAT_NAMES, stats.detach().cpu().tolist()))


def parse_ctl(text):
    conv = dict(seq=int, mode=str, k0=int, tau=float, renorm=int, min_rows=int,
                fallback=int, prod_tau=float, prod_min_rank=int, prod_min_rows=int, snap=int)
    out = {}
    for tok in text.split():
        if "=" not in tok:
            raise ValueError(f"bad token {tok!r}")
        key, value = tok.split("=", 1)
        if key not in conv or key in out:
            raise ValueError(f"unknown or repeated key {key!r}")
        out[key] = conv[key](value)
    if "seq" not in out:
        raise ValueError("missing seq=")
    return out


def _validated_config(req, current):
    cfg = dict(current)
    cfg.update({k: v for k, v in req.items() if k not in {"seq", "snap"}})
    if cfg["mode"] not in {"oea", "prod"} or not 1 <= cfg["k0"] <= 9:
        raise ValueError("unsupported mode or k0")
    if not 2 <= cfg["min_rows"] <= WIDTH:
        raise ValueError(f"unsupported min_rows={cfg['min_rows']!r}; allowed 2..{WIDTH}")
    fixed = dict(tau=0, renorm=0, fallback=1,
                 prod_tau=current["prod_tau"], prod_min_rank=current["prod_min_rank"],
                 prod_min_rows=current["prod_min_rows"])
    for key, value in fixed.items():
        if cfg[key] != value:
            raise ValueError(f"unsupported {key}={cfg[key]!r}; effective value is {value!r}")
    return cfg


def apply_control(text, device, st):
    rec = dict(pid=os.getpid(), t=time.time())
    try:
        seq_values = [tok[4:] for tok in text.split() if tok.startswith("seq=")]
        if len(seq_values) != 1:
            raise ValueError("expected exactly one seq=")
        seq = int(seq_values[0])
        rec["seq"] = seq
        if seq == st["seq"]:
            return None
        # Rejected sequences are recorded once, without changing configuration or device state.
        st["seq"] = seq
        req = parse_ctl(text)
        cfg = _validated_config(req, st["cfg_host"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        snap = read_stats(device)
        if device.type == "cuda":
            with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device)):
                _write_cell(st["cell"], cfg)
                torch.cuda.current_stream().synchronize()
        else:
            _write_cell(st["cell"], cfg)
        st["cfg_host"] = cfg
        rec.update(cfg=cfg, stats=snap)
    except (ValueError, RuntimeError) as exc:
        rec.update(error=str(exc), text=text, cfg=dict(st["cfg_host"]), stats=read_stats(device))
    return rec


def _write_status(ctl_file, rec):
    try:
        with open(ctl_file + ".status.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        logger.exception("Could not write OEA control status")


def _start_ctl_thread(device, st):
    ctl_file, serving = CTL_FILE, _SERVING  # bound at start (a module reload must not move a running thread)

    def loop():
        # oea-sd: one start record (no seq), so `oea status` shows k0/min_rows before the first switch.
        with _LOCK:
            _write_status(ctl_file, dict(pid=os.getpid(), t=time.time(), start=True, cfg=dict(st["cfg_host"])))
        # The first MoE call is the start's FlashInfer autotune (EagerRunner, before any graph runner); the target,
        # draft and adaptive graphs are captured after it. apply_control synchronizes the device, which would
        # invalidate a capture on the forward thread: read nothing until mark_serving().
        serving.wait()
        with _LOCK:
            _write_status(ctl_file, dict(pid=os.getpid(), t=time.time(), serving=True, cfg=dict(st["cfg_host"])))
        last = None
        while True:
            time.sleep(0.25)
            try:
                with open(ctl_file) as f:
                    text = f.read().strip()
            except OSError:
                continue
            if not text or text == last:
                continue
            last = text
            with _LOCK:
                rec = apply_control(text, device, st)
            if rec is not None:
                _write_status(ctl_file, rec)
    threading.Thread(target=loop, daemon=True, name="oea-ctl").start()
