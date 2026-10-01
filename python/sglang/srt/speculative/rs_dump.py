"""Post-verify sparse RS dumps for offline analysis; supports TP = 1 only.

Each chunk contains up to 50 CPU records. Normal process exit flushes the tail;
an abrupt termination can lose the current partial chunk. Dumping synchronizes
CPU copies and is intended for measurement runs only.
"""

import atexit
import os
from pathlib import Path
import time

import torch

_records = []
_directory = ""
_chunk = 0


def flush():
    global _chunk
    if not _records:
        return
    directory = Path(_directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(obj=_records, f=directory / f"rs-dump-{os.getpid()}-{_chunk:05d}.pt")
    _records.clear()
    _chunk += 1


def record_verify(
    *, directory, reqs, candidates, target_probs, target_index,
    draft_support_probs, draft_support_tokens, accept_len,
    temperatures, top_ks, top_ps, min_ps,
):
    global _directory
    if _directory != directory:
        flush()
        _directory = directory
    record = {
        "time": time.time(), "rid": [req.rid for req in reqs],
        "input_len": [len(req.origin_input_ids) for req in reqs],
    }
    for name, tensor, dtype in (
        ("candidates", candidates, torch.int32),
        ("target_probs", target_probs, torch.float32),
        ("target_index", target_index, torch.int32),
        ("draft_support_probs", draft_support_probs, torch.float32),
        ("draft_support_tokens", draft_support_tokens, torch.int32),
        ("accept_len", accept_len, torch.int32),
        ("temperatures", temperatures, torch.float32),
        ("top_ks", top_ks, torch.int32),
        ("top_ps", top_ps, torch.float32),
        ("min_ps", min_ps, torch.float32),
    ):
        record[name] = tensor.to(device="cpu", dtype=dtype, copy=True)
    _records.append(record)
    if len(_records) == 50:
        flush()


atexit.register(flush)
