# SPDX-License-Identifier: Apache-2.0
"""Device-only decode recording, with eager atomic NPZ snapshots."""
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .predict import demand

FIELDS = ('hdr', 'counts', 'dem', 'rank1', 'prob1', 'votes1', 'rank2', 'prob2', 'votes2')
STATE = ('slot_of', 'expert_of', 'key', 'step', 'stats')


def configuration():
    raw = os.environ.get('SGLANG_EXPERT_TIER_TRACE_CAP', '65536')
    try:
        cap = int(raw)
    except ValueError:
        raise ValueError('SGLANG_EXPERT_TIER_TRACE_CAP must be an integer in 1024..262144') from None
    if not 1024 <= cap <= 262144:
        raise ValueError('SGLANG_EXPERT_TIER_TRACE_CAP must be in 1024..262144')
    path = os.environ.get('SGLANG_EXPERT_TIER_TRACE', '')
    if path and not Path(path).is_absolute():
        raise ValueError('SGLANG_EXPERT_TIER_TRACE must be an absolute directory')
    return path, cap


class Recorder:
    def __init__(self, directory, cap, device):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cap = cap
        self.count = torch.zeros(1, dtype=torch.int64, device=device)
        self.ring = {}
        for name in FIELDS:
            dtype = torch.int32 if name in ('hdr', 'counts') else torch.float16 if name.startswith('prob') else torch.uint8
            self.ring[name] = torch.empty((cap, 4 if name == 'hdr' else 512), dtype=dtype, device=device)
        self.flushed = self.files = self.lost = 0
        self.warned = False

    def record(self, layer, ids, hidden, weights, route_weights=None):
        valid = (hidden is not None and hidden.dtype == torch.bfloat16
                 and hidden.shape == (ids.shape[0], weights.shape[2]))
        if not valid and not self.warned:
            logging.getLogger(__name__).warning('expert tier trace prediction skipped: hidden must be BF16 [M, H]')
            self.warned = True
        active = (ids >= 0).any(dim=1)
        # Graph capture forbids host-list copies: build the header with device fills only.
        hdr = torch.full((4,), layer, dtype=torch.int32, device=ids.device)
        hdr[1:2].fill_(ids.shape[0])
        hdr[2:3].copy_(active.sum())
        hdr[3:4].fill_(int(valid))
        # Preserve route multiplicity and exclude weight-only pruning as well.
        valid_ids = ids >= 0
        if route_weights is not None:
            valid_ids = valid_ids & (route_weights != 0)
        counts = torch.zeros(512, dtype=torch.int32, device=ids.device)
        counts.scatter_add_(0, ids.clamp_min(0).reshape(-1).long(),
                            valid_ids.reshape(-1).to(torch.int32))
        values = dict(hdr=hdr, counts=counts, dem=demand(ids).to(torch.uint8))
        for d in (1, 2):
            rank = torch.full((512,), 255, dtype=torch.int32, device=ids.device)
            prob = torch.zeros(512, dtype=torch.float32, device=ids.device)
            votes = torch.zeros(512, dtype=torch.int32, device=ids.device)
            if valid and layer+d < 48:
                logits = F.linear(hidden, weights[layer+d])
                top = logits.topk(16, dim=-1).indices
                ranks = torch.arange(16, dtype=torch.int32, device=ids.device).expand_as(top)
                rank.scatter_reduce_(0, top.reshape(-1), torch.where(active[:, None], ranks, 255).reshape(-1), reduce='amin')
                prob = torch.where(active[:, None], logits.float().softmax(-1), 0).amax(dim=0)
                votes.scatter_add_(0, top[:, :10].reshape(-1), active[:, None].expand(-1, 10).reshape(-1).int())
            values.update({f'rank{d}': rank.to(torch.uint8), f'prob{d}': prob.half(), f'votes{d}': votes.to(torch.uint8)})
        position = self.count.remainder(self.cap)
        for name in FIELDS:
            self.ring[name].index_copy_(0, position, values[name].unsqueeze(0))
        self.count.add_(1)

    def flush(self, layers):
        end = int(self.count.item())
        if end == self.flushed and self.files:
            return
        start = max(self.flushed, end-self.cap)
        lost = start-self.flushed
        indices = torch.arange(start, end, device=self.count.device).remainder(self.cap)
        data = {name: tensor.index_select(0, indices).cpu().numpy() for name, tensor in self.ring.items()}
        for name in STATE:
            data[name] = torch.stack([getattr(layers[l], name) for l in range(48)]).cpu().numpy()
        data['step'] = data['step'].reshape(48)
        data.update(seq_start=start, seq_end=end, lost=lost, time=time.time())
        path = self.directory / f'trace-{self.files:06d}.npz'
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez(stream, **data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.flushed, self.files, self.lost = end, self.files+1, self.lost+lost

    def report(self):
        return dict(dir=str(self.directory), cap=self.cap, count=int(self.count.item()), files=self.files, lost=self.lost)
