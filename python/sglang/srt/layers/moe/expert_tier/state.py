# SPDX-License-Identifier: Apache-2.0
"""Preallocated layer state and destination tensors; no SGLang dependency."""
import torch
from .reference import SlotReference

# stats: calls, demand, hits, copies, overflows, arena calls,
# arena copies from source, arena copies from cache.
SLOT_BYTES = 2764800


class TierState:
    def __init__(self, k=205, hot=(), g1=None, g2=None, device='cpu', *,
                 policy='lru', half=256, inc=16, skip_empty=False):
        g1 = torch.zeros(512,dtype=torch.float32,device=device) if g1 is None else g1
        g2 = torch.zeros(512,dtype=torch.float32,device=device) if g2 is None else g2
        ref = SlotReference(k,hot,policy=policy,half=half,inc=inc)
        self.k = k
        self.policy, self.half, self.inc = policy, half, inc
        if policy == 'lfu':
            self.count = torch.zeros(512,dtype=torch.int32,device=device)
        self.sync = torch.zeros(2,dtype=torch.int32,device=device)
        self.g1_alphas, self.g2_alphas = g1, g2
        for name in ('slot_of','expert_of','key','stats'):
            setattr(self,name,torch.tensor(getattr(ref,name),device=device))
        self.step = torch.zeros(1,dtype=torch.int64,device=device)
        self.overflow = torch.zeros(1,dtype=torch.int32,device=device)
        self.copy_list = torch.empty((k,2),dtype=torch.int32,device=device)
        self.n_copy = torch.zeros(1,dtype=torch.int32,device=device)
        if skip_empty:
            self.has_copy = torch.empty((), dtype=torch.bool, device=device)
        self.g1_slot = torch.zeros(k,dtype=g1.dtype,device=device)  # not the default dtype (bf16 while SGLang loads)
        self.g2_slot = torch.zeros(k,dtype=g2.dtype,device=device)
        if len(hot):
            ix = torch.tensor(list(hot),dtype=torch.long,device=device)
            self.g1_slot[:len(hot)] = g1[ix]
            self.g2_slot[:len(hot)] = g2[ix]


def allocate_slots(k, device='cpu'):
    # Scale buffers are bytes so copying also preserves arbitrary FP8 bit patterns.
    return (torch.empty((k,1280,1280),dtype=torch.uint8,device=device),
            torch.empty((k,2560,320),dtype=torch.uint8,device=device),
            torch.empty((k,1280,160),dtype=torch.float8_e4m3fn,device=device),
            torch.empty((k,2560,40),dtype=torch.float8_e4m3fn,device=device))
