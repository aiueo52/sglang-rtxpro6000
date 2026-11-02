# SPDX-License-Identifier: Apache-2.0
"""Read-only safetensors index, mmap addresses, and complete pinned storage."""
import json
import logging
import mmap
import os
from pathlib import Path
import re
import struct
import numpy as np
import torch

PATTERN = re.compile(r'^(.*)\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(down|gate|up)_proj\.(weight|weight_scale|weight_scale_2|input_scale)$')
FILES = re.compile(r'^layer-(\d{5})-experts-(\d{4})-(\d{4})\.safetensors$')
KINDS = (('down','weight'),('gate','weight'),('up','weight'),
         ('down','weight_scale'),('gate','weight_scale'),('up','weight_scale'))
SHAPES = {'down':{'weight':[2560,320],'weight_scale':[2560,40]},
          'gate':{'weight':[640,1280],'weight_scale':[640,160]},
          'up':{'weight':[640,1280],'weight_scale':[640,160]}}
SIZES = (819200,819200,819200,102400,102400,102400)
# Keep mapped addresses and pinned backing buffers alive until process exit.
_MAPPINGS = []
_PINNED = []


class SourceIndex:
    def __init__(self, directory, layers=48, experts=512, shard_size=128):
        self.directory = Path(directory)
        self.layers,self.experts = layers,experts
        self.files,self.entries,self.mmaps,self.views = [],{},{},{}
        self.table = np.full((layers,512,6,2),-1,np.int64)
        found = set()
        for path in sorted(Path(directory).iterdir()):
            match = FILES.fullmatch(path.name)
            if not match:
                continue
            l,lo,hi = map(int,match.groups())
            if l>=layers:
                continue
            if lo>=experts or hi!=min(lo+shard_size,experts)-1 or lo%shard_size:
                raise ValueError(f'invalid shard range: {path.name}')
            fi = len(self.files)
            self.files.append(path)
            with path.open('rb') as f:
                nraw=f.read(8)
                if len(nraw)!=8:
                    raise ValueError('short safetensors header')
                n=struct.unpack('<Q',nraw)[0]
                if n>path.stat().st_size-8:
                    raise ValueError('header exceeds file')
                header=json.loads(f.read(n))
            start=8+n
            if start%8:
                raise ValueError('data start must be 8-byte aligned')
            intervals=[]
            for name,item in header.items():
                if name=='__metadata__':
                    continue
                m=PATTERN.fullmatch(name)
                if m is None:
                    raise ValueError(f'unrecognized tensor: {name}')
                prefix,ls,es,proj,kind=m.groups()
                ll,e=int(ls),int(es)
                if ll!=l or not lo<=e<=hi:
                    raise ValueError('tensor outside shard')
                scalar=kind in ('input_scale','weight_scale_2')
                dtype='F32' if scalar else ('U8' if kind=='weight' else 'F8_E4M3')
                shape=[] if scalar else SHAPES[proj][kind]
                size=4 if scalar else (819200 if kind=='weight' else 102400)
                a,b=item['data_offsets']
                if item['dtype']!=dtype or item['shape']!=shape or b-a!=size:
                    raise ValueError(f'dtype/shape/size mismatch: {name}')
                if a<0 or b<a or start+b>path.stat().st_size:
                    raise ValueError('tensor extends outside file')
                if not scalar and (start+a)%8:
                    raise ValueError('tensor address must be 8-byte aligned')
                key=(l,e,proj,kind)
                if key in self.entries:
                    raise ValueError('duplicate tensor')
                self.entries[key]=(fi,start+a,size)
                found.add((l,e,proj,kind))
                intervals.append((a,b))
                if not scalar:
                    self.table[l,e,KINDS.index((proj,kind))]=(fi,start+a)
            intervals.sort()
            if any(b>c for (a,b),(c,d) in zip(intervals,intervals[1:])):
                raise ValueError('overlapping tensors')
        for l in range(layers):
            for e in range(experts):
                for p in ('down','gate','up'):
                    for k in ('weight','weight_scale','weight_scale_2','input_scale'):
                        if (l,e,p,k) not in found:
                            raise ValueError(f'missing tensor {(l,e,p,k)}')
        self.base=None
        self.offsets=None
        self.pinned=None

    def open_mmaps(self, willneed=None):
        if willneed is None:
            willneed=os.environ.get('SGLANG_EXPERT_TIER_WILLNEED','0')=='1'
        for fi,path in enumerate(self.files):
            if fi not in self.mmaps:
                with path.open('rb') as f:
                    self.mmaps[fi]=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
                self.views[fi]=np.frombuffer(self.mmaps[fi],dtype=np.uint8)
                _MAPPINGS.append((self.mmaps[fi],self.views[fi]))
                if willneed:
                    self.mmaps[fi].madvise(mmap.MADV_WILLNEED)
        return self

    def tensor_bytes(self,l,e,part):
        self.open_mmaps()
        fi,pos=self.table[l,e,part]
        if fi<0:
            raise ValueError('missing tensor')
        return self.views[int(fi)][int(pos):int(pos)+SIZES[part]]

    def cpu_buffer(self):
        """Test layout: concatenate complete fake files into one CPU tensor."""
        self.open_mmaps()
        starts=np.cumsum([0]+[len(self.views[i]) for i in range(len(self.files))])
        base=torch.from_numpy(np.concatenate([self.views[i] for i in range(len(self.files))]))
        off=np.zeros((self.layers,512,6),np.int64)
        valid=self.table[...,0]>=0
        off[valid]=starts[self.table[...,0][valid]]+self.table[...,1][valid]
        return base,torch.from_numpy(off)

    def bind(self,device='cuda',mode=None,hot=None,pin_memory=True,selection=None, source_shift=0):
        """Return a pointer anchor and offsets; retain host storage for lifetime.

        selection limits benchmark pinned allocation; normal use selects every
        expert. Offsets for deliberately omitted experts are not usable.
        """
        mode=mode or os.environ.get('SGLANG_EXPERT_TIER_SOURCE','mmap')
        self.open_mmaps()
        self.base=torch.empty(16,dtype=torch.uint8,device=device)
        offsets=np.zeros((self.layers,512,6),np.int64)
        if mode=='mmap':
            for l in range(self.layers):
                for e in range(self.experts):
                    for p in range(6):
                        fi,pos=self.table[l,e,p]
                        offsets[l,e,p]=self.views[int(fi)].ctypes.data+pos-self.base.data_ptr()
        elif mode=='pinned':
            selected=[(l,e) for l in range(self.layers) for e in range(self.experts)
                      if selection is None or (l,e) in selection]
            if source_shift not in (0,8):
                raise ValueError("source_shift must be 0 or 8")
            size=len(selected)*sum(SIZES)+source_shift
            logging.getLogger(__name__).warning('expert tier pinned source: %d bytes',size)
            self.pinned=torch.empty(size,dtype=torch.uint8,pin_memory=pin_memory)
            _PINNED.append(self.pinned)
            cursor=source_shift
            for l,e in selected:
                for p,n in enumerate(SIZES):
                    self.pinned[cursor:cursor+n].numpy()[:]=self.tensor_bytes(l,e,p)
                    offsets[l,e,p]=self.pinned.data_ptr()+cursor-self.base.data_ptr()
                    cursor+=n
        else:
            raise ValueError('source must be mmap or pinned')
        self.offsets=torch.tensor(offsets,device=device)
        return self.base,self.offsets


ROUTER_PATTERN = re.compile(r'^(.*)\.layers\.(\d+)\.mlp\.gate\.weight$')


def load_router_weights(directory, device='cpu'):
    """Read non-expert safetensors headers and BF16 bytes, without safetensors."""
    entries = {}
    hidden = None
    for path in sorted(Path(directory).glob('*.safetensors')):
        if FILES.fullmatch(path.name):
            continue
        size = path.stat().st_size
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f'short router safetensors header: {path.name}')
            n = struct.unpack('<Q', raw)[0]
            if n > size - 8:
                raise ValueError(f'router header exceeds file: {path.name}')
            header = json.loads(f.read(n))
        start = 8 + n   # bytes are read with f.read, so the data start need not be aligned
        for name, item in header.items():
            match = ROUTER_PATTERN.fullmatch(name)
            if match is None or name.startswith('mtp') or '.mtp.' in name:
                continue
            layer = int(match[2])
            if not 0 <= layer < 48:
                raise ValueError(f'router layer outside 0..47: {name}')
            if layer in entries:
                raise ValueError(f'duplicate router layer {layer}: {name}')
            shape = item['shape']
            if (item['dtype'] != 'BF16' or len(shape) != 2 or shape[0] != 512
                    or type(shape[1]) is not int or shape[1] <= 0
                    or (hidden is not None and shape[1] != hidden)):
                raise ValueError(f'router dtype/shape mismatch (BF16 [512, H]): {name}')
            hidden = shape[1]
            a, b = item['data_offsets']
            if a < 0 or b - a != 512 * hidden * 2 or start + b > size:
                raise ValueError(f'router byte range/size mismatch: {name}')
            entries[layer] = (path, start + a, b - a)
    missing = sorted(set(range(48)) - entries.keys())
    if missing:
        raise ValueError(f'missing router layers: {missing}')
    weights = torch.empty((48, 512, hidden), dtype=torch.bfloat16, device=device)
    for layer, (path, offset, size) in entries.items():
        with path.open('rb') as f:
            f.seek(offset)
            raw = bytearray(f.read(size))
        if len(raw) != size:
            raise ValueError(f'short router tensor: layer {layer} in {path.name}')
        weights[layer].copy_(torch.frombuffer(raw, dtype=torch.bfloat16).view(512, hidden))
    return weights
