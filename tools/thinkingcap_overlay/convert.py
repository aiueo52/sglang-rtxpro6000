#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

"""Minimal stdlib streaming helpers retained from the ThinkingCap converter.

This module does not convert quantized tensors or import a tensor framework.
Headers, shapes, offsets and complete payload sizes are checked before copying.
"""
import json
import math
import os
import struct
from pathlib import Path

CHUNK = 64 << 20
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
    "I64": 8, "U64": 8, "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def nbytes(dtype, shape):
    if dtype not in DTYPE_BYTES or not isinstance(shape, (list, tuple)):
        raise ValueError(f"unsupported dtype or shape: {dtype} {shape}")
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError(f"invalid shape: {shape}")
    return DTYPE_BYTES[dtype] * math.prod(shape)


def read_header(path):
    """(validated header without __metadata__, metadata, data_start)."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated header length")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > 200 << 20 or n > size - 8:
            raise ValueError(f"{path}: invalid header length {n}")
        h = json.loads(fh.read(n), object_pairs_hook=unique_object)
    if not isinstance(h, dict):
        raise ValueError(f"{path}: header is not an object")
    meta = h.pop("__metadata__", None)
    if meta is not None and (not isinstance(meta, dict) or
                             any(not isinstance(v, str) for v in meta.values())):
        raise ValueError(f"{path}: invalid metadata")
    ranges = []
    for name, tensor in h.items():
        if not name or not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"{path}: invalid tensor {name!r}")
        offsets = tensor["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(v) is not int or v < 0 for v in offsets)):
            raise ValueError(f"{path}: invalid offsets for {name}")
        a, b = offsets
        if b - a != nbytes(tensor["dtype"], tensor["shape"]):
            raise ValueError(f"{path}: shape/offset size mismatch for {name}")
        ranges.append((a, b))
    end = 0
    for a, b in sorted(ranges):
        if a != end:
            raise ValueError(f"{path}: overlapping or noncontiguous tensors")
        end = b
    if end != size - 8 - n:
        raise ValueError(f"{path}: truncated or excess tensor payload")
    return h, meta, 8 + n


class HeaderCache:
    def __init__(self):
        self._c = {}

    def get(self, path):
        path = str(path)
        if path not in self._c:
            self._c[path] = read_header(path)
        return self._c[path]

    def tensor(self, path, name):
        """(dtype, shape, abs_start, abs_end)."""
        h, _, ds = self.get(path)
        t = h[name]
        a, b = t["data_offsets"]
        return t["dtype"], list(t["shape"]), ds + a, ds + b


def iter_range(path, start, end, chunk=CHUNK):
    if start < 0 or end < start or chunk <= 0:
        raise ValueError("invalid byte range or chunk size")
    with open(path, "rb") as fh:
        fh.seek(start)
        left = end - start
        while left:
            data = fh.read(min(chunk, left))
            if not data:
                raise IOError(f"short read {path}")
            left -= len(data)
            yield data


class Writer:
    """Stream to an exclusively created partial file; never overwrite an output."""

    def __init__(self, path, entries, metadata=None):
        self.path = Path(path)
        if os.path.lexists(self.path):
            raise FileExistsError(self.path)
        header = {"__metadata__": metadata or {"format": "pt"}}
        off = 0
        self.order = []
        for name, dtype, shape in entries:
            if not name or name in header:
                raise ValueError(f"duplicate or invalid tensor name: {name}")
            n = nbytes(dtype, shape)
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
            self.order.append((name, n))
            off += n
        self.total = off
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)
        self.tmp = self.path.with_name(self.path.name + ".writing")
        self.fh = open(self.tmp, "xb")
        try:
            self.fh.write(struct.pack("<Q", len(raw)))
            self.fh.write(raw)
        except BaseException:
            self.abort()
            raise
        self.i = 0
        self.cur_left = None

    def begin(self, name):
        if self.cur_left is not None:
            raise RuntimeError("previous tensor not ended")
        exp_name, n = self.order[self.i]
        if exp_name != name:
            raise RuntimeError(f"writer order: expected {exp_name}, got {name}")
        self.cur_left = n

    def write(self, data):
        if self.cur_left is None or len(data) > self.cur_left:
            raise RuntimeError("tensor overflow or no active tensor")
        self.fh.write(data)
        self.cur_left -= len(data)

    def end(self):
        if self.cur_left != 0:
            raise RuntimeError(f"tensor {self.order[self.i][0]} short by {self.cur_left}")
        self.i += 1
        self.cur_left = None

    def close(self):
        if self.i != len(self.order) or self.cur_left is not None:
            raise RuntimeError("not all tensors written")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        # Atomic publication without replacing an existing file or following a link.
        os.link(self.tmp, self.path)
        self.tmp.unlink()

    def abort(self):
        self.fh.close()
        if self.tmp.exists():
            self.tmp.unlink()  # only the partial file this writer exclusively created


def shard_name(name):
    if (not isinstance(name, str) or not name.endswith(".safetensors") or
            "/" in name or "\\" in name or "\x00" in name or ":" in name or name.startswith(".")):
        raise ValueError(f"invalid shard filename: {name!r}")
    return name


def load_index(d):
    data = json.loads((Path(d) / "model.safetensors.index.json").read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object)
    if not isinstance(data, dict) or not isinstance(data.get("weight_map"), dict) or not data["weight_map"]:
        raise ValueError(f"{d}: missing or empty weight_map")
    wm = data["weight_map"]
    for name, filename in wm.items():
        if not name:
            raise ValueError(f"{d}: empty tensor name")
        shard_name(filename)
    return wm


def check_index(d, wm, hc, allow_missing=False):
    """Check complete shards and their exact index membership; return ready files."""
    by_file = {}
    for name, filename in wm.items():
        by_file.setdefault(filename, set()).add(name)
    ready = set()
    for filename, names in by_file.items():
        path = Path(d) / filename
        if allow_missing and not path.exists():
            continue
        h, _, _ = hc.get(path)
        if set(h) != names:
            raise ValueError(f"{path}: header and index disagree")
        ready.add(filename)
    return ready


def check_pause(pause_file):
    if pause_file and Path(pause_file).exists():
        print(f"PAUSE present ({pause_file}); stopping at a boundary", flush=True)
        return True
    return False


def mem_ok():
    """Allow two streaming buffers; absence of Linux memory statistics is OK."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 >= 2 * CHUNK
    except OSError:
        pass
    return True
