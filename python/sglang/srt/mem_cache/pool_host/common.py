# Modified by aiueo52 for flash-next-fast (2026); see MODIFICATIONS.md.

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict

import torch

from sglang.srt.mem_cache.storage.mmap import alloc_mmap

logger = logging.getLogger(__name__)

_PREPIN_ALIGNMENT = 2 * 1024 * 1024
_PREPIN_ROOM_SLACK = 64 * 1024 * 1024
_prepin_lock = threading.Lock()
_prepin_base = None
_prepin_size = 0
_prepin_cursor = 0
_prepin_initialized = False
_prepin_slices = {}
_prepin_hold = None


def prepin_host_arena() -> None:
    """Populate and register one process-wide host arena before loading weights."""
    global _prepin_base, _prepin_size, _prepin_initialized, _prepin_hold
    with _prepin_lock:
        if _prepin_initialized:
            return
        value = os.environ.get("SGLANG_HICACHE_PREPIN_BYTES", "")
        if not value:
            return
        if re.fullmatch(r"[+-]?[0-9]+", value) is None or int(value, 10) < 0:
            raise ValueError(
                "SGLANG_HICACHE_PREPIN_BYTES must be a non-negative decimal "
                f"integer (bytes); got {value!r}"
            )
        n = int(value, 10)
        if n == 0:
            return
        logger.info("hicache prepin: start bytes=%d", n)
        start = time.perf_counter()
        base = alloc_mmap((n,), torch.uint8)
        populated = time.perf_counter()
        logger.info(
            "hicache prepin: populated bytes=%d in %.2f s", n, populated - start
        )
        _cuda_host_register(base)
        registered = time.perf_counter()
        logger.info(
            "hicache prepin: registered bytes=%d in %.2f s (total %.2f s)",
            n, registered - populated, registered - start,
        )
        # Publish only after successful registration; retain the owning mapping.
        _prepin_base = base
        _prepin_hold = base
        _prepin_size = n
        _prepin_initialized = True


def prepin_room_bytes() -> int:
    """Conservative pool capacity remaining in the already registered arena."""
    with _prepin_lock:
        if _prepin_base is None:
            return 0
        off = (
            (_prepin_cursor + _PREPIN_ALIGNMENT - 1)
            // _PREPIN_ALIGNMENT * _PREPIN_ALIGNMENT
        )
        return max(0, _prepin_size - off - _PREPIN_ROOM_SLACK)


class HostTensorAllocator:
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        return alloc_mmap(dims, dtype)


class ShmHostTensorAllocator(HostTensorAllocator):
    def __init__(self):
        super().__init__()
        self.fds = []
        self.mms = []

    @property
    def fd(self):
        return self.fds[0] if self.fds else None

    @property
    def mm(self):
        return self.mms[0] if self.mms else None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        assert (
            device == "cpu"
        ), f"ShmHostTensorAllocator only supports CPU allocations; got device={device!r}"
        self.dtype = dtype
        self.dims = dims
        from sglang.srt.mem_cache.storage.mmap import alloc_shm

        tensor, fd, mm = alloc_shm(dims, dtype)
        self.fds.append(fd)
        self.mms.append(mm)
        return tensor

    def __del__(self):
        for fd in getattr(self, "fds", []):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fds = []


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    elif allocator_type == "mori":
        try:
            from sglang.srt.mem_cache.storage.umbp.umbp_host_allocator import (
                UMBPHostTensorAllocator,
            )

            return UMBPHostTensorAllocator()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "UMBPHostTensorAllocator unavailable (%s). "
                "Falling back to torch.empty-based allocator.",
                exc,
            )
            return HostTensorAllocator()
    elif allocator_type == "shm":
        return ShmHostTensorAllocator()
    else:
        return HostTensorAllocator()


def get_allocator_type() -> str:
    """The host-allocator kind the published HiCache configuration asks for."""
    from sglang.srt.runtime_context import get_memory

    backend = get_memory().hicache_storage_backend
    if backend == "shm":
        return "shm"
    if backend == "dynamic":
        extra_config_str = get_memory().hicache_storage_backend_extra_config
        if extra_config_str:
            try:
                config = json.loads(extra_config_str)
                if config.get("allocator") == "shm":
                    return "shm"
            except Exception:
                pass
    return backend or "default"


def _cuda_host_register(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    n_bytes = buffer.numel() * buffer.element_size()
    rc = cudart.cudaHostRegister(buffer.data_ptr(), n_bytes, 0)
    if int(rc) != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (rc={int(rc)}, "
            f"{cudart.cudaGetErrorString(rc)}) for ptr={buffer.data_ptr():#x} "
            f"size={n_bytes}; host buffer is not pinned and device transfers "
            f"may silently return stale data."
        )


def _cuda_host_unregister(buffer: torch.Tensor) -> None:
    global _prepin_base
    with _prepin_lock:
        if buffer.data_ptr() in _prepin_slices:
            del _prepin_slices[buffer.data_ptr()]
            if _prepin_slices:
                return
            # The last live slice releases the registration of the whole arena; the mapping itself stays alive in _prepin_hold so remaining views never dangle.
            base = _prepin_base
            try:
                _unregister_host_buffer(base)
            finally:
                _prepin_base = None
                _prepin_slices.clear()
            return
    _unregister_host_buffer(buffer)


def _unregister_host_buffer(buffer: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostUnregister(buffer.data_ptr())
    if int(rc) != 0:
        # Best-effort on shutdown: warn, don't raise -- a leak is reclaimed at exit.
        logger.warning(
            "cudaHostUnregister failed (rc=%d, %s) for ptr=%#x",
            int(rc),
            cudart.cudaGetErrorString(rc),
            buffer.data_ptr(),
        )


def alloc_with_host_register(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    no_room = None
    if pin_memory and type(allocator) is HostTensorAllocator:
        with _prepin_lock:
            if _prepin_base is not None:
                n_bytes = math.prod(dims) * dtype.itemsize
                if n_bytes > 0:
                    assert (
                        device == "cpu"
                    ), f"HostTensorAllocator only supports CPU allocations; got device={device!r}"
                    global _prepin_cursor
                    off = (
                        (_prepin_cursor + _PREPIN_ALIGNMENT - 1)
                        // _PREPIN_ALIGNMENT * _PREPIN_ALIGNMENT
                    )
                    left = max(0, _prepin_size - off)
                    if off + n_bytes <= _prepin_size:
                        ptr = _prepin_base.data_ptr() + off
                        array = (ctypes.c_uint8 * n_bytes).from_address(ptr)
                        buffer = torch.frombuffer(array, dtype=dtype).reshape(dims)
                        allocator.dtype = dtype
                        allocator.dims = dims
                        _prepin_slices[ptr] = array
                        _prepin_cursor = off + n_bytes
                        logger.info(
                            "hicache prepin: slice bytes=%d offset=%d left=%d",
                            n_bytes, off, _prepin_size - _prepin_cursor,
                        )
                        return buffer
                    no_room = (n_bytes, left)
    if _prepin_base is not None and pin_memory:
        n_bytes = math.prod(dims) * dtype.itemsize
        if n_bytes > 0:
            # Import lazily: base imports common while defining its pool classes.
            from sglang.srt.mem_cache.pool_host.base import host_free_budget_bytes

            free = host_free_budget_bytes()
            if n_bytes > free:
                raise ValueError(
                    "Not enough host memory available. Requesting "
                    f"{n_bytes / 1e9:.2f} GB but only have {free / 1e9:.2f} GB free "
                    "(the HiCache prepin arena could not hold this buffer). "
                    "Please reduce the size of the hierarchical cache or set "
                    "SGLANG_HICACHE_PREPIN_BYTES=0."
                )
    if no_room is not None:
        logger.warning(
            "hicache prepin: no room for bytes=%d (left=%d); "
            "using the normal path",
            *no_room,
        )
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        _cuda_host_register(buffer)
    return buffer


def alloc_with_pin_memory(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
        "musa": alloc_with_pin_memory,
    },
)
