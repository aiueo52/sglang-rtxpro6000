# SPDX-License-Identifier: Apache-2.0
"""Keep read-only expert file pages young; executable without package imports."""
import argparse
import ctypes
import json
import logging
import math
import mmap
import os
from pathlib import Path
import platform
import time
from contextlib import ExitStack

import numpy as np

PAGE = mmap.PAGESIZE
TOUCH_CHUNK = 64 * 1024 * 1024
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
_LIBC.mincore.restype = ctypes.c_int


class ParentGone(Exception):
    """Stop a pass between files when its owner has exited."""


def residency(mapping):
    """Return one boolean per page, without faulting the file into memory."""
    view = np.frombuffer(mapping, dtype=np.uint8)
    vector = np.empty((len(view) + PAGE - 1) // PAGE, dtype=np.uint8)
    if len(view) and _LIBC.mincore(view.ctypes.data, len(view), vector.ctypes.data):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return (vector & 1).astype(bool)


def resident_bytes(pages, size):
    """Count bytes, clipping the last (possibly partial) page to file size."""
    count = int(np.count_nonzero(pages)) * PAGE
    if len(pages) and pages[-1]:
        count -= (-size) % PAGE
    return count


def missing_ranges(pages, size):
    """Yield page-aligned starts and file-clipped lengths of absent runs."""
    edges = np.flatnonzero(np.diff(np.r_[False, ~pages, False]))
    for first, last in edges.reshape(-1, 2):
        start = int(first) * PAGE
        yield start, min(int(last) * PAGE, size) - start


def keep_pass(maps, *, parent_alive=None):
    """Prefetch absent runs, CPU-touch every page, then measure residency.

    maps is an iterable of read-only mmap objects. Residency counts are bytes,
    including only actual file bytes for a partial final page.
    """
    started = time.monotonic()
    stats = dict(files=0, bytes_total=0, resident_before=0, resident_after=0,
                 willneed_bytes=0, touch_seconds=0.0, pass_seconds=0.0)
    for mapping in maps:
        if parent_alive is not None and not parent_alive():
            raise ParentGone
        size = len(mapping)
        before = residency(mapping)
        stats['files'] += 1
        stats['bytes_total'] += size
        stats['resident_before'] += resident_bytes(before, size)
        for start, length in missing_ranges(before, size):
            mapping.madvise(mmap.MADV_WILLNEED, start, length)
            stats['willneed_bytes'] += length
        touching = time.monotonic()
        view = np.frombuffer(mapping, dtype=np.uint8)
        try:
            for start in range(0, size, TOUCH_CHUNK):
                # Reduction forces reads; slicing alone would not touch pages.
                view[start:min(start + TOUCH_CHUNK, size):PAGE].max()
        finally:
            del view
        stats['touch_seconds'] += time.monotonic() - touching
        stats['resident_after'] += resident_bytes(residency(mapping), size)
    stats['pass_seconds'] = time.monotonic() - started
    return stats


def parent_is_alive(parent, initial_ppid):
    if os.getppid() != initial_ppid:
        return False
    try:
        os.kill(parent, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def lower_priority():
    os.nice(19)
    # Linux syscall numbers; do not guess on unsupported architectures.
    number = {'x86_64': 251, 'aarch64': 30, 'riscv64': 30,
              'i386': 289, 'i686': 289}.get(platform.machine())
    if number is None:
        logging.warning('keeper: ioprio_set unsupported on %s', platform.machine())
    elif _LIBC.syscall(ctypes.c_long(number), ctypes.c_int(1), ctypes.c_int(0),
                       ctypes.c_int((2 << 13) | 7)) == -1:
        logging.warning('keeper: ioprio_set failed: %s', os.strerror(ctypes.get_errno()))


def write_stats(path, stats):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(stats, indent=2) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(files, period, parent, stats_path=None):
    if not math.isfinite(period) or not 5 <= period <= 3600:
        raise ValueError('keeper period must be in 5..3600 seconds')
    if parent <= 0:
        raise ValueError('keeper parent must be a positive pid')
    initial_ppid = os.getppid()
    alive = lambda: parent_is_alive(parent, initial_ppid)
    lower_priority()
    with ExitStack() as stack:
        maps = []
        for path in files:
            if not alive():
                return
            with open(path, 'rb') as file:
                maps.append(stack.enter_context(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)))
        number = 0
        while alive():
            started = time.monotonic()
            timestamp = time.time()
            try:
                stats = keep_pass(maps, parent_alive=alive)
            except ParentGone:
                return
            number += 1
            if stats_path is not None:
                write_stats(stats_path, dict(stats, pass_number=number, timestamp=timestamp))
            deadline = started + period
            while alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(1.0, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period', type=float, required=True)
    parser.add_argument('--parent', type=int, required=True)
    parser.add_argument('--stats')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    run(args.files, args.period, args.parent, args.stats)


if __name__ == '__main__':
    main()
