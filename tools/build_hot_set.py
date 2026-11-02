#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rank all experts per layer from expert-tier trace NPZ snapshots.

Recorder's hdr/counts format counts nonzero routes. Legacy hdr/dem counts calls.
An optional raw format (layers, ids, weights) counts individual nonzero routes.
"""

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS, EXPERTS = 48, 512


def count_trace(path):
    counts = np.zeros((LAYERS, EXPERTS), dtype=np.int64)
    with np.load(path, allow_pickle=False) as trace:
        if 'ids' in trace:
            ids, weights, layers = trace['ids'], trace['weights'], trace['layers']
            if ids.shape != weights.shape or ids.ndim < 2 or layers.shape != (ids.shape[0],):
                raise ValueError('raw trace requires matching ids/weights and one layer per record')
            if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < -1) | (ids >= EXPERTS)):
                raise ValueError('expert IDs must be integers in -1..511')
            for layer, row, weight in zip(layers, ids, weights):
                _check_layer(layer)
                valid = (row >= 0) & (weight != 0)
                counts[int(layer)] += np.bincount(row[valid], minlength=EXPERTS)
        else:
            hdr = trace['hdr']
            field = 'counts' if 'counts' in trace else 'dem'
            frequencies = trace[field]
            if hdr.ndim != 2 or hdr.shape[1] != 4 or frequencies.shape != (len(hdr), EXPERTS):
                raise ValueError('recorder trace requires hdr [N,4] and counts/dem [N,512]')
            if not np.issubdtype(frequencies.dtype, np.integer) or np.any(frequencies < 0):
                raise ValueError('recorder frequencies must be nonnegative integers')
            if field == 'dem' and not np.all((frequencies == 0) | (frequencies == 1)):
                raise ValueError('recorder dem must contain presence bits')
            for layer, presence in zip(hdr[:, 0], frequencies):
                _check_layer(layer)
                counts[int(layer)] += presence.astype(np.int64)
    return counts


def _check_layer(layer):
    if not np.issubdtype(type(layer), np.integer) or not 0 <= layer < LAYERS:
        raise ValueError('layer must be an integer in 0..47')


def build_hot_set(paths):
    counts = np.zeros((LAYERS, EXPERTS), dtype=np.int64)
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            raise ValueError(f'duplicate trace input: {path}')
        seen.add(resolved)
        counts += count_trace(path)
    # Stable sorting retains ascending expert ID for equal counts, including zero.
    return {'K': EXPERTS, 'layers': np.argsort(-counts, axis=1, kind='stable').tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path, help='trace NPZ file(s), in any order')
    parser.add_argument('-o', '--output', required=True, type=Path, help='hot-set JSON path')
    args = parser.parse_args()
    try:
        result = build_hot_set(args.traces)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'build_hot_set: {exc}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
