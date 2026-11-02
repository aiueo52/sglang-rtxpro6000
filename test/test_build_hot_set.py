# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the standalone trace converter."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/build_hot_set.py'
spec = importlib.util.spec_from_file_location('build_hot_set', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_recorder_trace_cli(tmp_path):
    # Actual Recorder schema: one distinct-demand presence bit per call.
    dem = np.zeros((4, 512), np.uint8)
    dem[0, [9, 2]] = 1
    dem[1, [9, 3]] = 1
    dem[2, [2, 3]] = 1
    dem[3, 511] = 1
    hdr = np.array([[0, 2, 2, 0], [0, 2, 2, 0], [0, 2, 2, 0], [47, 1, 1, 0]], np.int32)
    trace, output = tmp_path / 'trace.npz', tmp_path / 'hot set.json'
    np.savez(trace, hdr=hdr, dem=dem)
    result = subprocess.run([sys.executable, '-B', str(SCRIPT), str(trace), '-o', str(output)],
                            capture_output=True, text=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert set(data) == {'K', 'layers'}
    assert data['K'] == 512 and len(data['layers']) == 48
    assert data['layers'][0][:4] == [2, 3, 9, 0]
    assert data['layers'][47][:2] == [511, 0]
    assert data['layers'][1] == list(range(512))
    assert all(sorted(layer) == list(range(512)) for layer in data['layers'])


def test_raw_counts_pruning_and_multiple_files(tmp_path):
    trace = tmp_path / 'raw.npz'
    np.savez(trace, layers=np.array([0, 0, 47]),
             ids=np.array([[7, 7, -1, 4], [2, 3, 4, 8], [3, 2, -1, 9]]),
             weights=np.array([[1, .2, 1, 0], [1, 1, 0, 0], [0, .3, 0, 0]]))
    counts = builder.count_trace(trace)
    assert counts[0, 7] == 2 and counts[0, 2] == counts[0, 3] == 1
    assert counts[0, 4] == counts[0, 8] == counts[0, 511] == 0
    assert counts.sum() == 5 and counts[47, 2] == 1
    second = tmp_path / 'second.npz'
    dem = np.zeros((1, 512), np.uint8)
    dem[0, 2] = 1
    np.savez(second, hdr=np.array([[0, 1, 1, 0]]), dem=dem)
    assert builder.build_hot_set([trace, second])['layers'][0][:3] == [2, 7, 3]


def test_reject_duplicate_and_bad_layer(tmp_path):
    path = tmp_path / 'trace.npz'
    np.savez(path, hdr=np.array([[48, 1, 1, 0]]), dem=np.zeros((1, 512), np.uint8))
    with pytest.raises(ValueError, match='layer'):
        builder.build_hot_set([path])
    np.savez(path, hdr=np.array([[0, 1, 1, 0]]), dem=np.zeros((1, 512), np.uint8))
    with pytest.raises(ValueError, match='duplicate'):
        builder.build_hot_set([path, path])


def test_actual_recorder_excludes_pruned_routes(tmp_path, monkeypatch):
    import types
    torch = pytest.importorskip('torch')
    tier = ROOT / 'python/sglang/srt/layers/moe/expert_tier'
    package = types.ModuleType('hot_set_trace_test')
    package.__path__ = [str(tier)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ('predict', 'trace'):
        full = package.__name__ + '.' + name
        spec = importlib.util.spec_from_file_location(full, tier / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
    recorder = module.Recorder(tmp_path, 4, 'cpu')
    recorder.record(0, torch.tensor([[7, 7, -1, 4], [2, 3, 4, 8]]), None, None,
                    torch.tensor([[1., .2, 1., 0.], [1., 1., 0., 0.]]))
    # Flush the real NPZ schema using minimal CPU state buffers for all layers.
    state = types.SimpleNamespace(**{name: torch.zeros(1, dtype=torch.int64)
                                    for name in module.STATE})
    recorder.flush([state] * 48)
    path = tmp_path / 'trace-000000.npz'
    counts = builder.count_trace(path)
    assert counts[0, 7] == 2 and counts[0, 2] == counts[0, 3] == 1
    assert counts.sum() == 4
    assert builder.build_hot_set([path])['layers'][0][:4] == [7, 2, 3, 0]
