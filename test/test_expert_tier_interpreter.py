# SPDX-License-Identifier: Apache-2.0
"""Small CPU interpreter checks; no CUDA context or model imports."""
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TIER = ROOT / 'python/sglang/srt/layers/moe/expert_tier'


@pytest.fixture
def modules(monkeypatch):
    if os.environ.get('TRITON_INTERPRET') != '1':
        pytest.skip('run with TRITON_INTERPRET=1 for CPU kernels')
    torch = pytest.importorskip('torch')
    pytest.importorskip('triton')
    package = types.ModuleType('cpu_expert_tier')
    package.__path__ = [str(TIER)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    loaded = []
    for name in ('reference', 'state', 'kernels'):
        full = package.__name__ + '.' + name
        spec = importlib.util.spec_from_file_location(full, TIER / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return (torch, *loaded)


def test_decide_matches_reference(modules):
    torch, reference, state_module, kernels = modules
    g1, g2 = np.arange(512, dtype=np.float32), -np.arange(512, dtype=np.float32)
    ref = reference.SlotReference(4, [1, 3], g1, g2)
    state = state_module.TierState(4, [1, 3], torch.from_numpy(g1), torch.from_numpy(g2))
    for route in ([[1, 2, -1, 2]], [[3, 4, 5, -1]], [[4, 1, 5, 1]],
                  [[-1, -1, -1, -1]], [[0, 1, 2, 3, 4]]):
        ids = np.array(route, np.int32)
        expected, copies = ref.decide(ids)
        tensor = torch.from_numpy(ids)
        out = torch.empty_like(tensor)
        kernels.decide(tensor, state, out, variant='sort')
        np.testing.assert_array_equal(out.numpy(), expected)
        np.testing.assert_array_equal(state.copy_list[:state.n_copy.item()].numpy(), copies)
        for name in ('slot_of', 'expert_of', 'key', 'stats', 'g1_slot', 'g2_slot'):
            np.testing.assert_array_equal(getattr(state, name).numpy(), getattr(ref, name))
        assert state.step.item() == ref.step
        assert bool(state.overflow.item()) == ref.overflow


def test_gather_matches_numpy_layout(modules):
    torch, _, _, kernels = modules
    # One expert payload, with source offsets aliased by expert to keep memory small.
    sizes = [819200] * 3 + [102400] * 3
    starts = np.cumsum([0] + sizes[:-1])
    offsets = np.tile(starts, (512, 1)).astype(np.int64)
    base = torch.arange(sum(sizes) // 4, dtype=torch.int32)
    slots = tuple(torch.full((size,), -1, dtype=torch.int32)
                  for size in (409600, 204800, 51200, 25600))
    copies = torch.tensor([[0, 7]], dtype=torch.int32)
    kernels.gather(base.view(torch.uint8), torch.from_numpy(offsets), 0, copies,
                   torch.tensor([1], dtype=torch.int32), slots)
    destinations = [(0, 0), (0, 204800), (1, 0), (2, 0), (2, 25600), (3, 0)]
    for part, idx in enumerate((2, 1, 0, 5, 4, 3)):
        count = sizes[idx] // 4
        local = np.arange(count)
        src = local
        if part >= 3:
            n = 10 if part == 5 else 40
            src = ((local // (128*n))*128 + (local % 4)*32 + (local // 4) % 32)*n + (local // 128) % n
        expected = base.numpy()[starts[idx] // 4 + src]
        dest, start = destinations[part]
        np.testing.assert_array_equal(slots[dest].numpy()[start:start+count], expected)
