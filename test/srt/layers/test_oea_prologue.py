# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors

import ast
from contextlib import ExitStack
import importlib
import inspect
import json
import os
import re
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.environ import envs
from sglang.srt.layers.moe import oea_prologue as module
from sglang.srt.model_executor.forward_context import forward_batch_context

ROOT = Path(__file__).resolve().parents[3]
# The assembled FlashInfer can sit beside python/ or beside the release directory.
# OEA_SD_FLASHINFER overrides these candidates.
FI_CANDIDATES = [ROOT / 'flashinfer-vtoea', ROOT.parent / 'flashinfer-vtoea']


@pytest.fixture
def mod():
    with ExitStack() as stack:
        for field, value in ((envs.SGLANG_OEA_K0, 6), (envs.SGLANG_OEA_MODE, 'oea'),
                             (envs.SGLANG_OEA_STATS, True), (envs.SGLANG_OEA_CTL_FILE, ''),
                             (envs.SGLANG_OEA_MIN_ROWS, 16), (envs.SGLANG_OEA_WIDTH, 16),
                             (envs.SGLANG_OEA_MAX_BS, 1)):
            stack.enter_context(field.override(value))
        stack.enter_context(patch.dict('os.environ', FLASHINFER_MOE_PRUNE_SINGLETON_TAU='0.08',
                                       FLASHINFER_MOE_PRUNE_MIN_RANK='1',
                                       FLASHINFER_MOE_PRUNE_MIN_ROWS='2'))
        yield importlib.reload(module)
    importlib.reload(module)


def batch(mode='verify', width=16, bs=1):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: mode == 'verify'),
                           spec_info=SimpleNamespace(draft_token_num=width), batch_size=bs)


@pytest.mark.parametrize('value', [None, '', '0'])
def test_disarmed_allocates_nothing(value):
    with envs.SGLANG_OEA_K0.override(value), envs.SGLANG_OEA_STATS.override(True):
        with warnings.catch_warnings(record=True) as records:
            m = importlib.reload(module)
        assert not records
        with patch('torch.tensor', side_effect=AssertionError('allocation')):
            with m.target_forward_context(batch()):
                assert m.call_kwargs(16) == {}
        assert not m._STATE
    importlib.reload(module)


@pytest.mark.parametrize('mode,width,bs,rows,eligible', [
    ('verify', 16, 1, 16, True), ('verify', 8, 1, 16, False),
    ('verify', 4, 1, 16, False), ('verify', 16, 2, 32, False),
    ('verify', 16, 1, 15, True), ('verify', 16, 1, 2, True), ('verify', 16, 1, 1, False),
    ('draft_extend', 16, 1, 16, False),
    ('draft_decode', 16, 1, 16, False), ('prefill', 16, 1, 16, False),
    ('extend', 16, 1, 16, False),
])
def test_eligibility(mod, mode, width, bs, rows, eligible):
    with mod.target_forward_context(batch(mode, width, bs)):
        kw = mod.call_kwargs(rows, device='cpu')
        assert ('oea_k0' in kw) == eligible
        assert kw['oea_stats'].shape == (8,)
        if eligible:
            assert kw['oea_k0'].dtype == torch.int32
            assert kw['oea_k0'].shape == (2,)
            assert kw['oea_k0'].tolist() == [6, 16]
    assert set(mod.call_kwargs(16, device='cpu')) == {'oea_stats'}


def test_no_stats_and_import_configuration(mod):
    with envs.SGLANG_OEA_STATS.override(False), envs.SGLANG_OEA_MODE.override('prod'):
        m = importlib.reload(module)
        with m.target_forward_context(batch()):
            kw = m.call_kwargs(16, device='cpu')
            assert set(kw) == {'oea_k0'}
            assert kw['oea_k0'].tolist() == [0, 16]
        assert m._STATE[torch.device('cpu')]['stats'].dtype == torch.float64
        assert not any(m.read_stats().values())
        assert m.call_kwargs(16, device='cpu') == {}
    with envs.SGLANG_OEA_K0.override(10):
        with pytest.raises(ValueError):
            importlib.reload(module)
    importlib.reload(module)


def test_first_capture_does_not_allocate(mod, caplog):
    with patch('torch.cuda.is_current_stream_capturing', return_value=True), \
         patch('torch.tensor', side_effect=AssertionError('allocation')):
        with mod.target_forward_context(batch()):
            assert mod.call_kwargs(16, device='cuda:0') == {}
            assert mod.call_kwargs(16, device='cuda:0') == {}
    assert not mod._STATE
    assert len(caplog.records) == 1


def fake_target_class():
    source = (ROOT / 'python/sglang/srt/models/qwen4_exp.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'Qwen4ExpForConditionalGeneration')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    forward.decorator_list = []
    fake = ast.ClassDef(name=cls.name, bases=[ast.Name(id='Base', ctx=ast.Load())],
                        keywords=[], body=[forward], decorator_list=[])
    return ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[]))


def test_actual_target_hook_and_exception(mod):
    class Base:
        def forward(self, input_ids, positions, forward_batch, **kwargs):
            assert mod._CONTEXT.get() == (True, forward_batch.spec_info.draft_token_num, 1)
            assert ('oea_k0' in mod.call_kwargs(16, device='cpu')) == (forward_batch.spec_info.draft_token_num == 16)
            if kwargs.get('fail'):
                raise RuntimeError('forward failed')
            return object()
    scope = dict(Base=Base, LogitsProcessorOutput=type('LogitsProcessorOutput', (), {}),
                 forward_batch_context=forward_batch_context, _RND_PLE_TEST=None)
    exec(compile(fake_target_class(), 'target-hook', 'exec'), scope)
    target = scope['Qwen4ExpForConditionalGeneration']()
    target.model = SimpleNamespace(last_hc_hidden_states=None)
    for width in (4, 8, 16):
        target.forward(None, None, batch(width=width))
        assert mod._CONTEXT.get() is None
    with pytest.raises(RuntimeError):
        target.forward(None, None, batch(), fail=True)
    assert mod._CONTEXT.get() is None
    draft = (ROOT / 'python/sglang/srt/models/qwen4_exp_mtp.py').read_text()
    assert 'target_forward_context' not in draft
    assert 'class Qwen4ExpForCausalLMMTP(Qwen3_5ForCausalLMMTP)' in draft


LINE = ('seq={seq} mode={mode} k0=6 tau=0.0 renorm=0 min_rows=16 fallback=1 '
        'prod_tau=0.08 prod_min_rank=1 prod_min_rows=2')


def test_control_status(mod):
    dev = torch.device('cpu')
    st = mod._state(dev)
    for seq, mode, value in ((1, 'prod', 0), (2, 'oea', 6)):
        text = LINE.format(seq=seq, mode=mode)
        rec = mod.apply_control(text, dev, st)
        rec = json.loads(json.dumps(rec))
        assert rec['seq'] == seq and rec['cfg']['mode'] == mode
        assert st['cell'].tolist() == [value, 16]
        assert set(rec['stats']) == set(mod.STAT_NAMES)
        assert mod.apply_control(text, dev, st) is None


@pytest.mark.parametrize('field,value', [
    ('mode', 'off'), ('mode', 'p_oea'), ('mode', 'oea_p'), ('k0', '0'), ('k0', '10'),
    ('tau', '0.01'), ('renorm', '1'), ('fallback', '0'), ('min_rows', '1'), ('min_rows', '0'),
    ('min_rows', '17'), ('min_rows', '-2'), ('min_rows', '8.0'),
    ('prod_tau', '0'), ('prod_min_rank', '2'), ('prod_min_rows', '3'), ('unknown', '1'),
])
def test_control_rejects_without_change(mod, field, value):
    dev = torch.device('cpu')
    st = mod._state(dev)
    old = dict(st['cfg_host'])
    rec = mod.apply_control(f'seq=1 {field}={value}', dev, st)
    assert 'error' in rec
    assert st['cfg_host'] == old and st['cell'].tolist() == [6, 16]


def test_effective_production_env(mod):
    with patch.dict('os.environ', FLASHINFER_MOE_PRUNE_SINGLETON_TAU='0.125',
                    FLASHINFER_MOE_PRUNE_MIN_RANK='3', FLASHINFER_MOE_PRUNE_MIN_ROWS='4'):
        st = mod._state(torch.device('cpu'))
    assert st['cfg_host']['prod_tau'] == 0.125
    assert st['cfg_host']['prod_min_rank'] == 3
    assert st['cfg_host']['prod_min_rows'] == 4
    rec = mod.apply_control(LINE.format(seq=1, mode='prod'), torch.device('cpu'), st)
    assert 'error' in rec and rec['cfg']['prod_tau'] == 0.125


def test_adaptive_capture_metadata_contract(mod):
    runner = (ROOT / 'python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py').read_text()
    adaptive = (ROOT / 'python/sglang/srt/speculative/eagle_worker_v2.py').read_text()
    assert 'self.capture_forward_mode = ForwardMode.TARGET_VERIFY' in runner
    assert 'forward_mode=self.capture_forward_mode' in runner
    assert 'spec_info=spec_info' in runner
    assert 'speculative_num_draft_tokens=speculative_num_draft_tokens' in adaptive
    assert 'target_model_runner,' in adaptive
    tree = ast.parse(runner)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DecodeCudaGraphRunner')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_spec_info')
    scope = dict(torch=torch, get_spec=lambda: SimpleNamespace(speculative_eagle_topk=1),
                 CaptureHiddenMode=SimpleNamespace(NULL=0, FULL=1))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 'capture-spec-info', 'exec'), scope)
    info_module = SimpleNamespace(EagleVerifyInput=lambda **kw: SimpleNamespace(**kw))
    model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True, is_standalone=lambda: False),
        is_draft_worker=False, model_config=SimpleNamespace(hidden_size=2),
        dtype=torch.float32, device='cpu')
    with patch.dict('sys.modules', {'sglang.srt.speculative.eagle_info': info_module}):
        for width in (4, 8, 16):
            fake_runner = SimpleNamespace(model_runner=model_runner,
                                          buffers=SimpleNamespace(custom_mask=None),
                                          speculative_num_steps=width - 1,
                                          speculative_num_draft_tokens=width)
            info = scope['get_spec_info'](fake_runner, width)
            assert info.draft_token_num == width
            forward_batch = batch(width=width)
            forward_batch.spec_info = info
            with mod.target_forward_context(forward_batch):
                assert ('oea_k0' in mod.call_kwargs(width, device='cpu')) == (width == 16)
            assert mod._CONTEXT.get() is None


def test_existing_state_reused_during_capture(mod):
    cpu = torch.device('cpu')
    st = mod._state(cpu)
    mod._STATE[torch.device('cuda:0')] = st
    with patch('torch.cuda.is_current_stream_capturing', return_value=True), \
         patch('torch.tensor', side_effect=AssertionError('capture allocation')):
        with mod.target_forward_context(batch()):
            kw = mod.call_kwargs(16, device='cuda:0')
            assert kw['oea_k0'] is st['cell'] and kw['oea_stats'] is st['stats']
    mod.apply_control('seq=9 mode=prod', cpu, st)
    assert kw['oea_k0'].tolist() == [0, 16]
    mod.apply_control('seq=10 mode=oea min_rows=2', cpu, st)
    assert kw['oea_k0'].tolist() == [6, 2]


@pytest.mark.parametrize('field,value', [
    (envs.SGLANG_OEA_MODE, 'off'), (envs.SGLANG_OEA_K0, -1),
    (envs.SGLANG_OEA_WIDTH, 0), (envs.SGLANG_OEA_MAX_BS, 0),
    (envs.SGLANG_OEA_MIN_ROWS, 1), (envs.SGLANG_OEA_MIN_ROWS, 17),
])
def test_invalid_import_config(mod, field, value):
    with field.override(value):
        with pytest.raises(ValueError):
            importlib.reload(module)
    importlib.reload(module)


def test_rejected_sequence_has_only_one_status(mod):
    dev = torch.device('cpu')
    st = mod._state(dev)
    rec = mod.apply_control('seq=1 unknown=1', dev, st)
    assert rec['seq'] == 1 and 'error' in rec
    assert mod.apply_control('seq=1 mode=prod', dev, st) is None
    assert st['cell'].tolist() == [6, 16]
    assert mod.apply_control('seq=2 mode=prod', dev, st)['cfg']['mode'] == 'prod'


def flashinfer_core():
    import os
    override = os.environ.get('OEA_SD_FLASHINFER')
    roots = [Path(override)] if override else FI_CANDIDATES
    for root in roots:
        core = root / 'flashinfer/fused_moe/core.py'
        if core.is_file():
            return core
    pytest.fail(f'no flashinfer-vtoea core.py under {roots}')


def test_flashinfer_api_threads_both_optional_pointers():
    tree = ast.parse(flashinfer_core().read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name in {'cutlass_fused_moe', '_fake_cutlass_fused_moe'}]
    assert len(functions) == 3
    for function in functions:
        defaults = dict(zip([arg.arg for arg in function.args.args[-len(function.args.defaults):]],
                            function.args.defaults))
        for name in ('vt_lens', 'oea_k0', 'oea_stats'):
            assert isinstance(defaults[name], ast.Constant) and defaults[name].value is None
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'run_moe']
    assert len(calls) == 1
    # VT's optional vt_lens goes before the two OEA pointers (run_moe_vt takes vt_lens, oea_k0, oea_stats).
    assert [ast.unparse(arg) for arg in calls[0].args[-3:]] == [
        '*((vt_lens,) if vt_lens is not None else ())', 'oea_k0', 'oea_stats']


@pytest.mark.parametrize('rows', [16, 8, 2])
def test_control_switches_min_rows(mod, rows):
    dev = torch.device('cpu')
    st = mod._state(dev)
    rec = mod.apply_control(f'seq=1 min_rows={rows}', dev, st)
    assert 'error' not in rec and rec['cfg']['min_rows'] == rows
    assert st['cell'].tolist() == [6, rows]
    rec = mod.apply_control(LINE.replace('min_rows=16', f'min_rows={rows}').format(seq=2, mode='prod'), dev, st)
    assert 'error' not in rec and st['cell'].tolist() == [0, rows]
    rec = mod.apply_control('seq=3 mode=oea k0=4', dev, st)
    assert st['cell'].tolist() == [4, rows] and rec['cfg']['min_rows'] == rows


@pytest.mark.parametrize('rows', [2, 8, 16])
def test_import_min_rows_is_the_start_value(rows):
    with envs.SGLANG_OEA_K0.override(6), envs.SGLANG_OEA_MODE.override('oea'), \
         envs.SGLANG_OEA_CTL_FILE.override(''), envs.SGLANG_OEA_STATS.override(False), \
         envs.SGLANG_OEA_MIN_ROWS.override(rows):
        m = importlib.reload(module)
        st = m._state(torch.device('cpu'))
        assert st['cell'].tolist() == [6, rows] and st['cfg_host']['min_rows'] == rows
    importlib.reload(module)


def test_ple_test_switch_runs_inside_the_context(mod):
    seen = []

    class Base:
        def forward(self, input_ids, positions, forward_batch, **kwargs):
            seen.append(mod._CONTEXT.get())
            return object()

    class PleTest:
        def run(self, fn, forward_batch):
            seen.append(('run', mod._CONTEXT.get()))
            return fn()

    scope = dict(Base=Base, LogitsProcessorOutput=type('LogitsProcessorOutput', (), {}),
                 forward_batch_context=forward_batch_context, _RND_PLE_TEST=PleTest())
    module_ast = fake_target_class()
    cls = module_ast.body[0]
    source = (ROOT / 'python/sglang/srt/models/qwen4_exp.py').read_text()
    real = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                and n.name == 'Qwen4ExpForConditionalGeneration')
    cls.body.append(next(n for n in real.body if isinstance(n, ast.FunctionDef) and n.name == '_rnd_forward_once'))
    exec(compile(ast.fix_missing_locations(module_ast), 'target-hook-ple', 'exec'), scope)
    target = scope['Qwen4ExpForConditionalGeneration']()
    target.model = SimpleNamespace(last_hc_hidden_states=None)
    target.forward(None, None, batch())
    assert seen == [('run', (True, 16, 1)), (True, 16, 1)]
    assert mod._CONTEXT.get() is None


def test_forward_without_batch_passes_through(mod):
    with mod.target_forward_context(None):
        assert mod._CONTEXT.get() is None
        assert set(mod.call_kwargs(16, device='cpu')) == {'oea_stats'}


# -- REVIEW-1 S1: both words of the cell change with one store -------------------------------------------------------

class CellWrites(TorchDispatchMode):
    """Every aten op that writes into the cell's storage, as (op, dtype, numel)."""

    def __init__(self, cell):
        super().__init__()
        self.ptr = cell.untyped_storage().data_ptr()
        self.writes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        schema = func._schema.arguments
        values = list(args) + [kwargs.get(arg.name) for arg in schema[len(args):]]
        for arg, value in zip(schema, values):
            if (arg.alias_info is not None and arg.alias_info.is_write and isinstance(value, torch.Tensor)
                    and value.untyped_storage().data_ptr() == self.ptr):
                self.writes.append((func.overloadpacket.__name__, value.dtype, value.numel()))
        return func(*args, **kwargs)


def test_cell_changes_with_one_store(mod):
    cpu = torch.device('cpu')
    st = mod._state(cpu)
    for seq, line, cell in ((1, 'min_rows=8', [6, 8]), (2, 'mode=prod', [0, 8]), (3, 'mode=oea k0=4 min_rows=2', [4, 2]),
                            (4, 'k0=9 min_rows=12', [9, 12]), (5, 'mode=oea k0=6 min_rows=16', [6, 16])):
        with CellWrites(st['cell']) as spy:
            rec = mod.apply_control(f'seq={seq} {line}', cpu, st)
        assert 'error' not in rec and st['cell'].tolist() == cell, line
        assert spy.writes == [('fill_', torch.int64, 1)], (line, spy.writes)
    with CellWrites(st['cell']) as spy:
        assert 'error' in mod.apply_control('seq=6 k0=10', cpu, st)
        assert mod.apply_control('seq=6 k0=4', cpu, st) is None  # same seq: ignored
    assert spy.writes == [] and st['cell'].tolist() == [6, 16]


def test_cuda_branch_writes_the_cell_the_same_way():
    assert '["cell"][' not in inspect.getsource(module)
    body = inspect.getsource(module.apply_control)
    assert body.count('_write_cell(st["cell"], cfg)') == 2 and body.count('fill_') == 0
    cell = torch.zeros(2, dtype=torch.int32)
    for cfg in (dict(mode='oea', k0=6, min_rows=16), dict(mode='prod', k0=6, min_rows=2), dict(mode='oea', k0=1, min_rows=9)):
        module._write_cell(cell, cfg)
        assert cell.tolist() == module._cell_values(cfg)


# -- REVIEW-1 S2 / REVIEW-3 B1: the control thread reads nothing until the scheduler has captured every graph ---------

def test_start_order_runs_the_autotune_outside_capture_mode():
    """Why call_kwargs cannot tell (REVIEW-3 B1): the start's one FlashInfer autotune runs in EagerRunner.__init__,
    before any graph runner and outside model_capture_mode, as TARGET_VERIFY target forwards; the decode runner's
    capture() then finds the warmup done. So the flag is set by the scheduler after Scheduler(...) instead."""
    from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as cgs
    from sglang.srt.model_executor.runner import base_runner, decode_cuda_graph_runner, eager_runner

    src = inspect.getsource(cgs.capture_cuda_graphs)
    assert src.index('EagerRunner(model_runner)') < src.index('capture_prefill_graph(') < src.index(
        'capture_decode_graph(')
    init = inspect.getsource(eager_runner.EagerRunner.__init__)
    assert 'self.warmup()' in init and 'model_capture_mode' not in init
    warm = inspect.getsource(base_runner.BaseRunner.warmup)
    assert '_kernel_warmed_up' in warm and 'self._flashinfer_autotune(' in warm
    assert 'capture_forward_mode = ForwardMode.TARGET_VERIFY' in inspect.getsource(base_runner.BaseRunner._dummy_run)
    dec = inspect.getsource(decode_cuda_graph_runner.DecodeCudaGraphRunner.__init__)
    assert 'with model_capture_mode():\n                self.capture()' in dec


def test_scheduler_marks_serving_after_init():
    """mark_serving() runs once Scheduler(...) (model load, autotune, every graph capture) has returned and before
    the parent hears that the scheduler is ready, inside the same try (a failure still reaches the parent)."""
    source = (ROOT / 'python/sglang/srt/managers/scheduler.py').read_text()
    run = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_scheduler_process']
    assert len(run) == 1 and len(re.findall(r'\bmark_serving\b', source)) == 2  # the import and the one call
    body = next(n for n in ast.walk(run[0]) if isinstance(n, ast.Try)).body
    text = [ast.unparse(n) for n in body]
    created = next(i for i, t in enumerate(text) if t.startswith('scheduler = Scheduler('))
    marked = text.index('mark_serving()')
    sent = text.index('pipe_writer.send(scheduler.get_init_info())')
    assert created < marked < sent
    assert text[marked - 1] == 'from sglang.srt.layers.moe.oea_prologue import mark_serving'
    assert sum(t == 'mark_serving()' for t in text) == 1


@pytest.fixture
def ctl_mod(tmp_path):
    ctl = tmp_path / 'oea/flash-next.ctl'
    ctl.parent.mkdir()
    ctl.write_text('')
    with ExitStack() as stack:
        for field, value in ((envs.SGLANG_OEA_K0, 6), (envs.SGLANG_OEA_MODE, 'oea'),
                             (envs.SGLANG_OEA_STATS, True), (envs.SGLANG_OEA_CTL_FILE, str(ctl)),
                             (envs.SGLANG_OEA_MIN_ROWS, 16), (envs.SGLANG_OEA_WIDTH, 16),
                             (envs.SGLANG_OEA_MAX_BS, 1)):
            stack.enter_context(field.override(value))
        stack.enter_context(patch.dict('os.environ', FLASHINFER_MOE_PRUNE_SINGLETON_TAU='0.08'))
        yield importlib.reload(module), ctl
        ctl.write_text('')  # park the thread on a file that does not change again
    importlib.reload(module)


def status_records(ctl, at_least, timeout=5.0):
    path = Path(str(ctl) + '.status.jsonl')
    deadline = time.time() + timeout
    while True:
        recs = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if len(recs) >= at_least:
            return recs
        assert time.time() < deadline, recs
        time.sleep(0.05)


def test_autotune_forward_does_not_open_the_control_file(ctl_mod):
    """The autotune's TARGET_VERIFY forward runs outside capture mode, then a
    switch written while the graphs are captured. Nothing is read until mark_serving(); then the switch applies."""
    from sglang.srt.model_executor.runner_utils import capture_mode

    m, ctl = ctl_mod
    cpu, cuda = torch.device('cpu'), torch.device('cuda:0')
    st = m._state(cpu)  # the first MoE call (the autotune) allocates the state; the thread writes the start line
    m._STATE[cuda] = st
    assert status_records(ctl, 1)[0]['start'] is True
    autotune = SimpleNamespace(forward_mode=SimpleNamespace(is_target_verify=lambda: True),
                               spec_info=SimpleNamespace(draft_token_num=16), batch_size=1)
    assert capture_mode.get_is_capture_mode() is False
    with patch('torch.cuda.is_current_stream_capturing', return_value=False):
        for forward in (autotune, batch('extend'), batch()):
            with m.target_forward_context(forward):
                m.call_kwargs(16, device=cuda)
                m.call_kwargs(16, device=cpu)
    with patch.object(capture_mode, 'is_capture_mode', True):  # the target / draft / adaptive captures
        ctl.write_text('seq=1 min_rows=8\n')
        time.sleep(1.0)  # four polls of a thread that would read
        assert len(status_records(ctl, 1)) == 1 and st['cell'].tolist() == [6, 16]
    assert not m._SERVING.is_set()
    m.mark_serving()  # run_scheduler_process, after Scheduler(...)
    recs = status_records(ctl, 3)
    assert [(r.get('start'), r.get('serving'), r.get('seq')) for r in recs] == [
        (True, None, None), (None, True, None), (None, None, 1)]
    assert recs[2]['cfg']['min_rows'] == 8 and st['cell'].tolist() == [6, 8]


def test_control_thread_reads_nothing_before_serving(ctl_mod, monkeypatch):
    m, ctl = ctl_mod
    opened = []

    def spy(path, *args, **kwargs):
        opened.append((str(path), args[0] if args else kwargs.get('mode', 'r')))
        return open(path, *args, **kwargs)

    monkeypatch.setattr(m, 'open', spy, raising=False)  # the thread looks `open` up in the module first
    cpu = torch.device('cpu')
    st = m._state(cpu)
    assert status_records(ctl, 1)[0]['start'] is True
    ctl.write_text('seq=1 min_rows=8\n')  # typed while the start still captures graphs
    for forward in (batch(), batch('extend'), None):  # no forward, even a prefill outside capture, opens it
        with m.target_forward_context(forward):
            m.call_kwargs(16, device=cpu)
    time.sleep(1.0)
    assert len(status_records(ctl, 1)) == 1 and st['cell'].tolist() == [6, 16]
    assert (str(ctl), 'r') not in opened
    m.mark_serving()
    recs = status_records(ctl, 3)
    assert [(r.get('start'), r.get('serving'), r.get('seq')) for r in recs] == [
        (True, None, None), (None, True, None), (None, None, 1)]
    assert recs[1]['pid'] == os.getpid() and recs[1]['cfg'] == recs[0]['cfg'] and 'stats' not in recs[1]
    assert recs[2]['cfg']['min_rows'] == 8 and st['cell'].tolist() == [6, 8]
    assert (str(ctl), 'r') in opened
    m.mark_serving()  # idempotent: one serving line per thread
    time.sleep(0.3)
    assert len(status_records(ctl, 3)) == 3


def test_a_state_made_after_serving_reads_at_once(ctl_mod):
    m, ctl = ctl_mod
    m.mark_serving()
    m._state(torch.device('cpu'))
    recs = status_records(ctl, 2)
    assert recs[0].get('start') and recs[1].get('serving')
