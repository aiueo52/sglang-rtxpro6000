# SPDX-License-Identifier: Apache-2.0
"""Small SGLang hooks; all SGLang imports are local to integration functions."""
import atexit
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch

from . import kernels, predict, trace
from .source import SourceIndex, load_router_weights
from .state import SLOT_BYTES, TierState, allocate_slots

_PREFIX = re.compile(r'^(?:.*\.)?layers\.(\d+)\.mlp\.experts$')
_BIG = ('w13_weight', 'w2_weight', 'w13_weight_scale', 'w2_weight_scale')
_COUNTERS = ('calls', 'demand', 'hits', 'misses', 'overflows', 'arena_calls',
             'arena_source', 'arena_slots')
_RUNTIME = None


def conditional_nodes_available():
    return all(callable(getattr(torch.cuda.CUDAGraph, name, None)) for name in (
        'get_currently_capturing_graph', 'begin_capture_to_if_node',
        'end_capture_to_conditional_node'))


@contextmanager
def _if_copies(state):
    """Capture the gather body in an IF node; caller must be capturing."""
    torch.gt(state.n_copy.view(()), 0, out=state.has_copy)
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    graph.begin_capture_to_if_node(state.has_copy)
    try:
        yield
    finally:
        graph.end_capture_to_conditional_node()


def keeper_period():
    raw = os.environ.get('SGLANG_EXPERT_TIER_KEEP', '0')
    try:
        period = float(raw)
    except ValueError:
        raise ValueError('SGLANG_EXPERT_TIER_KEEP must be 0 or seconds in 5..3600') from None
    if not math.isfinite(period) or (period != 0 and not 5 <= period <= 3600):
        raise ValueError('SGLANG_EXPERT_TIER_KEEP must be 0 or seconds in 5..3600')
    return period


def stop_keeper(process):
    """Bound shutdown time, also handling a keeper that already exited."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def configured_k():
    k = int(os.environ.get('SGLANG_EXPERT_TIER_K', '0'))
    if not 0 <= k <= 512:
        raise ValueError('SGLANG_EXPERT_TIER_K must be in 0..512')
    return k


def layer_number(layer):
    name = getattr(layer, 'layer_name', '')
    match = _PREFIX.fullmatch(name)
    if name.startswith('mtp') or '.mtp.' in name or match is None:
        return None
    return int(match[1])


def is_tier_layer(layer):
    return configured_k() > 0 and layer_number(layer) is not None


def hot_experts(layer):
    k = configured_k()
    path = os.environ.get('SGLANG_EXPERT_TIER_HOT')
    if path:
        data = json.loads(Path(path).read_text())
        if data['K'] < k or len(data['layers']) != 48:
            raise ValueError('hot initialization requires 48 layers and at least K entries')
        hot = data['layers'][layer_number(layer)][:k]
    else:
        hot = list(range(k))
    if len(hot) != k or len(set(hot)) != k or any(type(e) is not int or not 0 <= e < 512 for e in hot):
        raise ValueError('invalid expert tier hot set')
    return hot


def loader_expert(layer, param, expert_id):
    """None skips a cold large parameter; scalar expert indices stay global."""
    if not is_tier_layer(layer) or not any(param is getattr(layer, n, None) for n in _BIG):
        return expert_id
    if not hasattr(layer, '_expert_tier_hot_slots'):
        layer._expert_tier_hot_slots = {e:s for s,e in enumerate(hot_experts(layer))}
    return layer._expert_tier_hot_slots.get(expert_id)


def validate_layer(layer, *, backend=None, method_type=None, in_prologue=None, armed=None):
    if backend is None:
        from sglang.srt.layers.moe.utils import get_moe_runner_backend
        backend = get_moe_runner_backend()
    if method_type is None:
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
        method_type = ModelOptNvFp4FusedMoEMethod
    if in_prologue is None:
        from sglang.srt.layers.moe.prune_singleton import IN_PROLOGUE
        in_prologue = IN_PROLOGUE
    if armed is None:
        from sglang.srt.layers.moe.oea_prologue import ARMED
        armed = ARMED
    checks = (
        (backend.is_flashinfer_cutlass(), 'backend must be flashinfer_cutlass'),
        (isinstance(layer.quant_method, method_type), 'quant method must be ModelOptNvFp4FusedMoEMethod'),
        (layer.moe_tp_size == layer.moe_ep_size == 1, 'moe_tp_size and moe_ep_size must be 1'),
        (not in_prologue, 'P1 required: prune_singleton.IN_PROLOGUE must be false'),
        (not armed, 'oea_prologue.ARMED must be false'),
        (getattr(layer, 'num_fused_shared_experts', 0) == 0, 'fused shared experts are unsupported'),
        (getattr(layer.quant_method, 'draft_moe_gemv', None) is None, 'draft_moe_gemv must be None'),
        (not getattr(layer, 'inference_moe_w13_interleaved', False), 'interleaved w13 is unsupported'),
        (os.environ.get('SGLANG_EXPERT_TIER_SOURCE', 'mmap') == 'mmap', 'integration requires mmap source; pinned is benchmark-only'),
    )
    for ok, message in checks:
        if not ok:
            raise ValueError('expert tier: ' + message)


class Runtime:
    """Shared source and arenas; optional prefill copy stream, no worker threads."""
    def __init__(self, source, base, offsets, arena, *, k, variant=None):
        if variant is None:
            variant = os.environ.get('SGLANG_EXPERT_TIER_DECIDE', 'matrix')
        self.keeper_period = keeper_period()
        self.keeper = None
        timing = os.environ.get('SGLANG_EXPERT_TIER_TIMING', '0')
        if timing not in ('0', '1'):
            raise ValueError('SGLANG_EXPERT_TIER_TIMING must be 0 or 1')
        self.timing_enabled = timing == '1'
        if self.timing_enabled:
            self.timing_cap = 20000
            self.timing_ring = torch.empty((self.timing_cap, len(_COUNTERS)),
                                           dtype=torch.int64, device=base.device)
            self.timing_records = []
            self.timing_previous = {}
            self.timing_tail = None
            self.timing_startup = None
            self.timing_dropped = 0
            self.timing_buckets = {
                name: dict(calls=0, tokens=0, presence_ms=0., arena_ms=0., gap_ms=0.,
                           host_experts=0, vram_experts=0, host_GBps=None)
                for name in ('le256', 'le2048', 'gt2048')}
            self.timing_buckets['gt2048']['layers'] = {}
        if variant not in ('matrix', 'sort'):
            raise ValueError('SGLANG_EXPERT_TIER_DECIDE must be matrix or sort')
        self.source, self.base, self.offsets, self.arena = source, base, offsets, arena
        self.k, self.variant = k, variant
        self.predict_mode = os.environ.get('SGLANG_EXPERT_TIER_PREDICT', '0')
        if self.predict_mode not in ('0', 'stats'):
            raise ValueError('SGLANG_EXPERT_TIER_PREDICT must be 0 or stats')
        self.predict_warned = False
        trace_dir, trace_cap = trace.configuration()
        if self.predict_mode == 'stats' or trace_dir:
            self.router_weights = load_router_weights(source.directory, device=base.device)
        if trace_dir:
            self.trace = trace.Recorder(trace_dir, trace_cap, base.device)
        if self.predict_mode == 'stats':
            self.pred_rank = torch.full((48, 512), 127, dtype=torch.int32, device=base.device)
            self.predict_stats = torch.zeros((48, 12), dtype=torch.int64, device=base.device)
        self.gather_variant = os.environ.get('SGLANG_EXPERT_TIER_GATHER', 'v1')
        if self.gather_variant not in ('v1', 'v2'):
            raise ValueError('SGLANG_EXPERT_TIER_GATHER must be v1 or v2')
        def integer(name, default, low, high, power=False):
            raw = os.environ.get(name, str(default))
            try:
                value = int(raw)
            except ValueError:
                raise ValueError(f'{name} must be an integer in {low}..{high}') from None
            if not low <= value <= high or (power and value & (value-1)):
                raise ValueError(f'{name} must be {"a power of two " if power else ""}in {low}..{high}')
            return value
        self.gather_chunk = integer('SGLANG_EXPERT_TIER_GATHER_CHUNK', 16384, 4096, 65536, True)
        self.gather_ctas = integer('SGLANG_EXPERT_TIER_GATHER_CTAS', 376, 1, 4096)
        self.policy = os.environ.get('SGLANG_EXPERT_TIER_POLICY', 'lru')
        if self.policy not in ('lru', 'lfu'):
            raise ValueError('SGLANG_EXPERT_TIER_POLICY must be lru or lfu')
        self.lfu_half = integer('SGLANG_EXPERT_TIER_LFU_HALF', 256, 1, 65536)
        self.lfu_inc = integer('SGLANG_EXPERT_TIER_LFU_INC', 16, 1, 4096)
        if self.lfu_half * self.lfu_inc > 4096:
            raise ValueError('SGLANG_EXPERT_TIER_LFU_HALF * SGLANG_EXPERT_TIER_LFU_INC must be <= 4096')
        if self.policy == 'lfu' and variant != 'sort':
            raise ValueError('SGLANG_EXPERT_TIER_POLICY=lfu requires SGLANG_EXPERT_TIER_DECIDE=sort')
        fused = os.environ.get('SGLANG_EXPERT_TIER_FUSED', '0')
        if fused not in ('0', '1'):
            raise ValueError('SGLANG_EXPERT_TIER_FUSED must be 0 or 1')
        self.fused = fused == '1'
        skip_empty = os.environ.get('SGLANG_EXPERT_TIER_SKIP_EMPTY', '0')
        if skip_empty not in ('0', '1'):
            raise ValueError('SGLANG_EXPERT_TIER_SKIP_EMPTY must be 0 or 1')
        self.skip_empty = skip_empty == '1'
        if self.skip_empty and self.fused:
            raise ValueError('SGLANG_EXPERT_TIER_SKIP_EMPTY=1 requires FUSED=0')
        if self.skip_empty and not conditional_nodes_available():
            raise RuntimeError('SGLANG_EXPERT_TIER_SKIP_EMPTY=1 requires torch CUDA conditional node APIs')
        if self.fused and (variant != 'sort' or self.gather_variant != 'v2'):
            raise ValueError('SGLANG_EXPERT_TIER_FUSED=1 requires GATHER=v2 and DECIDE=sort')
        self.arena_variant = os.environ.get('SGLANG_EXPERT_TIER_ARENA', 'v1')
        if self.arena_variant not in ('v1', 'v2'):
            raise ValueError('SGLANG_EXPERT_TIER_ARENA must be v1 or v2')
        prefetch = os.environ.get('SGLANG_EXPERT_TIER_PREFETCH', '0')
        if prefetch not in ('0', '1'):
            raise ValueError('SGLANG_EXPERT_TIER_PREFETCH must be 0 or 1')
        self.prefetch_enabled = prefetch == '1'
        self.prefetch_min = integer('SGLANG_EXPERT_TIER_PREFETCH_MIN', 2048, 1, 1048576)
        self.prefetch_ctas = integer('SGLANG_EXPERT_TIER_PREFETCH_CTAS', 188, 1, 4096)
        if self.prefetch_enabled:
            if self.arena_variant != 'v2':
                raise ValueError('SGLANG_EXPERT_TIER_PREFETCH=1 requires SGLANG_EXPERT_TIER_ARENA=v2')
            self.arenas = (arena, allocate_slots(512, base.device))
            self.prefetch_stream = torch.cuda.Stream(device=base.device) if base.is_cuda else None
            self.prefetch_marks = torch.ones(512, dtype=torch.int32, device=base.device)
            self.prefetch_stats = torch.zeros(len(_COUNTERS), dtype=torch.int64, device=base.device)
            self.pending = None
            self.prefetch_issued = self.prefetch_used = self.prefetch_stale = 0
            self.extra_arena_bytes = sum(t.numel()*t.element_size() for t in self.arenas[1])
            if self.timing_enabled:
                self.prefetch_timing_records = []
                self.prefetch_timing_ring = torch.empty((self.timing_cap, 2), dtype=torch.int64,
                                                         device=base.device)
                self.prefetch_timing = dict(calls=0, ms=0., source_experts=0, GBps=None)
            logging.getLogger(__name__).info('expert tier prefetch extra_arena_bytes=%d',
                                           self.extra_arena_bytes)
        self.pretouch_enabled = integer('SGLANG_EXPERT_TIER_PRETOUCH', 1, 0, 1)
        self.pretouch_seconds = None
        experts = torch.arange(source.experts, dtype=torch.int32, device=base.device)
        self.pretouch_copies = torch.stack((experts, experts), dim=1)
        self.pretouch_n_copy = torch.tensor([source.experts], dtype=torch.int32, device=base.device)
        self.layers = {}
        self.checked = False
        self.started = time.monotonic()
        self.last_write = self.started
        self.writes = 0
        self.stats_path = os.environ.get('SGLANG_EXPERT_TIER_STATS')
        self.stats_every = float(os.environ.get('SGLANG_EXPERT_TIER_STATS_EVERY', '30'))
        if self.stats_every < 0:
            raise ValueError('SGLANG_EXPERT_TIER_STATS_EVERY must be nonnegative')
        if self.keeper_period:
            files = [str(Path(path).resolve()) for path in source.files]
            size = sum(Path(path).stat().st_size for path in files)
            command = [sys.executable, '-B', str(Path(__file__).with_name('keeper.py').resolve()),
                       '--period', str(self.keeper_period), '--parent', str(os.getpid())]
            if self.stats_path:
                command += ['--stats', self.stats_path + '.keeper.json']
            self.keeper = subprocess.Popen(command + files, stdin=subprocess.DEVNULL)
            atexit.register(stop_keeper, self.keeper)
            logging.getLogger(__name__).info(
                'expert tier keeper pid=%d period=%s files=%d bytes=%d',
                self.keeper.pid, self.keeper_period, len(files), size)

    def before_route(self, state, capturing):
        if not self.checked:
            if capturing:
                raise RuntimeError('expert tier requires an eager route before graph capture')
            if set(self.layers) != set(range(48)):
                raise ValueError('expert tier requires exactly layers 0..47 before first route')
            if self.pretouch_enabled:
                pretouch(self)
            if self.timing_enabled:
                self.measure_startup()
            self.checked = True
            logging.getLogger(__name__).info(
                'expert tier K=%d layers=%d slot_bytes=%d arena_bytes=%d source=mmap files=%d decide=%s '
                'gather=%s chunk=%d ctas=%d pretouch_seconds=%s',
                self.k, len(self.layers), 48*self.k*SLOT_BYTES, 512*SLOT_BYTES,
                len(self.source.files), self.variant, self.gather_variant, self.gather_chunk,
                self.gather_ctas, self.pretouch_seconds)
        if state.layer_number == 0 and not capturing and hasattr(self, 'trace'):
            self.trace.flush(self.layers)
        if state.layer_number == 0 and not capturing and self.stats_path:
            now = time.monotonic()
            if now - self.last_write >= self.stats_every:
                self.write_stats(now)

    def write_stats(self, now):
        if hasattr(self, 'trace'):
            self.trace.flush(self.layers)
        if self.timing_enabled:
            self.resolve_timing()
        rows = {str(l): dict(zip(_COUNTERS, s.stats.cpu().tolist())) for l,s in self.layers.items()}
        totals = {n:sum(row[n] for row in rows.values()) for n in _COUNTERS}
        data = dict(layers=rows, totals=totals, K=self.k, source='mmap', decide=self.variant,
                    uptime_seconds=now-self.started, writes=self.writes+1)
        data['policy'] = (dict(name='lfu', half=self.lfu_half, inc=self.lfu_inc)
                          if self.policy == 'lfu' else dict(name='lru'))
        if self.skip_empty:
            data['skip_empty'] = 1
        if hasattr(self, 'trace'):
            data['trace'] = self.trace.report()
        if self.predict_mode == 'stats':
            data['predict'] = predict.report(self.predict_stats)
        if self.keeper is not None:
            data['keeper'] = dict(pid=self.keeper.pid, period=self.keeper_period)
        if self.prefetch_enabled:
            # The copy stream owns these counters; an ordinary .cpu() on the
            # forward stream alone would not order its read after the copies.
            if self.pending is not None and self.pending[2] is not None:
                self.pending[2].synchronize()
            data['prefetch'] = dict(issued=self.prefetch_issued, used=self.prefetch_used,
                stale=self.prefetch_stale, min_tokens=self.prefetch_min, ctas=self.prefetch_ctas,
                extra_arena_bytes=self.extra_arena_bytes,
                stats=dict(zip(_COUNTERS, self.prefetch_stats.cpu().tolist())))
        if self.timing_enabled:
            data['timing'] = dict(startup=self.timing_startup, buckets=self.timing_buckets,
                                  dropped=self.timing_dropped)
            if self.prefetch_enabled:
                data['timing']['prefetch'] = self.prefetch_timing
        path = Path(self.stats_path)
        temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
        try:
            temporary.write_text(json.dumps(data, indent=2)+'\n')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.last_write, self.writes = now, self.writes+1

    def measure_startup(self):
        """Probe the existing copy paths, restoring all routing counters and marks."""
        def measured(operation):
            if self.base.is_cuda:
                torch.cuda.synchronize()
            started = time.monotonic()
            operation()
            if self.base.is_cuda:
                torch.cuda.synchronize()
            return time.monotonic() - started

        def full_copy():
            for layer in range(self.source.layers):
                kernels.gather(self.base, self.offsets, layer, self.pretouch_copies,
                               self.pretouch_n_copy, self.arena, variant=self.gather_variant,
                               chunk=self.gather_chunk, ctas=self.gather_ctas)

        seconds = measured(full_copy)
        saved = {l: (s.stats.clone(), s.marks.clone()) for l, s in self.layers.items()}
        size = sum(int((s.slot_of < 0).sum().item()) * SLOT_BYTES
                   for s in self.layers.values())

        def arena_copy():
            for l, s in self.layers.items():
                s.marks.copy_(s.slot_of < 0)
                kernels.arena_gather(self.base, self.offsets, l, s, s.marks,
                                     s.slots, self.arena, **self.arena_options())

        try:
            probes = [measured(arena_copy) for _ in range(2)]
        finally:
            for l, s in self.layers.items():
                s.stats.copy_(saved[l][0])
                s.marks.copy_(saved[l][1])
        self.timing_previous = {l: stats.cpu().tolist() for l, (stats, _) in saved.items()}
        full_size = self.source.layers * self.source.experts * SLOT_BYTES
        rate = lambda n, t: n / t / 1e9 if t > 0 else None
        self.timing_startup = dict(pretouch2_seconds=seconds, pretouch2_bytes=full_size,
                                  pretouch2_GBps=rate(full_size, seconds),
                                  arena_probe_seconds=probes, arena_probe_bytes=size,
                                  arena_probe_GBps=[rate(size, t) for t in probes])
        logging.getLogger(__name__).info('expert tier timing startup %s',
                                       json.dumps(self.timing_startup))

    def begin_timing(self, state, tokens):
        if len(self.timing_records) >= self.timing_cap:
            self.timing_dropped += 1
            return None
        layer = state.layer_number
        if layer not in self.timing_previous:
            self.timing_previous[layer] = state.stats.clone()
        events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        self.record_forward(events[0])
        return (layer, tokens, *events, len(self.timing_records))

    def record_forward(self, event):
        if self.base.is_cuda:
            event.record(torch.cuda.current_stream(self.base.device))
        else:
            event.record()

    def resolve_timing(self):
        if self.prefetch_enabled:
            self.resolve_prefetch_timing()
        if not self.timing_records:
            return
        self.timing_records[-1][4].synchronize()
        rows = self.timing_ring[:len(self.timing_records)].cpu().tolist()
        for layer, tokens, e0, e1, e2, row in self.timing_records:
            name = 'le256' if tokens <= 256 else 'le2048' if tokens <= 2048 else 'gt2048'
            bucket = self.timing_buckets[name]
            if self.timing_tail is not None:
                last_layer, last_event, last_bucket = self.timing_tail
                if layer == last_layer + 1:
                    last_bucket['gap_ms'] += last_event.elapsed_time(e0)
            self.timing_tail = (layer, e2, bucket)
            previous = self.timing_previous[layer]
            if isinstance(previous, torch.Tensor):
                previous = previous.cpu().tolist()
            current = rows[row]
            host = current[_COUNTERS.index('arena_source')] - previous[_COUNTERS.index('arena_source')]
            vram = current[_COUNTERS.index('arena_slots')] - previous[_COUNTERS.index('arena_slots')]
            self.timing_previous[layer] = current
            arena_ms = e1.elapsed_time(e2)
            bucket['calls'] += 1
            bucket['tokens'] += tokens
            bucket['presence_ms'] += e0.elapsed_time(e1)
            bucket['arena_ms'] += arena_ms
            bucket['host_experts'] += host
            bucket['vram_experts'] += vram
            if name == 'gt2048':
                per_layer = bucket['layers'].setdefault(str(layer), dict(calls=0, arena_ms=0.))
                per_layer['calls'] += 1
                per_layer['arena_ms'] += arena_ms
        for bucket in self.timing_buckets.values():
            bucket['host_GBps'] = (bucket['host_experts'] * SLOT_BYTES / bucket['arena_ms'] / 1e6
                                   if bucket['arena_ms'] > 0 else None)
        self.timing_records.clear()

    def arena_options(self):
        # Preserve the original v1 call as well as its output by default.
        return (dict(variant='v2', chunk=self.gather_chunk, ctas=self.gather_ctas)
                if self.arena_variant == 'v2' else {})

    def wait_prefetch(self, layer, tokens):
        used = False
        if self.pending is not None:
            target, arena_index, event = self.pending
            if event is not None:
                torch.cuda.current_stream(self.base.device).wait_event(event)
            used = target == layer and tokens >= self.prefetch_min
            if used:
                assert arena_index == layer % 2
                self.prefetch_used += 1
            else:
                self.prefetch_stale += 1
            self.pending = None
        return used

    def issue_prefetch(self, layer, tokens):
        if tokens < self.prefetch_min or layer+1 >= self.source.layers:
            return
        next_state = self.layers[layer+1]
        # All copy-stream tensors are owned by Runtime or TierState. In
        # particular, neither this route's IDs nor its marks escape to it.
        proxy = SimpleNamespace(slot_of=next_state.slot_of, stats=self.prefetch_stats)
        stream = self.prefetch_stream
        if stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.base.device))
            stream.wait_event(ready)
        context = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with context:
            timing = None
            if self.timing_enabled:
                if len(self.prefetch_timing_records) < self.timing_cap:
                    row = len(self.prefetch_timing_records)
                    e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    self.prefetch_timing_ring[row, 0].copy_(self.prefetch_stats[6], non_blocking=True)
                    e0.record()
                    timing = (e0, e1, row)
                else:
                    self.timing_dropped += 1
            kernels.arena_gather(self.base, self.offsets, layer+1, proxy,
                self.prefetch_marks, next_state.slots, self.arenas[(layer+1) % 2],
                variant='v2', chunk=self.gather_chunk, ctas=self.prefetch_ctas)
            if timing is not None:
                e0, e1, row = timing
                e1.record()
                self.prefetch_timing_ring[row, 1].copy_(self.prefetch_stats[6], non_blocking=True)
                self.prefetch_timing_records.append(timing)
            done = None
            if stream is not None:
                done = torch.cuda.Event()
                done.record(stream)
        self.pending = (layer+1, (layer+1) % 2, done)
        self.prefetch_issued += 1

    def resolve_prefetch_timing(self):
        if not self.prefetch_timing_records:
            return
        # The completion event also covers the counter snapshot after e1.
        if self.prefetch_stream is not None:
            self.prefetch_stream.synchronize()
        else:
            self.prefetch_timing_records[-1][1].synchronize()
        rows = self.prefetch_timing_ring[:len(self.prefetch_timing_records)].cpu().tolist()
        result = self.prefetch_timing
        for e0, e1, row in self.prefetch_timing_records:
            result['calls'] += 1
            result['ms'] += e0.elapsed_time(e1)
            result['source_experts'] += rows[row][1] - rows[row][0]
        result['GBps'] = (result['source_experts'] * SLOT_BYTES / result['ms'] / 1e6
                          if result['ms'] > 0 else None)
        self.prefetch_timing_records.clear()


def pretouch(runtime):
    """Read every source expert into the shared arena without changing layer state."""
    started = time.monotonic()
    for layer in range(runtime.source.layers):
        kernels.gather(runtime.base, runtime.offsets, layer, runtime.pretouch_copies,
                       runtime.pretouch_n_copy, runtime.arena,
                       variant=runtime.gather_variant, chunk=runtime.gather_chunk,
                       ctas=runtime.gather_ctas)
    if runtime.base.is_cuda:
        torch.cuda.synchronize()
    runtime.pretouch_seconds = time.monotonic() - started
    size = runtime.source.layers * runtime.source.experts * SLOT_BYTES
    logging.getLogger(__name__).info('expert tier pretouch seconds=%.6f bytes=%d',
                                   runtime.pretouch_seconds, size)


def finalize_layer(layer, *, runtime=None, validation=None):
    if not is_tier_layer(layer):
        return None
    validate_layer(layer, **(validation or {}))
    global _RUNTIME
    k = configured_k()
    device = layer.w13_weight.device
    if runtime is None:
        if _RUNTIME is None:
            weights = os.environ.get('SGLANG_EXPERT_TIER_WEIGHTS')
            if weights is None:
                from sglang.srt.server_args import get_global_server_args
                weights = get_global_server_args().model_path
            source = SourceIndex(weights)
            base, offsets = source.bind(device=device, mode='mmap')
            _RUNTIME = Runtime(source, base, offsets, allocate_slots(512, device), k=k,
                               variant=os.environ.get('SGLANG_EXPERT_TIER_DECIDE', 'matrix'))
        runtime = _RUNTIME
    if runtime.k != k:
        raise ValueError('expert tier K changed within one process')
    number = layer_number(layer)
    if number not in range(48) or number in runtime.layers:
        raise ValueError('expert tier duplicate or out-of-range layer')
    state = TierState(k, hot_experts(layer), layer.g1_alphas, layer.g2_alphas, device,
                      policy=runtime.policy, half=runtime.lfu_half, inc=runtime.lfu_inc,
                      skip_empty=runtime.skip_empty)
    state.runtime, state.layer_number = runtime, number
    state.slots = (layer.w13_weight, layer.w2_weight,
                   layer.w13_blockscale_swizzled, layer.w2_blockscale_swizzled)
    state.marks = torch.empty(512, dtype=torch.int32, device=device)
    state.slot_ids = torch.empty(k, dtype=torch.int32, device=device)
    runtime.layers[number] = state
    layer.expert_tier = state
    return state


def apply_quant_info(layer, quant_info):
    state = layer.expert_tier
    quant_info.quant_scales[2] = state.g1_slot
    quant_info.quant_scales[5] = state.g2_slot
    quant_info.expert_tier = state


def route(state, topk_ids, w13_weight, w2_weight, quant_scales, hidden=None, topk_weights=None):
    """Choose solely by shape, using preallocated device buffers in capture."""
    if topk_ids.shape[0] == 0:
        return topk_ids, w13_weight, w2_weight, quant_scales
    capturing = topk_ids.is_cuda and torch.cuda.is_current_stream_capturing()
    runtime = state.runtime
    runtime.before_route(state, capturing)
    scales = list(quant_scales)
    if topk_ids.numel() <= state.k:
        if hasattr(runtime, 'trace'):
            runtime.trace.record(state.layer_number, topk_ids, hidden, runtime.router_weights, topk_weights)
        counting = False
        if getattr(runtime, 'predict_mode', '0') == 'stats':
            counting = (hidden is not None and hidden.dtype == torch.bfloat16
                        and hidden.shape == (topk_ids.shape[0], runtime.router_weights.shape[2]))
            if not counting and not runtime.predict_warned:
                logging.getLogger(__name__).warning(
                    'expert tier prediction skipped: hidden must be BF16 [M, H] (packed FP4 is unsupported)')
                runtime.predict_warned = True
        if counting:
            wanted = predict.before(runtime, state, topk_ids)
        ids = state.slot_ids[:topk_ids.numel()].view(topk_ids.shape)
        if getattr(runtime, 'fused', False):
            kernels.decide_gather(topk_ids, state, ids, runtime.base, runtime.offsets,
                                  state.layer_number, state.slots,
                                  chunk=runtime.gather_chunk, ctas=runtime.gather_ctas)
        else:
            kernels.decide(topk_ids, state, ids, variant=runtime.variant)
            if getattr(runtime, 'skip_empty', False) and capturing:
                with _if_copies(state):
                    kernels.gather(runtime.base, runtime.offsets, state.layer_number,
                                   state.copy_list, state.n_copy, state.slots,
                                   variant=runtime.gather_variant, chunk=runtime.gather_chunk,
                                   ctas=runtime.gather_ctas)
            else:
                kernels.gather(runtime.base, runtime.offsets, state.layer_number,
                               state.copy_list, state.n_copy, state.slots,
                               variant=runtime.gather_variant, chunk=runtime.gather_chunk,
                               ctas=runtime.gather_ctas)
        if counting:
            predict.after(runtime, state, topk_ids, hidden, wanted)
        w13, w2, scales[1], scales[4] = state.slots
        scales[2], scales[5] = state.g1_slot, state.g2_slot
    else:
        ids = topk_ids
        timing = None
        if getattr(runtime, 'timing_enabled', False) and not capturing:
            timing = runtime.begin_timing(state, topk_ids.shape[0])
        prefetch = getattr(runtime, 'prefetch_enabled', False) and not capturing
        arena = runtime.arenas[state.layer_number % 2] if prefetch else runtime.arena
        used = runtime.wait_prefetch(state.layer_number, topk_ids.shape[0]) if prefetch else False
        kernels.presence(ids, state.marks)
        if timing is not None:
            runtime.record_forward(timing[3])
        arena_options = runtime.arena_options() if hasattr(runtime, 'arena_options') else {}
        if used:
            arena_options['count_only'] = True
        kernels.arena_gather(runtime.base, runtime.offsets, state.layer_number,
                             state, state.marks, state.slots, arena,
                             **arena_options)
        if timing is not None:
            runtime.record_forward(timing[4])
            runtime.timing_ring[timing[5]].copy_(state.stats, non_blocking=True)
            runtime.timing_records.append(timing)
        if prefetch:
            runtime.issue_prefetch(state.layer_number, topk_ids.shape[0])
        w13, w2, scales[1], scales[4] = arena
        scales[2], scales[5] = state.g1_alphas, state.g2_alphas
    return ids, w13, w2, scales
