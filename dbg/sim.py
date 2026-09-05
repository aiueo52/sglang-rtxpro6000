"""Offline controller simulation with realistic per-batch accept distributions.

Distributions are shaped from the 2026-09-05 debug histograms (bimodal on
code-edit) and normalised to the measured fixed-profile means:
  drafts@15: code 10.85 agent 4.59 prose-en 1.99 prose-ja 1.64
  drafts@3 : code  2.77 agent 2.22 prose-en 1.50 prose-ja 1.46
"""
import random, sys, json
sys.path.insert(0, "python")
from sglang.srt.speculative.adaptive_spec_params import AdaptiveStepSlot

def pool(mean, steps, spread):  # noqa
    """A bimodal pool of integers in [0, steps] with the given mean."""
    if spread:   # bimodal: full chain or a short one
        hi, lo = steps, max(0, steps // 5)
        p = (mean - lo) / (hi - lo)
        return lambda r: hi if r.random() < p else lo
    return lambda r: min(steps, max(0, round(r.gauss(mean, 0.9))))

WL = {  # name -> {steps: (mean, bimodal)}
    "code-edit": {15: (10.85, True), 3: (2.77, True)},
    "agent-loop": {15: (4.59, True), 3: (2.22, False)},
    "prose-en": {15: (1.99, False), 3: (1.50, False)},
    "prose-ja": {15: (1.64, False), 3: (1.46, False)},
}
# Cost model: ms/iteration = 9.74 + 0.70*S (prof/OVERHEAD_REPORT.md)
STEP_MS = lambda s: 9.74 + 0.70 * s
import os
SWITCH_PENALTY_BATCHES = int(os.environ.get("SWPEN", "2"))  # cold batches after a swap

def run(cfg, wl, batches=600, lag=2, seed=0):
    r = random.Random(seed)
    draws = {s: pool(WL[wl][s][0], s, WL[wl][s][1]) for s in (3, 15)}
    slot = AdaptiveStepSlot(initial_steps=15, cfg=dict(cfg))
    inflight = [slot.current_steps] * lag
    switches = 0
    tokens = 0.0
    ms = 0.0
    cold = 0
    prev = slot.current_steps
    for _ in range(batches):
        live = slot.current_steps
        n = draws[live](r)
        if cold:                      # post-switch cold draft state
            n = min(n, 1); cold -= 1
        tokens += n + 1
        ms += STEP_MS(live)
        produced_by = inflight.pop(0)
        slot.update([draws[produced_by](r) if produced_by != live else n])
        inflight.append(slot.current_steps)
        if slot.current_steps != prev:
            switches += 1; cold = SWITCH_PENALTY_BATCHES; prev = slot.current_steps
    return switches, tokens / ms * 1000, slot.current_steps

def fixed(wl, steps, batches=600, seed=0):
    r = random.Random(seed)
    d = pool(WL[wl][steps][0], steps, WL[wl][steps][1])
    tok = sum(d(r) + 1 for _ in range(batches))
    return tok / (batches * STEP_MS(steps)) * 1000

cfgs = {
  "v1 (a2, shipped)": {"candidate_steps":[3,15],"ema_alpha":0.2,"update_interval":5,
                       "warmup_batches":10,"down_hysteresis":3.5,"up_hysteresis":0.0,
                       "reset_ema_on_switch":True},
  "v2 alpha.1/int20": {"candidate_steps":[3,15],"ema_alpha":0.1,"update_interval":20,
                       "warmup_batches":15,"down_hysteresis":3.5,"up_hysteresis":0.0,
                       "reset_ema_on_switch":True},
  "v3 grace40/noreseed": {"candidate_steps":[3,15],"ema_alpha":0.1,"update_interval":20,
                       "warmup_batches":15,"switch_grace_batches":40,"down_hysteresis":3.5,
                       "up_hysteresis":0.0,"reset_ema_on_switch":False},
}
for name, cfg in cfgs.items():
    print(f"--- {name}")
    for wl in WL:
        rows = [run(cfg, wl, seed=s) for s in range(5)]
        sw = sum(r[0] for r in rows)/5; tps = sum(r[1] for r in rows)/5
        end = [r[2] for r in rows]
        f3 = sum(fixed(wl,3,seed=s) for s in range(5))/5
        f15 = sum(fixed(wl,15,seed=s) for s in range(5))/5
        print(f"  {wl:11s} switches={sw:5.1f} tps={tps:6.0f}  (W4 {f3:.0f} / W16 {f15:.0f}) end={end}")

print("\n=== warm start: controller already settled at steps=3, then this workload arrives")
for name, cfg in cfgs.items():
    print(f"--- {name}")
    for wl in WL:
        res=[]
        for seed in range(5):
            r = random.Random(seed)
            draws = {s: pool(WL[wl][s][0], s, WL[wl][s][1]) for s in (3, 15)}
            slot = AdaptiveStepSlot(initial_steps=15, cfg=dict(cfg))
            slot.current_steps = 3; slot.ema_accept_len = 1.5; slot._batch_count = 100
            inflight=[3,3]; sw=0; tok=0.0; ms=0.0; cold=0; prev=3
            for _ in range(600):
                live=slot.current_steps; n=draws[live](r)
                if cold: n=min(n,1); cold-=1
                tok+=n+1; ms+=STEP_MS(live)
                pb=inflight.pop(0)
                slot.update([draws[pb](r) if pb!=live else n]); inflight.append(slot.current_steps)
                if slot.current_steps!=prev: sw+=1; cold=SWITCH_PENALTY_BATCHES; prev=slot.current_steps
            res.append((sw, tok/ms*1000, slot.current_steps))
        sw=sum(x[0] for x in res)/5; tps=sum(x[1] for x in res)/5
        f3=sum(fixed(wl,3,seed=s) for s in range(5))/5; f15=sum(fixed(wl,15,seed=s) for s in range(5))/5
        print(f"  {wl:11s} switches={sw:5.1f} tps={tps:6.0f}  (W4 {f3:.0f} / W16 {f15:.0f}) end={[x[2] for x in res]}")
