# SPDX-License-Identifier: Apache-2.0
"""CPU references for route pruning, ordered stack and slot LRU."""
import numpy as np

E, L, TOPK = 512, 48, 10


def validate_policy(policy, half, inc):
    if policy not in ('lru', 'lfu'):
        raise ValueError('policy must be lru or lfu')
    if type(half) is not int or not 1 <= half <= 65536:
        raise ValueError('LFU_HALF must be an integer in 1..65536')
    if type(inc) is not int or not 1 <= inc <= 4096:
        raise ValueError('LFU_INC must be an integer in 1..4096')
    if half * inc > 4096:
        raise ValueError('LFU_HALF * LFU_INC must be <= 4096')


def prune(ids, weights):
    out = np.asarray(ids, dtype=np.int32).copy()
    if 2 <= out.shape[0] <= 64:
        valid = out >= 0
        counts = np.bincount(out[valid], minlength=E)
        remove = valid & (counts[np.maximum(out, 0)] == 1) & (weights < .08)
        remove[:, 0] = False
        out[remove] = -1
    return out


def census_routes(path, pruned=True):
    with np.load(path) as z:
        rows = z['cyc_rows'].copy()
        repeat = z['cyc_repeat'].copy()
        # The census contains T=4 throughout; check rather than assume silently.
        if not np.all(z['row_T'] == 4):
            raise ValueError('census_routes expects the specified T=4 census')
        raw = z['ids'].astype(np.int32).reshape(-1, 4, TOPK)
        if not np.array_equal(z['row_off'], np.arange(len(raw)+1)*40):
            raise ValueError('unexpected census offsets')
        if pruned:
            weights = z['w'].reshape(raw.shape)
            # Count equality in 40-route rows without a dense [rows,512] array.
            for start in range(0, len(raw), 4096):
                a = raw[start:start+4096]
                counts = (a.reshape(-1,40,1) == a.reshape(-1,1,40)).sum(2).reshape(a.shape)
                remove = (a >= 0) & (counts == 1) & (weights[start:start+4096] < .08)
                remove[:,:,0] = False
                a[remove] = -1
        return raw[rows], repeat


class SlotReference:
    """Independent literal implementation of SPEC section 2."""
    def __init__(self, k, hot=(), g1=None, g2=None, *, policy='lru', half=256, inc=16):
        validate_policy(policy, half, inc)
        self.policy, self.half, self.inc = policy, half, inc
        if policy == 'lfu':
            self.count = np.zeros(E, np.int32)
        if not 1 <= k <= E:
            raise ValueError('K must be in 1..512')
        hot = list(hot)
        if len(hot)>k or len(set(hot)) != len(hot) or any(e<0 or e>=E for e in hot):
            raise ValueError('invalid hot set')
        self.k = k
        self.slot_of = np.full(E, -1, np.int32)
        self.expert_of = np.full(k, -1, np.int32)
        self.key = np.full(k, -1, np.int64)
        for r,e in enumerate(hot):
            self.slot_of[e], self.expert_of[r], self.key[r] = r,e,-2-r
        self.step = 0
        self.stats = np.zeros(8, np.int64)
        self.overflow = False
        self.g1 = np.zeros(E,np.float32) if g1 is None else np.asarray(g1)
        self.g2 = np.zeros(E,np.float32) if g2 is None else np.asarray(g2)
        self.g1_slot = np.zeros(k,np.float32)
        self.g2_slot = np.zeros(k,np.float32)
        self.g1_slot[:len(hot)] = self.g1[hot]
        self.g2_slot[:len(hot)] = self.g2[hot]

    def decide(self, ids):
        demand = np.unique(np.asarray(ids)[np.asarray(ids)>=0])
        if np.any(demand>=E):
            raise ValueError('invalid expert')
        hits = int((self.slot_of[demand]>=0).sum())
        self.stats[:3] += (1,len(demand),hits)
        if len(demand)>self.k:
            self.stats[4] += 1
            self.overflow = True
            return np.full_like(ids,-1), np.empty((0,2),np.int32)
        if self.policy == 'lfu' and self.step > 0 and self.step % self.half == 0:
            self.count >>= 1
        missing = demand[self.slot_of[demand]<0]
        eligible = np.flatnonzero(~np.isin(self.expert_of,demand))
        # slot number breaks ties between initially empty slots.
        if self.policy == 'lfu':
            frequency = np.where(self.expert_of[eligible] >= 0,
                                 self.count[np.maximum(self.expert_of[eligible], 0)], 0)
            chosen = eligible[np.lexsort((eligible, self.key[eligible], frequency))][:len(missing)]
        else:
            chosen = eligible[np.argsort(self.key[eligible],kind='stable')][:len(missing)]
        copies = np.column_stack((chosen,missing)).astype(np.int32)
        for s,e in copies:
            old = self.expert_of[s]
            if old>=0:
                self.slot_of[old] = -1
            self.slot_of[e],self.expert_of[s] = s,e
            self.g1_slot[s],self.g2_slot[s] = self.g1[e],self.g2[e]
        self.key[self.slot_of[demand]] = self.step*E + 511-demand
        self.step += 1
        if self.policy == 'lfu':
            self.count[demand] = np.minimum(8191, self.count[demand] + self.inc)
        self.stats[3] += len(missing)
        ids = np.asarray(ids)
        return np.where(ids>=0,self.slot_of[np.maximum(ids,0)],-1), copies


def replay_census(routes, repeat, k, deloop=True, *, policy='lru', half=256, inc=16):
    """Compare a vectorized slot reference with an independent hit oracle.

    LRU uses the ordered stack; LFU uses sets/dicts and three-part ordering.
    Both advance on every included step and compare every demanded hit bit.
    """
    validate_policy(policy, half, inc)
    count_lfu = np.zeros((L,E), np.int32) if policy == 'lfu' else None
    oracles = [DictReference(k, policy=policy, half=half, inc=inc) for _ in range(L)] if policy == 'lfu' else None
    slot = np.full((L,E),-1,np.int32)
    expert = np.full((L,k),-1,np.int32)
    keys = np.full((L,k),-1,np.int64)
    stack = np.broadcast_to(np.arange(E),(L,E)).copy()
    stack_size = np.zeros(L,np.int32)
    rr = np.arange(L)[:,None]
    total = hits = cycles = 0
    for cycle,ids in enumerate(routes):
        if deloop and repeat[cycle]:
            continue
        demand = np.zeros((L,E),bool)
        demand[rr,np.maximum(ids.reshape(L,-1),0)] = ids.reshape(L,-1)>=0
        # Assignment of duplicate index 0 could hide a real expert 0; set via counts.
        demand[:,0] = (ids.reshape(L,-1)==0).any(1)
        stack_hit = np.zeros_like(demand)
        pos = np.arange(E)[None,:]
        stack_hit[rr,stack] = pos < np.minimum(k,stack_size)[:,None]
        slot_hit = slot>=0
        if policy == 'lru' and not np.array_equal(stack_hit & demand, slot_hit & demand):
            raise AssertionError(f'stack/slot mismatch at cycle {cycle}, K={k}')
        if oracles is not None:
            for layer, oracle in enumerate(oracles):
                expected = oracle.decide(ids[layer])
                actual = set(np.flatnonzero(slot_hit[layer] & demand[layer]).tolist())
                if actual != expected:
                    raise AssertionError(f'dict/slot mismatch at cycle {cycle}, layer={layer}, K={k}')
        total += int(demand.sum())
        hits += int((slot_hit & demand).sum())
        missing = demand & ~slot_hit
        if policy == 'lfu' and cycles > 0 and cycles % half == 0:
            count_lfu >>= 1
        # Sort missing experts ascending, then non-missing entries.
        miss = np.argsort(~missing,axis=1,kind='stable')[:,:k]
        eligible = ~np.take_along_axis(demand,np.maximum(expert,0),1) | (expert<0)
        eviction = keys
        if policy == 'lfu':
            frequency = np.where(expert >= 0,
                                  np.take_along_axis(count_lfu, np.maximum(expert, 0), 1), 0)
            eviction = (frequency.astype(np.int64) << 50) | ((keys + k + 1) << 9)
        order = np.argsort(np.where(eligible,eviction,np.iinfo(np.int64).max),axis=1,kind='stable')
        count = missing.sum(1)
        active = np.arange(k)[None,:] < count[:,None]
        layers = np.broadcast_to(rr,(L,k))[active]
        slots = order[active]
        new = miss[active]
        old = expert[layers,slots]
        valid = old>=0
        slot[layers[valid],old[valid]] = -1
        expert[layers,slots] = new
        slot[layers,new] = slots
        dl,de = np.nonzero(demand)
        keys[dl,slot[dl,de]] = cycles*E+511-de
        if policy == 'lfu':
            count_lfu[demand] = np.minimum(8191, count_lfu[demand] + inc)
        # D ascending at top, followed by the surviving previous stack order.
        retained = np.take_along_axis(demand,stack,axis=1)
        old_order = np.argsort(retained,axis=1,kind='stable')
        survivors = np.take_along_axis(stack,old_order,1)
        front = np.argsort(~demand,axis=1,kind='stable')
        nd = demand.sum(1)
        stack = np.where(pos<nd[:,None], front,
                         np.take_along_axis(survivors,np.maximum(pos-nd[:,None],0),1))
        # Stack stores every expert after the first step, but unseen entries must
        # not count as hits. Track its actual populated length via seen set below.
        if cycles == 0:
            seen = demand.copy()
        else:
            seen |= demand
        stack_size = seen.sum(1)
        cycles += 1
    return dict(cycles=cycles,demand=total,hits=hits,misses=total-hits,
                demand_per_cycle=total/cycles,hit_percent=100*hits/total,
                misses_per_cycle=(total-hits)/cycles)


class DictReference:
    """Independent set/dict eviction oracle, with explicit three-part ordering.

    Empty slots are represented by negative identities. Last use is a step,
    rather than the packed slot key used by SlotReference and the kernels.
    """
    def __init__(self, k, hot=(), *, policy='lru', half=256, inc=16):
        validate_policy(policy, half, inc)
        self.k, self.policy, self.half, self.inc = k, policy, half, inc
        self.resident = set(hot)
        self.last = {e: -2-r for r, e in enumerate(hot)}
        self.count = {}
        self.step = 0

    def decide(self, ids):
        demand = set(int(e) for e in np.asarray(ids).flat if e >= 0)
        hits = demand & self.resident
        if len(demand) > self.k:
            return hits
        if self.policy == 'lfu' and self.step > 0 and self.step % self.half == 0:
            self.count = {e: n//2 for e, n in self.count.items()}
        missing = demand-self.resident
        if missing:
            candidates = self.resident - demand
            # Empty identities only affect eviction order, never hits or counts.
            candidates |= set(range(-1, -(self.k-len(self.resident))-1, -1))
            def priority(e):
                return (self.count.get(e, 0) if self.policy == 'lfu' else 0,
                        self.last[e] if e >= 0 else -1, -e)
            for victim, new in zip(sorted(candidates, key=priority), sorted(missing)):
                self.resident.discard(victim)
                self.resident.add(new)
        for e in demand:
            self.last[e] = self.step
            if self.policy == 'lfu':
                self.count[e] = min(8191, self.count.get(e, 0) + self.inc)
        self.step += 1
        return hits


def hot_layers(routes, repeat, k=307):
    counts = np.zeros((L,E),np.int64)
    for ids in routes[~repeat]:
        demand = np.zeros((L,E),bool)
        for layer in range(L):
            demand[layer,ids[layer][ids[layer]>=0]] = True
        counts += demand
    return np.argsort(-counts,axis=1,kind='stable')[:,:k].tolist()
