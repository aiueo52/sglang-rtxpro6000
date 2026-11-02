# SPDX-License-Identifier: Apache-2.0
"""One-CTA decide and fixed-chunk gather, usable in Triton CPU interpreter."""
import triton as tr
import triton.language as tl

CHUNK_BYTES = 65536
CHUNKS = tr.cdiv(2764800, CHUNK_BYTES)


@tr.jit(do_not_specialize=["N"])
def _decide_body(IDs, Out, Slot, Expert, Key, Step, Copies, NC, G1, G2,
                 GS1, GS2, Stats, Overflow, Count, N, K:tl.constexpr,
                 BK:tl.constexpr, BN:tl.constexpr, Sort:tl.constexpr,
                 LFU:tl.constexpr, Half:tl.constexpr, Inc:tl.constexpr):
    r = tl.arange(0,BN)
    ids = tl.load(IDs+r,r<N,-1).to(tl.int32)
    hist = tl.histogram(tl.maximum(ids,0).to(tl.int32),512,mask=(ids>=0)) 
    e = tl.arange(0,512)
    demand = hist>0
    old_slot = tl.load(Slot+e)
    missing = demand & (old_slot<0)
    nd = tl.sum(demand.to(tl.int32),0)
    nm = tl.sum(missing.to(tl.int32),0)
    tl.store(NC,0)
    tl.store(Stats,tl.load(Stats)+1)
    tl.store(Stats+1,tl.load(Stats+1)+nd)
    tl.store(Stats+2,tl.load(Stats+2)+nd-nm)
    if nd>K:
        tl.store(Overflow,1)
        tl.store(Stats+4,tl.load(Stats+4)+1)
        tl.store(Out+r,-1,r<N)
    else:
        step = tl.load(Step)
        if LFU:
            counts = tl.load(Count+e)
            if (step > 0) & (step % Half == 0):
                counts = counts >> 1
            tl.store(Count+e,counts)
        s = tl.arange(0,BK)
        old_e = tl.load(Expert+s,s<K,-1)
        # Histogram is already in registers. A gather does not touch shared state.
        protected = tl.gather(demand,tl.maximum(old_e,0),axis=0) & (old_e>=0)
        key = tl.load(Key+s,s<K,0)
        eligible = (s<K)&~protected
        if nm > 0:
            if Sort:
                eviction = ((key + K + 1) << 9) | s.to(tl.int64)
                if LFU:
                    frequency = tl.gather(counts,tl.maximum(old_e,0),axis=0)
                    frequency = tl.where(old_e>=0,frequency,0).to(tl.int64)
                    eviction = (frequency << 50) | eviction
                packed = tl.where(eligible, eviction, 0x7fffffffffffffff)
                ordered = tl.sort(packed, descending=False)
                tl.store(Copies+2*s, (ordered & 511).to(tl.int32), s<nm)
            else:
                before = (key[None,:]<key[:,None]) | ((key[None,:]==key[:,None]) & (s[None,:]<s[:,None]))
                slot_rank = tl.sum((before & eligible[None,:]).to(tl.int32),axis=1)
                chosen = eligible & (slot_rank<nm)
                tl.store(Copies+2*slot_rank,s,chosen)
        tl.debug_barrier()
        miss_rank = tl.cumsum(missing.to(tl.int32),axis=0)-1
        assigned = tl.load(Copies+2*tl.maximum(miss_rank,0),missing,0)
        victim = tl.load(Expert+assigned,missing,-1)
        tl.store(Slot+tl.maximum(victim,0),-1,missing&(victim>=0))
        tl.debug_barrier()
        tl.store(Expert+assigned,e,missing)
        tl.store(Slot+e,assigned,missing)
        tl.store(Copies+2*miss_rank+1,e,missing)
        a1 = tl.load(G1+e,missing,0)
        a2 = tl.load(G2+e,missing,0)
        tl.store(GS1+assigned,a1,missing)
        tl.store(GS2+assigned,a2,missing)
        tl.debug_barrier()
        final_slot = tl.load(Slot+e)
        tl.store(Key+tl.maximum(final_slot,0),step*512+511-e,demand)
        tl.store(Step,step+1)
        if LFU:
            tl.store(Count+e,tl.minimum(counts+Inc,8191),demand)
        tl.store(NC,nm)
        tl.store(Stats+3,tl.load(Stats+3)+nm)
        tl.debug_barrier()
        mapped = tl.load(Slot+tl.maximum(ids,0),r<N, -1)
        tl.store(Out+r,tl.where(ids>=0,mapped,-1),r<N)


@tr.jit(do_not_specialize=["N"])
def _decide(IDs, Out, Slot, Expert, Key, Step, Copies, NC, G1, G2,
            GS1, GS2, Stats, Overflow, N, K:tl.constexpr,
            BK:tl.constexpr, BN:tl.constexpr, Sort:tl.constexpr):
    # Keep the pre-C10 LRU launch signature, with no count pointer.
    _decide_body(IDs,Out,Slot,Expert,Key,Step,Copies,NC,G1,G2,GS1,GS2,
                 Stats,Overflow,None,N,K,BK,BN,Sort,False,256,16)


@tr.jit(do_not_specialize=["N"])
def _decide_lfu(IDs, Out, Slot, Expert, Key, Step, Copies, NC, G1, G2,
                GS1, GS2, Stats, Overflow, Count, N, K:tl.constexpr,
                BK:tl.constexpr, BN:tl.constexpr, Half:tl.constexpr, Inc:tl.constexpr):
    _decide_body(IDs,Out,Slot,Expert,Key,Step,Copies,NC,G1,G2,GS1,GS2,
                 Stats,Overflow,Count,N,K,BK,BN,True,True,Half,Inc)


def _policy_options(state, policy, half, inc):
    from .reference import validate_policy
    policy = state.policy if policy is None else policy
    half = state.half if half is None else half
    inc = state.inc if inc is None else inc
    validate_policy(policy, half, inc)
    return policy, half, inc


def decide(ids, state, out, variant="matrix", *, policy=None, half=None, inc=None):
    """Launch using only caller-owned buffers (safe to capture)."""
    if variant not in ("matrix", "sort"):
        raise ValueError("decide variant must be matrix or sort")
    policy, half, inc = _policy_options(state, policy, half, inc)
    if policy == 'lfu' and variant != 'sort':
        raise ValueError('lfu requires decide sort')
    args = (ids,out,state.slot_of,state.expert_of,state.key,state.step,
            state.copy_list,state.n_copy,state.g1_alphas,state.g2_alphas,
            state.g1_slot,state.g2_slot,state.stats,state.overflow)
    shape = (ids.numel(),state.k,tr.next_power_of_2(state.k),
             tr.next_power_of_2(max(ids.numel(),1)))
    if policy == 'lru':
        _decide[(1,)](*args,*shape,variant == "sort")
    else:
        _decide_lfu[(1,)](*args,state.count,*shape,half,inc)
    return out


@tr.jit
def _copy_chunk(Base, Off, W13, W2, S13, S2, layer, expert, slot,
                ch, B:tl.constexpr):
    # Two adjacent int32 lanes limit source transactions to eight bytes.
    group = ch*(B//8)+tl.arange(0,B//8)
    lane = tl.arange(0,2)
    q = group[:,None]*2+lane[None,:]
    byte = q*4
    # Linear destination segments: w13 up, gate; w2; s13 up, gate; s2.
    for part in tl.static_range(6):
        if part == 0:
            start=0
            size=819200
            idx=2
            dest = W13.to(tl.pointer_type(tl.int32))+slot*409600
        elif part == 1:
            start=819200
            size=819200
            idx=1
            dest = W13.to(tl.pointer_type(tl.int32))+slot*409600+204800
        elif part == 2:
            start=1638400
            size=819200
            idx=0
            dest = W2.to(tl.pointer_type(tl.int32))+slot*204800
        elif part == 3:
            start=2457600
            size=102400
            idx=5
            dest = S13.to(tl.pointer_type(tl.int32))+slot*51200
        elif part == 4:
            start=2560000
            size=102400
            idx=4
            dest = S13.to(tl.pointer_type(tl.int32))+slot*51200+25600
        else:
            start=2662400
            size=102400
            idx=3
            dest = S2.to(tl.pointer_type(tl.int32))+slot*25600
        local = (byte-start)//4
        mask = (byte>=start)&(byte<start+size)
        src_q = local
        if part>=3:
            if part==5:
                n=10
            else:
                n=40
            src_q = ((local//(128*n))*128+(local%4)*32+(local//4)%32)*n+(local//128)%n
        off = tl.load(Off+(layer*512+expert)*6+idx)
        source = (Base+tl.multiple_of(off,8)).to(tl.pointer_type(tl.int32))
        if part<3:
            # Preserve only eight-byte alignment through pointer casts.
            src_q = tl.max_contiguous(tl.multiple_of(src_q,(1,2)),(1,2))
            local = tl.max_contiguous(tl.multiple_of(local,(1,2)),(1,2))
        val = tl.load(source+src_q,mask,0)
        tl.store(dest+local,val,mask)


@tr.jit(do_not_specialize=["Layer"])
def _gather(Base,Off,Copies,NC,W13,W2,S13,S2, Layer,B:tl.constexpr):
    c,ch = tl.program_id(0),tl.program_id(1)
    if c<tl.load(NC):
        s,e = tl.load(Copies+2*c),tl.load(Copies+2*c+1)
        _copy_chunk(Base,Off,W13,W2,S13,S2,Layer,e,s,ch,B)


@tr.jit
def _gather_v2_body(Base, Off, Copies, NC, W13, W2, S13, S2, Layer,
               B:tl.constexpr, G:tl.constexpr):
    W:tl.constexpr = tr.cdiv(819200, B)
    S:tl.constexpr = tr.cdiv(102400, B)
    JOBS:tl.constexpr = 3 * (W + S)
    count = tl.load(NC)
    # The host grid is fixed even when NC changes during graph replay.
    for work in range(tl.program_id(0), count * JOBS, G):
        c = work // JOBS
        local = work % JOBS
        slot = tl.load(Copies + 2*c)
        expert = tl.load(Copies + 2*c + 1)
        if local < 3*W:
            part = local // W
            piece = local % W
            if part == 0:
                dest = W2.to(tl.pointer_type(tl.int32)) + slot*204800
            elif part == 1:
                dest = W13.to(tl.pointer_type(tl.int32)) + slot*409600+204800
            else:
                dest = W13.to(tl.pointer_type(tl.int32)) + slot*409600
            size = 204800
        else:
            part = 3 + (local-3*W)//S
            piece = (local-3*W)%S
            if part == 3:
                dest = S2.to(tl.pointer_type(tl.int32)) + slot*25600
            elif part == 4:
                dest = S13.to(tl.pointer_type(tl.int32)) + slot*51200+25600
            else:
                dest = S13.to(tl.pointer_type(tl.int32)) + slot*51200
            size = 25600
        # A two-lane axis caps source vectorization at two int32 (8 bytes).
        group = piece*(B//8) + tl.arange(0, B//8)
        lane = tl.arange(0, 2)
        j = group[:, None]*2 + lane[None, :]
        j = tl.max_contiguous(tl.multiple_of(j, (1, 2)), (1, 2))
        off = tl.load(Off + (Layer*512+expert)*6 + part)
        source = (Base + tl.multiple_of(off, 8)).to(tl.pointer_type(tl.int32))
        value = tl.load(source+j, j<size, 0)
        if local < 3*W:
            target = j
        else:
            n = tl.where(part == 3, 10, 40)
            row = j//n
            col = j%n
            target = (row//128)*128*n + col*128 + (row%32)*4 + (row%128)//32
        tl.store(dest+target, value, j<size)


@tr.jit(do_not_specialize=["Layer"])
def _gather_v2(Base, Off, Copies, NC, W13, W2, S13, S2, Layer,
               B:tl.constexpr, G:tl.constexpr):
    _gather_v2_body(Base,Off,Copies,NC,W13,W2,S13,S2,Layer,B,G)


@tr.jit(do_not_specialize=["N", "Layer"])
def _decide_gather(IDs, Out, Slot, Expert, Key, Step, Copies, NC, G1, G2,
                   GS1, GS2, Stats, Overflow, Count, Sync,
                   Base, Off, W13, W2, S13, S2, N, Layer,
                   K:tl.constexpr, BK:tl.constexpr, BN:tl.constexpr,
                   B:tl.constexpr, G:tl.constexpr, LFU:tl.constexpr,
                   Half:tl.constexpr, Inc:tl.constexpr):
    if tl.program_id(0) == 0:
        _decide_body(IDs,Out,Slot,Expert,Key,Step,Copies,NC,G1,G2,GS1,GS2,
                     Stats,Overflow,Count,N,K,BK,BN,True,LFU,Half,Inc)
        tl.atomic_xchg(Sync,1,sem="release",scope="gpu")
    elif G > 1:
        while tl.atomic_add(Sync,0,sem="acquire",scope="gpu") != 1:
            pass
    _gather_v2_body(Base,Off,Copies,NC,W13,W2,S13,S2,Layer,B,G)
    done = tl.atomic_add(Sync+1,1,sem="acq_rel",scope="gpu")
    if done == G-1:
        tl.atomic_xchg(Sync,0,sem="release",scope="gpu")
        tl.atomic_xchg(Sync+1,0,sem="release",scope="gpu")


def decide_gather(ids, state, out, base, offsets, layer, slots, *,
                  chunk=16384, ctas=376, policy=None, half=None, inc=None):
    validate_gather('v2', chunk, ctas)
    policy, half, inc = _policy_options(state, policy, half, inc)
    _decide_gather[(ctas,)](
        ids,out,state.slot_of,state.expert_of,state.key,state.step,
        state.copy_list,state.n_copy,state.g1_alphas,state.g2_alphas,
        state.g1_slot,state.g2_slot,state.stats,state.overflow,
        state.count if policy == 'lfu' else None,state.sync,
        base,offsets,*slots,ids.numel(),layer,state.k,tr.next_power_of_2(state.k),
        tr.next_power_of_2(max(ids.numel(),1)),chunk,ctas,policy == 'lfu',half,inc)
    return out


def validate_gather(variant, chunk, ctas):
    if variant not in ('v1', 'v2'):
        raise ValueError('gather variant must be v1 or v2')
    if type(chunk) is not int or not 4096 <= chunk <= 65536 or chunk & (chunk-1):
        raise ValueError('gather chunk must be a power of two in 4096..65536')
    if type(ctas) is not int or not 1 <= ctas <= 4096:
        raise ValueError('gather ctas must be in 1..4096')


def gather(base, offsets, layer, copies, n_copy, slots, cap=None, *,
           variant='v1', chunk=16384, ctas=376):
    validate_gather(variant, chunk, ctas)
    capacity = copies.shape[0] if cap is None else cap
    if variant == 'v1':
        _gather[(capacity, CHUNKS)](
            base,offsets,copies,n_copy,*slots,layer,CHUNK_BYTES)
    else:
        jobs = 3*(tr.cdiv(819200, chunk)+tr.cdiv(102400, chunk))
        grid = min(ctas, capacity*jobs)
        if grid:
            _gather_v2[(grid,)](base, offsets, copies, n_copy, *slots, layer, chunk, grid)


@tr.jit(do_not_specialize=["N"])
def _presence(IDs,P,N,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    ids=tl.load(IDs+x,x<N,-1).to(tl.int32)
    tl.store(P+ids,1,ids>=0)


def presence(ids, marks):
    marks.zero_()
    block=1024
    _presence[(max(1,tr.cdiv(ids.numel(),block)),)](ids,marks,ids.numel(),block)


@tr.jit(do_not_specialize=["Layer"])
def _arena(Base,Off,Slot,P,W13,W2,S13,S2,A13,A2,AS13,AS2,Stats,
           Layer,B:tl.constexpr):
    e,ch=tl.program_id(0),tl.program_id(1)
    if e==0 and ch==0:
        tl.atomic_add(Stats+5,1)
    if tl.load(P+e)!=0:
        s=tl.load(Slot+e)
        if ch==0:
            if s>=0:
                tl.atomic_add(Stats+7,1)
            else:
                tl.atomic_add(Stats+6,1)
        if s<0:
            _copy_chunk(Base,Off,A13,A2,AS13,AS2,Layer,e,e,ch,B)
        else:
            group=ch*(B//8)+tl.arange(0,B//8)
            lane=tl.arange(0,2)
            q=group[:,None]*2+lane[None,:]
            for part in tl.static_range(4):
                if part==0:
                    start=0
                    size=409600
                    src=W13.to(tl.pointer_type(tl.int32))
                    dst=A13.to(tl.pointer_type(tl.int32))
                elif part==1:
                    start=409600
                    size=204800
                    src=W2.to(tl.pointer_type(tl.int32))
                    dst=A2.to(tl.pointer_type(tl.int32))
                elif part==2:
                    start=614400
                    size=51200
                    src=S13.to(tl.pointer_type(tl.int32))
                    dst=AS13.to(tl.pointer_type(tl.int32))
                else:
                    start=665600
                    size=25600
                    src=S2.to(tl.pointer_type(tl.int32))
                    dst=AS2.to(tl.pointer_type(tl.int32))
                local=q-start
                local=tl.max_contiguous(tl.multiple_of(local,(1,2)),(1,2))
                m=(local>=0)&(local<size)
                v=tl.load(src+s*size+local,m,0)
                tl.store(dst+e*size+local,v,m)


@tr.jit(do_not_specialize=["Layer"])
def _arena_v2(Base, Off, Slot, P, W13, W2, S13, S2, A13, A2, AS13, AS2,
              Stats, Layer, B:tl.constexpr, G:tl.constexpr, E:tl.constexpr,
              CountOnly:tl.constexpr):
    W:tl.constexpr = tr.cdiv(819200, B)
    S:tl.constexpr = tr.cdiv(102400, B)
    JOBS:tl.constexpr = 3 * (W + S)
    if tl.program_id(0) == 0:
        tl.atomic_add(Stats+5, 1)
    for work in range(tl.program_id(0), E * JOBS, G):
        e = work // JOBS
        local = work % JOBS
        if tl.load(P+e) != 0:
            s = tl.load(Slot+e)
            if local == 0:
                if s < 0:
                    tl.atomic_add(Stats+6, 1)
                else:
                    tl.atomic_add(Stats+7, 1)
            if not CountOnly:
                if local < 3*W:
                    part = local // W
                    piece = local % W
                    size = 204800
                    if part == 0:
                        src = W2.to(tl.pointer_type(tl.int32)) + s*204800
                        dst = A2.to(tl.pointer_type(tl.int32)) + e*204800
                    elif part == 1:
                        src = W13.to(tl.pointer_type(tl.int32)) + s*409600+204800
                        dst = A13.to(tl.pointer_type(tl.int32)) + e*409600+204800
                    else:
                        src = W13.to(tl.pointer_type(tl.int32)) + s*409600
                        dst = A13.to(tl.pointer_type(tl.int32)) + e*409600
                else:
                    part = 3 + (local-3*W)//S
                    piece = (local-3*W)%S
                    size = 25600
                    if part == 3:
                        src = S2.to(tl.pointer_type(tl.int32)) + s*25600
                        dst = AS2.to(tl.pointer_type(tl.int32)) + e*25600
                    elif part == 4:
                        src = S13.to(tl.pointer_type(tl.int32)) + s*51200+25600
                        dst = AS13.to(tl.pointer_type(tl.int32)) + e*51200+25600
                    else:
                        src = S13.to(tl.pointer_type(tl.int32)) + s*51200
                        dst = AS13.to(tl.pointer_type(tl.int32)) + e*51200
                group = piece*(B//8) + tl.arange(0, B//8)
                lane = tl.arange(0, 2)
                j = group[:, None]*2 + lane[None, :]
                j = tl.max_contiguous(tl.multiple_of(j, (1, 2)), (1, 2))
                target = j
                if s < 0:
                    off = tl.load(Off + (Layer*512+e)*6 + part)
                    src = (Base + tl.multiple_of(off, 8)).to(tl.pointer_type(tl.int32))
                    if local >= 3*W:
                        n = tl.where(part == 3, 10, 40)
                        row = j//n
                        col = j%n
                        target = (row//128)*128*n + col*128 + (row%32)*4 + (row%128)//32
                value = tl.load(src+j, j<size, 0)
                tl.store(dst+target, value, j<size)


def arena_gather(base, offsets, layer, state, marks, slots, arena, *,
                 variant='v1', chunk=16384, ctas=376, count_only=False):
    validate_gather(variant, chunk, ctas)
    if variant == 'v1':
        if count_only:
            raise ValueError('arena count_only requires v2')
        _arena[(marks.numel(),CHUNKS)](base,offsets,state.slot_of,marks,*slots,*arena,
                                     state.stats,layer,CHUNK_BYTES)
    else:
        _arena_v2[(ctas,)](base, offsets, state.slot_of, marks, *slots, *arena,
                           state.stats, layer, chunk, ctas, marks.numel(), count_only)
