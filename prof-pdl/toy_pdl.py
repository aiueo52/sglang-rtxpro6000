"""Toy PDL validation: chain of small dependent Triton kernels under CUDA graph."""
import os, sys, time
import torch, triton, triton.language as tl

N_CHAIN = int(os.environ.get("CHAIN", "100"))

@triton.jit
def chain_kernel(X, Y, W, n, PDL: tl.constexpr, TRIG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    # independent prologue (weights): does not depend on the previous kernel
    w = tl.load(W + offs, mask=m, other=0.0)
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + offs, mask=m, other=0.0)   # dependent load
    y = x * 1.0009765625 + w * 0.0
    tl.store(Y + offs, y, mask=m)
    if TRIG:
        tl.extra.cuda.gdc_launch_dependents()

def run_chain(bufs, W, grid, BLOCK, n, mode):
    pdl = mode >= 2
    trig = mode >= 3
    for i in range(N_CHAIN):
        a = bufs[i % 2]; b = bufs[(i + 1) % 2]
        chain_kernel[grid](a, b, W, n, PDL=pdl, TRIG=trig, BLOCK=BLOCK,
                           launch_pdl=(mode >= 1), num_warps=4)

def bench(n, nblocks, mode):
    dev = 'cuda'
    bufs = [torch.randn(n, device=dev, dtype=torch.float32) for _ in range(2)]
    W = torch.randn(n, device=dev, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(max(1, n // nblocks))
    grid = (triton.cdiv(n, BLOCK),)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_chain(bufs, W, grid, BLOCK, n, mode)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_chain(bufs, W, grid, BLOCK, n, mode)
    torch.cuda.synchronize()
    for _ in range(5): g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        for _ in range(10): g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / 10)
    ts.sort(); med = ts[len(ts)//2]
    # eager (no graph) for reference
    del g
    return med / N_CHAIN * 1e6, grid[0]

def dump_ptx():
    n = 4096
    dev = 'cuda'
    a = torch.randn(n, device=dev); b = torch.empty_like(a); w = torch.randn(n, device=dev)
    k = chain_kernel[(4,)](a, b, w, n, PDL=True, TRIG=True, BLOCK=1024, launch_pdl=True, num_warps=4)
    ptx = k.asm['ptx']
    lines = ptx.splitlines()
    for i, l in enumerate(lines):
        if 'griddepcontrol' in l:
            print(f"   ptx[{i}]: {l.strip()}")
    # show ordering of the two global loads relative to the wait
    idx = [(i, l.strip()) for i, l in enumerate(lines)
           if 'griddepcontrol' in l or ('ld.global' in l)]
    print("   load/gdc order:")
    for i, l in idx[:12]:
        print(f"     {i}: {l[:90]}")

if __name__ == '__main__':
    torch.cuda.init()
    print(f"chain={N_CHAIN}  gpu={torch.cuda.get_device_name(0)}")
    print("PTX check (PDL+TRIG):"); dump_ptx()
    shapes = [(4096, 4), (65536, 16), (1048576, 128)]
    res = {}
    for mode in (0, 1, 2, 3):
        for n, nb in shapes:
            us, gr = bench(n, nb, mode)
            res[(mode, n)] = us
            print(f"mode={mode} n={n:>8} grid={gr:>4}: {us:6.3f} us/kernel")
    print()
    for n, nb in shapes:
        base = res[(0, n)]
        print(f"n={n:>8}: base {base:6.3f}  |  " + "  ".join(
            f"m{m}: {res[(m,n)]:6.3f} ({base-res[(m,n)]:+.3f})" for m in (1,2,3)))
