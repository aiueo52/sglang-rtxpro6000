"""API + graph-capture + correctness check for Triton PDL (no timing)."""
import torch, triton, triton.language as tl
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents

@triton.jit
def chain_kernel(X, Y, W, n, PDL: tl.constexpr, TRIG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    w = tl.load(W + offs, mask=m, other=0.0)     # independent prologue
    if PDL:
        gdc_wait()
    x = tl.load(X + offs, mask=m, other=0.0)     # dependent load
    tl.store(Y + offs, x * 1.0009765625 + w * 0.0, mask=m)
    if TRIG:
        gdc_launch_dependents()

n, BLOCK = 1 << 20, 1024
grid = (triton.cdiv(n, BLOCK),)
dev = 'cuda'
torch.manual_seed(0)
x0 = torch.randn(n, device=dev)
W = torch.randn(n, device=dev)

def chain(bufs, mode, k=64):
    for i in range(k):
        chain_kernel[grid](bufs[i % 2], bufs[(i + 1) % 2], W, n,
                           PDL=(mode >= 2), TRIG=(mode >= 3), BLOCK=BLOCK,
                           launch_pdl=(mode >= 1), num_warps=4)

# PTX inspection
k = chain_kernel[grid](x0, torch.empty_like(x0), W, n, PDL=True, TRIG=True,
                       BLOCK=BLOCK, launch_pdl=True, num_warps=4)
ptx = k.asm['ptx'].splitlines()
seq = [(i, l.strip()) for i, l in enumerate(ptx)
       if 'griddepcontrol' in l or 'ld.global' in l or 'st.global' in l]
print("PTX order (loads / stores / griddepcontrol):")
for i, l in seq[:16]:
    print(f"  {i:5d}  {l[:80]}")
print()

# correctness under graph capture
for mode in (0, 1, 2, 3):
    bufs = [x0.clone(), torch.empty_like(x0)]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain(bufs, mode)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    for b in bufs: b.copy_(x0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain(bufs, mode)
    bufs[0].copy_(x0); bufs[1].zero_()
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    out = bufs[0]  # 64 kernels, even -> result back in bufs[0]
    ref = x0 * (1.0009765625 ** 64)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"mode={mode} graph capture OK, max_abs_err={err:.3e}")
print("\ncapability:", torch.cuda.get_device_capability(), torch.version.cuda)
