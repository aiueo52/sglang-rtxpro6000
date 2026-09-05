# K1 PDL sweep — measured results (2026-09-05/06)

Branch opus/pdl-sweep, worktree /home/user/tools/sglang-pdl, base codex/perf-v1 ccf9d2faba.
Flag: SGLANG_TRITON_PDL=1 (default off). All A/B pairs are the same build in the same lock session.

## Toy (chain of 100 dependent Triton kernels under CUDA graph replay, us/kernel)
  grid    base   +attr    +attr+wait   +attr+wait+trigger
     4   1.137   0.726*      0.727 (+0.41)   0.677 (+0.46)
    16   1.953   1.617*      2.254 (-0.30)   2.118 (-0.17)
   128   3.113   2.773*      3.698 (-0.59)   3.901 (-0.79)
  * attribute-only is INCORRECT (no gdc_wait -> not a dependency chain); upper bound only.

## W4 prof A/B (pdl0 control -> pdl1)
  code-edit  wall 11.26 -> 10.63 ms (-0.63)  busy 11.00 -> 10.39  verify 9.99 -> 9.38  idle 0.14 -> 0.09
  prose-en   wall 10.86 -> 10.30 ms (-0.56)  busy 10.39 -> 10.07  verify 9.38 -> 9.06  idle 0.14 -> 0.09

## W16 prof A/B (pdl0 -> pdl1) — neutral
  code-edit  wall 17.67 -> 17.18 (-0.49)  busy 17.40 -> 17.16  verify 12.87 -> 12.64
  prose-en   wall 17.17 -> 17.55 (+0.38)  busy 16.94 -> 17.18  verify 12.52 -> 12.73

## Key per-kernel medians (W4 code-edit)
  _w8a16_gemv_silu_kernel  8.90 -> 6.27 us (-2.62, -181 us/step)   real win, same grid [20,5,1]
  _w8a16_gemv_kernel       5.82 -> 5.76 us (-182 us/step)
  _fused_sigmoid_mul_kernel 1.02 -> 16.78 us  <- ARTIFACT: starts 15.5us before its
      predecessor ends (300/300 launches), so the spin-wait is inside its measured duration.
      busy_ms (interval union) is the correct metric; per-kernel medians are not comparable
      across the flag for kernels that overlap their predecessor.

## Determinism control (why full-step greedy equality cannot be shown)
  OFF vs OFF, same server process:  identical on 1/4 workloads
  OFF vs OFF, server restart:       identical on 1/4
  OFF vs ON:                        identical on 1/4   <- same as baseline, no PDL signal
  Source: _hc_down_kernel accumulates t_raw with device-scope atomics ("summation order not
  reproducible across launches", hc_mix2_triton docstring).
