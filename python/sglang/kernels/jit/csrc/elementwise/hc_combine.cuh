#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

namespace sglang {

struct HcCombineParams {
  const void* block_output;     // [M, H]
  const void* residual;         // [M, HC * H]
  const void* normed_residual;  // [M, HC * H]
  const void* inject_weight;    // [HC, HC * H]
  void* output;                 // [M, HC * H]
};

/**
 * \brief Fused HyperConnection (gated residual) combine:
 *
 *   a[m, c] = 2 * sigmoid(dot(normed_residual[m, :], inject_weight[c, :]) / kHcCount)
 *   output[m, c*H + i] = residual[m, c*H + i] + a[m, c] * block_output[m, i]
 *
 * One CTA handles one token row. Phase 1 computes the kHcCount gate values with a
 * block reduction over the full HC*H row (fp32 accumulation). Phase 2 streams the
 * HC*H output elements with vectorized 16B accesses; the block_output row is only
 * H elements, so its re-read per branch stays in L2.
 *
 * \tparam kHcCount    Number of hyper-connection branches (4 in production).
 * \tparam kHiddenSize Per-branch hidden size H. HC*H must be a multiple of
 *                     kNumThreads * kVecLen so the row maps exactly onto the CTA.
 * \tparam kUsePDL     Whether to emit the PDL wait/trigger pair.
 * \tparam Float       Element type: bf16_t | fp16_t.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(256) void hc_combine_kernel(const HcCombineParams __grid_constant__ params) {
  using namespace device;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;  // 8 elements, 16 bytes
  constexpr uint32_t kVecLen = 8;
  constexpr uint32_t kNumThreads = 256;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;            // 1280 for 4x2560
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kNumThreads;  // 5 for 4x2560
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;      // 320 for 2560
  constexpr uint32_t kNumWarps = kNumThreads / kWarpThreads;

  const auto gmem = tile::Memory<Storage>::cta(kNumThreads);
  const uint32_t m = blockIdx.x;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  PDLWaitPrimary<kUsePDL>();

  // Phase 1: gate values a_c = 2 * sigmoid(dot(N[m, :], W[c, :]) / HC), fp32.
  // Each thread accumulates its kVecsPerThread strided vectors against all
  // kHcCount weight rows, then the partials are reduced across the CTA.
  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j] = gmem.load(n_ptr, j);
  }

  float acc[kHcCount];
#pragma unroll
  for (int c = 0; c < kHcCount; ++c) {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      const Storage w_vec = gmem.load(wc_ptr, j);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    acc[c] = warp::reduce_sum(sum);
  }

  __shared__ float smem[kHcCount][kNumWarps];
  const uint32_t warp_id = threadIdx.x / kWarpThreads;
  const uint32_t lane = threadIdx.x % kWarpThreads;
  if (lane == 0) {
#pragma unroll
    for (int c = 0; c < kHcCount; ++c) {
      smem[c][warp_id] = acc[c];
    }
  }
  __syncthreads();
  __shared__ float a_shared[kHcCount];
  if (threadIdx.x < kHcCount) {
    float total = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      total += smem[threadIdx.x][w];
    }
    a_shared[threadIdx.x] = 2.0f / (1.0f + math::exp(-total / kHcCount));
  }
  __syncthreads();

  // Phase 2: stream the output row. Vector `vec_idx` lies entirely inside
  // branch `vec_idx / kVecsPerBranch` (H is a multiple of kVecLen).
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = threadIdx.x + j * kNumThreads;
    const uint32_t branch = vec_idx / kVecsPerBranch;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch) * kVecLen;
    const float a = a_shared[branch];

    const Storage r_vec = gmem.load(r_ptr, j);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch / kVecLen);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    gmem.store(out_ptr, out_vec, j);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static_assert(kHcCount > 0, "kHcCount must be positive");
  static_assert(kHiddenSize > 0 && kHiddenSize % 8 == 0, "kHiddenSize must be a multiple of 8");
  static_assert((kHcCount * kHiddenSize) % (256 * 8) == 0, "kHcCount * kHiddenSize must be a multiple of 2048");
  static constexpr auto kernel = hc_combine_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr uint32_t kBlockSize = 256;

  /**
   * \brief Validate tensors and launch one CTA per token row.
   * \param block_output    [M, H] contiguous
   * \param residual        [M, HC * H] contiguous
   * \param normed_residual [M, HC * H] contiguous, same dtype/device as residual
   * \param inject_weight   [HC, HC * H] contiguous, same dtype/device as residual
   * \param output          [M, HC * H] contiguous, same shape/dtype/device as residual
   */
  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView output) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize})  // block_output
        .with_dtype<DType>()
        .with_device(device)
        .verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})  // residual, normed_residual, output
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(normed_residual)
        .verify(output);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize})  // inject_weight
        .with_dtype<DType>()
        .with_device(device)
        .verify(inject_weight);

    const auto params = HcCombineParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .output = output.data_ptr(),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(num_tokens, kBlockSize, device.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

struct HcCombineSplitParams {
  const void* block_output;
  const void* residual;
  const void* normed_residual;
  const void* inject_weight;
  void* output;
  float* partials;
};

namespace hc_combine_split_detail {

// One CTA per warp of the original kernel: CTA s replays exactly the work of
// threads [32s, 32s+32) of the 256-thread reference, so every float is
// accumulated in the reference order and results stay bit-identical.
constexpr uint32_t kSplit = 8;
constexpr uint32_t kRefThreads = 256;
constexpr uint32_t kGateThreads = 32;
constexpr uint32_t kApplyThreads = 160;
constexpr uint32_t kVecLen = 8;

}  // namespace hc_combine_split_detail

/**
 * \brief Stage 1 of the split combine: partial gate dots over a K slice.
 *
 * Grid is (rows, kSplit) so the [HC, HC*H] inject weight is read once across
 * the whole grid instead of once per row, and the row's traffic is spread over
 * kSplit CTAs. Each CTA writes its own partials slot, so no atomics and no
 * buffer clearing are needed.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(hc_combine_split_detail::kGateThreads) void hc_combine_gate_kernel(
    const HcCombineSplitParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_split_detail;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr uint32_t kVecLen = 8;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerThread = kVecsPerRow / kRefThreads;
  static_assert(kVecsPerRow % kRefThreads == 0);
  static_assert(kRefThreads / kGateThreads == kSplit);

  const uint32_t m = blockIdx.x;
  const uint32_t split = blockIdx.y / kHcCount;
  const uint32_t c = blockIdx.y % kHcCount;
  const uint32_t ref_tid = split * kGateThreads + threadIdx.x;

  const auto n_ptr = pointer::offset<Float>(params.normed_residual, static_cast<int64_t>(m) * kRowSize);
  const auto w_ptr = static_cast<const Float*>(params.inject_weight);

  PDLWaitPrimary<kUsePDL>();

  Storage n_vec[kVecsPerThread];
#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    n_vec[j].load(n_ptr, ref_tid + j * kRefThreads);
  }

  {
    const auto wc_ptr = w_ptr + static_cast<int64_t>(c) * kRowSize;
    float sum = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecsPerThread; ++j) {
      Storage w_vec;
      w_vec.load(wc_ptr, ref_tid + j * kRefThreads);
#pragma unroll
      for (uint32_t i = 0; i < kVecLen / 2; ++i) {
        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);
        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);
        sum += nx * wx + ny * wy;
      }
    }
    sum = warp::reduce_sum(sum);
    if (threadIdx.x == 0) {
      params.partials[(static_cast<int64_t>(m) * kSplit + split) * kHcCount + c] = sum;
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief Stage 2: reduce the partial dots and stream the combined row.
 *
 * Each CTA owns one contiguous vector slice of the row. kSplit divides
 * kHiddenSize, so a slice never straddles two branches and the gate is a
 * per-CTA scalar.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float>
__global__ __launch_bounds__(hc_combine_split_detail::kApplyThreads) void hc_combine_apply_kernel(
    const HcCombineSplitParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_split_detail;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerSplit = kVecsPerRow / kSplit;
  constexpr uint32_t kVecsPerThread = kVecsPerSplit / kApplyThreads;
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;
  static_assert(kVecsPerBranch % kVecsPerSplit == 0);

  const uint32_t m = blockIdx.x;
  const uint32_t split = blockIdx.y;
  const uint32_t vec_base = split * kVecsPerSplit;
  const uint32_t branch = vec_base / kVecsPerBranch;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  PDLWaitPrimary<kUsePDL>();

  float total = 0.0f;
#pragma unroll
  for (uint32_t s = 0; s < kSplit; ++s) {
    total += params.partials[(static_cast<int64_t>(m) * kSplit + s) * kHcCount + branch];
  }
  const float a = 2.0f / (1.0f + math::exp(-total / kHcCount));

#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = vec_base + threadIdx.x + j * kApplyThreads;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch);
    Storage r_vec;
    r_vec.load(r_ptr, vec_idx);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch);
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    out_vec.store(out_ptr, vec_idx);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType>
struct HcCombineSplitKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static constexpr auto gate_kernel = hc_combine_gate_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;
  static constexpr auto apply_kernel = hc_combine_apply_kernel<kHcCount, kHiddenSize, kUsePDL, DType>;

  /// Gate stage only: the split partial dots, for a caller that runs the apply
  /// stage later (SGLANG_HC_GATE_EARLY). Bit-identical to `run`'s first launch.
  static void
  run_gate(
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView partials) {
    using namespace host;
    using namespace hc_combine_split_detail;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(normed_residual);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(inject_weight);
    auto part_rows = SymbolicSize{"partial_rows"};
    TensorMatcher({part_rows, kSplit, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(partials);

    const auto params = HcCombineSplitParams{
        .block_output = nullptr,
        .residual = nullptr,
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .output = nullptr,
        .partials = static_cast<float*>(partials.data_ptr()),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(dim3(num_tokens, kSplit * kHcCount, 1), kGateThreads, device.unwrap())
        .enable_pdl(kUsePDL)(gate_kernel, params);
  }

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView normed_residual,
      const tvm::ffi::TensorView inject_weight,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView partials) {
    using namespace host;
    using namespace hc_combine_split_detail;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(normed_residual)
        .verify(output);
    TensorMatcher({kHcCount, kHcCount * kHiddenSize}).with_dtype<DType>().with_device(device).verify(inject_weight);
    auto part_rows = SymbolicSize{"partial_rows"};
    TensorMatcher({part_rows, kSplit, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(partials);

    const auto params = HcCombineSplitParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .normed_residual = normed_residual.data_ptr(),
        .inject_weight = inject_weight.data_ptr(),
        .output = output.data_ptr(),
        .partials = static_cast<float*>(partials.data_ptr()),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(dim3(num_tokens, kSplit * kHcCount, 1), kGateThreads, device.unwrap())
        .enable_pdl(kUsePDL)(gate_kernel, params);
    LaunchKernel(dim3(num_tokens, kSplit, 1), kApplyThreads, device.unwrap()).enable_pdl(kUsePDL)(apply_kernel, params);
  }
};

struct HcCombineApplyParams {
  const void* block_output;     // [M, H]  routed / block output
  const void* residual;         // [M, HC * H]
  const void* shared_output;    // [M, H], only read when kUseShared
  const float* shared_gate;     // [M],    only read when kUseShared
  const float* partials;        // [M, kNumPartials, HC]
  void* output;                 // [M, HC * H]
};

/**
 * \brief Apply-only stage of the HC combine, for a gate computed earlier.
 *
 * Same geometry and math as ``hc_combine_apply_kernel``, except that the number
 * of gate partial slots is a template parameter: the standalone gate kernel
 * emits kSplit=8 of them, while the HC-mix K0 epilogue (SGLANG_HC_GATE_EARLY)
 * emits one per branch CTA, i.e. kHcCount.
 *
 * With kUseShared the shared-expert join is folded in as well
 * (SGLANG_SHARED_GATE_EARLY):
 *
 *   y[m, i] = Float(routed[m, i] + gate[m] * shared[m, i])
 *   out[m, c*H + i] = residual[m, c*H + i] + a[m, c] * y[m, i]
 *
 * The intermediate rounding to ``Float`` reproduces the bf16 store that
 * ``fused_gate_sigmoid_mul_add`` performs before the combine reads it, so the
 * folded path is bit-identical to the separate-kernel one.
 */
template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename Float, bool kUseShared, int kNumPartials>
__global__ __launch_bounds__(hc_combine_split_detail::kApplyThreads) void hc_combine_apply2_kernel(
    const HcCombineApplyParams __grid_constant__ params) {
  using namespace device;
  using namespace hc_combine_split_detail;
  using Float2 = packed_t<Float>;
  using Storage = AlignedVector<Float2, 4>;
  constexpr int64_t kRowSize = kHcCount * kHiddenSize;
  constexpr uint32_t kVecsPerRow = kRowSize / kVecLen;
  constexpr uint32_t kVecsPerSplit = kVecsPerRow / kSplit;
  constexpr uint32_t kVecsPerThread = kVecsPerSplit / kApplyThreads;
  constexpr uint32_t kVecsPerBranch = kHiddenSize / kVecLen;
  static_assert(kVecsPerBranch % kVecsPerSplit == 0);

  const uint32_t m = blockIdx.x;
  const uint32_t split = blockIdx.y;
  const uint32_t vec_base = split * kVecsPerSplit;
  const uint32_t branch = vec_base / kVecsPerBranch;

  const auto y_ptr = pointer::offset<Float>(params.block_output, static_cast<int64_t>(m) * kHiddenSize);
  const auto r_ptr = pointer::offset<Float>(params.residual, static_cast<int64_t>(m) * kRowSize);
  const auto out_ptr = pointer::offset<Float>(params.output, static_cast<int64_t>(m) * kRowSize);

  PDLWaitPrimary<kUsePDL>();

  float total = 0.0f;
#pragma unroll
  for (int s = 0; s < kNumPartials; ++s) {
    total += params.partials[(static_cast<int64_t>(m) * kNumPartials + s) * kHcCount + branch];
  }
  const float a = 2.0f / (1.0f + math::exp(-total / kHcCount));

  float g = 0.0f;
  const void* s_ptr = nullptr;
  if constexpr (kUseShared) {
    g = params.shared_gate[m];
    s_ptr = pointer::offset<Float>(params.shared_output, static_cast<int64_t>(m) * kHiddenSize);
  }

#pragma unroll
  for (uint32_t j = 0; j < kVecsPerThread; ++j) {
    const uint32_t vec_idx = vec_base + threadIdx.x + j * kApplyThreads;
    const uint32_t col_in_branch = (vec_idx % kVecsPerBranch);
    Storage r_vec;
    r_vec.load(r_ptr, vec_idx);
    Storage y_vec;
    y_vec.load(y_ptr, col_in_branch);
    Storage s_vec;
    if constexpr (kUseShared) {
      s_vec.load(s_ptr, col_in_branch);
    }
    Storage out_vec;
#pragma unroll
    for (uint32_t i = 0; i < kVecLen / 2; ++i) {
      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);
      auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);
      if constexpr (kUseShared) {
        const auto [sx, sy] = cast<fp32x2_t>(s_vec[i]);
        // Round the routed+shared sum to Float exactly like the standalone
        // fused_gate_sigmoid_mul_add store does, then widen again.
        const Float2 rounded = cast<Float2>(fp32x2_t{yx + g * sx, yy + g * sy});
        const auto [wx, wy] = cast<fp32x2_t>(rounded);
        yx = wx;
        yy = wy;
      }
      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});
    }
    out_vec.store(out_ptr, vec_idx);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int64_t kHcCount, int64_t kHiddenSize, bool kUsePDL, typename DType, bool kUseShared, int kNumPartials>
struct HcCombineApplyKernel {
  static_assert(sizeof(DType) == 2, "HcCombine only supports 2-byte dtypes");
  static constexpr auto kernel =
      hc_combine_apply2_kernel<kHcCount, kHiddenSize, kUsePDL, DType, kUseShared, kNumPartials>;

  /// Apply without the shared-expert fold; `block_output` is the finished block output.
  static void
  run_plain(const tvm::ffi::TensorView block_output,
            const tvm::ffi::TensorView residual,
            const tvm::ffi::TensorView partials,
            const tvm::ffi::TensorView output) {
    static_assert(!kUseShared, "run_plain requires kUseShared == false");
    run(block_output, residual, block_output, partials, partials, output);
  }

  static void
  run(const tvm::ffi::TensorView block_output,
      const tvm::ffi::TensorView residual,
      const tvm::ffi::TensorView shared_output,
      const tvm::ffi::TensorView shared_gate,
      const tvm::ffi::TensorView partials,
      const tvm::ffi::TensorView output) {
    using namespace host;
    using namespace hc_combine_split_detail;
    auto M = SymbolicSize{"num_tokens"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(block_output);
    TensorMatcher({M, kHcCount * kHiddenSize})
        .with_dtype<DType>()
        .with_device(device)
        .verify(residual)
        .verify(output);
    auto part_rows = SymbolicSize{"partial_rows"};
    TensorMatcher({part_rows, kNumPartials, kHcCount}).with_dtype<fp32_t>().with_device(device).verify(partials);
    if constexpr (kUseShared) {
      TensorMatcher({M, kHiddenSize}).with_dtype<DType>().with_device(device).verify(shared_output);
      auto gate_rows = SymbolicSize{"gate_rows"};
      TensorMatcher({gate_rows}).with_dtype<fp32_t>().with_device(device).verify(shared_gate);
    }

    const auto params = HcCombineApplyParams{
        .block_output = block_output.data_ptr(),
        .residual = residual.data_ptr(),
        .shared_output = kUseShared ? shared_output.data_ptr() : nullptr,
        .shared_gate = kUseShared ? static_cast<const float*>(shared_gate.data_ptr()) : nullptr,
        .partials = static_cast<const float*>(partials.data_ptr()),
        .output = output.data_ptr(),
    };

    const auto num_tokens = static_cast<uint32_t>(M.unwrap());
    LaunchKernel(dim3(num_tokens, kSplit, 1), kApplyThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
