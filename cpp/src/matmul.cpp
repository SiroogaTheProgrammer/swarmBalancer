#include "swarm/matmul.hpp"

#include <algorithm>
#include <cstring>

#include "swarm/config.hpp"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SWARM_HAVE_NEON 1
#else
#define SWARM_HAVE_NEON 0
#endif

namespace swarm {

// ----------------------------------------------------------------------------
// Reference kernels
// ----------------------------------------------------------------------------

void gemm_f32_naive(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int k = 0; k < K; ++k) s += A[(std::size_t)i * lda + k] * B[(std::size_t)k * ldb + j];
      C[(std::size_t)i * ldc + j] = s;
    }
  }
}

void gemm_i8_naive(int M, int N, int K, const std::int8_t* A, int lda, const std::int8_t* B, int ldb, std::int32_t* C,
                   int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::int32_t s = 0;
      for (int k = 0; k < K; ++k)
        s += static_cast<std::int32_t>(A[(std::size_t)i * lda + k]) * static_cast<std::int32_t>(B[(std::size_t)k * ldb + j]);
      C[(std::size_t)i * ldc + j] = s;
    }
  }
}

// ----------------------------------------------------------------------------
// Fast kernels
// ----------------------------------------------------------------------------

namespace {

constexpr int MR = 4;    // rows of C per micro-kernel
constexpr int NR = 16;   // columns of C per micro-kernel = one 64-byte cache line of f32
constexpr int KC = 256;  // K block: the B panel (KC x NR) a tile streams through is 16 KB -> L1 resident

// Portable ROWS x NR tile of C over a K block. Accumulators are a fixed-size
// local array so the compiler keeps them in vector registers and vectorises
// along the columns (this is what happens for f32 with clang/gcc -O3).
template <class TA, class TAcc, int ROWS>
inline void micro_kernel_portable(int kc, const TA* SWARM_RESTRICT A, int lda, const TA* SWARM_RESTRICT B, int ldb,
                                  TAcc* SWARM_RESTRICT C, int ldc, bool accumulate) {
  TAcc acc[ROWS][NR];
  for (int r = 0; r < ROWS; ++r)
    for (int c = 0; c < NR; ++c) acc[r][c] = TAcc(0);

  for (int k = 0; k < kc; ++k) {
    const TA* SWARM_RESTRICT b = B + (std::size_t)k * ldb;
    for (int r = 0; r < ROWS; ++r) {
      const TAcc a = static_cast<TAcc>(A[(std::size_t)r * lda + k]);
      for (int c = 0; c < NR; ++c) acc[r][c] += a * static_cast<TAcc>(b[c]);
    }
  }

  for (int r = 0; r < ROWS; ++r) {
    TAcc* SWARM_RESTRICT crow = C + (std::size_t)r * ldc;
    if (accumulate) {
      for (int c = 0; c < NR; ++c) crow[c] += acc[r][c];
    } else {
      for (int c = 0; c < NR; ++c) crow[c] = acc[r][c];
    }
  }
}

#if SWARM_HAVE_NEON
// int8 x int8 -> int32 tile using widening multiply-accumulate (smlal).
// The auto-vectoriser turns the portable int8 loop into per-output dot
// products with scalar gathers (~10x slower than f32); this spells out the
// intended column-parallel shape: 4 MACs per instruction, 16 accumulator
// registers for the 4x16 tile.
template <int ROWS>
inline void micro_kernel_i8_neon(int kc, const std::int8_t* SWARM_RESTRICT A, int lda, const std::int8_t* SWARM_RESTRICT B,
                                 int ldb, std::int32_t* SWARM_RESTRICT C, int ldc, bool accumulate) {
  static_assert(NR == 16, "NEON int8 kernel assumes a 16-column tile");
  int32x4_t acc[ROWS][4];
  for (int r = 0; r < ROWS; ++r)
    for (int q = 0; q < 4; ++q) acc[r][q] = vdupq_n_s32(0);

  for (int k = 0; k < kc; ++k) {
    const int8x16_t b8 = vld1q_s8(B + (std::size_t)k * ldb);
    const int16x8_t b_lo = vmovl_s8(vget_low_s8(b8));
    const int16x8_t b_hi = vmovl_s8(vget_high_s8(b8));
    const int16x4_t b0 = vget_low_s16(b_lo), b1 = vget_high_s16(b_lo);
    const int16x4_t b2 = vget_low_s16(b_hi), b3 = vget_high_s16(b_hi);
    for (int r = 0; r < ROWS; ++r) {
      const std::int16_t a = A[(std::size_t)r * lda + k];
      acc[r][0] = vmlal_n_s16(acc[r][0], b0, a);
      acc[r][1] = vmlal_n_s16(acc[r][1], b1, a);
      acc[r][2] = vmlal_n_s16(acc[r][2], b2, a);
      acc[r][3] = vmlal_n_s16(acc[r][3], b3, a);
    }
  }

  for (int r = 0; r < ROWS; ++r) {
    std::int32_t* SWARM_RESTRICT crow = C + (std::size_t)r * ldc;
    for (int q = 0; q < 4; ++q) {
      int32x4_t v = acc[r][q];
      if (accumulate) v = vaddq_s32(v, vld1q_s32(crow + 4 * q));
      vst1q_s32(crow + 4 * q, v);
    }
  }
}
#endif

struct KernelF32 {
  template <int ROWS>
  static void run(int kc, const float* A, int lda, const float* B, int ldb, float* C, int ldc, bool accumulate) {
    micro_kernel_portable<float, float, ROWS>(kc, A, lda, B, ldb, C, ldc, accumulate);
  }
};

struct KernelI8 {
  template <int ROWS>
  static void run(int kc, const std::int8_t* A, int lda, const std::int8_t* B, int ldb, std::int32_t* C, int ldc,
                  bool accumulate) {
#if SWARM_HAVE_NEON
    micro_kernel_i8_neon<ROWS>(kc, A, lda, B, ldb, C, ldc, accumulate);
#else
    micro_kernel_portable<std::int8_t, std::int32_t, ROWS>(kc, A, lda, B, ldb, C, ldc, accumulate);
#endif
  }
};

// Columns that do not fill a whole NR tile (N % 16). Rare in practice
// because layers are laid out to keep N wide; plain scalar loop.
template <class TA, class TAcc>
inline void edge_columns(int M, int j0, int N, int kc, const TA* A, int lda, const TA* B, int ldb, TAcc* C, int ldc,
                         bool accumulate) {
  for (int i = 0; i < M; ++i) {
    for (int j = j0; j < N; ++j) {
      TAcc s = TAcc(0);
      for (int k = 0; k < kc; ++k)
        s += static_cast<TAcc>(A[(std::size_t)i * lda + k]) * static_cast<TAcc>(B[(std::size_t)k * ldb + j]);
      TAcc& c = C[(std::size_t)i * ldc + j];
      c = accumulate ? c + s : s;
    }
  }
}

template <class TA, class TAcc, class Kern>
void gemm_tiled(int M, int N, int K, const TA* A, int lda, const TA* B, int ldb, TAcc* C, int ldc) {
  if (M <= 0 || N <= 0) return;
  if (K <= 0) {
    for (int i = 0; i < M; ++i) std::memset(C + (std::size_t)i * ldc, 0, sizeof(TAcc) * N);
    return;
  }

  const int n_full = N - (N % NR);

  for (int k0 = 0; k0 < K; k0 += KC) {
    const int kc = std::min(KC, K - k0);
    const bool accumulate = k0 > 0;
    const TA* Ak = A + k0;
    const TA* Bk = B + (std::size_t)k0 * ldb;

    for (int j = 0; j < n_full; j += NR) {
      int i = 0;
      for (; i + MR <= M; i += MR)
        Kern::template run<MR>(kc, Ak + (std::size_t)i * lda, lda, Bk + j, ldb, C + (std::size_t)i * ldc + j, ldc,
                               accumulate);
      for (; i < M; ++i)
        Kern::template run<1>(kc, Ak + (std::size_t)i * lda, lda, Bk + j, ldb, C + (std::size_t)i * ldc + j, ldc,
                              accumulate);
    }
    if (n_full < N) edge_columns<TA, TAcc>(M, n_full, N, kc, Ak, lda, Bk, ldb, C, ldc, accumulate);
  }
}

}  // namespace

void gemm_f32(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
  gemm_tiled<float, float, KernelF32>(M, N, K, A, lda, B, ldb, C, ldc);
}

void gemm_i8(int M, int N, int K, const std::int8_t* A, int lda, const std::int8_t* B, int ldb, std::int32_t* C, int ldc) {
  gemm_tiled<std::int8_t, std::int32_t, KernelI8>(M, N, K, A, lda, B, ldb, C, ldc);
}

}  // namespace swarm
