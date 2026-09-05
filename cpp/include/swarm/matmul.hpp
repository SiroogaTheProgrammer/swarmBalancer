#pragma once
// Fast dense matrix multiplication kernels - the "multiplication tool" every
// brain layer is built on.
//
//   C[M,N] = A[M,K] * B[K,N]        (row-major, C is overwritten)
//
// Design notes (see docs/ARCHITECTURE.md, "Brain kernels"):
//  * register-tiled micro-kernel (MR=4 rows x NR=16 columns) - accumulators
//    live in SIMD registers for the whole K loop, so the inner loop does no
//    loads/stores on C at all;
//  * K is blocked so the B panel a tile streams through stays in L1;
//  * written as plain loops with fixed trip counts so clang/gcc auto-vectorise
//    it to NEON / SSE / AVX without any intrinsics -> the same source builds
//    for a Cortex-M, a Cortex-A, a satellite OBC or the host PC;
//  * int8 variant accumulates in int32 - this is what quantised layers use on
//    small devices (4x less weight memory, wider SIMD lanes).
// Layers are laid out so the *wide* dimension is N (e.g. output pixels of a
// conv, or output neurons of a dense layer) which is what the kernel
// vectorises along.

#include <cstdint>

namespace swarm {

// Straightforward triple loop. Reference for tests and the benchmark baseline.
void gemm_f32_naive(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_i8_naive(int M, int N, int K, const std::int8_t* A, int lda, const std::int8_t* B, int ldb, std::int32_t* C,
                   int ldc);

// Fast kernels.
void gemm_f32(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
void gemm_i8(int M, int N, int K, const std::int8_t* A, int lda, const std::int8_t* B, int ldb, std::int32_t* C, int ldc);

// Convenience overloads for packed (lda=K, ldb=N, ldc=N) matrices.
inline void gemm_f32(int M, int N, int K, const float* A, const float* B, float* C) { gemm_f32(M, N, K, A, K, B, N, C, N); }
inline void gemm_i8(int M, int N, int K, const std::int8_t* A, const std::int8_t* B, std::int32_t* C) {
  gemm_i8(M, N, K, A, K, B, N, C, N);
}

}  // namespace swarm
