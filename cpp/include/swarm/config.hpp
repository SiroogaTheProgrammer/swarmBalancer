#pragma once
// Build-time configuration and small portability macros for the brain.

#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define SWARM_SIMD_NAME "neon"
#elif defined(__AVX512F__)
#define SWARM_SIMD_NAME "avx512"
#elif defined(__AVX2__)
#define SWARM_SIMD_NAME "avx2"
#elif defined(__SSE2__) || defined(_M_X64)
#define SWARM_SIMD_NAME "sse2"
#else
#define SWARM_SIMD_NAME "scalar"
#endif

#if defined(__clang__)
#define SWARM_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
#define SWARM_COMPILER "gcc " __VERSION__
#elif defined(_MSC_VER)
#define SWARM_COMPILER "msvc"
#else
#define SWARM_COMPILER "unknown"
#endif

#if defined(_MSC_VER)
#define SWARM_RESTRICT __restrict
#else
#define SWARM_RESTRICT __restrict__
#endif
