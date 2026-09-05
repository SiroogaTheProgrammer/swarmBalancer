// Fast kernels must agree with the reference triple loop for every tile/edge
// combination (partial row tiles, partial column tiles, K blocking).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "swarm/matmul.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* what, int M, int N, int K) {
  if (!ok) {
    std::printf("FAIL %s M=%d N=%d K=%d\n", what, M, N, K);
    ++g_failures;
  }
}

void test_shape(int M, int N, int K, std::mt19937& rng) {
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  std::uniform_int_distribution<int> ui(-128, 127);

  std::vector<float> A((std::size_t)M * K), B((std::size_t)K * N), C1((std::size_t)M * N, 123.f), C2((std::size_t)M * N, -7.f);
  for (auto& v : A) v = uf(rng);
  for (auto& v : B) v = uf(rng);
  swarm::gemm_f32_naive(M, N, K, A.data(), K, B.data(), N, C1.data(), N);
  swarm::gemm_f32(M, N, K, A.data(), B.data(), C2.data());
  bool ok = true;
  for (std::size_t i = 0; i < C1.size(); ++i) {
    const float tol = 1e-4f * (1.0f + std::fabs(C1[i]));
    if (std::fabs(C1[i] - C2[i]) > tol) ok = false;
  }
  check(ok, "gemm_f32", M, N, K);

  std::vector<std::int8_t> A8((std::size_t)M * K), B8((std::size_t)K * N);
  std::vector<std::int32_t> D1((std::size_t)M * N, 5), D2((std::size_t)M * N, 9);
  for (auto& v : A8) v = static_cast<std::int8_t>(ui(rng));
  for (auto& v : B8) v = static_cast<std::int8_t>(ui(rng));
  swarm::gemm_i8_naive(M, N, K, A8.data(), K, B8.data(), N, D1.data(), N);
  swarm::gemm_i8(M, N, K, A8.data(), B8.data(), D2.data());
  check(D1 == D2, "gemm_i8", M, N, K);
}

void test_strided(std::mt19937& rng) {
  // Sub-matrix views: lda/ldb/ldc larger than the logical sizes.
  const int M = 5, N = 20, K = 33, lda = 40, ldb = 25, ldc = 31;
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  std::vector<float> A((std::size_t)M * lda), B((std::size_t)K * ldb), C1((std::size_t)M * ldc, 0.f), C2((std::size_t)M * ldc, 0.f);
  for (auto& v : A) v = uf(rng);
  for (auto& v : B) v = uf(rng);
  swarm::gemm_f32_naive(M, N, K, A.data(), lda, B.data(), ldb, C1.data(), ldc);
  swarm::gemm_f32(M, N, K, A.data(), lda, B.data(), ldb, C2.data(), ldc);
  bool ok = true;
  for (std::size_t i = 0; i < C1.size(); ++i)
    if (std::fabs(C1[i] - C2[i]) > 1e-4f * (1.0f + std::fabs(C1[i]))) ok = false;
  check(ok, "gemm_f32 strided", M, N, K);
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  const int Ms[] = {1, 2, 3, 4, 5, 8, 9, 16};
  const int Ns[] = {1, 7, 15, 16, 17, 32, 33, 100, 256};
  const int Ks[] = {1, 3, 9, 72, 255, 256, 257, 600};
  for (int M : Ms)
    for (int N : Ns)
      for (int K : Ks) test_shape(M, N, K, rng);
  test_strided(rng);
  test_shape(64, 64, 0, rng);  // K = 0 -> zeros

  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("test_matmul: all kernels match the reference\n");
  return 0;
}
