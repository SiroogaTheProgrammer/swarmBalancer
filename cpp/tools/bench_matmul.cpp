// Benchmarks the multiplication kernels the brain is built on.
// Prints MAC throughput for the shapes the layers actually produce.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "swarm/config.hpp"
#include "swarm/matmul.hpp"

namespace {

using Clock = std::chrono::steady_clock;

template <class F>
double time_per_call_ms(F&& fn, double min_seconds = 0.3) {
  // warm-up
  fn();
  int iters = 1;
  for (;;) {
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    if (s >= min_seconds) return s * 1e3 / iters;
    iters *= 2;
  }
}

struct Case {
  const char* name;
  int M, N, K;
};

void run_case(const Case& c, std::mt19937& rng) {
  std::uniform_real_distribution<float> uf(-1.f, 1.f);
  std::uniform_int_distribution<int> ui(-127, 127);
  const std::size_t M = c.M, N = c.N, K = c.K;

  std::vector<float> A(M * K), B(K * N), C(M * N);
  std::vector<std::int8_t> A8(M * K), B8(K * N);
  std::vector<std::int32_t> C32(M * N);
  for (auto& v : A) v = uf(rng);
  for (auto& v : B) v = uf(rng);
  for (auto& v : A8) v = static_cast<std::int8_t>(ui(rng));
  for (auto& v : B8) v = static_cast<std::int8_t>(ui(rng));

  const double macs = static_cast<double>(M) * N * K;
  const double t_naive = time_per_call_ms([&] { swarm::gemm_f32_naive(c.M, c.N, c.K, A.data(), c.K, B.data(), c.N, C.data(), c.N); });
  const double t_fast = time_per_call_ms([&] { swarm::gemm_f32(c.M, c.N, c.K, A.data(), B.data(), C.data()); });
  const double t_i8 = time_per_call_ms([&] { swarm::gemm_i8(c.M, c.N, c.K, A8.data(), B8.data(), C32.data()); });

  std::printf("%-26s %5d %5d %5d | %9.3f %7.2f | %9.3f %7.2f | %9.3f %7.2f | %5.1fx\n", c.name, c.M, c.N, c.K, t_naive,
              macs / t_naive * 1e-6, t_fast, macs / t_fast * 1e-6, t_i8, macs / t_i8 * 1e-6, t_naive / t_fast);
}

}  // namespace

int main() {
  std::printf("swarm_core build: %s / %s\n\n", SWARM_COMPILER, SWARM_SIMD_NAME);
  std::printf("%-26s %5s %5s %5s | %9s %7s | %9s %7s | %9s %7s | %s\n", "case", "M", "N", "K", "naive ms", "GMAC/s",
              "f32 ms", "GMAC/s", "int8 ms", "GMAC/s", "speedup");
  std::printf("%s\n", std::string(116, '-').c_str());

  std::mt19937 rng(42);
  const Case cases[] = {
      {"conv1 8ch 3x3 @32x32", 8, 1024, 9},       // tiny_cnn layer 1 (im2col GEMM)
      {"conv2 16ch 3x3 @16x16", 16, 256, 72},     // tiny_cnn layer 2
      {"conv 32ch 3x3 @64x64", 32, 4096, 144},    // a "satellite" sized conv
      {"dense 1024->32 (gemv)", 1, 32, 1024},     // tiny_cnn dense 1
      {"dense 4096->256 (gemv)", 1, 256, 4096},
      {"square 128", 128, 128, 128},
      {"square 256", 256, 256, 256},
      {"square 512", 512, 512, 512},
  };
  for (const Case& c : cases) run_case(c, rng);
  return 0;
}
