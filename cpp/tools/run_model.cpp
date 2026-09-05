// Runs a .swm brain on an input frame (raw float32 file) or on a synthetic
// frame, under an optional device RAM cap. Handy for checking a model on a
// target board without Python.
//
//   run_model model.swm [--input frame.f32] [--ram-cap BYTES] [--repeat N]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "swarm/config.hpp"
#include "swarm/model.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: run_model model.swm [--input frame.f32] [--ram-cap BYTES] [--repeat N]\n");
    return 2;
  }
  std::string model_path = argv[1];
  std::string input_path;
  std::size_t ram_cap = 0;
  int repeat = 1;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input_path = argv[++i];
    else if (!std::strcmp(argv[i], "--ram-cap") && i + 1 < argc) ram_cap = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }

  try {
    swarm::Model model = swarm::Model::load(model_path, ram_cap);
    const swarm::Shape in = model.input_shape();
    const swarm::Shape out = model.output_shape();
    std::printf("build      : %s / %s\n", SWARM_COMPILER, SWARM_SIMD_NAME);
    std::printf("model      : %s\n", model_path.c_str());
    std::printf("input      : %dx%dx%d   output: %dx%dx%d   layers: %zu\n", in.c, in.h, in.w, out.c, out.h, out.w,
                model.layers().size());
    std::printf("MACs/frame : %llu\n", static_cast<unsigned long long>(model.macs_per_run()));
    std::printf("weights    : %zu bytes\n", model.weights_bytes());
    std::printf("RAM needed : %zu bytes%s\n", model.required_bytes(),
                ram_cap ? (" (cap " + std::to_string(ram_cap) + ": FITS)").c_str() : "");
    for (const swarm::Layer& L : model.layers())
      std::printf("  %-10s %s -> %dx%dx%d  macs=%llu\n", swarm::layer_type_name(L.type),
                  L.dtype == swarm::DType::I8 ? "i8 " : "f32", L.out_shape.c, L.out_shape.h, L.out_shape.w,
                  static_cast<unsigned long long>(L.macs));

    std::vector<float> x(in.elems(), 0.0f), y(out.elems(), 0.0f);
    if (!input_path.empty()) {
      std::ifstream f(input_path, std::ios::binary);
      if (!f) {
        std::fprintf(stderr, "cannot open %s\n", input_path.c_str());
        return 1;
      }
      std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      if (raw.size() != x.size() * sizeof(float)) {
        std::fprintf(stderr, "input has %zu bytes, expected %zu\n", raw.size(), x.size() * sizeof(float));
        return 1;
      }
      std::memcpy(x.data(), raw.data(), raw.size());
    } else {
      for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i * 2654435761u) % 256) / 255.0f;
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) model.run(x.data(), y.data());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / repeat;

    std::size_t best = 0;
    for (std::size_t i = 1; i < y.size(); ++i)
      if (y[i] > y[best]) best = i;
    std::printf("host time  : %.4f ms/frame  (%.1f MMAC/s)\n", ms, model.macs_per_run() / ms * 1e-3);
    std::printf("output     :");
    for (std::size_t i = 0; i < y.size() && i < 16; ++i) std::printf(" %.4f", y[i]);
    std::printf("%s\n", y.size() > 16 ? " ..." : "");
    std::printf("argmax     : %zu\n", best);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
