#include "swarm/swarm_c_api.h"

#include <cstring>
#include <exception>
#include <string>

#include "swarm/config.hpp"
#include "swarm/matmul.hpp"
#include "swarm/model.hpp"

struct swm_model {
  swarm::Model impl;
};

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& msg) { g_last_error = msg; }

}  // namespace

extern "C" {

const char* swm_build_info(void) { return SWARM_COMPILER " / " SWARM_SIMD_NAME; }

const char* swm_last_error(void) { return g_last_error.c_str(); }

swm_model* swm_load(const char* path, size_t ram_cap_bytes) {
  if (!path) {
    set_error("path is NULL");
    return nullptr;
  }
  try {
    return new swm_model{swarm::Model::load(path, ram_cap_bytes)};
  } catch (const std::exception& e) {
    set_error(e.what());
    return nullptr;
  }
}

void swm_free(swm_model* m) { delete m; }

int swm_input_shape(const swm_model* m, int* c, int* h, int* w) {
  if (!m || !c || !h || !w) return SWM_ERR_ARGS;
  const swarm::Shape& s = m->impl.input_shape();
  *c = s.c;
  *h = s.h;
  *w = s.w;
  return SWM_OK;
}

int swm_output_shape(const swm_model* m, int* c, int* h, int* w) {
  if (!m || !c || !h || !w) return SWM_ERR_ARGS;
  const swarm::Shape& s = m->impl.output_shape();
  *c = s.c;
  *h = s.h;
  *w = s.w;
  return SWM_OK;
}

int swm_num_layers(const swm_model* m) { return m ? static_cast<int>(m->impl.layers().size()) : SWM_ERR_ARGS; }

int swm_run(swm_model* m, const float* input, size_t input_len, float* output, size_t output_len) {
  if (!m || !input || !output) {
    set_error("null argument");
    return SWM_ERR_ARGS;
  }
  if (input_len != m->impl.input_shape().elems() || output_len != m->impl.output_shape().elems()) {
    set_error("input/output length mismatch: expected " + std::to_string(m->impl.input_shape().elems()) + " / " +
              std::to_string(m->impl.output_shape().elems()));
    return SWM_ERR_ARGS;
  }
  try {
    m->impl.run(input, output);
    return SWM_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return SWM_ERR_RUN;
  }
}

uint64_t swm_macs_per_run(const swm_model* m) { return m ? m->impl.macs_per_run() : 0; }
size_t swm_weights_bytes(const swm_model* m) { return m ? m->impl.weights_bytes() : 0; }
size_t swm_required_bytes(const swm_model* m) { return m ? m->impl.required_bytes() : 0; }

void swm_gemm_f32(int M, int N, int K, const float* A, const float* B, float* C) { swarm::gemm_f32(M, N, K, A, B, C); }
void swm_gemm_f32_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  swarm::gemm_f32_naive(M, N, K, A, K, B, N, C, N);
}
void swm_gemm_i8(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C) { swarm::gemm_i8(M, N, K, A, B, C); }
void swm_gemm_i8_naive(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C) {
  swarm::gemm_i8_naive(M, N, K, A, K, B, N, C, N);
}

}  // extern "C"
