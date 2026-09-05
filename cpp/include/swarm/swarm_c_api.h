#pragma once
// Plain C ABI of the brain, consumed from Python via ctypes
// (python/swarm/brain/native.py) and usable from any other language / RTOS.
//
// All functions returning `int` return 0 on success and a negative code on
// failure; the message is available from swm_last_error().

#include <stddef.h>
#include <stdint.h>

#if defined(SWARM_BUILDING_DLL) && defined(_WIN32)
#define SWM_API __declspec(dllexport)
#elif defined(SWARM_BUILDING_DLL) && defined(__GNUC__)
#define SWM_API __attribute__((visibility("default")))
#else
#define SWM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct swm_model swm_model;

enum swm_status { SWM_OK = 0, SWM_ERR_ARGS = -1, SWM_ERR_LOAD = -2, SWM_ERR_RAM_CAP = -3, SWM_ERR_RUN = -4 };

/* Compiler + SIMD flavour this library was built with, e.g. "clang 21 / neon". */
SWM_API const char* swm_build_info(void);
/* Message of the last failed call on this thread. */
SWM_API const char* swm_last_error(void);

/* Loads a .swm model. ram_cap_bytes = 0 -> unlimited. Returns NULL on failure. */
SWM_API swm_model* swm_load(const char* path, size_t ram_cap_bytes);
SWM_API void swm_free(swm_model* m);

SWM_API int swm_input_shape(const swm_model* m, int* c, int* h, int* w);
SWM_API int swm_output_shape(const swm_model* m, int* c, int* h, int* w);
SWM_API int swm_num_layers(const swm_model* m);

/* Runs one forward pass. input_len / output_len are element counts and are checked. */
SWM_API int swm_run(swm_model* m, const float* input, size_t input_len, float* output, size_t output_len);

SWM_API uint64_t swm_macs_per_run(const swm_model* m);
SWM_API size_t swm_weights_bytes(const swm_model* m);
/* Weights + activations + scratch: the RAM a target device must provide. */
SWM_API size_t swm_required_bytes(const swm_model* m);

/* Raw kernels (packed row-major), for benchmarking/validation from Python. */
SWM_API void swm_gemm_f32(int M, int N, int K, const float* A, const float* B, float* C);
SWM_API void swm_gemm_f32_naive(int M, int N, int K, const float* A, const float* B, float* C);
SWM_API void swm_gemm_i8(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C);
SWM_API void swm_gemm_i8_naive(int M, int N, int K, const int8_t* A, const int8_t* B, int32_t* C);

#ifdef __cplusplus
}
#endif
