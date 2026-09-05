#pragma once
// Minimal neural-network layer set for "super minimalist" image recognition
// on constrained devices (satellites, drones). Layers operate on flat
// float32 CHW buffers; quantised (int8) layers quantise their input
// dynamically per tensor, run the int8 GEMM and dequantise the result.

#include <cstddef>
#include <cstdint>

#include "swarm/tensor.hpp"

namespace swarm {

enum class LayerType : std::uint32_t { Dense = 1, Conv2D = 2, ReLU = 3, MaxPool2D = 4, Flatten = 5, Softmax = 6 };
enum class DType : std::uint32_t { F32 = 0, I8 = 1 };

struct Layer {
  LayerType type{};
  DType dtype = DType::F32;

  int in = 0, out = 0;             // Dense: weights stored [in][out] so inference is x[1,in] * W[in,out]
  int in_c = 0, out_c = 0;         // Conv2D: weights stored [out_c][in_c*k*k]
  int k = 0, stride = 1, pad = 0;  // Conv2D / MaxPool2D
  float w_scale = 1.0f;            // I8 layers: real_weight = q * w_scale

  const float* w_f32 = nullptr;
  const std::int8_t* w_i8 = nullptr;
  const float* bias = nullptr;  // always f32, length `out` / `out_c`

  Shape in_shape{}, out_shape{};  // filled by plan_layer
  std::uint64_t macs = 0;         // multiply-accumulates per forward pass

  std::size_t weight_count() const;
  std::size_t bias_count() const;
  bool has_weights() const { return type == LayerType::Dense || type == LayerType::Conv2D; }
};

// Scratch buffers some layers need. Sized by the model planner from scratch_needs().
struct Scratch {
  std::int8_t* q_in = nullptr;     // dynamically quantised input activations (I8 layers)
  std::int8_t* cols_i8 = nullptr;  // im2col matrix (I8 conv)
  float* cols_f32 = nullptr;       // im2col matrix (F32 conv)
  std::int32_t* acc = nullptr;     // int32 GEMM accumulators (I8 layers)
};

struct ScratchNeeds {
  std::size_t q_in = 0, cols_i8 = 0, cols_f32 = 0, acc = 0;  // element counts
};

// Derives out_shape and macs from the input shape. Throws std::runtime_error on mismatch.
void plan_layer(Layer& L, const Shape& in);
ScratchNeeds scratch_needs(const Layer& L);

// Runs one layer. `in` and `out` must not alias.
void forward(const Layer& L, const float* in, float* out, const Scratch& s);

const char* layer_type_name(LayerType t);

}  // namespace swarm
