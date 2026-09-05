#include "swarm/nn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "swarm/config.hpp"
#include "swarm/matmul.hpp"

namespace swarm {

const char* layer_type_name(LayerType t) {
  switch (t) {
    case LayerType::Dense: return "dense";
    case LayerType::Conv2D: return "conv2d";
    case LayerType::ReLU: return "relu";
    case LayerType::MaxPool2D: return "maxpool2d";
    case LayerType::Flatten: return "flatten";
    case LayerType::Softmax: return "softmax";
  }
  return "?";
}

std::size_t Layer::weight_count() const {
  switch (type) {
    case LayerType::Dense: return (std::size_t)in * out;
    case LayerType::Conv2D: return (std::size_t)out_c * in_c * k * k;
    default: return 0;
  }
}

std::size_t Layer::bias_count() const {
  switch (type) {
    case LayerType::Dense: return (std::size_t)out;
    case LayerType::Conv2D: return (std::size_t)out_c;
    default: return 0;
  }
}

// ----------------------------------------------------------------------------
// Planning
// ----------------------------------------------------------------------------

static int conv_out_dim(int in, int k, int stride, int pad) {
  const int o = (in + 2 * pad - k) / stride + 1;
  if (o <= 0) throw std::runtime_error("layer produces empty output (input too small for kernel)");
  return o;
}

void plan_layer(Layer& L, const Shape& in) {
  L.in_shape = in;
  switch (L.type) {
    case LayerType::Dense:
      if (in.elems() != (std::size_t)L.in)
        throw std::runtime_error("dense layer expects " + std::to_string(L.in) + " inputs, got " +
                                 std::to_string(in.elems()));
      L.out_shape = Shape{L.out, 1, 1};
      L.macs = (std::uint64_t)L.in * L.out;
      break;
    case LayerType::Conv2D: {
      if (in.c != L.in_c)
        throw std::runtime_error("conv2d expects " + std::to_string(L.in_c) + " input channels, got " +
                                 std::to_string(in.c));
      const int oh = conv_out_dim(in.h, L.k, L.stride, L.pad);
      const int ow = conv_out_dim(in.w, L.k, L.stride, L.pad);
      L.out_shape = Shape{L.out_c, oh, ow};
      L.macs = (std::uint64_t)L.out_c * oh * ow * L.in_c * L.k * L.k;
      break;
    }
    case LayerType::MaxPool2D: {
      const int oh = conv_out_dim(in.h, L.k, L.stride, 0);
      const int ow = conv_out_dim(in.w, L.k, L.stride, 0);
      L.out_shape = Shape{in.c, oh, ow};
      L.macs = 0;
      break;
    }
    case LayerType::Flatten:
      L.out_shape = Shape{(int)in.elems(), 1, 1};
      L.macs = 0;
      break;
    case LayerType::ReLU:
    case LayerType::Softmax:
      L.out_shape = in;
      L.macs = 0;
      break;
  }
}

ScratchNeeds scratch_needs(const Layer& L) {
  ScratchNeeds n;
  const bool q = L.dtype == DType::I8;
  if (L.type == LayerType::Dense && q) {
    n.q_in = (std::size_t)L.in;
    n.acc = (std::size_t)L.out;
  } else if (L.type == LayerType::Conv2D) {
    const std::size_t K = (std::size_t)L.in_c * L.k * L.k;
    const std::size_t N = (std::size_t)L.out_shape.h * L.out_shape.w;
    if (q) {
      n.q_in = L.in_shape.elems();
      n.cols_i8 = K * N;
      n.acc = (std::size_t)L.out_c * N;
    } else {
      n.cols_f32 = K * N;
    }
  }
  return n;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

// Symmetric per-tensor quantisation to [-127, 127]. Returns the scale.
static float quantize_dynamic(const float* SWARM_RESTRICT x, std::size_t n, std::int8_t* SWARM_RESTRICT q) {
  float amax = 0.0f;
  for (std::size_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inv = 1.0f / scale;
  for (std::size_t i = 0; i < n; ++i) {
    long r = std::lrintf(x[i] * inv);
    r = std::max(-127L, std::min(127L, r));
    q[i] = static_cast<std::int8_t>(r);
  }
  return scale;
}

// cols[(c*k*k + ki*k + kj), (oy*ow + ox)] = in[c, oy*s - p + ki, ox*s - p + kj]  (0 outside the image)
template <class T>
static void im2col(const T* SWARM_RESTRICT in, int C, int H, int W, int k, int s, int p, int oh, int ow,
                   T* SWARM_RESTRICT cols) {
  const std::size_t N = (std::size_t)oh * ow;
  for (int c = 0; c < C; ++c) {
    for (int ki = 0; ki < k; ++ki) {
      for (int kj = 0; kj < k; ++kj) {
        T* SWARM_RESTRICT row = cols + (((std::size_t)c * k + ki) * k + kj) * N;
        for (int oy = 0; oy < oh; ++oy) {
          const int iy = oy * s - p + ki;
          T* SWARM_RESTRICT dst = row + (std::size_t)oy * ow;
          if (iy < 0 || iy >= H) {
            std::memset(dst, 0, sizeof(T) * ow);
            continue;
          }
          const T* SWARM_RESTRICT src = in + ((std::size_t)c * H + iy) * W;
          for (int ox = 0; ox < ow; ++ox) {
            const int ix = ox * s - p + kj;
            dst[ox] = (ix >= 0 && ix < W) ? src[ix] : T(0);
          }
        }
      }
    }
  }
}

static void add_bias_rows(float* SWARM_RESTRICT out, const float* SWARM_RESTRICT bias, int rows, std::size_t n) {
  for (int r = 0; r < rows; ++r) {
    float* SWARM_RESTRICT o = out + (std::size_t)r * n;
    const float b = bias[r];
    for (std::size_t i = 0; i < n; ++i) o[i] += b;
  }
}

static void dequant_rows(const std::int32_t* SWARM_RESTRICT acc, float* SWARM_RESTRICT out, const float* SWARM_RESTRICT bias,
                         int rows, std::size_t n, float scale) {
  for (int r = 0; r < rows; ++r) {
    const std::int32_t* SWARM_RESTRICT a = acc + (std::size_t)r * n;
    float* SWARM_RESTRICT o = out + (std::size_t)r * n;
    const float b = bias[r];
    for (std::size_t i = 0; i < n; ++i) o[i] = static_cast<float>(a[i]) * scale + b;
  }
}

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------

static void forward_dense(const Layer& L, const float* in, float* out, const Scratch& s) {
  if (L.dtype == DType::F32) {
    gemm_f32(1, L.out, L.in, in, L.in, L.w_f32, L.out, out, L.out);
    for (int j = 0; j < L.out; ++j) out[j] += L.bias[j];
  } else {
    const float a_scale = quantize_dynamic(in, (std::size_t)L.in, s.q_in);
    gemm_i8(1, L.out, L.in, s.q_in, L.in, L.w_i8, L.out, s.acc, L.out);
    const float scale = a_scale * L.w_scale;
    for (int j = 0; j < L.out; ++j) out[j] = static_cast<float>(s.acc[j]) * scale + L.bias[j];
  }
}

static void forward_conv2d(const Layer& L, const float* in, float* out, const Scratch& s) {
  const Shape& is = L.in_shape;
  const Shape& os = L.out_shape;
  const int K = L.in_c * L.k * L.k;
  const int N = os.h * os.w;
  // Dense-layer bias is per output neuron; conv bias is per output channel, i.e. per output row.
  if (L.dtype == DType::F32) {
    im2col<float>(in, is.c, is.h, is.w, L.k, L.stride, L.pad, os.h, os.w, s.cols_f32);
    gemm_f32(L.out_c, N, K, L.w_f32, K, s.cols_f32, N, out, N);
    add_bias_rows(out, L.bias, L.out_c, (std::size_t)N);
  } else {
    const float a_scale = quantize_dynamic(in, is.elems(), s.q_in);
    im2col<std::int8_t>(s.q_in, is.c, is.h, is.w, L.k, L.stride, L.pad, os.h, os.w, s.cols_i8);
    gemm_i8(L.out_c, N, K, L.w_i8, K, s.cols_i8, N, s.acc, N);
    dequant_rows(s.acc, out, L.bias, L.out_c, (std::size_t)N, a_scale * L.w_scale);
  }
}

static void forward_maxpool(const Layer& L, const float* in, float* out) {
  const Shape& is = L.in_shape;
  const Shape& os = L.out_shape;
  for (int c = 0; c < is.c; ++c) {
    const float* SWARM_RESTRICT src = in + (std::size_t)c * is.h * is.w;
    float* SWARM_RESTRICT dst = out + (std::size_t)c * os.h * os.w;
    for (int oy = 0; oy < os.h; ++oy) {
      for (int ox = 0; ox < os.w; ++ox) {
        float m = -INFINITY;
        for (int ky = 0; ky < L.k; ++ky) {
          const float* row = src + (std::size_t)(oy * L.stride + ky) * is.w + ox * L.stride;
          for (int kx = 0; kx < L.k; ++kx) m = std::max(m, row[kx]);
        }
        dst[(std::size_t)oy * os.w + ox] = m;
      }
    }
  }
}

static void forward_softmax(const float* in, float* out, std::size_t n) {
  float m = -INFINITY;
  for (std::size_t i = 0; i < n; ++i) m = std::max(m, in[i]);
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = std::exp(in[i] - m);
    sum += out[i];
  }
  const float inv = 1.0f / sum;
  for (std::size_t i = 0; i < n; ++i) out[i] *= inv;
}

void forward(const Layer& L, const float* in, float* out, const Scratch& s) {
  switch (L.type) {
    case LayerType::Dense: forward_dense(L, in, out, s); break;
    case LayerType::Conv2D: forward_conv2d(L, in, out, s); break;
    case LayerType::ReLU: {
      const std::size_t n = L.in_shape.elems();
      for (std::size_t i = 0; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
      break;
    }
    case LayerType::MaxPool2D: forward_maxpool(L, in, out); break;
    case LayerType::Flatten: std::memcpy(out, in, sizeof(float) * L.in_shape.elems()); break;
    case LayerType::Softmax: forward_softmax(in, out, L.in_shape.elems()); break;
  }
}

}  // namespace swarm
