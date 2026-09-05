#include "swarm/model.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace swarm {

namespace {

constexpr char kMagic[4] = {'S', 'W', 'M', '1'};
constexpr std::uint32_t kMaxLayers = 4096;
constexpr std::uint32_t kMaxDim = 1u << 16;
constexpr std::size_t kMaxElems = std::size_t(1) << 28;  // 1 GiB of f32 - sanity bound on any single tensor

// Bounds-checked little-endian reader over an in-memory file image.
struct Reader {
  const std::uint8_t* p;
  std::size_t size;
  std::size_t pos = 0;

  void need(std::size_t n) const {
    if (pos + n > size) throw std::runtime_error("truncated .swm file");
  }
  std::uint32_t u32() {
    need(4);
    std::uint32_t v;
    std::memcpy(&v, p + pos, 4);
    pos += 4;
    return v;
  }
  float f32() {
    need(4);
    float v;
    std::memcpy(&v, p + pos, 4);
    pos += 4;
    return v;
  }
  int dim(const char* what) {
    const std::uint32_t v = u32();
    if (v == 0 || v > kMaxDim) throw std::runtime_error(std::string("invalid ") + what + " in .swm file");
    return static_cast<int>(v);
  }
  const std::uint8_t* bytes(std::size_t n) {
    need(n);
    const std::uint8_t* r = p + pos;
    pos += n;
    return r;
  }
};

struct PendingWeights {
  const std::uint8_t* w = nullptr;  // raw bytes in the file image
  std::size_t w_bytes = 0;
  const std::uint8_t* b = nullptr;
  std::size_t b_bytes = 0;
};

}  // namespace

Model Model::load(const std::string& path, std::size_t ram_cap_bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open model file: " + path);
  std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return load_from_memory(data.data(), data.size(), ram_cap_bytes);
}

Model Model::load_from_memory(const std::uint8_t* data, std::size_t size, std::size_t ram_cap_bytes) {
  Reader r{data, size};
  r.need(4);
  if (std::memcmp(r.p, kMagic, 4) != 0) throw std::runtime_error("not a .swm model (bad magic)");
  r.pos = 4;

  Model m;
  const std::uint32_t n_layers = r.u32();
  if (n_layers == 0 || n_layers > kMaxLayers) throw std::runtime_error("invalid layer count");
  m.input_shape_ = Shape{r.dim("input c"), r.dim("input h"), r.dim("input w")};
  if (m.input_shape_.elems() > kMaxElems) throw std::runtime_error("input tensor too large");

  // Pass 1: parse layer descriptors, remember where weights live in the file image.
  std::vector<PendingWeights> pending(n_layers);
  Shape cur = m.input_shape_;
  std::size_t required = 0;  // arena bytes: sum of align_up(block) for every block we will allocate
  std::size_t max_act = 0;
  ScratchNeeds scratch;

  for (std::uint32_t li = 0; li < n_layers; ++li) {
    Layer L;
    const std::uint32_t type = r.u32();
    const std::uint32_t dtype = r.u32();
    if (type < 1 || type > 6) throw std::runtime_error("unknown layer type " + std::to_string(type));
    if (dtype > 1) throw std::runtime_error("unknown dtype " + std::to_string(dtype));
    L.type = static_cast<LayerType>(type);
    L.dtype = static_cast<DType>(dtype);

    switch (L.type) {
      case LayerType::Dense:
        L.in = r.dim("dense in");
        L.out = r.dim("dense out");
        L.w_scale = r.f32();
        break;
      case LayerType::Conv2D:
        L.in_c = r.dim("conv in_c");
        L.out_c = r.dim("conv out_c");
        L.k = r.dim("conv k");
        L.stride = r.dim("conv stride");
        {
          const std::uint32_t pad = r.u32();
          if (pad > kMaxDim) throw std::runtime_error("invalid conv pad");
          L.pad = static_cast<int>(pad);
        }
        L.w_scale = r.f32();
        break;
      case LayerType::MaxPool2D:
        L.k = r.dim("pool k");
        L.stride = r.dim("pool stride");
        break;
      default:
        if (L.dtype != DType::F32) throw std::runtime_error("only weight layers can be quantised");
        break;
    }

    if (L.has_weights()) {
      const std::size_t wc = L.weight_count();
      if (wc > kMaxElems) throw std::runtime_error("weight tensor too large");
      const std::size_t wb = wc * (L.dtype == DType::I8 ? 1 : 4);
      const std::size_t bb = L.bias_count() * 4;
      pending[li].w = r.bytes(wb);
      pending[li].w_bytes = wb;
      pending[li].b = r.bytes(bb);
      pending[li].b_bytes = bb;
      required += Arena::align_up(wb) + Arena::align_up(bb);
      m.weights_bytes_ += wb + bb;
    }

    plan_layer(L, cur);
    if (L.out_shape.elems() > kMaxElems) throw std::runtime_error("activation tensor too large");
    cur = L.out_shape;
    max_act = std::max({max_act, L.in_shape.elems(), L.out_shape.elems()});
    const ScratchNeeds n = scratch_needs(L);
    scratch.q_in = std::max(scratch.q_in, n.q_in);
    scratch.cols_i8 = std::max(scratch.cols_i8, n.cols_i8);
    scratch.cols_f32 = std::max(scratch.cols_f32, n.cols_f32);
    scratch.acc = std::max(scratch.acc, n.acc);
    m.macs_ += L.macs;
    m.layers_.push_back(L);
  }
  if (r.pos != r.size) throw std::runtime_error("trailing bytes after last layer in .swm file");

  required += 2 * Arena::align_up(max_act * sizeof(float));
  required += Arena::align_up(scratch.q_in) + Arena::align_up(scratch.cols_i8) +
              Arena::align_up(scratch.cols_f32 * sizeof(float)) + Arena::align_up(scratch.acc * sizeof(std::int32_t));

  if (ram_cap_bytes != 0 && required > ram_cap_bytes) {
    throw std::runtime_error("model needs " + std::to_string(required) + " bytes of RAM but the device cap is " +
                             std::to_string(ram_cap_bytes) + " bytes (" + std::to_string(m.weights_bytes_) +
                             " bytes are weights)");
  }

  // Pass 2: place everything in the arena.
  m.arena_ = Arena(required);
  for (std::uint32_t li = 0; li < n_layers; ++li) {
    Layer& L = m.layers_[li];
    if (!L.has_weights()) continue;
    void* w = m.arena_.alloc(pending[li].w_bytes);
    std::memcpy(w, pending[li].w, pending[li].w_bytes);
    if (L.dtype == DType::I8) L.w_i8 = static_cast<const std::int8_t*>(w);
    else L.w_f32 = static_cast<const float*>(w);
    float* b = m.arena_.alloc<float>(pending[li].b_bytes / 4);
    std::memcpy(b, pending[li].b, pending[li].b_bytes);
    L.bias = b;
  }
  m.act_a_ = m.arena_.alloc<float>(max_act);
  m.act_b_ = m.arena_.alloc<float>(max_act);
  m.scratch_.q_in = m.arena_.alloc<std::int8_t>(scratch.q_in);
  m.scratch_.cols_i8 = m.arena_.alloc<std::int8_t>(scratch.cols_i8);
  m.scratch_.cols_f32 = m.arena_.alloc<float>(scratch.cols_f32);
  m.scratch_.acc = m.arena_.alloc<std::int32_t>(scratch.acc);
  return m;
}

void Model::run(const float* input, float* output) {
  const std::size_t n = layers_.size();
  const float* src = input;
  float* bufs[2] = {act_a_, act_b_};
  int which = 0;
  for (std::size_t i = 0; i < n; ++i) {
    float* dst = (i + 1 == n) ? output : bufs[which];
    forward(layers_[i], src, dst, scratch_);
    src = dst;
    which ^= 1;
  }
}

}  // namespace swarm
