#pragma once
// A brain: an ordered list of layers loaded from a `.swm` file (see
// docs/MODEL_FORMAT.md), with all memory carved out of one Arena so the exact
// RAM footprint is known and a device RAM cap can be enforced at load time.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "swarm/arena.hpp"
#include "swarm/nn.hpp"
#include "swarm/tensor.hpp"

namespace swarm {

class Model {
public:
  // Loads a model. `ram_cap_bytes` = 0 means unlimited; otherwise loading fails
  // with a descriptive std::runtime_error if weights+activations+scratch exceed it.
  static Model load(const std::string& path, std::size_t ram_cap_bytes = 0);
  static Model load_from_memory(const std::uint8_t* data, std::size_t size, std::size_t ram_cap_bytes = 0);

  Model(Model&&) noexcept = default;
  Model& operator=(Model&&) noexcept = default;

  // input must hold input_shape().elems() floats, output output_shape().elems().
  void run(const float* input, float* output);

  const Shape& input_shape() const { return input_shape_; }
  const Shape& output_shape() const { return layers_.back().out_shape; }
  const std::vector<Layer>& layers() const { return layers_; }

  std::uint64_t macs_per_run() const { return macs_; }
  std::size_t weights_bytes() const { return weights_bytes_; }
  // Total RAM the brain needs on a device (weights + activations + scratch).
  std::size_t required_bytes() const { return arena_.used(); }

private:
  Model() = default;

  std::vector<Layer> layers_;
  Shape input_shape_{};
  Arena arena_;
  float* act_a_ = nullptr;
  float* act_b_ = nullptr;
  Scratch scratch_{};
  std::uint64_t macs_ = 0;
  std::size_t weights_bytes_ = 0;
};

}  // namespace swarm
