#pragma once
#include <cstddef>

namespace swarm {

// Activation shape in CHW order. Dense layers use (n, 1, 1).
struct Shape {
  int c = 1;
  int h = 1;
  int w = 1;
  std::size_t elems() const { return static_cast<std::size_t>(c) * h * w; }
  bool operator==(const Shape& o) const { return c == o.c && h == o.h && w == o.w; }
};

}  // namespace swarm
