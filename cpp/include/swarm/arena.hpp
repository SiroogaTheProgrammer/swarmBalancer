#pragma once
// Bump allocator with a hard capacity.
//
// Everything the brain touches at run time (weights, activations, scratch) is
// carved out of one Arena, so `used()` is the exact RAM footprint a target
// device would need, and a device profile's RAM limit can be enforced simply
// by constructing the Arena with that capacity. Blocks are never freed
// individually - a model's memory plan is static.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace swarm {

class Arena {
public:
  static constexpr std::size_t kAlign = 64;  // one cache line; also satisfies any SIMD alignment

  Arena() = default;
  explicit Arena(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ > 0) {
      storage_.reset(new unsigned char[capacity_ + kAlign]);
      base_ = align_up(reinterpret_cast<std::uintptr_t>(storage_.get()));
    }
  }

  Arena(Arena&&) noexcept = default;
  Arena& operator=(Arena&&) noexcept = default;
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  void* alloc(std::size_t bytes) {
    const std::size_t off = align_up(used_);
    if (off + bytes > capacity_) {
      throw std::length_error("arena overflow: requested " + std::to_string(bytes) + " bytes at offset " +
                              std::to_string(off) + ", capacity " + std::to_string(capacity_));
    }
    used_ = off + bytes;
    return reinterpret_cast<void*>(base_ + off);
  }

  template <class T>
  T* alloc(std::size_t count) {
    return static_cast<T*>(alloc(count * sizeof(T)));
  }

  std::size_t used() const { return used_; }
  std::size_t capacity() const { return capacity_; }

  // Size a block occupies once placed in the arena (next block starts aligned).
  template <class U>
  static constexpr U align_up(U v) {
    return static_cast<U>((v + (kAlign - 1)) & ~static_cast<U>(kAlign - 1));
  }

private:
  std::unique_ptr<unsigned char[]> storage_;
  std::uintptr_t base_ = 0;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
};

}  // namespace swarm
