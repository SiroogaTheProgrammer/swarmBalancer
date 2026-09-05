// Builds small .swm images in memory with hand-picked weights and checks the
// layer maths, the quantised path, and the device RAM cap.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "swarm/model.hpp"

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

bool near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol * (1.0f + std::fabs(b)); }

// Minimal .swm writer mirroring python/swarm/brain/formats.py
struct Writer {
  std::vector<std::uint8_t> buf;
  void u32(std::uint32_t v) {
    const std::size_t o = buf.size();
    buf.resize(o + 4);
    std::memcpy(buf.data() + o, &v, 4);
  }
  void f32(float v) {
    const std::size_t o = buf.size();
    buf.resize(o + 4);
    std::memcpy(buf.data() + o, &v, 4);
  }
  void f32s(const std::vector<float>& v) {
    for (float x : v) f32(x);
  }
  void i8s(const std::vector<std::int8_t>& v) {
    buf.insert(buf.end(), v.begin(), v.end());
  }
  void header(std::uint32_t n_layers, int c, int h, int w) {
    buf.assign({'S', 'W', 'M', '1'});
    u32(n_layers);
    u32(c);
    u32(h);
    u32(w);
  }
  void layer(swarm::LayerType t, swarm::DType d = swarm::DType::F32) {
    u32(static_cast<std::uint32_t>(t));
    u32(static_cast<std::uint32_t>(d));
  }
};

// conv(identity + constant) -> relu -> maxpool -> flatten -> dense -> softmax
void test_pipeline() {
  Writer w;
  w.header(6, 1, 4, 4);

  // Conv2D 1->2, k3, s1, p1. filter0 = centre tap (identity), filter1 = zeros with bias 1.
  w.layer(swarm::LayerType::Conv2D);
  w.u32(1); w.u32(2); w.u32(3); w.u32(1); w.u32(1); w.f32(1.0f);
  std::vector<float> cw(2 * 9, 0.0f);
  cw[4] = 1.0f;
  w.f32s(cw);
  w.f32s({0.0f, 1.0f});

  w.layer(swarm::LayerType::ReLU);

  w.layer(swarm::LayerType::MaxPool2D);
  w.u32(2); w.u32(2);

  w.layer(swarm::LayerType::Flatten);

  // Dense 8->2, weights stored [in][out]: out0 = 0.1*sum(ch0), out1 = sum(ch1)
  w.layer(swarm::LayerType::Dense);
  w.u32(8); w.u32(2); w.f32(1.0f);
  std::vector<float> dw(16, 0.0f);
  for (int i = 0; i < 4; ++i) dw[i * 2 + 0] = 0.1f;       // ch0 pooled values -> out0
  for (int i = 4; i < 8; ++i) dw[i * 2 + 1] = 1.0f;       // ch1 pooled values -> out1
  w.f32s(dw);
  w.f32s({0.0f, 0.0f});

  w.layer(swarm::LayerType::Softmax);

  swarm::Model m = swarm::Model::load_from_memory(w.buf.data(), w.buf.size());
  expect(m.layers().size() == 6, "layer count");
  expect(m.output_shape() == swarm::Shape{2, 1, 1}, "output shape");
  expect(m.macs_per_run() == 2ull * 16 * 9 + 8 * 2, "macs " + std::to_string(m.macs_per_run()));
  expect(m.weights_bytes() == (18 + 2 + 16 + 2) * 4, "weights bytes");

  // input: 0..15 with a few negatives (relu should clip them)
  std::vector<float> x(16), y(2);
  for (int i = 0; i < 16; ++i) x[i] = static_cast<float>(i);
  x[5] = -100.0f;  // in the top-left 2x2 block -> pooled max becomes 4 (indices 0,1,4,5 -> 0,1,4,-100)
  m.run(x.data(), y.data());

  // pooled ch0: blocks (0,1,4,5)->4, (2,3,6,7)->7, (8,9,12,13)->13, (10,11,14,15)->15 => sum 39 -> 3.9; ch1 all 1 => 4
  const float z0 = 3.9f, z1 = 4.0f;
  const float e0 = std::exp(z0 - z0), e1 = std::exp(z1 - z0);
  expect(near(y[0], e0 / (e0 + e1)), "softmax[0]=" + std::to_string(y[0]));
  expect(near(y[1], e1 / (e0 + e1)), "softmax[1]=" + std::to_string(y[1]));

  // RAM cap: too small must fail with a message naming both numbers.
  bool threw = false;
  try {
    swarm::Model::load_from_memory(w.buf.data(), w.buf.size(), 64);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("device cap") != std::string::npos;
  }
  expect(threw, "ram cap enforced");
  // Exactly the required size must succeed.
  swarm::Model m2 = swarm::Model::load_from_memory(w.buf.data(), w.buf.size(), m.required_bytes());
  expect(m2.required_bytes() == m.required_bytes(), "ram cap boundary");

  // Truncated file must be rejected, never read out of bounds.
  threw = false;
  try {
    swarm::Model::load_from_memory(w.buf.data(), w.buf.size() - 3);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  expect(threw, "truncated file rejected");
}

// int8 dense/conv must track the f32 result closely (dynamic activation quantisation).
void test_quantised_matches_f32() {
  const int in = 64, out = 16;
  std::vector<float> W((std::size_t)in * out), bias(out), x(in);
  std::uint32_t s = 12345;
  auto rnd = [&]() {
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
  };
  for (auto& v : W) v = rnd();
  for (auto& v : bias) v = rnd();
  for (auto& v : x) v = rnd();

  Writer wf;
  wf.header(1, in, 1, 1);
  wf.layer(swarm::LayerType::Dense);
  wf.u32(in); wf.u32(out); wf.f32(1.0f);
  wf.f32s(W);
  wf.f32s(bias);

  float amax = 0.0f;
  for (float v : W) amax = std::fmax(amax, std::fabs(v));
  const float scale = amax / 127.0f;
  std::vector<std::int8_t> Wq(W.size());
  for (std::size_t i = 0; i < W.size(); ++i) Wq[i] = static_cast<std::int8_t>(std::lrint(W[i] / scale));

  Writer wq;
  wq.header(1, in, 1, 1);
  wq.layer(swarm::LayerType::Dense, swarm::DType::I8);
  wq.u32(in); wq.u32(out); wq.f32(scale);
  wq.i8s(Wq);
  wq.f32s(bias);

  swarm::Model mf = swarm::Model::load_from_memory(wf.buf.data(), wf.buf.size());
  swarm::Model mq = swarm::Model::load_from_memory(wq.buf.data(), wq.buf.size());
  expect(mq.weights_bytes() == in * out + out * 4, "int8 weights are 1 byte each");
  expect(mq.required_bytes() < mf.required_bytes(), "int8 model needs less RAM");

  std::vector<float> yf(out), yq(out);
  mf.run(x.data(), yf.data());
  mq.run(x.data(), yq.data());
  float max_err = 0.0f, max_abs = 0.0f;
  for (int i = 0; i < out; ++i) {
    max_err = std::fmax(max_err, std::fabs(yf[i] - yq[i]));
    max_abs = std::fmax(max_abs, std::fabs(yf[i]));
  }
  if (max_err >= 0.03f * max_abs)
    for (int i = 0; i < out; ++i) std::printf("  y[%2d] f32=%9.5f i8=%9.5f\n", i, yf[i], yq[i]);
  expect(max_err < 0.03f * max_abs, "int8 dense within 3% of f32 (err=" + std::to_string(max_err) + ")");
}

}  // namespace

int main() {
  test_pipeline();
  test_quantised_matches_f32();
  if (g_failures) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("test_nn: ok\n");
  return 0;
}
