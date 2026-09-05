# `.swm` model format (version `SWM1`)

Little-endian, no alignment padding. Written by `python/swarm/brain/formats.py`, read by `cpp/src/model.cpp` and `formats.read_swm`.

```
char[4]  magic          "SWM1"
u32      n_layers       1..4096
u32      in_c, in_h, in_w

repeat n_layers:
  u32 type             1 dense | 2 conv2d | 3 relu | 4 maxpool2d | 5 flatten | 6 softmax
  u32 dtype            0 f32 | 1 i8        (only dense / conv2d may be i8)

  dense:      u32 in, u32 out, f32 w_scale
              w[in*out]           f32 or i8, layout [in][out]
              bias[out]           f32
  conv2d:     u32 in_c, u32 out_c, u32 k, u32 stride, u32 pad, f32 w_scale
              w[out_c*in_c*k*k]   f32 or i8, layout [out_c][in_c*k*k], inner index = c*k*k + ki*k + kj
              bias[out_c]         f32
  maxpool2d:  u32 k, u32 stride   (numpy reference supports k == stride only)
  relu, flatten, softmax: no fields
```

Semantics

* Activations are CHW float32. Dense layers take the flattened input.
* `i8` weights: `real = q * w_scale`, symmetric per-tensor, `q` in [-127, 127].
* At run time an `i8` layer quantises its **input** dynamically: `scale_a = max|x| / 127`, GEMM in int32, output `acc * scale_a * w_scale + bias`.
* The C++ reader validates every dimension (1..65536), rejects truncated files, unknown ids, quantised non-weight layers and trailing bytes, and checks shapes layer by layer (`plan_layer`).

Sizes for the default `tiny_cnn` (1x32x32 -> 5 classes): 34 213 params, 401 568 MACs/frame; `f32` file 137 048 B / RAM 276 224 B; `int8` file 34 592 B / RAM 153 344 B (fits a 192 KiB Cortex-M4).
