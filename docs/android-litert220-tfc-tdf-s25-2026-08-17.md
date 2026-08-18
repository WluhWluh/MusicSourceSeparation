# Android LiteRT 2.2.0 benchmark: TFC-TDF compact candidate on S25

Status: S25 tensor execution passed on LiteRT CPU and bounded OpenCL GPU. This
is a single real-song first-window benchmark; it does not establish full-song
throughput, thermal sustainability, or product separation quality.

## Scope and frozen identities

The benchmark uses the real-song first-window fixture for the compact TFC-TDF
candidate. The fixture contains the same FP32 NHWC TFLite model, input, ONNX
golden, and host-TFLite golden used by the host validation.

| Item | Identity |
| --- | --- |
| Candidate | `tfc_tdf_default_vocals_core_fp32@tflite-1` |
| Fixture | `tfc_tdf_default_real_window_0@1` |
| Model bytes | `3,989,856` |
| Model SHA-256 | `0ee7bbc0bd5a1194745ebf4df1753ba6ef32a256cbb55fca8098a43912591f5e` |
| Fixture manifest SHA-256 | `111cf80d8c4b4cf592368badc05db53d558429a4f8200fe426a099c15d9206f0` |
| LiteRT runtime | `2.2.0-bss.2` |
| Runtime AAR SHA-256 | `35b55a0ef9a6d28e56271a9bc3b6b6cc8a84b16732b17b34b2a6b51ee7be3124` |
| Benchmark source | `f5172a59f16a540a008ed599a8e0ebe7e4d99624`, clean |

The device was a Samsung Galaxy S25 (`SM-S9310`, `pa1q`, Qualcomm `SM8750`,
Android 15/API 35, `arm64-v8a`). The runtime reported both CPU and GPU as
available accelerators.

## Method

Each run used a new application process, two warmups, and ten measured
invocations of the same `1 x 1025 x 128 x 4` real-audio tensor. CPU runs used
LiteRT XNNPACK with the requested thread count. GPU runs used the bounded
FP32 OpenCL profile `gpu-opencl-bounded-fp32-v1`, with kernel batch size 1 and
command queue window size 1.

`setup` includes runtime loading, environment/model compilation, buffer
creation, and input preparation. `invoke` measures model execution only;
`total` also includes reading the output tensor. The 95th percentile is the
sample-order percentile used by the instrumented test, not an interpolated
statistic.

The acceptance gate was finite output, at least 90 dB SNR against the ONNX
golden, and a maximum absolute error no greater than the larger of `2e-4` and
`2e-6 * referencePeak`.

## CPU results

| Run | Threads | Setup ms | Invoke median / p95 ms | Total median ms | Peak PSS KB | Battery temperature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s25-cpu-t1-r1` | 1 | 20.24 | 1,709.995 / 1,712.535 | 1,710.948 | 364,887 | 17.1 C |
| `s25-cpu-t2-r1` | 2 | 20.98 | 874.709 / 890.235 | 875.520 | 354,513 | 17.1 C |
| `s25-cpu-t4-r2` | 4 | 20.61 | 452.257 / 456.209 | 453.117 | 380,361 | 16.7 C |
| `s25-cpu-t8-r1` | 8 | 19.07 | 358.038 / 497.740 | 358.977 | 380,497 | 15.8 C |

Eight threads reduced the median relative to four threads by about 21%, but
had a wider tail: the p95 was higher than the four-thread p95. This makes
four threads the steadier CPU point in this small sample; a larger repeated
and full-song run is needed before choosing a product thread default.

## Bounded GPU results

| Run | Setup ms | Invoke median / p95 ms | Total median ms | Peak PSS KB | Dispatches / waits | Battery temperature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s25-gpu-bounded-r1` | 696.95 | 77.209 / 96.876 | 77.959 | 318,577 | 1,060 / 1,060 | 15.8 C |
| `s25-gpu-bounded-r2` | 527.99 | 96.348 / 104.621 | 98.912 | 319,863 | 1,060 / 1,060 | 15.8 C |

Both GPU sessions passed the runtime capability check and reported actual
bounded-runtime dispatch evidence. The difference between the two cold
sessions is visible in both setup and steady-state timing, so the GPU result
should be reported as a range rather than a single guaranteed latency.

Compared with the four-thread CPU median, the GPU invoke median was about
`5.86x` faster in run 1 and `4.70x` faster in run 2. Compared with eight CPU
threads, the corresponding ratios were about `4.64x` and `3.72x`. GPU setup
was roughly `0.53-0.70 s`, versus about `0.02 s` for CPU. For this fixture,
the GPU amortizes that extra setup after approximately two to three model
windows, depending on which cold GPU session is used.

## Numerical and resource observations

All six accepted runs produced finite output and the same output file identity
within each backend. Representative comparisons are:

| Backend | ONNX SNR | ONNX max error | ONNX normalized max error | Host-TFLite SNR | Host-TFLite max error |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU (all thread counts) | 120.690 dB | `1.602e-4` | `1.027e-6` | 120.664 dB | `1.411e-4` |
| bounded GPU (both runs) | 119.714 dB | `1.373e-4` | `8.806e-7` | 119.641 dB | `1.450e-4` |

Thermal status remained Android status `0` throughout every run, and the
reported battery temperature did not change during an individual session.
These are coarse device indicators, not a measurement of SoC hotspot
temperature or sustained throttling. Peak PSS was lower for the GPU sessions
than for CPU (`~319 MB` versus `~355-380 MB`), while post-close PSS remained
above the starting process footprint because the GPU/runtime process was not
fully reclaimed until force-stop.

## Raw reports

The untracked device reports remain under the ignored output tree:

```text
outputs/tfc-tdf-default-s25-litert220/s25-cpu-t1-r1/report.json
outputs/tfc-tdf-default-s25-litert220/s25-cpu-t2-r1/report.json
outputs/tfc-tdf-default-s25-litert220/s25-cpu-t4-r2/s25-cpu-t4-r2/report.json
outputs/tfc-tdf-default-s25-litert220/s25-cpu-t8-r1/report.json
outputs/tfc-tdf-default-s25-litert220/s25-gpu-bounded-r1/report.json
outputs/tfc-tdf-default-s25-litert220/s25-gpu-bounded-r2/report.json
```

## Decision

LiteRT 2.2.0 is technically viable for this candidate on the S25 for both
CPU and bounded GPU execution. The GPU path is the clear tensor-throughput
winner for work containing multiple windows, but its setup cost and cold-run
variance must be handled by session reuse and background preparation. This
benchmark does not justify enabling the model as a product default: the
listening assessment remains substantially below MDX 9662, and a full-song
Android audio run is still required to measure DSP, memory, and sustained
behavior.

