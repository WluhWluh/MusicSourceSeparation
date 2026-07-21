# Android LiteRT benchmark: UVR MDXNET 9482

Date: 2026-07-19

## Scope

This benchmark compares the existing ONNX Runtime CPU path with LiteRT CPU and
GPU paths for `UVR_MDXNET_9482`. It measures model inference only. Decode,
STFT/ISTFT, cache I/O, and playback are outside this benchmark.

The production Booming SS configuration uses ORT 1.26.0, sequential execution,
one inter-op thread, and eight intra-op threads. The benchmark uses the same ORT
settings.

## Devices

| Device | SoC / GPU | Android |
| --- | --- | --- |
| Samsung Galaxy S10 SM-G9730 | Snapdragon 855 / Adreno 640 | 12 (API 31) |
| Samsung Galaxy S25 SM-S9310 | Snapdragon 8 Elite / Adreno 830 | 15 (API 35) |

Both devices used battery power and reported Android thermal status 0.

## Model and runtimes

- ONNX input/output: `float32 [1, 4, 2048, 256]` (NCHW).
- LiteRT input/output: `float32 [1, 2048, 256, 4]` (NHWC).
- TFLite model: static batch-1 conversion from the ONNX model.
- ORT: `onnxruntime-android:1.26.0`, CPU, eight threads.
- LiteRT: `litert:2.1.5`, CPU or OpenCL GPU.
- GPU delegation: 183/183 TFLite nodes in one `LITERT_CL` partition on both
  devices.
- GPU timing includes asynchronous dispatch and the output read that waits for
  completion. Dispatch alone is not a valid inference measurement.

## Stable inference results

Each result uses two warmups followed by five measured inferences with the same
generated input. CPU time is total process CPU time, so values above wall time
represent parallel CPU work.

| Device | Backend | Median wall ms | Mean CPU ms | CPU/wall | SNR vs ORT | PSS delta MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S10 | ORT CPU | 3403.8 | 22340 | 6.55 | reference | 119.5 |
| S10 | LiteRT CPU | 2417.4 | 15369 | 6.40 | 101.0 dB | 147.7 |
| S10 | LiteRT GPU FP32 | 2207.8 | 89 | 0.04 | 100.9 dB | 52.8 |
| S10 | LiteRT GPU FP16 | 1098.5 | 70 | 0.06 | 23.1 dB | 55.6 |
| S25 | ORT CPU | 864.7 | 6579 | 7.56 | reference | 83.6 |
| S25 | LiteRT CPU | 657.5 | 4904 | 7.50 | 101.1 dB | 66.8 |
| S25 | LiteRT GPU FP32 | 246.3 | 5 | 0.02 | 101.0 dB | 80.6 |
| S25 | LiteRT GPU FP16 | 216.5 | 6 | 0.03 | 23.2 dB | 81.6 |

Compared with ORT, LiteRT GPU FP32 was about 1.54x faster on the S10 and 3.51x
faster on the S25. More importantly, it removed nearly all process CPU work.

LiteRT CPU was 24-29% faster and used 25-31% less total CPU time than ORT. Its
memory result was device-dependent; XNNPACK weight packing made it the largest
PSS result on the S10.

## Real music input

A real MDX spectrogram was generated from the first window of
`_-_Coast_Town__decoded.wav`. Its RMS was 1.556 and its range was approximately
-142.9 to 146.2, unlike the small generated benchmark input.

| Device | Backend | Median wall ms | SNR vs ORT | Max absolute error |
| --- | --- | ---: | ---: | ---: |
| S10 | LiteRT CPU | 2249.0 | 102.10 dB | 0.000049 |
| S10 | LiteRT GPU FP32 | 2202.3 | 101.72 dB | 0.000041 |
| S10 | LiteRT GPU FP16 | 1104.3 | 24.70 dB | 0.279 |
| S25 | LiteRT CPU | 629.0 | 102.03 dB | 0.000046 |
| S25 | LiteRT GPU FP32 | 244.2 | 101.66 dB | 0.000050 |
| S25 | LiteRT GPU FP16 | 217.6 | 24.63 dB | 0.339 |

The FP16 error is therefore not an artifact of the generated input. FP16 is not
a suitable production default without end-to-end audio quality evidence. GPU
FP32 preserves the converted model's output closely enough to proceed to an
audio-level prototype.

## Thermal observations

The AP sensor updates coarsely, so these values show direction rather than
precise power. Runs started after cooldown.

| Device | Backend | First AP C | Maximum AP C | Change C |
| --- | --- | ---: | ---: | ---: |
| S10 | ORT CPU | 30.8 | 53.6 | +22.8 |
| S10 | LiteRT CPU | 32.9 | 51.5 | +18.6 |
| S10 | LiteRT GPU FP32 | 32.4 | 38.7 | +6.3 |
| S10 | LiteRT GPU FP16 | 30.7 | 35.6 | +4.9 |
| S25 | ORT CPU | 24.2 | 54.0 | +29.8 |
| S25 | LiteRT CPU | 25.5 | 46.9 | +21.4 |
| S25 | LiteRT GPU FP32 | 26.4 | 31.0 | +4.6 |

Battery current and charge-counter readings were too coarse and vendor-specific
for a defensible energy comparison. Use a long fixed-duration run with an
external power meter for energy-per-window conclusions.

## Runtime size

For arm64, the stock AAR native libraries are:

- ORT 1.26.0: about 26.25 MiB uncompressed.
- LiteRT 2.1.5 CPU plus OpenCL GPU: about 7.68 MiB uncompressed.

Replacing ORT with LiteRT can save about 18.6 MiB of installed arm64 native
code. Keeping both runtimes adds LiteRT's footprint instead. A real size win
therefore requires either removing ORT, publishing separate runtime variants,
or accepting that custom ONNX models continue to require an ORT-enabled build.

## Recommendation

1. Prototype a reusable LiteRT GPU FP32 runner for the preset 9482 TFLite model.
2. Fall back to LiteRT CPU when GPU compilation or full delegation fails.
3. Keep FP16 experimental and disabled by default.
4. Reuse the compiled model and evaluate LiteRT's serialization cache; GPU setup
   took roughly 0.6-1.2 seconds in these cold-process tests.
5. Run a full Booming SS window and full-song benchmark, including decode,
   STFT/ISTFT, cache writes, foreground-service behavior, playback deadlines,
   and cancellation.
6. Add Mali, Exynos, MediaTek, and Tensor devices before making GPU the default.
