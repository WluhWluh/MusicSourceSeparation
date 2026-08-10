# S25 MDX bounded GPU and QNN double-buffer experiment

Date: 2026-08-10  
Branch: `experiment/demucs-double-buffer`  
Runtime: LiteRT 2.1.5 / local `litert-android-2.1.5-bss.2`  
Model: `uvr_mdxnet_3_9662`  
Artifact SHA-256: `f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378`

## Scope and method

The new `MdxDoubleBufferInstrumentedTest` uses the frozen 9662 contract and
`native-packed` FFT with four DSP workers. It keeps one `CompiledModel` and two input/output
buffer slots. While the previous window is executing, the host prepares the next window's
STFT and input buffer. After the next invocation is queued, the host reads and iSTFTs the
previous output. This is a short model-window experiment, not a full-song playback test:
WAV I/O, residual, and application buffering are excluded.

Each profile used a clean S25 process, two warmups, and measured windows. Outputs were finite.
The GPU profile used the BSS bounded runtime guard (`beginInference/endInference`), OpenCL
FP32, kernel batch N=1, and command queue window N=1. The QNN profile used the `qnnV79` flavor,
HTP sustained-high-performance, and inference optimization.

## Results

| Profile | Windows | Sequential mean / median | Double-buffer pipeline | Change |
| --- | ---: | ---: | ---: | ---: |
| S25 bounded GPU FP32 | 10 | 330.75 / 331.36 ms | 291.79 ms/window | -11.8% |
| S25 QNN HTP v79 | 5 | 148.81 / 148.46 ms | 124.16 ms/window | -16.6% |

The bounded GPU report recorded `gpu-opencl-bounded-fp32-v1`, artifact `2.1.5-bss.2`,
`2540` dispatches and `2540` event waits across the measured runs. This is positive
delegation evidence, not CPU fallback.

The QNN report exposed `CPU`, `GPU`, and `NPU`, and produced one non-empty IR file:
`qnn-ir/qnn_partition_0.json` (396,223 bytes). The QNN result is therefore qualified as
actual NPU delegation. A control run with the ordinary Maven LiteRT package had no QNN IR
and approximately 4.6 seconds per window; it is intentionally rejected as fallback-only and
is not included in the table.

## Stage behavior

With bounded GPU, sequential native-packed windows had roughly 9-11 ms STFT/input, 277-304
ms model/output synchronization, and 14-22 ms iSTFT. In the overlapped run, the next STFT
usually remained 12-16 ms, while the previous inference wait was about 248-260 ms. The
pipeline hides most of the host preparation and previous-window iSTFT behind GPU work.

With QNN, sequential inference was about 118-122 ms, STFT/input 9-13 ms, and iSTFT 12-15 ms.
The double-buffer wait was 85-107 ms, with 11-15 ms next-window preparation and 15-19 ms
previous iSTFT. The single-session two-slot arrangement is materially better than compiling
two independent model instances and avoids duplicating the graph's memory.

## Decision

- Keep double buffering as an experimental optimization for accelerated MDX profiles. The
  measured scoped gain is 11.8% on bounded GPU and 16.6% on QNN for 9662.
- Do not directly project these percentages to full-song RTF. Residual, OLA, PCM, writes,
  thermal behavior, and QNN JIT are outside this test. The likely end-to-end gain is lower,
  especially once native OLA becomes the dominant stage.
- Product integration requires a full-song 30-second then 273.699-second gate, numerical
  parity/quality checks, cancellation and model-switch lifecycle checks, and QNN cold-JIT
  versus cache-hit accounting.
- Use one `CompiledModel` with two buffer slots. Do not create two complete GPU/QNN model
  contexts; the initial GPU two-instance probe failed compilation and adds avoidable memory
  pressure.

Raw reports are retained under the ignored `outputs/android-benchmark/mdx-double-buffer-s25`
directory. The test does not modify Booming SS application integration.

