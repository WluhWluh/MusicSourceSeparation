# S25 MDX full-song double-buffer experiment

Date: 2026-08-10  
Branch: `experiment/demucs-double-buffer`  
Input: Coast Town, 273.699093 seconds, 48 windows, 47 joins  
Model: `uvr_mdxnet_3_9662`, native-packed DSP, four workers  
LiteRT: 2.1.5; BSS runtime `2.1.5-bss.2`

## Method

This test extends the short-window double-buffer runner through the complete processing
path: MP3/WAV decode, context-window extraction, native STFT, LiteRT inference, native
iSTFT, model scaling, residual, PCM16 quantization, and two WAV writers. One CompiledModel
is used with two independent input/output buffer slots. While window `n` is running, the
host prepares window `n+1`; after queuing that run, it reads, iSTFTs, and writes window `n`.

The GPU run used the bounded OpenCL FP32 guard (N=1). The QNN run used the `qnnV79` flavor,
HTP sustained-high-performance and inference optimization. Both reports completed all 48
windows with finite output.

## Results

| Backend | Setup/JIT | Processing | Processing RTF | Decode + setup + processing | Total RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| S25 bounded GPU FP32 | 1.001 s | 14.467 s | 0.05286 | 15.963 s | 0.05832 |
| S25 QNN HTP v79 | 13.212 s | 5.995 s | 0.02190 | 19.670 s | 0.07187 |

For comparison, the same device data retained from the prior sequential native-packed GPU
run measured 18.879 s processing time (RTF 0.06898). The double-buffer GPU run is 23.4%
faster in that paired full-song comparison. Its two WAV outputs are byte-identical to the
prior sequential run:

| Output | SHA-256 |
| --- | --- |
| Model output | `f92f45c52a4902763eae031f80c8ad8b499296ee1fdf56318224988d9a24553a` |
| Residual | `4af386eefae21c7fb7f9f04bf17d0aa15bbdb37af594053c0b4ebf91d7f08fe4` |

The GPU report recorded 6,096 dispatches and 6,096 event waits, with profile
`gpu-opencl-bounded-fp32-v1`. The QNN report exposed `NPU` and generated
`qnn-ir/qnn_partition_0.json` (396,223 bytes), so it is qualified as actual QNN delegation.
QNN output is finite and complete; it is not claimed byte-identical to FP32 because the
existing QNN contract has a separate numerical-quality gate.

## Stage totals

| Stage | GPU | QNN |
| --- | ---: | ---: |
| Initial + next-window preparation | 0.999 s | 0.854 s |
| Inference wait | 10.567 s | 2.379 s |
| Output read | 0.316 s | 0.204 s |
| iSTFT | 0.798 s | 0.824 s |
| Residual | 0.404 s | 0.399 s |
| PCM/WAV write | 1.378 s | 1.330 s |

These stage sums overlap by design and therefore exceed the wall-clock processing time.
The GPU host preparation is materially slower while the GPU is active than in isolation,
but remains hidden behind inference for most windows. QNN has much lower steady-state
processing time; its 13.2-second cold setup/JIT dominates first-song total RTF.

## Decision

- Full-song bounded GPU double buffering is numerically safe for the tested 9662 artifact
  and provides a real 23% paired processing improvement. Keep it as an accelerated research
  profile; validate thermals and cancellation before product use.
- QNN double buffering reduces steady-state processing to 5.995 s for the song, but cold
  setup makes total first-song RTF worse than GPU. A resident/cached QNN session is required
  before using the steady-state number for product decisions.
- The test still does not cover playback prebuffer, seek, background, process death, or UI
  lifecycle. Those remain application-integration gates, not DSP evidence.

Raw reports and WAVs are retained under the ignored
`outputs/android-benchmark/mdx-double-buffer-full-song-s25/` directory.
