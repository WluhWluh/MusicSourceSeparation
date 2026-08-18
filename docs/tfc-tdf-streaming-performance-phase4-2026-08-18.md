# TFC-TDF streaming performance Phase 4

Status: complete for the isolated Android prototype. This phase measures the
real local-audio read-ahead, TFC-TDF DSP, LiteRT CPU/GPU session, wet-window
publication, and dry/wet selection path on one Galaxy S25. It is not yet an
integration test for the product Media3 renderer or AudioTrack.

## Frozen run

| Item | Identity |
| --- | --- |
| Device | Samsung SM-S9310 / `pa1q` / Snapdragon `SM8750` |
| Android | API 35, `arm64-v8a` |
| Runtime | LiteRT `2.2.0-bss.2`, bounded OpenCL FP32 profile |
| Runtime AAR | 13,102,718 bytes, SHA-256 `35b55a0ef9a6d28e56271a9bc3b6b6cc8a84b16732b17b34b2a6b51ee7be3124` |
| TFC-TDF model | 3,989,856 bytes, SHA-256 `0ee7bbc0bd5a1194745ebf4df1753ba6ef32a256cbb55fca8098a43912591f5e` |
| Source | `coast_town.mp3`, 11,019,555 bytes, SHA-256 `f66be47fc846459f8ac92543b38dab2019765b556cff98539339bbb44cabcae3` |
| Source format | MediaCodec decoded, 44.1 kHz stereo |
| Playback window | 30 s at real-time pace, block size 1,024 frames |
| Seek | at 12 s, target 45 s |
| CPU | LiteRT CPU, XNNPACK, 4 threads |
| GPU | LiteRT bounded OpenCL FP32; one session per epoch, reused across windows |

The source revision was `fb155e4` with a dirty working tree containing the
Phase 4 changes. The reproducible entry point is
`tools/run_tfc_tdf_streaming_benchmark.ps1`.

## Data path measured

```text
MediaCodec playback reader ---------------------> dry PCM
                                                   |
MediaCodec analysis reader -> bounded input ring -> STFT
                                                    -> LiteRT session
                                                    -> iSTFT + residual mix
                                                    -> bounded wet snapshot
                                                   |
dry PCM + wet snapshot -> block selector -> output digest
```

The selector never waits for the analysis worker. A block uses wet PCM only
when its complete range is already published; otherwise it uses dry PCM. A
late result cannot replace an already emitted dry block. The benchmark checks
every selected output block for finite samples.

`fullChainRtf` is the sum of session setup plus STFT, LiteRT, iSTFT, and
residual processing divided by processed audio time. It intentionally does
not hide normal playback decode or read-ahead decode inside that number.
`wallRtf` includes the real-time pacing loop and is the end-to-end scheduling
indicator for this prototype.

## S25 results

| Metric | CPU 4 threads | bounded GPU |
| --- | ---: | ---: |
| First wet block ready | 1,073 ms | 1,115 ms |
| First wet block selected | 1,073 ms | 1,115 ms |
| Seek to wet latency | 720 ms | 720 ms |
| Dry blocks / wet blocks | 79 / 1,213 | 79 / 1,213 |
| Positive wet lead median | 6,037 ms | 6,548 ms |
| Positive wet lead maximum | 7,616 ms | 7,988 ms |
| Real-time wall RTF | 1.0022 | 1.0026 |
| DSP + LiteRT compute RTF | 0.3181 | 0.1364 |
| Playback reader decode | 1,515.5 ms | 1,756.5 ms |
| Analysis read-ahead decode | 1,244.7 ms | 1,517.1 ms |
| Late/missed windows | 0 | 0 |
| Discarded epoch outputs | 0 | 0 |

Both runs completed 17 neural windows across two epochs (7 before seek and 10
after seek). The CPU sessions took 46.9 ms and 8.5 ms to initialize; the GPU
sessions took 636.3 ms and 486.3 ms. The second setup cost is the expected new
session after seek, not per-window creation.

The aggregate DSP/inference timing was:

| Timing | CPU 4 threads | bounded GPU |
| --- | ---: | ---: |
| Session setup wall | 55.4 ms | 1,122.5 ms |
| Window compute wall | 9,488.9 ms | 2,969.3 ms |
| STFT wall | 565.3 ms | 587.1 ms |
| LiteRT invocation wall | 8,467.6 ms | 1,600.8 ms |
| iSTFT wall | 405.5 ms | 705.6 ms |
| Analysis worker CPU time | 34,797 ms | 2,025 ms |
| Analysis worker thread CPU time | 9,402.7 ms | 1,493.9 ms |

The GPU evidence was positive in both generations: each session reported
profile `gpu-opencl-bounded-fp32-v1`, `1,802` dispatches, and `1,802` event
waits. This is a verified bounded GPU run, not an inference result that merely
requested a GPU accelerator.

## Seek latency decomposition

The benchmark has a second trace run which records the critical path after the
seek request. The values below are from separate 30-second S25 runs using the
same source and seek position. The small difference between the two totals and
the earlier baseline is normal scheduling variance.

| Seek stage | CPU 4 threads | bounded GPU |
| --- | ---: | ---: |
| `seek()` call itself | 0.31 ms | 0.18 ms |
| Seek return to new worker/session start | 38.14 ms | 33.59 ms |
| Second session initialization | 8.22 ms | 442.40 ms |
| Session end to first window process start | 44.91 ms | 49.60 ms |
| Of that, MediaCodec read wall time | 39.42 ms | 44.19 ms |
| First window STFT | 12.49 ms | 13.97 ms |
| First window LiteRT inference | 583.64 ms | 108.37 ms |
| First window iSTFT | 12.46 ms | 19.84 ms |
| Remaining first-window processing | about 2.98 ms | about 6.73 ms |
| Window end to wet block selection | 16.66 ms | 22.26 ms |
| Total seek return to wet selection | 719.53 ms | 696.77 ms |

The analysis reader performed eight reads totaling 131,072 frames before the
first post-seek window. The useful model window is 119,808 frames, so this is
the expected bounded read-ahead plus decoder granularity rather than a full-song
decode. The `seek()` API is not the bottleneck. On CPU, the first window's
inference dominates. On GPU, rebuilding the second `CompiledModel` session
dominates.

These measurements also explain why CPU and GPU seek totals happened to be
similar despite very different steady-state inference speed. They should not
be interpreted as equal backend seek performance.

## Latency reduction options

The following are ordered by expected impact for this data path:

1. Keep `Environment`, `CompiledModel`, tensor buffers, and the bounded GPU
   runtime alive for the current model/accelerator. Seek should reset the input
   and wet rings and advance the epoch, not recreate the session. The trace
   gives an upper-bound saving of about 442 ms on GPU and only about 8 ms on
   CPU for this run.
2. Keep the analysis decoder alive and use a safe decoder flush/seek path, or
   feed the analysis ring from PCM already decoded by the Media3 side. This
   targets the 39-44 ms first-fill cost and avoids repeated extractor/codec
   setup. WAV/FLAC random access may be cheaper than MP3, but must be measured
   separately.
3. Replace cancellation plus executor resubmission with a persistent analysis
   worker receiving an epoch/seek command. The current handoff costs about
   34-38 ms. Old results can still be rejected by epoch without stopping the
   worker thread.
4. Start decoder refill and any unavoidable session initialization in parallel
   on bounded workers. This can overlap part of the roughly 50 ms input-fill
   span, but it is secondary to GPU session reuse.
5. Reuse STFT, iSTFT, tensor, and residual buffers. This primarily reduces
   memory pressure and GC; the measured first-window DSP portion is only about
   25-34 ms, so it will not by itself remove the 720 ms delay.
6. For CPU-only devices, reduce the first-window model cost or use a smaller
   streaming model. The current CPU inference slice is about 584 ms, so model
   selection has a much larger effect than audio-thread or block-selector
   micro-optimizations.

As a projection, retaining the GPU session could move this S25 seek path from
about 697 ms toward roughly 250-300 ms before decoder and product-integration
changes. That is an arithmetic upper-bound estimate, not a product benchmark.
The corresponding CPU path would remain near 700 ms unless its model inference
is optimized. With a persistent decoder and a shared PCM read-ahead, the GPU
path could plausibly approach the 200 ms range, but this needs a dedicated
implementation and measurement.

## Resource observations

The instrumented report records start/end PSS and GC counters; the external
poller records the in-run peak. Values below are from the final CPU and GPU
runs, not from the earlier exploratory runs.

| Resource | CPU 4 threads | bounded GPU |
| --- | ---: | ---: |
| Peak PSS during run | 367,759 KiB | 306,219 KiB |
| PSS after close | 71,988 KiB | 101,917 KiB |
| Peak native heap allocated | 313,293,520 bytes | 56,481,360 bytes |
| Process CPU time | 40,951 ms | 11,087 ms |
| ART GC count | 2 -> 30 | 2 -> 27 |
| Thermal status | 0 -> 0 | 0 -> 0 |
| Battery broadcast temperature | 13.3 -> 13.6 C | 14.9 -> 14.4 C |

The Android thermal service exposes several inconsistent cached and current
sensor values on this device. The thermal status transition is therefore the
accepted thermal result; the battery broadcast temperature is retained as a
raw observation, not a skin or SoC temperature claim.

## Interpretation

The compact TFC-TDF path has enough isolated S25 margin to keep selecting wet
blocks without pausing: both CPU and bounded GPU runs stayed at approximately
one wall-clock second per audio second and had no missed windows. The GPU
reduces the measured neural/DSP compute RTF from about 0.318 to 0.136 and
reduces process CPU use substantially, at the cost of roughly 1.1 s of total
session setup across the two epochs.

The measured first-wet latency is dominated by session setup, the first
MediaCodec read-ahead, and one complete model window. Seek re-wet latency is
about 0.72 s in this run because seek starts a new generation and session. The
large positive lead values reflect the prototype's four-window wet retention,
not an additional audible delay.

The CPU peak native/PSS footprint is too high to treat this implementation as
a product-ready memory profile. The benchmark currently allocates full
window tensors and reconstruction buffers per window; later work should reuse
those buffers and measure under the product process budget before enabling a
streaming model by default.

## Evidence files

```text
outputs/tfc-tdf-streaming-s25/s25-cpu-4t-20260818-r3/report.json
outputs/tfc-tdf-streaming-s25/s25-cpu-4t-20260818-r3/host-resource-samples.jsonl
outputs/tfc-tdf-streaming-s25/s25-gpu-bounded-20260818-r2/report.json
outputs/tfc-tdf-streaming-s25/s25-gpu-bounded-20260818-r2/host-resource-samples.jsonl
outputs/tfc-tdf-streaming-s25/s25-cpu-seek-trace-20260818-r2/report.json
outputs/tfc-tdf-streaming-s25/s25-gpu-seek-trace-20260818/report.json
```

These directories are ignored local evidence. The report's output SHA-256 is
an execution trace identity only: CPU and GPU can choose different dry/wet
boundaries as asynchronous read-ahead progresses, so their whole-output
digests are not a numerical backend-parity gate.

## Limitations and next boundary

- The source was one local MP3 and one S25 codec implementation; this does not
  qualify AAC, FLAC, Media3 source formats, or other device codec behavior.
- The normal reader in this test is a second MediaCodec reader, not the actual
  Media3 renderer and audio sink used by Booming SS.
- No speaker output, audio focus, service lifecycle, screen-off behavior, or
  product UI was exercised.
- The experiment does not create persistent PCM or FLAC cache data, but it
  also does not yet connect to the product cache and model-selection policy.

The next decision point is an integration prototype that feeds the same
engine from the real playback renderer while retaining the dry fallback. It
should first reduce/reuse window allocations, then repeat the S25 test with
screen-off and service lifecycle transitions before any default product
exposure.
