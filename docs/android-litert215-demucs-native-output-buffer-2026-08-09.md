# LiteRT 2.1.5 HTDemucs native output buffer experiment

Date: 2026-08-09

Branch: `experiment/demucs-native-output-buffer`

Devices: Samsung Galaxy S25 (SM8750) and Galaxy S10 (SM8150)

## Purpose

The existing benchmark reads both LiteRT outputs through `TensorBuffer.readFloat()` before
native postprocessing. That creates a 66 MiB frequency array and a 16 MiB time array for each
six-stem window, or 44 MiB and 11 MiB for each four-stem window. Repeating those allocations
causes avoidable Java heap pressure and GC.

LiteRT 2.1.5 does not expose a supported Kotlin API that lets this runner lock the managed
output buffer and pass its address to JNI for the duration of postprocessing. Accessing private
Kotlin or C++ wrapper object layouts was rejected because their ABI is not public. This
experiment instead uses the public LiteRT C API exported by `libLiteRt.so`.

## Experimental path

The `native-packed-litert-c` mode loads the LiteRT 2.1.5 C API with `dlopen`/`dlsym`, compiles
the same canonical artifact for CPU/XNNPACK, and retains both inputs and outputs in native
managed tensor buffers. CPU options use the stable opaque-options interface:

- identifier: `xnnpack`
- NUL-terminated TOML payload: `num_threads = 4\n`

The frequency and time outputs remain locked only while packed-real iSTFT and time-branch
addition execute. No Java output array is created. The final combined waveform is still a
reused Java `FloatArray`, so this is an output-boundary experiment rather than a completely
native E2E pipeline. The `libLiteRt.so` handle deliberately remains open for the process
lifetime so cached function pointers stay valid.

The comparison path is the existing `native-packed` DSP using the LiteRT Kotlin
`CompiledModel` runner and `TensorBuffer.readFloat()` outputs. Both modes use CPU four threads,
four iSTFT workers, the same input, fused reusable postprocessing, and the same model artifact.

## Contracts

| Candidate | Stems | Frequency output | Time output | Artifact SHA-256 |
| --- | ---: | ---: | ---: | --- |
| HTDemucs official 6s | 6 | 16,515,072 floats (66 MiB) | 4,127,760 floats (16 MiB) | `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7` |
| HTDemucs 6s guitar-ft | 6 | 16,515,072 floats (66 MiB) | 4,127,760 floats (16 MiB) | Frozen by its executable contract and device report |
| HTDemucs official 4s | 4 | 11,010,048 floats (44 MiB) | 2,751,840 floats (11 MiB) | Frozen by its executable contract and device report |

The four-stem output ABI remains independent. It is not a truncated six-stem output. The
guitar-ft model reuses the six-stem DSP shape but retains separate model identity and
research-only license status.

## Numerical result

The S25 two-second smoke produced bit-identical PCM16 for all six stems: 176,400 samples per
stem and maximum delta zero. The full 30-second matrix covered both devices, all three models,
and both output paths. All 12 runs completed, all outputs were finite, and every paired final
stem WAV had the same SHA-256. The direct native boundary therefore preserves the tested final
PCM exactly.

## Allocation and GC

Each row is one 30-second, six-window run. Values are E2E ART allocation deltas.

| Device and model | Array allocation | Direct allocation | Reduction | Array GC | Direct GC |
| --- | ---: | ---: | ---: | ---: | ---: |
| S25 official 6s | 631.3 MiB | 64.0 MiB | 89.9% | 6 | 1 |
| S25 guitar-ft 6s | 631.3 MiB | 64.0 MiB | 89.9% | 6 | 1 |
| S25 official 4s | 442.2 MiB | 63.9 MiB | 85.5% | 6 | 1 |
| S10 official 6s | 631.9 MiB | 64.6 MiB | 89.8% | 14 | 2 |
| S10 guitar-ft 6s | 631.9 MiB | 64.6 MiB | 89.8% | 14 | 2 |
| S10 official 4s | 442.5 MiB | 64.3 MiB | 85.5% | 12 | 4 |

The remaining approximately 64 MiB is primarily other runner and postprocessing activity,
including the reused final waveform boundary. Direct output removes the dominant repeated
Java arrays but does not make the complete pipeline allocation-free.

## Timing

The target boundary is array `outputRead + iSTFT` versus direct-buffer `iSTFT`; direct mode
reports output read as zero because lock/map overhead and branch combination occur inside the
iSTFT stage.

| Device and model | Array boundary | Direct boundary | Direct saving |
| --- | ---: | ---: | ---: |
| S25 official 6s, stable repeat | 338.2 ms | 174.8 ms | 163.4 ms |
| S25 guitar-ft 6s | 377.7 ms | 176.2 ms | 201.5 ms |
| S25 official 4s | 254.5 ms | 143.7 ms | 110.8 ms |
| S10 official 6s | 705.1 ms | 528.3 ms | 176.8 ms |
| S10 guitar-ft 6s | 813.8 ms | 578.4 ms | 235.4 ms |
| S10 official 4s | 554.8 ms | 405.6 ms | 149.2 ms |

This is only 18-39 ms per window. Model inference remains dominant, so the boundary saving is
mostly absorbed by run-to-run inference and device-state variation.

The first 30-second matrix illustrates that variation:

| Device and model | Array RTF | Direct RTF |
| --- | ---: | ---: |
| S25 official 6s | 0.542 | 0.740 |
| S25 official 4s | 0.606 | 0.714 |
| S25 guitar-ft 6s | 0.571 | 0.558 |
| S10 official 6s | 1.111 | 1.157 |
| S10 official 4s | 1.350 | 1.388 |
| S10 guitar-ft 6s | 1.234 | 1.226 |

Both paths had inference process-CPU/wall ratios near 3.8, confirming four active CPU threads.
The S25 official direct inference nevertheless slowed from about 2.44 to 3.37 seconds per
window in the noisy run. Android thermal status did not fully explain it and battery
temperature was not yet recorded.

A reversed-order S25 official repeat added battery temperature:

| Mode | E2E | RTF | Inference | Output read | iSTFT | Allocation | GC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct C API | 16.384 s | 0.546 | 13.312 s | 0 | 174.8 ms | 64.0 MiB | 1 |
| Kotlin array | 16.393 s | 0.546 | 13.147 s | 150.6 ms | 187.6 ms | 631.3 MiB | 6 |

Direct mode stayed at 31.9 C and thermal status 0. Array mode started at 32.1 C and ended at
33.8 C while thermal status still remained 0. The stable repeat makes the two E2E results
effectively equal and shows that the earlier direct result was device-state noise, not an
intrinsic C API regression. Battery temperature is now captured at prepare, each core sample,
each window, and final state because Android thermal status alone was insufficient.

## Recommendation

Treat direct native LiteRT outputs as a memory and stability improvement, not as an RTF
optimization. Exact PCM equality and roughly 86-90% lower ART allocation make it valuable when
heap churn, GC jitter, or OOM risk is the deciding constraint. Current evidence does not justify
adopting the extra C API invocation path solely for speed; stable S25 E2E was performance-neutral.

Keep this path experimental and pinned to LiteRT 2.1.5 until it passes cancellation, repeated
create/close, 100-window, and 180-second lifecycle tests. It has not been tested with GPU or
QNN and is not integrated into Booming SS. A future supported Kotlin direct-buffer/read-into
interface should be preferred if it can provide the same zero-copy native boundary without a
second invocation implementation.

Raw device reports and paired WAV evidence are under the ignored directory
`outputs/android-benchmark/htdemucs-native-output-buffer-20260809/`.
