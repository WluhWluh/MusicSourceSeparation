# LiteRT 2.1.5 MDX native tensor buffer experiment

Date: 2026-08-09

Branch: `experiment/mdx-native-output-buffer`

Devices: Samsung Galaxy S25 (SM8750) and Galaxy S10 (SM8150)

## Purpose

The existing `native-packed` MDX runner uses native packed-real DSP but crosses the LiteRT
tensor boundary through Kotlin `writeFloat()` and `readFloat()`. The output read creates one
Java `FloatArray` per window: about 8 MiB for 9662, 10 MiB for HQ4, and 12 MiB for Kim Inst.
Ten measured windows therefore allocate about 80, 100, or 120 MiB before other runner costs.

This experiment closes that gap with a `native-packed-litert-c` profile. It uses the public
LiteRT 2.1.5 C API exported by `libLiteRt.so`, locks the managed input buffer while native STFT
writes NHWC data, runs the compiled model, then locks the managed output buffer while native
iSTFT reads it. No tensor-sized Java input or output array is created. Reusable separated
waveform, residual, and PCM arrays remain outside this tensor boundary.

The implementation is pinned to the custom bounded runtime
`litert-android-2.1.5-bss.2.aar`, SHA-256
`88cd2f7eaf1443d1c570085b1c24f239db87eb24c788a590adf5158e17443d0e`.

## Runtime profiles

CPU uses opaque options identifier `xnnpack` with `num_threads = 4`.

The final bounded GPU C profile uses identifier `gpu_options` with both controls:

```toml
backend = 1
precision = 2
kernel_batch_size = 1
num_steps_of_command_buffer_preparations = 1
```

`kernel_batch_size = 1` is required to obtain complete bounded dispatch evidence through the C
API. `num_steps_of_command_buffer_preparations = 1` keeps the C profile aligned with the Kotlin
public API profile. All final GPU reports embed these exact options, source revision
`7bf9048c64f066a37de71221ca25aa942768afcb`, and `sourceDirty=false`.

Two intermediate GPU configurations are not qualification evidence:

- Using only `num_steps_of_command_buffer_preparations` produced only 20 dispatches on the S25.
  It was fast but did not implement the requested bounded profile.
- Using only `kernel_batch_size` produced full dispatch counts, but Kim Inst on the S10 had
  unstable 7-11 second invokes. It is retained only as option-attribution evidence.

## Method

The three shape sentinels were 9662 (`2048 x 256`), HQ4 (`2560 x 256`), and Kim Inst
(`3072 x 256`). Each profile used four DSP workers, two warmups, and ten measured rotating
fixture windows. The comparison was the existing Kotlin CompiledModel plus native packed DSP
against the C CompiledModel plus direct managed buffers.

The CPU matrix was recorded at implementation revision `0e0abba`; later revisions only change
GPU opaque options and reporting, so the CPU implementation is identical. The final GPU matrix
was rebuilt and recorded from clean revision `7bf9048`.

## Numerical and delegation gates

All 12 CPU pairs and all 12 final GPU reports completed with finite outputs. For every device,
backend, and model, the old and direct profiles produced byte-identical PCM preview files. The
preview SHA differs between CPU and GPU as expected, but each paired comparison has exactly the
same SHA.

Every final GPU session also has complete dispatch and wait evidence:

| Model | Expected dispatch/wait | S25 | S10 |
| --- | ---: | ---: | ---: |
| 9662 | 1270 / 1270 | 1270 / 1270 | 1270 / 1270 |
| HQ4 | 1390 / 1390 | 1390 / 1390 | 1390 / 1390 |
| Kim Inst | 1390 / 1390 | 1390 / 1390 | 1390 / 1390 |

## CPU results

Times are means per window. Allocation and GC cover ten measured windows.

| Device | Model | Old total | Direct total | Change | Old allocation | Direct allocation | Old/direct GC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S25 | 9662 | 1295.34 ms | 1285.82 ms | -0.7% | 80.13 MiB | 0.03 MiB | 1 / 0 |
| S25 | HQ4 | 3671.84 ms | 3629.75 ms | -1.1% | 100.20 MiB | 0.06 MiB | 2 / 0 |
| S25 | Kim Inst | 4346.83 ms | 4462.43 ms | +2.7% | 120.20 MiB | 0.09 MiB | 2 / 1 |
| S10 | 9662 | 2523.03 ms | 2733.04 ms | +8.3% | 80.18 MiB | 0.05 MiB | 4 / 0 |
| S10 | HQ4 | 11860.29 ms | 11018.85 ms | -7.1% | 100.16 MiB | 0.03 MiB | 4 / 0 |
| S10 | Kim Inst | 9705.55 ms | 9411.92 ms | -3.0% | 120.23 MiB | 0.03 MiB | 6 / 0 |

The direct profile reports input write and output read as zero. Buffer lock/map time is included
inside STFT and iSTFT. The old path spent about 3.2-7.1 ms per S25 window and 4.4-15.9 ms per
S10 window on those two explicit copies. Invoke variation is larger than this boundary cost,
which explains the mixed E2E changes.

## Final bounded GPU results

| Device | Model | Old total | Direct total | Change | Old allocation | Direct allocation | Old/direct GC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S25 | 9662 | 344.00 ms | 332.98 ms | -3.2% | 80.10 MiB | 0.03 MiB | 1 / 0 |
| S25 | HQ4 | 803.71 ms | 785.02 ms | -2.3% | 100.16 MiB | 0.09 MiB | 2 / 1 |
| S25 | Kim Inst | 1176.42 ms | 1154.31 ms | -1.9% | 120.20 MiB | 0.13 MiB | 2 / 1 |
| S10 | 9662 | 2437.12 ms | 2426.04 ms | -0.5% | 80.18 MiB | 0.05 MiB | 4 / 0 |
| S10 | HQ4 | 3998.80 ms | 3981.64 ms | -0.4% | 100.18 MiB | 0.05 MiB | 4 / 0 |
| S10 | Kim Inst | 6017.13 ms | 6307.08 ms | +4.8% | 120.27 MiB | 0.05 MiB | 7 / 0 |

The five small improvements do not establish a stable speedup, and the ordered S10 Kim pair is
4.8% slower because direct invoke averaged 6181 ms versus 5866 ms. A nearby diagnostic pair
also showed device-state-sensitive Kim inference. This experiment therefore does not claim an
RTF improvement.

Live PSS must be taken from the in-session `memorySamples` rather than `memoryAfterClose`. GPU
resources are intentionally released before the latter snapshot, so after-close PSS is only a
lifecycle check. The final in-session results were:

| Device | Model | Old live PSS | Direct live PSS | Difference |
| --- | --- | ---: | ---: | ---: |
| S25 | 9662 | 526.3 MiB | 525.6 MiB | -0.7 MiB |
| S25 | HQ4 | 862.8 MiB | 832.3 MiB | -30.5 MiB |
| S25 | Kim Inst | 1000.1 MiB | 940.5 MiB | -59.6 MiB |
| S10 | 9662 | 506.3 MiB | 473.6 MiB | -32.7 MiB |
| S10 | HQ4 | 867.3 MiB | 837.1 MiB | -30.2 MiB |
| S10 | Kim Inst | 973.4 MiB | 938.3 MiB | -35.1 MiB |

This is consistent with eliminating Java tensor arrays. PSS is process-wide and sessions ran
sequentially, so these differences are supporting evidence rather than a precise attribution of
every released byte. All final GPU runs remained at Android thermal status 0. Battery temperature
is recorded in the raw reports; unusually low S25 readings are device telemetry and should not
be interpreted as ambient temperature.

## Conclusion

The missing MDX tensor boundary is closed. Direct managed buffers reduce ten-window ART
allocation from approximately 80/100/120 MiB to 0.03-0.13 MiB, normally eliminate measured GC,
preserve exact final PCM, and maintain complete bounded GPU dispatch. This is a material heap,
GC jitter, and OOM-risk improvement across all three MDX shape families.

Keep it classified as a memory and stability optimization. Current short ordered sessions do
not prove an RTF improvement, and the C invocation path adds a second runtime integration that
must remain pinned and tested against LiteRT 2.1.5.

Before application integration, run the existing 48-window/full-song fixture, 100-window
create/run/close and model-switch stress, cancellation, and repeated GPU lifecycle tests. These
are the remaining qualification gates; no Booming SS integration was performed here.

Raw reports and paired PCM previews are under the ignored directory
`outputs/android-benchmark/mdx-native-output-buffer-20260809/`.
