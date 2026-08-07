# LiteRT 2.1.5 HTDemucs six-stem unified CPU baseline

Date: 2026-08-07

## Decision

The clean unified baseline is complete on Galaxy S25 and Galaxy S10. The same
APK, Android test APK, official six-stem artifact, runner, 7.8-second window,
25% overlap, and 30-second Athletics II PCM prefix were used on both devices.

Use four LiteRT CPU threads and four parity-preserving iSTFT workers as the
stable comparison profile. Six LiteRT threads are a useful S25 throughput
candidate, but this single ordered screen heated the S25 to thermal status 1
and its ten-sample core P95 became worse than the four-thread result. On S10,
six threads did not materially improve 30-second E2E over four threads.

The run is a short optimization baseline, not product qualification. It has
one E2E execution per configuration, one song, no 180-second thermal run, and
no cancellation or application playback test.

## Frozen identity

- Branch: `experiment/demucs-rtf-optimization`
- Facility revision: `37e79d208d289e05f8ad33b67c8e760f4a389c88`
- Source dirty: `false`
- LiteRT: `2.1.5`
- Resolved Gradle-cache AAR: 10,058,192 bytes, SHA-256
  `a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37`
- App APK: 144,538,875 bytes, SHA-256
  `aa2e50eddc79f8e20f7fd611a65082e302ae35ca8b7eeed1bba2072552af63bd`
- Test APK: 509,152 bytes, SHA-256
  `eade97bb15906ffdda3480261a47a464bfa373769e6ddad0808380707b4b17b7`
- Model SHA-256:
  `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7`
- S25: `SM-S9310`, `SM8750`, Android 15 / API 35
- S10: `SM-G9730`, `SM8150`, Android 12 / API 31

The APK runtime field says `maven-unresolved` because the existing Gradle
build only embeds an automatic SHA when `-PliteRtAar` is used. The exact AAR
above was independently hashed from the resolved Gradle cache. APK SHA and
installed APK SHA matched on both devices. A later facility cleanup should
embed the resolved Maven artifact SHA automatically; this does not change the
code or runtime used by this batch.

## Measurement contract

Each configuration uses one compiled model session:

1. Prepare DSP, compile the CPU model, and allocate buffers.
2. Prepare one fixed canonical input window.
3. Run two unmeasured neural-core warmups.
4. Record ten neural-core invocations.
5. Start a separate timer and process the complete 30-second prefix through
   STFT, tensor writes, inference, output reads, iSTFT, branch combine, OLA,
   PCM16 conversion, writes, commit, and output validation.

`warm E2E` excludes session preparation and the 12 calibration invocations.
`cold RTF` below adds session preparation to warm E2E, but also excludes the
calibration invocations. The report's old `total` field includes calibration
and must not be used as song RTF for this experiment.

Every stage records wall time, process CPU time, caller-thread CPU time, ART
allocated-byte delta, and ART GC-count delta where Android exposes the runtime
counter. Native XNNPACK allocation is not represented by the ART allocation
counter. LiteRT 2.1.5's Java `CompiledModel` CPU API exposes threads and
XNNPACK flags but no per-op profiler callback/result API. Reports therefore
record per-op profiling as explicitly unsupported instead of presenting
inferred operator data.

## S25 result

| LiteRT threads | iSTFT workers | Core mean / P95 | Warm E2E | Warm / cold RTF | Inference / iSTFT share | Peak PSS | Thermal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 2 | 3661 / 3674 ms | 27.367 s | 0.912 / 0.930 | 80.3% / 7.3% | 1101.5 MiB | 0 |
| 2 | 4 | 3663 / 3673 ms | 27.960 s | 0.932 / 0.950 | 82.5% / 4.7% | 1107.4 MiB | 0 |
| 4 | 2 | 2267 / 2287 ms | 26.453 s | 0.882 / 0.900 | 68.3% / 11.3% | 1104.3 MiB | 0, 1 |
| 4 | 4 | 2267 / 2325 ms | 25.513 s | 0.850 / 0.869 | 71.9% / 6.9% | 1107.7 MiB | 1 |
| 6 | 2 | 1965 / 2570 ms | 21.520 s | 0.717 / 0.736 | 63.0% / 13.3% | 1106.0 MiB | 1 |
| 6 | 4 | 2056 / 2553 ms | 21.721 s | 0.724 / 0.742 | 65.7% / 8.4% | 1107.8 MiB | 1 |

Four iSTFT workers reduce iSTFT time, but whole-run ordering and thermal state
are large enough that they do not improve every single E2E result. The
six-thread core median was about 1.83-1.84 seconds, while slow samples raised
P95 above 2.55 seconds. Four threads were slower but substantially tighter.

## S10 result

| LiteRT threads | iSTFT workers | Core mean / P95 | Warm E2E | Warm / cold RTF | Inference / iSTFT share | Peak PSS | Thermal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 2 | 6655 / 6901 ms | 68.368 s | 2.279 / 2.304 | 56.8% / 40.3% | 1093.3 MiB | 0 |
| 2 | 4 | 6794 / 7070 ms | 52.134 s | 1.738 / 1.762 | 78.5% / 17.5% | 1096.5 MiB | 0 |
| 4 | 2 | 5198 / 5408 ms | 62.036 s | 2.068 / 2.102 | 50.4% / 44.8% | 1088.8 MiB | 0 |
| 4 | 4 | 5234 / 5527 ms | 45.104 s | 1.503 / 1.537 | 69.2% / 24.5% | 1091.8 MiB | 0 |
| 6 | 2 | 5088 / 5396 ms | 59.381 s | 1.979 / 2.014 | 49.1% / 46.1% | 1088.7 MiB | 0 |
| 6 | 4 | 5150 / 5490 ms | 44.888 s | 1.496 / 1.530 | 68.3% / 25.3% | 1091.6 MiB | 0 |

Four outer iSTFT workers remain mandatory on S10 for this JTransforms policy.
Two workers leave iSTFT at about 27.4-27.8 seconds per six-window song; four
workers reduce it to about 9.1-11.4 seconds. Six LiteRT threads improve the
ten-sample core mean only 2.1% over four threads and improve warm E2E only
0.5%, so four threads remain the conservative baseline.

This clean run is slower than the earlier dirty-source S10 experiment
(`RTF 1.258` for the comparable Athletics control). The new ten-sample core
mean is about 5.23 seconds instead of roughly 4.55 seconds, and four-worker
iSTFT is about 11.06 seconds per song instead of 8.96 seconds. Thermal stayed
at 0. This confirms the previously documented Java/JTransforms and CPU
scheduling variance; a second cold-session replicate is required before
attributing the difference to source or runtime changes.

## Numerical and resource gates

- All 12 runs completed with finite output and valid six-stem WAV contracts.
- Within each device, all six configurations produced identical per-stem WAV
  SHA-256 values.
- S25 and S10 produced two device-specific SHA sets. Cross-device bit identity
  is therefore not claimed; later parity work must compare each device against
  the canonical host oracle with numeric tolerances.
- Peak sampled PSS remained approximately 1.09-1.11 GiB.
- Core calibration reported no ART GC. ART allocation deltas were zero on S10
  and 32-64 KiB total across ten S25 invocations; these counters exclude native
  XNNPACK allocation.
- S10 thermal status remained 0. S25 began at 0 and reached 1 for later runs.

## Next gate

Repeat the selected `threads=4, iSTFT workers=4` profile as three independent
cold sessions per device, with ordering alternated against the six-thread S25
candidate. Do not proceed to workspace fusion or pipeline changes until the
clean replicate explains the S10 regression and establishes a stable P95.

After that replicate, the next parity-preserving implementation batch should
reuse DSP/tensor workspaces and fuse branch combine, denormalization, OLA,
clamp, PCM16 conversion, and writing. Pipeline overlap belongs in a separate
experiment because it changes CPU contention and timing semantics.

## Evidence

- `outputs/demucs-rtf-baseline-s25/`
- `outputs/demucs-rtf-baseline-s10/`
- `tools/run_htdemucs_cpu_matrix.py`
- `app/src/androidTest/java/com/example/musicsourceseparation/benchmark/HtdemucsCanonicalE2eBenchmark.kt`
