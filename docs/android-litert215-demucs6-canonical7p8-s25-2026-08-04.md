# LiteRT 2.1.5 Canonical HTDemucs 6-Stem S25 Experiment

Date: 2026-08-04

Status: device execution and performance evidence complete; numerical device
gate failed; not accepted as a product candidate.

## Decision

The project-owned 7.8-second FP32 HTDemucs neural core is practical on the
Galaxy S25 CPU as an offline or producer-ahead-of-playback experiment. The
final 180-second run completed 31 windows in 99.912 seconds, an end-to-end RTF
of 0.555. A timing projection retained 2.724 seconds of minimum refill margin
with one ready window and 8.593 seconds with two ready windows.

This is not a product qualification:

- the frozen single-window device-vs-host numerical gate failed on CPU and on
  GPU+CPU;
- GPU+CPU delegated only a small part of the graph and did not improve useful
  sustained performance;
- BoomingMusic Phase 3 N-stem playback and Phase 4 HTDemucs adapter are not
  implemented; and
- underrun evidence is a producer/consumer timing projection, not an
  `AudioTrack` observation.

Keep `readyWindowCount=2` for a first integration. The measured results make
one window plausible, but reducing the product default should wait for the
real playback data plane, device contention, and background execution tests.

## Identity

Device:

```text
Samsung SM-S9310 (Galaxy S25)
SoC: QTI SM8750
Android API: 35
ABI: arm64-v8a
ADB serial: private device endpoint; pass as <device-serial>
```

Runtime:

```text
com.google.ai.edge.litert:litert:2.1.5
AAR SHA-256:
a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37
```

Source state embedded in the final APK:

```text
revision: 3510d5c263ab8236cd96cfeb07feef1c1a5c0dff
dirty: true
branch: experiment/model-matrix-demucs-multistem
```

Canonical artifact:

```text
modelId: htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0
file: htdemucs_6s.core.canonical_7p8s.fp32.tflite
bytes: 117,624,880
SHA-256:
8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7
```

ABI:

```text
args_0    [1,2,343980]
args_1    [1,4,2048,336]
output_0  [1,6,4,2048,336]
output_1  [1,6,2,343980]
stem order: drums,bass,other,vocals,guitar,piano
```

The generated host manifest is 47,505 bytes with SHA-256
`e22708ecbb1e43f528a3f1ff2ab33a8062c42fc36ffed1134f837426865f33e2`.
The Android loader now parses and cross-validates its complete host DSP
contract instead of only checking that the object exists.

## Host DSP Boundary

JTransforms was checked directly against the frozen Torch fixtures:

| Layer | SNR | Maximum absolute error | RMS ratio |
| --- | ---: | ---: | ---: |
| waveform to spectrum | 133.606 dB | `4.768e-7` | 0.999999982 |
| frequency output to waveform | 133.544 dB | `1.192e-7` | 0.999999931 |
| frequency plus time branch | exact | 0 | 1.0 |

All three pass the `80 dB / 1e-3` fixture gate. The streaming planner also
tests centered `TensorChunk` padding, odd-delta crop direction, triangle-prefix
weighting, FP32 ascending OLA, global normalization, and the exact two-window
frozen plan.

The frozen Python export evidence remains separate: Torch core OLA versus
official `apply_model` was bitwise equal; LiteRT versus Torch full OLA was
98.427 dB with maximum error `5.55e-6`, and the lowest full-track per-stem SNR
was 85.356 dB.

## Neural-Core CPU

The cold single-window probe completed with finite outputs:

```text
prepare: 489.14 ms
inference: 2163.16 ms
point PSS: 1,006,358 KiB
native allocated: 970,637,216 bytes
XNNPACK: 3271 / 3432 nodes, 75 partitions
```

The clean sustained pass used two warmups and ten measured invocations on one
`CompiledModel`:

| Metric | Wall time |
| --- | ---: |
| minimum | 2083.07 ms |
| median | 2123.77 ms |
| mean | 2121.88 ms |
| P95 / maximum | 2152.27 ms |

Neural-core mean RTF is approximately `2.122 / 7.8 = 0.272`.

The sampled sustained pass observed:

```text
peak total PSS:       1,060,917 KiB
peak total RSS:       1,144,196 KiB
peak native heap PSS:   821,318 KiB
graphics PSS:                  0 KiB
minimum MemAvailable: 4,255,712 KiB
minimum SwapFree:     5,067,140 KiB
```

## Neural-Core GPU+CPU FP32

This was a real hybrid graph, not strict GPU execution:

```text
OpenCL: 158 / 3432 nodes, 3 partitions
XNNPACK remainder: 3115 / 3275 nodes, 77 partitions
```

Sustained two-warmup/ten-measured results:

| Metric | CPU | GPU+CPU FP32 |
| --- | ---: | ---: |
| prepare | 547.93 ms | 1593.43 ms |
| median inference | 2123.77 ms | 2108.03 ms |
| mean inference | 2121.88 ms | 2152.75 ms |
| P95 / maximum | 2152.27 ms | 2483.22 ms |
| sampled peak PSS | 1,060,917 KiB | 1,772,181 KiB |
| sampled graphics PSS | 0 KiB | 451,792 KiB |

The median improvement was only about 0.7 percent, while mean, tail latency,
prepare time, and memory were worse. Do not carry this GPU profile into a
product path. Strict GPU and FP16 remain out of scope for this route.

## Numerical Device Gate

Both device profiles produced finite tensors, but both final statuses are
`failed` under the layered per-stem gate.

| Layer aggregate | CPU | GPU+CPU FP32 |
| --- | ---: | ---: |
| raw frequency SNR | 81.352 dB | 82.824 dB |
| frequency iSTFT SNR | 79.686 dB | 81.503 dB |
| time waveform SNR | 107.842 dB | 104.998 dB |
| combined waveform SNR | 82.404 dB | 84.199 dB |

CPU failed stems:

```text
combined: other, vocals
frequency iSTFT: guitar, other, vocals
time waveform: drums
```

GPU+CPU FP32 failed stems:

```text
combined: other, vocals
frequency iSTFT: other, vocals
time waveform: drums
```

The raw frequency layer passed its special latent gate on both profiles. The
failures above are low-signal absolute-error failures or, for CPU guitar
iSTFT, the normal SNR gate. Aggregate SNR must not hide those stem failures.

## Real-Song E2E

Input was derived from `data/samples/_-_Coast_Town__decoded.wav`, a real 44.1
kHz stereo PCM16 song. Exact test inputs:

```text
30 seconds, 5,292,044 bytes
SHA-256 940c2ad55fad34a52ba7660706150300cfea84d315c491012e54de41d653aa6b

180 seconds, 31,752,044 bytes
SHA-256 b39fd8b9d1c506a37ddea5cc1189d0af24f7446d53e39f655b3d0e05a64adf97
```

The harness performs one whole-track stereo-reference mean/sample-standard-
deviation scan, random-access normalized window reads, canonical STFT, both
LiteRT inputs, frequency iSTFT, time-branch addition, exact 25 percent OLA,
inverse normalization, PCM16 conversion, and six atomic streaming WAV writes.
It retains only the OLA carry rather than all six tracks in Java memory.

Final digest-pinned runs:

| Metric | 30 seconds | 180 seconds |
| --- | ---: | ---: |
| windows | 6 | 31 |
| emitted frames | 1,323,000 | 7,938,000 |
| total wall time | 19.935 s | 99.912 s |
| end-to-end RTF | 0.664 | 0.555 |
| core inference total | 12.691 s | 66.097 s |
| STFT total | 0.624 s | 1.863 s |
| iSTFT total | 2.827 s | 13.713 s |
| branch combine total | 0.194 s | 0.998 s |
| OLA total | 0.171 s | 0.906 s |
| PCM conversion total | 2.207 s | 12.910 s |
| WAV write total | 0.014 s | 0.087 s |

The 180-second inference distribution was 2092.54 ms minimum, 2130.78 ms
median, and 2216.76 ms maximum. All six output WAVs had the exact frame count,
zero non-finite samples, and zero clipped samples.

The pinned 180-second sampled run observed:

```text
peak total PSS:       1,128,721 KiB
peak total RSS:       1,214,192 KiB
peak native heap PSS:   821,624 KiB
graphics PSS:                  0 KiB
minimum MemAvailable: 4,230,636 KiB
minimum SwapFree:     5,122,948 KiB
Java heap maximum:      268,435,456 bytes
```

ADB sampling perturbs the run, so the clean 30-second run is the cleaner short
latency baseline. The 180-second sampled result is still stable and slightly
faster than the earlier unpinned diagnostic pass.

## EOF, Seams, and Seek

The 30-second input covers the difficult shortened penultimate and final
windows:

```text
window 4: actual=291060, crop=26460/26460, next carry=33075
window 5: actual=33075, crop=155452/155453, right pad=155453
```

This found and fixed an early harness defect that incorrectly retained the
full 85,995-sample carry after a shortened penultimate window.

The output analyzer performed 18 random seeks per run: start, midpoint, and
last 4096 frames for all six stems. All WAV shape and read checks passed.

For every stride boundary it compared the boundary PCM16 delta with the local
derivative P95. The 180-second maximum ratios were:

| Stem | Boundaries | Maximum ratio |
| --- | ---: | ---: |
| drums | 30 | 1.000 |
| bass | 30 | 1.015 |
| other | 30 | 1.394 |
| vocals | 30 | 1.500 |
| guitar | 30 | 1.068 |
| piano | 30 | 1.125 |

No boundary is an isolated large derivative relative to its local audio. This
is objective seam evidence, not a listening-quality verdict.

## Cancel and Resume

An intentional cancellation after two completed windows produced 515,970
frames in staging, closed the LiteRT session, and removed the entire partial
output directory. A new session then restarted at window zero and completed
all six stems and 1,323,000 frames.

The tested resume policy is therefore:

```text
cancel -> discard partial stem set -> compile a new session -> restart at zero
```

It is not checkpoint continuation. One initial attempt was externally killed
when a parallel instrumentation launch force-stopped the package; that run is
retained as test-infrastructure collision evidence and excluded from product
behavior. The isolated retry passed.

## Playback Buffer Projection

The projection uses each measured window's complete production wall time and a
real-time consumer. It does not invoke `AudioTrack`.

| Run | Ready windows | Audio at start | Producer wall to start | Minimum refill margin | Underruns |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30 s | 1 | 5.85 s | 4.326 s | 2.731 s | 0 |
| 30 s | 2 | 11.70 s | 7.445 s | 8.593 s | 0 |
| 180 s | 1 | 5.85 s | 4.282 s | 2.724 s | 0 |
| 180 s | 2 | 11.70 s | 7.407 s | 8.593 s | 0 |

`readyWindowCount=2` remains the conservative integration setting. One window
has enough margin in these isolated S25 runs, but the product still needs
contention, service lifecycle, audio routing, and actual playback-underrun
tests.

## Thermal Limitation

Android thermal severity remained `0`. `dumpsys thermalservice` exposed
duplicate AP/BAT/SKIN sensor names, cached primary values, and inconsistent
secondary readings. It is not a trustworthy temperature-rise curve. Only the
severity result is accepted; no temperature delta is claimed.

## Evidence

Neural-core results:

```text
outputs/htdemucs-canonical7p8-s25-20260804/
```

Final E2E results:

```text
outputs/htdemucs-canonical7p8-s25-e2e-20260804/
  30s-cpu-pinned-clean-final/
  180s-cpu-pinned-sampled/
  30s-cancel-resume-retry/
```

Pinned evidence digests:

```text
30s report:   c39dd00663de3392b962f818ede1244841394312cb8a50b9004b36d1c75c0678
30s analysis: 2bacd977b5b51de7b938e37447c37a68d8c9486a3f7a0935a03b93bb45344513
180s report:  f24989b2c54736c1f568d52b04bba65daf44d68b7e5a6d6472830b019abb306a
180s analysis: a7af3ace98722828d14e166bdc0bf0d39edec5570332c5002e6ccd66965f96d5
```

The analyzer output beside each run records WAV seeks, all seam measurements,
and both playback-buffer projections. `tools/analyze_htdemucs_e2e_outputs.py`
reproduces those checks.

## Product Boundary

This experiment justifies continuing a CPU-only HTDemucs adapter prototype.
It does not justify exposing a Demucs preset. Required next evidence is:

1. implement the roadmap's N-stem cache/hydration/playback data plane;
2. run the real `AudioTrack` path with seek, cancellation, service restart,
   background contention, and underrun counters;
3. investigate the frozen device numerical failures before quality acceptance;
4. compare actual rendered audio to a Torch full-song reference; and
5. optimize PCM conversion and iSTFT, the two largest non-core costs.

QNN is intentionally excluded from this experiment.
