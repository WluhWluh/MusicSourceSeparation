# LiteRT 2.1.5 HTDemucs fused postprocess experiment

Date: 2026-08-07

## Decision

The parity-preserving `fused-reuse` profile passes the host and device gates.
It should replace the legacy implementation in the next optimization baseline,
but its measured E2E gain is modest: a stable 1.2-1.7% on S25 and a noisy
-0.6% to 4.8% on S10. The main durable result is 28.2% less ART allocation per
30-second run and faster postprocessing with byte-identical PCM output.

Do not attribute unrelated inference or iSTFT variation to this patch. The
profile does not change LiteRT, FFT arithmetic, windowing, overlap, model
outputs, or the six-stem contract.

## Identity and scope

- Branch: `experiment/demucs-rtf-optimization`
- Source revision: `1fe0b7ac3bdfaa0298fb10dd33aaf5c3874a7662`
- Source dirty: `false`
- LiteRT: `2.1.5`, AAR SHA-256
  `a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37`
- App APK SHA-256:
  `4fc074de9991aa5212b0d7086a8e2d44b51a9f4bda38e49063eecdd1496f32e5`
- Test APK SHA-256:
  `ab9736eb10bf35e73b0ffad807a7027e42736f9fbad1ef80af3f62c26328ba28`
- Model: official six-stem 7.8-second FP32 core, SHA-256
  `8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7`
- CPU profile: four LiteRT threads, four parallel iSTFT workers
- Input: first 30 seconds of the canonical Athletics II PCM
- Devices: S25 `SM8750` / API 35 and S10 `SM8150` / API 31
- Three independent legacy/optimized pairs per device with alternating order

## Implementation

The optimized profile:

- reuses the normalized waveform buffer;
- reuses STFT output, padding, and FFT buffers;
- reuses the iSTFT output buffer and existing per-worker FFT/overlap buffers;
- combines frequency and time branches while performing streaming OLA;
- performs inverse normalization, clipping, PCM16 encoding, and stem writes in
  the same postprocess pass;
- reuses one bounded PCM byte buffer per stem and writes only its active range.

The LiteRT 2.1.5 Java `TensorBuffer.readFloat()` API always creates a new
array. Frequency and time model-output arrays therefore remain unavoidable in
this implementation. Packed `realInverse`, native NEON FFT, reduced stem
materialization, and residual-derived outputs are not part of this batch.

## Paired results

| Device | Metric | Legacy mean | Optimized mean | Paired change |
| --- | --- | ---: | ---: | ---: |
| S25 | 30 s E2E | 17.696 s | 17.422 s | -1.55% mean; -1.22% to -1.74% |
| S25 | RTF median | 0.590 | 0.580 | -1.68% median pair |
| S25 | postprocess + write | 2542.9 ms | 2318.0 ms | -8.84% |
| S25 | ART allocation/run | 694.3 MiB | 498.3 MiB | -28.23% |
| S10 | 30 s E2E | 38.976 s | 38.298 s | -1.73% mean; +0.60% to -4.76% |
| S10 | RTF median | 1.304 | 1.290 | -1.01% median pair |
| S10 | postprocess + write | 860.0 ms | 616.4 ms | -28.33% |
| S10 | ART allocation/run | 694.3 MiB | 498.2 MiB | -28.25% |

S25 legacy postprocess was consistently dominated by PCM conversion:
branch combine 194 ms, OLA 171 ms, PCM 2166 ms, and writes 12 ms. S10 measured
312 ms, 184 ms, 341 ms, and 23 ms respectively. The earlier S25/S10 PCM
disparity is therefore reproducible under the same source, APK, input, and
contract; it is not explained by the old harness difference. Fusion reduces
S25 postprocess only about 9%, so a dedicated PCM conversion microbenchmark is
the next useful diagnostic.

The optimized profile did not materially reduce sampled PSS because the model,
LiteRT buffers, and native XNNPACK state dominate the approximately 1.1 GiB
working set. It reduced S10 ART GC counts from 42 to 36 across three runs; S25
remained 18 versus 18.

## Numerical gate

- Host reusable-workspace STFT and iSTFT match legacy raw float bits across
  three repeated calls.
- All 12 device runs completed with finite six-stem outputs.
- All six S25 reports share one identical per-stem WAV SHA set.
- All six S10 reports share one identical per-stem WAV SHA set.
- Legacy and optimized output byte identity therefore holds on both devices.
- Thermal status stayed 0 for every measured window.

## Interpretation

The optimization is accepted as a memory-allocation and postprocess
improvement, not as a 5-15% E2E win. On S25 it delivers a small but repeatable
E2E reduction. On S10, inference and JTransforms scheduling noise is larger
than the overall gain even though the targeted postprocess stage improves in
all three runs.

The next parity-preserving experiment should isolate S25 PCM16 conversion:
benchmark scalar `roundToInt`/clamp, a reusable direct `ByteBuffer`, and a JNI
NEON float-to-PCM kernel against the exact current quantization rule. Only
after that should packed real-IFFT be introduced as a separate rounding-changing
profile with host, device, full-song, and join-boundary numeric gates.

Selective-stem iSTFT is not used here because this remains a six-stem output
contract. It belongs to a separate vocals/accompaniment product contract; it
cannot be counted as a six-stem performance improvement.

## Evidence

- `outputs/demucs-postprocess-ab-s25/`
- `outputs/demucs-postprocess-ab-s25-rep2/`
- `outputs/demucs-postprocess-ab-s25-rep3/`
- `outputs/demucs-postprocess-ab-s10/`
- `outputs/demucs-postprocess-ab-s10-rep2/`
- `outputs/demucs-postprocess-ab-s10-rep3/`
