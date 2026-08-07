# LiteRT 2.1.5 HTDemucs PCM16 profile comparison

Date: 2026-08-08

## Decision

Admit `fused-jni-neon` as the preferred experimental PCM profile. Reject
`fused-direct`: a direct `ByteBuffer` is slower than the reusable byte-array
scalar path on both devices. Keep `fused-reuse` as the portable reference and
non-ARM fallback until native packaging is integrated into the application
repository.

JNI NEON preserves the existing quantization rule byte-for-byte and reduces
the complete fused postprocess stage by 18.8% on S25 and at least 35% in every
S10 pair. Its median paired 30-second E2E improvement is about 2.4% on S25 and
2.2% on S10. Do not claim the larger mean E2E changes: inference/iSTFT
scheduling and later S10 thermal status dominate whole-run variance.

## Implementations

- `fused-reuse`: Kotlin scalar clamp, multiply, `roundToInt`, reusable
  `ByteArray`, and `RandomAccessFile.write`.
- `fused-direct`: the same Kotlin arithmetic, reusable direct little-endian
  `ByteBuffer`, and `FileChannel.write`.
- `fused-jni-neon`: reusable interleaved float workspace, JNI critical arrays,
  AArch64 NEON clamp/multiply/`floor(x + 0.5)`, and reusable `ByteArray` write.
- Non-AArch64 ABIs compile a scalar native fallback; neither device used it.

The NEON rounding sequence is deliberately equivalent to Kotlin/JVM
`roundToInt` for finite values after clipping. It does not use nearest-even
conversion.

## Frozen identity

Full E2E batch:

- Source revision: `ddf7f65f4ab726abcdaa7961739d5cb8cfcd5bce`
- Source dirty: `false`
- App APK: 144,539,168 bytes, SHA-256
  `2fae3930425abbddd854076ffa663317052b728db5e1b136fcdc7be1581df218`
- Test APK: 632,543 bytes, SHA-256
  `cb038c86cccbbe80a716870b1f36349916fe29d10ae05c79486f633219bc8c85`

Pure PCM microbenchmark:

- Source revision: `1504f36785d70f329b6d8878e0ef6fa2f46056bc`
- Source dirty: `false`
- App APK: 144,539,168 bytes, SHA-256
  `34e894464175fba4daad0d13e61978c1efc2394ab86c97d634122190d20a2a6b`
- Test APK: 632,562 bytes, SHA-256
  `3200740e1643a1b538070615ba2cf18f231bdaccc5363bfe91ea23b11ac5ba20`

Both use LiteRT 2.1.5 AAR SHA-256
`a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37`,
the same official six-stem model, four LiteRT CPU threads, four parallel iSTFT
workers, and the same 30-second Athletics II canonical PCM prefix.

## Pure PCM microbenchmark

The device test quantizes 515,970 deterministic samples (1,031,940 PCM bytes),
uses two warmup rounds and 20 measured rounds in rotating order, and compares
all output bytes.

| Device | Mode | Mean | Median | P95 | Relative to scalar mean |
| --- | --- | ---: | ---: | ---: | ---: |
| S25 | scalar byte array | 14.577 ms | 14.568 ms | 14.579 ms | baseline |
| S25 | direct ByteBuffer | 25.024 ms | 24.980 ms | 25.437 ms | 71.7% slower |
| S25 | JNI NEON | 0.160 ms | 0.161 ms | 0.171 ms | 91.2x faster |
| S10 | scalar byte array | 9.800 ms | 9.798 ms | 9.898 ms | baseline |
| S10 | direct ByteBuffer | 14.596 ms | 14.603 ms | 14.728 ms | 48.9% slower |
| S10 | JNI NEON | 0.287 ms | 0.282 ms | 0.300 ms | 34.1x faster |

This isolates quantization and buffer writes, not file I/O or OLA. It confirms
that the previously observed S25 scalar PCM cost is device/runtime-specific
and reproducible; S25 scalar conversion is about 49% slower than S10 for the
same values and code.

## Full 30-second results

Three independently launched runs per profile and device were executed with
rotated ordering. Each run contains six model windows.

| Device | Mode | Fused postprocess mean | Paired change vs scalar | Median RTF | Mean sampled PSS |
| --- | --- | ---: | ---: | ---: | ---: |
| S25 | scalar byte array | 2349.8 ms | baseline | 0.582 | 1100.5 MiB |
| S25 | direct ByteBuffer | 2730.2 ms | 16.2% slower | 0.593 | 1099.9 MiB |
| S25 | JNI NEON | 1907.9 ms | 18.8% faster | 0.568 | 1112.2 MiB |
| S10 | scalar byte array | 1006.2 ms | baseline | 1.564 | 1083.2 MiB |
| S10 | direct ByteBuffer | 1196.0 ms | noisy; generally slower | 1.524 | 1084.9 MiB |
| S10 | JNI NEON | 510.7 ms | 35.2-62.5% faster | 1.529 | 1095.9 MiB |

S25 paired JNI postprocess changes are -17.8%, -19.4%, and -19.3%. S10 changes
are -35.2%, -62.5%, and -37.1%; the middle scalar run was a scheduling outlier,
but JNI is faster in every pair. Median paired E2E changes are -2.4% on S25 and
-2.2% on S10. Direct E2E has no credible benefit.

S25 remained at thermal status 0. S10 was 0 in the first triplet and reached
status 1 in later triplets. This is why target-stage and pure-microbenchmark
results, rather than S10 whole-run means, determine the decision.

The JNI profile adds approximately 12 MiB sampled PSS because it stages one
interleaved float workspace per stem. ART allocation and GC remain essentially
the same as the scalar fused profile. A follow-up native encoder can read the
existing two planar channel regions directly, removing the staging buffers
while retaining the same quantization kernel.

## Numerical gate

- A 100,003-value edge/random device test passes byte-for-byte on S25 and S10.
- The pure microbenchmark confirms all three output buffers are identical.
- All 18 full E2E reports complete with finite six-stem output.
- All nine S25 reports share one per-stem WAV SHA set.
- All nine S10 reports share one per-stem WAV SHA set.
- Both devices report native implementation `neon-aarch64`.

## Next step

Retain `fused-jni-neon` and remove its interleaved float staging by accepting
two planar channel offsets in JNI. Re-run only the PCM parity test, pure
microbenchmark, and one 30-second A/B per device. This should recover roughly
12 MiB PSS; performance may improve slightly by eliminating the Kotlin staging
copy.

Packed real-IFFT remains a separate experiment because it changes floating
point rounding before PCM quantization and requires waveform/join numeric
gates, not only byte identity at the PCM encoder.

## Evidence

- `outputs/demucs-pcm-abc-s25/`
- `outputs/demucs-pcm-abc-s25-rep2/`
- `outputs/demucs-pcm-abc-s25-rep3/`
- `outputs/demucs-pcm-abc-s10/`
- `outputs/demucs-pcm-abc-s10-rep2/`
- `outputs/demucs-pcm-abc-s10-rep3/`
