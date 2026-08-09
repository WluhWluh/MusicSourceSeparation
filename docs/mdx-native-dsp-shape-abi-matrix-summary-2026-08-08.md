# MDX native DSP shape and ABI matrix summary

## Canonical batches

Contract v3 extends the earlier three-shape ARM64 campaign to every known MDX
DSP shape and the remaining Android ABIs. It compares the product candidate,
native pocketfft packed-real, directly with Kotlin/JTransforms. Native-full is
not part of these batches.

| Campaign | Target | Shapes / workers | Runs | Qualified rows |
| --- | --- | --- | ---: | ---: |
| `mdx-dsp-shape-remaining10-arm64-v3` | S25 + S10 ARM64 | remaining 10 / 4w | 2 | 40 / 40 |
| `mdx-dsp-risk4-arm64-applive-v3` | S25 collection control | risk 4 / 4w | 1 | 8 / 8 |
| `mdx-dsp-all13-x86_64-v3` | x86_64 emulator | all 13 / 4w | 1 | 26 / 26 |
| `mdx-dsp-risk4-armeabi-v7a-v3-r2` | S10 32-bit process | risk 4 / 2w + 4w | 1 | 16 / 16 |
| `mdx-dsp-x86-3-v3-r2` | API 26 x86 emulator | control + maximum tensor/FFT / 4w | 1 | 6 / 6 |
| **Total** | four ABIs | 13 unique shapes | **6** | **96 / 96** |

The same strict merger accepted all five campaign directories together. The
combined ignored output is under `.tmp/mdx-dsp-shape-abi-canonical-merged/`.
Every relay run was downloaded with file SHA verification and retained on the
remote relay.

Two non-canonical attempts remain as failure-stage evidence. The first arm32
run completed DSP but mislabeled the summary ABI because it reported the first
device-supported ABI. Commit `9ac916c` added the active process ABI. The first
API 26 x86 attempt stopped before DSP because `PackageInfo.longVersionCode`
was unavailable; commit `2c57145` added the API 26 fallback. Both were rerun in
new campaigns after the fixes.

## Numerical envelope

There are 48 canonical native-packed rows. All are finite and all pass the
80 dB SNR and `1e-3` maximum-absolute-error gates.

| Metric | Worst canonical result |
| --- | ---: |
| Packed STFT SNR | 133.6109179 dB |
| Packed iSTFT SNR | 136.8495527 dB |
| STFT maximum absolute error | `9.1552734375e-5` |
| iSTFT maximum absolute error | `7.45058059692383e-8` |

The ten newly covered ARM64 shapes favor native-packed over Kotlin on both S25
and S10. Speedup ranges are 3.60-7.86x on S25 and 2.98-9.52x on S10. The
separate risk-four S25 package reproduced 3.99-5.88x. These are DSP-only
comparisons and must not be presented as complete model RTF speedups.

x86_64 completed all 13 shapes. Its 15.48-36.80x ratios reflect emulator and
JTransforms scheduling behavior and are correctness evidence only. On
armeabi-v7a, four workers are tied with two workers for 4096/128 and improve
the three long/high-load shapes by 1.18-1.25x. The x86 control and two maximum
shapes also complete, but x86 remains test-only.

## Frozen packages

The following ignored binary artifacts and independent checksum files are
retained under `C:\Users\User\Documents\BSSUploadRelay\app-live-apks\`:

| ABI / purpose | Artifact |
| --- | --- |
| ARM64 BrowserStack risk four | `BSS-AppLive-MDX-DSP-Risk4-v3-89355d1c.apk` |
| x86_64 all 13 | `BSS-MDX-DSP-All13-v3-x86_64-60f0e7ec.apk` |
| armeabi-v7a risk four | `BSS-MDX-DSP-Risk4-v3-armeabi-v7a-758b2516.apk` |
| x86 three sentinels | `BSS-MDX-DSP-X86-3-v3-a1952e52.apk` |

All four local APK hashes match their `.sha256` files. The ARM64 package also
has full build metadata and a BrowserStack procedure. It completed the S25
install, automatic run, deferred upload, download, and strict-merge path.

## Product decision

1. Use one contract-derived native packed-real implementation for all 13 MDX
   DSP shapes on ARM64. No shape-specific FFT algorithm or full/packed device
   selector is justified.
2. Use native packed-real for all 13 shapes in x86_64 emulator CI. Do not use
   emulator latency as a product threshold.
3. Native packed-real is functionally valid on armeabi-v7a and four workers
   remain the preferred common worker policy. Keep arm32 model support gated by
   LiteRT allocation, complete-song memory, and low-memory recovery.
4. Keep x86 as legacy emulator compatibility only. DSP success does not fix
   the existing 2 GiB HQ4 model-allocation failure.
5. Retain Kotlin/JTransforms as the conservative fallback for native library
   load, plan creation, or execution failure. Native-full remains a research
   numerical reference, not a product runtime policy.

This experiment qualifies the shared DSP stage, not the 30 model artifacts.
Weights, LiteRT invoke, backend delegation, stems, complete-song joins, and
application lifecycle must continue to use their own qualification evidence.
