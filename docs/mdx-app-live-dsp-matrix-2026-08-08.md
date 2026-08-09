# MDX App Live DSP matrix

## Purpose

This branch adds a self-contained arm64 App Live package for comparing the
shared MDX DSP implementations without model-inference noise. A fresh install
automatically runs the matrix once and defers all relay traffic until every
timed sample and numerical check has completed.

The frozen shapes are:

| Sentinel | FFT | Hop | dimF | Frames |
| --- | ---: | ---: | ---: | ---: |
| UVR MDXNET 3 9662 | 6144 | 1024 | 2048 | 256 |
| Kim Inst | 7680 | 1024 | 3072 | 256 |
| UVR MDXNET Inst HQ4 | 5120 | 1024 | 2560 | 256 |

Each shape compares four-worker Kotlin/JTransforms, native pocketfft
full-complex, and native pocketfft packed-real. The runner uses two warmups and
ten measured runs per profile in a balanced cross-over order. STFT consumes a
deterministic stereo multisine, while every iSTFT consumes the same frozen
Kotlin NHWC tensor. Native parity must remain finite, reach at least 80 dB SNR,
and stay within `1e-3` maximum absolute error.

## Artifact

The current package is contract v2. Contract v1 remains historical and should
not be used for new BrowserStack runs.

| Field | Value |
| --- | --- |
| Branch | `experiment/mdx-app-live-dsp-matrix` |
| Source commit | `4a26d2850f97f98fa6fc4338249dedd425a03bda` |
| Source dirty | `false` |
| Variant | `dspMatrixDebug` |
| Package | `com.example.musicsourceseparation.dspmatrix` |
| Contract | 2 |
| Bundle ID | `eeccfead0c30e7022afdecace644e82bb8bbb2dda8a8e27988d4bb676ec7ebc0` |
| APK bytes | 97,887,073 |
| APK SHA-256 | `56d6499307ea9b42f7ad25f12fa0e859adf098121a5c50e99ec8ebcea4278a65` |
| Campaign | `app-live-mdx-dsp-matrix-v2` |

The private package, checksum, build metadata, and App Live procedure are under
`C:\Users\User\Documents\BSSUploadRelay\app-live-apks\` with prefix
`BSS-AppLive-MDX-DSP-Matrix-v2-eeccfead`.

Successful v2 runs upload eight files: `artifact-manifest.json`, `identity.json`,
the full `dsp-matrix-report.json`, batch-ready JSON and CSV summaries,
process-filtered logcat, the application log, and terminal `complete.json`.
Identity includes exact APK/native-library hashes,
source and bundle IDs, firmware, SoC, ABI, and memory. Reports include every
raw sample, P50/P95, parity, PSS/native heap, ART allocation/GC, thermal, and
battery evidence. A write-only temporary relay credential is packaged; the
relay read credential is not present.

The summary JSON contains nine fully flattened shape/profile rows, an explicit
qualification count, native winner counts, and a uniform-winner field. The CSV
contains the same rows under a frozen 74-column header. Each row repeats the
run, device, fingerprint hash, artifact identity, timing, parity, speedup,
memory, GC, and thermal data needed for direct concatenation. `complete.json`
pins both summary files by byte count and SHA-256.

`tools/summarize_app_live_dsp_matrix.py` discovers downloaded v2 summaries and
writes combined JSON/CSV. It rejects wrong contracts, incomplete or failed
runs, missing rows, sample-count changes, field drift, and conflicting
duplicate identities.

## Final-artifact S10 control

The v2 exact-artifact control is
`local-sm-g9730-dsp-matrix-20260809T024326Z-eeccfead`. It uploaded eight files,
reported 9/9 qualified rows, generated a 74-column CSV, and merged to one run
and nine rows without reading the raw report. The APK SHA, summary JSON/CSV SHA,
bundle, clean source commit, and terminal evidence all match. Logcat contains no
fatal exception, ANR, OOM, or native abort.

The table below is the earlier v1 S10 performance control; v2 does not change
the DSP algorithm or measurement order.

Run `local-sm-g9730-dsp-matrix-20260809T015731Z-b8dfcf52` completed in
32.282 seconds on the Galaxy S10 / SM8150. Relay download verified all six file
hashes. The device-reported base APK hash exactly matches the published APK,
and the report records the clean source commit above.

| Shape | Profile | STFT P50 | iSTFT P50 | Combined P50 / P95 |
| --- | --- | ---: | ---: | ---: |
| 9662 | Kotlin | 46.47 ms | 137.27 ms | 188.73 / 264.98 ms |
| 9662 | native-full | 33.01 ms | 36.60 ms | 69.26 / 85.24 ms |
| 9662 | native-packed | 23.06 ms | 36.53 ms | 59.08 / 75.93 ms |
| Kim Inst | Kotlin | 54.63 ms | 187.94 ms | 241.94 / 366.73 ms |
| Kim Inst | native-full | 37.80 ms | 48.25 ms | 86.84 / 103.48 ms |
| Kim Inst | native-packed | 26.44 ms | 48.93 ms | 75.18 / 97.72 ms |
| HQ4 | Kotlin | 52.67 ms | 253.75 ms | 294.65 / 325.67 ms |
| HQ4 | native-full | 27.56 ms | 35.04 ms | 62.42 / 78.57 ms |
| HQ4 | native-packed | 27.74 ms | 37.76 ms | 65.14 / 75.89 ms |

Kotlin is the bit-exact reference. Native STFT SNR spans 133.59-135.69 dB and
native iSTFT SNR spans 136.82-137.62 dB. All maximum errors remain below
`1e-3`. Thermal status stayed `0`; battery temperature moved from 30.4 to
32.0 C. Process PSS moved from 120,273 to 156,851 KiB after all three shapes;
this is an end snapshot after released plans, not a peak-memory claim.

## Campaign result

The final contract-v2 artifact completed an 81-run campaign spanning 77 unique
firmware fingerprints, 72 device models, nine manufacturers, and 37 reported
SoCs. The complete analysis and final uniform native-packed product decision
are recorded in
[`mdx-app-live-dsp-matrix-v2-batch-results-2026-08-08.md`](mdx-app-live-dsp-matrix-v2-batch-results-2026-08-08.md).
