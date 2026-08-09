# MDX native DSP risk-four App Live package

## Published artifact

The ARM64 contract-v3 package is published locally for BrowserStack App Live
under `C:\Users\User\Documents\BSSUploadRelay\app-live-apks\`.

| Field | Value |
| --- | --- |
| APK | `BSS-AppLive-MDX-DSP-Risk4-v3-89355d1c.apk` |
| Bytes | `98,046,850` |
| SHA-256 | `dff29c749814691fcebf93e5d8c7ccfa4a209eece44487bf3a91b51bd6cd3b6c` |
| Package | `com.example.musicsourceseparation.dspshapeabi` |
| ABI | `arm64-v8a` |
| Source commit | `2aca162099c884a561c93c9e31db946980862e5f` |
| Source dirty | `false` |
| Bundle ID | `89355d1c63c5b05651f9f95f7e3c25638238440ef605e0a2616a4f82d8945713` |
| Campaign | `mdx-dsp-risk4-arm64-applive-v3` |

`apksigner` verifies APK Signature Scheme v2 with one debug signer. The signer
certificate SHA-256 is
`8134730b8e406e138b1cfc252fd197c42e18f98ac74714d25c672ff0545d322f`.
The package directory also contains an independent checksum, build-info file,
and App Live procedure.

## Matrix contract

The package compares Kotlin/JTransforms and native packed-real for four worker
threads, two warmups, and five balanced alternating measured rounds. It covers:

- 4096 / 2048 / 128;
- 4096 / 2048 / 512;
- 6144 / 3072 / 512;
- 16384 / 2048 / 512.

Measurements and numerical checks finish before relay traffic begins. A
successful run uploads eight files and exposes eight flattened summary rows.

## S25 collection-path control

The published APK was installed from the relay package directory onto the S25,
application data was cleared, and automatic execution was started through the
launcher activity. It produced terminal `complete.json`, deferred upload until
after all measurements, and was then downloaded and SHA-verified from the
relay without remote cleanup.

The strict contract-v3 merger accepted one run and eight rows; all eight rows
qualified.

| Shape | Packed median | Packed P95 | Speedup vs Kotlin |
| --- | ---: | ---: | ---: |
| 4096 / 2048 / 128 | 7.95 ms | 10.97 ms | 3.99x |
| 4096 / 2048 / 512 | 32.11 ms | 44.23 ms | 5.88x |
| 6144 / 3072 / 512 | 51.91 ms | 52.57 ms | 4.02x |
| 16384 / 2048 / 512 | 71.74 ms | 78.91 ms | 4.93x |

The matrix took 19.002 seconds. Thermal status remained `0/0`, battery
temperature rose 0.2 C, and process PSS rose 64.6 MiB. The lowest packed STFT
and iSTFT SNR were 133.721 dB and 137.446 dB. The complete retained evidence is
under
`C:\Users\User\Documents\BSSUploadRelay\results\mdx-dsp-risk4-arm64-applive-v3\`.

## Use

This package is ready for additional fresh-install arm64 App Live devices.
Further runs should use the frozen APK rather than rebuilding the branch. The
upload token is write-only and should be rotated after the campaign ends.
