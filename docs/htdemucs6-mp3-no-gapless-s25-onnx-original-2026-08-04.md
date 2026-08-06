# S25 HTDemucs six-stem full-song three-way batch

Tracks: **8**; total audio: **36.46 min**.

All three backends consumed the exact PCM16 canonical decode exported by S25. The original safetensors/Torch FP32 path is the primary numerical reference.

## Performance

| Track | Duration (s) | Windows | S25 RTF | ONNX RTF | Original RTF | Peak S25 PSS (MiB) | Thermal max | Clipped samples |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Athletics - II.mp3 | 555.598 | 95 | 0.554 | 0.698 | 0.327 | 1086.9 | 0 | 414 |
| Joel Hanson - Traveling Light.mp3 | 208.744 | 36 | 0.626 | 0.437 | 0.335 | 1086.5 | 1 | 170 |
| John Lennon - Imagine.MP3 | 185.208 | 32 | 0.551 | 0.409 | 0.337 | 1086.5 | 0 | 0 |
| Josiah James - Chasing The Wind.MP3 | 219.664 | 38 | 0.599 | 0.411 | 0.321 | 1086.4 | 0 | 1366 |
| Kygo Ed Sheeran - I See Fire (Kygo Remix).mp3 | 316.891 | 55 | 0.592 | 0.414 | 0.372 | 1086.7 | 0 | 119 |
| Nylon - Eventide.mp3 | 201.064 | 35 | 0.623 | 0.409 | 0.344 | 1086.7 | 1 | 7 |
| Sleeping at Last - Already Gone.MP3 | 240.849 | 42 | 0.634 | 0.411 | 0.330 | 1086.2 | 0 | 0 |
| Sleeping at Last - North.mp3 | 259.422 | 45 | 0.596 | 0.430 | 0.334 | 1086.4 | 0 | 0 |

Aggregate separation RTF:

- S25 LiteRT CPU: `0.5907`
- Desktop ONNX CPU: `0.4887`
- Original safetensors/Torch CPU: `0.3373`
- S25 non-finite samples: `0`; clipped before PCM16 write: `2076`

## Numerical difference

| Pair | Track SNR min / median / max (dB) | Max error (LSB) | Min correlation |
|---|---:|---:|---:|
| s25-cpu_vs_original-safetensors-torch | 80.99 / 84.39 / 87.81 | 15 | 0.999999996 |
| desktop-onnx_vs_original-safetensors-torch | 64.84 / 70.49 / 73.16 | 108 | 0.999999836 |
| desktop-onnx_vs_s25-cpu | 64.99 / 70.64 / 73.43 | 107 | 0.999999842 |

The published desktop ONNX stores weights in FP16 and comes from a legacy `.th` export. Its difference from the original control is therefore not attributable only to ONNX Runtime. S25 uses the project-owned FP32 neural-core artifact generated from the official safetensors.

A track-level aggregate above 80 dB does not imply every low-signal stem passed 80 dB. Use the JSON report for per-stem SNR, absolute error, and correlation.

## Outputs

- Batch JSON: `outputs/htdemucs6-mp3-no-gapless-s25-onnx-original-20260804/batch-report.json`
- Blind audio: `outputs/htdemucs6-mp3-no-gapless-s25-onnx-original-20260804/blind`
- Blind key: private local file `docs/htdemucs6-mp3-no-gapless-s25-onnx-original-blind-map-2026-08-04.md` (ignored by Git)
