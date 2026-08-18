# TFC-TDF readFloatInto experiment

Status: complete as a local benchmark experiment. The product runtime was not
changed and no runtime release was published.

## Change

`TfcTdfStreamingFullChainInstrumentedTest` and
`run_tfc_tdf_streaming_benchmark.ps1` now accept `-OutputRead allocating|reuse`.
The `reuse` mode allocates one output array per live LiteRT session and calls
the local `TensorBuffer.readFloatInto(FloatArray)` experiment API. The default
`allocating` mode continues to call the existing `readFloat()` API.

Both full-chain and two-slot reports record `outputReadMode` and output-read
wall time. Existing output ordering and finite-sample checks remain enabled.

Example A/B commands:

```powershell
.\tools\run_tfc_tdf_streaming_benchmark.ps1 `
  -Test double-buffer -Backend gpu-bounded `
  -DspProfile native-packed -DspWorkers 4 -Postprocess fused `
  -OutputRead allocating

.\tools\run_tfc_tdf_streaming_benchmark.ps1 `
  -Test double-buffer -Backend gpu-bounded `
  -DspProfile native-packed -DspWorkers 4 -Postprocess fused `
  -OutputRead reuse
```

## Results

The local runtime AAR was LiteRT `2.2.0-bss.2` plus the arm64
`readFloatInto` experiment. On the S25 bounded GPU, output-read time fell from
2.655 to 0.612 ms per sequential window and from 1.749 to 0.295 ms per
double-buffered window. The corresponding double-buffer means were 85.351 ms
with allocation and 82.968 ms with reuse; output ordering passed in both.

On S25 CPU, output-read time was 0.834/1.071 ms with allocation versus
0.155/0.142 ms with reuse (sequential/double-buffered). On S10 CPU it was
0.977/2.949 ms versus 0.231/0.800 ms. Total CPU window time remained governed
by inference and scheduling.

The full-chain S25 GPU smoke measured 720.76 ms versus 720.96 ms for first-wet
and 233.23 ms versus 210.94 ms for seek re-wet (allocating versus reuse). These
single-run values do not establish a seek-latency improvement.

## Evidence

Device reports remain in the ignored local output tree:

```text
outputs/tfc-tdf-streaming-s25/s25-gpu-readfloat-alloc-r1-20260818/report.json
outputs/tfc-tdf-streaming-s25/s25-gpu-readfloat-reuse-r1-20260818/report.json
outputs/tfc-tdf-streaming-s25/s25-cpu-readfloat-alloc-r1-20260818/report.json
outputs/tfc-tdf-streaming-s25/s25-cpu-readfloat-reuse-r1-20260818/report.json
outputs/tfc-tdf-streaming-s25/s10-cpu-readfloat-alloc-r1-20260818/report.json
outputs/tfc-tdf-streaming-s25/s10-cpu-readfloat-reuse-r1-20260818/report.json
```

The local runtime contract and AAR identity are recorded in
`bss-litert-android/docs/read-float-into-experiment-2026-08-18.md`.
