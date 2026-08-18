# TFC-TDF DSP optimization Phase 6

Status: the two-slot host/inference pipeline is implemented as an isolated
Android benchmark. It overlaps preparation of the next window with LiteRT
execution of the current window, verifies output ordering, and is not yet
connected to `NonCausalStreamingSeparatedPlaybackEngine`.

## Pipeline

The benchmark owns two independent sets of:

- LiteRT input and output `TensorBuffer` objects;
- native TFC-TDF DSP plans;
- STFT tensor and postprocess workspaces.

The main test thread prepares the next slot while a resident single-thread
executor runs the current LiteRT invocation. The executor remains single
threaded because one compiled model session must not receive concurrent calls.
After the invocation future completes, the test postprocesses that slot and
publishes no result until the previous slot has been consumed. This overlaps
DSP input preparation with inference without changing logical output order.

The two slot inputs are deliberately different. Every double-buffered output
is compared with the corresponding sequential output, with a maximum absolute
error threshold of `2e-3`. A passing run therefore checks more than timing: it
also checks slot ownership and ordering.

The benchmark does not perform file I/O, blocking work, or LiteRT calls on an
audio callback. It is a measurement harness, not the product playback engine.

## Device measurements

Each run used two warmups and eight measured windows, native packed-real DSP,
fused iSTFT residual postprocess, and the same TFC-TDF model artifact.

| Device/backend | Setup | Sequential mean | Double-buffer mean | Change | Order | Thermal | GPU dispatches |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| S25 bounded GPU, 4 DSP lanes | 514.5 ms | 107.06 ms | 85.56 ms | -20.1% | pass | 0 -> 0 | 1,696 |
| S25 CPU, 4 DSP lanes | 15.7 ms | 469.39 ms | 461.40 ms | -1.7% | pass | 0 -> 0 | n/a |
| S10 CPU, 2 DSP lanes | 18.6 ms | 769.98 ms | 745.43 ms | -3.2% | pass | 0 -> 0 | n/a |

The largest absolute difference between sequential and double-buffered output
was `0` in all three runs. The small CPU changes are within the variability
expected from a short benchmark and do not justify making the pipeline a CPU
default. Earlier S10 measurements with four DSP lanes showed contention and a
large regression; the two-lane choice remains the safer S10 profile.

The S25 GPU result is consistent with the earlier roughly 18% improvement: the
GPU invocation is long enough for input preparation and part of postprocess to
be hidden. It is evidence for a GPU-specific streaming optimization, not a
claim that every model or device will gain the same percentage.

A four-window S25 GPU repeat measured `107.82 ms` sequential versus `88.85 ms`
double-buffered (`-17.6%`), with output ordering verified and no thermal-status
change. The short repeat confirms the direction but is not combined with the
eight-window table above as a formal aggregate.

## Validation command

The reproducible entry point is:

```powershell
.\tools\run_tfc_tdf_streaming_benchmark.ps1 `
  -Test double-buffer `
  -Serial 192.168.8.197:35131 `
  -Backend gpu-bounded `
  -DspProfile native-packed `
  -DspWorkers 4 `
  -Postprocess fused
```

Use `-Backend cpu` and the device-specific lane count for the CPU comparison.
The script also retains the existing default `-Test full-chain` path.

## Product boundary

This phase does not change the product streaming engine. The benchmark still
has no seek epoch, cancellation, model switch, process-death recovery, or
playback-service lifecycle barrier around the two slots. Reusing this code in
the product without those boundaries could let an old invocation publish into
a new seek or model generation.

The next implementation step, if this optimization is selected, is a small
`PipelinedStreamingInferenceSession` abstraction with explicit epoch checks,
slot state transitions, cancellation, and output-order tests. It should be
enabled only for measured GPU profiles. The LiteRT `readFloat()` allocation is
a separate issue and still requires a supported runtime API; this phase does
not attempt to access internal tensor-buffer handles.

## Evidence

```text
outputs/tfc-tdf-streaming-s25/s25-gpu-double-buffer-r1-20260818/report.json
outputs/tfc-tdf-streaming-s25/s25-gpu-double-buffer-r2-20260818/report.json
outputs/tfc-tdf-streaming-s25/s25-cpu-double-buffer-r1-20260818/report.json
outputs/tfc-tdf-streaming-s25/s10-cpu-double-buffer-r1-20260818/report.json
```
