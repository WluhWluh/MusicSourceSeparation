# S25 LiteRT 2.1.5 HTDemucs 6-stem probe

日期: 2026-08-03 (host local time)

状态: 完成一次固定 7.8 秒 neural-core 窗口的 CPU、GPU 混合和 QNN
prepare 探索。结果只用于前置性能和可行性实验，不代表产品支持、整曲稳定性、
电量或分离质量。

本报告只记录官方 `com.google.ai.edge.litert:litert:2.1.5`。BandBuddy
原项目使用的 LiteRT 1.0.1/QNN 2.48 是另一套隔离运行时，不能把两者的 GPU
或 HTP 数字合并。

## 结论

* `CompiledModel + CPU/XNNPACK` 是本次原图中最快且数值稳定的路径: 单窗口
  `2194.12 ms`，4 threads，输出 finite。
* `GPU+CPU` 可以运行，但只覆盖 `158/3504` 个节点，剩余图由 XNNPACK 分成
  105 个 partition。FP32 的 AUTO、OpenCL、OpenGL 都通过 80 dB 数值门禁，
  但分别约 `2632/2534/2811 ms`，没有超过 CPU，因此暂不构成 GPU 性能方案。
* OpenCL FP16 (AUTO、显式 OpenCL、BUFFER) 的两路输出全部 non-finite。
  OpenGL FP16 虽 finite，但 combined SNR 只有 `39.18 dB`，不能接受。
* GPU-only 的 AUTO/OpenCL/OpenGL 都在 prepare 失败。日志显示 GPU 只覆盖
  `158/3504` 节点、3 个 partition；这是原图覆盖不足的 fail-closed，不是
  6022/OOM，也不是静默 CPU fallback。
* 后续把 132 个严格证明为恒等映射的 `GATHER_ND` 原地改写为 `RESHAPE`。
  派生图通过了 S25 Compiled CPU bitwise 门禁，GPU 覆盖升到 `1337/3504`；
  但 GPU+CPU FP32 的 combined SNR 为 `-5.20 dB`，因此判为数值失败。其
  `1738-1781 ms` 单窗口时延只代表错误输出下的性能上界，不能作为可用 GPU 结果。
* QNN HTP 确实加载并生成了非空 IR，选中 `3497/3504` 节点、8 个 partition，
  但 SM8750 的首个长序列 Conv2d 单算子需要 `0xdff800` TCM，设备只有
  `0x800000` (8 MiB)。HMX on、HMX off、HMX off + HVX1 在约 58 秒后均因同一
  单算子失败；没有开始推理。
* `HTP_OPTIMIZE_FOR_INFERENCE` 的探针运行到约 143 秒、host native heap 约
  4 GiB，仍未 finalize，已按资源上限停止，结论为 indeterminate；不能当作
  成功或失败。

因此当前推荐的可复现实验基线仍是 Compiled CPU。恒等 `GATHER_ND` 重写证明
了图障碍会显著压低 GPU 覆盖，但覆盖增加不等于数值正确。任何后续 rank-5
lowering 或短窗口契约仍必须逐个派生身份重复同一套 LiteRT 2.1.5 门禁。

## 固定身份

| 项目 | 值 |
| --- | --- |
| branch | `experiment/model-matrix-demucs-multistem` |
| source revision | `3510d5c263ab8236cd96cfeb07feef1c1a5c0dff`, dirty worktree |
| device | Samsung `SM-S9310`, SoC `SM8750`, Android 15/API 35, `arm64-v8a` |
| ADB | private device endpoint; pass as `<device-serial>` |
| model | `htdemucs_6s.core.tflite`, 117,784,760 bytes |
| model SHA-256 | `a9fcc89e84aa65313e0540b582e710007ed12064969a0d49a3c85e49f1ae4e3d` |
| contract | `bandbuddy_htdemucs_6s_core_v1_0_0.json`, 5,022 bytes |
| contract SHA-256 | `ce0dfb4481eeb97a5d8dc32554ce90bcd6d2786bfbd63766909f836794288e09` |
| LiteRT AAR | `com.google.ai.edge.litert:litert:2.1.5`, 10,058,192 bytes |
| LiteRT AAR SHA-256 | `a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37` |
| accelerator bundle SHA-256 | `8e20d10a6b27107fcb80460cc27892d5c2b797107c09ecbebeb095fbfe09c297` |
| app APK SHA-256 | `3545eb87a0fe46d17ac165fe50e2dafd61cfe6ee44e7ddb61db00362b98cacf4` |
| test APK SHA-256 | `fbe454e48c7b71172b577cf262eb0f7e735f0b501c8688837f4e1d95b9e19495` |

Native identity recorded by the probe:

* `libLiteRt.so`: 5,328,296 bytes,
  `366e3e040b00692158f9f8f9105870672c93348a3d8e9024120b40045a074b0b`
* `libLiteRtClGlAccelerator.so`: 2,728,920 bytes,
  `d22d9490c43a9428a6047564560dae83ce32a616658baa324b43843bfb066e89`
* Qualcomm compiler/dispatch and QNN V79 libraries are included in the result
  JSON; their hashes must be checked again when moving to another device image.

The probe uses the `serving_default` signature with two FLOAT32 inputs:

```text
args_0: [1, 2, 343980]
args_1: [1, 4, 2048, 336]
output_0: [1, 6, 4, 2048, 336]
output_1: [1, 6, 2, 343980]
```

Inputs are deterministic non-zero fixtures. Every run is a new instrumentation
process and writes a stage checkpoint before model creation, buffer allocation,
inference, and output reads.

## CPU baseline

| Candidate | Prepare wall | Inference wall | Inference CPU | Output |
| --- | ---: | ---: | ---: | --- |
| `interpreter-cpu-xnnpack` | 515.33 ms | 5853.68 ms | native 5825.03 ms | finite |
| `compiled-cpu-xnnpack` | 413.01 ms | **2194.12 ms** | 8423 ms | finite |

The two Compiled CPU outputs are bitwise identical to the Interpreter output for
this fixture. The Compiled CPU result is the reference for all GPU SNR values.

## GPU-only prepare

All strict candidates use `CompiledModel` with only `Accelerator.GPU`; they are
intentionally fail-closed. The observed common coverage is:

```text
GPU delegate: 158 / 3504 nodes, 3 partitions
```

Unsupported operations include `GATHER_ND`, `BROADCAST_TO`, rank-5
`RESHAPE/MUL/ADD`, and some `FULLY_CONNECTED` nodes. The following all failed at
`CompiledModel.create` with `LiteRtException: Failed to compile model`:

| Candidate | Backend/precision | Prepare result |
| --- | --- | --- |
| `compiled-gpu-auto-fp16` | AUTO, FP16 | fail |
| `compiled-gpu-auto-fp32` | AUTO, FP32 | fail |
| `compiled-gpu-opencl-fp16` | OpenCL, FP16 | fail |
| `compiled-gpu-opencl-fp32` | OpenCL, FP32 | fail |
| `compiled-gpu-opencl-buffer-fp16` | OpenCL BUFFER, FP16 | fail |
| `compiled-gpu-opencl-buffer-fp32` | OpenCL BUFFER, FP32 | fail |
| `compiled-gpu-opengl-fp16` | OpenGL, FP16 | fail |

The `WEBGPU` enum was also probed. Android LiteRT 2.1.5 did not create a
WebGPU-native backend; the log shows the OpenCL path instead. It is retained only
as a negative-control record, not as WebGPU evidence.

Adding more strict GPU flags cannot make the unchanged graph fully delegated:
the remaining nodes have no CPU fallback when `Accelerator.CPU` is absent.

## GPU plus CPU single-window runs

Each row is one 7.8-second fixture. `SNR` is against `compiled-cpu-xnnpack`; the
acceptance gate is 80 dB per output and combined. `finite` means no NaN/Inf, not
that the result is accurate.

| Candidate | Prepare | Inference | GPU backend | finite | combined SNR | Decision |
| --- | ---: | ---: | --- | --- | ---: | --- |
| `compiled-gpu-cpu-auto-fp16` | 1565.17 ms | 2260.12 ms | OpenCL, 158 nodes | no | n/a | numerical fail |
| `compiled-gpu-cpu-auto-fp32` | 1448.28 ms | 2632.21 ms | OpenCL, 158 nodes | yes | 109.90 dB | valid, slower |
| `compiled-gpu-cpu-opencl-fp16` | 1575.90 ms | 2361.50 ms | OpenCL, 158 nodes | no | n/a | numerical fail |
| `compiled-gpu-cpu-opencl-fp32` | 1545.15 ms | 2534.22 ms | OpenCL, 158 nodes | yes | 109.90 dB | valid, slower |
| `compiled-gpu-cpu-opencl-buffer-fp16` | 1754.05 ms | 2327.29 ms | OpenCL BUFFER, 158 nodes | no | n/a | numerical fail |
| `compiled-gpu-cpu-opencl-buffer-fp32` | 1456.64 ms | 2298.17-2352.09 ms | OpenCL BUFFER, 158 nodes | yes | 109.93 dB | valid, no gain |
| `compiled-gpu-cpu-opengl-fp16` | 1015.24 ms | 2766.53 ms | OpenGL, 158 nodes | yes | **39.18 dB** | parity fail |
| `compiled-gpu-cpu-opengl-fp32` | 992.19 ms | 2811.03 ms | OpenGL, 158 nodes | yes | 110.00 dB | valid, slower |

For the successful OpenCL FP32 BUFFER run, output SNR is `109.93 dB` and
`99.88 dB` for `output_0` and `output_1`; maximum absolute errors are
`5.18e-5` and `4.88e-7`. The exported repeat was stable within the reported
hashes. OpenGL FP16 is a useful diagnostic: finite output alone would have
incorrectly suggested success, while the raw comparison exposes a large error.

The logs show the same CPU remainder in every hybrid row:

```text
GPU: 158 / 3504 nodes, 3 partitions
XNNPACK: 3152 / 3347 nodes, 105 partitions
```

This partition count and the extra synchronization explain why none of the
hybrid rows beats the 2.19-second Compiled CPU baseline.

## Identity GATHER_ND derivative follow-up

The first exact graph rewrite was isolated behind a derived-model contract. It
is the in-place FlatBuffer patch, not the earlier append-shape-tensor prototype.

| Item | Value |
| --- | --- |
| rewrite tool | `tools/rewrite_tflite_identity_gather_nd.py` |
| tool SHA-256 | `514bd175c57bd461b80cffbab8c812a8de13daa70b0cdbc816184bf8dfdf66b1` |
| derived model | `htdemucs_6s.core.gather-reshape.tflite`, 117,784,760 bytes |
| derived SHA-256 | `ce971b195c20f1233c407b145ce8123f4d4b0834c6d946f0786504b999fb26a6` |
| derived contract | `bandbuddy_htdemucs_6s_core_gather_reshape_v1_0_0@1` |
| manifest SHA-256 | `de4c1caa72b3c471dcd4e3c6d54ac9c0453e4b79a4c3d2ae5d3fbc6935417b8b` |
| graph delta | 132 `GATHER_ND -> RESHAPE`, 14 shared shape tensors |
| graph totals | 3,504 operators, 4,332 tensors, 0 `GATHER_ND`, 1,073 `RESHAPE` |

The rewrite accepts an operator only when its data is the static one-dimensional
flattening of the output, INT32 indices are exactly `0..N-1`, the indices shape
is `output_shape + [1]`, and the constant tensor and inline buffer have no
unproven users. Regeneration with the pinned project `.venv` produced the same
derived SHA-256. A byte audit found all differences inside the expected opcode,
shape-vector, and buffer-vector fields.

### S25 gates

| Gate | Original | Derivative | Decision |
| --- | ---: | ---: | --- |
| Compiled CPU prepare | 468.36 ms | 338.89-387.10 ms | pass |
| Compiled CPU inference | 2192.62 ms | 2191.46-2192.91 ms | bitwise pass |
| strict OpenCL FP32 BUFFER | 158/3504 GPU nodes | 1337/3504, 3 partitions | compile fail |
| hybrid OpenCL FP32 BUFFER prepare | 1456.64 ms | 2727.89-2866.64 ms | correctness-blocked |
| hybrid OpenCL FP32 BUFFER inference | 2298.17-2352.09 ms | 1738.37-1781.15 ms | correctness-blocked |

Both derivative CPU output files exactly match the original CPU hashes:

```text
output_0  3b46317613de97922af3ecd8a5085d226d76f66b5bdf68846e0956691f12d421
output_1  5eedb65802959a51f6b1b8c0aea19b7024c9df04825af3f99209adec571a45f2
```

Strict GPU still fails `CompiledModel.create`. `GATHER_ND` is absent from the
unsupported-op list; the remaining diagnostics name rank-5 `ADD`, `MUL`, and
`RESHAPE`, `BROADCAST_TO`, and two unsupported `FULLY_CONNECTED` forms. The
hybrid run is proven to use the GPU by the `LITERT_CL` replacement line and
OpenCL delegate initialization, rather than by accelerator availability alone.

| Output | finite | SNR vs derivative CPU | RMSE | max abs error |
| --- | --- | ---: | ---: | ---: |
| `output_0` | yes | -5.20136 dB | 0.107192 | 9.41117 |
| `output_1` | yes | 10.91224 dB | 0.000390325 | 0.00959243 |
| combined | yes | -5.20079 dB | n/a | n/a |

The hybrid output hashes are `d0560b934d301259c8044ae84bf15f140099e6eb7484527a586983c06e068060`
and `fea008b8a26592383472b2e4c0a1a9b799815964e8e1db78fa387b2e1754e6d0`.
All values are finite, but every SNR gate is below the required 80 dB. The
observed 18.7-20.7% lower single-window inference latency is therefore rejected,
and no thermal or long-window performance batch was run.

Probe limitation: `status: complete` in the current result JSON means that
model execution and output export completed; it is not a quality verdict. The
80 dB decision above was computed from the exported raw tensors. Likewise,
`availableAccelerators: [CPU, GPU]` is not delegation proof; the proof is the
captured logcat line naming `LITERT_CL`, 1,337 nodes, and three partitions.
Future batches should make the host runner persist instrumentation plus logcat,
pin their SHA-256 values and APK identities, compute the quality verdict, and
use a run ID instead of overwriting the same model/candidate/phase directory.

## QNN HTP prepare

The qnnV79 flavor contains the official LiteRT Qualcomm compiler/dispatch and
QAIRT V79 libraries. `BuiltinNpuAcceleratorProvider` reports NPU availability,
but availability alone is not treated as delegation proof. Proof requires a
non-empty QNN IR and successful graph finalize.

### HMX on, prepare optimization

`compiled-npu-qnn-prepare-opt` selected `3497/3504` nodes in 8 partitions and
wrote `qnn_partition_0.json` (about 1.64 MB). This is real QNN compilation, not
a CPU fallback. Finalization failed after about 58 seconds:

```text
A single op, q::ConvLayer.opt.activations_to_vtcm, requires 0xdff800 bytes
of TCM, which is greater than the TCM size of 0x800000
failing op: .../Conv2d_3672_LiteRt_OpId_27
QnnGraph_finalize done. status 0x3ea
```

The failing Conv2d is the first long-sequence encoder convolution. Its QNN IR
shapes are approximately:

```text
input  [1, 85995, 1, 48]
kernel [6, 3, 1, 48]
output [1, 85995, 1, 6]
```

`SM8750` is mapped by the LiteRT 2.1.5 Qualcomm table to 8 MiB VTCM. The
Kotlin option `vtcmSize=0` already means device maximum; setting 8 is equivalent
and a value above 8 cannot create physical TCM.

### Targeted option probes

| Candidate | Accelerators | QNN options | Result |
| --- | --- | --- | --- |
| `compiled-npu-cpu-qnn-prepare-opt-no-hmx` | NPU + CPU | level 1, HMX off | same Conv2d/TCM failure, 58.33 s |
| `compiled-npu-cpu-qnn-prepare-opt-no-hmx-hvx1` | NPU + CPU | level 1, HMX off, HVX1 | same Conv2d/TCM failure, 58.36 s |
| `compiled-npu-qnn-inference-opt` | NPU | level 2 | stopped after about 143 s; finalize indeterminate |
| `compiled-npu-qnn-inference-o3` | NPU | level 3 | candidate added, not run after level-2 resource cap |

The no-HMX rows use explicit `NPU+CPU`, which is the required form for a graph
with residual CPU nodes. They show that CPU fallback does not help when QNN
cannot finalize this single convolution. The HVX1 option is recorded in the
LiteRT evidence (`HvxThread: 1`), but the device environment still reports its
normal 8 HVX workers and the TCM requirement is unchanged.

NPU+GPU+CPU was not redundantly compiled: the QNN compiler sees the same eligible
partition and has no public Kotlin option to exclude OpId 27 or impose a maximum
partition size. A different accelerator list cannot reduce this single-op
scratch requirement.

## What must change before another NPU/GPU attempt

1. Preserve the current contract and use Compiled CPU as the numerical oracle.
2. Keep the completed identity-`GATHER_ND` derivative as a diagnostic artifact,
   not a GPU candidate: it passes CPU bitwise parity and raises delegation, but
   its hybrid GPU result is severely wrong. Do not stack rank-5 rewrites onto
   this artifact until the bad GPU partition is isolated. Any rank-5 lowering
   must use another derived identity and repeat CPU plus GPU parity gates.
3. For QNN, a partition boundary alone is insufficient because the failing
   operation itself needs about 14 MiB TCM. Practical experiments are:
   * teach/fork the Qualcomm compiler plugin to leave OpId 27 (and any similar
     long-axis convolutions) on explicit CPU;
   * implement a time-axis `SAME`, kernel-3, stride-1 convolution in chunks with
     one-sample halo, then concatenate only valid regions; or
   * export a shorter-window core and give it a separate contract.
4. Only after a non-empty IR, `QnnGraph_finalize status 0`, and a created
   `CompiledModel` should a one-window NPU run be attempted. Require finite
   outputs and the same 80 dB gate.
5. The smallest next experiment should be a new `smoke_2s` contract with
   waveform length `88,200` (87 spectral frames), using the reference
   `export_short_litert_model.py` shape/export mechanics. This is expected to
   reduce the long-axis scratch estimate from about 14 MiB to about 3.6 MiB,
   but that estimate is not evidence: require non-empty QNN IR,
   `QnnGraph_finalize status 0`, and a created model before running. Pin the
   official safetensors provenance (the reference script's legacy `.th` file
   must not be silently treated as equivalent), then require per-output and
   combined SNR >=80 dB. Compare latency as RTF, not absolute milliseconds
   across 2-second and 7.8-second windows. If short-window QNN still fails,
   implement the time-axis halo/chunk contract; defer rank-5 GPU lowering until
   the bad `LITERT_CL` partition in the current derivative is isolated.

Changing `htpPerformanceMode`, profiling, dump/IR logging, weight sharing, or
`useFoldRelu` does not address this activation-TCM failure. AOT targets the same
SM8750 8 MiB hardware and is not an automatic workaround.

## Reproduction and evidence

The probe source is
[`ExternalLiteRt215InstrumentedTest.kt`](../app/src/androidTest/java/com/example/musicsourceseparation/benchmark/ExternalLiteRt215InstrumentedTest.kt).
For example, a one-window run is:

```powershell
$adb="$env:LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe"
& $adb -s '<device-serial>' shell am instrument -w -r `
  -e class com.example.musicsourceseparation.benchmark.ExternalLiteRt215InstrumentedTest `
  -e probeDerivationManifest bandbuddy_htdemucs_6s_core_gather_reshape_v1_0_0.json `
  -e probeCandidate compiled-gpu-cpu-opencl-buffer-fp32 `
  -e probePhase run -e probeThreads 4 `
  com.example.musicsourceseparation.test/androidx.test.runner.AndroidJUnitRunner
```

Raw device results, QNN IR, logcat, and exported FP32 outputs are archived under
`outputs/bandbuddy-s25-20260803/litert215/` (ignored generated evidence). The
probe result JSON records model/contract/runtime/device hashes for every row.
The in-place derivative follow-up is also staged under
`.tmp/bandbuddy-htdemucs-6s/s25-gather-reshape/`; do not substitute the separate
`htdemucs_6s.core.gather_nd_reshape_v1.tflite` prototype. Its `device-evidence`
directory includes the final CPU/hybrid result JSON and the strict/hybrid GPU
logcat files that contain the `LITERT_CL` replacement evidence.

Build verification for this batch:

```text
./gradlew :app:assembleQnnV79Debug :app:assembleQnnV79DebugAndroidTest ...
BUILD SUCCESSFUL
```

No full-song, thermal, energy, S10, or product-integration claim is made by this
short experiment.
