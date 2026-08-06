# S25 LiteRT 2.1.5 HTDemucs 2 秒 smoke probe

日期: 2026-08-04 (设备日志为 `Asia/Shanghai`，host 文档日期按实验目录命名)

状态: 完成 host 导出门禁、S25 Compiled CPU 单窗口、GPU strict prepare、GPU+CPU
hybrid 单窗口；NPU QNN prepare 在 VTCM 分配阶段被主动停止。本文只记录固定
`smoke_2s` 的前置可行性证据，不代表产品支持、7.8 秒性能、整曲稳定性、功耗或
音质结论。

## 结论

- 官方 safetensors 导出得到的 2 秒 neural core 通过 host parity；固定 ABI 是
  waveform `[1,2,88200]`、host-STFT `[1,4,2048,87]`、频域输出
  `[1,6,4,2048,87]`、时域输出 `[1,6,2,88200]`。
- S25 `compiled-cpu-xnnpack` 创建、分配和推理完成。raw 频域、raw 时域和聚合
  combined 与 host golden 都达到 80 dB 门槛以上且 finite。按每个 stem 也要求
  80 dB 时，`bass`、`other`、`vocals` 分别低于门槛；现有 `host-parity.json`
  的 `status: passed` 是分支/combined 聚合门禁，不应解释为逐 stem 全部通过。
- GPU-only strict `compiled-gpu-opencl-buffer-fp32` 在 prepare 失败。日志明确记载
  GPU 只替换 `158/3504` 个节点、形成 3 个 partition，并列出
  rank-5 `ADD/MUL/RESHAPE`、`BROADCAST_TO`、两类 `FULLY_CONNECTED` 和
  `GATHER_ND` blocker；没有 inference。
- `compiled-gpu-cpu-opencl-buffer-fp32` 成功创建并完成一次 inference，日志有真实
  `LITERT_CL` 委派（`158/3504`、3 partitions）及 XNNPACK CPU remainder
  （`3152/3347`、105 partitions）。聚合 host parity 通过；逐 stem 的同一三个
  低于 80 dB，故不能把它记为严格逐 stem 通过的 GPU 方案。
- NPU `compiled-npu-qnn-prepare-opt` 生成了非空 QNN IR，并进入了
  `QnnGraph_finalize started`；随后 QNN 日志明确出现 VTCM `0x800000` 分配失败和
  retry。该进程在第二轮准备仍未结束时由 `user_req_or_stop` force-stop。没有
  `QnnGraph_finalize done/status`、没有 `compiled-model-created`、没有
  `QnnGraph_execute`，因此 NPU 没有创建可运行的 `CompiledModel`，也没有跑
  inference。该行应记为 `STOPPED_DURING_PREPARE`，不是 NPU PASS，也不是完整的
  自然返回 FAIL。

## 固定身份

| 项目 | 值 |
| --- | --- |
| branch | `experiment/model-matrix-demucs-multistem` |
| app source revision | `3510d5c263ab8236cd96cfeb07feef1c1a5c0dff`，dirty worktree |
| device | Samsung `SM-S9310`, SoC `SM8750`, Android 15/API 35, `arm64-v8a` |
| ADB | private device endpoint; pass as `<device-serial>` |
| LiteRT | `com.google.ai.edge.litert:litert:2.1.5` |
| LiteRT AAR | 10,058,192 bytes, SHA-256 `a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37` |
| QNN v79 bundle manifest | `app/src/qnnV79/assets/app-live/runtime-manifest.json`, SHA-256 `8e20d10a6b27107fcb80460cc27892d5c2b797107c09ecbebeb095fbfe09c297` |
| model ID | `htdemucs_6s_core_smoke_2s_fp32_v1_0_0` |
| contract ID | `htdemucs_6s_core_smoke_2s_fp32_v1_0_0@1` |
| contract sidecar | `app/src/main/assets/benchmark-contracts/htdemucs_6s_core_smoke_2s_fp32_v1_0_0.json`, 15,303 bytes, SHA-256 `a14474d308734c84003cfc83e2ccd746564022d34ad79d6066c87d50a3dd8d0c` |
| LiteRT model | `models/demucs/generated/htdemucs_6s_core_smoke_2s_fp32_v1_0_0/htdemucs_6s.core.smoke_2s.fp32.tflite`, 112,924,120 bytes, SHA-256 `35ec0361b8ee6b415d435dc6d35d52dbf13bf9492b3f044e3259f414dbf23e61` |
| export report | `models/demucs/generated/htdemucs_6s_core_smoke_2s_fp32_v1_0_0/export-report.json`, 13,025 bytes, SHA-256 `f7b8672875f4db58b8ef823bcf1f513b57ba4d878303b2bbd40dc2d38a6bc1e2` |

The report also records a 111,392,345-byte ONNX intermediate with SHA-256
`16464486d8791e673d1ee9bed9aecc7045b4042b3616c41af2b3b0276801879c`. It is a
diagnostic parity intermediate, not the Android LiteRT input.

## 来源与导出

The runtime candidate references
`htdemucs_6s_waveform_7p8s_onnx@3` only as
`upstream-model-identity-only`; it does not claim to be converted from that
community ONNX artifact. The actual model source pinned by the contract is:

| Artifact | Path / revision | Size | SHA-256 |
| --- | --- | ---: | --- |
| official weight | `models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors`, `adefossez/HTDemucs-6s@053e1404489b3dc58bf718224fac4b7316de8c93` | 54,885,744 | `d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411` |
| metadata | `models/demucs/official-hf/htdemucs_6s/5c90dfd2.json` | 10,398 | `72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e` |
| bag manifest | `models/demucs/official-hf/htdemucs_6s/htdemucs_6s.yaml` | 21 | `207405151270af8fd81c2373c25d27950916682ac91dca7884a11ce13dad6f58` |

The legacy checkpoint is retained for provenance comparison only:
`models/demucs/checkpoints/5c90dfd2-34c22ccb.th` is 54,996,327 bytes with SHA-256
`34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd`. The export
contract's selected source format remains canonical safetensors.

The project-owned export recipe is pinned in the contract as
`tools/export_htdemucs_litert_candidate.py` (26003 bytes,
SHA-256 `a1229069ebee6e48d03507e730b3a4fa428067eba4222d75b3e6e5872bcf50c5`).
That is the historical execution path. The byte-exact source remains available
at commit `673d565` and under
`tools/frozen-exporters/a1229069ebee6e48d03507e730b3a4fa428067eba4222d75b3e6e5872bcf50c5/`;
the current exporter at the historical path is a later revision.
The recorded environment is Python `3.12.3`, torch `2.11.0+cpu`, NumPy `2.5.1`,
 safetensors `0.8.0`, LiteRT Torch `0.9.1`, and ai-edge LiteRT `2.1.5`.

## Static ABI

The TFLite FlatBuffer inspection used by the contract reports `TFL3`, schema 3,
one subgraph, 3,504 operators, 4,301 tensors, and zero custom operators. The
signature is `serving_default`:

```text
args_0   FLOAT32 [1, 2, 88200]
args_1   FLOAT32 [1, 4, 2048, 87]
output_0 FLOAT32 [1, 6, 4, 2048, 87]  tensorIndex=4295
output_1 FLOAT32 [1, 6, 2, 88200]     tensorIndex=4300
```

The logical feature order is `L.real, L.imag, R.real, R.imag`; stem order is
`drums, bass, other, vocals, guitar, piano`. The FlatBuffer operator histogram
still contains 132 `GATHER_ND` nodes; this smoke artifact is not the earlier
GATHER-to-RESHAPE derivative.

There is an evidence-format discrepancy that is intentionally not resolved here:
the lightweight `liteRtArtifact.inspection.tensorCount` field in
`export-report.json` says `4308`, while the standalone
`flatbuffer-inspection.json` and the contract say `4301`. ABI shapes, signature
indices, and device allocation agree; no cause is inferred from this discrepancy.

## Host parity

Evidence: `models/demucs/generated/htdemucs_6s_core_smoke_2s_fp32_v1_0_0/export-report.json`.
The Torch core reconstruction is bitwise equal (`maxAbsoluteError=0`). LiteRT
versus Torch on the frozen 2-second fixture is finite and passes the
`SNR >= 80 dB`, `maxAbsoluteError <= 0.001` gate:

| Output | SNR | max absolute error |
| --- | ---: | ---: |
| frequency | 116.231 dB | 2.622604e-06 |
| waveform | 130.349 dB | 1.192093e-07 |
| combined | 118.620 dB | 5.941838e-07 |

Combined per-stem SNR is: drums `118.573 dB`, bass `84.087 dB`, other `87.452 dB`,
vocals `80.596 dB`, guitar `123.805 dB`, piano `83.692 dB`. The report marks the
candidate accepted for device testing.

Frozen fixture files and identities are recorded in the same report and contract:

```text
fixtures/waveform_input.f32le.raw   705600 bytes  a644f187d8c6d6644838fd637463695877e5d0c3680ba7fa7364e4b0fe6aa071
fixtures/spectrum_input.f32le.raw  2850816 bytes  c02deb000bb84026487bf9885faba4f15b5282052c36890a53fae052f3791a6d
fixtures/frequency_golden.f32le.raw 17104896 bytes 8cf015f76b00208013fcb0ba72436b0b0236d3af8de9ac0e634c6bdd0062d02f
fixtures/waveform_golden.f32le.raw 4233600 bytes 981824eb40608db997b61dfed92d0a28ed2826ab876fc287e28431a7a38903b4
fixtures/combined_golden.f32le.raw 4233600 bytes 3f2a30b8cdff473b568aea32bbe4824c8a0fcde500e08185e0d548ba25928b40
```

## S25 CPU

Evidence root:
`outputs/htdemucs-smoke2s-s25-20260804/compiled-cpu-xnnpack-run/`.

| Metric | Value |
| --- | ---: |
| candidate / mode | `compiled-cpu-xnnpack`, `CompiledModel`, XNNPACK FP32, 4 threads |
| prepare wall / CPU | 597.322 / 592 ms |
| inference wall / CPU | 1,148.981 / 4,216 ms |
| model-only RTF | 0.5745 (`1.148981 / 2`) |
| output_0 / output_1 read wall | 11.512 / 3.099 ms |
| output_0 non-finite | 0 |
| output_1 non-finite | 0 |
| retained result PSS | 88,925 KiB |

`host-parity.json` reports finite aggregate parity: frequency `100.395 dB`,
waveform `123.847 dB`, combined `102.678 dB`; their max absolute errors are
`1.379661e-05`, `3.986061e-07`, and `4.101545e-06`, respectively. The combined
per-stem values are
drums `104.679 dB`, guitar `108.771 dB`, piano `81.707 dB`, bass `77.986 dB`,
vocals `79.990 dB`, other `70.801 dB`. Thus the aggregate row is usable as the
device CPU baseline, but a policy requiring every stem to reach 80 dB is not met.

The device output hashes are recorded in `run/result.json`:

```text
output_0: d8d2c4aacc42223940a2af78f2f824003fd98b009598f4418f62fbdcc0eb602d
output_1: 2ad4b939182d4bae124896ac4ba98c4a35283a6bda91961bbf69b2a56e11352d
```

## GPU strict prepare

Mode: `compiled-gpu-opencl-buffer-fp32`, GPU-only, OpenCL BUFFER FP32, high
priority, one command-buffer preparation step, `phase=prepare`.

Evidence:

- `outputs/htdemucs-smoke2s-s25-20260804/compiled-gpu-opencl-buffer-fp32-prepare/prepare/failure.json`
  reports `status=error`, `LiteRtException`, message `Failed to compile model`.
- `.../logcat.txt:3764-3775` reports rank-5 `ADD/MUL/RESHAPE`, unsupported
  `BROADCAST_TO`, constant/multi-input `FULLY_CONNECTED`, and `GATHER_ND`,
  followed by `Replacing 158 out of 3504 node(s) with delegate (LITERT_CL)
  node, yielding 3 partitions`.
- `.../logcat.txt:8824` reports the compiled-model creation failure.

No `result.json`, inference, output readback, or numerical parity exists for this
strict GPU row. The retained prepare checkpoint PSS was 46,406 KiB; it is not a
peak measurement.

## GPU + CPU hybrid run

Mode: `compiled-gpu-cpu-opencl-buffer-fp32`, accelerators `[GPU, CPU]`, OpenCL
BUFFER FP32, `phase=run`, frozen raw fixtures, output export enabled.

Evidence:

- `.../logcat.txt:1064-1079` shows the same unsupported-op list and real
  `LITERT_CL` replacement of `158/3504` nodes into 3 GPU partitions.
- `.../logcat.txt:1390-1391` shows XNNPACK replacing `3152/3347` remainder nodes
  into 105 CPU partitions.
- `.../logcat.txt:1404-1458` records `compiled-model-created`, buffer setup, and
  inference start; `...:2820` records inference completion.

| Metric | Value |
| --- | ---: |
| prepare wall / CPU | 1,696.919 / 1,668 ms |
| inference wall / CPU | 637.109 / 2,222 ms |
| model-only RTF | 0.3186 (`0.637109 / 2`) |
| output_0 / output_1 read wall | 6.469 / 2.387 ms |
| output_0 / output_1 non-finite | 0 / 0 |
| retained result PSS | 114,628 KiB |

The host-parity aggregate is frequency `98.915 dB`, waveform `122.326 dB`,
combined `101.238 dB`; their max absolute errors are `1.649000e-05`,
`5.587935e-07`, and `4.734844e-06`, respectively. Per-stem combined SNR is drums
`108.060 dB`, guitar
`105.889 dB`, piano `81.420 dB`, bass `77.982 dB`, vocals `79.956 dB`, other
`69.543 dB`; apply the same per-stem caveat as CPU. Output hashes are:

```text
output_0: 745409bafbb2a0cc5dae3fcb3ddd7b94e17d119aac07fc91bcb56e2927ae267d
output_1: b841dcb062c410428a925abbba6f96d2e57c97385ce386033cd74604a92f392c
```

## NPU prepare: VTCM stop

Mode: `compiled-npu-qnn-prepare-opt`, NPU-only, `HTP_OPTIMIZE_FOR_PREPARE`,
Conv HMX enabled, `phase=prepare`, no output export.

The probe produced a non-empty static IR at
`.../prepare/qnn-ir/qnn_partition_0.json`: 1,621,074 bytes, SHA-256
`4d8a8211e3d558b029249741433ba7555dcc94b545edde0f0dd92ce86eb32446`.
The JSON contains 1,340 node entries and 1,569 tensor entries. These are static
IR-shape observations only; they are not causal attribution of the failure.
The compiler log selected `3497/3504` LiteRT operators into 8 partitions; the
archived JSON is only `qnn_partition_0`, so its 1,340 lowered QNN nodes are not
the whole selected graph.

The causal evidence actually present in `.../logcat.txt` is:

```text
20:14:59.299  QnnGraph_create done. status 0x0
20:14:59.377  QnnGraph_finalize started
20:14:59.378  VTCM total_sz=8388608 (0x800000)
20:19:16.129  Failure to allocate within the VTCM size of 0x800000 bytes
20:19:16.161  Cannot allocate for schedule due to vtcm limits; possible retry
20:19:16.167  RETRY_PREPARE try 1 with return status = 1
20:21:08.629  ActivityManager force stop (user_req_or_stop)
20:21:08.757  process exited due to signal 9 (Killed)
```

There is exactly one `QnnGraph_finalize started` line and no matching
`QnnGraph_finalize done/status` line, no `compiled-model-created` probe stage, and
no `QnnGraph_execute` line. `prepare/progress.json` remains
`status=running`, stage `compiled-model-create-start`; there is no completed
`result.json`. Therefore this run establishes a VTCM allocation failure/retry
boundary and an active-stop condition only. It does not establish a successful or
completed unsuccessful QNN graph, and there are no NPU outputs or NPU latency
numbers. The 46,304 KiB PSS in the initial checkpoint is neither retained-final nor
peak PSS.

### QNN IR static observations

The JSON is the single `qnn_partition_0`, not a dump of the whole LiteRT graph.
It contains 1,340 QNN node entries and 1,569 tensor entries. Its 1,292 unique
source LiteRT OpIds cover `0..1291`; 48 source IDs are lowered into a
`Conv2d`/`Transpose` pair, so the QNN-node count must not be compared directly
with the 3,504 FlatBuffer operator count. The node histogram is:

```text
ElementWiseBinary 429   Reshape 352          Transpose 140
StridedSlice 120        ReduceSum 68          GatherNd 56
Conv2d 48                ElementWiseNeuron 48  ElementWiseSubtract 34
ElementWiseUnary 34     Concat 8              Pad 3
```

The largest native FP32 tensors contain 4,276,224 elements (17,104,896 bytes,
16.3125 MiB), with 19 instances in the shape family
`512x87x1x96`, `512x87x96`, `512x96x87`, `512x1x96x87`, and
`1x512x87x96`. Representative producers are source OpIds 242, 307, and 342.
Long-axis convolution examples include:

```text
Op20       1x88200x1x2   -> 1x22050x1x48
Op27/112   1x22050x1x48 -> 1x22050x1x6
Op79/164   1x22050x1x6  -> 1x22050x1x96
Op197      1x22050x1x48 -> 1x22050x1x96
```

These shapes are candidate memory-risk clues only. The archived log does not
identify a responsible operator or prove that any one of these tensors is the
failed schedule group. In particular, the 7.8-second artifact's OpId 27 and its
separate `0xdff800` single-convolution observation must not be applied to this
2-second IR. The IR SHA-256 is
`4d8a8211e3d558b029249741433ba7555dcc94b545edde0f0dd92ce86eb32446`.

## Reproduction commands

All rows used the same S25 device, contract, model, 4 threads, and LiteRT 2.1.5
QNN-v79 build. The common instrumentation command shape was:

```powershell
$adb="$env:LOCALAPPDATA/Android/Sdk/platform-tools/adb.exe"
& $adb -s '<device-serial>' shell am instrument -w -r `
  -e class com.example.musicsourceseparation.benchmark.ExternalLiteRt215InstrumentedTest `
  -e probeContract htdemucs_6s_core_smoke_2s_fp32_v1_0_0.json `
  -e probeCandidate <candidate> `
  -e probePhase <prepare|run> -e probeThreads 4 `
  -e probeExportOutputs <true|false> `
  com.example.musicsourceseparation.test/androidx.test.runner.AndroidJUnitRunner
```

Rows and modes:

```text
compiled-cpu-xnnpack                    run     exportOutputs=true
compiled-gpu-opencl-buffer-fp32        prepare exportOutputs=false
compiled-gpu-cpu-opencl-buffer-fp32    run     exportOutputs=true
compiled-npu-qnn-prepare-opt           prepare exportOutputs=false
```

The raw device evidence is under
`outputs/htdemucs-smoke2s-s25-20260804/`; the corresponding on-device result
directories are isolated by model ID under
`litert215-probe/htdemucs_6s_core_smoke_2s_fp32_v1_0_0/`.

## PSS and performance interpretation

`result.json.process.pssKb` is a final retained-process snapshot after the probe's
output/cleanup path, not a peak-PSS sample. The higher `progress.json` checkpoints
at `read-output-output_1` were 531,184 KiB (CPU) and 758,958 KiB (GPU hybrid), but
they are also point samples, not a continuously observed peak. No formal peak PSS,
thermal series, swap series, median, or P95 was collected in this one-window run.

The CPU and hybrid inference wall times are therefore single-window observations
only. The hybrid inference observation is 44.5% lower than CPU (`637.109` versus
`1148.981 ms`), but prepare is about 1.10 seconds higher. Prepare plus one
inference is therefore about `2.334 s` for hybrid versus `1.746 s` for CPU, so
this cold single-window experiment does not establish an end-to-end throughput
win. Reusing a compiled model across several windows might amortize prepare, but
that was not measured.

## Scope boundary and next step

This is the `smoke_2s` static profile for allocation and operator/backend coverage.
It must not be linearly extrapolated to `canonical_7p8s`: attention and activation
cost do not scale linearly with represented audio duration. A follow-up NPU attempt
needs a bounded watchdog and must first obtain a completed `QnnGraph_finalize` and a
created `CompiledModel`; only then should it run one window and compare raw and
combined outputs. GPU work should retain the strict/hybrid distinction and the
per-stem parity caveat above.

Recommended next-batch order:

1. Keep this exact GPU+CPU row as the LiteRT 2.1.5 mixed-backend baseline. Do
   not relabel its 158-node OpenCL coverage as strict GPU support.
2. If GPU graph work continues, create one derived identity per rewrite family.
   Start by isolating rank-5 `ADD/MUL/RESHAPE` and `BROADCAST_TO`; leave
   unsupported `FULLY_CONNECTED` or other nodes on explicit CPU until each
   lowering passes Compiled CPU and exported GPU-output parity. Do not copy the
   prior 7.8-second identity-`GATHER_ND` derivative blindly: its CPU result was
   exact but its hybrid GPU output was severely wrong.
3. Do not rerun the unchanged short graph with QNN O3, HMX-off, or another
   accelerator list. The next NPU experiment must change the workload: use a
   separately contracted shorter window, time-axis convolution chunks with the
   correct halo, or a compiler/partition boundary that leaves the long-axis
   operations on CPU. Repeat host parity before device prepare.
4. Generate the four-stem smoke artifact as a distinct official-safetensors
   contract before comparing 4-versus-6 stem pressure. The official four-stem
   model has more stored parameters despite fewer outputs, so neither model can
   be used as a memory proxy for the other.
5. Treat a canonical 7.8-second CPU window as another independent batch with a
   fresh process, watchdog, and actual peak-PSS sampling. Do not use the 2-second
   RTF or point-sampled PSS to admit S10 canonical execution.

Related contract and policy documents:

- `docs/android-litert-demucs-multistem-feasibility-2026-08-03.md`
- `docs/android-litert215-demucs6-s25-2026-08-03.md`
- `docs/model_contracts.md`
