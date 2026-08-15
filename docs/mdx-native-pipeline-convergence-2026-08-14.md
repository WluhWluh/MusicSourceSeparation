# MDX native managed-buffer two-slot convergence

Date: 2026-08-14  
Branch: `experiment/mdx-native-pipeline-convergence`  
Final report-harness revision: `9c3ba62e34a0688bfc0c30704a17e34042f13d8b`

## Scope

This branch converges three previously independent experiments:

- `9988137`: all-shape and ABI qualification of native packed-real MDX DSP;
- `9e69417`: LiteRT 2.1.5 C API with managed input/output buffers;
- `5930d0a` and `3bc8fb0`: one-model, two-slot short-window and full-song pipelines.

The converged path owns one native LiteRT C `CompiledModel`, two managed input buffers,
two managed output buffers, and one native packed-real DSP plan. The main thread prepares
window N+1 while the executor invokes window N. Postprocessing begins only after the
corresponding invocation completes, so the single DSP workspace is not accessed
concurrently. The public Kotlin wrapper remains source compatible with `slotCount = 1` and
rejects counts outside 1..2.

QNN is intentionally excluded. The native C wrapper currently implements CPU and bounded
GPU profiles only; existing Kotlin `CompiledModel` QNN results must not be presented as
direct-buffer convergence evidence.

## Frozen inputs

| Item | Identity |
|---|---|
| Model | `UVR_MDXNET_3_9662_static_float32.tflite` |
| Runtime | `litert-android-2.1.5-bss.2` |
| Runtime AAR SHA-256 | `88cd2f7eaf1443d1c570085b1c24f239db87eb24c788a590adf5158e17443d0e` |
| GPU profile | bounded OpenCL FP32, kernel batch 1, queue window 1 |
| DSP | native packed-real, four workers |
| Short test | two warmups, ten sequential and ten two-slot windows |
| Full song | 273.699093 s, 48 windows |

Local verification passed `testStandardDebugUnitTest`, `assembleStandardDebug`, and
`assembleStandardDebugAndroidTest`. CMake compiled the JNI implementation for arm64-v8a,
armeabi-v7a, x86, and x86_64.

## Short-window results

Each boundary ran in a separate instrumentation process. Both devices recorded 2,540
bounded dispatches and 2,540 bounded waits per measured profile.

| Device / run order | Java tensor two-slot | Native managed two-slot | Native change |
|---|---:|---:|---:|
| S25, Java then native | 292.14 ms/window | 283.71 ms/window | -2.9% |
| S25, native then Java | 325.13 ms/window | 301.96 ms/window | -7.1% |
| S10, Java then native | 2,577.22 ms/window | 2,330.65 ms/window | -9.6% |
| S10, native then Java | 2,373.86 ms/window | 2,791.69 ms/window | +17.6% |

S25 improved in both orders. S10 changed direction and has a much larger spread, so this
batch establishes S10 compatibility and memory behavior, not a stable RTF improvement.
A temperature-controlled repeated S10 batch is required before making a performance claim.

### S10 cold-state repeat

Revision `ef4403b` added thermal status to the short-run report. Three new paired repeats
then ran from separate processes with two warmups and ten measured windows per sequential
and two-slot phase. Every profile recorded thermal status `0 -> 0`; the order alternated
native/Java, Java/native, native/Java.

| Repeat | Java tensor two-slot | Native managed two-slot | Native change |
|---|---:|---:|---:|
| 1 | 2,379.93 ms/window | 2,585.44 ms/window | +8.6% |
| 2 | 2,383.53 ms/window | 2,418.04 ms/window | +1.45% |
| 3 | 2,452.98 ms/window | 2,416.64 ms/window | -1.48% |
| Median | 2,383.53 ms/window | 2,418.04 ms/window | +1.45% |

The cold batch resolves the earlier direction-changing S10 result: managed buffers should
be treated as approximately performance-neutral with a small median regression, not as an
S10 RTF optimization. The memory result is stable. Every Java profile allocated about
160.5 MiB and ran eight GCs; every native-managed profile allocated 279,856 bytes and ran
zero GCs. Mean PSS was 558,932 KiB for Java and 508,835 KiB for native-managed, a 48.9 MiB
reduction.

The second run added matched ART and memory evidence:

| Device | Boundary | ART allocated | GC count | PSS | Native heap |
|---|---|---:|---:|---:|---:|
| S25 | Java tensor | 168,247,344 B | 3 | 572,234 KiB | 43,530,256 B |
| S25 | Native managed | 295,520 B | 0 | 549,841 KiB | 39,558,464 B |
| S10 | Java tensor | 168,263,680 B | 8 | 452,628 KiB | 37,257,424 B |
| S10 | Native managed | 279,856 B | 0 | 408,942 KiB | 33,257,552 B |

The direct boundary removes about 160 MiB of short-batch ART allocation, all observed GC,
21.9 MiB S25 PSS, and 42.7 MiB S10 PSS. This is the strongest result of the convergence and
does not depend on the noisy S10 timing comparison.

## S25 full-song pair

Both profiles started and ended at Android thermal status 2. Treat absolute timing as a
paired hot-condition result, not a cool-device qualification.

| Metric | Java tensor | Native managed | Change |
|---|---:|---:|---:|
| Processing | 15,911.54 ms | 13,953.60 ms | -12.3% |
| Processing RTF | 0.05814 | 0.05098 | -12.3% |
| Setup | 1,055.07 ms | 1,722.47 ms | +667.40 ms |
| PSS | 690,060 KiB | 589,814 KiB | -97.9 MiB |
| Native heap | 43,583,552 B | 39,566,512 B | -3.8 MiB |
| Inference wait | 11,626.33 ms | 9,486.99 ms | -18.4% |
| Output read + iSTFT | 1,082.71 ms | 953.95 ms | -11.9% |

The native full-song path still allocated 200,810,544 ART bytes and ran four GCs because
the runner creates per-window context waveforms and performs residual/PCM work in Kotlin.
That number is not a matched tensor-boundary delta: the corresponding Java full-song run
predated the added ART counter. The short fixed-window test provides the valid matched
boundary allocation comparison.

Both complete outputs are byte-identical to the established Java two-slot baseline:

| Output | Bytes | SHA-256 |
|---|---:|---|
| Model stem | 48,280,564 | `f92f45c52a4902763eae031f80c8ad8b499296ee1fdf56318224988d9a24553a` |
| Residual stem | 48,280,564 | `4af386eefae21c7fb7f9f04bf17d0aa15bbdb37af594053c0b4ebf91d7f08fe4` |

## Decision

The converged native-managed two-slot path passes the 9662 bounded-GPU experimental gate:

- one model instance and two managed buffer pairs operate correctly;
- bounded GPU dispatch evidence is complete;
- short and 48-window full-song outputs are finite;
- full-song PCM is byte-identical to the previous two-slot path;
- ART allocation, GC, and PSS improve materially;
- S25 paired processing time improves, including under a hot condition.

Keep `native-packed` as the single native DSP candidate; do not add a product-time
native-full/native-packed selector. Before application integration, complete a 100-window
create/run/close stability batch and run all 13 qualified MDX shapes through the same
two-slot managed-buffer boundary. QNN needs a separate C API extension and must remain a
later experiment.
