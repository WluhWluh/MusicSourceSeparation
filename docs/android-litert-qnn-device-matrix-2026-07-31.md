# Android LiteRT QNN device matrix

This report closes the initial Qualcomm HTP exploration performed with LiteRT
2.1.5, QAIRT 2.44.0.260225, `UVR_MDXNET_3_9662` FP32, and the self-contained
App Live Quick and Full validation profiles. It records research evidence, not
a Booming SS production support list.

## Evidence contract

A result is classified as delegated only when all of these conditions hold:

- the Qualcomm provider accepts the device and reports its libraries ready;
- LiteRT exposes the NPU accelerator;
- model setup and inference complete with finite output; and
- every claimed QNN tensor or audio session emits at least one non-empty QNN
  IR partition.

Provider readiness or an NPU label alone is insufficient. A completed run with
no QNN IR is classified as CPU fallback. This fail-closed rule was added after
the Galaxy Tab S8 session exposed a silent fallback.

## Device results

| Device | SoC | Runtime | Coverage | Result |
| --- | --- | --- | --- | --- |
| Galaxy S25 / SM-S931B | SM8750 | HTP v79 | Quick and Full | Delegated |
| Galaxy S25 Ultra / SM-S938B | SM8750 | HTP v79 | Quick | Delegated |
| OnePlus 13R / CPH2691 | SM8650 | HTP v75 | Quick and Full | Delegated |
| OnePlus 12R / CPH2585 | SM8550 | HTP v73 | Quick and Full | Delegated |
| Galaxy S23 Ultra / SM-S918B | SM8550 | HTP v73 | Quick and Full | Delegated |
| OnePlus 11R / CPH2487 | SM8475 | HTP v69 | Quick and Full | Delegated |
| China-market Galaxy S22 / SM-S9010 | SM8450 | HTP v69 | Quick and Full | Delegated |
| Galaxy Tab S8 / SM-X706B | SM8450 | HTP v69 | Quick and Full | CPU fallback; remote device unhealthy |
| Motorola Edge 50 Fusion | SM7435 | HTP v73 | Quick diagnostics | CPU fallback; no QNN IR |
| Exynos Galaxy S22 controls | Exynos 2200 | HTP v69 | Preflight | Correctly rejected |

The successful v69, v73, v75, and v79 sessions emitted the same complete
277-node partition: 40 Conv2d, 65 ElementWiseBinary, 66 ElementWiseNeuron,
22 MatMul, 79 Transpose, and 5 TransposeConv2d nodes. Same-generation devices
produced stable output. In particular, the two v73 phones produced
byte-identical tensors and full-song WAV files, as did the SM8450 and SM8475
v69 devices.

Representative Full-profile results for the 273.699-second fixture were:

| Device | QNN window median | End-to-end time | Processing seconds per audio second |
| --- | ---: | ---: | ---: |
| Galaxy S25 v79 | 120.1 ms | 39.70 s | 0.1451 |
| OnePlus 13R v75 | 126.9 ms | 55.65 s | 0.2033 |
| OnePlus 12R v73 | 156.6 ms | 50.25 s | 0.1836 |
| Galaxy S23 Ultra v73 | 145.2 ms | 28.55 s | 0.1043 |
| OnePlus 11R v69 | 163.3 ms | 39.55 s | 0.1445 |
| Galaxy S22 v69 | 142.6 ms | 43.26 s | 0.1581 |

The large same-generation end-to-end difference between the OnePlus 12R and
Galaxy S23 Ultra came primarily from ISTFT and PCM conversion rather than HTP
inference. BrowserStack scheduling and OEM behavior remain confounders for
non-inference timing.

## Numerical and audio interpretation

QNN output is not byte-identical to CPU, bounded GPU, or ORT output. Window
SNR versus ORT was approximately 34.6-35.0 dB across the accepted generations,
with cosine similarity near one. Cross-generation full-song PCM comparisons
were materially closer than the raw tensor metric, and informal listening did
not reveal a difference from the other backends. This is useful research
evidence but not a formal blind-listening qualification.

All accepted Full runs completed 48 of 48 windows, produced finite stems, and
closed their sessions. Sparse instrumental endpoint clipping was consistent
across QNN generations and the other backends, so it is not treated as a
QNN-specific failure.

## Failed and confounded boundaries

The Galaxy Tab S8 session emitted no QNN IR, set up in less than one second,
and took about 4.9 seconds per window with CPU time matching wall time. Its
output matched the explicit CPU path. The remote tablet was also extremely
laggy, required repeated connection attempts, and could not later be reached
for a clean repeat. The result must not be generalized to all SM8450 tablets;
the successful Galaxy S22 excludes a general SM8450, Samsung, v69 payload, or
model incompatibility.

The Edge 50 Fusion provider passed the research allowlist and loaded its
libraries, but emitted no QNN IR and reproduced the CPU result at CPU speed.
This proves only that the current LiteRT 2.1.5 and QAIRT v73 package does not
verify HTP delegation on that SM7435 device. A newer plugin or explicit SoC
configuration would require a separate experiment.

## Closure decision

The exploration establishes that generation-specific Qualcomm packages can
run the complete model on representative Snapdragon 8 Gen 1 through 8 Elite
devices. It does not justify a production SoC allowlist, redistribution of the
QAIRT payload, or integration into Booming SS yet.

Further Qualcomm devices, MediaTek bring-up, Tab S8 recovery, and formal
listening are deferred. The next runtime experiment should validate downloading
and loading the common LiteRT CPU core from app-private storage. Vendor packs
can be composed only after that core contract is proven.

Raw App Live reports, QNN IR, logcat, tensors, WAV outputs, and relay sync
manifests remain local test material. They are intentionally excluded from Git.
