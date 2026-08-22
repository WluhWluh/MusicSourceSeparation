# Inst 3 V-R Short-Event Audio-Domain Objective

Status: completed locally on 2026-08-22. This is a non-commercial MUSDB18
research experiment. It does not modify product code or publish a runtime
model.

## Objective

The preceding event-centered spectral-target experiment improved the general
positive-projection distribution but did not change the global worst short
leak hotspot. This experiment changed only the loss domain: the model still
outputs residual vocals, but the loss is evaluated after a differentiable
`torch.istft` in PCM space.

The fixed semantic contract is:

```text
M       = mixtureGt
V_T     = M - Inst3Instrumental
V_H50   = H50 residual output
V_hat   = student residual output
```

The objective is a weighted audio Charbonnier loss:

```text
L_event  = Charbonnier(V_hat - V_T) on the event weights
L_anchor = Charbonnier(V_hat - V_H50) on the complement
L_total  = L_event + L_anchor
```

Two arms were trained using the identical schedule:

- `audio-event-core`: exact 50 ms event interval;
- `audio-event-guard25`: exact event interval with a 25 ms linear guard.

The target and anchor PCM cache was generated from the existing
event-centered `targetResidualSpec` and `anchorResidualSpec`; no new teacher
inference or source decoding was performed.

## Frozen Contract

- 80 MUSDB18 training songs and 20 song-disjoint calibration/internal-test
  songs;
- four event-centered records per training song;
- 1,600 records total, 320 records per pass;
- five passes and 400 optimizer updates per arm;
- batch size 4, AdamW, learning rate `1e-6`, weight decay 0;
- initialization from `V-R-H50` pass-50;
- frozen BatchNorm running statistics and gradient clipping at 1.0;
- CUDA on the RTX 4060 Laptop GPU;
- 44.1 kHz, 2,048 FFT, 1,024 hop, 128 TFC-TDF frames;
- no official final-test songs, 24-frame export, QNN, or product integration.

The torch iSTFT implementation matched the existing NumPy contract with a
smoke maximum absolute error of `2.98e-7` and RMSE `2.65e-8`. Both arms ran an
8-update CUDA gradient smoke with finite loss and gradients before the full
run.

## Holdout Results

Values below are relative to the fixed H50 pass-50 checkpoint. Negative
projection deltas are better. The aggregate maximum is the important
individual-hotspot diagnostic; p95 describes the broader event distribution.

| arm | instrumental SDR delta | low-vocal SDR delta | accompaniment vocal projection delta | 100 ms miss p95 delta | 100 ms positive projection p95 delta | 100 ms positive projection max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `audio-event-core` pass 5 | +0.001 dB | -0.028 dB | -0.045 dB | +0.039 dB | -0.082 dB | -0.0003 dB |
| `audio-event-guard25` pass 5 | +0.001 dB | -0.024 dB | -0.035 dB | +0.034 dB | -0.064 dB | -0.0003 dB |

The same pattern is visible at 50 and 200 ms. Core was marginally stronger
than guard25, but neither approached the predeclared `0.5 dB` hotspot gate.
The core arm improved the per-song 100 ms positive-projection p95 on 19/20
evaluation songs; guard25 did so on 19/20. This is a small distribution
improvement, not a reliable solution for the worst single event.

The raw miss p95 moved slightly worse at 100 ms for both arms. This mismatch
between projection p95 and raw miss confirms that the small gain should not be
interpreted as uniformly removing all H50-to-Inst-3 residual content.

## Artifact Checks

| checkpoint | derivative excess p95 | derivative excess max | clipped samples | non-finite windows |
| --- | ---: | ---: | ---: | ---: |
| H50 pass 50 | 0.870 dB | 3.281 dB | 0 | 0 |
| audio-event-core pass 5 | 0.877 dB | 3.277 dB | 0 | 0 |
| audio-event-guard25 pass 5 | 0.877 dB | 3.279 dB | 0 | 0 |

The first-difference proxy is diagnostic rather than a perceptual score. No
new clipping, non-finite window, or CUDA failure occurred.

## Listening Outputs

The runner produced 36 validated PCM16 FLAC files for the 12-song private
listening set:

`data/musdb18-inst3-vr-event-audio/listening-12/`

Variants:

- `H50-pass-50`;
- `audio-event-core@pass-5`;
- `audio-event-guard25@pass-5`.

The machine report is:

`data/musdb18-inst3-vr-event-audio/reports/inst3-vr-event-audio-report.json`

No human listening judgment is included yet.

## Decision

The audio-domain loss is numerically correct and produces a modest aggregate
p95 improvement, but it does not solve the global worst short-leak hotspot.
`audio-event-core` is slightly preferable to `guard25`, while both remain far
below the threshold needed to justify retaining a new listening winner.

Do not export or integrate either arm. The next experiment should change model
capacity or temporal resolution, such as a separately trained shorter-window
student, rather than continue tuning the same 128-frame loss with more guard
width or more passes. Keep V-R-H50 as the product/listening baseline until a
candidate passes both the hotspot gate and blind listening.

## Provenance

| artifact | SHA-256 |
| --- | --- |
| H50 pass-50 checkpoint | `0E49F154B8F66F9827BAA197C287DB7A34C6902C932D72332EA34CDD33870705` |
| audio-event-core pass-5 checkpoint | `AFE956E9B201F7C03C47C48D9A2F0067D8D81A2E6F707703CA644F09863802C8` |
| audio-event-guard25 pass-5 checkpoint | `4378E60C00CB6DC75D7CF25D82F0CA99355B5A581DC00BD68724929A11B17373` |
| runner | `1EE2EA52ADECF4FCD881847DC7E1CE7E318A473F30DF8721C7C92CAA54EA3FE` |

The checkpoint, teacher-derived cache, and all audio remain local under the
MUSDB18 non-commercial research restrictions.
