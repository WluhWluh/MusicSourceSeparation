# Inst 3 TFC-TDF 24-Frame Dedicated Retraining

Status: completed locally on 2026-08-22. This is a non-commercial MUSDB18
research experiment. It does not modify product code, export a runtime, or
change the 128-frame Booming SS candidate.

## Objective

The prior 24-frame work only exported the 128-frame checkpoint at a shorter
static time dimension. This experiment rebuilt the input and Inst 3 residual
target spectra under a real 24-frame contract and retrained from the best
H50 residual-vocals checkpoint.

The fixed short-window contract is:

- 24 STFT frames;
- 2,048 FFT and 1,024 hop at 44.1 kHz;
- four left and four right context hops;
- 23,552 input samples;
- 15-hop / 15,360-sample useful output stride;
- continuous-context overlap-save assembly;
- 65.22% output efficiency.

The target semantic remains:

```text
M   = mixtureGt
V_T = M - Inst3Instrumental
```

Every short input and target spectrum was regenerated from PCM. No 128-frame
spectrum was cropped or reused.

## Frozen Training Contract

- 80 MUSDB18 training songs;
- 20 song-disjoint calibration/internal-test songs;
- eight training records per song per pass;
- 640 records per pass and 1,600 optimizer updates per arm;
- 10 passes, checkpoints at 0/5/10 passes;
- batch size 4, AdamW, learning rate `1e-6`, weight decay 0;
- warm-start from `V-R-H50` pass 50;
- frozen BatchNorm running statistics and gradient clipping at 1.0;
- CUDA on the RTX 4060 Laptop GPU;
- no official final-test songs, audio-domain loss, 16-frame model, QNN, or
  product integration.

Two schedules used the same song order and seed:

- `F24-Inst3-U`: eight uniform windows;
- `F24-Inst3-H25`: four uniform windows plus four H50 event-centered windows.

The H25 schedule had no uniform/event start collisions. Each cache contains
12 union records with shape `[12, 4, 1025, 24]`; each arm trains on eight of
those records per song per pass.

## Short-Window Baseline

`F24-static-H50` means the unmodified H50 weights run under the new 24-frame
assembly. It is the correct zero-update baseline for this experiment. The
128-frame H50 pass-50 result remains the quality baseline.

| checkpoint | instrumental SDR | low-vocal instrumental SDR | accompaniment vocal projection | derivative excess p95 |
| --- | ---: | ---: | ---: | ---: |
| 128-frame H50 pass 50 | 13.777 dB | 19.272 dB | -16.632 dB | 0.870 dB |
| F24-static-H50 | 13.302 dB | 16.641 dB | -16.631 dB | 1.105 dB |

The static short contract therefore loses `0.475 dB` instrumental SDR and
`2.632 dB` low-vocal instrumental SDR before any retraining. Its aggregate
vocal projection is essentially unchanged, but its boundary artifact proxy
is already higher.

## Retraining Results

Values in the delta columns are relative to `F24-static-H50`; negative vocal
projection deltas are better.

| checkpoint | instrumental SDR delta | low-vocal SDR delta | accompaniment vocal projection delta | 100 ms miss p95 delta | 100 ms positive projection p95 delta | 100 ms positive projection max delta | derivative excess p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F24-Inst3-U pass 5 | -0.326 dB | +0.003 dB | +2.648 dB | +0.685 dB | +2.863 dB | +0.153 dB | 2.807 dB |
| F24-Inst3-U pass 10 | -0.565 dB | -0.168 dB | +3.705 dB | +1.233 dB | +4.322 dB | +0.159 dB | 3.000 dB |
| F24-Inst3-H25 pass 5 | -0.216 dB | +0.072 dB | +2.364 dB | +0.536 dB | +2.392 dB | +0.136 dB | 2.749 dB |
| F24-Inst3-H25 pass 10 | -0.320 dB | +0.043 dB | +3.175 dB | +0.662 dB | +3.262 dB | +0.146 dB | 2.762 dB |

Both retrained arms worsened the main vocal-projection metric on all 20
evaluation songs. They also raised the short-window derivative artifact
proxy substantially. No run produced clipped samples or non-finite windows,
so this is a quality/training failure rather than a numerical crash.

The H25 arm is less bad than U at pass 5 and pass 10, but neither is close to
the predeclared quality gate. The worse projection is not explained by a
single outlier: the aggregate 100 ms positive-projection p95 deltas are
positive on every evaluation song for both final arms.

## Listening Outputs

The runner generated 36 validated PCM16 FLAC files for the 12-song private
listening set:

`data/musdb18-inst3-f24-retrain/listening-12/`

Variants:

- `F24-static-H50`;
- `F24-Inst3-U@pass-10`;
- `F24-Inst3-H25@pass-10`.

The machine report is:

`data/musdb18-inst3-f24-retrain/reports/inst3-tfc-f24-retrain-report.json`

No human listening judgment is included yet. The files should be treated as
diagnostic A/B material; neither retrained arm is a candidate replacement.

## Decision

The first short-window retraining phase fails the host quality gate:

1. The static 24-frame contract already damages instrumental fidelity,
   especially in low-vocal passages.
2. Ten-pass U/H25 retraining moves further away from both H50 and the Inst 3
   removal target on the held-out projection metrics.
3. The boundary artifact proxy rises from `1.105 dB` for static F24 to about
   `2.75-3.00 dB` after retraining.

Do not proceed to 16-frame retraining or Android benchmarking from these
weights. Keep the artifacts for diagnosis and retain H50 as the baseline.
The next short-window investigation, if resumed, should first isolate the
F24 target/optimization mismatch with a tiny supervised reconstruction test
and lower learning-rate/partial-layer adaptation; it should not combine a
new window size with more aggressive teacher sampling.

## Provenance

| artifact | SHA-256 |
| --- | --- |
| H50 source checkpoint | `0E49F154B8F66F9827BAA197C287DB7A34C6902C932D72332EA34CDD33870705` |
| F24 U pass-10 checkpoint | `F05A9598539FD3A43ED6CC0AFED4203218249FEAF4A7FE2D1298E89BDD1E0C64` |
| F24 H25 pass-10 checkpoint | `DD1B846D09FFDB8F1F46ACB5DE44A8E788C4801F7FD916D128D7D4156E2197A3` |
| runner | `FBAD5B8B8282969870A449D89B389C52EFA61368EA735DA3A16C1FA383303125` |

All MUSDB18 audio, teacher-derived targets, caches, checkpoints, and listening
files remain local under the dataset's non-commercial restrictions.
