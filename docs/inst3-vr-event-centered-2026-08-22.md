# Inst 3 V-R Event-Centered Local-Target Experiment

Status: completed locally on 2026-08-22. This is a non-commercial MUSDB18
research experiment. It does not modify product code or publish a runtime
model.

## Objective

The preceding miss-driven sampling run did not improve the most abrupt vocal
leak hotspots. This run tested whether giving each difficult H50 miss a fresh
context window, with the event near the useful-span center, would make the
remaining event easier for the 128-frame TFC-TDF model to learn.

The event definition was:

```text
teacherResidualMiss = H50Instrumental - Inst3Instrumental
```

For every one of the 80 training songs, all Stage 1 candidate windows were
rescanned with the H50 pass-50 model and Inst 3. The top four non-overlapping
50/100 ms blocks were selected by positive projection RMS, with raw miss RMS
as a tie-break. In this run all 320 selected blocks were 50 ms blocks.

Each selected block received a 119,808-sample context window centered on its
block center and clamped to the song boundaries. The event interval was mapped
to TFC-TDF frames by centered FFT-support overlap.

## Frozen Contract

- 80 MUSDB18 training songs;
- 20 song-disjoint calibration/internal-test songs;
- four event-centered records per training song;
- 1,600 records total, 320 records per pass;
- five passes and 400 optimizer updates per arm;
- batch size 4, AdamW, learning rate `1e-6`, weight decay 0;
- initialization from `V-R-H50` pass 50;
- residual-vocals output semantic;
- frozen BatchNorm running statistics and gradient clipping at 1.0;
- CUDA on the RTX 4060 Laptop GPU;
- no official final-test songs, 24-frame export, QNN, or product integration.

The two arms used the exact same centered records and schedule:

- `event-centered-continuation`: Inst 3 residual target on every frame;
- `event-centered-local-anchor`: Inst 3 target on event frames and frozen H50
  residual output outside the event mask.

## Results

Negative deltas are better for the vocal projection and miss metrics. Deltas
are relative to the fixed H50 pass-50 checkpoint.

| checkpoint | instrumental SDR delta | accompaniment vocal projection delta | 100 ms miss p95 delta | 100 ms positive projection p95 delta | 100 ms positive projection max delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| H50 pass 50 | 0.000 dB | 0.000 dB | 0.000 dB | 0.000 dB | 0.000 dB |
| continuation pass 5 | -0.032 dB | -0.266 dB | +0.080 dB | -0.341 dB | -0.003 dB |
| local-anchor pass 5 | -0.000 dB | -0.006 dB | +0.025 dB | -0.008 dB | -0.000 dB |

The continuation arm improved the 100 ms positive-projection p95 on 19/20
evaluation songs and the local-anchor arm improved it on 18/20. This is an
aggregate distribution improvement, not a solution to the worst individual
leak: the continuation maximum changed by only `-0.003 dB`, and the
local-anchor maximum was effectively unchanged.

The continuation arm also increased the 100 ms raw miss p95 by `+0.080 dB`
and the raw miss maximum by `+0.045 dB`. Its ordinary instrumental SDR fell
by `0.032 dB`. The local-anchor arm stayed close to H50 and did not produce a
meaningful audible-scale objective change.

No checkpoint produced clipped samples or non-finite evaluation windows. The
first-difference artifact proxy was:

| checkpoint | derivative excess p95 | derivative excess max | clipped samples | non-finite windows |
| --- | ---: | ---: | ---: | ---: |
| H50 pass 50 | 0.870 dB | 3.281 dB | 0 | 0 |
| continuation pass 5 | 0.794 dB | 3.187 dB | 0 | 0 |
| local-anchor pass 5 | 0.871 dB | 3.281 dB | 0 | 0 |

## Listening Outputs

The runner produced 36 validated PCM16 FLAC files for the 12-song private
listening set:

`data/musdb18-inst3-vr-event-centered/listening-12/`

Variants:

- `H50-pass-50`;
- `event-centered-continuation@pass-5`;
- `event-centered-local-anchor@pass-5`.

The machine-generated report is:

`data/musdb18-inst3-vr-event-centered/reports/inst3-vr-event-centered-report.json`

No human listening judgment is included in this report yet.

## Decision

Event-centered placement helped the continuation arm reduce the general
positive-projection p95, but it did not reduce the worst short-leak hotspot.
The local-anchor arm preserved H50 well but was too weak to create a useful
audible change. Neither arm should replace V-R-H50 as the listening baseline
without a favorable blind-listening result.

The result suggests that the remaining abrupt leak is not primarily caused by
window placement. The next useful direction is a dedicated short-event
audio-domain objective or a model/capacity change, evaluated against the fixed
V-R-H50 listening baseline. More event-centered continuation at the same
learning rate is not justified by this run.

## Provenance

| artifact | SHA-256 |
| --- | --- |
| H50 pass-50 checkpoint | `0E49F154B8F66F9827BAA197C287DB7A34C6902C932D72332EA34CDD33870705` |
| continuation pass-5 checkpoint | `984A05C3B2432AF95A2A23D7F4DF13BFE0904C04247062D063C606473A1785EC` |
| local-anchor pass-5 checkpoint | `3C34E459B87B9DFE57BA5CB37003FEC5A4AA9A5B9AEE973A02EBF216E538B7D5` |
| report execution runner | `CCA60926AC508EB1B2E7F5164BE1FD71A8FA522E0111DE6462033417A99A17F2` |

The local-anchor checkpoint hash and machine report hash are recorded in the
JSON report and are intentionally not repeated as release metadata. The
checkpoint, teacher-derived cache, and all audio remain local under the
MUSDB18 non-commercial research restrictions.
