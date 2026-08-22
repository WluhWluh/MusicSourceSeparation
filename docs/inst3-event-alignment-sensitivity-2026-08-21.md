# Inst 3 Event Alignment Sensitivity Diagnostic

Status: completed locally on 2026-08-21. This is a non-commercial MUSDB18
research diagnostic. It did not train a model, export a runtime artifact, or
use the official final-test split.

## Purpose

This test checks whether the brief residual-vocal leaks targeted by the Inst 3
hard-event work are caused primarily by where the event falls inside the
TFC-TDF useful output span. It compares the existing useful-stride placement
with an event-centered placement and two quarter-span shifts. The three
student checkpoints are evaluated on the same event and placement:

- `initial`: the original `vocals_epoch=891.ckpt` residual-vocals model;
- `h50-pass-50`: the V-R-H50 checkpoint;
- `lambda-0.50-pass-50`: the H50 event-weighted checkpoint.

The target remains the Inst 3 residual-vocal target:

```text
mixtureGt = vocals + drums + bass + other
teacherRemoved = mixtureGt - Inst3Instrumental
studentOutput = residual vocals
```

The local metrics are measured over 50, 100, and 200 ms around each event:

- `missRms`: residual error between the student residual and
  `teacherRemoved`;
- `positiveProjectionRms`: the positive projection of that error onto the
  teacher-removed signal.

More negative dBFS is better for both metrics. The alignment deltas below are
`center - stride`; a negative delta is an improvement.

## Frozen Contract

- 80 MUSDB18 train songs and 20 song-disjoint calibration/internal-test songs;
- top 3 100 ms events per song (Stage 1 hard-event entries for train songs;
  events derived with the same diagnostic for evaluation songs);
- 300 events total and 3,600 model/placement rows;
- useful output span: 119,808 samples, or 2.716735 seconds at 44.1 kHz;
- placements: `stride`, `center`, `center-minus-quarter`,
  `center-plus-quarter`;
- center start: `eventCenter - usefulSamples / 2`;
- quarter shifts: plus or minus `usefulSamples / 4`, clamped to the song;
- local windows: 50, 100, and 200 ms;
- CUDA execution on the repository environment, with the Inst 3 ONNX teacher
  using `CUDAExecutionProvider` and `CPUExecutionProvider` fallback;
- no official final-test songs and no private listening songs in the metrics.

Command:

```powershell
.\.tmp\vr-hard-env\Scripts\python.exe tools\analyze_inst3_event_alignment.py `
  --device cuda --threads 8 --top-events 3 `
  --output-root data\musdb18-inst3-event-alignment-corrected --force
```

The run completed with 300 events, 3,600 rows, and no CUDA, non-finite, or
shape failure. Elapsed time was 2,107.9 seconds on an NVIDIA GeForce RTX
4060 Laptop GPU.

## Correctness Correction

The first full run was discarded. The Stage 1 event JSON does not contain a
rank field; the initial reader therefore assigned rank `0` to all three
selected events, causing their placement map entries to overwrite one
another. The corrected runner assigns rank `0`, `1`, and `2` by the sorted
selection order. A smoke run confirmed that each song/rank has four distinct
placements, and the full corrected run has exactly 300 event groups with 12
rows per group.

The discarded output remains in the ignored data tree for forensic comparison
only and must not be used as an experiment result.

## Event Position and Coverage

The current stride placement puts a substantial fraction of event centers at
the useful-span edge or beyond it:

| role | events | center within 10% of edge | event center outside useful span |
| --- | ---: | ---: | ---: |
| train | 240 | 123 (51.3%) | 45 (18.8%) |
| calibration | 30 | 14 (46.7%) | 10 (33.3%) |
| internal-test | 30 | 13 (43.3%) | 7 (23.3%) |

Here, outside means that the event center is outside the selected useful
output span. The event's input audio can still overlap the window; it does not
mean that the whole model input is absent.

At 100 ms, full local-event coverage was:

| role | stride | center | center-minus-quarter | center-plus-quarter |
| --- | ---: | ---: | ---: | ---: |
| train | 147/240 | 240/240 | 238/240 | 240/240 |
| calibration | 14/30 | 30/30 | 30/30 | 30/30 |
| internal-test | 18/30 | 30/30 | 29/30 | 30/30 |

The paired comparisons below include only events for which both the stride and
the comparison placement have full coverage. This avoids treating a partial
window as a quality regression, but it also means the paired test cannot
fully answer the question for the most extreme boundary events.

## Held-Out Paired Results

The most relevant result is the 20-song internal-test set. Values are dB
deltas relative to the same event rendered with the current stride placement.
`miss med/mean` is the center-minus-stride delta for `missRms`; `projection
med/mean` is the corresponding delta for `positiveProjectionRms`.

| model | window | paired n | miss med / mean | projection med / mean |
| --- | ---: | ---: | ---: | ---: |
| initial | 50 ms | 18 | +0.291 / +0.501 | -0.004 / -0.089 |
| initial | 100 ms | 18 | +0.261 / +0.405 | -0.044 / -0.134 |
| initial | 200 ms | 14 | +0.279 / +0.522 | -0.030 / -0.121 |
| H50 | 50 ms | 18 | +0.053 / +0.270 | 0.000 / -0.020 |
| H50 | 100 ms | 18 | +0.081 / +0.123 | 0.000 / -0.025 |
| H50 | 200 ms | 14 | +0.360 / +0.347 | +0.001 / -0.016 |
| lambda-0.50 | 50 ms | 18 | +0.143 / +0.243 | 0.000 / -0.004 |
| lambda-0.50 | 100 ms | 18 | +0.113 / +0.092 | -0.001 / -0.011 |
| lambda-0.50 | 200 ms | 14 | +0.363 / +0.311 | 0.000 / -0.001 |

The calibration set points in the same direction for the miss metric. At 100
ms, the center miss median/mean deltas are `+0.192/+0.667 dB` for initial,
`+0.097/+0.256 dB` for H50, and `+0.134/+0.247 dB` for lambda-0.50. Its small
paired sample is not strong evidence of a general effect, but it does not
support a large center-placement gain.

The quarter shifts were similarly close to zero on internal-test. For H50 at
100 ms, center-minus-quarter versus stride had projection median/mean
`-0.025/-0.051 dB` and miss median/mean `+0.039/+0.127 dB`; center-plus-quarter
had projection `-0.005/-0.095 dB` and miss `+0.060/+0.103 dB`.

## Interpretation

The experiment separates two effects:

1. Centering reliably makes the short event fully covered. This is useful for
   diagnostics and could matter for a future event-specific inference path.
2. On the held-out events that permit a fair paired comparison, centering does
   not materially improve the student's match to the Inst 3 target. The
   positive-projection change is at most a few hundredths of a dB for H50 and
   lambda-0.50, while the full miss metric is slightly worse on average.

Therefore the proposed roughly 1 dB alignment gate is not met. The remaining
short leaks are unlikely to be explained by window placement alone. More
likely causes are model capacity, the ambiguity of the Inst 3 target, or the
fact that a single fixed-window output cannot represent the event cleanly at
the current residual-vocal model boundary.

This diagnostic does not test overlap-add reconstruction, crossfades, or the
Android streaming scheduler. It also does not prove that a centered window
could never help an individual boundary event; it shows that centering is not
a strong enough general training signal to justify C25/C50 training by itself.

## Decision and Next Step

Do not start a centered-window training run based on this result. Keep
V-R-H50 as the current listening baseline. The next focused experiment should
be a short, low-learning-rate local fine-tune from H50 pass-50 using the
frozen hard-event manifest, with an H50-output anchor outside the event region.
That isolates model adaptation at the remaining hotspots while preserving the
ordinary regions that already sound good. It should be evaluated with the
same 50/100/200 ms event metrics and the 12-song listening set.

## Artifacts and Provenance

Corrected JSON artifacts remain local and ignored:

- `data/musdb18-inst3-event-alignment-corrected/reports/inst3-event-alignment-report.json`
- `data/musdb18-inst3-event-alignment-corrected/alignment-rows.json`
- `data/musdb18-inst3-event-alignment-corrected/event-manifest.json`

Key hashes from the completed run:

| artifact | SHA-256 |
| --- | --- |
| alignment runner | `AB54170F2D8F0965D45CB243A8582AD5B759FEF129810D09CA551D2615D037A9` |
| initial checkpoint | `101921DAC943E1683452F293DC0D32DF53B694886541952BE26D9592D819C20D` |
| H50 checkpoint | `0E49F154B8F66F9827BAA197C287DB7A34C6902C932D72332EA34CDD33870705` |
| lambda-0.50 checkpoint | `DE55073AF5B8991607CFAF2836EDD3FA3B94DBAA528413F1EFF04423FDD557F1` |
| Inst 3 teacher | `2B7834E2972158D8C9864E7376E3A7D084079C80A23F38DC31C4B0A4E901A1CB` |
| oracle split manifest | `3A580E39C8754ABB902784F257FB251F1BA52E6BE678F5022343A1846252AB2B` |
| Stage 1 event report | `3BAF35352EDCA46D5CC7EC2B3C69956500D8A55BAE27B0642F83BE14BF2534B8` |

All generated audio, teacher-derived data, and checkpoints remain local and
under the MUSDB18 non-commercial research restrictions.
