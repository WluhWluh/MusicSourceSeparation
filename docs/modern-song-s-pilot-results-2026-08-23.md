# Modern Song S-Focused Pilot

Status: completed locally on 2026-08-23. This is a local non-commercial
research experiment. Audio, caches, and checkpoints remain local artifacts.

## Objective

Test whether prioritizing manually labeled `S` events, a high-severity subset
of `R`, reduces conspicuous short vocal leakage more effectively than ordinary
equal-budget MUSDB continuation.

`S` means that residual loudness and/or clear consonants makes the lyric
especially easy to hear. It uses the same Inst 3 residual target as `R`; only
sampling priority changes.

## Frozen split and contract

- 57 style-approved modern songs were available;
- 16 songs / 32 unique S clusters used for S1 training;
- 12 S-containing songs held out by category and whole song;
- remaining songs were not used as modern holdout evidence;
- one representative per 3-second S cluster, maximum two clusters per song;
- 32 unique S windows repeated to 64 external records per pass;
- 640 MUSDB records plus 64 extra records per pass;
- external share: `64/704 = 9.09%`;
- five passes, 880 updates, batch 4;
- 128-frame continuous-context overlap-save;
- AdamW state restored from H50-continuation+5 step 1600;
- learning rate `1e-6`, frozen BatchNorm, gradient clipping 1.0;
- 100 ms event core plus 25 ms soft guard;
- no official MUSDB test songs.

The control `S0-control` uses deterministic repeated MUSDB records for the 64
extra slots. `S1-focused` uses the frozen S event windows.

## Checkpoints

| Arm | File | SHA-256 |
| --- | --- | --- |
| S0 control | `runs/S0-control/step-2480.pt` | `d96a22de4f200c8c306286de5f27cfe70b60b0fa7781acf2554d0481bff92211` |
| S1 focused | `runs/S1-focused/step-2480.pt` | `c624b3953db4183a52cd2f7989f88ce99f947c2273a23adb1ec7876461632f9e` |

The training run completed with finite loss and gradients. Preloading all
cache files kept GPU use high: S0 sampled mean was approximately `95.2%`, and
S1 was approximately `88.6%` with most samples between 83% and 99%.

## Modern S holdout

There are 61 S, 71 R, 53 K, and 7 I events in the 12-song holdout.
S1-S0 local target-error deltas are negative when S1 moves closer to Inst 3.

| Mark | 50 ms mean | 100 ms mean | 200 ms mean | 100 ms improved |
| --- | ---: | ---: | ---: | ---: |
| S | -0.990 dB | -0.897 dB | -0.747 dB | 56/61 |
| R | -0.628 dB | -0.597 dB | -0.564 dB | 56/71 |
| K | -0.323 dB | -0.278 dB | -0.210 dB | 33/53 |
| I | -0.460 dB | -0.306 dB | -0.145 dB | 4/7 |

The mean S improvement is strong and exceeds the initial 0.5 dB target. It
also transfers beyond the selected training songs. However, the tail is not
uniformly safe:

- S 100 ms delta p95: `+0.037 dB`;
- S 100 ms worst delta: `+0.892 dB`;
- S 100 ms median: `-0.484 dB`.

Thus most S events improve, but the worst S event is not guaranteed to improve.
The K/I counts are small in this holdout and should not be treated as a stable
safety estimate by themselves.

## MUSDB safety

Relative to S0 control, S1 changes the 20-song MUSDB mean by:

- instrumental SDR: `-0.162 dB`;
- positive vocal projection: `-0.441 dB`;
- Inst 3 removed-content projection: `-1.551 dB`;
- teacher residual miss RMS: `+0.180 dB`.

At 100 ms, positive projection p95 improves `-0.768 dB`, while raw miss p95
worsens `+0.247 dB`. This is a larger aggressive-removal tradeoff than C1 and
must be checked by ear.

Mean seam p95 ratio is `1.850` for S1 versus `1.796` for S0; the worst seam max
is lower (`8.66` versus `9.55`), but the p95 movement is a mild regression.

## Private 12-song listening set

The continuous listening files are under:

`data/modern-song-s-pilot/listening-12/`

Variants:

- `Source-H50-continuation+5`;
- `S0-control`;
- `S1-focused`.

S1-S0 mean private changes are:

- coherent retained relative content: `-1.847 dB`;
- target error: `+0.227 dB`;
- 100 ms positive projection p95: `-1.238 dB`;
- 100 ms positive projection max: `-0.934 dB`.

S1 is therefore audibly worth checking: it moves more strongly than C1, but
the target-error increase means stronger removal may also include non-vocal
content. The 12-song blind listening decision remains the final product gate.

## Human listening result

The user found that S1 does change the sound at the marked leak hotspots. The
remaining vocal residue is slightly reduced, but the difference is small: it
usually requires immediate neighboring replay of the short excerpts to hear
the advantage over S0 or the earlier C1 result. Full-song blind listening does
not reliably distinguish S1, and the conspicuous leaks are not fully removed.
No clearly audible additional accompaniment damage was reported.

This is a useful partial success rather than a product-quality win. It
confirms that S-aware sampling changes the intended local content, while also
showing that the current 128-frame model/loss path is reaching a low audible
return for this intervention.

## Decision

S1 passes the machine-learning signal test:

- 56/61 held-out S events improve at 100 ms;
- mean held-out S gain is approximately 0.90 dB;
- ordinary R events also improve;
- private short-event metrics move strongly toward Inst 3.

It does not yet pass the safety/product gate automatically:

- worst S holdout event worsens;
- MUSDB SDR and raw miss regress;
- seam p95 rises modestly;
- private target error rises.

The S1 checkpoint is currently the best machine- and excerpt-level S-focused
candidate, but it is not yet a replacement for the existing H50 baseline. The
next experiment should be one bounded continuation from S1, not a new data
pool or a larger S sampling fraction:

- continue S1 for five additional passes, restoring its AdamW state;
- keep the same 16-song S training split, 32 unique cluster representatives,
  and 64 external records per pass;
- keep S0 continuation as an equal-budget control;
- save pass 1/3/5 continuation checkpoints;
- evaluate the same 12 private songs and the frozen 12-song S holdout;
- compare the worst S hotspots, not only average p95;
- stop if the next five passes do not produce a further clearly audible change.

This tests whether the small excerpt-level gain has remaining optimization
headroom without confounding it with new music. It should be the final bounded
training attempt under the current 128-frame contract. If the continuation
again improves numeric S metrics but remains blind-indistinguishable and leaves
the same hotspots intact, stop increasing S data and move to temporal
resolution, phase behavior, or model capacity.

## Artifacts

- Main report: `data/modern-song-s-pilot/reports/s-report.json`;
- modern holdout report: `data/modern-song-s-pilot/reports/modern-holdout-evaluation.json`;
- private analysis: `data/modern-song-s-pilot/reports/private-inst3-analysis.json`;
- listening report: `data/modern-song-s-pilot/listening-report.json`;
- S0/S1 final checkpoints under `data/modern-song-s-pilot/runs/`.
