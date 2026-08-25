# FMA S86 Review Comparison

## Result

The new review covers 790 events from 158 style-approved FMA songs. The
current model is `S86-event-only@pass-5`, rendered with continuous 128-frame
assembly. The current labels are:

| mark | meaning | events | share |
| --- | --- | ---: | ---: |
| `S` | conspicuous residual vocal, a subset of `R` | 48 | 6.1% |
| `R` | more vocal/harmony/spoken/effect removal desired | 72 | 9.1% |
| `K` | current result acceptable | 430 | 54.4% |
| `I` | H50/S86 removed extra non-vocal content relative to Inst 3 | 240 | 30.4% |

Thus the current aggressive-removal fraction (`S+R`) is 15.2%. `I` is not a
claim that Inst 3 removed non-vocal content; it records that the current model
went farther than Inst 3 at that event.

## Comparison With Earlier H50 Reviews

The older reviews used the `H50-continuation+5` checkpoint and different
candidate locations. The current set was rescanned after S86 rendering, so
raw event percentages are descriptive rather than a strict paired A/B result.
Same-song aggregate rates are the more stable comparison:

| shared comparison | current `S+R` | prior `S+R` | current minus prior | current `K` | prior `K` | current `I` | prior `I` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batch 1, 57 songs, current 5 vs prior 16 events/song | 23.5% | 52.1% | -28.6 pp | 44.6% | 41.4% | 31.9% | 6.5% |
| batch 2, 101 songs, current 5 vs prior 5 events/song | 10.5% | 56.2% | -45.7 pp | 60.0% | 37.4% | 29.5% | 6.3% |
| batch 2, 60 shared songs, current 5 vs prior selected 16 | 17.3% | 65.3% | -48.0 pp | 54.0% | 29.5% | 28.7% | 5.2% |

The direction is consistent across all three comparisons: the audible
residual-vocal problem has substantially receded, while extra non-vocal
removal is now the dominant risk. This agrees with the listening reports.

### Time-neighbour matching

For an additional, weaker check, current events were matched one-to-one to the
nearest prior event in the same song within 0.5 seconds:

| prior set | matched events | exact four-way mark agreement | aggressive agreement | current-only aggressive | prior-only aggressive |
| --- | ---: | ---: | ---: | ---: | ---: |
| batch 1 prior 16-event set | 208 | 56.7% | 71.6% | 6 | 53 |
| batch 2 prior 5-event set | 233 | 49.4% | 63.9% | 0 | 84 |
| batch 2 prior selected 16-event set | 195 | 49.2% | 63.1% | 0 | 72 |

The large `prior-only aggressive` counts are compatible with genuine
improvement, but they also include different candidate selection and subjective
listening-order effects. They should not be interpreted as a quantitative
separation score.

## Distribution and Training Safety

Current `S+R` counts by explicit prior split role are:

| role | events | `S+R` | `I` |
| --- | ---: | ---: | ---: |
| train | 295 | 32 | 81 |
| calibration | 85 | 12 | 27 |
| holdout | 125 | 9 | 41 |
| pending style role | 285 | 67 | 91 |

Only 10 explicit train songs have both aggressive events and zero `I` events,
providing 28 current aggressive event examples. The pending-role songs must
not be silently treated as training songs. The current labels are one review
round, so `S`/`R` should be treated as high-value perceptual examples, not
perfect target truth.

The historical S-only pressure run is also a warning. From pass 5 to pass 10,
the 100 ms positive-projection p95 moved only within roughly 0.03 dB, while
100 ms raw-miss p95 worsened from about `-29.22` to `-29.12 dBFS`. More
unconstrained S-only training is therefore unlikely to be the next useful
lever; it risks increasing the `I` side of the tradeoff.

The machine-readable comparison is:

`C:\Users\User\Documents\MusicSourceSeparation\data\modern-song-fma-all-s86-event-pass5\review-analysis.json`

## Recommended Sequence

### Stage 1: targeted residual-vocal continuation

Run a small, split-aware experiment before any accompaniment restoration.

1. Start from `S86-event-only@pass-5` and keep a frozen no-update control with
   the same number of updates.
2. Use the current `S`/`R` event core with the existing Inst 3 residual target
   and keep the S86 residual as the non-event anchor. Do not remove the anchor
   and do not increase the learning rate.
3. For the first formal arm, use only explicit train songs with no current `I`
   marks. Sample `S` at a higher priority than `R`, cap one event per song per
   pass, and keep the external records at roughly 5--10% of updates. This is a
   deliberately small pilot rather than another all-S stress test.
4. Keep calibration and holdout songs untouched. Assign the pending-role batch
   1 songs to train/calibration/holdout only after a frozen song-level split;
   do not select their role based on the new labels.
5. Use a low, bounded continuation such as three passes at the established
   learning rate, saving pass 1 and pass 3. Compare 50/100/200 ms projection
   p95 and maximum, raw miss, I-region damage, seams, and continuous 12-song
   listening.

The first arm should answer whether a small set of cleaner, human-confirmed
S/R examples can reduce the remaining audible leaks without repeating the
large `I` increase. If it cannot, stop adding FMA vocal events and move to
model/time-resolution work rather than increasing the S dose again.

### Stage 2: separate accompaniment restoration

After Stage 1 is selected, restore accompaniment in separate arms. Do not mix
restoration and new vocal-removal sampling in the first run.

- **MUSDB restoration arm:** use true MUSDB instrumental
  `drums + bass + other` as the reference. In the residual-vocals output
  representation, the corresponding target is `mixture - trueInstrumental`.
  Select windows by measurable S86 excess attenuation and keep a song-disjoint
  validation set.
- **FMA preference arm:** use only manually reviewed `I` events and move the
  accompaniment toward the native Inst 3 output. This is a preference target,
  not proof of the correct instrumental stem.
- **Combined arm:** only after the two arms are understood, use domain-gated
  targets: MUSDB true instrumental on MUSDB restoration regions, Inst 3 on FMA
  `I` regions, Inst 3 residual target on S/R regions, and the current S86
  output as the anchor elsewhere.

For all restoration arms, evaluate `I`-region instrumental error separately
from S/R residual projection. Require no meaningful regression on known S/R
hotspots, low-vocal instrumental SDR, seams, clipping, or mechanical-noise
proxies. Re-listen a stratified sample of the strongest `I` events because a
numeric difference from Inst 3 does not imply equal perceptual importance.

The restoration loss should remain gated to the reviewed region. Applying a
whole-song MUSDB target would erase the current Inst 3-directed behavior and
would not answer whether the restoration specifically repairs the audible
damage.
