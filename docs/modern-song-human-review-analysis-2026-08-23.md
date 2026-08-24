# Modern Song S/R/K/I Review Analysis

Status: completed locally on 2026-08-23. This report analyzes the manual
review of 912 events from the 57 style-approved modern-song candidates. It
does not start training or alter any audio.

## Mark semantics

`S` is a strict perceptual subset of `R`:

- `S`: especially conspicuous residual; its loudness and/or clear consonants
  make the lyric unusually easy to hear;
- `R`: residual vocal/harmony/effect should be removed, but it is less
  conspicuous than S;
- `K`: current H50 result is acceptable;
- `I`: H50 removed extra non-vocal content relative to Inst 3.

S and R use the same Inst 3 residual target. S changes priority and sampling,
not target direction.

## Totals

| Mark | Events | Share |
| --- | ---: | ---: |
| `S` | 210 | 23.0% |
| `R` | 265 | 29.1% |
| `K` | 378 | 41.4% |
| `I` | 59 | 6.5% |

Aggressive-removal events (`S+R`) account for 475/912 events (52.1%). S occurs
in 39 of the 57 songs.

## Does S separate from ordinary R?

Yes, the label is reflected in the existing local metrics:

| Metric | S | R | S minus R |
| --- | ---: | ---: | ---: |
| Event-score mean | -20.00 dBFS | -23.03 dBFS | +3.04 dB |
| Event-score median | -19.42 dBFS | -23.13 dBFS | +3.71 dB |
| 50 ms projection mean | -20.56 dBFS | -23.79 dBFS | +3.23 dB |
| 100 ms projection mean | -22.66 dBFS | -25.19 dBFS | +2.53 dB |
| 200 ms projection mean | -23.84 dBFS | -27.45 dBFS | +3.61 dB |

80% of S events are stronger than the median event-score of ordinary R. This
does not replace listening, but it confirms that S is a useful high-priority
label rather than arbitrary relabeling.

## Category distribution

| Category | S | R | K | I | S share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hip-Hop/R&B | 73 | 47 | 51 | 5 | 41.5% |
| Pop/Synth | 53 | 47 | 65 | 11 | 30.1% |
| Dance/Electronic | 19 | 11 | 23 | 11 | 29.7% |
| Latin modern | 27 | 48 | 59 | 10 | 18.8% |
| Singer-songwriter | 27 | 62 | 70 | 1 | 16.9% |
| Pop Rock | 11 | 50 | 110 | 21 | 5.7% |

The strongest S concentration is in Hip-Hop/R&B and Pop/Synth, which is useful
for the intended short-consonant problem. Pop Rock has relatively few S events
and more K/I material, so it should not dominate an S-focused training arm.

## Concentration and deduplication

Using a 3-second gap to group nearby S centers:

- 210 S events;
- 187 temporal S clusters;
- 18 clusters contain multiple S events.

The largest song-level concentrations are:

| Track | Category | S/R/K/I |
| --- | --- | ---: |
| In The Night - Game of Thrones Remake | Hip-Hop/R&B | 14/2/0/0 |
| Love Love Love | Dance/Electronic | 14/1/1/0 |
| Say Goodbye | Pop/Synth | 13/2/1/0 |
| Stronger Living Feat Wordsmith | Hip-Hop/R&B | 13/2/1/0 |
| El Juego | Latin modern | 13/0/3/0 |
| Dance Like A Stripper | Hip-Hop/R&B | 11/4/1/0 |
| Ballade somnifère | Pop/Synth | 9/2/5/0 |
| Intergalactique | Hip-Hop/R&B | 9/6/1/0 |
| Where The Comet Falls | Singer-songwriter | 9/7/0/0 |

The first pilot must cap one event per S cluster and at most two clusters per
song. Otherwise a few songs can supply most of the gradient and recreate the
narrow-data problem seen in earlier experiments.

The complete S-only sortable list is:

`data/modern-song-inst3-event-listening/s-focused-event-candidates.csv`

## Recommended next experiment

### Decision

Run one S-focused pilot using the current 57 style-approved songs before
acquiring another candidate batch. The present pool already provides enough
high-confidence S events to test salience-aware sampling. Do not add more
ordinary R dose in this round.

### Equal-budget arms

Start both arms from `H50-continuation+5/step-1600.pt`:

- `S0-control`: 640 MUSDB continuous records plus 64 deterministic repeated
  MUSDB records per pass;
- `S1-focused`: the same 640 MUSDB records plus 64 S-event records per pass.

For S1, select 16-20 training songs across categories, one event per 3-second
S cluster and no more than two clusters per song. Repeat the selected unique
events only as needed to reach 64 external records. Freeze a separate
artist/song-disjoint holdout of 10-12 S-containing songs before selecting
training windows. The remaining style-approved songs stay controls or stress
tests.

Keep the first S1 loss contract identical to C1 so the experiment isolates the
label/sampling effect:

- 128-frame continuous-context assembly;
- residual-vocal output;
- `V_T = mixture - Inst3Instrumental`;
- 100 ms event core with 25 ms soft guard;
- H50 anchor outside the event;
- AdamW, `1e-6`, batch 4, frozen BatchNorm, five passes;
- external fraction `64/704 = 9.09%`.

The S label changes which windows are sampled, not the target or loss sign.

### Optional second arm

Only after the first smoke test, compare temporal masks:

- `S50`: 50 ms core + 50 ms guard;
- `S100`: 100 ms core + 25 ms guard.

This tests whether consonant-scale supervision needs a narrower audio-domain
core. It should not be combined with new data, a new teacher, or a higher
learning rate in the same run.

## Evaluation gates

Primary metrics:

- held-out S events at 50/100/200 ms, especially 100 ms p95 and maximum;
- fraction of held-out S events moving toward Inst 3;
- known short-hotspot listening on the 12 private songs.

Safety metrics:

- ordinary R, K, and I events reported separately;
- MUSDB 20-song calibration/internal-test;
- seam ratios, first-difference mechanical-noise proxy, clipping, and finite
  values;
- a new modern-song holdout not used to choose S windows.

Suggested go/no-go thresholds:

- held-out S 100 ms p95 and maximum improve by at least 0.5 dB in the same
  direction;
- at least 70% of held-out S events improve;
- K/I local error does not worsen by more than 0.1 dB;
- no new mechanical sound in blind listening;
- the known private hotspots become repeatedly distinguishable or audibly less
  intelligible, not merely numerically different.

If S1 again produces a roughly 0.5 dB machine gain but remains impossible to
distinguish blind and leaves the same hotspots intact, stop increasing the
external training dose. The likely bottleneck is then temporal resolution,
phase/model capacity, or the loss/assembly path; test a dedicated short-window
retraining or higher-capacity direction instead.

## Artifacts

- Analysis JSON:
  `data/modern-song-inst3-event-listening/human-review-analysis.json`
- S candidate CSV:
  `data/modern-song-inst3-event-listening/s-focused-event-candidates.csv`
- Full review CSV:
  `data/modern-song-inst3-event-listening/human-review-template.csv`
