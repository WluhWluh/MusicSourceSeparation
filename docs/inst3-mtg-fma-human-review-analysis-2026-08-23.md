# MTG-Jamendo/FMA Human Review Analysis

Status: completed locally on 2026-08-23. The analysis uses the completed
manual review in `human-review-template.csv`; it does not modify the review
marks or start training.

## Correct interpretation of the marks

The paired file is ordered as:

```text
H50-continuation+5 accompaniment -> 300 ms silence -> Inst 3 accompaniment
```

The mark describes the current H50 result relative to Inst 3 and the user's
listening preference:

- `R`: H50 retains vocal, harmony, spoken, or vocal-effect content that should
  be removed further. The desired correction moves the student toward Inst 3
  by increasing the residual-vocal estimate.
- `I`: H50 has removed extra non-vocal content that should be retained relative
  to Inst 3. This is a student over-removal/restoration signal, not an error
  attributed to Inst 3. The desired correction moves the student toward Inst 3
  in the opposite direction by reducing the residual-vocal estimate.
- `K`: the current H50 result is acceptable; no teacher-directed correction is
  justified by this event.

Consequently, both `R` and `I` can be teacher-aligned correction examples, but
they should be kept as separately reported directions. `K` is a preservation
control.

## Review totals

| Mark | Count | Share |
| --- | ---: | ---: |
| `R` | 161 | 35.9% |
| `K` | 222 | 49.6% |
| `I` | 65 | 14.5% |

All 448 rows are marked and all event IDs resolve to the generated report.

| Pool | R | K | I |
| --- | ---: | ---: | ---: |
| Expansion candidate pool (256 events) | 95 (37.1%) | 133 (52.0%) | 28 (10.9%) |
| Supplemental train role (128 events) | 31 (24.2%) | 70 (54.7%) | 27 (21.1%) |
| Supplemental holdout role (64 events) | 35 (54.7%) | 19 (29.7%) | 10 (15.6%) |

The supplemental-train role has the weakest R-to-I balance. That is a warning
against treating the role name as evidence that these tracks are suitable for
aggressive training.

## What the song-level marks say

### Clean R-heavy examples

These songs have enough reviewed R events and little or no I evidence. They
are reasonable candidates for a small correction pilot, subject to their
style limitations:

| Track | Tags/context | R/K/I |
| --- | --- | ---: |
| C'etait comme danser | French chanson/blues | 6/10/0 |
| Chacun son tour | metal vocal onsets | 9/6/1 |
| Stray - Akuma | hip-hop | 4/12/0 |
| Mr. & Mrs. Smith - Cold Black Oil | country/folk singer-songwriter | 7/9/0 |
| Monk Turner + Fascinoma - Where's my Horse? | country/pop singer-songwriter | 3/12/1 |
| Kate Orange | electronic vocal context | 7/9/0 |

The first four are the conservative core for an initial external-event arm.
`Monk Turner` and `Kate Orange` are useful secondary candidates but should not
be allowed to dominate the schedule.

### High-I restoration/safety examples

These are valuable for testing H50 over-removal, but they should not be used
as ordinary full-song aggressive pseudo-targets:

| Track | Tags/context | R/K/I |
| --- | --- | ---: |
| Mueve | Spanish pop/Latin | 1/3/12 |
| In (Feat Arrpa) | electronic | 1/4/11 |
| Bessemer | jazz vocal | 0/5/11 |
| Amor Brejeiro | Portuguese pop/electronic/experimental | 5/6/5 |
| I Gotta Tell You Something | electronic vocal effect | 5/4/7 |
| Crocodiles Boogie | blues vocal | 5/5/6 |

For these tracks, `I` is evidence that the current student has gone too far
relative to the practical reference. The tracks are useful as a restoration
arm or held-out safety test. It is still an open question whether their
arrangements resemble the product's common modern-pop workload.

### Stress-test rather than distribution data

`The Praties Go Small`, `Soon and Very Soon`, `We Hymn Thee`, and `We Three
Kings` are strongly R-marked, but they are folk/choral/classical textures.
`o my`, `Welcome Wizard`, `Earthquake`, and several ambient/electronic tracks
are similarly useful for difficult vocal-effect or texture stress tests. Their
marks are informative about failure modes, but they should not define the
main training distribution.

`Primavera Vuela`, `Wilde Eye`, and `Earthquake` are all-K neutral controls in
this review. They are useful for checking that a training change does not
alter already-acceptable material, not for pushing the model toward Inst 3.

## Important metric finding

The numeric event score cannot replace the human mark:

| Mark | Mean event score (dBFS) | Median (dBFS) |
| --- | ---: | ---: |
| `R` | -27.34 | -26.16 |
| `K` | -27.84 | -25.49 |
| `I` | -25.72 | -26.04 |

`I` events are, on average, numerically stronger than `R` events. A sampler
that simply takes the largest Inst 3 projection would therefore over-select
student over-removal cases. The manual mark must gate the training direction.

## Recommended next experiment

Do not train on all 28 songs. Use the following bounded matrix, starting from
the current best continuous H50 checkpoint (`H50-continuation+5`, step 1600):

### Arm C0: MUSDB18 control

- Same continuous 128-frame MUSDB18 event schedule and optimizer budget as the
  external-data arms.
- No MTG/FMA records.
- This isolates ordinary continued fine-tuning drift.

### Arm C1: reviewed-R correction

- External songs: `C'etait comme danser`, `Chacun son tour`, `Stray - Akuma`,
  and `Cold Black Oil`.
- Use only manually marked `R` events, balanced by song; start with four event
  centers per song and a 100 ms core with a small temporal guard.
- Use the Inst 3 residual target only inside the R core. Keep the H50 output as
  the anchor over the rest of the useful window.
- Keep external records at roughly 10-20% of each pass; MUSDB18 remains the
  dominant distribution.

### Arm C2: reviewed-R-plus-I correction

- Same schedule and songs as C1, plus the marked events from `Amor Brejeiro`
  as a deliberately small mixed Pop/Electronic probe (`R=5, I=5`), rather
  than all high-I songs. This song should be treated as an experimental
  auxiliary stratum, not as evidence of broad modern-pop coverage.
- On I cores, use the Inst 3 residual target to restore the content H50
  removed; on K and non-event regions, anchor to H50.
- Report R and I metrics separately. Do not average them into one “teacher
  improvement” number.

Use 128-frame continuous-context assembly, BatchNorm frozen, AdamW, learning
rate `1e-6`, batch size 4, and a bounded five-pass run. Match C0/C1/C2 for
updates, seed, and MUSDB18 records. Do not export 24-frame/TFLite artifacts in
this stage.

## Evaluation and go/no-go

Use three disjoint evidence groups:

1. The 12 private modern songs, with continuous rendering and blind listening,
   as the primary product-facing test.
2. The existing 20 MUSDB18 calibration/internal-test songs for instrument
   safety, raw miss, and mechanical artifacts.
3. MTG/FMA songs held out by whole song, especially `Mueve`, `In (Feat Arrpa)`,
   `Burdeos`, `Bessemer`, `Primavera Vuela`, and the four predesignated
   supplemental holdout songs.

Keep a candidate only if it improves marked R events or the private listening
set without worsening marked I events, K controls, low-vocal instrument
quality, or mechanical-noise metrics. A practical first threshold is about
`0.5 dB` improvement on the relevant local projection with no audible new
damage. If C1 improves only choral/experimental tracks and not the private
modern set, stop treating this pool as a useful training source.

## Data-distribution conclusion

The concern about realism is justified. This 28-track pool was selected for
voice tags and difficult local differences, not sampled from the distribution
of common contemporary pop releases. The human marks show that it is best
used as a small, label-guided correction and stress-test set. Before a larger
run, acquire a separate 30-50-track slice filtered toward Pop, Hip-Hop/R&B,
Dance/Latin, and conventional singer-songwriter production, with artist-level
disjoint holdout. Exclude Experimental/Noise/Electroacoustic tracks from the
main distribution and retain them as a named stress-test stratum.

The current MUSDB18 backbone remains the main training distribution until that
more representative slice is available.

## C1 follow-up

The subsequent C1 pilot used four reviewed-R songs at 9.09% of each pass and
produced approximately 0.5 dB improvement on unseen reviewed-R events. On the
12-song private set, however, the user could not reliably distinguish C1 from
the equal-budget MUSDB control, and the known conspicuous leak hotspots were
not fully removed. This confirms the distribution warning above: do not add
more passes on the same external subset. Acquire and review a more
representative modern-production slice before the next external-data arm.

## Artifacts

- Machine-readable analysis:
  `data/inst3-mtg-fma-event-listening/human-review-analysis.json`
- Review CSV:
  `data/inst3-mtg-fma-event-listening/human-review-template.csv`
- Event generation and analysis tools remain local research code; no audio or
  teacher-derived artifact is published.
