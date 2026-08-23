# Reviewed-R MTG/FMA C1 Pilot

Status: completed locally on 2026-08-23. This is a non-commercial research
experiment. Source audio, teacher caches, checkpoints, and listening outputs
remain ignored local artifacts and must not be published.

## Question

Does a small amount of human-reviewed MTG/FMA `R` material improve the current
H50-continuation+5 model beyond an equal-budget MUSDB-only continuation, while
preserving accompaniment quality?

`R` means that the current H50 accompaniment still contains vocal, harmony,
spoken, or vocal-effect content that should be removed. `I` means the opposite
student error: H50 has removed extra non-vocal content relative to Inst 3.

## Frozen experiment

Both arms start from:

`data/musdb18-inst3-vr-continuation/runs/H50-continuation-plus5/step-1600.pt`

SHA-256:

`cc0ce960e2d305cc0f03e69fc58941988be720d69fdd192f6b746b692959a674`

Common settings:

- 128-frame continuous-context overlap-save;
- AdamW state restored from source;
- BatchNorm running statistics frozen;
- learning rate `1e-6`, batch size 4, gradient clip 1.0;
- five passes, 704 records/pass, 176 updates/pass, 880 updates total;
- checkpoints at pass 1, 3, and 5;
- seed 891;
- no MUSDB official test tracks.

Every pass includes the same 640 original MUSDB continuous records. The
remaining 64 records make the arms equal-budget:

- `C0-musdb-control`: 64 deterministic repeated MUSDB records;
- `C1-reviewed-R`: 16 reviewed external R events repeated four times;
  external share `64 / 704 = 9.09%`.

The four external songs and selected-event counts are:

- Fabrice Collette - C'etait comme danser: 4;
- Burnogson - Chacun son tour: 4;
- Stray - Akuma: 4;
- Mr. & Mrs. Smith - Cold Black Oil: 4.

The external loss uses the Inst 3 residual target on a 100 ms event core, a
25 ms soft guard on each side, and an H50 source-output anchor outside the
event (`anchorBeta=0.25`). It never applies an Inst 3 target to the whole
external song.

## Runtime correction

The first C0 run initially averaged approximately 44% sampled GPU utilization,
with peaks at 99% and repeated zero-utilization gaps. The cause was the new
runner's default eight-song LRU cache: random song scheduling repeatedly
decompressed NPZ files while the GPU waited.

The run was stopped after the exact update-600 rolling checkpoint, changed to
preload every cache used by the arm, and resumed without redoing updates.
Twenty subsequent samples measured mean GPU utilization `95.4%`, minimum
`87%`, and maximum `99%`. C1 used the corrected preload path from its first
update.

## Checkpoints

| Arm | Final checkpoint | SHA-256 |
| --- | --- | --- |
| C0 MUSDB control | `runs/C0-musdb-control/step-2480.pt` | `f907b4326a112f2e35f8d34328884b75eb3bf01897ce10c7408e4a06f51bef36` |
| C1 reviewed R | `runs/C1-reviewed-R/step-2480.pt` | `c3897cad5c6095d4c8a0dfc4feabaee5404d48994833fc0a634b3d7fde7837a9` |

The C1 external event loss decreased from a first-half mean of approximately
`0.0696` to a second-half mean of `0.0668`; no non-finite loss or gradient was
observed.

## Reviewed-event results

The most important comparison is C1 minus C0, because both arms have exactly
the same update budget and retain all 640 MUSDB records per pass.

### R events

| Scope | Events | 50 ms mean | 100 ms mean | 200 ms mean | Better at 100 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected training events | 16 | -1.126 dB | -1.096 dB | -1.014 dB | 15/16 |
| Same songs, unselected R | 10 | -0.684 dB | -0.656 dB | -0.619 dB | 7/10 |
| Song-disjoint/unseen R | 135 | -0.500 dB | -0.501 dB | -0.504 dB | 109/135 |
| All reviewed R | 161 | -0.573 dB | -0.569 dB | -0.562 dB | 131/161 |

Negative is better: lower error to Inst 3 than the equal-budget C0. The gain is
not limited to memorized windows. Unseen-song R events improve by about 0.5 dB
at all three resolutions, which is meaningful evidence of transfer.

At the two-second event scale, C1 improves all R events by approximately
`0.445 dB` relative to C0; selected windows improve by `0.709 dB`.

### K and I events

| Mark | 50 ms C1-C0 | 100 ms C1-C0 | 200 ms C1-C0 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| K | -0.204 dB | -0.200 dB | -0.192 dB | Small broad movement toward Inst 3; listening still required |
| I | -0.019 dB | -0.010 dB | +0.005 dB | Essentially unchanged; no restoration benefit |

C1 is deliberately R-only, so it does not fix H50 over-removal on I events.
At the two-second scale I error worsens about 0.066 dB versus C0. This is the
main reason not to promote C1 solely from metrics: its aggressive-removal
benefit must be checked against the marked I cases and private listening.

## MUSDB holdout

Relative to equal-budget C0, C1 changes the 20-song holdout mean as follows:

- instrumental SDR: `-0.052 dB`;
- positive vocal projection: `-0.379 dB` (less vocal-like residue);
- Inst 3 removed-content positive projection: `-1.173 dB`;
- teacher residual miss RMS: `+0.071 dB`.

For 50/100/200 ms blocks, C1 improves positive-projection p95 over C0 by
approximately `0.478/0.514/0.523 dB`, while raw-miss p95 worsens by
`0.094/0.135/0.097 dB`. This is the expected aggressive-removal tradeoff and
is small but not zero.

Join/seam behavior did not regress materially. Mean seam p95 ratios are
`1.795` for C0 and `1.784` for C1; the worst max ratio is `9.55` for C0 and
`9.33` for C1.

## Private 12-song set

Relative to C0, C1 changes the Inst 3-directed mean metrics:

- coherent retained content: `-0.862 dB`;
- 50/100/200 ms positive-projection p95: approximately
  `-0.490/-0.541/-0.535 dB`;
- 50/100/200 ms positive-projection max: approximately
  `-0.236/-0.476/-0.455 dB`;
- relative target error: `+0.049 dB`.

All 12 songs improve in coherent retained content and in 100 ms projection
p95. The largest 100 ms p95 gains occur on `coast-town`, `i-see-fire`,
`unseen-sea`, and `traveling-light`. The 100 ms max is slightly worse only on
`lugu-lake` (`+0.042 dB`); all other songs improve or are effectively flat.

The relative-error warning means the model moves more aggressively in some
content that is not coherently aligned with the teacher residual. Artificial
metrics do not decide whether this is audible damage.

All 36 private instrumental files are finite and frame-aligned. Across the 12
songs, clipping-count and first-difference RMS do not increase in C1 relative
to source/C0. The MUSDB seam metric also shows no new mechanical-boundary
signal.

## Listening set

The continuous full-song comparison is under:

`data/inst3-mtg-fma-c1/listening-12/`

Variants:

- `Source-H50-continuation+5`;
- `C0-musdb-control`;
- `C1-reviewed-R`.

Each directory contains 12 instrumental and 12 residual FLAC files. The
primary listening comparison is C1 versus C0; source is included to identify
ordinary five-pass continuation drift.

## Human listening result

The completed 12-song listening review found:

- `C1-reviewed-R` may have slightly less vocal residue than
  `C0-musdb-control` in some details;
- the difference is too subtle for reliable blind identification;
- none of the previously known conspicuous short vocal-leak hotspots is fully
  removed;
- no clear new accompaniment damage was reported in this comparison.

This result is consistent with the approximately 0.5 dB machine-level local
gain, but it does not meet the product-facing bar. A measurable movement
toward Inst 3 is not enough when the listener cannot reliably distinguish it
and the salient defects remain.

## Decision

C1 passes the machine-level learning test:

- selected R events improve strongly;
- unselected R events from the same songs improve;
- 109/135 unseen-song R events improve at 100 ms;
- MUSDB and private projection metrics move in the intended direction;
- no seam, non-finite, clipping-count, or first-difference warning is visible.

The human review does not pass the audible-benefit gate. C1 is not a replacement
for H50-continuation+5 and should not be extended for more passes on the same
four songs. More updates would increase exposure to a narrow, stylistically
biased pool while the present five-pass result has already failed to remove
the known private hotspots.

The next priority is a new, artist-disjoint collection that more closely
matches contemporary Pop, Hip-Hop/R&B, Dance/Latin, and conventional modern
singer-songwriter production. Run the same H50-versus-Inst 3 event review on
that collection before training. The current 28-song MTG/FMA pool remains
useful as:

- a named stress-test set for choir, folk, unusual effects, and experimental
  textures;
- a small auxiliary source for reviewed R/I events;
- evidence that the label-guided training mechanism works.

Only after a representative modern pool exists should C2 test R-plus-I
correction. This avoids spending another training round optimizing a data
distribution that has not produced reliably audible product gains.

## Artifacts

- Main report: `data/inst3-mtg-fma-c1/reports/c1-report.json`
- Exact event report:
  `data/inst3-mtg-fma-c1/reports/review-event-evaluation.json`
- Private Inst 3 analysis:
  `data/inst3-mtg-fma-c1/reports/private-inst3-analysis.json`
- Listening report: `data/inst3-mtg-fma-c1/listening-report.json`
