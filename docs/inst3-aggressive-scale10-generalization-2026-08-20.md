# Inst 3 aggressive-target scale-10 generalization

Status: complete as a local, non-commercial research experiment, including a
user listening review. The run does not establish a releasable model.
MUSDB18-derived audio, teacher output, checkpoints, and listening files remain
under the ignored `data` tree.

Runner: [run_inst3_aggressive_scale10.py](../tools/run_inst3_aggressive_scale10.py)

Machine-readable evidence:

`data/musdb18-inst3-aggressive-scale10/reports/inst3-aggressive-scale10-report.json`

## Purpose

This is the follow-up to the user listening review that found the direct
`alpha=1.00` target audibly more aggressive than the initial compact student.
It tests whether that behavior generalizes beyond the two training songs used
in the earlier pilot.

The two students are identical except for the audio-domain target:

```text
alpha=0.00: MUSDB18 instrumental = drums + bass + other
alpha=1.00: UVR-MDX-NET-Inst_3 instrumental
```

The student remains the 128-frame `vocals_epoch=891.ckpt` neural core and
continues to predict the vocal residual. The target is converted to the
student STFT contract after the audio-domain mix. There is no anchor loss or
vocal-projection loss in this experiment.

## Frozen controls

- Manifest: `musdb18-inst3-oracle-split@1`.
- The first ten entries of the frozen 80-song `train` split were selected by
  rank and member name:
  `The So So Glos - Emergency`, `Dark Ride - Burning Bridges`,
  `Faces On Film - Waiting For Ga`, `Atlantis Bound - It Was My Fault For Waiting`,
  `BigTroubles - Phantom`, `Alexander Ross - Velvet Curtain`,
  `Traffic Experiment - Sirens`, `Flags - 54`,
  `Jokers, Jacks & Kings - Sea Of Leaves`, and `Giselle - Moss`.
- Each selected training song was decoded across its full duration. Thirty-two
  activity/temporal-coverage windows per song were used for training and
  sixteen disjoint windows per song were held out.
- The ten calibration and ten internal-test songs were kept song-disjoint from
  training and evaluated with sixteen coverage windows each.
- Official final-test songs were not extracted, loaded, or evaluated.
- Seed: `891`; learning rate: `1e-5`; AdamW with zero weight decay; gradient
  clipping at `1.0`; BatchNorm running statistics frozen; 8192 updates.
- Milestones: `0`, `2048`, and `8192`.
- Student execution: CUDA on an RTX 4060 Laptop GPU.
- Teacher: `uvr_mdxnet_inst_3@2`, with ONNX Runtime CUDA and CPU providers.
- Initial checkpoint SHA-256:
  `101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d`.
- Inst 3 ONNX SHA-256:
  `2b7834e2972158d8c9864e7376e3a7d084079c80a23f38dc31c4b0a4e901a1cb`.

The report records the full manifest, selection, runner hash, provider list,
checkpoint hashes, and listening-file hashes.

## Results

`teacherRemovalResidualSdrDb` measures similarity between the student's
removed residual and the Inst 3 removed residual. Higher is closer to the
aggressive teacher behavior; it is not a direct measurement of perceptual
vocal leakage. `instrumentalSdrDb` remains a conventional accompaniment
fidelity safety diagnostic. `accompanimentVocalProjectionDb` is a MUSDB18
vocal-stem proxy where a more negative value is better.

### Calibration plus internal-test, 20 unseen songs

| Variant | Aggressive-target SDR | Teacher-residual SDR | Instrumental SDR | Low-vocal instrumental SDR | Vocal projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| alpha 0.00, 8192 | 13.90 dB | 10.48 dB | 13.90 dB | 33.49 dB | -17.33 dB |
| alpha 1.00, 8192 | 16.70 dB | 10.69 dB | 13.94 dB | 34.07 dB | -16.87 dB |
| alpha 1.00 minus alpha 0.00 | +2.80 dB | +0.21 dB | +0.04 dB | +0.58 dB | +0.46 dB |

At the song level, the Inst 3 residual metric improved for 19 of 20 unseen
songs. Conventional instrumental SDR improved for 16 of 20. The vocal
projection proxy moved in the wrong direction in aggregate by `0.46 dB`; this
is a real safety signal and prevents treating the experiment as an unqualified
quality win.

### Ten-song training holdout

| Variant | Aggressive-target SDR | Teacher-residual SDR | Instrumental SDR | Low-vocal instrumental SDR | Vocal projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| alpha 0.00, 8192 | 14.25 dB | 9.58 dB | 14.25 dB | 36.20 dB | -15.41 dB |
| alpha 1.00, 8192 | 17.11 dB | 9.84 dB | 14.05 dB | 35.87 dB | -14.55 dB |
| alpha 1.00 minus alpha 0.00 | +2.86 dB | +0.26 dB | -0.20 dB | -0.33 dB | +0.86 dB |

The holdout result is directionally consistent for teacher similarity but
shows a larger conventional-fidelity tradeoff. This is why the 20-song
unseen result, not the training holdout, is the main generalization gate.

### Aggregate selected set

The ten training songs and twenty held-out songs together contain 30 songs.
At 8192 updates, `alpha=1.00` versus `alpha=0.00` changed the aggregate
metrics by:

- teacher-residual SDR: `+0.23 dB`;
- instrumental SDR: `-0.06 dB`;
- low-vocal instrumental SDR: `+0.33 dB`;
- vocal projection: `+0.62 dB` (worse by this proxy).

The per-song teacher-residual direction was favorable for 28 of 30 songs.
The main negative outlier was `Atlantis Bound - It Was My Fault For Waiting`
at `-0.29 dB`; `James May - On The Line` was approximately neutral at
`-0.01 dB`.

## Listening artifacts

The runner generated initial, teacher, and 2048/8192-step student outputs for
the same three external 30-second samples used in the prior listening review:

`data/musdb18-inst3-aggressive-scale10/listening-extra/`

- `yoru-ni-kakeru` from `data/samples/yoru_ni_kakeru.flac`;
- `coldplay-tove-lo` from `data/samples/fun.mp3`;
- `coast-town` from `data/samples/coast_town.mp3`.

Each file is a 15.0-45.0 second, 44.1 kHz stereo PCM16 FLAC render. The
previous user review found the three-song `alpha=1.00-step-2048` outputs had
audibly less residual vocal content than initial without an apparent
accompaniment-quality regression. The new 8192 files are the corresponding
longer-run comparison set.

## User listening review

The user reviewed the scale-10 `alpha=1.00-step-2048` and
`alpha=1.00-step-8192` outputs against the earlier four-song, complete
30-second pilot's `alpha=1.00-step-2048` output.

- The scale-10 outputs had a small but perceptible overall reduction in vocal
  residue volume relative to the earlier small pilot.
- The 8192-step and 2048-step scale-10 outputs both moved in this favorable
  average direction.
- In `yoru-ni-kakeru`, some isolated short syllables again leaked through as
  conspicuous events. Those local defects were slightly worse than in the
  earlier small-pilot output.
- No audible accompaniment-quality regression was noticed in the scale-10
  outputs.
- If selecting only by the current listening set, the user would probably
  prefer the earlier small-pilot result because it contained fewer abrupt
  vocal-leak artifacts, despite the scale-10 result having slightly lower
  overall residue.
- The listening set contains only three external songs, so this preference is
  evidence for the next experiment rather than a general model-ranking claim.

The listening result is consistent with the aggregate teacher-residual metric
improving while also showing why that average cannot select the product
candidate alone: a few brief, salient failures can outweigh a small reduction
in continuous background residue.

## Decision and next step

This run supports continuing the aggressive-removal student direction, but it
does not justify product integration or publication. The direct Inst 3 target
is learnable on unseen songs and stays close to standard instrumental fidelity
in aggregate. The listening review found no accompaniment regression, so the
next experiment should not yet add an anchor or ordinary-instrumental loss
that could weaken the desired aggressive behavior.

The first variable to isolate is update density. The small pilot used 16
training windows for 2048 updates, or about 128 visits per window. Scale-10
used 320 windows for 8192 updates, or about 25.6 visits per window. Wider data
coverage improved the average result, but rare short vocal events may simply
be under-trained.

The next controlled run should therefore:

1. keep the 128-frame model, `alpha=1.00`, ten-song training set, frozen
   BatchNorm policy, seed, and learning rate;
2. train from the same initial checkpoint to `8192/20480/40960` updates, which
   correspond to approximately `25.6/64/128` visits per training window;
3. save optimizer state so a long run can resume without resetting AdamW;
4. retain uniform full-song coverage, but separately report windows with
   short, high-energy teacher-removal bursts and high initial-to-teacher error;
5. render the same three external songs at every milestone and add exact
   listening markers for the conspicuous `yoru-ni-kakeru` syllables;
6. expand the private external listening set to roughly 8-12 diverse songs
   before choosing between the small-pilot and scale-10 trajectories.

If equal update density does not reduce the isolated leaks, the following run
should add hard-window sampling or a time-local residual loss. Candidate hard
windows should be selected using Inst 3 removed-residual activity and current
student-to-teacher error, not mixture RMS alone. Drums/bass preservation and
initial-output anchoring remain deferred safety options unless broader
listening reveals actual accompaniment damage.

Keep the 128-frame contract. Do not export a 24-frame candidate, integrate
into Booming SS, or upload derived weights to `bss-tflite` until the candidate
reduces both average residue and brief salient leak events across the expanded
listening set.

All derived weights and audio remain local non-commercial research artifacts.
