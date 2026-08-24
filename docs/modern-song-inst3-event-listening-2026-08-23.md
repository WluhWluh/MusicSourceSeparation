# Modern Song Inst 3 Event Listening Set

Status: completed locally on 2026-08-23. This preparation uses all and only
the 57 songs marked `Y` in the original-style review. No song has yet been
assigned to training or validation.

## Frozen source selection

Input style review:

`data/modern-song-original-candidates/original-style-review.csv`

The review contains 72 valid decisions:

- `Y`: 57;
- `N`: 15;
- blank or invalid: 0.

Selected category counts:

| Category | Songs |
| --- | ---: |
| Pop Rock | 12 |
| Pop / Synth Pop | 11 |
| Hip-Hop / R&B | 11 |
| Singer-songwriter | 10 |
| Latin modern | 9 |
| Dance / Electronic | 4 |

The frozen selected-source manifest is:

`data/modern-song-inst3-event-listening/selected-source-manifest.json`

It records each original MP3 hash, style decision, source, license, and the
pending-event-review role.

## Model references

- Student: `H50-continuation+5`, step 1600;
- student assembly: 128-frame continuous-context overlap-save;
- teacher: native `uvr_mdxnet_inst_3@2` CUDA full-song rendering;
- sample rate: 44.1 kHz stereo.

The full-song student and teacher annotation caches are PCM16 FLAC files. They
exist to make this listening workflow resumable and auditable. If a song later
enters training, native float teacher windows must be regenerated from the
original source and teacher contract. The annotation FLAC must not silently
become a training target.

## Candidate selection

The selection contract matches the previous MTG/FMA human-review set:

- 16 events per song;
- 912 events total;
- scan hop: 50 ms;
- local measurements: 50, 100, and 200 ms;
- eight temporal coverage bins per song;
- local non-maximum suppression before filling remaining positions;
- complete two-second snippets only, with no song-edge zero padding.

This deliberately supplies broad time coverage rather than allowing every
event from a song to come from one chorus or one unusual sound.

## Paired listening contract

Every paired file is exactly 4.6 seconds:

```text
H50-continuation+5 accompaniment: 2.0 s
silence:                            0.3 s
Inst 3 accompaniment:              2.0 s
trailing silence:                  0.3 s
```

At 44.1 kHz this is exactly 202,860 frames:

```text
88,200 + 13,230 + 88,200 + 13,230
```

Paired files:

`data/modern-song-inst3-event-listening/events/pairedListening/`

File names use a continuous four-digit serial followed by `eventId`, for
example:

`0001-modern-003-fma-my-bubba-mi-nothing-much-01.flac`

The separate two-second H50, Inst 3, and mixture references remain available
under their corresponding `events/` directories.

## Human review

Review sheet:

`data/modern-song-inst3-event-listening/human-review-template.csv`

The first columns are `mark,eventId,serial,sourceOrder`. Enter:

- `R`: H50 still retains vocal, harmony, spoken, or vocal-effect content that
  should be removed;
- `S`: a particularly conspicuous subset of `R`: residual loudness and/or
  clear consonants make the lyric especially easy to hear;
- `K`: current H50 accompaniment is acceptable;
- `I`: H50 removed additional non-vocal content that Inst 3 retains and that
  should remain;
- blank: not yet reviewed.

The renderer preserves existing `mark` and `notes` values by `eventId` if the
report or paired files are regenerated.

## Verification

- completed songs: 57/57;
- events per song: 16/16;
- event rows: 912;
- paired files: 912;
- separate H50 files: 912;
- separate Inst 3 files: 912;
- mixture files: 912;
- full-song H50 caches: 57 FLAC + 57 JSON;
- full-song Inst 3 caches: 57 FLAC + 57 JSON;
- every paired file is finite, stereo, 44.1 kHz, and 202,860 frames;
- both 300 ms regions are sample-zero silence;
- H50 and Inst 3 segments match their separate two-second files sample for
  sample within PCM16 readback tolerance;
- serial values are continuous from 1 through 912;
- all source and license fields are retained.

Machine-readable report:

`data/modern-song-inst3-event-listening/event-listening-report.json`

All source and derived audio remains local. Do not assign train/validation
roles until this event review is complete and a song-level, artist-disjoint
split is frozen.
