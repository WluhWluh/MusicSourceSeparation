# S86 FMA Hotspot Review

## Scope

This annotation set covers every song that survived the two frozen local FMA
style reviews:

- batch 1: 57 `Y` songs;
- batch 2: 101 `Y` songs;
- total: 158 unique songs and 790 events, exactly 5 events per song;
- the 32 songs marked `N` were excluded and were not rendered.

The previous `train`, `calibration`, `holdout`, and pending roles are retained
as metadata only. No song or event is automatically assigned to a training
pool by this preparation step.

## Rendering Contract

The student is `S86-event-only@pass-5`, checkpoint:

`C:\Users\User\Documents\MusicSourceSeparation\data\modern-song-s-r-continuation\runs\S86-event-only\step-3360.pt`

Checkpoint SHA-256:

`117a48816e14ac4a4957698ec1ae5e7e365ec23efabd83e316a04ae5085e6523`

Student output uses the residual-vocals representation and is converted to an
accompaniment with `mixture - residual`. Rendering uses the continuous
128-frame overlap-save contract (`5 + 5` context hops, 117-hop output stride).
Isolated zero-filled windows are not used. Inst 3 is the native full-song
cached instrumental output, and each source cache was checked against the
source SHA-256 before use.

Each paired listening file is exactly:

1. H50 accompaniment, 2.0 seconds;
2. silence, 0.3 seconds;
3. Inst 3 instrumental, 2.0 seconds;
4. trailing silence, 0.3 seconds.

The total is 4.6 seconds / 202,860 frames at 44.1 kHz. The first and third
audio files in each event directory are also retained separately for scripted
checks and alternate listening workflows.

## Selected-Hotspot Metrics

The metrics below describe the five selected 50 ms-hop candidates per song.
They are teacher-directed residual proxies, not ordinary instrumental SDR and
not human labels. `positiveProjection` measures the part of the H50
accompaniment that projects onto content removed by Inst 3; a less-positive
(more negative) value is better.

| block | projection p50 | projection p90 | projection p95 | projection max | miss p95 | miss max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 ms | -26.18 dBFS | -19.39 dBFS | -17.39 dBFS | -10.30 dBFS | -11.56 dBFS | -6.43 dBFS |
| 100 ms | -28.77 dBFS | -20.53 dBFS | -18.45 dBFS | -10.57 dBFS | -12.37 dBFS | -6.90 dBFS |
| 200 ms | -32.69 dBFS | -22.02 dBFS | -20.04 dBFS | -11.55 dBFS | -13.48 dBFS | -7.43 dBFS |

At 100 ms, 68 of 790 selected events are at or above `-20 dBFS` positive
projection and 235 are at or above `-25 dBFS`. These are screening counts for
manual listening, not an automatic decision that an event is vocal or should
be used for training. The strongest numerical candidates are intentionally
spread across each song's temporal bins; repeated events in one song can still
reflect a song-specific failure mode and should be considered together.

## Manual Review Files

The primary table is:

`C:\Users\User\Documents\MusicSourceSeparation\data\modern-song-fma-all-s86-event-pass5\human-review-template.csv`

The paired files are in:

`C:\Users\User\Documents\MusicSourceSeparation\data\modern-song-fma-all-s86-event-pass5\events\pairedListening`

The CSV begins with `mark,eventId`. Marks are intentionally blank. Use:

- `S`: especially conspicuous residual vocal; this is a subset of `R`;
- `R`: further vocal, harmony, spoken, or vocal-effect removal is desired;
- `K`: the current result is acceptable;
- `I`: H50 removed extra non-vocal content relative to Inst 3. This does not
  mean that Inst 3 removed non-vocal content;
- blank: not reviewed.

The machine-readable report is:

`C:\Users\User\Documents\MusicSourceSeparation\data\modern-song-fma-all-s86-event-pass5\event-listening-report.json`

## Validation

- selected songs: 158;
- unique source IDs: 158;
- rejected `N` songs included: 0;
- candidates per song: 5 for all 158 songs;
- event files per type (`h50ContinuationPlus5`, `inst3Instrumental`,
  `mixture`, `pairedListening`): 790 each;
- full-song H50 caches and metadata: 158 each;
- paired-file frame count: 202,860 for all 790 files;
- both silence regions: exact zero for all 790 paired files;
- initial manual marks: blank for all 790 rows.
