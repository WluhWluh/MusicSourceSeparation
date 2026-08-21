# Inst 3 V-R hard-event analysis, Stage 1

Status: complete.  The full 80-song run and its interpretation are recorded
in [the Stage 1 results](inst3-vr-hard-events-stage1-results-2026-08-21.md).
This stage performs no training and does not create an Android or releasable
model.  MUSDB18 audio, Inst 3 outputs, event arrays, and all generated data
remain under the ignored `data` tree.

## Purpose

The W-D direct-instrumental continuation introduced intermittent mechanical
artifacts and was not a useful product candidate.  This stage keeps the
original `vocals_epoch=891.ckpt` residual-vocal output and measures where it
fails to reproduce the more aggressive Inst 3 removal behavior.

For each complete song in the frozen 80-song MUSDB18 `train` role:

```text
teacherRemoved = mixtureGt - Inst3Instrumental
studentRemoved = mixtureGt - initialInstrumental
miss = teacherRemoved - studentRemoved
```

`miss` is the teacher-removed content that remains in the student's
accompaniment.  It is the diagnostic quantity for short, abrupt leaks.  The
student model is never updated in this stage.

## Frozen inputs

- Manifest: `musdb18-inst3-oracle-split@1`.
- Songs: all 80 entries with role `train`.
- Official calibration, internal-test, and final-test entries are not used.
- Student: the original `vocals_epoch=891.ckpt`, residual-vocals semantic.
- Teacher: `uvr_mdxnet_inst_3@2`, using the existing verified DSP contract.
- Mixture: `vocals + drums + bass + other`, not the encoded mixture stream.
- Student contract: 44.1 kHz, 2,048 FFT, 1,024 hop, 128 frames.

The runner processes one song at a time.  It stores compact event arrays and
per-song JSON summaries, then removes raw/decoded/teacher intermediates by
default.  This avoids retaining tens of gigabytes of derived audio.

## Event features

For 50 ms, 100 ms, and 200 ms blocks it records:

- Inst 3 removed-content RMS;
- student miss RMS;
- positive projection of `miss` onto the Inst 3 removed content;
- true MUSDB18 vocal RMS;
- cosine between Inst 3 removed content and the true vocal stem;
- deterministic hard-event score:
  `positiveProjectionRms * sqrt(max(teacherRemovedRms, activeFloor))`.

Blocks below `-60 dBFS` teacher-removed RMS are inactive.  The report also
maps the 100 ms events to the model's approximately 2.7 second useful student
windows, ranked for a later sampling experiment.

The three event labels are deliberately heuristic:

- `vocal-aligned`: high true-vocal activity and positive teacher/vocal cosine;
- `vocal-like-or-non-vocal`: strong teacher removal but low true-vocal activity;
- `mixed-or-uncertain`: everything else.

They are not annotations of ground-truth vocal identity.  In particular,
vocal-like harmony and effected vocal material can appear in the second or
third category, which is consistent with the intended aggressive-removal use
case.

## Reproduction

Full train analysis, with CUDA required:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe `
  tools\analyze_inst3_vr_hard_events.py `
  --device cuda --require-teacher-cuda --threads 8
```

Smoke analysis for the first two train entries:

```powershell
C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe `
  tools\analyze_inst3_vr_hard_events.py `
  --device cuda --require-teacher-cuda --max-songs 2
```

The report is written to
`data/musdb18-inst3-vr-hard-events/reports/inst3-vr-hard-events-report.json`.
Per-song event arrays are written under the same directory's `events/` tree.

## Decision after Stage 1

Do not choose training windows from the private listening songs.  First inspect
the all-train distribution by song, duration, event resolution, and heuristic
category.  The next experiment should preserve song diversity rather than
select only the globally loudest events.  A likely candidate is a per-song
quota with a 25% or 50% hard-event share, but the exact quota must wait for the
completed distribution report.
