# MTG-Jamendo/FMA Short-Event Listening Set

Status: generated locally on 2026-08-23 for human review. The source audio,
teacher outputs, model outputs, and snippets are local non-commercial research
artifacts. No track has been added to training or validation by this step.

## Purpose

The set extends the earlier `musdb18-inst3-training-event-listening` review
pattern to every currently planned MTG-Jamendo/FMA track:

- 16-track expansion pool;
- 12-track supplemental pool (8 intended supplemental-train and 4 intended
  supplemental-holdout records).

The purpose is to let listening decide which local events are genuinely
undesired vocal, harmony, spoken, or vocal-effect residue. Numeric severity is
only a candidate-ranking aid; it is not a label and does not automatically
decide whether a song belongs in training.

## Output

The event root is:

`data/inst3-mtg-fma-event-listening/`

There are 28 songs and 448 events, exactly 16 per song. Each event has three
matching two-second FLAC files with the same event ID, plus one combined
four-point-three-second comparison file:

- `events/h50ContinuationPlus5/`: current student accompaniment;
- `events/inst3Instrumental/`: native Inst 3 accompaniment reference;
- `events/mixture/`: source mixture for orientation.
- `events/pairedListening/`: H50 (2 seconds), 300 ms silence, then Inst 3
  (2 seconds).

The main machine-readable report is:

`data/inst3-mtg-fma-event-listening/mtg-fma-event-listening-report.json`

The quick-review spreadsheet-compatible file is:

`data/inst3-mtg-fma-event-listening/human-review-template.csv`

The CSV starts with the empty `mark` column followed immediately by `eventId`.
It contains absolute paths for the matching H50, Inst 3, mixture, and paired
files, plus artist/track/license metadata and event time.

## Fast marking key

Enter one letter in the CSV `mark` column:

| Mark | Meaning |
| --- | --- |
| `R` | H50 retains vocal, harmony, spoken material, or vocal effect that should be removed |
| `K` | Current result is acceptable; keep as-is |
| `I` | H50 removed extra non-vocal content that should be retained relative to Inst 3 |
| blank | Not reviewed yet |

The same key is stored in:

`data/inst3-mtg-fma-event-listening/human-review-key.txt`

The comparison direction is important: the first half of each pair is the
current H50 accompaniment and the second half is Inst 3. Therefore `I` means
that H50 has removed more non-vocal content than Inst 3 and that this extra
removal sounds wrong. It does **not** mean that Inst 3 made the mistake.

Use `notes` for short comments such as `harmony`, `spoken`, `timbre`, or
`drum/transient`. The intended comparison is the paired file for the same
`eventId`, with H50 first and Inst 3 after the 300 ms gap; the separate mixture
is optional context.

## Candidate selection

The scanner evaluates full songs at 50 ms steps and scores centered 50, 100,
and 200 ms blocks. It combines the strongest local positive projection onto
the content removed by Inst 3 with a secondary raw miss term. To keep the set
useful for listening, it also:

- selects across eight temporal coverage bins per song;
- applies local non-maximum suppression before filling the remaining slots;
- excludes song edges and zero-padding;
- emits exactly two seconds per event.

This deliberately produces a broad candidate set rather than only the global
worst events. Some candidates will be acceptable instruments or effects; that
is expected and is precisely what the human pass is intended to identify.

## Roles before review

The manifest role is retained in every CSV/report row:

- expansion records: `candidate-pool`;
- supplemental records: `supplement-train` or `supplement-holdout`.

These are acquisition/planning roles, not final decisions. After marking:

- `R` events can form a confirmed aggressive-removal correction subset;
- `I` events are confirmed H50 over-removal corrections. They can be used as
  teacher-aligned restoration examples in a separate arm, or as safety
  validation; they must not be treated as evidence that Inst 3 itself made
  the error;
- `K` events are useful for checking that ordinary separation should remain
  unchanged;
- a song should enter the supplemental training pool only when its event mix
  and license terms are reviewed together;
- the four predesignated supplemental holdout songs remain untouched while
  training choices are made.

Do not use the private listening set or its labels to change this source-level
role split without recording the decision in a follow-up report.

## Provenance and restrictions

Every row preserves source ID, attribution metadata, license, license URL,
source URL, download URL, and raw SHA-256. Source manifests are:

- `data/inst3-mtg-fma-expansion/source-manifest.json`;
- `data/inst3-mtg-fma-supplement/source-manifest.json`.

Keep the audio and derived pseudo-targets local. Do not redistribute the
recordings, teacher caches, or derived weights, and do not upload them to
`bss-tflite`.

## Verification

- 28/28 songs processed;
- 16/16 events per song;
- 448/448 files in each of the three two-second event directories;
- 448/448 paired files, each 4.3 seconds with a 300 ms zero-valued gap;
- all event files finite, stereo, 44.1 kHz, and exactly 88,200 frames;
- all rows contain the required license and provenance fields.
