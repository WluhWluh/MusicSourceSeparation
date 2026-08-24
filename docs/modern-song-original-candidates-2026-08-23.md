# Modern Song Original Candidate Pool

Status: 72 raw original files downloaded locally on 2026-08-23 for style
review only. No audio decoding, probing, resampling, analysis, separation,
teacher inference, student inference, metric calculation, or snippet creation
was performed.

## Purpose

This pool is a replacement candidate search after the reviewed-R C1 pilot
produced measurable but not reliably audible improvement. The pool is intended
to find arrangements closer to common contemporary music before any further
teacher-directed training.

## Selection contract

- 72 tracks and 72 distinct artists;
- all artists from the prior 28-track MTG/FMA pool excluded;
- metadata duration between 150 and 330 seconds;
- explicit Creative Commons Attribution-family license;
- every NoDerivatives license excluded;
- `track_instrumental=1` excluded;
- explicit instrumental/karaoke/backing-track titles or tags excluded;
- Classical, Choir-like historic material, Ambient, Noise, Electroacoustic,
  Experimental, Avant-Garde, Field Recording, Sound Collage, Chiptune,
  Post-Rock, Metal, Hardcore, and similar stress-test tags excluded from the
  main pool;
- one track per artist;
- no train/validation role assigned before original-song style review.

Category quotas:

| Category | Count |
| --- | ---: |
| Pop / Synth Pop / Power-Pop | 14 |
| Hip-Hop / R&B / Rap | 14 |
| Dance / House / specific electronic styles | 12 |
| Latin / Brazilian / Spanish modern styles | 10 |
| Pop Rock / Indie-Rock with vocal evidence | 12 |
| Singer-songwriter | 10 |

## Source result

All 72 selected files come from FMA. The local MTG-Jamendo autotagging subset
contained only two new tracks that met modern genre, voice, artist, and length
requirements; both had a NoDerivatives license and were rejected. Tracks were
not admitted by weakening the license or vocal-evidence rules merely to create
a mixed source count.

Language metadata among the selected records:

- English: 34;
- Spanish: 6;
- Portuguese: 1;
- Japanese: 1;
- blank/unknown: 30.

The Japanese candidate is `Gunhead - Duck Rock Fever (Gunhead Remix)`, which
has FMA language `ja`, Electronic/Hip-Hop/House tags, and an explicit CC
Attribution-Noncommercial 2.1 Japan license. It still requires style and vocal
content review like every other candidate.

No acceptable Mandarin modern-pop candidate was downloaded. The two modern
Chinese FMA tracks found had NoDerivatives licenses. Broader Internet Archive
candidates had no explicit CC license in their item metadata. This remains an
explicit acquisition gap; language was not guessed from titles or artist
names.

## Local files

Raw originals, in listening order:

`data/modern-song-original-candidates/raw-originals/`

Style-review sheet:

`data/modern-song-original-candidates/original-style-review.csv`

Enter `Y` in `keep` for a suitable contemporary-production candidate, `N` for
reject, or leave the cell blank if undecided. The numeric file prefix matches
the CSV `order` column.

Source, attribution, per-track license, download URL, byte count, and SHA-256:

`data/modern-song-original-candidates/source-manifest.json`

## Verification

- manifest status: `completed`;
- records: 72;
- raw files: 72;
- total bytes: 626,454,332;
- unique artists: 72;
- byte size and SHA-256 match for every file;
- no downloaded file begins with an HTML response;
- no directory other than `raw-originals` exists below the pool root;
- all audio-processing flags in the manifest are `false`.

Metadata remains imperfect: some FMA records marked non-instrumental may still
be instrumentals or may not sound like modern commercial production. That is
why this stage ends at original-song listening instead of automatically
creating training targets.

After the user accepts the style subset, freeze an artist-disjoint split before
running H50/Inst 3 or generating event snippets. Do not use rejected tracks to
fill a quota.
