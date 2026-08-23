# Inst 3 Teacher Training Data Research

Date: 2026-08-23

This is a research and licensing triage note for local, non-commercial
experiments. It is not legal advice. A dataset page license does not
necessarily clear the underlying composition, recording, performer, or
third-party source rights. No new dataset audio was downloaded in this scan.

## Short answer

There are useful additional sources, but no single dataset solves the problem.
The best practical combination is:

1. **MTG-Jamendo** or filtered **FMA** for diverse real full-song mixtures.
2. **SingStyle111**, **M4Singer**, and **JVS-MuSiC** for Chinese/Japanese
   vocal phonetics, styles, and short events.
3. **VocalSet** and a small choir corpus for harmony, breath, and vocal-effect
   controls.
4. **MIR-1K** only after obtaining explicit permission, because it is highly
   relevant but has no clear license statement in the current Zenodo record.

Inst 3 can generate pseudo-instrumental targets for full-song sources. Vocal-
only datasets must first be mixed with separately licensed accompaniment; they
should be augmentation data, not the sole training distribution.

## Candidate triage

| Dataset | Coverage and useful content | License/evidence found | Inst 3 use | Decision |
| --- | --- | --- | --- | --- |
| [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) | 55k+ full tracks, 195 tags, broad genres/instruments/moods | Audio has per-track Creative Commons licenses; metadata is CC BY-NC-SA 4.0; repository states non-commercial academic use only | Run Inst 3 on selected full tracks; use tags to balance genre/instrument/vocal content | **Best first real-mix source** |
| [Free Music Archive / FMA](https://github.com/mdeff/fma) | 106,574 full tracks, 16k artists, 161 genres | Audio license is selected by each artist; repository explicitly says it does not hold audio copyright and data is for research | Filter per-track license metadata, then run Inst 3 | **Best second real-mix source** |
| [SingStyle111](https://doi.org/10.5281/zenodo.10265401) | 111 songs, 8 professional singers, 12.8 h; English/Chinese/Italian; dry mono vocal phrases with phoneme alignment | Zenodo record: CC BY 4.0 | Mix vocal phrases with licensed instrumentals, then teacher-label; emphasize Chinese phonetic/onset events | **High-value augmentation** |
| [VocalSet](https://doi.org/10.5281/zenodo.1492453) | 10.1 h, 20 singers, 17 vocal techniques, a cappella | Zenodo record: CC BY 4.0 | Mix technique clips with licensed accompaniment; target breath, consonant, vibrato, and vocal-effect cases | **High-value technique augmentation** |
| [JVS-MuSiC](https://arxiv.org/abs/2001.07044) | 100 Japanese singers; common Japanese song plus a unique song per singer | Paper states audio/MPD may be used for academic research, non-commercial research including research in commercial organizations, and personal use; matrices have CC BY-SA 4.0 | Mix/teacher-label Japanese vocal material; hold out singers, not random clips | **Best Japanese candidate, verify project terms before ingestion** |
| [M4Singer](https://github.com/M4Singer/M4Singer) | 20 professional singers, 700 Mandarin pop songs, SATB styles and detailed alignment | `dataset_license.md`: CC BY-NC-SA 4.0, with an explicit responsibility/indemnity agreement | Mix vocal-only material with licensed instrumentals; use Chinese syllable/onset events | **Very relevant, local-only pending license review** |
| [Opencpop](https://github.com/wenet-e2e/opencpop) | 100 Mandarin songs, about 5.2 h, studio vocal recordings and phoneme boundaries | Official license page says non-commercial CC BY-NC-ND 4.0; page metadata contains an inconsistent CC BY description | Vocal augmentation only; ND makes derived training use/weights legally ambiguous | **Do not use without written clarification** |
| [MIR-1K](https://doi.org/10.5281/zenodo.3532216) | 1,000 Chinese-pop clips, 133 min; mixture and accompaniment/vocal channels; unvoiced labels | Current Zenodo record has no explicit license; official page offers a download but does not provide a clear permissive license | Directly useful for Chinese short consonants and true accompaniment targets | **Request author permission first** |
| [NUS-48E](https://zenodo.org/records/19595152) | New NUS-48E release; approximately 1 GB archive; singing/spoken-lyrics corpus lineage | Zenodo record: CC BY 4.0 | Candidate for multilingual sung/spoken onset augmentation after inspecting contents | **Audit release contents before use** |
| [ESMUC Choir Dataset](https://doi.org/10.5281/zenodo.5848990) | 12 individual singers, SATB sections, room mics, multitrack Western choral music; about 31 min | Zenodo record: CC BY 4.0 | Harmony/choral leakage and room/reverb controls | **Small supplement** |
| [Choral Singing Dataset](https://doi.org/10.5281/zenodo.2649950) | 16 individual singers, three a cappella pieces, Latin/Spanish/Catalan | Zenodo record: CC BY 4.0 | Harmony and unison controls; synthetic mix with licensed accompaniment | **Small supplement** |
| [Japanese Singing Voice Dataset](https://huggingface.co/datasets/tts-dataset/japanese-singing-voice) | Dataset card claims about 1,000 h of Japanese vocal audio | Card declares CC BY-NC 4.0, but provenance and underlying recording rights need independent audit | Potentially large Japanese augmentation source | **Do not ingest yet; provenance risk is high** |

## Most useful findings

### MTG-Jamendo

This is the strongest immediately actionable source for real mixtures. It has
full tracks and rich genre/instrument tags, unlike vocal-only SVS corpora. The
repository documents per-track audio licenses in `audio_licenses.txt` and
states that the dataset is solely for non-commercial research/academic use.

For this project, filter out tracks with unknown or `ND` terms for the first
pilot. Retain explicit `CC BY`, `CC BY-SA`, `CC BY-NC`, and `CC BY-NC-SA` only
after recording each track's exact license. Do not redistribute downloaded
audio or teacher caches.

The full high-quality collection is about 508 GB for the raw 30-second-capable
subset, so the first pilot should download a metadata-selected slice rather
than the whole dataset.

### FMA

FMA is larger and more genre-diverse, but its license is not one dataset-wide
license. The repository explicitly says the audio is distributed under the
license chosen by each artist. Use the per-track license field as a hard
filter, preserve attribution, and exclude `ND` and unknown tracks from model
training until their treatment is reviewed.

FMA is useful as a source of accompaniment diversity and English/international
real mixes. It is less attractive than MTG-Jamendo for language-balanced
selection because language metadata is weaker.

### Language-targeted vocal corpora

`M4Singer`, `Opencpop`, `SingStyle111`, and `JVS-MuSiC` are not source-
separation datasets. That is acceptable for teacher training, but their use
should be structured as synthetic mixture augmentation:

```text
licensed accompaniment + vocal-only phrase
    -> loudness/pan/room/codec augmentation
    -> Inst 3 pseudo-instrumental target
```

Keep the real full-song mixture ratio high. Synthetic mixtures can teach
phonetic onsets and vocal effects, but they do not reproduce real mastering,
bleed, masking, or arrangement statistics.

`JVS-MuSiC` is especially attractive for Japanese diversity because it covers
100 singers. `M4Singer` is especially attractive for Mandarin because it
covers 700 songs and 20 singers, but its CC BY-NC-SA agreement is more
restrictive than the CC BY sources.

## Recommended acquisition order

### Tier 1: start locally

1. Build a license manifest from MTG-Jamendo metadata and select 200-500 full
   tracks across genre, instrument, vocal tags, and artist. Keep a fixed
   song/artist-disjoint holdout.
2. Add a smaller FMA slice, perhaps 100-300 tracks, using only explicit
   acceptable per-track licenses.
3. Add SingStyle111 and VocalSet as vocal-technique augmentation, mixed only
   with Tier 1 instrumentals.
4. Add JVS-MuSiC after checking the current project download terms and storing
   the exact terms alongside the manifest.

### Tier 2: permission or terms confirmation

1. Request explicit research/training permission for MIR-1K. It is likely the
   most valuable Chinese control because it includes real accompaniment and
   vocal channels plus unvoiced labels.
2. Confirm whether M4Singer's CC BY-NC-SA agreement permits training a
   teacher-distilled separation model and what attribution/share-alike notice
   is expected for weights.
3. Obtain written clarification for Opencpop because the official page says
   CC BY-NC-ND 4.0 and its metadata is inconsistent.
4. Audit the source provenance and release-specific terms of the large
   Japanese Singing Voice Dataset before downloading any part of it.

### Tier 3: do not use as training input yet

- YouTube/DALI or other scraped corpora with unclear redistribution/training
  rights.
- Any Hugging Face mirror whose card license is not traceable to the original
  dataset and recording rights.
- Opencpop until the ND restriction is clarified.
- MIR-1K until the author or official terms provide permission.

## Proposed first experiment

Do not replace MUSDB18. Add a small supplemental corpus and compare against
the current continuous H50 continuation:

1. 80% of training windows from the existing MUSDB18 schedule.
2. 10% from filtered MTG-Jamendo/FMA real full tracks.
3. 10% from language-targeted synthetic mixtures, initially balanced between
   Mandarin and Japanese vocal phrases.
4. Run Inst 3 once per source mixture and cache only local pseudo-targets.
5. Keep all songs, artists, and singers disjoint between training and
   validation; do not use the 12 private listening songs for selection.
6. Compare a uniform supplemental sampler with a phonetic-event sampler.
7. Evaluate separately on English, Mandarin, and Japanese held-out singers or
   songs, using 50/100/200 ms positive projection maxima and the fixed private
   listening set.

The first budget should be roughly 20-40 hours of supplemental audio, not a
full MTG-Jamendo/FMA download. A successful result would show improvement on
held-out Chinese/Japanese short-event maxima without increasing the raw-miss
maximum or damaging clearly non-vocal accompaniment.

## Provenance requirements

For every source file, retain locally:

- dataset/version and download URL;
- exact per-track license and license URL;
- artist/singer/source attribution;
- source SHA-256 and decoded PCM hash;
- whether the source is real mixture, vocal-only, speech, or synthetic mix;
- teacher checkpoint hash and runner/DSP contract;
- split assignment by song, artist, and singer.

Derived audio, teacher caches, and checkpoints from NC/SA/ND sources should
remain local. Do not upload them to `bss-tflite` or use them in a commercial
release without a separate rights review.
