# MTG-Jamendo/FMA Supplemental Pilot

Status: selected, downloaded, and teacher-cached locally on 2026-08-23. This
is non-commercial research data. The audio, decoded files, Inst 3 outputs, and
all derived caches remain ignored local artifacts and are not publishable.

## Selection policy

The pilot requires:

- explicit `Attribution-NonCommercial` or `Attribution-NonCommercial-ShareAlike`
  audio license;
- no `NoDerivatives`/`NoDerivs` license;
- explicit non-instrumental/voice metadata where available;
- unique artist per selected record;
- duration suitable for full-song Inst 3 inference;
- a mix of vocal genres and languages rather than one artist or one style.

The metadata sources were downloaded from the official repositories and their
SHA-256 values are recorded in:

`data/inst3-mtg-fma-supplement/source-manifest.json`

## Selected records

| Role | Source | Artist - track | License | Intended value |
| --- | --- | --- | --- | --- |
| train | MTG-Jamendo | Fabrice Collette - C'etait comme danser | CC BY-NC-SA | French chanson/blues phrasing |
| train | MTG-Jamendo | Burnogson - Chacun son tour | CC BY-NC-SA | dense rock/metal vocal onset |
| train | MTG-Jamendo | Viejo Den - Primavera Vuela | CC BY-NC-SA | Spanish/Latin featured vocal |
| train | MTG-Jamendo | Franck Camu - Crocodiles Boogie | CC BY-NC-SA | short blues phrasing |
| train | FMA | AWOL - Street Music | BY-NC-SA 3.0 | English hip-hop consonants/rap onsets |
| train | FMA | Dona Onete - Amor Brejeiro | BY-NC-SA 3.0 | Portuguese pop/electronic vocal |
| train | FMA | Lido Pimienta - Mueve | BY-NC 2.0 Chile | Spanish/Latin pop vocal |
| train | FMA | Monk Turner + Fascinoma - Where's my Horse? | BY-NC | English singer-songwriter/pop |
| holdout | MTG-Jamendo | unfa - I Gotta Tell You Something | CC BY-NC | electronic vocal-effect control |
| holdout | MTG-Jamendo | DeguMoth Studios - We Three Kings | CC BY-NC-SA | ambient/choral texture control |
| holdout | FMA | /'angstalt/ - wie weich | BY-NC-SA 3.0 | German rock vocal |
| holdout | FMA | D. Charles Speer - Aman Yiala Yiala | BY-NC-SA 3.0 | international/folk vocal |

All 12 direct audio URLs returned `audio/mpeg` successfully before download.
The exact URLs, source metadata, licenses, raw file hashes, and decoded PCM
hashes are in the manifest.

The completed local preparation produced 12 raw MP3 files and 12 finite Inst 3
NPZ caches. The teacher residual RMS values range from approximately `-18.36`
to `-31.07 dBFS`, so every selected record contains non-trivial removed
content; the four holdout records remain unused by the training schedule.

## Teacher preparation

The local `uvr_mdxnet_inst_3@2` CUDA teacher completed all 12 full songs:

- 8 supplemental training songs;
- 4 supplemental holdout songs;
- 44.1 kHz stereo decoded PCM;
- finite instrumental and residual outputs for every record.

Teacher cache location:

`data/inst3-mtg-fma-supplement/teacher/`

The manifest records teacher providers, timing, checkpoint contract, output
hashes, and residual/instrumental RMS. The shortest holdout is approximately
53 seconds; it is retained as a control rather than used to fill the training
quota.

## Current status and next use

The files have **not** yet been mixed into the TFC-TDF training schedule. The
next controlled experiment should keep MUSDB18 as the dominant source and add
these records as a small supplemental arm. Recommended first comparison:

1. existing continuous H50 continuation schedule;
2. the same schedule plus supplemental real-song windows;
3. identical optimizer budget and song/artist-disjoint supplemental holdout;
4. Inst 3-directed 50/100/200 ms event metrics and the fixed private listening
   set.

The next integration should consume only records with role
`supplement-train`; `supplement-holdout` is reserved for a separate report and
must not be used to choose training hyperparameters.

The FMA and MTG selections contain no Chinese or Japanese track in this first
pilot because their public metadata did not provide a reliable enough language
filter for those languages. Language-specific material should be added through
M4Singer/JVS-MuSiC or a separately cleared source rather than guessing from
track titles.

## Provenance and restrictions

- MTG-Jamendo: [repository](https://github.com/MTG/mtg-jamendo-dataset),
  per-track `audio_licenses.txt`, direct Jamendo MP3 endpoint.
- FMA: [repository](https://github.com/mdeff/fma), per-track license fields in
  `raw_tracks.csv`, direct FMA storage endpoint.
- Every selected track requires attribution from its source record.
- The current artifact set is local research-only. Do not upload source audio,
  teacher caches, or derived weights to `bss-tflite`.
