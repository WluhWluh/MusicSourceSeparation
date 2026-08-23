# Private Listening Language-Group Quality

Status: completed locally on 2026-08-23. This is a diagnostic comparison of
the twelve private songs against the native Inst 3 accompaniment output. It
does not use MUSDB18 ground-truth stems and is not a causal language study.

## Groups

The grouping is the user-confirmed grouping:

- Chinese: `coast-town`, `lugu-lake`, `unseen-sea` (3)
- Japanese: `odd-future`, `yoru-ni-kakeru` (2)
- English: `already-gone`, `chasing-the-wind`, `coldplay-tove-lo`,
  `i-see-fire`, `imagine`, `north`, `traveling-light` (7)

Machine-readable report:

`data/inst3-private-language-quality/private-language-quality-report.json`

Analysis tool:

`tools/analyze_inst3_private_language_quality.py`

## Reference and metrics

Inst 3 native instrumental output is the only reference, as requested.

- `teacherMatchSnrDb`: waveform energy SNR against Inst 3; higher is closer.
- `relativeErrorToRemovedDb`: candidate-minus-Inst 3 RMS relative to the
  signal Inst 3 removed; lower is closer.
- `coherentRetainedRelativeDb`: positive projection of the candidate error
  onto the Inst 3 removed signal; lower generally means less coherent residual
  content, but an excessively negative value can also indicate over-removal.
- Short-event metrics: the same quantities over non-overlapping 50/100/200 ms
  blocks, including p95, maximum, and active-block threshold rate.

The short-event positive-projection metrics are the most relevant diagnostics
for the reported abrupt vocal leaks. Whole-song averages can hide them.

## Current model: H50-continuation+5

### Whole-song group means

| Group | Inst 3 match SNR | Relative error / removed | Coherent retained / removed | Target error RMS |
| --- | ---: | ---: | ---: | ---: |
| English (7) | `13.667 dB` | `-8.657 dB` | `-22.622 dB` | `-28.249 dBFS` |
| Japanese (2) | `15.309 dB` | `-10.669 dB` | `-19.299 dB` | `-28.384 dBFS` |
| Chinese (3) | `10.422 dB` | `-2.896 dB` | `-19.880 dB` | `-23.186 dBFS` |

On whole-song waveform matching, the Chinese group is clearly worse than
English. Japanese is not worse on this aggregate metric; it is slightly
better in match SNR and relative error.

### 100 ms short-event means

| Group | Positive projection p95 | Positive projection max | Raw miss p95 | Raw miss max | > -30 dBFS rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| English (7) | `-29.993 dBFS` | `-20.922 dBFS` | `-22.162 dBFS` | `-16.379 dBFS` | `11.12%` |
| Japanese (2) | `-27.396 dBFS` | `-19.628 dBFS` | `-23.424 dBFS` | `-18.511 dBFS` | `10.26%` |
| Chinese (3) | `-28.018 dBFS` | `-18.190 dBFS` | `-16.081 dBFS` | `-11.066 dBFS` | `11.40%` |

For positive projection, higher dB is worse. Relative to English:

- Japanese p95 is `+2.60 dB` worse and max is `+1.29 dB` worse.
- Chinese p95 is `+1.97 dB` worse and max is `+2.73 dB` worse.

This short-event result agrees with the user's listening impression for both
non-English groups, even though Japanese does not look worse in whole-song
SNR. The normalized threshold rate is similar across groups, so the main
difference is event intensity rather than simply a larger number of active
blocks.

## Per-song ranking

### Short-leak ranking

Ranked by 100 ms positive-projection maximum, lower is better:

| Rank | Song | Group | Positive max |
| ---: | --- | --- | ---: |
| 1 | `imagine` | English | `-24.983 dBFS` |
| 2 | `i-see-fire` | English | `-24.907 dBFS` |
| 3 | `already-gone` | English | `-23.830 dBFS` |
| 4 | `yoru-ni-kakeru` | Japanese | `-20.361 dBFS` |
| 5 | `chasing-the-wind` | English | `-20.345 dBFS` |
| 6 | `lugu-lake` | Chinese | `-19.740 dBFS` |
| 7 | `odd-future` | Japanese | `-18.895 dBFS` |
| 8 | `north` | English | `-18.733 dBFS` |
| 9 | `coldplay-tove-lo` | English | `-18.230 dBFS` |
| 10 | `coast-town` | Chinese | `-17.489 dBFS` |
| 11 | `unseen-sea` | Chinese | `-17.340 dBFS` |
| 12 | `traveling-light` | English | `-15.429 dBFS` |

The 100 ms p95 ranking is similar but not identical. The English group has
the best concentration of high-ranked songs, while all three Chinese songs
are in the lower half. Japanese occupies the middle, with `yoru-ni-kakeru`
better than `odd-future`.

### Whole-song Inst 3 match ranking

Ranked by whole-song Inst 3 match SNR, higher is better:

`yoru-ni-kakeru`, `imagine`, `coldplay-tove-lo`, `north`,
`chasing-the-wind`, `odd-future`, `already-gone`, `i-see-fire`,
`unseen-sea`, `coast-town`, `traveling-light`, `lugu-lake`.

This ranking demonstrates why whole-song SNR alone is not enough: `yoru-ni-
kakeru` ranks first by average match while still containing subjectively
noticeable short leaks.

## Interpretation

The data supports a **language-associated short-event difficulty signal**,
especially for the Chinese group and, on coherent short-event peaks, for the
Japanese group. It does not establish that language itself is causal. The
groups are confounded with singer, genre, arrangement, vocal style, mastering,
encoding, and only 2 Japanese songs are available.

There are also strong song-level effects. `traveling-light` is the weakest
English song by the short-event maximum, while `imagine` is among the best.
Therefore the likely issue is a combination of phonetic/transient patterns and
training distribution, not a simple language switch.

The result justifies adding more Chinese and Japanese training material, but
the training examples should be selected by vocal event type and phonetic
structure rather than language label alone. In particular, add short
consonant/onset, syllabic, harmony, and vocal-effect examples, while retaining
instrument-like controls.
