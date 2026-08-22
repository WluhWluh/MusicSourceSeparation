# Inst 3 V-R Soft Guard Comparison

Status: completed locally on 2026-08-22. This is a local, non-commercial
MUSDB18 research experiment. It does not modify product code or publish a
runtime model.

## Purpose

The preceding `3e-6` local-anchor run improved the aggregate 100 ms positive
projection p95 by only about `0.043 dB`, and the short-leak hotspot itself did
not improve. This comparison tests whether the selected 100 ms hard-event
mask was too narrow in time for the model to learn the surrounding context.

Both arms use the same fixed H50 schedule. The only changed variable between
the two local-anchor runs is the soft guard width around the existing event
mask:

- `H50-continuation`: the Inst 3 residual target on every frame. This is the
  training control and does not use the event mask.
- `H50-local-anchor`: the Inst 3 target on the event region and the frozen H50
  output on the complement of the event weights.
- `guard=25`: linear decay from weight `1.0` to `0.0` over 25 ms on each side.
- `guard=50`: linear decay from weight `1.0` to `0.0` over 50 ms on each side.

The original FFT-support-overlap event region remains at weight `1.0` in both
runs. Candidate windows, event selection, schedule, target audio, and model
initialization are unchanged.

## Frozen Contract

- 80 MUSDB18 training songs and 20 song-disjoint calibration/internal-test
  songs;
- 8 selected windows per song per pass, using the existing V-R-H50 selection;
- 5 passes and 800 optimizer updates per arm;
- batch size 4, AdamW, learning rate `3e-6`, weight decay 0;
- `anchorBeta=1.0`, seed 891, frozen BatchNorm running statistics;
- gradient clip 1.0, milestones 0/1/2/5 passes;
- 44.1 kHz, 128-frame TFC-TDF, residual-vocals output semantic;
- target semantic `mixtureGt - Inst3Instrumental`;
- no official final-test songs, 24-frame export, QNN, or product integration.

Commands:

```powershell
.\.tmp\vr-hard-env\Scripts\python.exe tools\run_inst3_vr_local_anchor.py `
  --device cuda --threads 8 --passes 5 --milestones 0,1,2,5 `
  --learning-rate 3e-6 --guard-ms 25 `
  --output-root data\musdb18-inst3-vr-local-anchor-guard25ms --no-resume

.\.tmp\vr-hard-env\Scripts\python.exe tools\run_inst3_vr_local_anchor.py `
  --device cuda --threads 8 --passes 5 --milestones 0,1,2,5 `
  --learning-rate 3e-6 --guard-ms 50 `
  --output-root data\musdb18-inst3-vr-local-anchor-guard50ms --no-resume
```

## Mask Coverage

The mask statistics below are the mean weighted event-frame fraction across
the 80 training songs. They show that the guard implementation changed the
loss coverage as intended without changing the number or location of
candidate windows.

| guard | mean weighted fraction | minimum | maximum |
| ---: | ---: | ---: | ---: |
| 25 ms | 0.295428 | 0.166578 | 0.519816 |
| 50 ms | 0.318290 | 0.179768 | 0.547802 |

## Holdout Results

The values in the delta columns are relative to the H50 pass-50 checkpoint.
Negative is better for projection and miss metrics. The `100 ms` columns are
the primary comparison because the hard-event manifest is defined at 100 ms.

| run | arm | d instrumental SDR | d low-vocal SDR | d accompaniment vocal projection | d 100 ms miss p95 | d 100 ms positive projection p95 | d 100 ms positive projection max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 ms | continuation | +0.000187 dB | +0.001566 dB | +0.029569 dB | +0.024979 dB | +0.045943 dB | +0.000133 dB |
| 25 ms | local-anchor | +0.000736 dB | -0.012583 dB | -0.028282 dB | +0.034442 dB | -0.043182 dB | -0.000269 dB |
| 50 ms | continuation | +0.000182 dB | +0.001564 dB | +0.029630 dB | +0.024594 dB | +0.046510 dB | +0.000136 dB |
| 50 ms | local-anchor | +0.000787 dB | -0.011520 dB | -0.024617 dB | +0.035124 dB | -0.039784 dB | -0.000258 dB |

For the local-anchor arm, the corresponding positive-projection p95 deltas
were:

| guard | 50 ms window | 100 ms window | 200 ms window |
| ---: | ---: | ---: | ---: |
| 25 ms | -0.031058 dB | -0.043182 dB | -0.040198 dB |
| 50 ms | -0.029048 dB | -0.039784 dB | -0.039982 dB |

The local-anchor arm improved the 100 ms positive-projection p95 on 20/20
evaluation songs for both guard widths. The 100 ms miss p95 improved on 12/20
songs for both widths. This is the same per-song pattern as the no-guard
`3e-6` run, so the guard did not add a new generalization effect.

The largest 100 ms positive-projection improvement was only about `0.00027
dB`. The worst individual hotspot therefore remains effectively unchanged.
The 25 ms guard is marginally better than 50 ms on the primary projection
metric, but the difference is only `0.0034 dB` and is not meaningful.

## Preservation and Artifact Checks

| run | arm | derivative excess p95 | derivative excess max | clipped samples | non-finite windows |
| --- | --- | ---: | ---: | ---: | ---: |
| 25 ms | continuation | 0.831077 dB | 3.264605 dB | 0 | 0 |
| 25 ms | local-anchor | 0.863384 dB | 3.271782 dB | 0 | 0 |
| 50 ms | continuation | 0.831145 dB | 3.264648 dB | 0 | 0 |
| 50 ms | local-anchor | 0.864769 dB | 3.272506 dB | 0 | 0 |

The local-anchor coefficient preservation diagnostic was `-58.77 dB` for the
25 ms guard and `-59.43 dB` for the 50 ms guard on the non-event weighted
region. This is a relative spectral diagnostic, not an audio dBFS score. No
guard width introduced a mechanical-artifact or numerical failure.

## Listening Outputs

Both runs produced 48 finite PCM16 FLAC files: four model variants across the
12-song private listening set. The render reports verified 44.1 kHz, stereo,
and exact source frame counts for all outputs.

- 25 ms listening root:
  `C:\Users\User\Documents\MusicSourceSeparation\data\musdb18-inst3-vr-local-anchor-guard25ms\listening-12\`
- 50 ms listening root:
  `C:\Users\User\Documents\MusicSourceSeparation\data\musdb18-inst3-vr-local-anchor-guard50ms\listening-12\`

The model variants are `H50-pass-50`, `lambda-0.50-pass-50`,
`H50-continuation@pass-5`, and `H50-local-anchor@pass-5`.

The objective changes are far below the scale expected to produce a reliable
blind-listening difference. Neither guard width should be treated as an
audibly improved replacement for the current V-R-H50 listening baseline
without a separate human listening result.

## Decision

The soft guard hypothesis is not supported as a useful next optimization:

1. Widening the context from 25 to 50 ms does not improve the result.
2. Both widths reproduce the earlier approximately `0.04 dB` projection-p95
   change rather than solving the individual short-leak hotspots.
3. Miss p95 at 100 ms moves slightly worse in aggregate, and low-vocal SDR
   decreases by about `0.012 dB`.
4. The continuation control moves in the opposite direction, confirming that
   the small local-anchor effect is not a generic benefit of another 5-pass
   continuation.

Stop this soft-guard/local-anchor fine-tune family. Keep V-R-H50 as the
current best listening baseline and do not spend more runs on 75 ms guards or
blind learning-rate increases. The next useful research direction is a
dedicated short-event objective or a higher-capacity/short-context model that
can represent the remaining event, evaluated against the fixed V-R-H50
baseline and the same private listening set.

## Provenance

Implementation and tests were committed in:

- `09a68e8 experiment: add soft guard local-anchor comparison`
- `296e1e2 experiment: add Inst3 event alignment and local anchor studies`

| artifact | SHA-256 |
| --- | --- |
| runner | `805F28EAE4D63BCFB32CB4DEF4F94FC28C9657C9196D1AC1A1F8261C16FED2D1` |
| unit test | `C70B35603EF8F4FC7F88D4924FFBE5E7AC70C01BB23347E6A4AD1E1EDBBF664E` |
| H50 checkpoint | `0E49F154B8F66F9827BAA197C287DB7A34C6902C932D72332EA34CDD33870705` |
| 25 ms local-anchor checkpoint | `9CBB9AFDB6B890273E306083564A64A663B456026AAB0DA23C49B572B67E6543` |
| 25 ms continuation checkpoint | `63CEDF61F82B7C1D74571F53A8AC7FFE7D94207D2DE5F9ECBA138B7A958A4D92` |
| 25 ms evaluation report | `697CD823EA8F05BC42953A8ED0F388E9E0F32A3C13294C4BE8EBDAD34CABA59E` |
| 25 ms listening report | `9373B71C623136BD718EFBD63296709931D5AB203DC0CEDC31279BA7679289B7` |
| 50 ms local-anchor checkpoint | `C8DA8D59451DB09C39600D96D45845983624D36B7818DC9B290A8CC65D96183E` |
| 50 ms continuation checkpoint | `D3FA9093A9EA0684D15F1523CCC284CF047007A6419D502A0F8B546D3DD19182` |
| 50 ms evaluation report | `A7BEA5EB0564755FB007705D67B5B663960650E4324676A1F88D50249EB699E6` |
| 50 ms listening report | `4ABF4B9834F8970E6926721B1B17657974F35EC309F15FFCBCBADEA2F1FD01E5` |

All checkpoints, teacher-derived targets, and audio remain local under the
MUSDB18 non-commercial research restrictions.
