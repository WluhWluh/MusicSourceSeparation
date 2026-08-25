# Modern S Pilot Comparison

This note compares the original `S0/S1` pilot with the expanded-pool
`C0/C1` pilot and its bounded continuation. All values below come from the
continuous-context reports; negative projection/error deltas mean movement
toward lower residual energy, while higher Inst 3 match SNR is better.

## Experimental contracts

| Experiment | External pool | Train/holdout evidence | Budget |
| --- | --- | --- | --- |
| S0/S1 | 32 S events from 16 songs | 12 modern-song holdout songs plus 20 MUSDB songs | 5 passes, 704 records/pass |
| C0/C1 | 24 prior S records plus 40 current consensus-S records | 20 MUSDB songs plus the same 12 private songs | 5 passes, 704 records/pass |
| C0/C1 continuation | Same C0/C1 schedules and caches | Same 20 MUSDB songs plus the same 12 private songs | 5 additional passes from step 2480 |

All three stages use the same source H50 checkpoint, 128-frame continuous
assembly, batch size 4, learning rate `1e-6`, frozen BatchNorm, 100 ms core,
25 ms guard, and 9.09% external budget. The C0 control reproduces S0's
MUSDB-only extra slots; its small numerical differences are floating-point or
run-order noise.

## Modern-song holdout: original S0 versus S1

Only the original S pilot has a frozen modern-song holdout report. Direct S1
minus S0 target-error deltas are:

| Mark | 50 ms mean | 100 ms mean | 200 ms mean | 100 ms improved |
| --- | ---: | ---: | ---: | ---: |
| S | -0.990 dB | -0.897 dB | -0.747 dB | 56/61 |
| R | -0.628 dB | -0.597 dB | -0.564 dB | 56/71 |
| K | -0.323 dB | -0.278 dB | -0.210 dB | 33/53 |
| I | -0.460 dB | -0.306 dB | -0.145 dB | 4/7 |

The S 100 ms worst event still worsened by `+0.892 dB`. The K/I movement is
evidence that the original S pilot learned a generally more aggressive
direction, not only a selective correction of S events.

The expanded C1 pilot did not regenerate this exact modern-song holdout, so
there is no defensible C1-versus-S1 modern-event generalization claim. Its
external pool is larger and drawn from a different set of reviewed songs.

## 20-song MUSDB evaluation

The following table is treatment minus control for the whole-song means.

| Metric | S1 - S0 | C1 - C0 | C1-cont - C0-cont |
| --- | ---: | ---: | ---: |
| instrumental SDR | -0.162 dB | -0.155 dB | -0.206 dB |
| positive vocal projection | -0.441 dB | -0.593 dB | -0.532 dB |
| teacher-removed positive projection | -1.551 dB | -2.042 dB | -1.831 dB |
| teacher residual-miss RMS | +0.180 dB | +0.174 dB | +0.223 dB |

For the 100 ms window, positive-projection p95 changed by `-0.768 dB` for
S1/S0, `-0.786 dB` for C1/C0, and `-0.869 dB` for C1-cont/C0-cont. Raw miss
p95 changed by `+0.247 dB`, `+0.262 dB`, and `+0.304 dB`, respectively.

Thus the larger pool produced a stronger aggressive direction than S1 on the
MUSDB aggregate, but did not improve the metric that measures content still
missing relative to Inst 3. The continuation increased this mismatch further.

## Private 12-song Inst 3 analysis

Treatment minus control deltas are:

| Metric | S1 - S0 | C1 - C0 | C1-cont - C0-cont |
| --- | ---: | ---: | ---: |
| coherent retained relative content | -1.847 dB | -1.927 dB | -2.337 dB |
| teacher match SNR | -0.227 dB | -0.194 dB | -0.264 dB |
| target error RMS | +0.227 dB | +0.194 dB | +0.264 dB |
| 100 ms positive projection p95 | -1.238 dB | -1.173 dB | -1.515 dB |
| 100 ms positive projection max | -0.934 dB | -0.788 dB | -1.016 dB |
| 100 ms raw miss p95 | +0.211 dB | +0.185 dB | +0.244 dB |
| 100 ms raw miss max | +0.176 dB | +0.160 dB | +0.214 dB |

Absolute private-set teacher-match SNR is `13.071 dB` for S0, `12.844 dB`
for S1, `13.071 dB` for C0, `12.877 dB` for C1, `13.036 dB` for
C0-continuation, and `12.772 dB` for C1-continuation. C1 is only `0.033 dB`
above S1 on this aggregate, and the two treatments use different external
pools; that small difference is not evidence that the expanded pool is
intrinsically better. The robust conclusion is that both treatments trade
higher aggressive removal for higher target error. C1 has a marginally
smaller private raw-miss regression than S1, but the difference is not large
enough to establish a quality improvement.

The C1 continuation adds another `0.403 dB` of coherent removal relative to
C1 and another `0.105 dB` of target error, without reducing the worst-leak
tail. This is optimization drift toward aggressiveness, not evidence of
better short-event reconstruction.

## Conclusion

1. The original S1 pilot already demonstrated the strongest clear modern
   holdout signal: S events improved on average and in most individual cases.
2. The expanded C1 pool increased the aggressive shift on the 20-song MUSDB
   evaluation, while preserving nearly the same private-set tradeoff.
3. C1 continuation made the tradeoff worse without solving raw miss hotspots.
4. The current evidence does not justify more S-only passes or a higher S
   fraction. The next meaningful comparison should change temporal resolution
   or the local audio-domain objective, with a matched continuation control.

Source reports:

- `data/modern-song-s-pilot/reports/s-report.json`
- `data/modern-song-s-pilot/reports/modern-holdout-evaluation.json`
- `data/modern-song-s-pilot/reports/private-inst3-analysis.json`
- `data/modern-song-s-combined-pilot/reports/combined-s-report.json`
- `data/modern-song-s-combined-pilot/reports/musdb-evaluation.json`
- `data/modern-song-s-combined-pilot/reports/private-inst3-analysis.json`
- `data/modern-song-s-combined-continuation/reports/continuation-report.json`
