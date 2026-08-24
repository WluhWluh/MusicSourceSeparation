# Combined Modern-S Continuation

Status: completed locally on 2026-08-24. This is a local non-commercial
research experiment. Source audio, derived caches, and teacher-derived
checkpoints remain local artifacts.

## Objective

Test whether the combined stable-`S` pilot still had useful optimization
headroom. Both final arms from the preceding experiment were continued for
the same additional budget:

- `C0-continuation`: equal-budget MUSDB control;
- `C1-continuation`: the combined previous/current stable-`S` event pool.

The continuation was deliberately bounded. It did not add songs, change the
S fraction, refresh event windows, or alter the target semantics.

## Frozen contract

- C0 source: `data/modern-song-s-combined-pilot/runs/C0-combined-control/step-2480.pt`;
- C1 source: `data/modern-song-s-combined-pilot/runs/C1-combined-stable-S/step-2480.pt`;
- both model and AdamW states restored;
- 640 MUSDB records plus 64 extra records per pass (`704` total);
- C1 extra pool: 24 previous S records plus 40 current consensus-S records;
- five additional passes, `880` updates per arm, batch size `4`;
- AdamW learning rate `1e-6`, weight decay `0`, gradient clipping `1.0`;
- frozen BatchNorm statistics, seed `891`;
- 128-frame continuous-context overlap-save assembly;
- 100 ms event core plus 25 ms guard, H50 anchor beta `0.25`;
- the official MUSDB final test was not used.

The regenerated C0/C1 schedule hashes exactly match the preceding pilot. The
target caches remain tied to the original H50 checkpoint, so the continuation
does not silently change the teacher target.

## Training and runtime

Both arms completed at global step `3360`, with checkpoints at passes `1`,
`3`, and `5`. The run used Python `3.12.10`, PyTorch `2.11.0+cu128`, and an
NVIDIA GeForce RTX 4060 Laptop GPU. GPU utilization was sampled at roughly
`89-100%` during training, with about `5.7 GB` of the `8.2 GB` GPU memory in
use. All recorded losses and gradient norms were finite.

Final checkpoints:

| Arm | File | SHA-256 |
| --- | --- | --- |
| C0-continuation | `runs/C0-continuation/step-3360.pt` | `5d4790c748cd860e3ff18f2cedddbdeab8a0534c3579c735feefaa94cceda432` |
| C1-continuation | `runs/C1-continuation/step-3360.pt` | `afa9d98763f80191e28048544dc7f4dc30d9acd32f4ff5798878d54f450e83de` |

## Private 12-song analysis

Continuous renders are under
`data/modern-song-s-combined-continuation/listening-12/`. Each of the three
variants has 24 FLAC files (instrumental and residual for 12 songs), all
stereo 44.1 kHz with matching full-song frame counts.

Relative to the preceding final checkpoints:

| Change after five-pass continuation | C0 | C1 |
| --- | ---: | ---: |
| coherent retained relative content | `+0.01 dB` | `-0.40 dB` |
| teacher-match SNR | `-0.04 dB` | `-0.11 dB` |
| 100 ms positive-projection p95 | `-0.01 dB` | `-0.36 dB` |
| 100 ms positive-projection max | `-0.00 dB` | `-0.23 dB` |
| 100 ms raw miss p95 | `+0.04 dB` | `+0.10 dB` |
| 100 ms raw miss max | `+0.01 dB` | `+0.06 dB` |

The C1 continuation therefore made the output more aggressive, but did not
reduce the residual-miss measure that represents the remaining content absent
from Inst 3. Its private teacher-match SNR also declined. This is consistent
with stronger removal rather than more precise correction of the known short
leaks.

## Continuous MUSDB analysis

Relative to the preceding final checkpoint, whole-song means changed by:

- C0: instrumental SDR `-0.013 dB`, positive vocal projection `+0.032 dB`,
  teacher residual miss `+0.018 dB`;
- C1: instrumental SDR `-0.063 dB`, positive vocal projection `+0.093 dB`,
  teacher residual miss `+0.067 dB`.

Relative to H50, C1 continuation still reduced 100 ms positive-projection p95
on all 20 songs, but raw miss p95 improved on only 9/20 songs. Its mean raw
miss-p95 delta versus H50 was `+0.320 dB`, with a worst-song regression of
`4.386 dB`. The continuation did not solve the maximum-hotspot problem.

## Decision

This is a negative continuation result for the current 128-frame S-focused
route. The combined S pool can make the student more aggressive, but another
five passes mainly increase removal and teacher mismatch while leaving the
most important residual-miss tail unresolved. Do not continue C1 again, raise
the S fraction, or add another S pool under this loss contract.

The next useful experiment should change the mechanism that can represent or
optimize a short leak while retaining a strict control: a short-window
retraining arm (24-frame continuous contract) or a localized audio-domain
event objective with a C1 continuation control. The choice should be made as
one isolated variable change; further S-only continuation has already shown
poor return.

## Reports and listening paths

- `data/modern-song-s-combined-continuation/reports/continuation-report.json`
- `data/modern-song-s-combined-continuation/reports/musdb-evaluation.json`
- `data/modern-song-s-combined-continuation/reports/private-inst3-analysis.json`
- `data/modern-song-s-combined-continuation/listening-12/C0-continuation/`
- `data/modern-song-s-combined-continuation/listening-12/C1-continuation/`
- `data/modern-song-s-combined-continuation/listening-12/Source-H50-continuation+5/`
