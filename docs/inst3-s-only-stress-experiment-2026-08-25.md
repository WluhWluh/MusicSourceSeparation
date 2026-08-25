# Inst 3 S-Only Stress Experiment

Date: 2026-08-25

## Purpose

Measure the limit of an all-S reviewed-event schedule when the H50 stabilizer
is either retained locally or removed entirely. Both arms start from the same
`H50-continuation+5` checkpoint and restore its AdamW state.

## Frozen Contract

- Source: `data/musdb18-inst3-vr-continuation/runs/H50-continuation-plus5/step-1600.pt`
- Source SHA-256: `cc0ce960e2d305cc0f03e69fc58941988be720d69fdd192f6b746b692959a674`
- Reviewed S pool: 86 records from 38 songs (32 previous + 54 current)
- 128-frame continuous overlap-save assembly
- 704 records per pass: 86 records repeated 8 times plus 16 deterministic balanced repeats
- 5 passes, 880 updates, batch 4, AdamW, learning rate `1e-6`, weight decay 0
- BatchNorm running statistics frozen, seed 891, gradient clip 1.0
- No MUSDB training records, R/K/I records, or official final-test songs used

Arms:

- `S86-masked-anchor`: 100 ms core plus 25 ms guard toward Inst 3; H50 anchor
  outside the event with beta `0.25`.
- `S86-max-aggressive`: full useful-window Inst 3 target with no H50 anchor.

## Results Relative To This Run's H50 Source

The table uses the `H50-continuation+5` source in this experiment, rather than
the older generic `H50-continuous` evaluator baseline. Values are dB deltas;
negative positive-projection and raw-miss values are improvements, while a
negative instrumental SDR is a regression.

| Arm | Pass | Instrumental SDR | 100 ms positive projection | 100 ms raw miss p95 |
| --- | ---: | ---: | ---: | ---: |
| masked-anchor | 1 | -0.372 | -1.925 | +0.519 |
| masked-anchor | 3 | -0.744 | -3.239 | +1.126 |
| masked-anchor | 5 | -0.880 | -3.528 | +1.312 |
| max-aggressive | 1 | -0.114 | -0.809 | +0.120 |
| max-aggressive | 3 | -0.236 | -1.178 | +0.244 |
| max-aggressive | 5 | -0.295 | -1.212 | +0.300 |

On the 12-song private set at pass 5, masked-anchor reduced the 100 ms
positive-projection p95 by about 4.40 dB but increased Inst 3 target error by
about 1.29 dB. Max-aggressive reduced the p95 by about 1.72 dB and increased
target error by about 0.10 dB. These figures indicate stronger removal, not
proof of accurate vocal removal; raw miss and target error remain the safety
checks.

## Artifacts

The complete local run is under:

`data/modern-song-s-only-stress/`

Important files:

- `reports/stress-report.json`
- `reports/musdb-source-relative.json`
- `reports/musdb-evaluation.json`
- `reports/private-inst3-analysis.json`
- `listening-12/`
- `event-listening/event-listening-report.json`
- `event-listening/clips/` (144 files; 2 s H50, 300 ms silence, 2 s candidate,
  300 ms silence, 2 s Inst 3)
- `runs/S86-masked-anchor/`
- `runs/S86-max-aggressive/`

## Runtime Verification

The two 8-update CUDA smoke arms passed. During formal training on the RTX
4060 Laptop GPU, sampled `nvidia-smi` utilization was predominantly 90--100%
with roughly 6.2--7.1 GB used memory. Lower utilization during evaluation was
caused by per-song audio decoding and file I/O, not the training loop.

The repository test suite completed with 140/140 tests passing.

## Decision

This is a limit/diagnostic experiment, not an automatic promotion candidate.
The masked-anchor arm is too damaging on the MUSDB safety metrics despite its
large projection reduction. The max-aggressive arm is the safer of the two,
but its raw-miss increase and modest private-set target degradation require
human listening before any reuse. Continuous-rendered listening files are the
authoritative audio comparison for this run.
