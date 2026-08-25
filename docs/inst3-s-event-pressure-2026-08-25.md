# Inst 3 S-Event Pressure Test

Date: 2026-08-25

## Objective

Continue the listening-best `S86-event-only@pass-5` model with the identical
all-S event-only contract until the continuous 20-song 100 ms positive vocal
projection p95 stops showing a meaningful numerical improvement. The test
prioritizes vocal removal; accompaniment recovery is deliberately excluded.

## Frozen Contract

- Source: `data/modern-song-s-r-continuation/runs/S86-event-only/step-3360.pt`
- Optimizer state restored; AdamW, learning rate `1e-6`, weight decay `0`
- Batch size `4`, gradient clip `1.0`, BatchNorm running statistics frozen
- Seed `891`, 128-frame continuous overlap-save assembly
- 86 reviewed S records, repeated eight times plus 16 balanced repeats: 704
  records per pass and 176 updates per pass
- 100 ms Inst 3 target core plus 25 ms guard; S86 output anchor outside event
- No MUSDB training records, no R records, no private listening render, and
  no official final-test songs
- The reconstructed first-five-pass schedule matched the committed schedule
  hash `832e2d77f71407cd258fcd68050ab1aafc7a97841c2242b6a3ebc917585c63b7`

## Training

One additional five-pass block was completed. Every pass was checkpointed:

| Pressure pass | Global step | Checkpoint |
| ---: | ---: | --- |
| 6 | 3536 | `data/modern-song-s-event-pressure/runs/S86-event-only/step-3536.pt` |
| 7 | 3712 | `data/modern-song-s-event-pressure/runs/S86-event-only/step-3712.pt` |
| 8 | 3888 | `data/modern-song-s-event-pressure/runs/S86-event-only/step-3888.pt` |
| 9 | 4064 | `data/modern-song-s-event-pressure/runs/S86-event-only/step-4064.pt` |
| 10 | 4240 | `data/modern-song-s-event-pressure/runs/S86-event-only/step-4240.pt` |

GPU monitoring during training showed approximately 90% utilization and
6.5/8.2 GB used on the RTX 4060 Laptop GPU.

## Continuous Evaluation

The metric is the mean across the same 20 calibration/internal-test songs.
Negative change in the p95 column means lower residual vocal projection.

| Node | 100 ms projection p95 (dBFS) | Delta vs pass 5 | 100 ms projection max (dBFS) | Instrumental SDR (dB) |
| --- | ---: | ---: | ---: | ---: |
| S86-event-only pass 5 | -41.5385 | 0.0000 | -29.9460 | 13.5624 |
| Pressure pass 6 | -41.5240 | +0.0145 | -29.9523 | 13.5477 |
| Pressure pass 8 | -41.5511 | -0.0126 | -29.9709 | 13.5158 |
| Pressure pass 10 | -41.5212 | +0.0173 | -29.9852 | 13.4967 |

The p95 changes are all below `0.1 dB` and alternate direction. Pass 8 is the
numerical p95 minimum, but its advantage over pass 5 is only `0.0126 dB`.
Pass 10 has the lowest maximum projection, while its p95 and instrumental SDR
are worse than pass 8. This is a plateau, not evidence that longer S-only
training is still producing a reliable generalization gain. The pressure run
therefore stops at pass 10.

## Nodes To Render Later

Keep the original `S86-event-only@pass-5` as the listening reference. If a
small follow-up listening comparison is useful, render only pressure pass 8
(best p95) and pressure pass 10 (best maximum projection/latest endpoint).
Pass 6 is an optional near-source control; pass 7 and pass 9 have not been
continuous-evaluated and are not recommended for the first listening batch.

The pressure phase intentionally produced no audio files. The complete
machine-readable curve and checkpoint hashes are in
`data/modern-song-s-event-pressure/reports/pressure-report.json`.
