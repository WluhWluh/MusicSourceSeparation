# FMA S/R Leakage Survey Continuation

Date: 2026-08-26

This is a local, non-commercial research run. The official MUSDB18 final test
set was not used.

## Contract

- Source checkpoint: `data/modern-song-fma-sr-event-only-continuation/extensions/from-step-5120/runs/step-5648.pt`
- Source checkpoint SHA-256: `8992ec23afc000381211a8493cca83729ccd843ed1aa387b37875cc7cbb3e482`
- Model: residual-vocals output, continuous 128-frame overlap-save assembly
- Event target: `mixture - Inst 3 instrumental` on a 100 ms core plus a 25 ms linear guard
- Non-event target: source checkpoint residual output (`anchorBeta=0.25`)
- Optimizer: restored AdamW state, learning rate `1e-6`, weight decay `0`, batch size `4`
- BatchNorm running statistics frozen; gradient clipping `1.0`; seed `891`

## Pool And Sampling

Eight complete survey batches were included: `002`, `004`, `006`, `010`,
`011`, `012`, `013`, and `014`. Incomplete batches were skipped because they
lacked a completed survey report and review CSV. Filtering was per event:
`S=43`, `R=35`, `K` and `I` excluded. An `I` event in a song did not exclude
that song's marked `S/R` events.

The pool contains 30 songs and 78 event records. Each pass has 704 records and
176 updates. The repeated schedule represents each S event twice and each R
event once before filling the remainder with deterministic S draws. In pass 1
the schedule contains 505 S and 199 R records, giving an average per-event
frequency ratio of `2.066:1`; later passes use the same rule with shuffled
ordering.

## Results

The stopping signal is the all-pool 100 ms positive-projection p95. Lower is
better. Values are dBFS.

| Node | Step | p95 | Improvement from prior node |
| --- | ---: | ---: | ---: |
| source | 5648 | -15.495 | - |
| pass 1 | 5824 | -16.250 | 0.755 dB |
| pass 2 | 6000 | -16.890 | 0.640 dB |
| pass 3 | 6176 | -17.376 | 0.486 dB |
| pass 4 | 6352 | -17.755 | 0.379 dB |
| pass 5 | 6528 | -18.340 | 0.585 dB |
| pass 6 | 6704 | -19.099 | 0.758 dB |
| pass 7 | 6880 | -19.515 | 0.416 dB |
| pass 8 | 7056 | -19.753 | 0.238 dB |
| pass 9 | 7232 | -19.776 | 0.023 dB |
| pass 10 | 7408 | -19.783 | 0.007 dB |

Two consecutive improvements below `0.10 dB` occurred at passes 9 and 10,
so training stopped after pass 10. Relative to the source, the final all-pool
improvement is `4.288 dB` at 100 ms. The corresponding p95 improvements are
`2.333 dB` at 50 ms and `5.086 dB` at 200 ms.

At 100 ms, the S subset improved by `4.138 dB` and the R subset by `3.369 dB`
relative to the source. These are in-pool measurements and should not be
interpreted as held-out generalization or a direct listening guarantee.

## Artifacts

- Training report: `data/modern-song-fma-s-leakage-survey-continuation/reports/training-report.json`
- Pool manifest: `data/modern-song-fma-s-leakage-survey-continuation/pool-selection.json`
- Schedule: `data/modern-song-fma-s-leakage-survey-continuation/schedule.json`
- Final checkpoint: `data/modern-song-fma-s-leakage-survey-continuation/runs/step-7408.pt`
- All checkpoints: `step-5824.pt` through `step-7408.pt` in the same `runs` directory

The final checkpoint SHA-256 is recorded in the training report. The 30 cache
files are local generated artifacts and are not tracked by Git.

## GPU Note

The RTX 4060 Laptop GPU was used with CUDA (`torch 2.11.0+cu128`). The 8-update
smoke test reached 100% maximum utilization, 86% median, and 100% P90 after
the CUDA warm-up. Formal update batches stayed around 0.30 seconds with the
event pool resident on GPU. The combined formal monitor includes CPU-heavy
full-song evaluation passes, so its aggregate utilization is not a pure
training-utilization measure.
