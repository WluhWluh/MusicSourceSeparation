# Inst 3 S/R Continuation

Date: 2026-08-25

## Objective

Continue the current best `S86-masked-anchor@pass-5` model and separate the
effect of adding consensus R events from the effect of retaining a MUSDB
stability base.  All four arms use the same 128-frame continuous-context
contract, optimizer state, seed, learning rate, batch size, and 704 records per
pass.

## Frozen Contract

- Source checkpoint:
  `data/modern-song-s-only-stress/runs/S86-masked-anchor/step-2480.pt`
- Source SHA-256:
  `e0a03ac51d85fe43fc35e39417cc0991672da48ec380ae5a651193a164984485`
- AdamW state restored from the source; learning rate `1e-6`, weight decay 0
- Batch size `4`, gradient clip `1.0`, BatchNorm running statistics frozen
- Seed `891`, five passes, 176 updates per pass, milestones pass 1/3/5
- 128-frame continuous overlap-save assembly
- External event target: `V_T = mixture - Inst3 instrumental` on a 100 ms
  core plus 25 ms guard; outside the mask, anchor to the S86 source residual
- MUSDB records, where present, use the full Inst 3 residual target
- Official MUSDB final-test songs were not used

## Arms

| Arm | Records per pass | Event pool |
| --- | ---: | --- |
| `S86-continuation-control` | 640 MUSDB + 64 S | MUSDB base plus reviewed S |
| `SR-balanced` | 640 MUSDB + 32 S + 32 R | MUSDB base plus balanced S/R |
| `S86-event-only` | 704 external | all 86 reviewed S records |
| `SR-event-only` | 704 external | all 86 S + 46 consensus R records |

The event-only S arm exposes all 86 S records eight times per pass plus 16
balanced repeats.  The event-only S/R arm exposes all 132 records five times
per pass plus 44 balanced repeats.

## Runtime

- CUDA smoke: four arms, 8 updates each, all finite
- Formal training: 4 arms x 880 updates, completed successfully
- GPU: RTX 4060 Laptop; sampled utilization approximately 88%, roughly
  5.9--6.0 GB of 8.2 GB memory in use
- No non-finite state tensors were found in the 44 saved rolling/milestone
  checkpoints

## Continuous MUSDB Results

Values below are pass-5 deltas relative to the S86 source on 20 calibration and
internal-test songs. Negative projection deltas are lower residual vocal
projection; negative instrumental-SDR deltas are regressions.

| Arm | Instrumental SDR | Whole-song projection | Whole-song raw miss | 100 ms projection p95 | Songs improved at 100 ms p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S86-continuation-control` | +0.230 dB | +1.256 dB | -0.390 dB | +1.326 dB | 0/20 |
| `SR-balanced` | +0.253 dB | +1.364 dB | -0.425 dB | +1.425 dB | 0/20 |
| `S86-event-only` | -0.200 dB | -0.233 dB | +0.299 dB | -0.368 dB | 19/20 |
| `SR-event-only` | -0.141 dB | +0.025 dB | +0.210 dB | -0.200 dB | 16/20 |

The MUSDB-base arms appear to move toward a more conservative/general output
on the held-out songs, but they do not reduce the projection tail.  The
event-only arms reduce projection modestly while increasing raw-miss metrics;
this is evidence of a more aggressive change, not proof of more accurate
vocal removal.

## Private 12-Song Machine Metrics

Relative to the S86 source, pass-5 100 ms positive-projection p95 changed by:

- `S86-continuation-control`: `+0.952 dB`
- `SR-balanced`: `+1.095 dB`
- `S86-event-only`: `-0.730 dB`
- `SR-event-only`: `-0.464 dB`

The event-only S arm improved this metric on all 12 private songs; the event-
only S/R arm improved 11/12.  Its private target error increased by about
`0.157 dB`, while the S-only event arm increased it by about `0.267 dB`.
These values must be checked against the continuous listening files before
selecting a practical model.

## Artifacts

Run root:

`data/modern-song-s-r-continuation/`

Report:

`data/modern-song-s-r-continuation/reports/s-r-continuation-report.json`

Continuous private listening output:

`data/modern-song-s-r-continuation/listening-12/`

Each arm has checkpoints at `step-2656` (pass 1), `step-3008` (pass 3), and
`step-3360` (pass 5).  The report records SHA-256 values for every milestone.

## Decision Pending Listening

The event-only arms are the relevant candidates for the user's stated goal of
removing remaining S/R hotspots.  The MUSDB-base arms are useful controls, but
their held-out projection results do not justify promoting them on numerical
evidence alone.  The event-only S/R arm is less aggressive than S-only on the
private machine metrics and should be listened to beside both S-only and the
current S86 source before deciding whether R coverage helps or dilutes the
strong S-event effect.

## Subjective Listening Result

The listening review selected `S86-event-only@pass-5` as the best arm. It was
clearly better than `SR-event-only`, and it was also an audible improvement
over the previous round: residual vocal hotspots were reduced even at
locations that had been only mild leaks. The S-only arm sounded more natural
than the mixed S/R event-only arm while removing the reviewed S events more
effectively. Accompaniment damage was audible, but the listener judged it
acceptable for this stage and requested that vocal-removal progress be
prioritized before attempting accompaniment recovery.

`S86-event-only@pass-5` is therefore the current listening reference. The next
step is an S-only continuation pressure test to determine whether additional
training still lowers the 100 ms projection tail. It will preserve continuous
overlap-save rendering and will not generate new listening files until the
numerical curve identifies a small set of checkpoints worth comparing.
