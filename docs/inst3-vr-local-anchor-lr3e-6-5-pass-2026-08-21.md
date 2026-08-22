# Inst 3 V-R Local Anchor at 3e-6

Status: completed locally on 2026-08-21. This is a non-commercial MUSDB18
research experiment. It does not modify product code or publishable model
artifacts.

## Purpose

The preceding local-anchor run used a learning rate of `1e-6` and produced an
inaudible change. This run keeps the same event-local target and H50 schedule,
but increases the learning rate to `3e-6` to test whether the small effect was
only an update-size limitation.

Two arms were trained independently from the same H50 pass-50 state:

- `H50-continuation`: Inst 3 target on every frame;
- `H50-local-anchor`: Inst 3 target on event frames and frozen H50 output on
  non-event frames.

## Contract

- 80 training songs, 20 song-disjoint evaluation songs;
- existing V-R-H50 selection and schedule, 8 windows per song per pass;
- 5 passes and 800 updates per arm;
- batch size 4, AdamW, learning rate `3e-6`, weight decay 0;
- `anchorBeta=1.0`, seed 891, frozen BatchNorm running statistics;
- gradient clip 1.0, milestones 0/1/2/5 passes;
- same 44.1 kHz TFC-TDF spectral and Inst 3 residual-vocals contract;
- no centered-window, 24-frame, QNN, or product integration changes.

Command:

```powershell
.\.tmp\vr-hard-env\Scripts\python.exe tools\run_inst3_vr_local_anchor.py `
  --device cuda --threads 8 --passes 5 --milestones 0,1,2,5 `
  --learning-rate 3e-6 `
  --output-root data\musdb18-inst3-vr-local-anchor-lr3e-6 --no-resume
```

Both arms completed on CUDA without non-finite loss, CUDA failure, or failed
checkpoint. The final local-anchor loss was `0.135485`, consisting of
`0.135415` event loss and `0.0000699` anchor loss.

## Holdout Results

All deltas below are relative to H50 pass-50; negative is better for the
projection and miss metrics.

| checkpoint | instrumental SDR | low-vocal SDR | accompaniment vocal projection | 100 ms miss p95 | 100 ms positive projection p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| continuation pass-5 | -0.0004 dB | +0.0016 dB | +0.0294 dB | +0.0240 dB | +0.0464 dB |
| local-anchor pass-1 | -0.0007 dB | -0.0064 dB | -0.0063 dB | +0.0233 dB | -0.0108 dB |
| local-anchor pass-2 | -0.0004 dB | -0.0086 dB | -0.0131 dB | +0.0283 dB | -0.0163 dB |
| local-anchor pass-5 | +0.0006 dB | -0.0133 dB | -0.0293 dB | +0.0353 dB | -0.0431 dB |

At local-anchor pass-5, positive-projection p95 improved on all 20 evaluation
songs. The mean per-song improvements were:

| local window | positive projection p95 | miss p95 |
| ---: | ---: | ---: |
| 50 ms | -0.0405 dB | +0.0023 dB |
| 100 ms | -0.0512 dB | -0.0020 dB |
| 200 ms | -0.0401 dB | +0.0099 dB |

The aggregate miss p95 and low-vocal SDR moved in the wrong direction. The
worst 100 ms positive-projection maximum improved only `0.00026 dB`; the
individual short-leak hotspot remains unsolved. The improvement is larger
than the prior `1e-6` run (`0.0105 dB` aggregate 100 ms projection p95), but
still below the `0.1 dB` signal threshold and far below the `0.5 dB` practical
gate.

## Preservation and Artifact Checks

The spectral coefficient preservation diagnostic for local-anchor pass-5 was
approximately `-59.0` on the non-event coefficient RMS scale, compared with
`-46.9` for unrestricted continuation. This is a relative spectral diagnostic,
not audio dBFS. Increasing the learning rate therefore caused more ordinary-
region drift, while the anchor still constrained it substantially.

| checkpoint | derivative excess p95 | derivative excess max | clipped samples | non-finite windows |
| --- | ---: | ---: | ---: | ---: |
| H50 pass-50 | 0.8698 dB | 3.2815 dB | 0 | 0 |
| continuation pass-5 | 0.8310 dB | 3.2646 dB | 0 | 0 |
| local-anchor pass-5 | 0.8633 dB | 3.2714 dB | 0 | 0 |

The 12-song render had 48 finite outputs. Local-anchor output clipping was
effectively unchanged from H50 (26,568 versus 26,567 counted samples in the
decoded diagnostic), but this count is dominated by the existing renderer and
is not a perceptual score.

## Listening Outputs

The four-variant, 12-song listening set is under:

`C:/Users/User/Documents/MusicSourceSeparation/data/musdb18-inst3-vr-local-anchor-lr3e-6/listening-12/`

The actual directory is:

`C:\Users\User\Documents\MusicSourceSeparation\data\musdb18-vr-local-anchor-lr3e-6\listening-12\`

Variants:

- `H50-pass-50`;
- `lambda-0.50-pass-50`;
- `H50-continuation@pass-5`;
- `H50-local-anchor@pass-5`.

The local-anchor versus H50 full-song RMS difference was approximately
`-74.3` to `-66.0 dB`; it remains a very small change. The render report is:

`C:/Users/User/Documents/MusicSourceSeparation/data/musdb18-vr-local-anchor-lr3e-6/listening-12/render-report.json`

## Decision

`3e-6` confirms that a stronger local update increases the objective effect,
but not enough to produce a reliable audible improvement. It also begins to
reduce low-vocal instrumental safety quality and increases non-event drift.
Do not continue with a blind `6e-6` learning-rate sweep.

The next technically useful experiment is to keep `3e-6` and `anchorBeta=1`,
but broaden the event mask with a fixed 25-50 ms soft guard band on each side
of the selected 100 ms event. That tests whether the sparse frame mask is
missing the model's temporal context. Keep the same continuation control and
5-pass budget. If that remains below 0.1 dB or remains inaudible, stop this
fine-tune family and revisit model capacity or a dedicated short-event target.

## Provenance

| artifact | SHA-256 |
| --- | --- |
| runner | `38320DEF27BD9AAFB2F98ED79463E29BE0353CBF8C4BE7E0ADD51B2B1CB675E8` |
| local-anchor pass-5 checkpoint | `ED3DE0325B780F0FE04DC75E3E0BFFC2BAFBB66C0E2FDE645A2B68F6C4FCF4DA` |
| continuation pass-5 checkpoint | `C30233F4B231E1562BD4578859DCB9DDC4E08F27C4761AB3CE8E899049608CE7` |
| evaluation report | `2C5E9FE4AEEC765591224634B2888E4FE327BBCC6893AC529170B4EE5EB01A6E` |
| listening report | `C73BAD7D831749AD6CD2C4ACF122B0F1A5A9B00470CD5EC3C5AC4C877F0F3CC4` |

All checkpoints, teacher-derived targets, and audio remain local under the
MUSDB18 non-commercial research restrictions.
