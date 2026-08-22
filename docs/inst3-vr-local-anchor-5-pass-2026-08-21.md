# Inst 3 V-R Local Anchor Fine-Tune

Status: completed locally on 2026-08-21. This is a non-commercial MUSDB18
research experiment. No product runtime or release asset was changed.

## Objective

The preceding alignment diagnostic showed that moving an event to the center
of the TFC-TDF useful span did not materially improve the held-out result. This
experiment therefore kept the V-R-H50 schedule fixed and changed only the
training target around the selected event frames.

Both arms started from the exact H50 pass-50 checkpoint:

`data/musdb18-inst3-vr-hard-sampling-h50/runs/V-R-H50/step-8000.pt`

- `H50-continuation`: the Inst 3 residual-vocal target on every frame;
- `H50-local-anchor`: the Inst 3 target on event frames and the frozen H50
  output on all non-event frames.

The student remained the original residual-vocals model. The target semantic
was unchanged:

```text
mixtureGt = vocals + drums + bass + other
teacherRemoved = mixtureGt - Inst3Instrumental
```

## Frozen Contract

- 80 training songs and 20 song-disjoint calibration/internal-test songs;
- existing V-R-H50 selection and schedule, 8 windows per song per pass;
- 5 passes, 640 records per pass, 800 updates per arm;
- batch size 4, AdamW, learning rate `1e-6`, weight decay 0;
- seed 891, frozen BatchNorm running statistics, gradient clip 1.0;
- milestones at 0, 1, 2, and 5 passes;
- `anchorBeta=1.0`;
- 44.1 kHz and the existing TFC-TDF spectral contract;
- no centered-window placement, 24-frame model, QNN, or product integration.

The local loss was:

```text
L_event  = mean(abs(student - Inst3Target) on event frames)
L_anchor = mean(abs(student - H50Output) on non-event frames)
L_total  = L_event + L_anchor
```

The H50 output was generated in memory from the same cached input spectra. No
new teacher inference or full-song teacher cache was created.

Command:

```powershell
.\.tmp\vr-hard-env\Scripts\python.exe tools\run_inst3_vr_local_anchor.py `
  --device cuda --threads 8 --passes 5 --milestones 0,1,2,5 `
  --output-root data\musdb18-inst3-vr-local-anchor --no-resume
```

Both arms completed on CUDA without non-finite loss, CUDA failure, or failed
checkpoint. Training elapsed times were approximately 248 seconds for
continuation and 312 seconds for local-anchor.

## Training Signals

| arm | final total loss | final event loss | final anchor loss | mean event-frame fraction (last pass) |
| --- | ---: | ---: | ---: | ---: |
| continuation | 0.048055 | 0.048055 | 0 | 1.000 |
| local-anchor | 0.135658 | 0.135632 | 0.0000256 | 0.305 |

The local arm received event supervision on roughly 30% of the selected cache
frames. The small non-event anchor loss confirms that the model stayed close
to H50 outside the event mask, but it also explains why a short 5-pass run can
only produce a very small audible change.

## Holdout Results

Values are on the same 20-song evaluation set. More negative projection and
miss values are better. Deltas are relative to H50 pass-50.

| checkpoint | instrumental SDR delta | low-vocal SDR delta | accompaniment vocal projection delta | 100 ms miss p95 delta | 100 ms positive projection p95 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| lambda-0.50 pass-50 | +0.0173 dB | -0.0893 dB | -0.2903 dB | +0.0200 dB | -0.3488 dB |
| continuation pass-1 | -0.0001 dB | -0.0055 dB | -0.0056 dB | +0.0102 dB | -0.0132 dB |
| continuation pass-2 | -0.0006 dB | -0.0037 dB | +0.0032 dB | +0.0120 dB | +0.0228 dB |
| continuation pass-5 | -0.0008 dB | +0.0021 dB | +0.0326 dB | +0.0159 dB | +0.0483 dB |
| local-anchor pass-1 | -0.0002 dB | -0.0027 dB | -0.0042 dB | +0.0008 dB | -0.0087 dB |
| local-anchor pass-2 | -0.0002 dB | -0.0029 dB | -0.0041 dB | +0.0015 dB | -0.0039 dB |
| local-anchor pass-5 | +0.0002 dB | -0.0051 dB | -0.0108 dB | +0.0091 dB | -0.0105 dB |

At local-anchor pass-5, the 100 ms positive-projection p95 improved on all
20 evaluation songs, but the mean improvement was only `0.0193 dB` and the
100 ms miss p95 improved on only 10/20 songs. The 50 ms and 200 ms projection
p95 mean improvements were `0.0132 dB` and `0.0148 dB`, respectively. These
are far below the approximately 0.5 dB gate for a meaningful follow-up.

The worst 100 ms positive-projection value changed by only `0.0001 dB`; the
local target did not solve the individual short-leak hotspot. The continuation
control drifted slightly in the wrong direction by pass 5, which supports
using the local anchor rather than simply training H50 longer, but does not
make the local result large enough to retain as a new listening winner.

## Artifact Checks

| checkpoint | derivative excess p95 | derivative excess max | clipped samples | non-finite windows |
| --- | ---: | ---: | ---: | ---: |
| H50 pass-50 | 0.8698 dB | 3.2815 dB | 0 | 0 |
| continuation pass-5 | 0.8432 dB | 3.2734 dB | 0 | 0 |
| local-anchor pass-5 | 0.8675 dB | 3.2778 dB | 0 | 0 |

The anchor-preservation diagnostic measured spectral-coefficient RMS outside
the event mask. It is not audio dBFS. For local-anchor pass-5 the coefficient
difference was approximately `-67.1` on that diagnostic scale, versus
`-48.9` for continuation, confirming that the anchor materially reduced
ordinary-region drift.

## Listening Outputs

The runner produced 48 validated PCM16 FLAC files: four variants across the
12-song private listening set:

`data/musdb18-inst3-vr-local-anchor/listening-12/`

Variants:

- `H50-pass-50`
- `lambda-0.50-pass-50`
- `H50-continuation@pass-5`
- `H50-local-anchor@pass-5`

All files match their source frame count, sample rate, channel count, and are
finite. The local-anchor versus H50 full-song RMS difference was between
approximately `-74 dB` and `-83 dB` across the 12 songs, so an audible
difference is not expected from this 5-pass setting. Continuation differed by
roughly `-61 dB` to `-72 dB`.

The listening render report is:

`data/musdb18-inst3-vr-local-anchor/listening-12/render-report.json`

## Decision

The experiment verifies that the local-anchor objective is implemented and
preserves ordinary regions better than unrestricted continuation, but five
passes at `1e-6` are insufficient to produce a meaningful reduction in the
remaining short leaks. Treat `H50-local-anchor@pass-5` as a diagnostic output,
not a new best model.

The next worthwhile experiment is not more passes with the exact same tiny
update. First test a modestly stronger local adaptation while retaining the
anchor: `learning-rate=3e-6`, `anchorBeta=1` for 5 passes, with the same
continuation control. If that still stays below roughly `0.1 dB` improvement,
stop this objective and reconsider event-mask breadth or model capacity rather
than extending training indefinitely. Any candidate must still pass the 12-song
blind listening check before further export work.

## Provenance

| artifact | SHA-256 |
| --- | --- |
| runner | `38320DEF27BD9AAFB2F98ED79463E29BE0353CBF8C4BE7E0ADD51B2B1CB675E8` |
| local-anchor pass-5 checkpoint | `E38ED69C41EF54E7CBA87BE7B5EF816C6C712D08B60BDDDEB4A0B7B9947A295A` |
| continuation pass-5 checkpoint | `FCCD7C4198F6D9253D6A6A5D00B250CE7E94EB1112E299C2BD8C69C4A7987FBB` |
| evaluation report | `C5B1C48C502EB10E7C822FDDD029130D3429E73BA5D7E9EC09AA3F6D7B5B3FD5` |
| listening report | `C54D8E7F0E7CE24E0A195EDD224A6091B33F070C95BFC5B37E3BE851F26A5C74` |

All checkpoints, teacher-derived targets, and audio remain local under the
MUSDB18 non-commercial research restrictions.
