# Continuous H50 Bounded Continuation

Status: completed locally on 2026-08-23. This is a non-commercial MUSDB18
research experiment. Checkpoints, caches, and listening audio remain ignored
local artifacts.

## Objective

The user found that `H50-continuation@step-800` audibly reduced vocal residue
in some passages relative to the original H50. This run tests whether that
improvement has useful headroom without changing event selection, assembly, or
loss semantics.

## Frozen contract

- Source: `H50-continuation@step-800`.
- Source optimizer: AdamW state restored from the source checkpoint.
- Additional training: five passes, 640 records/pass, 800 updates.
- Milestones: `+1` at global step 960, `+3` at 1280, `+5` at 1600.
- Records: the same 80-song continuous central-event records used by the
  source continuation run; no dynamic event refresh.
- Target: full useful-window `V_T = mixtureGt - Inst3Instrumental` audio
  Charbonnier loss.
- Assembly: 128-frame continuous-context overlap-save.
- Learning rate: `1e-6`; batch size 4; seed 891; BatchNorm statistics frozen.
- Evaluation: 10 calibration and 10 internal-test songs; official final-test
  songs unused.

Runner:

`tools/run_inst3_vr_continuation.py`

Machine report:

`data/musdb18-inst3-vr-continuation/reports/continuation-report.json`

## Holdout learning curve

Values are relative to the original continuous H50 checkpoint. Negative is
better for Inst 3 projection; positive raw miss is a warning.

| Milestone | 50 ms projection p95 | 100 ms projection p95 | 200 ms projection p95 | 100 ms raw miss p95 | 200 ms raw miss p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `+1` | `-0.962 dB` | `-1.034 dB` | `-1.074 dB` | `+0.004 dB` | `+0.042 dB` |
| `+3` | `-1.006 dB` | `-1.081 dB` | `-1.134 dB` | `+0.009 dB` | `+0.079 dB` |
| `+5` | `-1.045 dB` | `-1.119 dB` | `-1.169 dB` | `+0.027 dB` | `+0.112 dB` |

Positive-projection p95 improved on all 20 holdout songs at every milestone.
The maximum positive projection also improved on 18/20 songs at 50 ms and
20/20 songs at 100/200 ms. The raw miss distribution does not follow the same
direction: at +5 the worst per-song 200 ms raw-miss p95 delta is approximately
`+4.44 dB`.

This is a genuine tradeoff, not a boundary-only effect. The training records
exclude song edges, join neighborhoods, and useful-span edges, and the holdout
evaluation uses continuous context.

## Checkpoints

| Milestone | File | SHA-256 |
| --- | --- | --- |
| `+1` | `runs/H50-continuation-plus5/step-960.pt` | `49bafba10f5fd3bd34e92332a69931c8cb730da43c90edc39679477e34ac89d9` |
| `+3` | `runs/H50-continuation-plus5/step-1280.pt` | `a974330a0461d17dd2809e8252cb33e13dfb0a7116143060273a4109cce8e5a5` |
| `+5` | `runs/H50-continuation-plus5/step-1600.pt` | `cc0ce960e2d305cc0f03e69fc58988be720d69fdd192f6b746b692959a674` |

The source checkpoint is
`H50-continuation@step-800`, SHA-256
`d02c9629bc1c25039f6dbb17a04bca5b151ffb001d2a740d55be9511adde121a`.

## Listening set

The runner generated 120 validated PCM16 FLAC files for the 12-song private
set under:

`data/musdb18-inst3-vr-continuation/listening-12/`

Variants:

- `H50-pass-50`
- `H50-continuation@step-800`
- `H50-continuation+1`
- `H50-continuation+3`
- `H50-continuation+5`

The user listening decision is still pending. The primary listening question
is whether +1 or +3 removes more conspicuous short syllables than step800
without introducing the raw-miss warning as audible scrape, tonal loss, or
instrument damage. +5 should be treated as a possible over-training candidate,
not assumed to be best because its projection p95 is lowest.

## Runtime observation

During this continuation run, instantaneous GPU samples reached approximately
99% utilization with about 6.5 GB memory used. This differs from the earlier
low sampled averages and confirms that batch-4 compute phases can fully occupy
the GPU; host scheduling and sampling interval explain much of the apparent
average under-utilization.

## Training-event listening excerpts

After the full-song review found no clear improvement at the conspicuous short
leak points, the renderer selected the global top 40 events from the actual
continuous training manifest. Ranking uses the H50 positive-projection RMS,
with raw miss RMS as the tie-break. Duplicate event intervals are removed, but
multiple independent events from the same song are retained.

Each approximately two-second event includes:

- mixture;
- Inst 3 target instrumental and removed residual;
- H50 pass-50 instrumental/residual;
- continuation step-800 instrumental/residual;
- continuation +1/+3/+5 instrumental/residual.

The local listening root and machine index are:

`data/musdb18-inst3-training-event-listening/`

`data/musdb18-inst3-training-event-listening/event-listening-report.json`

File names share the `event-NNN-song-slug` prefix across all variant
directories. The report records absolute source time, event interval inside
the snippet, continuous training score, target/model checkpoint hashes, and
PCM output hashes.

## Decision boundary

Do not automatically continue beyond +5. Select the best milestone from the
12-song continuous listening set while requiring that the 100/200 ms raw-miss
maxima do not materially worsen. If +1 or +3 is audibly preferred and raw
misses remain controlled, use that checkpoint for the next targeted experiment.
If only +5 sounds better despite the raw-miss increase, inspect the affected
holdout songs before accepting it.
