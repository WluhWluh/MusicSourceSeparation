# Inst 3 objective alignment experiment

Status: complete as a local, non-commercial research experiment. The
checkpoint, listening files, and MUSDB18-derived audio remain under the
ignored `data` tree and are not releasable artifacts.

## Purpose

The preceding stability sweep showed that freezing BatchNorm removed the
earlier catastrophic drift, but a plain `0.10 * Inst 3` soft target produced
no useful improvement. This experiment first audits the teacher against the
existing TFC-TDF checkpoint, then tests an anchor loss and an audio-domain
vocal-projection penalty before considering a larger training run.

The frozen input is `mixture_gt = vocals + drums + bass + other`. The student
keeps the existing `vocals_epoch=891.ckpt` neural core and defines its direct
output as `mixture_gt - predicted_vocal_residual`.

Runner: [run_inst3_objective_alignment.py](../tools/run_inst3_objective_alignment.py)

Machine-readable report:

`data/musdb18-inst3-objective-alignment/reports/inst3-objective-alignment-report.json`

Listening artifacts are in the ignored
`data/musdb18-inst3-objective-alignment/listening` directory. They include the
initial checkpoint, the Inst 3 teacher, and every trained variant for all four
30-second segments. Blind listening has not been scored automatically.

## Controls

- Frozen manifest: `musdb18-inst3-oracle-split@1`.
- Songs: two `train`, one `calibration`, and one `internal-test`.
- Official `final-test`: not extracted or evaluated.
- Segment: 15.0 to 45.0 seconds for every song.
- Training windows: 8 activity-stratified windows per training song, selected
  using vocal RMS and mixture RMS; the remaining training windows are held
  out.
- Active-vocal threshold: `0.05696107` RMS; 12 of 16 training windows qualify.
- Optimizer: AdamW, zero weight decay, gradient norm clip 1.0.
- BatchNorm: running statistics frozen and model kept in evaluation mode during
  updates.
- Learning rate: `3e-6`.
- Updates: 32, with milestones `0/1/4/8/16/32`.
- Device: RTX 4060 Laptop GPU; student `cuda`; teacher
  `CUDAExecutionProvider, CPUExecutionProvider`.

## Teacher audit

Metrics are energy-SNR proxies. Vocal projection is the projection of
instrumental error onto the reference vocal stem; more negative is better.

| Four-song aggregate | Instrumental SDR | Vocal SDR | Vocal projection | Low-vocal instrumental SDR |
| --- | ---: | ---: | ---: | ---: |
| Initial TFC-TDF checkpoint | 14.109 dB | 7.698 dB | -15.923 dB | 41.458 dB |
| Inst 3 teacher | 11.961 dB | 5.549 dB | -19.332 dB | 1.669 dB |
| Inst 3 minus initial | -2.149 dB | -2.148 dB | -3.409 dB | -39.789 dB |

Inst 3 reduces vocal-correlated error in this aggregate, but it does so with
substantial instrumental damage. The failure is especially visible on the
calibration song `lushlife-toynbee-suite`: teacher instrumental SDR is
`0.518 dB` and low-vocal instrumental SDR is `0.502 dB`, versus `42.561 dB`
and `42.892 dB` for the initial checkpoint. The teacher is therefore not a
safe global soft target for this compact student.

## Objective matrix

All variants start from the same checkpoint, use the same window order and
seed, and are evaluated on the complete four-song segments at step 32.

| Variant | Instrumental SDR | Vocal projection | Low-vocal instrumental SDR | Projection delta vs initial |
| --- | ---: | ---: | ---: | ---: |
| S0 ground truth | 14.131 dB | -15.575 dB | 42.815 dB | +0.347 dB |
| S0 anchor | 14.128 dB | -15.803 dB | 42.261 dB | +0.119 dB |
| S0 vocal projection | 14.132 dB | -15.737 dB | 42.699 dB | +0.186 dB |
| S1 anchor + Inst 3 (0.01) | 14.128 dB | -15.801 dB | 42.269 dB | +0.122 dB |
| S1 anchor + Inst 3 (0.03) | 14.129 dB | -15.795 dB | 42.287 dB | +0.128 dB |
| S1 anchor + Inst 3 (0.10) | 14.129 dB | -15.776 dB | 42.345 dB | +0.146 dB |

The projection delta is positive for every trained variant, meaning the
instrumental error became more vocal-correlated than the zero-update
checkpoint on the four-song aggregate. The held-out windows show the same
direction. The anchor reduces drift, but does not create the desired quality
improvement. Inst 3 weight changes from `0.01` to `0.10` are too small to
alter that conclusion.

## Gate decision

The objective gate required at least `0.25 dB` vocal-projection improvement,
with no more than `0.10 dB` loss in instrumental SDR or low-vocal instrumental
SDR. No variant passed. The numerical gate therefore does not justify
exporting these variants to 24-frame, training on 80 songs, integrating them
into Booming SS, or publishing them to `bss-tflite`.

## Subjective listening follow-up

The follow-up review is recorded in
`docs/inst3-aggressive-vocal-removal-listening-review-2026-08-20.md`.
On the three additional listening tracks, the initial checkpoint and all six
32-step student variants were not audibly distinguishable. Inst 3 was clearly
more aggressive and was judged to leave effectively no vocal residue, without
an objectionable accompaniment change in the reviewed material. The content
removed from `Lushlife - Toynbee Suite` sounded like effected harmony or
backing-vocal material to the listener and is desirable for the intended
karaoke/vocal-removal use case.

This does not invalidate the conventional instrumental-fidelity gate above. It
does change the next research question: Inst 3 should be tested as an explicit
aggressive vocal-removal target, with collateral removal reported separately
from standard instrumental preservation.

## Recommendation

Do not expand the current anchored objective or export it to 24-frame. Instead,
run a separate aggressive-removal experiment with direct mixed targets:

```text
target(alpha) = (1 - alpha) * musdb18_instrumental + alpha * inst3_instrumental
```

Test `alpha=0.5`, `0.75`, and `1.0` at 128, 512, and 2048 update milestones,
while retaining the frozen song split, activity coverage, BatchNorm freeze, and
held-out evaluation. The practical gate is an audible reduction of lead,
backing, and harmony residue relative to initial, with acceptable retention of
clearly non-vocal accompaniment. Standard instrumental SDR remains a reported
diagnostic, but is not the sole gate for this purpose-built model.

All derived checkpoints and audio from this experiment remain local research
artifacts under the MUSDB18 and teacher-source usage restrictions.
