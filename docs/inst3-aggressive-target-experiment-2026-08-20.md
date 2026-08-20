# Inst 3 aggressive-target experiment

Status: complete as a local, non-commercial, four-song pilot. This experiment
implements the follow-up direction from the listening review: it treats Inst 3
as a valid aggressive karaoke/vocal-removal teacher rather than rejecting it
solely because it removes vocal-like accompaniment content.

Runner: [run_inst3_aggressive_target.py](../tools/run_inst3_aggressive_target.py)

Machine-readable report:

`data/musdb18-inst3-aggressive-target/reports/inst3-aggressive-target-report.json`

All checkpoints, teacher audio, MUSDB18-derived material, and listening files
remain under ignored local `data` directories. They are not releasable model
artifacts.

## Target contract

The 128-frame student keeps the existing `vocals_epoch=891.ckpt` neural core.
It still predicts a vocal residual and exposes:

```text
student_instrumental = mixture_gt - predicted_vocal_residual
```

Unlike the prior anchored objective, the supervision target is constructed in
the audio domain first and then converted using the student STFT:

```text
target(alpha) = (1 - alpha) * musdb18_instrumental
              + alpha * inst3_instrumental
```

There is no anchor loss, standard-ground-truth side loss, or vocal-projection
penalty. The sole training loss is L1 against the packed student spectrum of
that audio-domain target. This makes `alpha=1.0` a direct attempt to imitate
Inst 3's aggressive removal behavior.

## Controls

- Frozen manifest: `musdb18-inst3-oracle-split@1`.
- Songs: two train, one calibration, one internal-test; official final-test
  tracks were not extracted or evaluated.
- Segment: 15.0 to 45.0 seconds.
- Training: 8 vocal/mixture-RMS-stratified windows per train song, 16 total.
- Held out: 3 non-tail windows per train song, 6 total.
- Student: `vocals_epoch=891.ckpt`, SHA-256
  `101921dac943e1683452f293dc0d32df53b694886541952be26d9592d819c20d`.
- Teacher: `uvr_mdxnet_inst_3@2`, ONNX SHA-256
  `2b7834e2972158d8c9864e7376e3a7d084079c80a23f38dc31c4b0a4e901a1cb`.
- CUDA: RTX 4060 Laptop GPU; student `cuda`; teacher
  `CUDAExecutionProvider, CPUExecutionProvider`.
- BatchNorm running statistics frozen; AdamW with zero weight decay; gradient
  norm clipping at `1.0`; learning rate `1e-5`; fixed seed `891`.
- Alpha values: `0.50`, `0.75`, and `1.00`.
- Milestones: `0`, `128`, `512`, and `2048` steps.

## Numerical result

`aggressiveTargetSdrDb` measures similarity to the alpha-specific mixed target.
`teacherRemovalResidualSdrDb` measures similarity between the student's removed
residual and Inst 3's removed residual. Higher is closer. Standard instrumental
SDR remains reported, but is not the sole gate for this aggressive-removal
candidate.

At 2048 steps:

| Variant | Full target delta | Full teacher-residual delta | Train-held-out target delta | Full standard instrumental SDR |
| --- | ---: | ---: | ---: | ---: |
| alpha 0.50 | +0.71 dB | +0.34 dB | +0.53 dB | 14.73 dB |
| alpha 0.75 | +0.53 dB | +0.35 dB | +0.73 dB | 14.69 dB |
| alpha 1.00 | +0.35 dB | +0.35 dB | +0.81 dB | 14.63 dB |

The calibration/internal-test target metric changed by only about `-0.02` to
`+0.01 dB`. Therefore the fixed two-song training set does teach its own
windows and some held-out portions, but has not established a robust result on
unseen songs. This is not a release-quality model-selection result.

## Extra-track listening set

The runner generated 33 valid 30-second files, all 44.1 kHz stereo PCM16 FLAC:

`data/musdb18-inst3-aggressive-target/listening-extra/`

For each of `yoru-ni-kakeru`, `coldplay-tove-lo`, and `coast-town`, the folder
contains:

- `initial`;
- `teacher-inst3`;
- every alpha at steps `128`, `512`, and `2048`.

On the external samples, the 2048-step candidates move toward Inst 3 but remain
far from matching it. The reduction in waveform error to Inst 3 is:

| Song | alpha 0.50 | alpha 0.75 | alpha 1.00 |
| --- | ---: | ---: | ---: |
| yoru-ni-kakeru | +1.87 dB | +1.93 dB | +1.99 dB |
| coldplay-tove-lo | +0.98 dB | +1.05 dB | +1.02 dB |
| coast-town | +0.33 dB | +0.46 dB | +0.60 dB |

Positive values mean less waveform error relative to the initial checkpoint.
This is a diagnostic, not a quality score. The external files must be judged by
listening for vocal, backing-vocal, and harmony removal versus accompaniment
damage.

## Practical candidate for the next listening decision

`alpha-1.00-step-2048` has the most direct target semantics and is at least as
close to Inst 3 as the mixed variants on every extra song, although the margin
is modest. Compare it first with `initial` and `teacher-inst3`, then compare
the same-step `alpha-0.50` and `alpha-0.75` files to decide whether partial
teacher influence produces a more useful trade-off.

Do not move to 24-frame export, Booming SS integration, or `bss-tflite`
publication until the external listening set shows a repeatable, useful change
relative to the initial compact model. If listening does show that effect, the
next training experiment should expand beyond the two train songs while keeping
song-level held-out and final-test isolation.
