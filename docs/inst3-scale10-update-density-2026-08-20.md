# Inst 3 scale-10 update-density experiment

Status: complete as a local, non-commercial research experiment, including a
user listening review. The run isolates update density for the ten-song
`alpha=1.00` aggressive-removal student and renders twelve complete private
listening songs. It does not establish a releasable model.

Runner: [run_inst3_scale10_density.py](../tools/run_inst3_scale10_density.py)

Machine-readable reports:

```text
data/musdb18-inst3-scale10-density/reports/inst3-scale10-density-report.json
data/musdb18-inst3-scale10-density/reports/full-listening-report.json
```

MUSDB18-derived audio, teacher output, checkpoints, private source audio, and
rendered listening files remain under the ignored `data` tree. They must not
be redistributed or uploaded to `bss-tflite`.

## Question

The earlier four-song pilot trained each of 16 windows about 128 times. The
scale-10 experiment trained 320 windows only 25.6 times at its 8192-step
milestone. Listening found lower average vocal residue in scale-10, but some
short syllables in `yoru-ni-kakeru` were more conspicuous than in the small
pilot.

This run tests whether increasing visits over the exact same 320 windows
recovers those local events without changing data coverage or the objective.

## Frozen contract

- Student: 128-frame TFC-TDF initialized from `vocals_epoch=891.ckpt`.
- Target: direct Inst 3 instrumental, `alpha=1.00`.
- Train data: the same ten frozen scale-10 songs and 32 windows per song.
- Training windows: 320; disjoint train holdout: 16 windows per song.
- Evaluation: ten calibration and ten internal-test songs, 16 windows each.
- Official MUSDB18 final-test songs were not loaded or evaluated.
- Seed: `891`; learning rate: `1e-5`; AdamW with zero weight decay.
- BatchNorm running statistics frozen; gradient clipping at `1.0`.
- Milestones: `0`, `8192`, `20480`, and `40960` updates.
- Visits per training window: `0`, `25.6`, `64`, and `128`.
- Student execution: CUDA on an RTX 4060 Laptop GPU.
- Teacher execution: ONNX Runtime CUDA with CPU fallback registered.
- Run contract ID:
  `68d30520ac6d14d2bf773223afb9700c35e60a9aaf7ea4f90ef555aa9abdcb2e`.
- Runner SHA-256:
  `da6de82fe6658624623053849dcb1b476aa08b46a0db737f1ea3faf0738c5f80`.

The optimizer, model, window order, NumPy RNG, PyTorch CPU RNG, and CUDA RNG
were persisted every 1024 updates. The host session ended after step 37888;
the run resumed once from that exact state and completed with `resumeCount=1`.
Training and milestone evaluation took 3124.08 seconds in total.

Checkpoint SHA-256 values:

| Step | SHA-256 |
| ---: | --- |
| 8192 | `a7bdac5a0d684416d06acea42c9b76f04d1d43ece7fda46db25a0ce08f2606c5` |
| 20480 | `23a896df6f5038bef31626ffde9bf72af366723de8fa56a02073817bd58a86b0` |
| 40960 | `255b46c0e9e7807156653ea8cd76947da52358128c505421da8bd83b97b168d8` |

## Held-out numerical result

The table covers the song-disjoint calibration plus internal-test set. Higher
teacher-residual SDR means closer removed-residual behavior to Inst 3. More
negative vocal projection is better. These are diagnostics rather than a
replacement for listening.

| Step | Visits/window | Aggressive-target SDR | Teacher-residual SDR | Instrumental SDR | Low-vocal instrumental SDR | Vocal projection |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0 | 16.522 dB | 10.520 dB | 13.913 dB | 35.430 dB | -17.103 dB |
| 8192 | 25.6 | 16.695 dB | 10.693 dB | 13.944 dB | 34.071 dB | -16.873 dB |
| 20480 | 64.0 | 16.663 dB | 10.661 dB | 13.937 dB | 33.446 dB | -17.043 dB |
| 40960 | 128.0 | 16.591 dB | 10.589 dB | 13.917 dB | 33.639 dB | -17.077 dB |

The teacher-target metrics peak at 8192 and then decline. At 40960, the
teacher-residual gain over the initial model is only about `0.069 dB`, versus
`0.173 dB` at 8192. Conventional instrumental SDR follows the same shallow
trajectory. Uniformly repeating the same windows therefore does not produce
a monotonic improvement.

## Twelve-song complete listening set

The frozen private set contains:

`already-gone`, `chasing-the-wind`, `coast-town`, `coldplay-tove-lo`,
`i-see-fire`, `imagine`, `lugu-lake`, `north`, `odd-future`,
`traveling-light`, `unseen-sea`, and `yoru-ni-kakeru`.

For each complete song, the runner produced:

```text
teacher-inst3.flac
initial.flac
alpha-1.00-step-8192.flac
alpha-1.00-step-20480.flac
alpha-1.00-step-40960.flac
```

All 60 outputs are 44.1 kHz stereo PCM16 FLAC, have the same frame count as
their decoded source, and passed a second file-size and SHA-256 audit. The set
occupies about 1.374 GiB.

The following values are unweighted means over the twelve songs. Lower error
RMS and more negative 100 ms retained-residual values are better.

| Variant | Teacher-residual SDR | Instrumental-to-teacher SDR | Error RMS | 100 ms retained p95 | 100 ms retained max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 7.722 dB | 13.299 dB | -27.176 dBFS | -25.753 dBFS | -16.766 dBFS |
| Step 8192 | 7.924 dB | 13.501 dB | -27.378 dBFS | -26.181 dBFS | -17.299 dBFS |
| Step 20480 | 7.911 dB | 13.488 dB | -27.365 dBFS | -26.233 dBFS | -17.218 dBFS |
| Step 40960 | 7.802 dB | 13.380 dB | -27.256 dBFS | -26.035 dBFS | -17.142 dBFS |

Step 8192 is best on mean teacher similarity and worst-event magnitude. Step
20480 is narrowly best on the mean 100 ms p95. Step 40960 regresses on every
mean relative to the two earlier milestones, while remaining slightly better
than initial on these aggregate private-set diagnostics.

For `yoru-ni-kakeru`, the 100 ms retained-residual p95 changes from
`-23.196 dBFS` initially to `-25.299`, `-25.211`, and `-24.856 dBFS`. The
maximum changes from `-15.802 dBFS` to `-16.753`, `-16.738`, and
`-16.720 dBFS`. A persistent hotspot near 97.7 seconds remains approximately
`-16.7 dBFS` at every trained milestone. Increased uniform update density did
not fix that local event.

## Interpretation

This experiment rejects the simple hypothesis that matching the small
pilot's 128 visits per window is sufficient. The useful region appears to be
between 8192 and 20480 updates for this fixed data and objective. More uniform
epochs begin to overfit or drift without solving the most salient short
residuals.

## User listening review

The user compared the complete private-song outputs at all three density
milestones with the preceding scale-10 result and the original four-song
aggressive-target pilot.

- Increasing update density did not produce a perceptual improvement over the
  preceding scale-10 result.
- The conspicuous residual-vocal hotspots did not improve. This agrees with
  the 50-200 ms diagnostics and the persistent `yoru-ni-kakeru` hotspot.
- None of the 8192, 20480, or 40960 density checkpoints displaced the original
  four-song small-pilot result in listening preference.
- The original four-song `alpha=1.00-step-2048` result remains the most
  pleasing of the trained students heard so far.

This preference is a model-development signal rather than a generalization
claim. The four-song pilot saw only 16 training windows and may have benefited
from its specific songs or window distribution. It remains unsuitable for
publication or product integration without broader validation.

## Decision and next experiment

Do not continue uniform training beyond 8192 steps, select the 40960 checkpoint,
expand this trajectory to the full 80-song training split, retrain a 24-frame
variant, integrate it into Booming SS, or publish its derived weights. More
uniform exposure is not addressing the perceptually dominant failure mode.

Keep the original four-song `alpha=1.00-step-2048` checkpoint as the current
listening reference. The next controlled experiment should determine which
part of the small pilot produced its more pleasing behavior:

1. preserve the ten-song scale-10 data and direct Inst 3 target;
2. rank candidate windows using 50-200 ms Inst 3 removed-residual activity and
   current student-to-teacher error;
3. replace a controlled fraction of uniform updates with those hard windows;
4. include the known external residual timestamps only in evaluation, never
   in training selection;
5. compare against both the scale-10 step-8192 checkpoint and the four-song
   step-2048 listening reference.

This isolates sampling and short-event emphasis without weakening the desired
aggressive-removal semantic with an ordinary-instrumental anchor.
