# Inst 3 Continuous Baseline Evaluation

Status: completed locally on 2026-08-23. This is a non-commercial MUSDB18
research evaluation. The report, teacher-derived audio, and private listening
files remain under the ignored `data/` tree and are not release artifacts.

## Question

Does the V-R-H50 checkpoint still improve the Inst 3-directed removal behavior
when both the original checkpoint and H50 use the same continuous-context
assembly? The comparison also separates useful-window centers, useful-window
edges, internal joins, and song edges.

## Frozen comparison

- Original student: `models/tfc-tdf/source/vocals_epoch=891.ckpt`.
- H50 student: `data/musdb18-inst3-vr-hard-sampling-h50/runs/V-R-H50/step-8000.pt`.
- Student contract: 128 frames, 2,048 FFT, 1,024 hop, five context hops on
  each side, 119,808-sample output stride.
- Original and H50: identical continuous-context overlap-save renderer.
- Inst 3: native `uvr_mdxnet_inst_3@2` full-song MDX output. MUSDB18 evaluation
  uses the already frozen native teacher cache.
- Evaluation: 10 calibration and 10 internal-test songs; official final-test
  songs were not used.
- Private listening: the existing 12-song private set; H50 files were reused
  from the previously validated continuous renderer.

The continuous output region is partitioned without overlap using this
precedence:

1. first/last 100 ms of the song;
2. +/-100 ms around every internal output boundary;
3. the central 80% of each useful output chunk;
4. the remaining useful-chunk edge samples.

50/100/200 ms events are assigned by event center. Song-edge event counts are
very small because the native MDX teacher has little active removed content in
the first/last 100 ms; those rows are retained for audit but are not a strong
quality estimate.

Machine-readable report:

`data/musdb18-inst3-continuous-baseline-evaluation/continuous-baseline-report.json`

Runner:

`tools/evaluate_inst3_continuous_baseline.py`

## Whole-song result

Values are `H50-continuous - initial-continuous`; negative is better for
residual/projection/error metrics.

| Metric | Delta |
| --- | ---: |
| Instrumental SDR | `+0.013 dB` |
| Accompaniment vocal projection | `-0.172 dB` |
| Inst 3 removed-content positive projection | `-0.471 dB` |
| Teacher-residual miss RMS | `-0.097 dB` |

The H50 direction therefore survives the assembly correction. It is not an
isolated-window artifact: both candidates were rendered with the same real
PCM context around internal windows.

## Region result

These are unweighted per-song mean deltas. The song-edge row is included for
completeness but is too sparse and edge-sensitive for model selection.

| Region | Instrumental SDR | Vocal projection | Inst 3 positive projection | Raw miss RMS |
| --- | ---: | ---: | ---: | ---: |
| Central 80% | `-0.099 dB` | `-0.147 dB` | `-0.247 dB` | `+0.047 dB` |
| Join +/-100 ms | `-0.275 dB` | `-0.581 dB` | `-1.131 dB` | `+0.189 dB` |
| Window edge 10% | `-0.296 dB` | `-0.299 dB` | `-0.575 dB` | `+0.246 dB` |
| Song edge 100 ms | `+22.474 dB`* | `+8.079 dB`* | `-1.104 dB`* | `-2.239 dB`* |

`*` The song-edge aggregate is dominated by near-silent/zero-padded native
teacher edge material and should not be interpreted as a real quality gain.

H50 improves coherent projection in all three usable internal regions. The
small positive raw-miss deltas in the center, join, and window edge show that
H50 is not simply reducing every component of the waveform error; it is moving
the error away from the coherent Inst 3-removed direction.

## Short-event result

The table uses all active blocks over the 20 songs. Negative is better.

| Block | Positive projection p95 | Positive projection max | Raw miss p95 | Raw miss max | Count above -30 dBFS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 ms | `-0.698 dB` | `-0.185 dB` | `-0.252 dB` | `+2.835 dB` | `-302` |
| 100 ms | `-0.797 dB` | `-0.201 dB` | `-0.234 dB` | `+2.349 dB` | `-149` |
| 200 ms | `-0.828 dB` | `-0.185 dB` | `-0.185 dB` | `+3.296 dB` | `-76` |

Per-song positive-projection p95 improved on 15/20 songs at 50 ms and 16/20
at both 100 and 200 ms. The corresponding maximum improved on 16/20, 15/20,
and 15/20 songs. Raw miss maximum improved on only 10/20, 11/20, and 12/20
songs, which is the main remaining safety warning.

## Join diagnostics

The continuous assembly seam diagnostic compares each internal output boundary
with its local derivative distribution:

| Model | Mean seam p95 ratio | Mean seam max ratio | Boundaries >=2x | Songs with a >=2x boundary |
| --- | ---: | ---: | ---: | ---: |
| Initial continuous | `1.906` | `4.064` | `99` | `19/20` |
| H50 continuous | `1.754` | `4.163` | `75` | `18/20` |

H50 has fewer elevated joins and a lower mean p95 ratio, while its mean maximum
ratio is slightly higher. This is an artifact diagnostic, not a perceptual
score; it does not show a new systematic continuous-assembly failure.

## Private listening files

The 12-song set contains 72 validated 44.1 kHz stereo PCM16 FLAC files:

`data/musdb18-inst3-continuous-baseline-evaluation/private-listening-12/`

Variants:

- `initial-continuous/`
- `H50-continuous/`
- `Inst3-native/`

The files preserve each source frame count. H50 uses the previously validated
continuous instrumental files from
`data/musdb18-inst3-f24-h50-comparison/h50-128-continuous-instrumental/`;
the report records those source files and hashes.

## Decision

The H50 improvement is present in the central useful region and internal join
neighborhoods under a matched continuous assembly. The earlier H50 result
therefore cannot be dismissed as only an isolated-window boundary artifact.

It remains an aggressive-removal diagnostic, not a product-quality claim. The
increase in worst raw miss means the model can remove more Inst 3-aligned
content while also producing larger non-coherent deviations in some songs.
The next H50-based experiment should use the continuous contract and target
the worst actual miss events with a top-k audio-domain local objective, while
keeping the H50 output as an anchor outside those events. More H75 sampling,
more event-weight multiplier, and another isolated-window comparison are not
justified by this evaluation.
