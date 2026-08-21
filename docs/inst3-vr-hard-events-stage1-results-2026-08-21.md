# Inst 3 V-R hard-event analysis, Stage 1 results

Status: complete diagnostic run.  No model was trained, exported, or changed.
All MUSDB18-derived arrays and reports remain under the ignored local
`data/musdb18-inst3-vr-hard-events/` tree.

## Run identity

- Manifest: `musdb18-inst3-oracle-split@1`.
- Scope: all 80 entries whose manifest role is `train`.
- Calibration, internal-test, and official final-test songs were not used.
- Mixture: `vocals + drums + bass + other` decoded from the five-stream
  MUSDB18 source.
- Student: original `vocals_epoch=891.ckpt`, residual-vocals semantic.
- Teacher: verified `uvr_mdxnet_inst_3@2` contract.
- Student contract: 44.1 kHz, 2,048 FFT, 1,024 hop, 128 frames.
- Device: NVIDIA CUDA for both the Inst 3 teacher and the student diagnostic.
- Total analyzed duration: 17,715.1 seconds, approximately 4.92 hours.

Machine-readable report:

`data/musdb18-inst3-vr-hard-events/reports/inst3-vr-hard-events-report.json`

The run produced 80 per-song JSON reports and 80 compressed event-array files.
Raw, decoded, and teacher intermediate directories were empty after the run.

## Diagnostic quantity

The original model output is kept as a vocal residual.  For each song:

```text
teacherRemoved = mixtureGt - Inst3Instrumental
studentRemoved = mixtureGt - initialInstrumental
miss           = teacherRemoved - studentRemoved
```

`miss` measures content that Inst 3 removes but the initial student leaves in
the accompaniment.  The hard-event score is a deterministic ranking aid:

```text
positiveProjectionRms =
    max(dot(miss, teacherRemoved) / power(teacherRemoved), 0)
    * teacherRemovedRms

hardEventScore = positiveProjectionRms
                 * sqrt(max(teacherRemovedRms, activeFloor))
```

Blocks below `-60 dBFS` teacher-removed RMS are inactive.  This score is not a
perceptual quality metric or a vocal ground-truth label.

## Distribution

| Block size | Total blocks | Active blocks | Active fraction |
| ---: | ---: | ---: | ---: |
| 50 ms | 354,347 | 267,117 | 75.38% |
| 100 ms | 177,196 | 134,229 | 75.75% |
| 200 ms | 88,619 | 67,575 | 76.25% |

The fixed activity floor is therefore only an activity gate; it does not by
itself isolate rare short leaks.  Across songs, the 100 ms active-fraction
10th/50th/90th percentiles were 60.5% / 79.9% / 95.3%.

The 100 ms positive-projection p95 across songs had 10th/50th/90th percentiles
of `-41.19 / -36.75 / -30.22 dBFS`; the per-song maximum had
10th/50th/90th percentiles of `-28.01 / -24.54 / -19.84 dBFS`.

The event mapper produced 6,534 valid student windows across the 80 songs.
5,974 windows (91.43%) contain at least one active 100 ms event, so future
sampling must use score ranking or within-song quantiles rather than a binary
active/inactive filter.

Using the frozen per-song window scores:

| Global score slice | Windows | Songs represented | Share of score mass |
| ---: | ---: | ---: | ---: |
| top 5% | 327 | 51 | 24.8% |
| top 10% | 654 | 69 | 39.4% |
| top 25% | 1,634 | 79 | 67.0% |
| top 50% | 3,267 | 80 | 90.9% |

The top 25% is broad enough to cover nearly every song, but a global ranking
still gives long or unusually energetic songs disproportionate influence.  A
per-song hard pool is preferable.

## Event character

For each song and each of 50/100/200 ms, the report retains the top 24 active
events.  Across those 5,760 top-event rows:

| Heuristic category | Count | Share |
| --- | ---: | ---: |
| `vocal-aligned` | 2,393 | 41.5% |
| `vocal-like-or-non-vocal` | 384 | 6.7% |
| `mixed-or-uncertain` | 2,983 | 51.8% |

At 100 ms only, the 1,920 top rows were 765 vocal-aligned, 130
vocal-like-or-non-vocal, and 1,025 mixed-or-uncertain.  Twenty-three songs had
at least one `vocal-like-or-non-vocal` event in their 100 ms top-24 list.

These categories are intentionally heuristic.  The second category includes
the kind of harmony/effected vocal content that the listening review judged
desirable to remove; it must not be filtered out as ordinary accompaniment.
The mixed category is large enough that a hard sampler should retain some of
it rather than training only on clearly isolated vocals.

The highest 100 ms per-song hard scores occurred in `celestial-shore-die-for-
us`, `voelund-comfort-lives-in-belief`, `triviul-angelsaint`, `actions-south-
of-the-water`, and `skelpolu-together-alone`.  They are useful diagnostics,
not a valid exclusive training set.

## Decision for the next experiment

Do not use all globally highest-scoring windows.  The next V-R experiment
should retain the pretrained checkpoint and compare a uniform control with a
per-song hard sampler under the same total update budget:

1. Build a hard pool from the top 25% of valid student windows within each of
   the 80 train songs, using the frozen 100 ms score.  Keep at least one hard
   candidate for every song with a valid window.
2. Use the same total window count in both cells.  A practical first pilot is
   eight uniform windows per song, 640 total, with the hard cell replacing
   25% or 50% of those draws from the per-song hard pools.
3. Keep calibration and internal-test songs completely song-disjoint.  Do not
   use the private listening set or any known listening timestamps to choose
   windows.
4. Retain both `vocal-aligned` and `vocal-like-or-non-vocal` events.  Use the
   heuristic category for stratified reporting, not as a hard rejection rule.
5. Evaluate 50/100/200 ms miss p95/max, short-event hotspot count, ordinary
   instrumental safety metrics, and mechanical-artifact proxies before
   rendering listening files.

The first comparison should be `V-R-U` versus `V-R-H25`; `H50` is the next
cell only if H25 improves short-event metrics without increasing artifacts.
No training was started in Stage 1, and no ONNX/TFLite or `bss-tflite`
publication is authorized by this result.
