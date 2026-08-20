# Inst 3 aggressive vocal-removal listening review

Status: recorded from the user's listening review on 2026-08-20. This is a
subjective follow-up to the objective-alignment experiment and does not replace
its numerical evaluation.

## Listening set

The comparison used the same 15.0-45.0 second, 44.1 kHz stereo, PCM16 FLAC
specification for:

- `yoru-ni-kakeru`
- `coldplay-tove-lo` (source file: `data/samples/fun.mp3`)
- `coast-town`

The extra-track renders are local ignored artifacts under:

`data/inst3-objective-alignment-listening-extra/`

They contain the initial checkpoint, six objective-alignment student variants,
and the Inst 3 teacher output for each song. The source and render identities
are recorded in `render-report.json`.

## User listening result

The following outputs were not audibly distinguishable from one another:

- initial checkpoint;
- `S0-ground-truth`;
- `S0-anchor`;
- `S0-vocal-projection`;
- `S1-anchor-inst3-0.01`;
- `S1-anchor-inst3-0.03`;
- `S1-anchor-inst3-0.1`.

The vocal-removal impression was also the same across those variants. This is
consistent with the measured student changes being roughly -43 to -52 dB
relative to the initial output on the three extra tracks: the 32-step updates
were too small to establish a new audible separation behavior.

Inst 3 was clearly distinguishable and was judged to have effectively no
audible vocal residue. No negative accompaniment artifact was noticed in this
listening set. On `Lushlife - Toynbee Suite`, the content removed by Inst 3
that the numerical evaluation classified as accompaniment sounded more like
effected harmony or backing-vocal material to the listener, and should count
as desirable removal for this product use case.

## Interpretation

The previous objective gate measured conventional instrumental preservation.
That gate remains valid for a general-purpose instrumental stem, but it is not
the sole quality criterion for an aggressive karaoke/vocal-removal model. The
target for this candidate should instead explicitly prioritize:

1. removal of lead vocals, backing vocals, harmonies, and vocal-like effects;
2. retention of clearly non-vocal drums, bass, and accompaniment;
3. acceptable artifacts when vocal-like material overlaps with accompaniment.

The Inst 3 result is therefore not rejected as a teacher for this use case.
Its previous low-vocal instrumental penalty is a warning about collateral
removal, not an automatic failure. Future reports must separate `standard
instrumental fidelity` from `aggressive vocal-removal quality`.

## Evidence limits

This review is a user blind-listening result, but it was not collected with a
randomized ABX protocol and the three extra songs have no isolated reference
stems. It supports a new experimental direction and model-selection hypothesis;
it does not establish a quantitative quality claim.

## Next experiment

Keep the 128-frame student contract and train direct audio-domain target
mixtures:

```text
target(alpha) = (1 - alpha) * musdb18_instrumental + alpha * inst3_instrumental
alpha = 0.5, 0.75, 1.0
```

Use the same frozen four-song split, activity coverage, BatchNorm freeze, and
held-out evaluation. Run 128, 512, and 2048 update milestones at a controlled
learning rate, and render the three extra listening songs at each useful
milestone. Compare initial, Inst 3, and the alpha candidates on vocal residue,
backing-vocal removal, clearly non-vocal accompaniment retention, and audible
artifacts. Do not export a 24-frame model or publish derived weights until one
candidate is audibly distinguishable from initial and passes the practical
listening review.

All MUSDB18-derived audio, Inst 3-derived audio, and student checkpoints remain
local non-commercial research artifacts.
