# F24-H50 Assembly Listening Review

Status: completed locally on 2026-08-23. This records the user's listening
review and the matched signal comparison; it is not a product quality claim.

## Human Review

The twelve-song comparison used:

- `F24-static-H50`: H50 weights, 24 frames, four hops of context, continuous
  overlap-save, accompaniment output;
- `H50-128-isolated`: H50 weights, 128 frames, isolated zero-padded windows,
  accompaniment output from the retained earlier renderer;
- `H50-128-continuous`: the same H50 weights and 128-frame contract, rendered
  with continuous overlap-save, accompaniment output.

The user reported:

1. `F24-static-H50` sometimes has a short, audible accompaniment-quality
   loss, although some abrupt vocal-leak points present in the old 128-frame
   files disappear.
2. `F24-Inst3-U@pass-10`, `F24-Inst3-H25@pass-10`, and the original static
   F24 export are audibly worse than untrained `F24-static-H50`.
3. `H50-128-continuous` sounds clearly better than the old 128-frame files.

The resulting interpretation is that the apparent F24 advantage was largely
caused by the old 128-frame reference using isolated zero-padding. It should
not be treated as evidence that shortening the neural context improved the
separation model.

## Matched Signal Evidence

The retained 128-frame isolated files were re-rendered from the same H50
checkpoint. The verification SNR between the retained files and the
re-rendered isolated control ranged from approximately `40.9` to `85.9 dB`
across the twelve songs, confirming that the old files represent the intended
isolated assembly rather than corrupted outputs.

All three paths were compared as accompaniment tracks. Twelve-song means:

| Pair | Mean direct SNR | Mean correlation | Mean difference RMS |
| --- | ---: | ---: | ---: |
| F24-static-H50 vs H50-128-continuous | 19.34 dB | 0.9934 | -33.36 dBFS |
| H50-128-continuous vs H50-128-isolated | 26.01 dB | 0.9986 | -40.04 dBFS |
| F24-static-H50 vs H50-128-isolated | 18.88 dB | 0.9927 | -32.90 dBFS |

The continuous 128-frame output is therefore materially closer to the old
H50 result than F24 is, while also being the version the user prefers by
listening. This isolates the assembly change as the dominant explanation for
the earlier perceived leak reduction.

Relative to the isolated H50 accompaniment, the whole-song residual-energy
change averaged `+0.192 dB` for F24 and `-0.062 dB` for 128-frame continuous.
The F24 path therefore changes the removed-content energy more than the
continuous control, consistent with the reported accompaniment damage. This
metric cannot identify whether the changed content is vocal or instrumental
without private-song stems; it is supporting evidence, not a separation
score.

## Decision

Use `H50-128-continuous` as the fair current H50 listening baseline. Treat the
old isolated 128-frame files as a historical assembly variant, not as the
primary reference for judging F24. Do not promote F24-static-H50 solely on
the basis of the earlier A/B impression, and do not continue F24 retraining or
F16 work until a model is shown to improve over this continuous baseline.

The comparison report and generated 128-frame continuous accompaniment files
are under:

`data/musdb18-inst3-f24-h50-comparison/`

The direct comparison tool now labels all analyzed variants as accompaniment
tracks and records the assembly control explicitly.
