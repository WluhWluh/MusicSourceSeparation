# Frozen experiment tools

These snapshots preserve byte-identical tools named by completed experiment
evidence. They are provenance records, not current entry points.

| SHA-256 | Evidence-recorded name | Experiment |
| --- | --- | --- |
| `57735d1ed36d5b5fa61160233e873e84115febef19ac50e3680ab7cc6601521f` | `rewrite_identity_gather_nd.py` | Legacy six-stem append-shape `GATHER_ND -> RESHAPE` prototype |

The ignored raw manifest
`outputs/bandbuddy-s25-20260803/litert215-gather-rewrite/manifest.json`
records only the tool basename and SHA-256, not its absolute execution path.
The surviving untracked `tools/rewrite_identity_gather_nd.py` matched that SHA
exactly and is archived under its digest here; this does not invent a stronger
historical path claim than the evidence supports.

That prototype generated the 117,789,992-byte artifact
`htdemucs_6s.core.gather_nd_reshape_v1.tflite` (SHA-256
`199fd2c3a63fcbc840ab0be4b003730522b4b1503222f03a5f6d7e4a8a0c47e9`)
by repacking the object graph and appending 14 shape tensors. It is distinct
from the byte-size-preserving artifact used by the S25 device gates. The
current supported rewriter remains
`tools/rewrite_tflite_identity_gather_nd.py`.
