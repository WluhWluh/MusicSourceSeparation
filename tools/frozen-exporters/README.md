# Frozen HTDemucs exporters

These byte-exact snapshots preserve the exporter identities embedded in generated
LiteRT contracts and host manifests:

| SHA-256 | Contract |
| --- | --- |
| `a1229069ebee6e48d03507e730b3a4fa428067eba4222d75b3e6e5872bcf50c5` | 2-second six-stem smoke candidate |
| `532f5e1b7c9c30aa53963c891c94e3379f5aa18d0297d6ae21ce1e4d82e1b9ba` | canonical 7.8-second six-stem candidate |

The immutable reports retain their original execution path,
`tools/export_htdemucs_litert_candidate.py`. The current file at that path is a
later compatible exporter with derived-weight validation. The canonical freezer
uses the declared SHA to validate the matching snapshot while preserving the
historical path in the reproduced manifest.

For an exact historical rerun, use the Git commit that contains the required
snapshot at the original path. The archived copies are provenance inputs, not
independent command entry points.
