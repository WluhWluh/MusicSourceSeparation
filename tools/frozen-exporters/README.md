# Frozen HTDemucs exporters

These byte-exact snapshots preserve the exporter identities embedded in generated
LiteRT contracts and host manifests:

| SHA-256 | Contract |
| --- | --- |
| `a1229069ebee6e48d03507e730b3a4fa428067eba4222d75b3e6e5872bcf50c5` | 2-second six-stem smoke candidate |
| `532f5e1b7c9c30aa53963c891c94e3379f5aa18d0297d6ae21ce1e4d82e1b9ba` | canonical 7.8-second six-stem candidate |
| `2d0ee2f15cd6e3369a3adb2bde0202aa19311d6ccda1159cf1ac4947e6cfd1c1` | guitar-ft variant wrapper used by the research diagnostic |

The immutable reports retain their original execution path,
`tools/export_htdemucs_litert_candidate.py`. The current files at the historical
paths are later compatible revisions. The freezers use each declared SHA to
validate the matching snapshot while preserving the historical path in
reproduced manifests.

For an exact historical rerun, use the Git commit that contains the required
snapshot at the original path. The archived copies are provenance inputs, not
independent command entry points.
