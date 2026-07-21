# Third-party notices

The Apache-2.0 license in this repository applies to source code and
documentation authored for MusicSourceSeparation unless a file states
otherwise. It does not relicense third-party dependencies, binaries, model
weights, converted model artifacts, or audio files.

## Source and DSP reference

The MDX tensor layout, periodic Hann STFT/ISTFT behavior, trimming, and window
assembly were implemented with reference to
[`seanghay/uvr-mdx-infer`](https://github.com/seanghay/uvr-mdx-infer), licensed
under Apache-2.0. The local NumPy and Kotlin implementations are maintained in
this repository and retain this attribution.

## Runtime and build dependencies

The project directly or transitively uses the following software. Consult the
linked projects and resolved dependency artifacts for complete license text
and attribution requirements.

| Component | License |
| --- | --- |
| [Google LiteRT](https://github.com/google-ai-edge/LiteRT) | Apache-2.0 |
| [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT |
| [ONNX](https://github.com/onnx/onnx) | Apache-2.0 |
| [JTransforms](https://github.com/wendykierp/JTransforms) | BSD-2-Clause |
| [JLargeArrays](https://gitlab.com/ICM-VisLab/JLargeArrays) | BSD-2-Clause |
| [Apache Commons Math](https://commons.apache.org/proper/commons-math/) | Apache-2.0 |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause |
| [SciPy](https://github.com/scipy/scipy) | BSD-3-Clause |
| [Gradle](https://github.com/gradle/gradle) | Apache-2.0 |
| [JUnit 4](https://github.com/junit-team/junit4) | EPL-1.0 |

## Supplemental LiteRT x86 artifact

`app/libs/litert-2.1.5-x86.aar` is a historical benchmark artifact derived
from LiteRT 2.1.5. It is not original MusicSourceSeparation code and is not
relicensed by this repository. Its native payload is reproduced and released
with complete build provenance and resolved license material by
[`WluhWluh/bss-litert-android`](https://github.com/WluhWluh/bss-litert-android)
in release
[`v2.1.5-bss.1`](https://github.com/WluhWluh/bss-litert-android/releases/tag/v2.1.5-bss.1).

Use the canonical release's `LICENSE-LiteRT.txt`,
`THIRD_PARTY_LICENSES.txt`, `THIRD_PARTY_NOTICES.md`, and build manifest when
redistributing that native runtime.

## Model weights and audio

UVR/MDX ONNX and TFLite files are intentionally excluded from Git. Their
compatibility with this project does not grant permission to use,
redistribute, sublicense, or commercialize them. Each model remains governed
by the terms, provenance, and permissions associated with its source.

Source audio, generated stems, benchmark input tensors, and other user data
are also excluded from the repository and are not covered by its Apache-2.0
license.
