# Inst 3 teacher Oracle Stage 0.5

Status: complete for the local, non-commercial teacher baseline. The student
model has not been trained, and no derived audio, checkpoint, or teacher weight
has been published.

## Scope

This stage freezes a song-level MUSDB18 split and renders the UVR-MDX-NET
Inst 3 teacher for the complete calibration and internal-test songs. Each
selected song is rendered twice:

- `mixture-gt`: `vocals + drums + bass + other`, computed from decoded stems;
- `mixture-encoded`: the encoded MUSDB18 mixture stream, retained only to
  measure AAC/container input error.

The official MUSDB18 test split remains untouched for the final evaluation.
No selected window sampling is used.

## Frozen identities

| Item | Identity |
| --- | --- |
| Archive | `C:/Users/User/Downloads/musdb18.zip` |
| Dataset record | MUSDB18, Zenodo `1117372` |
| Split seed | `20260819` |
| Split | 80 train / 10 calibration / 10 internal-test / 50 final-test |
| Split algorithm | sort `SHA256(seed + NUL + archive member)` |
| Teacher contract | `uvr_mdxnet_inst_3@2` |
| Teacher ONNX SHA-256 | `2b7834e2972158d8c9864e7376e3a7d084079c80a23f38dc31c4b0a4e901a1cb` |
| Runner and manifest generator SHA-256 | `665cc378a3b4ff2194eba6a4aef430635427464ccad9934922479148db33ef5f` |
| Oracle git revision | `dcc60f3e6879e8b94514dfa784af5973d56135cb` |
| Manifest SHA-256 | `af8a9e8568772d0f84615f25caca0a098bce7c1e87ce22abcebe105814e42a50` |

The frozen DSP contract is 44.1 kHz, FFT 7680, hop 1024, `dimF=3072`, 256
model time frames, periodic Hann, trim 3840, and output scale `1.028`. The
MUSDB18 stream map is mixture 0, drums 1, bass 2, other 3, vocals 4. All 150
archive members have a source SHA-256 in the manifest.

## Execution evidence

The primary session was required to use CUDA and reported:

```text
CUDAExecutionProvider, CPUExecutionProvider
```

The host environment used PyTorch `2.11.0+cu128`, ONNX Runtime `1.26.0`,
Python `3.12.10`, and an NVIDIA GeForce RTX 4060 Laptop GPU. The selected 20
complete songs produced 40 unique cache keys and 1,842 teacher windows. No
teacher window failed, produced a non-finite result, or had a missing output
cache. The aggregate GPU render RTF was approximately `0.051` over about
1.454 hours of unique audio.

The first two calibration songs were rendered again with
`CPUExecutionProvider` for parity:

| Output | Maximum absolute error | Minimum SNR |
| --- | ---: | ---: |
| Instrumental | `5.923137e-4` | `71.11 dB` |
| Vocals/residual | `5.923137e-4` | `66.41 dB` |

The parity run also completed all CPU windows without failure. The CUDA
requirement was enforced; a silent CPU fallback was not accepted.

## Quality baseline

The following values are means across ten songs in each group. Leakage is the
projection of the estimated instrumental onto the reference vocal stem. The
low-vocal metrics use the lowest 20 percent of vocal-energy frames as an
instrument-preservation proxy.

| Split / input | Instrumental SI-SDR | Vocal leakage projection | High-vocal leakage projection | Low-vocal SI-SDR | Low-vocal error RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration / `mixture-gt` | 16.214 dB | -27.784 dB | -30.564 dB | 28.556 dB | -50.533 dBFS |
| Calibration / `mixture-encoded` | 16.024 dB | -27.783 dB | -30.567 dB | 25.037 dB | -47.061 dBFS |
| Internal-test / `mixture-gt` | 16.375 dB | -27.982 dB | -29.900 dB | 32.936 dB | -53.487 dBFS |
| Internal-test / `mixture-encoded` | 16.185 dB | -28.005 dB | -29.906 dB | 27.893 dB | -48.512 dBFS |

The encoded mixture differs from the exact stem sum by about `-50.2` to
`-50.7 dBFS` RMS on average. The overall instrumental score changes by only
about `0.19 dB`, while the low-vocal preservation proxy is more sensitive.
This is why `mixture-gt` remains the student-training target and the encoded
variant remains an evaluation control.

The runner records per-song peak, RMS level, residual/reconstruction error,
window-boundary error, and failure lists. MUSDB18 does not label backing vocals
or reverb tails separately, so those aspects are represented only by the
high-vocal, low-vocal, and boundary proxies here; they are not claimed as
independent objective measurements.

## Local artifact locations

The reproducible implementation is:

```text
tools/run_inst3_teacher_oracle.py
requirements-inst3-distill-pilot.txt
```

The run was made with an explicit durable ignored output root:

```powershell
python tools/run_inst3_teacher_oracle.py `
  --output-root data/musdb18-inst3-oracle `
  --manifest data/musdb18-inst3-oracle/musdb18-inst3-oracle-manifest.json `
  --require-teacher-cuda
```

The report and frozen split are:

```text
data/musdb18-inst3-oracle/reports/inst3-teacher-oracle-report.json
data/musdb18-inst3-oracle/musdb18-inst3-oracle-manifest.json
```

Decoded PCM, teacher outputs, and source extracts are local ignored artifacts.
The active CUDA environment is outside the repository at
`C:/Users/User/AppData/Local/MusicSourceSeparation/musdb18-inst3-oracle-venv`.

## Gate for the next stage

The operational teacher baseline is reproducible and complete. This does not
establish that Inst 3 distillation improves the current TFC-TDF candidate. The
next stage is the small instrumental student pilot: fixed song-level train,
calibration, and internal-test subsets; a ground-truth-only baseline; and a
ground-truth plus Inst 3 soft-target variant. It must remain separate from the
current vocals-primary checkpoint. Do not begin 24-frame export, TFLite
publication, QNN work, or `bss-tflite` upload until that comparison shows a
stable reduction in accompaniment vocal residue without unacceptable
instrument damage.
