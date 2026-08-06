# HTDemucs-6s guitar-ft host experiment

Machine-readable analysis: `outputs/htdemucs6-guitar-ft-host-20260804/analysis.json`.

Tracks: **8**; total audio: **36.46 min**.

This corpus has no isolated ground truth. SNR below measures change from the official model, not separation quality.

| Track | Role | RTF | All-stem delta SNR (dB) | Guitar delta SNR (dB) | Guitar RMS change (dB) | Mix residual change (dB) |
|---|---|---:|---:|---:|---:|---:|
| john-lennon-imagine | piano-heavy | 0.318 | 7.44 | 0.57 | -14.40 | -8.59 |
| athletics-ii | guitar-heavy | 0.306 | 8.22 | 14.13 | +0.16 | -10.62 |
| kygo-ed-sheeran-i-see-fire-kygo-remix | guitar-and-electronic | 0.314 | 6.96 | 8.61 | +1.03 | -13.32 |
| sleeping-at-last-north | piano-heavy-low-guitar | 0.301 | 10.24 | 2.24 | -3.83 | -7.66 |
| sleeping-at-last-already-gone | piano-heavy-low-guitar | 0.313 | 9.52 | 7.13 | -0.24 | -3.55 |
| josiah-james-chasing-the-wind | guitar-present | 0.314 | 8.20 | 12.70 | +0.93 | -7.36 |
| joel-hanson-traveling-light | guitar-heavy | 0.314 | 9.42 | 15.80 | +0.50 | -9.67 |
| nylon-eventide | guitar-heavy | 0.308 | 5.11 | 18.47 | +0.19 | -18.27 |

Aggregate host separation RTF: `0.3101`.
Peak observed VmHWM: `1242.5 MiB`.
Non-finite outputs: `0`; clipped samples before PCM16: `95`.

Historical automated device-escalation gate: **FAILED**. Mixture reconstruction
SNR worsened on all eight tracks, from `-18.27` to `-3.55 dB`; no S25 smoke was
run in this host-screening stage.

These measurements alone cannot validate the publisher's claimed guitar SDR
gain. Subsequent blind listening supplied the missing perceptual evidence:
guitar and piano were repeatedly preferred for completeness and reduced
cross-talk, including substantially less audible vocal content in the Imagine
piano stem. A smaller vocals regression, heard as slight drum leakage, was also
repeatable. That evidence justified a bounded research-only S25 diagnostic; it
does not retroactively make the residual proxy pass. See
[`htdemucs6-guitar-ft-litert-s25-diagnostic-2026-08-05.md`](htdemucs6-guitar-ft-litert-s25-diagnostic-2026-08-05.md).

## Pinned provenance

- Hugging Face repository: `adityalakhani/htdemucs-6s-guitar-ft`.
- Revision: `163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad`.
- Training checkpoint: `guitar_htdemucs_6s.pt`, 329654071 bytes, SHA-256 `4fde369e41582ba5c2759b6ab926a44af467c64d4566bf914374ab267b19260e`.
- Extracted artifact: `htdemucs_6s_guitar_ft_fp32.safetensors`, 109716096 bytes, SHA-256 `e83f1e6ae8aaef5a177beaf6c6ff6bc3864c244dca123af7398eb1016a24fc6a`.
- Candidate manifest SHA-256: `b584473224a5ec8991eecda6f2c2fadf43eee27abce0a9f782e0442a9cc66930`.
- The checkpoint was loaded with `torch.load(weights_only=True)` and only `model_state_dict` was retained. All 525 keys and shapes match the official `5c90dfd2` HTDemucs-6s state; strict load passed. Stem order is `drums, bass, other, vocals, guitar, piano`.
- Checkpoint metadata records epoch `60`, validation loss `0.24393394`, and validation SDR `6.0798273 dB`; these are not the model-card guitar SDR claim and are not treated as a reproduced benchmark.

## Historical device decision

The host screen failed before any LiteRT conversion or S25 run. The screening thresholds were a maximum `3 dB` mixture-reconstruction-SNR worsening per track and a maximum `3 dB` absolute median RMS shift for each non-target stem. Observed mixture-reconstruction changes were `-18.27` to `-3.55 dB` on all eight tracks; non-target median RMS changes were drums `-3.89 dB`, bass `-7.71 dB`, other `-0.60 dB`, vocals `+0.03 dB`, and piano `-4.27 dB`. This was treated as a quality-screen failure, not a LiteRT conversion failure, so the initially planned S25 30-second CPU smoke was deliberately not executed in this stage. Later listening evidence changed the research decision, and the separately identified diagnostic was then run without relaxing this gate.

## Listening artifacts

- Blind audio: `outputs/htdemucs6-guitar-ft-host-20260804/blind`.
- Blind key JSON: `docs/htdemucs6-guitar-ft-blind-map-2026-08-04.json`.
- Blind key Markdown: `docs/htdemucs6-guitar-ft-blind-map-2026-08-04.md`.
- Reference-audio manifest with lossless FLAC-to-PCM identity checks: `outputs/htdemucs6-guitar-ft-host-20260804/reference-audio-manifest.json`.

## Licensing caution

The model card declares Apache-2.0 but the repository has no standalone LICENSE file. The card states that training used MoisesDB; the dataset terms are research/non-commercial and require separate legal review. This artifact is therefore retained as an internal, research-only experiment and is not a product-support candidate.

Source references: model card `https://huggingface.co/adityalakhani/htdemucs-6s-guitar-ft/blob/163ec83135ee06e6f10cb8cd94d2ecef8f3f34ad/README.md`; MoisesDB license `https://huggingface.co/datasets/wearemusicai/moisesdb/blob/3162378eb0653d9f9831a6051110822ab067b8b6/LICENSE`; Music AI research terms `https://music.ai/research/`.
