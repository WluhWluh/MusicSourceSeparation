# Model Contracts

This document records model-specific assumptions that the Android implementation must eventually reproduce. It is intentionally separate from the general progress log because tensor contracts and DSP parameters need to stay precise.

## Candidate 1: UVR-MDX-NET-Inst_Main

Status: first local candidate

Local development path:

- `models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx`

Do not commit the model file. It is stored locally only for development and personal testing.

Source:

- Hugging Face mirror: `https://huggingface.co/seanghay/uvr_models`
- Reference implementation checked during research: `https://github.com/seanghay/uvr-mdx-infer`

License note:

- The app is currently for personal use only.
- Model redistribution rights are not confirmed.
- If this project is ever packaged for wider distribution, model licensing must be resolved before bundling weights.

ONNX graph:

- IR version: `6`
- Opset: `ai.onnx` version `13`
- Input name: `input`
- Input dtype: `float32`
- Input shape: `[batch_size, 4, 2048, 256]`
- Output name: `output`
- Output dtype: `float32`
- Output shape: `[batch_size, 4, 2048, 256]`

DSP parameters:

- Sample rate: `44100 Hz`
- Channels: stereo
- FFT size: `6144`
- Hop length: `1024`
- Frequency bins used by model: `2048`
- Full real FFT bins: `3073`
- Time frames per model window: `256`
- Window length in samples: `261120`
- Generation stride before trim: `254976`
- Trim per window: `3072` samples at each edge
- Window function: periodic Hann

Tensor layout:

- The model does not consume waveform samples directly.
- Each input item is a stereo STFT representation with four channels:
  - left real
  - left imaginary
  - right real
  - right imaginary
- Frequency bins are truncated from `3073` to `2048` before inference.
- Model output has the same layout and is reconstructed with inverse STFT.

Android implication:

- Phase 4 cannot just pass PCM into ONNX Runtime.
- Android needs a matching STFT/ISTFT implementation or a model variant with DSP folded into the graph.
- For the first Android ONNX integration, using this exact model means implementing periodic Hann windows, reflect padding, chunking, frequency padding, and overlap-add behavior.

