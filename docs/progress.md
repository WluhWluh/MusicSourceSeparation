# Music Source Separation Android MVP Progress

## Project Goal

Build a simple Android app for personal daily use that performs fully on-device 2-stem music source separation. The user should install the app, choose a local audio file, wait for processing, and receive two local output files:

- `vocals.wav`
- `instrumental.wav`

The MVP must avoid server-side processing, account setup, manual model configuration, command-line usage, and complicated environment setup for the end user.

## Current Status

Date: 2026-06-14

The workspace has been prepared as the project root. A minimal Kotlin Android shell now builds, installs, and launches on the local emulator. The app currently provides a file-selection entry point, audio metadata reading, and a verified pass-through WAV export path. A desktop MDX ONNX reference scaffold has been added for model contract inspection and short WAV separation tests. Android model inference is not implemented yet.

## Local Development Environment Check

Workspace:

- Path: `C:\Users\User\Documents\MusicSourceSeparation`
- Existing project files before setup: none detected
- Git repository before setup: not initialized
- Git repository after setup: initialized on branch `main`

Available tools:

- Git: available, `2.54.0.windows.1`
- Java: available, Temurin OpenJDK `21.0.11`
- `JAVA_HOME`: set to `C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot\`
- Python: available, `3.14.5`
- Node.js: available, `24.16.0`
- npm: available, `11.13.0`
- Android Studio: installed at `C:\Program Files\Android\Android Studio`

Android SDK:

- SDK path detected: `C:\Users\User\AppData\Local\Android\Sdk`
- Installed platforms: `android-36.1`, `android-37.0`
- Installed build tools: `36.0.0`, `36.1.0`, `37.0.0`
- ADB exists at `C:\Users\User\AppData\Local\Android\Sdk\platform-tools\adb.exe`
- ADB version: `37.0.0-14910828`

Missing or not configured:

- `ANDROID_HOME` is not set.
- `ANDROID_SDK_ROOT` is not set.
- `adb` is not available on `PATH`, although it exists inside the SDK.
- `gradle` is not available globally on `PATH`.
- Android command-line tools directory was not detected.
- Android NDK was not detected.
- Android SDK CMake was not detected.
- Ninja was not detected.

Cached Gradle assets:

- Gradle wrapper distribution cache includes `gradle-9.4.1-bin`.
- Android Gradle Plugin cache includes `com.android.tools.build:gradle:9.2.1`.

Environment implication:

- A Kotlin/Android app scaffold should be feasible once Gradle wrapper files are added to the repository.
- Native C++ work for audio DSP or ONNX Runtime custom integration will require installing or configuring Android NDK, CMake, and Ninja.
- Android SDK environment variables should be set or replaced with a checked-in `local.properties` file when the Android project is created.

## MVP Product Scope

In scope:

- Android-only app.
- Fully offline, fully on-device processing.
- Local audio file selection through the Android system file picker.
- 2-stem output only: vocals and instrumental.
- Progress display with cancel support.
- Save output files locally.
- Basic in-app playback or system share/open action after completion.
- A single default separation quality mode.

Out of scope for the first MVP:

- 4-stem separation.
- Real-time separation.
- Batch processing.
- Cloud upload or account sync.
- Manual model selection.
- Advanced DSP settings.
- Desktop GUI.
- iOS support.
- Commercial distribution readiness.

## Recommended Technical Direction

Model strategy:

- Start with an MDX/UVR-style 2-stem ONNX model instead of porting Demucs first.
- Use Demucs as a quality benchmark and possible future high-quality mode.
- Keep the model engine behind a small interface so the implementation can later switch between ONNX Runtime, LiteRT, or another runtime.

Initial runtime strategy:

- Use ONNX Runtime Android for the first working prototype.
- Start with CPU execution to reduce device-specific accelerator risk.
- Benchmark real devices before investing in GPU, NPU, NNAPI, or vendor-specific acceleration.

Audio pipeline:

1. Select an audio file with Android Storage Access Framework.
2. Decode audio with Android media APIs into PCM.
3. Convert to the model sample rate and stereo format.
4. Process audio in chunks with overlap.
5. Run model inference per chunk.
6. Reconstruct separated audio with overlap-add.
7. Apply simple limiter or clipping protection.
8. Save `vocals.wav` and `instrumental.wav`.

Output strategy:

- Use WAV for the first MVP because it is simple and avoids early encoder/container issues.
- Add AAC/M4A export later if WAV file size becomes inconvenient.

## Development Roadmap

### Phase 0: Repository and Planning

Status: done

Tasks:

- Initialize the workspace as a Git repository.
- Create `docs/progress.md`.
- Keep project notes and implementation comments in English.
- Record the local development environment state.

Done criteria:

- Git repository exists.
- Planning document exists.
- Current toolchain gaps are documented.

### Phase 1: Android Project Scaffold

Status: done

Tasks:

- Create a minimal Android project using Kotlin.
- Use dependency-light native Android Views for the first scaffold.
- Add Gradle wrapper files.
- Configure SDK path through `local.properties` or environment setup.
- Add a basic app screen with file selection entry point.
- Build and run an empty app on emulator or device.

Done criteria:

- `./gradlew assembleDebug` works.
- App installs and opens on the local emulator.
- Main screen renders correctly.

Implementation notes:

- Android Gradle Plugin: `9.2.1`
- Gradle wrapper: `9.4.1`
- Kotlin support: AGP 9 built-in Kotlin support
- Compile SDK: `37`
- Minimum SDK: `26`
- UI approach: a single native Android `Activity` with a system audio picker entry point
- Debug APK: `app/build/outputs/apk/debug/app-debug.apk`
- Debug APK SHA-256: `F30EFD2B0835B9D95F82B08ED297C33FE708810E0313BEAB1B024381D9F274D0`
- Verification: `assembleDebug` succeeded, APK installed successfully on `emulator-5554`, and the main screen launched without visible layout issues.

### Phase 2: File Selection and Audio I/O

Status: done

Tasks:

- Add Android Storage Access Framework file selection. Done.
- Accept common local audio formats such as MP3, M4A, AAC, WAV, and FLAC where platform codecs support them. Initial MP3 path verified.
- Decode selected audio into PCM. Done for the verified MP3 path.
- Implement a WAV writer. Done.
- Create a pass-through test path that writes the decoded audio back to a WAV file. Done.

Done criteria:

- User can select a local audio file.
- App can decode it without model inference.
- App can save a valid WAV output file.

Implementation notes:

- Metadata reader: `AudioMetadataReader`
- PCM decoder: `AudioPassthroughExporter`
- WAV writer: `WavFileWriter`
- Decoder APIs: Android `MediaExtractor` and `MediaCodec`
- Output location: app-specific external music directory under `exports`
- Debug APK SHA-256 after Phase 2 partial work: `9A6F1DD4C927EDC68246A882D54F8CAAAE93BED19B50B4BA4B5B839995729D97`
- Verification: `assembleDebug` succeeded, APK installed successfully on `emulator-5554`, and the updated empty-state UI launched without visible layout issues.
- User verification: an MP3 file from the emulator Downloads directory was selected in the app, exported to WAV, moved back to Downloads, played successfully, and had the expected duration.

### Phase 3: Desktop Reference Model Pipeline

Status: in progress

Tasks:

- Select 2 or 3 candidate ONNX 2-stem models. First candidate selected.
- Build a Python reference script for inference. Initial scaffold added.
- Fix the expected sample rate, channel layout, chunk size, overlap, normalization, and output gain. Initial MDX contract documented.
- Generate short golden test outputs for later Android comparison. Pending real music sample.

Done criteria:

- Reference inference works on a short audio sample.
- Candidate model performance and quality are compared.
- Input and output tensor contracts are documented.

Implementation notes:

- First candidate: `UVR-MDX-NET-Inst_Main.onnx`
- Local model path: `models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx`
- Model file size: `52786726` bytes
- Model files are ignored by Git and must not be committed.
- Python environment: project-local `.venv`
- Dependency lock for this phase: `requirements-phase3.txt`
- Reference script: `tools/mdx_reference.py`
- Contract document: `docs/model_contracts.md`
- Verified ONNX input: `float32 [batch_size, 4, 2048, 256]`
- Verified ONNX output: `float32 [batch_size, 4, 2048, 256]`
- Smoke tests completed:
  - `python tools/mdx_reference.py self-test`
  - `python tools/mdx_reference.py inspect --model models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx --smoke-run`
  - `python tools/mdx_reference.py separate --model models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx --input data/smoke/synthetic_mix.wav --output-dir outputs/reference --limit-seconds 3 --no-denoise`
- Synthetic smoke output verification: vocals and instrumental WAV files were both `44100 Hz`, stereo, `3.0` seconds, and non-empty.
- Real audio reference test:
  - Source: app-exported WAV pulled from emulator Downloads, stored locally as `data/samples/_-_Coast_Town__decoded.wav`
  - Command: `python tools/mdx_reference.py separate --model models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx --input data/samples/_-_Coast_Town__decoded.wav --output-dir outputs/reference/coast_town_20s --limit-seconds 20 --no-denoise`
  - Runtime on local desktop: about `7.05` seconds for a `20.0` second excerpt
  - Outputs: vocals and instrumental WAV files were both `44100 Hz`, stereo, `20.0` seconds, and non-empty
  - Emulator playback files pushed to Downloads:
    - `/sdcard/Download/coast_town_20s_reference_vocals.wav`
    - `/sdcard/Download/coast_town_20s_reference_instrumental.wav`
- Important finding: this MDX model consumes STFT tensors, not raw PCM waveform samples. Android Phase 4 will need matching STFT/ISTFT DSP or a model variant with preprocessing folded into ONNX.

### Phase 4: Android ONNX Runtime Integration

Status: pending

Tasks:

- Add ONNX Runtime Android dependency.
- Bundle one default model with the app or load it from app assets.
- Implement inference on a short test tensor.
- Connect PCM chunks to model input.
- Compare Android output against the desktop reference output.

Done criteria:

- ONNX inference runs locally on Android.
- Short sample separation completes successfully.
- Output difference from reference pipeline is understood and acceptable.

### Phase 5: Chunked Full-Song Processing

Status: pending

Tasks:

- Implement chunk scheduling for full songs.
- Add overlap-add reconstruction.
- Track progress by processed chunk count.
- Add cancel behavior.
- Avoid loading the whole song and all intermediate tensors into memory at once.

Done criteria:

- A typical 3-minute song can be processed without running out of memory.
- Progress updates are visible.
- Cancel stops processing cleanly.

### Phase 6: Long-Running Task UX

Status: pending

Tasks:

- Run separation as a long-running foreground operation.
- Keep the app responsive during processing.
- Show elapsed time and estimated remaining time if reliable enough.
- Preserve partial state safely on interruption.
- Avoid producing broken output files after failure or cancellation.

Done criteria:

- Processing can continue while the screen is off.
- User sees a clear processing notification.
- Failure, cancellation, and completion states are handled clearly.

### Phase 7: Device Benchmarking

Status: pending

Tasks:

- Test at least one real Android device.
- Measure runtime factor, peak memory, output quality, thermal behavior, and battery impact.
- Test 30-second, 3-minute, and 10-minute inputs.
- Record failures and unsupported codecs.

Target MVP acceptance:

- A 3-minute song completes on a mid-range phone in a tolerable time.
- Peak memory remains below a practical safety limit.
- The app does not crash on normal inputs.
- Output quality is good enough for personal listening and practice use.

### Phase 8: Polish for Personal Use

Status: pending

Tasks:

- Improve output file naming.
- Add simple result playback.
- Add share/open actions.
- Add a small settings screen only if needed.
- Add clear error messages for unsupported files, insufficient storage, or interrupted processing.

Done criteria:

- The app is usable without developer intervention.
- The main workflow is simple enough for daily personal use.

## Immediate Next Steps

1. Listen to `/sdcard/Download/coast_town_20s_reference_vocals.wav` and `/sdcard/Download/coast_town_20s_reference_instrumental.wav`.
2. Record subjective output quality notes for `UVR-MDX-NET-Inst_Main.onnx`.
3. Add one additional candidate model for comparison if licensing and download path are clear.
4. Decide whether Android Phase 4 should implement MDX STFT/ISTFT in Kotlin first or introduce a native DSP layer.
5. Start Android ONNX Runtime integration once the first model choice is acceptable.

## Open Technical Decisions

- Whether the first model should be bundled in the APK/AAB or downloaded once into app-private storage.
- Whether the first implementation should use pure Kotlin plus ONNX Runtime Java APIs or introduce C++ early for audio DSP.
- Whether the first WAV writer should be Kotlin-only or native.
- Whether the native Android Views scaffold should later move to Jetpack Compose.
- Which real test device should be used for performance targets.

## Risk Register

- Model license risk: third-party source separation weights may not have clear redistribution or commercial-use terms.
- Performance risk: desktop inference speed does not predict phone inference speed.
- Thermal risk: long CPU inference may slow down after sustained processing.
- Memory risk: careless chunking can still create large intermediate buffers.
- Codec risk: Android platform codec support varies by device and file container.
- Native tooling risk: ONNX Runtime and future DSP optimization may require NDK/CMake setup.
