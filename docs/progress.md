# Music Source Separation Android MVP Progress

## Project Goal

Build a simple Android app for personal daily use that performs fully on-device 2-stem music source separation. The user should install the app, choose a local audio file, wait for processing, and receive two local output files:

- `vocals.wav`
- `instrumental.wav`

The MVP must avoid server-side processing, account setup, manual model configuration, command-line usage, and complicated environment setup for the end user.

## Current Status

Date: 2026-06-14

The workspace has been prepared as the project root. The Android app now provides local audio selection, metadata reading, WAV export, Android ONNX Runtime inference, and 2-stem MDX separation for custom ranges or full tracks. User testing confirmed full-song separation quality on the Coast Town and Brave Heart tracks. The current stage focuses on robustness for personal device testing: automatic `44100 Hz` model-rate resampling, MDX-style context windows, and a debug APK that bundles the local ONNX model asset for direct installation.

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
- Real audio reference retest after offset correction:
  - Added `--start-seconds` support to `tools/mdx_reference.py`
  - Command: `python tools/mdx_reference.py separate --model models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx --input data/samples/_-_Coast_Town__decoded.wav --output-dir outputs/reference/coast_town_37s_20s --start-seconds 37 --limit-seconds 20 --no-denoise`
  - Runtime on local desktop: about `7.18` seconds for a `20.0` second excerpt
  - Outputs: vocals and instrumental WAV files were both `44100 Hz`, stereo, `20.0` seconds, and non-empty
  - Emulator playback files pushed to Downloads:
    - `/sdcard/Download/coast_town_37s_20s_reference_vocals.wav`
    - `/sdcard/Download/coast_town_37s_20s_reference_instrumental.wav`
  - This is the more meaningful comparison region because the vocal entry begins around `37s`.
- Output quality note:
  - User confirmed the corrected vocal-entry segment separates vocals and instruments very cleanly.
  - The first reference script version mislabeled the stems.
  - For `UVR-MDX-NET-Inst_Main.onnx`, the ONNX output is instrumental, and vocals are `mixture - instrumental`.
- Important finding: this MDX model consumes STFT tensors, not raw PCM waveform samples. Android Phase 4 will need matching STFT/ISTFT DSP or a model variant with preprocessing folded into ONNX.

### Phase 4: Android ONNX Runtime Integration

Status: in progress

Tasks:

- Add ONNX Runtime Android dependency. Done.
- Bundle one default model with the app or load it from app assets. Initial local app-private loader done.
- Implement inference on a short test tensor. Done.
- Connect PCM chunks to model input.
- Compare Android output against the desktop reference output.

Done criteria:

- ONNX inference runs locally on Android.
- Short sample separation completes successfully.
- Output difference from reference pipeline is understood and acceptable.

Implementation notes:

- First Android dependency choice: `com.microsoft.onnxruntime:onnxruntime-android:1.26.0`
- Goal for the first Android inference step: load the bundled model and run a zero-tensor smoke test before wiring the audio pipeline.
- The current Android app already handles file selection, metadata reading, and WAV export; this phase will add model loading and inference on top of that baseline.
- Local model load path for development: `/data/user/0/com.example.musicsourceseparation/files/UVR-MDX-NET-Inst_Main.onnx`
- Current debug APK size after ONNX Runtime integration: about `115 MB`
- Current debug APK SHA-256: `271B42D761C3B75E504F1B87A084810DC26DAB7858824D7A0E159DA79335EE64`
- Android ONNX smoke test result on `emulator-5554`:
  - Input name: `input`
  - Output name: `output`
  - Output shape: `[1, 4, 2048, 256]`
  - Output absolute mean: `1.8588207`
  - This matches the desktop Python zero-input smoke test closely enough for Phase 4.
- Android DSP scaffold:
  - `MdxDspConfig`
  - `MdxSpectrogram`
  - JVM test: `MdxSpectrogramTest`
  - FFT dependency: `com.github.wendykierp:JTransforms:3.1`
  - Validation: `testDebugUnitTest` and `assembleDebug` both passed
  - The next step is to wire these DSP classes into the actual Android audio pipeline and then into ONNX inference for a real short excerpt.
- Audio pipeline preparation:
  - Extracted `AudioPcmDecoder` from `AudioPassthroughExporter`
  - Added `DecodedPcmAudio` as an in-memory PCM container
  - WAV pass-through export now reuses the shared PCM decoder
  - Validation: `testDebugUnitTest`, `assembleDebug`, APK install, and launch check passed
- One-window separation prototype:
  - Added `MdxOneWindowSeparator`
  - Added a development UI button: `Separate one window`
  - Added a development UI button: `Separate 37s window`
  - The prototype decodes selected audio, resamples it to the MDX model rate when needed, uses one MDX-sized window, runs STFT, ONNX inference, ISTFT, then writes vocals and instrumental WAV files.
  - Current limitation: this is one window only, not full-song chunking.
  - Validation so far: `testDebugUnitTest`, `assembleDebug`, APK install, and model-file presence check passed.
  - Prepared a reusable vocal-entry test sample: `data/samples/coast_town_vocal_entry_37s_12s.wav`
  - Pushed the same sample to emulator Downloads for manual validation.
  - User verification: `coast_town_vocal_entry_37s_12s.wav` produced clean vocals and instrumental output.
- Range separation UI:
  - Added editable start and end time inputs in minutes, seconds, and milliseconds.
  - Default selection is the full audio duration after import.
  - Added a development button: `Separate range`
  - Added `MdxRangeSeparator` for sequential windowed range processing.
  - Range separation now decodes selected audio, resamples to `44100 Hz` for MDX when needed, processes each window with `trim` samples of context, advances by `generationSize`, and writes only the stable center region.
  - Separated WAV outputs are written at `44100 Hz`.
  - End time input now treats `0` as "to the end of the imported file" so the default full-range selection remains safe even when metadata duration is unavailable.
  - Validation so far: `testDebugUnitTest`, `assembleDebug`, and emulator reinstall passed after the UI and range-separation additions.
  - User verification: full-track range separation was tested on Coast Town and Brave Heart, and both outputs sounded correct.
- Model packaging for personal device testing:
  - Debug APK builds now include the local ignored model file from `models/uvr-mdx/UVR-MDX-NET-Inst_Main.onnx` as an Android asset when that file exists locally.
  - At runtime, `MdxModelFile` copies the bundled asset into app-private storage before ONNX Runtime opens the model.
  - This keeps the direct-install Samsung S25 test path simple while still keeping model weights out of Git.
  - Verification: `testDebugUnitTest` and `assembleDebug` passed after the resampling, context-window, and model-asset updates.
  - APK asset check: `app/build/outputs/apk/debug/app-debug.apk` contains `assets/UVR-MDX-NET-Inst_Main.onnx`.
  - Current bundled debug APK size: `166725001` bytes.
  - Current bundled debug APK SHA-256: `C4050084246D0189A5290012C3F8BC35916C838130DB56475E367BBDA3F241B7`.
  - Personal-test copy: `dist/MusicSourceSeparation-s25-debug.apk`.
  - Emulator verification: installing `dist/MusicSourceSeparation-s25-debug.apk` with `adb install -r` succeeded, and `MainActivity` launched and became the focused window on `emulator-5554`.

### Phase 5: Chunked Full-Song Processing

Status: in progress

Tasks:

- Implement chunk scheduling for full songs. Done for sequential MDX range windows.
- Use MDX-style context windows and stable center writes. Done.
- Track progress by processed chunk count. Done.
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

Status: in progress

Tasks:

- Test at least one real Android device. Done on Samsung S25.
- Measure runtime factor, peak memory, output quality, thermal behavior, and battery impact. In progress.
- Test 30-second, 3-minute, and 10-minute inputs.
- Record failures and unsupported codecs.

Implementation notes:

- Samsung S25 user verification: all current app functions work correctly, and separated output quality remains good across several songs.
- Initial S25 performance observation: the Coast Town track that took about `150s` on the desktop Android emulator took about `268s` on the Samsung S25.
- This confirms that device benchmarking must separate desktop-emulator CPU behavior from sustained real-phone CPU behavior.

Short-term performance plan:

1. Add range-separation timing instrumentation for decode, resample, model/session setup, STFT, ONNX inference, ISTFT, PCM conversion, and WAV writing. Done.
2. Build an S25 APK with the timing report visible in-app and saved next to the separated stems. Done for the first timing-test APK.
3. Use the S25 timing report to identify whether the first optimization target should be ONNX Runtime inference, Kotlin DSP, audio decode/resample, or output writing. Done: ONNX Runtime inference dominates.
4. Sweep ONNX Runtime CPU session settings such as graph optimization level and intra-op thread count after a timing baseline exists. In progress.
5. Test ONNX Runtime XNNPACK as the first low-risk execution-provider experiment.
6. Consider NNAPI, Qualcomm QNN, FP16 conversion, or quantization only after the baseline and CPU/XNNPACK results justify the extra complexity.

Timing instrumentation notes:

- `MdxRangeSeparator` now reports decode, resample, model-file preparation, output setup, ONNX session setup, window input, STFT, tensor creation, ONNX inference, output flattening, ISTFT, stem subtraction, PCM conversion, WAV writing, and control overhead.
- The same timing report is shown in the app after range separation and written as a `_timing.txt` file next to the separated WAV files.
- Verification: `testDebugUnitTest` and `assembleDebug` passed after timing instrumentation.
- Timing-test APK: `dist/MusicSourceSeparation-s25-timing-debug.apk`.
- Timing-test APK size: `166725001` bytes.
- Timing-test APK SHA-256: `DE2F60ACC804B8C0EF1A6D93F7DB8C10E92427BE222ED4E8E39688989A86CEF4`.
- Timing-test APK asset check: contains `assets/UVR-MDX-NET-Inst_Main.onnx`.
- Emulator verification: installing `dist/MusicSourceSeparation-s25-timing-debug.apk` with `adb install -r` succeeded, and `MainActivity` launched and became the focused window on `emulator-5554`.
- S25 Coast Town timing baseline with ORT default CPU settings: `273.70s` audio, `48` windows, `286.46s` total, `1.05x` audio duration, `5.97s/window`.
- S25 baseline stage split: decode `2.38s`, STFT `5.96s`, ONNX inference `264.10s`, ISTFT `10.29s`, PCM convert `2.09s`; ONNX inference accounts for `92.20%` of total time.

ONNX CPU settings experiment:

- Added `MdxRuntimeSettings` with configurable CPU intra-op thread count.
- Range separation now uses `ALL_OPT`, `SEQUENTIAL` execution, `interOp=1`, and user-provided `intraOp` when CPU threads is greater than `0`.
- The app exposes `ONNX CPU threads (0 = default)` on the main screen.
- Each timing report records the runtime settings used for that run.
- Verification: `testDebugUnitTest` and `assembleDebug` passed after the runtime-settings update.
- CPU thread-test APK: `dist/MusicSourceSeparation-s25-cpu-threads-debug.apk`.
- CPU thread-test APK size: `166725001` bytes.
- CPU thread-test APK SHA-256: `B7F34A09C45590C4B919A19E7BEC3FCFAFE5C253838F487A905C5761DAB0480F`.
- CPU thread-test APK asset check: contains `assets/UVR-MDX-NET-Inst_Main.onnx`.
- Emulator verification: installing `dist/MusicSourceSeparation-s25-cpu-threads-debug.apk` with `adb install -r` succeeded, and `MainActivity` launched and became the focused window on `emulator-5554`.
- S25 CPU thread sweep results for Coast Town:
  - `8` threads, cold phone: `165.31s` total, `142.88s` ONNX inference, `0.60x` audio duration.
  - `8` threads, hot phone after consecutive runs: `239.69s` total, `208.01s` ONNX inference, `0.88x` audio duration.
  - `6` threads, hot phone: `285.24s` total, `255.78s` ONNX inference, `1.04x` audio duration.
  - `4` threads, hot phone: `347.12s` total, `322.81s` ONNX inference, `1.27x` audio duration.
  - `2` threads, hot phone: `346.12s` total, `321.80s` ONNX inference, `1.26x` audio duration.
  - `1` thread, hot phone: `407.08s` total, `389.65s` ONNX inference, `1.49x` audio duration.
  - ORT default, hot phone: `341.83s` total, `316.14s` ONNX inference, `1.25x` audio duration.
- CPU thread sweep conclusion: `8` threads is the fastest tested setting even after thermal throttling, but hot-phone performance drops substantially from the cold-phone result.
- Next experiment: test XNNPACK with explicit thread counts, starting with `8` threads cold and hot, then compare against CPU-only `8` threads.

XNNPACK experiment:

- Added a `Use XNNPACK` checkbox to the main screen.
- When XNNPACK is enabled, ORT intra-op threads are set to `1`, ORT spinning is disabled, and the CPU thread input is passed to XNNPACK as `intra_op_num_threads`.
- Each timing report records whether the run used CPU-only or XNNPACK runtime settings.
- Verification: `testDebugUnitTest` and `assembleDebug` passed after adding the XNNPACK toggle.
- XNNPACK test APK: `dist/MusicSourceSeparation-s25-xnnpack-debug.apk`.
- XNNPACK test APK size: `166725001` bytes.
- XNNPACK test APK SHA-256: `47ABC320C28306B3D3CB50398BA89BB5F66E683E3190096B9FDF3B278DE39ABD`.
- XNNPACK test APK asset check: contains `assets/UVR-MDX-NET-Inst_Main.onnx`.
- Emulator verification: installing `dist/MusicSourceSeparation-s25-xnnpack-debug.apk` with `adb install -r` succeeded, and `MainActivity` launched and became the focused window on `emulator-5554`.
- S25 sequential XNNPACK comparison from a cool phone:
  - Round 1, XNNPACK `8` threads: `148.36s` total, `131.29s` ONNX inference, `0.54x` audio duration.
  - Round 2, CPU-only `8` threads: `145.71s` total, `125.79s` ONNX inference, `0.53x` audio duration.
  - Round 3, XNNPACK `8` threads on warmer phone: `176.82s` total, `157.00s` ONNX inference, `0.65x` audio duration.
  - Round 4, CPU-only `8` threads on hot phone: `235.79s` total, `202.63s` ONNX inference, `0.86x` audio duration.
- XNNPACK conclusion: XNNPACK works, but CPU-only `8` threads remains the best tested path for this MDX model on the S25. XNNPACK may retain better hot-phone behavior than CPU-only in some run orders, but the best practical default is CPU-only `8` threads until more devices or repeated alternating runs justify changing it.
- Default runtime update: the main-screen CPU thread input now defaults to `8`, with `Use XNNPACK` unchecked by default.
- Default-8 APK: `dist/MusicSourceSeparation-s25-default8-debug.apk`.
- Default-8 APK size: `166725001` bytes.
- Default-8 APK SHA-256: `EF24A29024A6BA78463823EC1CF5576942B2C1698A34AE1377392EE96D8615EC`.
- Default-8 APK asset check: contains `assets/UVR-MDX-NET-Inst_Main.onnx`.
- Emulator verification: installing `dist/MusicSourceSeparation-s25-default8-debug.apk` with `adb install -r` succeeded, and `MainActivity` launched and became the focused window on `emulator-5554`.

Small MDX model experiment:

- Added `UVR_MDXNET_9482.onnx` as a second bundled model while keeping `UVR-MDX-NET-Inst_Main.onnx` unchanged.
- Local ignored model path: `models/uvr-mdx/UVR_MDXNET_9482.onnx`.
- Model source: `https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx`.
- Local model size: `29704738` bytes.
- Local model SHA-256: `9D78F8566FA8198065214AB628BE1DE966A500C57786695AA4B13E2B27A7727D`.
- Desktop contract inspection passed: input and output are both `float32 [batch_size, 4, 2048, 256]`, opset `13`, and the zero-tensor smoke run returned `float32 [1, 4, 2048, 256]`.
- Android now exposes a model dropdown with `UVR-MDX-NET Inst Main` and `UVR MDXNET 9482`.
- Range separation, one-window separation, and ONNX smoke test all use the selected model.
- Output file names include the selected model tag to avoid mixing results from different models.
- Per-model output stem semantics are now encoded in `MdxModelVariant`:
  - `UVR-MDX-NET Inst Main` model output is treated as instrumental, and vocals are computed as `mixture - instrumental`.
  - `UVR MDXNET 9482` model output is treated as vocals, and instrumental is computed as `mixture - vocals`.
- Verification: `testDebugUnitTest` and `assembleDebug` passed after adding model selection.
- APK asset check: `app/build/outputs/apk/debug/app-debug.apk` contains both `assets/UVR-MDX-NET-Inst_Main.onnx` and `assets/UVR_MDXNET_9482.onnx`.
- Current two-model debug APK size: `194397049` bytes.
- Current two-model debug APK SHA-256: `65DEAABDC5175D407410D7D12DEE1982DBA58156A10E705D17672889BDC094BF`.
- Two-model test APK: `dist/MusicSourceSeparation-s25-two-mdx-models-debug.apk`.
- Two-model test APK size: `194397049` bytes.
- Two-model test APK SHA-256: `65DEAABDC5175D407410D7D12DEE1982DBA58156A10E705D17672889BDC094BF`.
- Two-model test APK asset check: contains both `assets/UVR-MDX-NET-Inst_Main.onnx` and `assets/UVR_MDXNET_9482.onnx`.
- Emulator verification: installing `dist/MusicSourceSeparation-s25-two-mdx-models-debug.apk` with `adb install -r` succeeded, and `MainActivity` launched and became the focused window on `emulator-5554`.
- S25 test result: `UVR MDXNET 9482` reduced total processing time to about `25%` of source duration, more than twice as fast as the previous model, while still sounding good enough to keep.
- S25 timing shift: `ONNX inference` share dropped from about `90%` with `UVR-MDX-NET Inst Main` to about `75%` with `UVR MDXNET 9482`.
- S25 labeling finding: `UVR MDXNET 9482` has opposite output stem semantics from `Inst Main`; initial files were mislabeled until the per-model stem mapping was added.
- Stem-labeling fix APK: `dist/MusicSourceSeparation-s25-stemfix-debug.apk`.
- Stem-labeling fix APK size: `194397049` bytes.
- Stem-labeling fix APK SHA-256: `313CB11906AB363F9ECA2C0D11102D0CBD7E584B35E631B8244989679A74DFAF`.
- Stem-labeling fix APK asset check: contains both `assets/UVR-MDX-NET-Inst_Main.onnx` and `assets/UVR_MDXNET_9482.onnx`.
- Emulator verification: installing `dist/MusicSourceSeparation-s25-stemfix-debug.apk` with `adb install -r` succeeded, and `MainActivity` launched and became the focused window on `emulator-5554`.

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

### Later Package Size Reduction

Status: deferred

Current package size snapshot:

- Analyzed APK: `dist/MusicSourceSeparation-s25-stemfix-debug.apk`.
- APK size: `194397049` bytes, or about `185.39 MiB`.
- ONNX Runtime native libraries: `108.32 MiB` compressed, about `58.43%` of the APK.
- Bundled ONNX models: `73.33 MiB` compressed, about `39.56%` of the APK.
- Dex code: `2.49 MiB` compressed, about `1.34%` of the APK.
- Remaining assets, resources, manifest, and metadata are negligible.

Native library ABI breakdown:

- `x86_64`: `31.76 MiB`.
- `x86`: `31.63 MiB`.
- `arm64-v8a`: `26.25 MiB`.
- `armeabi-v7a`: `18.68 MiB`.

Bundled model breakdown:

- `UVR-MDX-NET-Inst_Main.onnx`: `46.94 MiB` compressed, `50.34 MiB` uncompressed.
- `UVR_MDXNET_9482.onnx`: `26.39 MiB` compressed, `28.33 MiB` uncompressed.

Decision:

- Do not reduce package size yet. Keep the current multi-ABI, two-model debug package while model quality, performance, and workflow polish are still being tested.
- The first future size reduction should be an `arm64-v8a`-only S25 build. This should avoid changing runtime behavior and could remove roughly `82 MiB` of x86, x86_64, and 32-bit ARM libraries.
- If `UVR MDXNET 9482` remains the daily-use default, a later lightweight build can omit `UVR-MDX-NET-Inst_Main.onnx`.
- Deeper runtime slimming through `onnxruntime-mobile`, ORT format models, or a custom reduced-operator ONNX Runtime build should stay lower priority because it adds more conversion, compatibility, and regression-test risk.

## Immediate Next Steps

1. Install the stem-labeling fix APK on the Samsung S25.
2. Run `UVR MDXNET 9482` again on Coast Town and verify the vocals and instrumental file names are now correct.
3. Keep `UVR MDXNET 9482` as the likely daily-use default candidate if additional songs confirm quality.
4. Add cancellation and foreground-service handling because full-song runs remain long enough to need interruption or background continuity.
5. Replace the development-only one-window buttons with a cleaner personal-use UI after performance experiments.

## Open Technical Decisions

- Whether a future release-style build should bundle the model, download it once into app-private storage, or ask the user to import a model file.
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
