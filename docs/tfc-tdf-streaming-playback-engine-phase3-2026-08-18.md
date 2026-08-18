# TFC-TDF non-causal streaming playback engine Phase 3

Status: prototype complete. The engine is an isolated Android data-plane
prototype. It is not connected to the existing WAV/FLAC cache playback path,
Media3 renderer chain, or product UI.

## Data path

```text
normal Media3 decoder
        |
        v
dry PCM block -> selectBlock() -> output PCM
                         ^
                         |
wet snapshot <- background analysis worker
       ^                |
       |                +-- reusable CPU/GPU inference session
       +-- MediaCodecStreamingAudioReader
           bounded input ring and window scheduler
```

The normal player remains the source of the playback clock and dry PCM. The
analysis worker owns a second absolute-frame reader, a bounded input ring, the
TFC-TDF zero-padded window assembly, and one injected inference session. No
full-song PCM buffer or cache file is created by this path.

## Playback-thread contract

`selectBlock()` accepts a caller-owned dry block and caller-owned output
buffer. Its hot path only performs atomic reads and copies from an immutable
wet snapshot. It does not:

- acquire a monitor or wait for a worker;
- call `Future.get()` or any other blocking primitive;
- read files, access `MediaCodec`, or invoke LiteRT;
- allocate a window, model tensor, or result object.

If the complete wet range for the requested block is present, the block is
wet. Otherwise the dry block is copied immediately. The switch occurs only at
the requested PCM block boundary, with no crossfade, silence, or pause.

When a wet window finishes after dry audio has already advanced, publication
clips its visible start to the current playback sample. Consequently, an old
dry block cannot be replaced retroactively; only subsequent blocks can switch
to wet. Snapshot reads can span adjacent wet windows.

## Epoch and lifecycle contract

Every start, seek, and model/accelerator switch increments an epoch, clears
the wet snapshot, cancels the old worker future, and starts a fresh analysis
generation. A result is published only if its epoch is still current. A
single-thread executor serializes old and new reader access, while epoch
checks provide logical cancellation even when a model invocation does not
honor interruption immediately.

The inference factory opens one session per generation and reuses it for all
windows in that generation. `StreamingAccelerator.CPU` and `.GPU` are explicit
factory inputs. The current repository does not yet provide a TFC-TDF-specific
LiteRT session implementation; this boundary is intentionally injected so the
same scheduler can be connected to the audited LiteRT CPU/GPU profiles later.

## Android read-ahead decoder

`MediaCodecStreamingAudioReader` owns a separate `MediaExtractor` and
`MediaCodec`. It supports sequential absolute-frame reads and recreates the
decoder for a non-sequential request, discarding codec seek preroll until the
requested frame. PCM16, PCM float, and PCM 8-bit output are converted to
stereo float PCM. The prototype requires 44.1 kHz output and mono or stereo
input; duration metadata supplies the bounded frame count.

This reader is analysis-only. It is not used by the normal Media3 player and
has not yet been validated across the device codec matrix. Codec timestamp
behavior, compressed-format seek accuracy, end padding, and sample-rate
conversion remain device-test work.

## Validation

The five new JVM tests cover:

- immediate dry output while a background session is blocked;
- wet transition after a window completes;
- rejection of late backfill for an emitted dry block;
- seek epoch invalidation and old-result rejection;
- CPU/GPU session selection, session reuse, bounded input memory, and
  cross-window wet reads.

The full standard unit-test task passed:

```text
85 tests
0 failures
0 errors
```

The focused test class is:

```text
com.example.musicsourceseparation.streaming.NonCausalStreamingSeparatedPlaybackEngineTest
```

## Files

- `app/src/main/java/com/example/musicsourceseparation/streaming/NonCausalStreamingSeparatedPlaybackEngine.kt`
- `app/src/main/java/com/example/musicsourceseparation/streaming/MediaCodecStreamingAudioReader.kt`
- `app/src/test/java/com/example/musicsourceseparation/streaming/NonCausalStreamingSeparatedPlaybackEngineTest.kt`

## Phase 4 result

The real TFC-TDF LiteRT 2.2.0 CPU/GPU session, centered STFT/iSTFT DSP, and
MediaCodec read-ahead path were measured on an S25 with a local MP3. Both
backends completed a 30-second real-time run with a seek, no late windows, and
no discarded epoch outputs. Bounded GPU execution reported positive dispatch
and event-wait evidence in both generations. The full metrics, resource
samples, input identities, and limitations are recorded in
`tfc-tdf-streaming-performance-phase4-2026-08-18.md`.

The result qualifies this isolated data-plane prototype for an integration
experiment, not for default product exposure. The next boundary is to feed
the engine from the real Media3 renderer and service lifecycle, while first
reducing the current per-window allocation and memory peak.
