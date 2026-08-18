# TFC-TDF streaming read-ahead Phase 2

Status: complete. This phase turns the Phase 1 logical input timing into an
independent bounded local-audio reader. It remains a host-side experiment and
does not yet integrate with Android Media3 or run the TFC-TDF neural model.

## Goal

The playback decoder and analysis decoder are separate concerns:

```text
normal playback decoder -> playback clock -> dry output

independent local reader -> bounded input ring -> complete model window
                         -> inference task queue -> wet output ring
```

The analysis reader does not write a full-song PCM file. It reads by absolute
frame position, keeps a finite amount of source PCM, and releases the source
range after the corresponding padded model window has been submitted.

## Reader contract

`LocalAudioReadAhead` accepts an `AudioFrameSource` with an absolute-frame
`read(start, count)` operation. The current TFC-TDF profile requires 44.1 kHz
stereo input and uses:

| Item | Value |
| --- | ---: |
| Model input | 130,048 samples |
| Valid source region | 119,808 samples |
| Zero padding before valid region | 5,120 samples |
| Zero padding after valid region | 5,120 samples |
| Default input ring capacity | 3 valid-output windows |

For a playback position inside a window, the reader starts at that window's
boundary so the required prefix cannot be lost before the model input is
complete. It advances the read horizon from the current retained ring floor,
which keeps seek-in-window reads bounded as well as normal sequential reads.

Each ready result contains:

- the window index, epoch, absolute start, and actual valid length;
- the valid source samples for synthetic host scheduling;
- a zero-padded `130,048 x 2` model input.

The reader submits windows strictly in ascending window order. An incomplete
window is not submitted. If playback passes an incomplete window, it is marked
late and skipped, allowing later windows to be considered after the source
reader catches up.

## Session and cancellation behavior

`reset(start_sample, epoch)` clears the input ring and starts at the containing
window boundary. Seek and model changes call this reset before any new analysis
read. `cancel(epoch)` clears the ring, marks the reader inactive, and prevents
future reads or submissions. The simulator also drops inference tasks from the
old epoch.

The simulator pumps the reader independently of whether the playback output is
currently dry or wet. Its deterministic host budget is:

```text
max(1, ceil(playback_block_samples * read_ahead_rate))
```

This models the existing Phase 1 fast and slow producer profiles without
pretending to measure a disk or Android codec. Inference latency starts only
after a complete input window has actually been read and submitted.

## Source adapters

Implemented adapters are:

- `ArrayAudioFrameSource` for deterministic simulator fixtures;
- `WavPcm16FrameSource` for an independent real local PCM16 WAV reader.

`WavPcm16FrameSource` owns its own file cursor and supports mono-to-stereo
expansion. It intentionally rejects compressed WAV, non-PCM16 data, more than
two channels, and non-44.1 kHz input. Android compressed-file support should
implement the same absolute-frame interface with a second
`MediaExtractor`/`MediaCodec` instance in the product repository; it is not
silently approximated by reusing the normal Media3 decoder here.

## Validation

The dedicated reader tests cover:

- bounded chunk sizes and window-boundary submission;
- exact zero padding and valid sample placement;
- seek reset and epoch replacement;
- incomplete-window withholding during slow reads;
- cancellation preventing future reads and results;
- stereo and mono PCM16 WAV decoding.

The Phase 1 scheduler tests continue to cover dry/wet output, late results,
window-crossing reads, seek, model changes, pause/resume, disable/enable, and
ring bounds. Current result:

```text
5 read-ahead tests passed
9 scheduler tests passed
```

Representative 12-second simulator runs after integration:

| Profile | Dry | Wet | Total late windows | Read-ahead submitted | Max input ring |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20x / 50 ms | 8 blocks | 509 blocks | 0 | 5 | 132,096 samples |
| 0.5x / 0 ms | 517 blocks | 0 blocks | 4 | 0 | 60,416 samples |
| seek at 1.5 s, target 5 s | 16 blocks | 351 blocks | 0 | 9 | 132,096 samples |

The seek run dropped two old inference tasks and rebuilt the reader at epoch 2.
The fast run switches to wet at approximately 0.186 s. The slow run remains
dry without inserting silence or a playback gap.

## Implementation boundary

Implemented in:

- `tools/tfc_tdf_read_ahead.py`
- `tools/tfc_tdf_stream_simulator.py`
- `tests/test_tfc_tdf_read_ahead.py`
- `tests/test_tfc_tdf_stream_simulator.py`

This phase does not yet perform actual TFC-TDF STFT/iSTFT, LiteRT inference,
Media3 integration, or device timing. The host read budget is a deterministic
scheduling control, not a disk-throughput benchmark. Phase 3 adds the Android
engine and a concrete MediaCodec reader; the remaining device validation is
documented in `tfc-tdf-streaming-playback-engine-phase3-2026-08-18.md`.
