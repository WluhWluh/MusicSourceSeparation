# TFC-TDF streaming scheduler Phase 1

Status: complete. This phase is a host-only scheduling experiment. It does
not invoke LiteRT, Android audio APIs, or the real TFC-TDF DSP graph.

## Scope

The simulator freezes the current compact TFC-TDF window contract:

| Item | Value |
| --- | ---: |
| Sample rate | 44,100 Hz |
| Input window | 130,048 samples |
| Trim on each side | 5,120 samples |
| Valid output per window | 119,808 samples |
| Playback block | 1,024 samples in the default profile |
| Channels | 2 |

The valid output is divided into fixed playback blocks. The simulator uses a
deterministic scaled copy of the source as wet output so that scheduling
transitions can be asserted without confusing them with model quality. It is
not a numerical or audio-quality result for the TFC-TDF model.

## Scheduling contract

The modeled data path is:

```text
source PCM -> bounded dry ring
           -> logical read-ahead and bounded window task queue
           -> wet window result
           -> bounded wet ring
           -> dry/wet block selector
           -> rendered PCM
```

The selector follows these rules:

1. Emit wet PCM when the complete requested playback block is available in the
   wet ring.
2. Emit the source PCM immediately when wet PCM is unavailable. This is a
   block-boundary hard switch with no crossfade, pause, or silence insertion.
3. A wet result that arrives after a dry block was emitted can only be used for
   future samples. It can never replace already emitted audio.
4. Seek and model changes increment the transport epoch, clear old wet data,
   and discard pending work from older epochs.
5. Pause advances neither the playback sample position nor output blocks;
   producer timing may continue to advance.
6. Played samples are removed from both rings. The producer is only scheduled
   within a finite playback-ahead horizon, including at most one partial
   boundary window beyond the configured sample capacity.

Adjacent wet blocks are read as one logical range. This is required when a
   playback block crosses a TFC-TDF window boundary. The ring keeps blocks in
   sample order after partial eviction so a trimmed first block cannot hide
   later contiguous data.

## Validation

The simulator validates:

- no gap or duplicate position within an output segment;
- valid channel count, non-empty blocks, and final block sizing;
- no output beyond the source track;
- finite output samples and known dry/wet modes;
- no late result replacing an emitted dry range;
- epoch invalidation for seek, disable/enable, and model switching;
- wet reads spanning two adjacent model windows;
- bounded sample and pending-task memory;
- pause/resume continuity.

The standard-library test entry point is:

```text
python -m unittest discover -s tests -p 'test_tfc_tdf_stream_simulator.py' -v
```

Result: 9 tests passed. The module and tests also pass `py_compile` and
`git diff --check`.

## Representative traces

The CLI reports event-level JSON as well as compact mode ranges. With a
12-second synthetic source, the following deterministic profiles were run.

| Profile | Read-ahead | Inference latency | Dry blocks | Wet blocks | Late windows | Mode trace |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fast | 20x realtime | 50 ms | 9 | 508 | 0 | `0.000-0.209 dry`, then `0.209-12.000 wet` |
| Slow | 1x realtime | 500 ms | 517 | 0 | 4 | `0.000-12.000 dry` |
| Seek | 20x realtime | 50 ms | 18 | 349 | 0 | `0.000-0.209 dry`, `0.209-1.509 wet`, seek to `5.000`, `5.000-5.209 dry`, then wet |

The seek run dropped one pending task from the old epoch. The fast profile's
wet ring reached 359,424 samples, equal to three configured valid-output
windows. The slow profile kept all 517 output blocks dry without inserting a
gap. The output hash is included in each CLI report for deterministic replay,
but it is not an audio-quality identity.

## Implementation boundary

Implemented in:

- `tools/tfc_tdf_stream_simulator.py`
- `tests/test_tfc_tdf_stream_simulator.py`

The simulator currently abstracts the input queue and inference completion as
logical wall-clock events. It does not run the frozen TFLite model, perform
STFT/iSTFT, decode a real song, measure CPU/GPU throughput, or integrate with
`PlaybackService`. It also does not claim that a real device can sustain the
configured producer rate.

## Next phase

Phase 2 should replace the synthetic wet transform with the actual host
TFC-TDF window pipeline and real fixture audio, while preserving this dry/wet
timeline contract. It should measure window preparation, inference, DSP,
result publication, and sustained realtime margin before any Android playback
integration.
