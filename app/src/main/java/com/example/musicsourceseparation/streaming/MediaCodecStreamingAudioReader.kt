package com.example.musicsourceseparation.streaming

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToLong

data class MediaCodecReadTiming(
    val startSample: Long,
    val frameCount: Int,
    val startNanos: Long,
    val endNanos: Long,
    val outputAllocationNanos: Long,
    val pendingCopyNanos: Long,
    val decodeOneCount: Int,
    val inputDequeueNanos: Long,
    val inputQueueNanos: Long,
    val extractorReadNanos: Long,
    val extractorAdvanceNanos: Long,
    val outputDequeueNanos: Long,
    val pcmConversionNanos: Long,
    val pendingAppendNanos: Long,
)

data class MediaCodecStreamingAudioReaderStats(
    val codecTimeoutUs: Long,
    val codecCreateCount: Int,
    val codecFlushCount: Int,
    val codecReleaseCount: Int,
    val decoderSeekCount: Int,
    val flushWallNanos: Long,
    val readCallCount: Int,
    val readFrameCount: Long,
    val readWallNanos: Long,
    val outputAllocationNanos: Long,
    val pendingCopyNanos: Long,
    val decodeOneCount: Long,
    val inputDequeueNanos: Long,
    val inputQueueNanos: Long,
    val extractorReadNanos: Long,
    val extractorAdvanceNanos: Long,
    val outputDequeueNanos: Long,
    val pcmConversionNanos: Long,
    val pendingAppendNanos: Long,
    val readTimings: List<MediaCodecReadTiming>,
)

/**
 * A second, analysis-only decoder. It owns its extractor and codec and exposes
 * sequential absolute-frame reads. A non-sequential read first seeks the
 * existing extractor and flushes the existing codec. If that operation is not
 * supported by a device codec, it falls back to recreating the decoder.
 *
 * This class is intentionally not used by Media3 or the existing WAV/FLAC
 * cache path. All calls are expected on the engine's analysis executor.
 */
class MediaCodecStreamingAudioReader(
    context: Context,
    private val uri: Uri,
    private val codecTimeoutUs: Long = DEFAULT_TIMEOUT_US,
) : StreamingAudioReader {
    private val appContext = context.applicationContext
    private val stateLock = Any()
    private val metadata = readMetadata(appContext, uri)

    override val sampleRate: Int = metadata.sampleRate
    override val channelCount: Int = 2
    override val frameCount: Long = metadata.frameCount

    private var extractor: MediaExtractor? = null
    private var codec: MediaCodec? = null
    private var inputEnded = false
    private var outputEnded = false
    private var outputEncoding = AudioFormat.ENCODING_PCM_16BIT
    private var decoderFrameCursor = 0L
    private var seekTarget = 0L
    private var nextFrame = 0L
    private var pending = FloatArray(0)
    private var pendingOffsetFrames = 0
    private var closed = false
    private var codecCreateCount = 0
    private var codecFlushCount = 0
    private var codecReleaseCount = 0
    private var decoderSeekCount = 0
    private var flushWallNanos = 0L
    private var readCallCount = 0
    private var readFrameCount = 0L
    private var readWallNanos = 0L
    private var outputAllocationNanos = 0L
    private var pendingCopyNanos = 0L
    private var decodeOneCount = 0L
    private var inputDequeueNanos = 0L
    private var inputQueueNanos = 0L
    private var extractorReadNanos = 0L
    private var extractorAdvanceNanos = 0L
    private var outputDequeueNanos = 0L
    private var pcmConversionNanos = 0L
    private var pendingAppendNanos = 0L
    private val readTimings = ArrayList<MediaCodecReadTiming>()

    init {
        require(codecTimeoutUs > 0L) { "codecTimeoutUs must be positive" }
        require(sampleRate == SAMPLE_RATE) {
            "Analysis reader requires 44,100 Hz output, got $sampleRate"
        }
        require(metadata.sourceChannels in 1..2) {
            "Analysis reader requires mono or stereo source, got ${metadata.sourceChannels} channels"
        }
    }

    override fun read(startSample: Long, frameCount: Int): FloatArray {
        require(startSample >= 0) { "startSample must not be negative" }
        require(frameCount > 0) { "frameCount must be positive" }
        require(startSample + frameCount <= this.frameCount) {
            "Read exceeds audio duration"
        }

        synchronized(stateLock) {
            check(!closed) { "Reader is closed" }
            val timing = ReadTimingAccumulator(
                startSample = startSample,
                frameCount = frameCount,
                startNanos = SystemClock.elapsedRealtimeNanos(),
            )
            try {
                if (startSample != nextFrame || codec == null) {
                    resetDecoder(startSample)
                }
                val outputStarted = SystemClock.elapsedRealtimeNanos()
                val output = FloatArray(frameCount * CHANNEL_COUNT)
                timing.outputAllocationNanos += SystemClock.elapsedRealtimeNanos() - outputStarted
                var copiedFrames = 0
                while (copiedFrames < frameCount) {
                    val available = pendingFrameCount()
                    if (available > 0) {
                        val copied = minOf(frameCount - copiedFrames, available)
                        val copyStarted = SystemClock.elapsedRealtimeNanos()
                        pending.copyInto(
                            output,
                            copiedFrames * CHANNEL_COUNT,
                            pendingOffsetFrames * CHANNEL_COUNT,
                            (pendingOffsetFrames + copied) * CHANNEL_COUNT,
                        )
                        timing.pendingCopyNanos += SystemClock.elapsedRealtimeNanos() - copyStarted
                        pendingOffsetFrames += copied
                        copiedFrames += copied
                        nextFrame += copied
                        if (pendingFrameCount() == 0) {
                            pending = FloatArray(0)
                            pendingOffsetFrames = 0
                        }
                    } else {
                        check(!outputEnded) { "Decoder ended before requested frames" }
                        decodeOne(timing)
                    }
                }
                return output
            } finally {
                recordReadTiming(timing.finish(SystemClock.elapsedRealtimeNanos()))
            }
        }
    }

    override fun close() {
        synchronized(stateLock) {
            if (closed) return
            closed = true
            releaseDecoder()
            pending = FloatArray(0)
            pendingOffsetFrames = 0
        }
    }

    fun stats(): MediaCodecStreamingAudioReaderStats = synchronized(stateLock) {
        MediaCodecStreamingAudioReaderStats(
            codecTimeoutUs = codecTimeoutUs,
            codecCreateCount = codecCreateCount,
            codecFlushCount = codecFlushCount,
            codecReleaseCount = codecReleaseCount,
            decoderSeekCount = decoderSeekCount,
            flushWallNanos = flushWallNanos,
            readCallCount = readCallCount,
            readFrameCount = readFrameCount,
            readWallNanos = readWallNanos,
            outputAllocationNanos = outputAllocationNanos,
            pendingCopyNanos = pendingCopyNanos,
            decodeOneCount = decodeOneCount,
            inputDequeueNanos = inputDequeueNanos,
            inputQueueNanos = inputQueueNanos,
            extractorReadNanos = extractorReadNanos,
            extractorAdvanceNanos = extractorAdvanceNanos,
            outputDequeueNanos = outputDequeueNanos,
            pcmConversionNanos = pcmConversionNanos,
            pendingAppendNanos = pendingAppendNanos,
            readTimings = readTimings.toList(),
        )
    }

    private fun resetDecoder(startSample: Long) {
        decoderSeekCount++
        val activeExtractor = extractor
        val activeCodec = codec
        if (activeExtractor != null && activeCodec != null) {
            val flushStarted = System.nanoTime()
            try {
                activeCodec.flush()
                activeExtractor.seekTo(
                    startSample * 1_000_000L / sampleRate,
                    MediaExtractor.SEEK_TO_CLOSEST_SYNC,
                )
                // flush() returns a running decoder to MediaCodec's flushed
                // executing sub-state; calling start() here is invalid.
                codecFlushCount++
                flushWallNanos += System.nanoTime() - flushStarted
                inputEnded = false
                outputEnded = false
                decoderFrameCursor = 0L
                seekTarget = startSample
                nextFrame = startSample
                pending = FloatArray(0)
                pendingOffsetFrames = 0
                return
            } catch (_: Throwable) {
                flushWallNanos += System.nanoTime() - flushStarted
                releaseDecoder()
            }
        }
        createDecoder(startSample)
    }

    private fun createDecoder(startSample: Long) {
        releaseDecoder()
        val newExtractor = MediaExtractor()
        var newCodec: MediaCodec? = null
        try {
            newExtractor.setStreamingDataSource(appContext, uri)
            val track = findAudioTrack(newExtractor)
            require(track >= 0) { "No audio track was found" }
            newExtractor.selectTrack(track)
            val format = newExtractor.getTrackFormat(track)
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: error("Audio track has no MIME type")
            newCodec = MediaCodec.createDecoderByType(mime)
            newCodec.configure(format, null, null, 0)
            val seekUs = startSample * 1_000_000L / sampleRate
            newExtractor.seekTo(seekUs, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
            newCodec.start()
            extractor = newExtractor
            codec = newCodec
            codecCreateCount++
            inputEnded = false
            outputEnded = false
            outputEncoding = format.optionalInteger(MediaFormat.KEY_PCM_ENCODING)
                ?: AudioFormat.ENCODING_PCM_16BIT
            decoderFrameCursor = 0L
            seekTarget = startSample
            nextFrame = startSample
            pending = FloatArray(0)
            pendingOffsetFrames = 0
        } catch (throwable: Throwable) {
            newCodec?.let { activeCodec ->
                try {
                    activeCodec.stop()
                } catch (_: Throwable) {
                    // The codec may not have reached the started state.
                }
                activeCodec.release()
            }
            try {
                newExtractor.release()
            } catch (_: Throwable) {
                // Preserve the original decoder setup failure.
            }
            throw throwable
        }
    }

    private fun decodeOne(timing: ReadTimingAccumulator) {
        val activeCodec = checkNotNull(codec)
        val activeExtractor = checkNotNull(extractor)
        val info = MediaCodec.BufferInfo()
        timing.decodeOneCount++
        decodeOneCount++
        if (!inputEnded) {
            queueNextInput(
                activeCodec = activeCodec,
                activeExtractor = activeExtractor,
                timing = timing,
                timeoutUs = codecTimeoutUs,
            )
        }
        while (true) {
            val outputWaitStarted = SystemClock.elapsedRealtimeNanos()
            when (val outputIndex = activeCodec.dequeueOutputBuffer(info, codecTimeoutUs)) {
                MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    val outputWaitElapsed = SystemClock.elapsedRealtimeNanos() - outputWaitStarted
                    timing.outputDequeueNanos += outputWaitElapsed
                    outputDequeueNanos += outputWaitElapsed
                    if (!inputEnded) {
                        queueNextInput(
                            activeCodec = activeCodec,
                            activeExtractor = activeExtractor,
                            timing = timing,
                            timeoutUs = 0L,
                        )
                    }
                    continue
                }
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val outputWaitElapsed = SystemClock.elapsedRealtimeNanos() - outputWaitStarted
                    timing.outputDequeueNanos += outputWaitElapsed
                    outputDequeueNanos += outputWaitElapsed
                    val outputFormat = activeCodec.outputFormat
                    val outputRate = outputFormat.optionalInteger(MediaFormat.KEY_SAMPLE_RATE)
                    require(outputRate == null || outputRate == sampleRate) {
                        "Decoder output rate changed to $outputRate"
                    }
                    val outputChannels = outputFormat.optionalInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    require(outputChannels == null || outputChannels in 1..2) {
                        "Decoder output channel count changed to $outputChannels"
                    }
                    outputEncoding = outputFormat.optionalInteger(MediaFormat.KEY_PCM_ENCODING)
                        ?: outputEncoding
                }
                else -> {
                    val outputWaitElapsed = SystemClock.elapsedRealtimeNanos() - outputWaitStarted
                    timing.outputDequeueNanos += outputWaitElapsed
                    outputDequeueNanos += outputWaitElapsed
                    if (outputIndex < 0) continue
                    try {
                        if (info.size > 0) {
                            val outputBuffer = activeCodec.getOutputBuffer(outputIndex)
                                ?: error("Decoder returned a null output buffer")
                            val conversionStarted = SystemClock.elapsedRealtimeNanos()
                            val values = decodePcm(outputBuffer, info, outputEncoding)
                            val conversionElapsed = SystemClock.elapsedRealtimeNanos() - conversionStarted
                            timing.pcmConversionNanos += conversionElapsed
                            pcmConversionNanos += conversionElapsed
                            val frameCount = values.size / CHANNEL_COUNT
                            val timestampFrame = if (info.presentationTimeUs >= 0) {
                                (info.presentationTimeUs * sampleRate / 1_000_000.0).roundToLong()
                            } else {
                                decoderFrameCursor
                            }
                            decoderFrameCursor = maxOf(
                                decoderFrameCursor,
                                timestampFrame + frameCount,
                            )
                            val appendStarted = SystemClock.elapsedRealtimeNanos()
                            appendFromTarget(timestampFrame, values)
                            val appendElapsed = SystemClock.elapsedRealtimeNanos() - appendStarted
                            timing.pendingAppendNanos += appendElapsed
                            pendingAppendNanos += appendElapsed
                        }
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            outputEnded = true
                        }
                    } finally {
                        activeCodec.releaseOutputBuffer(outputIndex, false)
                    }
                    return
                }
            }
        }
    }

    private fun queueNextInput(
        activeCodec: MediaCodec,
        activeExtractor: MediaExtractor,
        timing: ReadTimingAccumulator,
        timeoutUs: Long,
    ): Boolean {
        val inputWaitStarted = SystemClock.elapsedRealtimeNanos()
        val inputIndex = activeCodec.dequeueInputBuffer(timeoutUs)
        val inputWaitElapsed = SystemClock.elapsedRealtimeNanos() - inputWaitStarted
        timing.inputDequeueNanos += inputWaitElapsed
        inputDequeueNanos += inputWaitElapsed
        if (inputIndex < 0) return false

        val inputBuffer = activeCodec.getInputBuffer(inputIndex)
            ?: error("Decoder returned a null input buffer")
        val extractorReadStarted = SystemClock.elapsedRealtimeNanos()
        val size = activeExtractor.readSampleData(inputBuffer, 0)
        val extractorReadElapsed = SystemClock.elapsedRealtimeNanos() - extractorReadStarted
        timing.extractorReadNanos += extractorReadElapsed
        extractorReadNanos += extractorReadElapsed
        val queueStarted = SystemClock.elapsedRealtimeNanos()
        if (size < 0) {
            activeCodec.queueInputBuffer(
                inputIndex,
                0,
                0,
                0L,
                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
            )
            inputEnded = true
        } else {
            activeCodec.queueInputBuffer(
                inputIndex,
                0,
                size,
                activeExtractor.sampleTime,
                0,
            )
        }
        val queueElapsed = SystemClock.elapsedRealtimeNanos() - queueStarted
        timing.inputQueueNanos += queueElapsed
        inputQueueNanos += queueElapsed
        if (size >= 0) {
            val extractorAdvanceStarted = SystemClock.elapsedRealtimeNanos()
            activeExtractor.advance()
            val extractorAdvanceElapsed = SystemClock.elapsedRealtimeNanos() - extractorAdvanceStarted
            timing.extractorAdvanceNanos += extractorAdvanceElapsed
            extractorAdvanceNanos += extractorAdvanceElapsed
        }
        return true
    }

    private fun recordReadTiming(timing: MediaCodecReadTiming) {
        readCallCount++
        readFrameCount += timing.frameCount.toLong()
        readWallNanos += timing.endNanos - timing.startNanos
        outputAllocationNanos += timing.outputAllocationNanos
        pendingCopyNanos += timing.pendingCopyNanos
        if (readTimings.size >= MAX_READ_TIMING_RECORDS) {
            readTimings.removeAt(0)
        }
        readTimings += timing
    }

    private class ReadTimingAccumulator(
        private val startSample: Long,
        private val frameCount: Int,
        private val startNanos: Long,
    ) {
        var outputAllocationNanos = 0L
        var pendingCopyNanos = 0L
        var decodeOneCount = 0
        var inputDequeueNanos = 0L
        var inputQueueNanos = 0L
        var extractorReadNanos = 0L
        var extractorAdvanceNanos = 0L
        var outputDequeueNanos = 0L
        var pcmConversionNanos = 0L
        var pendingAppendNanos = 0L

        fun finish(endNanos: Long) = MediaCodecReadTiming(
            startSample = startSample,
            frameCount = frameCount,
            startNanos = startNanos,
            endNanos = endNanos,
            outputAllocationNanos = outputAllocationNanos,
            pendingCopyNanos = pendingCopyNanos,
            decodeOneCount = decodeOneCount,
            inputDequeueNanos = inputDequeueNanos,
            inputQueueNanos = inputQueueNanos,
            extractorReadNanos = extractorReadNanos,
            extractorAdvanceNanos = extractorAdvanceNanos,
            outputDequeueNanos = outputDequeueNanos,
            pcmConversionNanos = pcmConversionNanos,
            pendingAppendNanos = pendingAppendNanos,
        )
    }

    private fun appendFromTarget(frameStart: Long, values: FloatArray) {
        val frameCount = values.size / CHANNEL_COUNT
        val frameEnd = frameStart + frameCount
        if (frameEnd <= seekTarget) return
        val clippedStart = maxOf(frameStart, seekTarget)
        val offsetFrames = (clippedStart - frameStart).toInt()
        val clipped = if (offsetFrames == 0) {
            values
        } else {
            values.copyOfRange(offsetFrames * CHANNEL_COUNT, values.size)
        }
        if (pendingFrameCount() > 0) {
            val expectedStart = seekTarget + pendingOffsetFrames + pendingFrameCount()
            require(expectedStart == clippedStart) {
                "Decoder output timestamp gap: expected $expectedStart, got $clippedStart"
            }
            pending += clipped
        } else {
            pending = clipped
            pendingOffsetFrames = 0
        }
    }

    private fun decodePcm(
        buffer: ByteBuffer,
        info: MediaCodec.BufferInfo,
        encoding: Int,
    ): FloatArray {
        val duplicate = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        duplicate.position(info.offset)
        duplicate.limit(info.offset + info.size)
        val sourceChannels = metadata.sourceChannels
        val sourceValues = when (encoding) {
            AudioFormat.ENCODING_PCM_16BIT -> {
                val values = ShortArray(duplicate.remaining() / 2)
                duplicate.asShortBuffer().get(values)
                FloatArray(values.size) { index -> values[index] / 32768f }
            }
            AudioFormat.ENCODING_PCM_FLOAT -> {
                val values = FloatArray(duplicate.remaining() / 4)
                duplicate.asFloatBuffer().get(values)
                values
            }
            AudioFormat.ENCODING_PCM_8BIT -> {
                FloatArray(duplicate.remaining()) {
                    ((duplicate.get().toInt() and 0xFF) - 128) / 128f
                }
            }
            else -> error("Unsupported decoder PCM encoding: $encoding")
        }
        require(sourceValues.size % sourceChannels == 0) {
            "PCM output is not aligned to source channels"
        }
        if (sourceChannels == CHANNEL_COUNT) return sourceValues
        val frames = sourceValues.size
        val stereo = FloatArray(frames * CHANNEL_COUNT)
        for (frame in 0 until frames) {
            val value = sourceValues[frame]
            stereo[frame * CHANNEL_COUNT] = value
            stereo[frame * CHANNEL_COUNT + 1] = value
        }
        return stereo
    }

    private fun pendingFrameCount(): Int {
        return pending.size / CHANNEL_COUNT - pendingOffsetFrames
    }

    private fun releaseDecoder() {
        codec?.let { activeCodec ->
            codecReleaseCount++
            try {
                activeCodec.stop()
            } catch (_: Throwable) {
                // Codec may already have been interrupted or released.
            }
            try {
                activeCodec.release()
            } catch (_: Throwable) {
                // Preserve the surrounding lifecycle operation.
            }
        }
        codec = null
        try {
            extractor?.release()
        } catch (_: Throwable) {
            // Preserve the surrounding lifecycle operation.
        }
        extractor = null
        inputEnded = false
        outputEnded = false
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int {
        for (index in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)
                ?: continue
            if (mime.startsWith("audio/")) return index
        }
        return -1
    }

    private data class Metadata(
        val sampleRate: Int,
        val sourceChannels: Int,
        val frameCount: Long,
    )

    private companion object {
        const val SAMPLE_RATE = 44_100
        const val CHANNEL_COUNT = 2
        const val DEFAULT_TIMEOUT_US = 10_000L
        const val MAX_READ_TIMING_RECORDS = 2_048

        fun readMetadata(context: Context, uri: Uri): Metadata {
            val extractor = MediaExtractor()
            try {
                extractor.setStreamingDataSource(context, uri)
                var selected: MediaFormat? = null
                for (index in 0 until extractor.trackCount) {
                    val format = extractor.getTrackFormat(index)
                    val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                    if (mime.startsWith("audio/")) {
                        selected = format
                        break
                    }
                }
                val format = selected ?: error("No audio track was found")
                val rate = format.optionalInteger(MediaFormat.KEY_SAMPLE_RATE)
                    ?: error("Audio track has no sample rate")
                val channels = format.optionalInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    ?: error("Audio track has no channel count")
                val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) {
                    format.getLong(MediaFormat.KEY_DURATION)
                } else {
                    error("Audio track has no duration")
                }
                return Metadata(
                    sampleRate = rate,
                    sourceChannels = channels,
                    frameCount = maxOf(1L, (durationUs * rate / 1_000_000.0).roundToLong()),
                )
            } finally {
                extractor.release()
            }
        }

        fun MediaFormat.optionalInteger(key: String): Int? {
            return if (containsKey(key)) getInteger(key) else null
        }

        fun MediaExtractor.setStreamingDataSource(context: Context, uri: Uri) {
            if (uri.scheme == "file") {
                setDataSource(requireNotNull(uri.path) { "File Uri has no path" })
            } else {
                setDataSource(context, uri, null)
            }
        }
    }
}
