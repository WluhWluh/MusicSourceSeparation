package com.example.musicsourceseparation.streaming

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.min

enum class StreamingAccelerator {
    CPU,
    GPU,
}

enum class StreamingPlaybackMode {
    DRY,
    WET,
}

data class StreamingModelConfig(
    val modelId: String,
    val inputSamples: Int = 130_048,
    val trimSamples: Int = 5_120,
    val usefulSamples: Int = 119_808,
) {
    init {
        require(modelId.isNotBlank()) { "modelId must not be blank" }
        require(inputSamples > 0) { "inputSamples must be positive" }
        require(trimSamples >= 0) { "trimSamples must not be negative" }
        require(usefulSamples > 0) { "usefulSamples must be positive" }
        require(trimSamples * 2 + usefulSamples == inputSamples) {
            "inputSamples must equal trimSamples * 2 + usefulSamples"
        }
    }
}

/**
 * Absolute-frame reader owned by the analysis side of the streaming engine.
 * Implementations must not be called from the playback thread.
 */
interface StreamingAudioReader : AutoCloseable {
    val sampleRate: Int
    val channelCount: Int
    val frameCount: Long

    /** Returns interleaved stereo float PCM with size frameCount * 2. */
    fun read(startSample: Long, frameCount: Int): FloatArray
}

/**
 * A reusable CPU/GPU model session. The engine reuses one session across
 * generations while the model and accelerator identity stay unchanged.
 */
interface StreamingInferenceSession : AutoCloseable {
    /** Returns only the valid, unpadded stereo output for actualSamples. */
    fun process(inputPcm: FloatArray, actualSamples: Int): FloatArray
}

fun interface StreamingInferenceSessionFactory {
    fun open(
        model: StreamingModelConfig,
        accelerator: StreamingAccelerator,
    ): StreamingInferenceSession
}

data class StreamingEngineSnapshot(
    val epoch: Long,
    val playbackSample: Long,
    val active: Boolean,
    val publishedWindowCount: Long,
    val discardedEpochOutputCount: Long,
    val lateWindowCount: Long,
    val readCallCount: Long,
    val readFrameCount: Long,
    val maxInputRingSamples: Long,
    val wetWindowCount: Int,
    val wetStartSample: Long?,
    val wetEndSample: Long?,
    val wetLeadSamples: Long,
    val readAheadDecodeWallNanos: Long,
)

/**
 * Hostable Android prototype for non-causal dry/wet separated playback.
 *
 * The normal Media3 decoder supplies dry PCM blocks through [selectBlock]. A
 * single background worker owns the independent reader, input ring and
 * inference session. The playback path only reads an immutable wet snapshot,
 * copies already-prepared PCM, and never performs I/O, inference, waits, or
 * locking.
 */
class NonCausalStreamingSeparatedPlaybackEngine(
    private val reader: StreamingAudioReader,
    private val sessionFactory: StreamingInferenceSessionFactory,
    private val maxInputWindows: Int = 3,
    private val maxWetWindows: Int = 4,
    private val readChunkSamples: Int = 16_384,
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor(),
) : AutoCloseable {
    init {
        require(reader.sampleRate == SAMPLE_RATE) {
            "Streaming reader must be 44,100 Hz"
        }
        require(reader.channelCount == CHANNEL_COUNT) {
            "Streaming reader must be stereo"
        }
        require(reader.frameCount > 0) { "Streaming reader must not be empty" }
        require(maxInputWindows > 0) { "maxInputWindows must be positive" }
        require(maxWetWindows > 0) { "maxWetWindows must be positive" }
        require(readChunkSamples > 0) { "readChunkSamples must be positive" }
    }

    private val closed = AtomicBoolean(false)
    private val enabled = AtomicBoolean(false)
    private val epoch = AtomicLong(0)
    private val playbackSample = AtomicLong(0)
    private val wetSnapshot = AtomicReference(WetSnapshot.EMPTY)
    private val publishedWindowCount = AtomicLong(0)
    private val discardedEpochOutputCount = AtomicLong(0)
    private val lateWindowCount = AtomicLong(0)
    private val readCallCount = AtomicLong(0)
    private val readFrameCount = AtomicLong(0)
    private val readAheadDecodeWallNanos = AtomicLong(0)
    private val maxInputRingSamples = AtomicLong(0)
    private val lifecycleLock = Any()

    @Volatile
    private var currentModel: StreamingModelConfig? = null

    @Volatile
    private var currentAccelerator: StreamingAccelerator = StreamingAccelerator.CPU

    private var analysisFuture: Future<*>? = null

    @Volatile
    private var reusableSession: SessionHolder? = null

    /** Starts a new analysis generation at startSample. */
    fun start(
        startSample: Long = 0,
        model: StreamingModelConfig,
        accelerator: StreamingAccelerator = StreamingAccelerator.CPU,
    ) {
        requirePosition(startSample)
        synchronized(lifecycleLock) {
            check(!closed.get()) { "Engine is closed" }
            restartLocked(startSample, model, accelerator)
        }
    }

    /** Stops using wet output without stopping the normal player. */
    fun setEnabled(value: Boolean) {
        check(!closed.get()) { "Engine is closed" }
        enabled.set(value)
    }

    /** Starts a fresh analysis generation from the window containing targetSample. */
    fun seek(targetSample: Long) {
        requirePosition(targetSample)
        synchronized(lifecycleLock) {
            check(!closed.get()) { "Engine is closed" }
            val model = currentModel ?: return
            restartLocked(targetSample, model, currentAccelerator)
        }
    }

    /** Switches model/accelerator and invalidates every older wet result. */
    fun switchModel(
        model: StreamingModelConfig,
        accelerator: StreamingAccelerator,
    ) {
        synchronized(lifecycleLock) {
            check(!closed.get()) { "Engine is closed" }
            restartLocked(playbackSample.get(), model, accelerator)
        }
    }

    /**
     * Supplies one normal-player PCM block and writes the selected output.
     * This is the audio-thread entry point. It performs no synchronization,
     * allocation, file access, inference, or Future wait.
     */
    fun selectBlock(
        positionSample: Long,
        dryPcm: FloatArray,
        outputPcm: FloatArray,
    ): StreamingPlaybackMode {
        require(positionSample >= 0) { "positionSample must not be negative" }
        require(dryPcm.isNotEmpty() && dryPcm.size % CHANNEL_COUNT == 0) {
            "dryPcm must contain non-empty stereo frames"
        }
        require(outputPcm.size == dryPcm.size) {
            "outputPcm must have the same size as dryPcm"
        }

        playbackSample.lazySet(positionSample + dryPcm.size / CHANNEL_COUNT)
        if (!enabled.get()) {
            dryPcm.copyInto(outputPcm)
            return StreamingPlaybackMode.DRY
        }

        val frames = dryPcm.size / CHANNEL_COUNT
        val snapshot = wetSnapshot.get()
        if (snapshot.copy(positionSample, frames, outputPcm)) {
            return StreamingPlaybackMode.WET
        }
        dryPcm.copyInto(outputPcm)
        return StreamingPlaybackMode.DRY
    }

    /** Updates the read-ahead clock when the player has no output block yet. */
    fun observePlaybackPosition(positionSample: Long) {
        requirePosition(positionSample)
        playbackSample.lazySet(positionSample)
    }

    fun snapshot(): StreamingEngineSnapshot {
        val snapshot = wetSnapshot.get()
        val currentPlaybackSample = playbackSample.get()
        val wetStartSample = snapshot.windows.firstOrNull()?.startSample
        val wetEndSample = snapshot.windows.lastOrNull()?.endSample
        return StreamingEngineSnapshot(
            epoch = epoch.get(),
            playbackSample = currentPlaybackSample,
            active = currentModel != null && !closed.get(),
            publishedWindowCount = publishedWindowCount.get(),
            discardedEpochOutputCount = discardedEpochOutputCount.get(),
            lateWindowCount = lateWindowCount.get(),
            readCallCount = readCallCount.get(),
            readFrameCount = readFrameCount.get(),
            maxInputRingSamples = maxInputRingSamples.get(),
            wetWindowCount = snapshot.windows.size,
            wetStartSample = wetStartSample,
            wetEndSample = wetEndSample,
            wetLeadSamples = maxOf(0L, (wetEndSample ?: currentPlaybackSample) - currentPlaybackSample),
            readAheadDecodeWallNanos = readAheadDecodeWallNanos.get(),
        )
    }

    override fun close() {
        synchronized(lifecycleLock) {
            if (!closed.compareAndSet(false, true)) return
            enabled.set(false)
            epoch.incrementAndGet()
            wetSnapshot.set(WetSnapshot.EMPTY)
            analysisFuture?.cancel(true)
            analysisFuture = null
        }
        analysisExecutor.shutdownNow()
        try {
            analysisExecutor.awaitTermination(5, TimeUnit.SECONDS)
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        closeReusableSession()
        reader.close()
        currentModel = null
    }

    private fun restartLocked(
        startSample: Long,
        model: StreamingModelConfig,
        accelerator: StreamingAccelerator,
    ) {
        val generation = epoch.incrementAndGet()
        analysisFuture?.cancel(true)
        wetSnapshot.set(WetSnapshot.EMPTY)
        playbackSample.set(startSample)
        currentModel = model
        currentAccelerator = accelerator
        enabled.set(true)
        analysisFuture = analysisExecutor.submit {
            analyze(generation, startSample, model, accelerator)
        }
    }

    private fun analyze(
        generation: Long,
        startSample: Long,
        model: StreamingModelConfig,
        accelerator: StreamingAccelerator,
    ) {
        val analysisStart = (startSample / model.usefulSamples) * model.usefulSamples
        var readCursor = analysisStart
        var nextWindowIndex = analysisStart / model.usefulSamples
        val inputCapacity = (maxInputWindows + 1L) * model.usefulSamples
        val inputRing = InputRing(inputCapacity.toInt())
        val session = try {
            acquireSession(model, accelerator)
        } catch (throwable: Throwable) {
            if (isCurrent(generation)) {
                currentModel = null
                wetSnapshot.set(WetSnapshot.EMPTY)
            }
            return
        }

        try {
            while (isCurrent(generation)) {
                val currentPosition = playbackSample.get().coerceIn(0, reader.frameCount - 1)
                val readHorizon = min(
                    reader.frameCount,
                    maxOf(
                        currentPosition + maxInputWindows.toLong() * model.usefulSamples,
                        analysisStart + model.usefulSamples.toLong(),
                    ),
                )
                var progressed = false

                if (readCursor < readHorizon) {
                    val count = min(
                        readChunkSamples.toLong(),
                        readHorizon - readCursor,
                    ).toInt()
                    val readStarted = System.nanoTime()
                    val chunk = reader.read(readCursor, count)
                    readAheadDecodeWallNanos.addAndGet(System.nanoTime() - readStarted)
                    require(chunk.size == count * CHANNEL_COUNT) {
                        "Reader returned ${chunk.size} values for $count frames"
                    }
                    inputRing.write(readCursor, chunk)
                    readCursor += count
                    readCallCount.incrementAndGet()
                    readFrameCount.addAndGet(count.toLong())
                    maxInputRingSamples.accumulateAndGet(inputRing.maxFrames.toLong()) { old, value ->
                        maxOf(old, value)
                    }
                    progressed = true
                }

                val processLimit = min(
                    reader.frameCount,
                    currentPosition + maxInputWindows.toLong() * model.usefulSamples,
                )
                while (isCurrent(generation)) {
                    val windowStart = nextWindowIndex * model.usefulSamples
                    if (windowStart >= reader.frameCount) break
                    val actual = min(
                        model.usefulSamples.toLong(),
                        reader.frameCount - windowStart,
                    ).toInt()
                    val windowEnd = windowStart + actual
                    if (windowEnd <= currentPosition) {
                        nextWindowIndex++
                        inputRing.discardBefore(windowEnd)
                        lateWindowCount.incrementAndGet()
                        progressed = true
                        continue
                    }
                    if (windowEnd > processLimit) break
                    val valid = inputRing.read(windowStart, actual) ?: break
                    val input = FloatArray(model.inputSamples * CHANNEL_COUNT)
                    valid.copyInto(input, model.trimSamples * CHANNEL_COUNT)
                    val wet = try {
                        session.process(input, actual).also { output ->
                            require(output.size == actual * CHANNEL_COUNT) {
                                "Session returned ${output.size} values for $actual frames"
                            }
                        }
                    } catch (throwable: Throwable) {
                        discardSession(session)
                        throw throwable
                    }
                    if (!isCurrent(generation)) {
                        discardedEpochOutputCount.incrementAndGet()
                        return
                    }
                    publishWet(generation, windowStart, wet)
                    nextWindowIndex++
                    inputRing.discardBefore(windowEnd)
                    progressed = true
                }

                if (!progressed) {
                    try {
                        Thread.sleep(2L)
                    } catch (interrupted: InterruptedException) {
                        Thread.currentThread().interrupt()
                        return
                    }
                }
            }
        } catch (interrupted: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (throwable: Throwable) {
            if (isCurrent(generation)) {
                currentModel = null
                wetSnapshot.set(WetSnapshot.EMPTY)
            }
        }
    }

    private fun acquireSession(
        model: StreamingModelConfig,
        accelerator: StreamingAccelerator,
    ): StreamingInferenceSession {
        reusableSession?.let { holder ->
            if (holder.model == model && holder.accelerator == accelerator) {
                return holder.session
            }
        }
        closeReusableSession()
        return sessionFactory.open(model, accelerator).also { session ->
            reusableSession = SessionHolder(model, accelerator, session)
        }
    }

    private fun discardSession(session: StreamingInferenceSession) {
        val holder = reusableSession
        if (holder?.session !== session) return
        reusableSession = null
        try {
            session.close()
        } catch (_: Throwable) {
            // Preserve the original processing failure.
        }
    }

    private fun closeReusableSession() {
        val holder = reusableSession ?: return
        reusableSession = null
        try {
            holder.session.close()
        } catch (_: Throwable) {
            // Session cleanup must not prevent reader or engine cleanup.
        }
    }

    private fun publishWet(
        generation: Long,
        windowStart: Long,
        wet: FloatArray,
    ) {
        if (!isCurrent(generation)) {
            discardedEpochOutputCount.incrementAndGet()
            return
        }
        val windowEnd = windowStart + wet.size / CHANNEL_COUNT
        val visibleStart = maxOf(windowStart, playbackSample.get())
        if (visibleStart >= windowEnd) {
            discardedEpochOutputCount.incrementAndGet()
            return
        }
        val visible = if (visibleStart == windowStart) {
            wet
        } else {
            wet.copyOfRange(
                ((visibleStart - windowStart) * CHANNEL_COUNT).toInt(),
                wet.size,
            )
        }
        val old = wetSnapshot.get()
        val retained = old.windows
            .filter { it.endSample > playbackSample.get() }
            .filterNot { it.startSample == visibleStart }
            .toMutableList()
        retained += WetWindow(visibleStart, visible)
        retained.sortBy { it.startSample }
        while (retained.size > maxWetWindows) {
            retained.removeAt(0)
        }
        wetSnapshot.set(WetSnapshot(retained.toTypedArray()))
        publishedWindowCount.incrementAndGet()
    }

    private fun isCurrent(generation: Long): Boolean {
        return !closed.get() && epoch.get() == generation
    }

    private fun requirePosition(position: Long) {
        require(position >= 0 && position < reader.frameCount) {
            "Position $position is outside ${reader.frameCount} frames"
        }
    }

    private class InputRing(private val capacityFrames: Int) {
        private val blocks = ArrayList<InputBlock>()
        var maxFrames: Int = 0
            private set

        fun write(startFrame: Long, samples: FloatArray) {
            require(samples.isNotEmpty() && samples.size % CHANNEL_COUNT == 0)
            val frameCount = samples.size / CHANNEL_COUNT
            val endFrame = startFrame + frameCount
            require(blocks.none { startFrame < it.endSample && it.startFrame < endFrame }) {
                "Input ring received overlapping blocks"
            }
            blocks += InputBlock(startFrame, samples.copyOf())
            blocks.sortBy { it.startFrame }
            while (totalFrames() > capacityFrames && blocks.isNotEmpty()) {
                blocks.removeAt(0)
            }
            maxFrames = maxOf(maxFrames, totalFrames())
        }

        fun read(startFrame: Long, frameCount: Int): FloatArray? {
            require(frameCount > 0)
            val output = FloatArray(frameCount * CHANNEL_COUNT)
            var position = startFrame
            var remaining = frameCount
            var outputOffset = 0
            for (block in blocks) {
                if (block.endSample <= position) continue
                if (block.startFrame > position) return null
                val offset = (position - block.startFrame).toInt()
                val available = block.frameCount - offset
                val copied = min(remaining, available)
                block.samples.copyInto(
                    output,
                    outputOffset,
                    offset * CHANNEL_COUNT,
                    (offset + copied) * CHANNEL_COUNT,
                )
                position += copied
                remaining -= copied
                outputOffset += copied * CHANNEL_COUNT
                if (remaining == 0) return output
            }
            return null
        }

        fun discardBefore(positionFrame: Long) {
            val replacement = ArrayList<InputBlock>()
            for (block in blocks) {
                if (block.endSample <= positionFrame) continue
                if (block.startFrame < positionFrame) {
                    val offset = (positionFrame - block.startFrame).toInt()
                    replacement += InputBlock(
                        positionFrame,
                        block.samples.copyOfRange(offset * CHANNEL_COUNT, block.samples.size),
                    )
                } else {
                    replacement += block
                }
            }
            blocks.clear()
            blocks += replacement
        }

        private fun totalFrames(): Int = blocks.sumOf { it.frameCount }
    }

    private data class InputBlock(
        val startFrame: Long,
        val samples: FloatArray,
    ) {
        val frameCount: Int get() = samples.size / CHANNEL_COUNT
        val endSample: Long get() = startFrame + frameCount
    }

    private data class WetWindow(
        val startSample: Long,
        val samples: FloatArray,
    ) {
        val endSample: Long get() = startSample + samples.size / CHANNEL_COUNT
    }

    private data class SessionHolder(
        val model: StreamingModelConfig,
        val accelerator: StreamingAccelerator,
        val session: StreamingInferenceSession,
    )

    private class WetSnapshot(val windows: Array<WetWindow>) {
        fun copy(startSample: Long, frameCount: Int, destination: FloatArray): Boolean {
            var position = startSample
            var remaining = frameCount
            var destinationOffset = 0
            for (window in windows) {
                if (window.endSample <= position) continue
                if (window.startSample > position) return false
                val offset = (position - window.startSample).toInt()
                val available = window.samples.size / CHANNEL_COUNT - offset
                val copied = min(remaining, available)
                window.samples.copyInto(
                    destination,
                    destinationOffset * CHANNEL_COUNT,
                    offset * CHANNEL_COUNT,
                    (offset + copied) * CHANNEL_COUNT,
                )
                position += copied
                remaining -= copied
                destinationOffset += copied
                if (remaining == 0) return true
            }
            return false
        }

        companion object {
            val EMPTY = WetSnapshot(emptyArray())
        }
    }

    private companion object {
        const val SAMPLE_RATE = 44_100
        const val CHANNEL_COUNT = 2
    }
}
