package com.example.musicsourceseparation.model

import org.jtransforms.fft.FloatFFT_1D
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sqrt

class HtdemucsDsp(
    val windowSamples: Int = 343_980,
    val istftMode: IstftMode = IstftMode.SERIAL,
    val istftWorkers: Int = 1,
    val reuseIoWorkspaces: Boolean = false,
) : HtdemucsDspSession {
    override val frameCount: Int = ceil(windowSamples.toDouble() / HOP_LENGTH).toInt()

    private val fft = FloatFFT_1D(N_FFT.toLong())
    private val hann = FloatArray(N_FFT) { index ->
        (0.5 - 0.5 * cos(2.0 * PI * index / N_FFT)).toFloat()
    }
    private val forwardScale = (1.0 / sqrt(N_FFT.toDouble())).toFloat()
    private val inverseScale = sqrt(N_FFT.toDouble()).toFloat()
    private val outerLength = frameCount * HOP_LENGTH + OUTER_PAD_LEFT * 2
    private val outerPadRight = outerLength - windowSamples - OUTER_PAD_LEFT
    private val fullFrameCount = frameCount + FRAME_PAD_LEFT + FRAME_PAD_RIGHT
    private val overlapLength = N_FFT + HOP_LENGTH * (fullFrameCount - 1)
    private val windowSquareSum = FloatArray(overlapLength).also { sum ->
        repeat(fullFrameCount) { frame ->
            val start = frame * HOP_LENGTH
            hann.indices.forEach { sample ->
                sum[start + sample] += hann[sample] * hann[sample]
            }
        }
    }
    private val validatedIstftWorkers = validateIstftConfig(istftMode, istftWorkers)
    private val inverseWorkspaces = when (istftMode) {
        IstftMode.SERIAL -> emptyArray()
        IstftMode.PARALLEL_LANES -> Array(validatedIstftWorkers) { InverseWorkspace() }
    }
    private val inverseExecutor: ExecutorService? = when (istftMode) {
        IstftMode.SERIAL -> null
        IstftMode.PARALLEL_LANES -> Executors.newFixedThreadPool(
            validatedIstftWorkers,
        ) { runnable ->
            Thread(
                runnable,
                "htdemucs-istft-${ISTFT_THREAD_SEQUENCE.incrementAndGet()}",
            )
        }.also { executor ->
            (executor as ThreadPoolExecutor).prestartAllCoreThreads()
        }
    }
    private val closed = AtomicBoolean(false)
    private val reusableForwardOutput = if (reuseIoWorkspaces) {
        FloatArray(FEATURE_COUNT * FREQUENCY_BINS * frameCount)
    } else {
        null
    }
    private val reusableForwardPadded = if (reuseIoWorkspaces) FloatArray(outerLength) else null
    private val reusableForwardFftBuffer = if (reuseIoWorkspaces) FloatArray(N_FFT * 2) else null
    private var reusableInverseOutput = FloatArray(0)

    init {
        require(windowSamples > 0)
        require(outerPadRight >= 0)
    }

    @Synchronized
    override fun waveformToSpectrum(planarStereoWaveform: FloatArray): FloatArray {
        check(!closed.get()) { "HtdemucsDsp is closed." }
        require(planarStereoWaveform.size == CHANNEL_COUNT * windowSamples) {
            "Expected planar stereo waveform with " + (CHANNEL_COUNT * windowSamples) + " values."
        }
        val output = reusableForwardOutput
            ?: FloatArray(FEATURE_COUNT * FREQUENCY_BINS * frameCount)
        val padded = reusableForwardPadded ?: FloatArray(outerLength)
        val fftBuffer = reusableForwardFftBuffer ?: FloatArray(N_FFT * 2)
        repeat(CHANNEL_COUNT) { channel ->
            val channelOffset = channel * windowSamples
            padded.indices.forEach { index ->
                val sourceIndex = reflectIndex(index - OUTER_PAD_LEFT, windowSamples)
                padded[index] = planarStereoWaveform[channelOffset + sourceIndex]
            }
            repeat(frameCount) { frame ->
                val start = frame * HOP_LENGTH
                fftBuffer.fill(0f)
                repeat(N_FFT) { sample ->
                    fftBuffer[sample] = padded[start + sample] * hann[sample]
                }
                fft.realForwardFull(fftBuffer)
                val realFeature = channel * 2
                val imaginaryFeature = realFeature + 1
                repeat(FREQUENCY_BINS) { frequency ->
                    output[spectrumIndex(realFeature, frequency, frame)] =
                        fftBuffer[frequency * 2] * forwardScale
                    output[spectrumIndex(imaginaryFeature, frequency, frame)] =
                        fftBuffer[frequency * 2 + 1] * forwardScale
                }
            }
        }
        return output
    }

    @Synchronized
    fun frequencyToWaveform(packedFrequency: FloatArray): FloatArray =
        frequencyToWaveform(packedFrequency, DEFAULT_STEM_COUNT)

    override fun frequencyToWaveform(
        packedFrequency: FloatArray,
        stemCount: Int,
    ): FloatArray {
        check(!closed.get()) { "HtdemucsDsp is closed." }
        val expected = stemCount * FEATURE_COUNT * FREQUENCY_BINS * frameCount
        require(packedFrequency.size == expected) {
            "Expected packed frequency tensor with $expected values."
        }
        val outputSize = stemCount * CHANNEL_COUNT * windowSamples
        val output = if (reuseIoWorkspaces) {
            if (reusableInverseOutput.size != outputSize) {
                reusableInverseOutput = FloatArray(outputSize)
            }
            reusableInverseOutput
        } else {
            FloatArray(outputSize)
        }
        val planeCount = stemCount * CHANNEL_COUNT
        val executor = inverseExecutor
        if (executor == null) {
            processSerialInverse(
                stemCount = stemCount,
                packedFrequency = packedFrequency,
                output = output,
            )
        } else {
            val activeWorkers = min(istftWorkers, planeCount)
            val cancellationRequested = AtomicBoolean(false)
            val tasks = (0 until activeWorkers).map { workerIndex ->
                Callable {
                    processInversePlanes(
                        workerIndex = workerIndex,
                        workerCount = activeWorkers,
                        planeCount = planeCount,
                        packedFrequency = packedFrequency,
                        output = output,
                        cancellationRequested = cancellationRequested,
                    )
                }
            }
            try {
                executor.invokeAll(tasks).forEach { future -> future.get() }
            } catch (error: InterruptedException) {
                cancellationRequested.set(true)
                Thread.currentThread().interrupt()
                throw IllegalStateException("Interrupted while running parallel iSTFT.", error)
            }
        }
        return output
    }

    private fun processSerialInverse(
        stemCount: Int,
        packedFrequency: FloatArray,
        output: FloatArray,
    ) {
        val fftBuffer = FloatArray(N_FFT * 2)
        val overlap = FloatArray(overlapLength)
        repeat(stemCount) { stem ->
            repeat(CHANNEL_COUNT) { channel ->
                overlap.fill(0f)
                val realFeature = channel * 2
                val imaginaryFeature = realFeature + 1
                repeat(frameCount) { frame ->
                    fftBuffer.fill(0f)
                    repeat(FREQUENCY_BINS) { frequency ->
                        val real = packedFrequency[
                            frequencyIndex(stem, realFeature, frequency, frame)
                        ]
                        val imaginary = packedFrequency[
                            frequencyIndex(stem, imaginaryFeature, frequency, frame)
                        ]
                        setComplex(fftBuffer, frequency, real, imaginary)
                        if (frequency > 0) {
                            setComplex(fftBuffer, N_FFT - frequency, real, -imaginary)
                        }
                    }
                    fft.complexInverse(fftBuffer, true)
                    val start = (frame + FRAME_PAD_LEFT) * HOP_LENGTH
                    repeat(N_FFT) { sample ->
                        overlap[start + sample] +=
                            fftBuffer[sample * 2] * inverseScale * hann[sample]
                    }
                }
                val outputOffset = (stem * CHANNEL_COUNT + channel) * windowSamples
                repeat(windowSamples) { sample ->
                    val reconstructionIndex = OUTER_PAD_LEFT + sample
                    val overlapIndex = CENTER_TRIM + reconstructionIndex
                    val divisor = windowSquareSum[overlapIndex]
                    check(divisor > 0f)
                    output[outputOffset + sample] = overlap[overlapIndex] / divisor
                }
            }
        }
    }

    private fun processInversePlanes(
        workerIndex: Int,
        workerCount: Int,
        planeCount: Int,
        packedFrequency: FloatArray,
        output: FloatArray,
        cancellationRequested: AtomicBoolean,
    ) {
        val workspace = inverseWorkspaces[workerIndex]
        var plane = workerIndex
        while (plane < planeCount) {
            checkNotCancelled(cancellationRequested)
            val stem = plane / CHANNEL_COUNT
            val channel = plane % CHANNEL_COUNT
            workspace.overlap.fill(0f)
            val realFeature = channel * 2
            val imaginaryFeature = realFeature + 1
            repeat(frameCount) { frame ->
                checkNotCancelled(cancellationRequested)
                workspace.fftBuffer.fill(0f)
                repeat(FREQUENCY_BINS) { frequency ->
                    val real = packedFrequency[
                        frequencyIndex(stem, realFeature, frequency, frame)
                    ]
                    val imaginary = packedFrequency[
                        frequencyIndex(stem, imaginaryFeature, frequency, frame)
                    ]
                    setComplex(workspace.fftBuffer, frequency, real, imaginary)
                    if (frequency > 0) {
                        setComplex(workspace.fftBuffer, N_FFT - frequency, real, -imaginary)
                    }
                }
                workspace.fft.complexInverse(workspace.fftBuffer, true)
                val start = (frame + FRAME_PAD_LEFT) * HOP_LENGTH
                repeat(N_FFT) { sample ->
                    workspace.overlap[start + sample] +=
                        workspace.fftBuffer[sample * 2] * inverseScale * hann[sample]
                }
            }
            val outputOffset = plane * windowSamples
            repeat(windowSamples) { sample ->
                val reconstructionIndex = OUTER_PAD_LEFT + sample
                val overlapIndex = CENTER_TRIM + reconstructionIndex
                val divisor = windowSquareSum[overlapIndex]
                check(divisor > 0f)
                output[outputOffset + sample] = workspace.overlap[overlapIndex] / divisor
            }
            plane += workerCount
        }
    }

    private fun checkNotCancelled(cancellationRequested: AtomicBoolean) {
        if (cancellationRequested.get() || Thread.currentThread().isInterrupted) {
            throw InterruptedException("Parallel iSTFT worker was cancelled.")
        }
    }

    fun combineBranches(
        frequencyWaveform: FloatArray,
        timeWaveform: FloatArray,
    ): FloatArray {
        require(frequencyWaveform.size == timeWaveform.size)
        return FloatArray(frequencyWaveform.size) { index ->
            frequencyWaveform[index] + timeWaveform[index]
        }
    }

    private fun spectrumIndex(feature: Int, frequency: Int, frame: Int): Int =
        (feature * FREQUENCY_BINS + frequency) * frameCount + frame

    private fun frequencyIndex(
        stem: Int,
        feature: Int,
        frequency: Int,
        frame: Int,
    ): Int = ((stem * FEATURE_COUNT + feature) * FREQUENCY_BINS + frequency) *
        frameCount + frame

    private fun setComplex(buffer: FloatArray, bin: Int, real: Float, imaginary: Float) {
        buffer[bin * 2] = real
        buffer[bin * 2 + 1] = imaginary
    }

    private fun reflectIndex(index: Int, size: Int): Int {
        var reflected = index
        while (reflected < 0 || reflected >= size) {
            reflected = if (reflected < 0) {
                -reflected
            } else {
                2 * size - reflected - 2
            }
        }
        return reflected
    }

    @Synchronized
    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        val executor = inverseExecutor ?: return
        executor.shutdown()
        if (!awaitTerminationPreservingInterrupt(executor)) {
            executor.shutdownNow()
            check(awaitTerminationPreservingInterrupt(executor)) {
                "Parallel iSTFT executor did not terminate."
            }
        }
    }

    private fun awaitTerminationPreservingInterrupt(executor: ExecutorService): Boolean {
        var interrupted = Thread.interrupted()
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(
            EXECUTOR_CLOSE_TIMEOUT_SECONDS,
        )
        try {
            while (true) {
                val remaining = deadline - System.nanoTime()
                if (remaining <= 0L) return executor.isTerminated
                try {
                    return executor.awaitTermination(remaining, TimeUnit.NANOSECONDS)
                } catch (_: InterruptedException) {
                    interrupted = true
                }
            }
        } finally {
            if (interrupted) Thread.currentThread().interrupt()
        }
    }

    enum class IstftMode(
        val wireValue: String,
    ) {
        SERIAL("serial"),
        PARALLEL_LANES("parallel-lanes"),
        ;

        companion object {
            fun fromWireValue(value: String): IstftMode = entries.firstOrNull {
                it.wireValue == value
            } ?: error("Unknown iSTFT mode '$value'.")
        }
    }

    private inner class InverseWorkspace {
        val fft = FloatFFT_1D(N_FFT.toLong())
        val fftBuffer = FloatArray(N_FFT * 2)
        val overlap = FloatArray(overlapLength)
    }

    companion object {
        const val SAMPLE_RATE = 44_100
        const val N_FFT = 4_096
        const val HOP_LENGTH = 1_024
        const val FREQUENCY_BINS = 2_048
        const val CHANNEL_COUNT = 2
        const val FEATURE_COUNT = 4
        const val DEFAULT_STEM_COUNT = 6
        const val OUTER_PAD_LEFT = 1_536
        const val CENTER_TRIM = N_FFT / 2
        const val FRAME_PAD_LEFT = 2
        const val FRAME_PAD_RIGHT = 2
        const val MAX_ISTFT_WORKERS = 16

        private const val EXECUTOR_CLOSE_TIMEOUT_SECONDS = 5L
        private val ISTFT_THREAD_SEQUENCE = AtomicInteger()

        private fun validateIstftConfig(mode: IstftMode, workers: Int): Int {
            require(workers in 1..MAX_ISTFT_WORKERS)
            require(mode != IstftMode.SERIAL || workers == 1) {
                "Serial iSTFT requires exactly one worker."
            }
            require(mode != IstftMode.PARALLEL_LANES || workers >= 2) {
                "Parallel-lanes iSTFT requires at least two workers."
            }
            return workers
        }
    }
}
