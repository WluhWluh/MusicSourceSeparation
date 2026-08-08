package com.example.musicsourceseparation.model

import org.jtransforms.fft.FloatFFT_1D
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.ThreadFactory
import kotlin.math.PI
import kotlin.math.cos

class MdxSpectrogram(
    private val config: MdxDspConfig = MdxDspConfig(),
    val workerCount: Int = 1,
) : AutoCloseable {
    init {
        require(workerCount in 1..MAX_WORKER_COUNT) {
            "DSP worker count must be from 1 through $MAX_WORKER_COUNT."
        }
    }

    private val lanes = Array(maxOf(workerCount, MdxDspConfig.STEREO_CHANNELS)) {
        FftLane(config.nFft)
    }
    private val executor: ExecutorService? = if (workerCount > 1) {
        Executors.newFixedThreadPool(workerCount, DspThreadFactory())
    } else {
        null
    }
    private val window = FloatArray(config.nFft) { index ->
        (0.5 - 0.5 * cos(2.0 * PI * index / config.nFft)).toFloat()
    }
    private val reusableWorkspace by lazy { ReusableWorkspace(config, window) }

    fun waveformToTensor(waveform: Array<FloatArray>): FloatArray {
        requireStereoChunk(waveform)

        val tensor = FloatArray(config.tensorElementCount)
        val padded = Array(MdxDspConfig.STEREO_CHANNELS) { channel ->
            reflectPad(waveform[channel], config.trim)
        }
        waveformToTensorInto(tensor, padded, nhwc = false)
        return tensor
    }

    fun waveformToNhwcTensorInto(waveform: Array<FloatArray>, tensor: FloatArray) {
        requireStereoChunk(waveform)
        require(tensor.size == config.tensorElementCount)
        val workspace = reusableWorkspace
        for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
            reflectPadInto(waveform[channel], config.trim, workspace.padded[channel])
        }
        waveformToTensorInto(tensor, workspace.padded, nhwc = true)
    }

    private fun waveformToTensorInto(
        tensor: FloatArray,
        padded: Array<FloatArray>,
        nhwc: Boolean,
    ) {
        runLanes(workerCount) { laneIndex ->
            val lane = lanes[laneIndex]
            val firstFrame = laneIndex * config.dimT / workerCount
            val lastFrame = (laneIndex + 1) * config.dimT / workerCount
            for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
                for (frameIndex in firstFrame until lastFrame) {
                    val start = frameIndex * config.hopLength
                    lane.buffer.fill(0f)
                    for (sampleIndex in 0 until config.nFft) {
                        lane.buffer[sampleIndex] =
                            padded[channel][start + sampleIndex] * window[sampleIndex]
                    }
                    lane.fft.realForwardFull(lane.buffer)

                    val realChannel = channel * 2
                    val imaginaryChannel = realChannel + 1
                    for (frequencyIndex in 0 until config.dimF) {
                        tensor[tensorIndex(realChannel, frequencyIndex, frameIndex, nhwc)] =
                            lane.buffer[frequencyIndex * 2]
                        tensor[tensorIndex(imaginaryChannel, frequencyIndex, frameIndex, nhwc)] =
                            lane.buffer[frequencyIndex * 2 + 1]
                    }
                }
            }
        }
    }

    fun tensorToWaveform(tensor: FloatArray): Array<FloatArray> {
        require(tensor.size == config.tensorElementCount) {
            "Expected tensor size ${config.tensorElementCount}, got ${tensor.size}."
        }

        val outputLength = config.chunkSize + config.nFft
        val output = Array(MdxDspConfig.STEREO_CHANNELS) { FloatArray(outputLength) }
        val windowSum = FloatArray(outputLength)
        val result = Array(MdxDspConfig.STEREO_CHANNELS) { FloatArray(config.chunkSize) }
        tensorToWaveformInto(tensor, output, windowSum, result, nhwc = false, precomputedWindowSum = false)
        return result
    }

    fun nhwcTensorToWaveformInto(tensor: FloatArray, result: Array<FloatArray>) {
        require(tensor.size == config.tensorElementCount)
        require(result.size == MdxDspConfig.STEREO_CHANNELS && result.all { it.size == config.chunkSize })
        val workspace = reusableWorkspace
        workspace.inverseOutput.forEach { it.fill(0f) }
        tensorToWaveformInto(
            tensor,
            workspace.inverseOutput,
            workspace.windowSum,
            result,
            nhwc = true,
            precomputedWindowSum = true,
        )
    }

    private fun tensorToWaveformInto(
        tensor: FloatArray,
        output: Array<FloatArray>,
        windowSum: FloatArray,
        result: Array<FloatArray>,
        nhwc: Boolean,
        precomputedWindowSum: Boolean,
    ) {

        val inverseLaneCount = if (workerCount > 1) MdxDspConfig.STEREO_CHANNELS else 1
        runLanes(inverseLaneCount) { laneIndex ->
            val firstChannel = if (workerCount > 1) laneIndex else 0
            val lastChannel = if (workerCount > 1) firstChannel + 1 else MdxDspConfig.STEREO_CHANNELS
            val lane = lanes[laneIndex]
            for (channel in firstChannel until lastChannel) {
                for (frameIndex in 0 until config.dimT) {
                    lane.buffer.fill(0f)
                    val realChannel = channel * 2
                    val imaginaryChannel = realChannel + 1

                    for (frequencyIndex in 0 until config.dimF) {
                        val real = tensor[tensorIndex(realChannel, frequencyIndex, frameIndex, nhwc)]
                        val imaginary = tensor[tensorIndex(imaginaryChannel, frequencyIndex, frameIndex, nhwc)]
                        setComplexBin(lane.buffer, frequencyIndex, real, imaginary)
                        if (frequencyIndex in 1 until config.nBins - 1) {
                            setComplexBin(lane.buffer, config.nFft - frequencyIndex, real, -imaginary)
                        }
                    }

                    lane.fft.complexInverse(lane.buffer, true)
                    val start = frameIndex * config.hopLength
                    for (sampleIndex in 0 until config.nFft) {
                        val value = lane.buffer[sampleIndex * 2] * window[sampleIndex]
                        output[channel][start + sampleIndex] += value
                        if (!precomputedWindowSum && channel == 0) {
                            windowSum[start + sampleIndex] += window[sampleIndex] * window[sampleIndex]
                        }
                    }
                }
            }
        }

        for (sampleIndex in output.indicesForNormalization(windowSum)) {
            val divisor = windowSum[sampleIndex]
            output[0][sampleIndex] /= divisor
            output[1][sampleIndex] /= divisor
        }

        for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
            output[channel].copyInto(
                result[channel],
                startIndex = config.trim,
                endIndex = config.trim + config.chunkSize,
            )
        }
    }

    fun tensorShape(): IntArray {
        return intArrayOf(
            MdxDspConfig.STEM_COMPLEX_CHANNELS,
            config.dimF,
            config.dimT,
        )
    }

    override fun close() {
        executor?.shutdown()
    }

    private fun runLanes(count: Int, action: (Int) -> Unit) {
        val laneExecutor = executor
        if (laneExecutor == null) {
            action(0)
            return
        }
        val futures = laneExecutor.invokeAll(
            (0 until count).map { laneIndex -> Callable { action(laneIndex) } },
        )
        futures.forEach(Future<Unit>::get)
    }

    private fun requireStereoChunk(waveform: Array<FloatArray>) {
        require(waveform.size == MdxDspConfig.STEREO_CHANNELS) {
            "Expected stereo waveform."
        }
        for (channel in waveform.indices) {
            require(waveform[channel].size == config.chunkSize) {
                "Expected channel $channel to contain ${config.chunkSize} samples, got ${waveform[channel].size}."
            }
        }
    }

    private fun reflectPad(input: FloatArray, pad: Int): FloatArray {
        val result = FloatArray(input.size + 2 * pad)
        reflectPadInto(input, pad, result)
        return result
    }

    private fun reflectPadInto(input: FloatArray, pad: Int, result: FloatArray) {
        require(result.size == input.size + 2 * pad)
        for (index in result.indices) {
            val sourceIndex = reflectIndex(index - pad, input.size)
            result[index] = input[sourceIndex]
        }
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

    private fun tensorIndex(channel: Int, frequencyIndex: Int, frameIndex: Int, nhwc: Boolean = false): Int {
        return if (nhwc) {
            (frequencyIndex * config.dimT + frameIndex) * MdxDspConfig.STEM_COMPLEX_CHANNELS + channel
        } else {
            (channel * config.dimF + frequencyIndex) * config.dimT + frameIndex
        }
    }

    private fun setComplexBin(buffer: FloatArray, bin: Int, real: Float, imaginary: Float) {
        val index = bin * 2
        buffer[index] = real
        buffer[index + 1] = imaginary
    }

    private fun Array<FloatArray>.indicesForNormalization(windowSum: FloatArray): IntRange {
        return 0 until minOf(first().size, windowSum.size)
    }

    private class FftLane(nFft: Int) {
        val fft = FloatFFT_1D(nFft.toLong())
        val buffer = FloatArray(nFft * 2)
    }

    private class ReusableWorkspace(config: MdxDspConfig, window: FloatArray) {
        val padded = Array(MdxDspConfig.STEREO_CHANNELS) {
            FloatArray(config.chunkSize + 2 * config.trim)
        }
        val inverseOutput = Array(MdxDspConfig.STEREO_CHANNELS) {
            FloatArray(config.chunkSize + config.nFft)
        }
        val windowSum = FloatArray(config.chunkSize + config.nFft).also { sum ->
            for (frameIndex in 0 until config.dimT) {
                val start = frameIndex * config.hopLength
                for (sampleIndex in 0 until config.nFft) {
                    sum[start + sampleIndex] += window[sampleIndex] * window[sampleIndex]
                }
            }
        }
    }

    private class DspThreadFactory : ThreadFactory {
        override fun newThread(runnable: Runnable): Thread = Thread(runnable).apply {
            name = "mdx-dsp-${nextThreadId()}"
            isDaemon = true
        }
    }

    companion object {
        private const val MAX_WORKER_COUNT = 8
        private var threadId = 0

        @Synchronized
        private fun nextThreadId(): Int = ++threadId
    }
}
