package com.example.musicsourceseparation.model

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class HtdemucsWindowPlan(
    val index: Int,
    val trackSamples: Int,
    val offset: Int,
    val actualSamples: Int,
    val contextStart: Int,
    val contextEnd: Int,
    val sourceStart: Int,
    val sourceEnd: Int,
    val padLeft: Int,
    val padRight: Int,
    val cropLeft: Int,
    val cropRight: Int,
) {
    val copiedSamples: Int
        get() = sourceEnd - sourceStart
}

data class HtdemucsWindowOutput(
    val offset: Int,
    val planarSamples: FloatArray,
)

data class HtdemucsOlaResult(
    val planarSamples: FloatArray,
    val accumulatedWeights: FloatArray,
)

data class HtdemucsGlobalNormalization(
    val mean: Float,
    val sampleStandardDeviation: Float,
    val epsilon: Float,
    val divisor: Float,
)

class HtdemucsStreamingPlan(
    val readyWindowCount: Int = DEFAULT_READY_WINDOW_COUNT,
) {
    val readySamples: Int
    val readyDurationSeconds: Double

    init {
        require(readyWindowCount > 0) { "readyWindowCount must be positive." }
        readySamples = Math.multiplyExact(readyWindowCount, STRIDE_SAMPLES)
        readyDurationSeconds = readySamples.toDouble() / SAMPLE_RATE
    }

    fun windowPlans(trackSamples: Int): List<HtdemucsWindowPlan> {
        require(trackSamples > 0) { "trackSamples must be positive." }
        require(trackSamples <= Int.MAX_VALUE - WINDOW_SAMPLES) {
            "trackSamples is too large for canonical window planning."
        }

        val plans = mutableListOf<HtdemucsWindowPlan>()
        var offset = 0
        while (offset < trackSamples) {
            val actualSamples = min(trackSamples - offset, WINDOW_SAMPLES)
            val delta = WINDOW_SAMPLES - actualSamples
            val cropLeft = delta / 2
            val cropRight = delta - cropLeft
            val contextStart = offset - cropLeft
            val contextEnd = contextStart + WINDOW_SAMPLES
            val sourceStart = max(0, contextStart)
            val sourceEnd = min(trackSamples, contextEnd)
            plans += HtdemucsWindowPlan(
                index = plans.size,
                trackSamples = trackSamples,
                offset = offset,
                actualSamples = actualSamples,
                contextStart = contextStart,
                contextEnd = contextEnd,
                sourceStart = sourceStart,
                sourceEnd = sourceEnd,
                padLeft = sourceStart - contextStart,
                padRight = contextEnd - sourceEnd,
                cropLeft = cropLeft,
                cropRight = cropRight,
            )
            offset = Math.addExact(offset, STRIDE_SAMPLES)
        }
        return plans
    }

    fun extractPaddedWindow(
        planarTrack: FloatArray,
        planeCount: Int,
        window: HtdemucsWindowPlan,
    ): FloatArray {
        require(planeCount > 0) { "planeCount must be positive." }
        require(planarTrack.size % planeCount == 0) {
            "Planar track size must be divisible by planeCount."
        }
        val trackSamples = planarTrack.size / planeCount
        require(window.trackSamples == trackSamples) { "Window belongs to a different track." }
        val expectedPlans = windowPlans(trackSamples)
        require(window.index in expectedPlans.indices && expectedPlans[window.index] == window) {
            "Window does not match the canonical ascending plan."
        }

        val padded = FloatArray(Math.multiplyExact(planeCount, WINDOW_SAMPLES))
        repeat(planeCount) { plane ->
            planarTrack.copyInto(
                destination = padded,
                destinationOffset = plane * WINDOW_SAMPLES + window.padLeft,
                startIndex = plane * trackSamples + window.sourceStart,
                endIndex = plane * trackSamples + window.sourceEnd,
            )
        }
        return padded
    }

    fun triangleWeight(index: Int): Float {
        require(index in 0 until WINDOW_SAMPLES) { "Weight index is outside the canonical window." }
        val numerator = if (index < WEIGHT_PEAK_SAMPLES) {
            index + 1
        } else {
            WINDOW_SAMPLES - index
        }
        return numerator.toFloat() / WEIGHT_PEAK_SAMPLES.toFloat()
    }

    fun triangleWeightPrefix(sampleCount: Int): FloatArray {
        require(sampleCount in 0..WINDOW_SAMPLES) {
            "Weight prefix must fit in the canonical window."
        }
        return FloatArray(sampleCount, ::triangleWeight)
    }

    fun overlapAdd(
        trackSamples: Int,
        outputPlaneCount: Int,
        windowOutputs: List<HtdemucsWindowOutput>,
    ): HtdemucsOlaResult {
        require(outputPlaneCount > 0) { "outputPlaneCount must be positive." }
        val windows = windowPlans(trackSamples)
        require(windowOutputs.size == windows.size) {
            "Expected ${windows.size} window outputs, received ${windowOutputs.size}."
        }

        val output = FloatArray(Math.multiplyExact(outputPlaneCount, trackSamples))
        val accumulatedWeights = FloatArray(trackSamples)

        // Matching apply_model requires window-order FP32 adds and the prefix of the
        // triangle after TensorChunk center trimming, including the EOF window.
        windows.forEachIndexed { windowIndex, window ->
            val windowOutput = windowOutputs[windowIndex]
            require(windowOutput.offset == window.offset) {
                "Window outputs must be supplied in ascending offset order."
            }
            require(
                windowOutput.planarSamples.size ==
                    Math.multiplyExact(outputPlaneCount, WINDOW_SAMPLES),
            ) { "Each model output must contain one complete canonical window." }

            repeat(window.actualSamples) { localSample ->
                val trackSample = window.offset + localSample
                val weight = triangleWeight(localSample)
                accumulatedWeights[trackSample] = accumulatedWeights[trackSample] + weight
                repeat(outputPlaneCount) { plane ->
                    val outputIndex = plane * trackSamples + trackSample
                    val windowIndexInPlane =
                        plane * WINDOW_SAMPLES + window.cropLeft + localSample
                    val weighted = windowOutput.planarSamples[windowIndexInPlane] * weight
                    output[outputIndex] = output[outputIndex] + weighted
                }
            }
        }

        accumulatedWeights.forEachIndexed { sample, weight ->
            check(weight > 0f) { "OLA weight is not positive at sample $sample." }
            repeat(outputPlaneCount) { plane ->
                val index = plane * trackSamples + sample
                output[index] = output[index] / weight
            }
        }
        return HtdemucsOlaResult(output, accumulatedWeights)
    }

    fun globalNormalization(planarStereo: FloatArray): HtdemucsGlobalNormalization {
        require(planarStereo.size % CHANNEL_COUNT == 0) {
            "Expected a planar stereo track."
        }
        val sampleCount = planarStereo.size / CHANNEL_COUNT
        require(sampleCount >= 2) {
            "Sample standard deviation with correction=1 requires at least two samples."
        }

        var sum = 0.0
        repeat(sampleCount) { sample ->
            val left = planarStereo[sample]
            val right = planarStereo[sampleCount + sample]
            require(left.isFinite() && right.isFinite()) { "Normalization input must be finite." }
            sum += (left.toDouble() + right.toDouble()) * 0.5
        }
        val mean = sum / sampleCount

        var squaredDeviationSum = 0.0
        repeat(sampleCount) { sample ->
            val reference = (
                planarStereo[sample].toDouble() +
                    planarStereo[sampleCount + sample].toDouble()
                ) * 0.5
            val deviation = reference - mean
            squaredDeviationSum += deviation * deviation
        }
        val sampleStandardDeviation = sqrt(squaredDeviationSum / (sampleCount - 1)).toFloat()
        val meanFloat = mean.toFloat()
        val divisor = sampleStandardDeviation + NORMALIZATION_EPSILON
        require(meanFloat.isFinite() && divisor.isFinite() && divisor > 0f) {
            "Normalization parameters must be finite with a positive divisor."
        }
        return HtdemucsGlobalNormalization(
            mean = meanFloat,
            sampleStandardDeviation = sampleStandardDeviation,
            epsilon = NORMALIZATION_EPSILON,
            divisor = divisor,
        )
    }

    fun normalize(
        planarStereo: FloatArray,
        parameters: HtdemucsGlobalNormalization = globalNormalization(planarStereo),
    ): FloatArray {
        require(planarStereo.size % CHANNEL_COUNT == 0) { "Expected a planar stereo track." }
        return FloatArray(planarStereo.size) { index ->
            (planarStereo[index] - parameters.mean) / parameters.divisor
        }
    }

    fun denormalize(
        normalizedSamples: FloatArray,
        parameters: HtdemucsGlobalNormalization,
    ): FloatArray = FloatArray(normalizedSamples.size) { index ->
        normalizedSamples[index] * parameters.divisor + parameters.mean
    }

    companion object {
        const val SAMPLE_RATE = 44_100
        const val CHANNEL_COUNT = 2
        const val WINDOW_SAMPLES = 343_980
        const val STRIDE_SAMPLES = 257_985
        const val OVERLAP_SAMPLES = 85_995
        const val DEFAULT_READY_WINDOW_COUNT = 2
        const val NORMALIZATION_EPSILON = 1e-8f

        private const val WEIGHT_PEAK_SAMPLES = WINDOW_SAMPLES / 2
    }
}
