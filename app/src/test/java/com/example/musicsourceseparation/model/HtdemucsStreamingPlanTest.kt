package com.example.musicsourceseparation.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class HtdemucsStreamingPlanTest {
    private val streamingPlan = HtdemucsStreamingPlan()

    @Test
    fun canonicalTwoWindowPlanMatchesFrozenTensorChunkPlan() {
        val windows = streamingPlan.windowPlans(TRACK_SAMPLES)

        assertEquals(2, windows.size)
        assertEquals(
            HtdemucsWindowPlan(
                index = 0,
                trackSamples = TRACK_SAMPLES,
                offset = 0,
                actualSamples = HtdemucsStreamingPlan.WINDOW_SAMPLES,
                contextStart = 0,
                contextEnd = HtdemucsStreamingPlan.WINDOW_SAMPLES,
                sourceStart = 0,
                sourceEnd = HtdemucsStreamingPlan.WINDOW_SAMPLES,
                padLeft = 0,
                padRight = 0,
                cropLeft = 0,
                cropRight = 0,
            ),
            windows[0],
        )
        assertEquals(
            HtdemucsWindowPlan(
                index = 1,
                trackSamples = TRACK_SAMPLES,
                offset = 257_985,
                actualSamples = 257_985,
                contextStart = 214_988,
                contextEnd = 558_968,
                sourceStart = 214_988,
                sourceEnd = 515_970,
                padLeft = 0,
                padRight = 42_998,
                cropLeft = 42_997,
                cropRight = 42_998,
            ),
            windows[1],
        )
        assertEquals(300_982, windows[1].copiedSamples)
        assertEquals(515_970, streamingPlan.readySamples)
        assertEquals(11.7, streamingPlan.readyDurationSeconds, 1e-12)
    }

    @Test
    fun eofWindowUsesCenteredTensorChunkCopyAndRightPadding() {
        val track = FloatArray(TRACK_SAMPLES) { index -> (index + 1).toFloat() }
        val eof = streamingPlan.windowPlans(TRACK_SAMPLES)[1]

        val padded = streamingPlan.extractPaddedWindow(track, planeCount = 1, window = eof)

        assertEquals(HtdemucsStreamingPlan.WINDOW_SAMPLES, padded.size)
        assertEquals(track[214_988], padded[0])
        assertEquals(track[257_985], padded[eof.cropLeft])
        assertEquals(track.last(), padded[eof.copiedSamples - 1])
        assertEquals(0f, padded[eof.copiedSamples])
        assertEquals(0f, padded.last())
        assertEquals(eof.actualSamples, padded.size - eof.cropLeft - eof.cropRight)
    }

    @Test
    fun overlapAddUsesTrianglePrefixAndRequiresAscendingOutputs() {
        val prefix = streamingPlan.triangleWeightPrefix(257_985)
        val peak = HtdemucsStreamingPlan.WINDOW_SAMPLES / 2

        assertEquals(257_985, prefix.size)
        assertEquals(1f / peak, prefix.first())
        assertEquals(1f, prefix[peak - 1])
        assertEquals(1f, streamingPlan.triangleWeight(peak))
        assertEquals(85_996f / peak, prefix.last())
        assertEquals(
            1f / peak,
            streamingPlan.triangleWeight(HtdemucsStreamingPlan.WINDOW_SAMPLES - 1),
        )

        val first = HtdemucsWindowOutput(
            offset = 0,
            planarSamples = FloatArray(HtdemucsStreamingPlan.WINDOW_SAMPLES) { 1f },
        )
        val second = HtdemucsWindowOutput(
            offset = HtdemucsStreamingPlan.STRIDE_SAMPLES,
            planarSamples = FloatArray(HtdemucsStreamingPlan.WINDOW_SAMPLES) { 3f },
        )

        val result = streamingPlan.overlapAdd(
            trackSamples = TRACK_SAMPLES,
            outputPlaneCount = 1,
            windowOutputs = listOf(first, second),
        )
        val overlapStart = HtdemucsStreamingPlan.STRIDE_SAMPLES
        val firstWeight = streamingPlan.triangleWeight(overlapStart)
        val secondWeight = streamingPlan.triangleWeight(0)
        var expectedNumerator = 0f
        expectedNumerator += 1f * firstWeight
        expectedNumerator += 3f * secondWeight
        var expectedWeight = 0f
        expectedWeight += firstWeight
        expectedWeight += secondWeight

        assertEquals(1f, result.planarSamples[overlapStart - 1])
        assertEquals(expectedNumerator / expectedWeight, result.planarSamples[overlapStart])
        assertEquals(expectedWeight, result.accumulatedWeights[overlapStart])
        assertEquals(3f, result.planarSamples[HtdemucsStreamingPlan.WINDOW_SAMPLES])
        assertEquals(3f, result.planarSamples.last())
        assertThrows(IllegalArgumentException::class.java) {
            streamingPlan.overlapAdd(
                trackSamples = TRACK_SAMPLES,
                outputPlaneCount = 1,
                windowOutputs = listOf(second, first),
            )
        }
    }

    @Test
    fun globalNormalizationUsesStereoReferenceAndCorrectionOne() {
        val stereo = floatArrayOf(
            1f, 3f, 5f,
            3f, 5f, 7f,
        )

        val parameters = streamingPlan.globalNormalization(stereo)
        val normalized = streamingPlan.normalize(stereo, parameters)
        val restored = streamingPlan.denormalize(normalized, parameters)

        assertEquals(4f, parameters.mean)
        assertEquals(2f, parameters.sampleStandardDeviation)
        assertEquals(HtdemucsStreamingPlan.NORMALIZATION_EPSILON, parameters.epsilon)
        assertArrayEquals(floatArrayOf(-1.5f, -0.5f, 0.5f, -0.5f, 0.5f, 1.5f), normalized, 1e-6f)
        assertArrayEquals(stereo, restored, 1e-6f)

        val constant = floatArrayOf(3f, 3f, 3f, 3f)
        val constantParameters = streamingPlan.globalNormalization(constant)
        assertEquals(0f, constantParameters.sampleStandardDeviation)
        assertEquals(HtdemucsStreamingPlan.NORMALIZATION_EPSILON, constantParameters.divisor)
        assertTrue(streamingPlan.normalize(constant, constantParameters).all { it == 0f })
    }

    companion object {
        private const val TRACK_SAMPLES = 515_970
    }
}
