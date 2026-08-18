package com.example.musicsourceseparation.benchmark

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlin.math.abs
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TfcTdfStreamingDspInstrumentedTest {
    @Test
    fun reusableForwardAndInverseWorkspacesMatchAllocatingContract() {
        val dsp = TfcTdfStreamingDsp()
        val input = FloatArray(TfcTdfStreamingDsp.INPUT_SAMPLES * TfcTdfStreamingDsp.CHANNELS) {
            index -> (((index * 17) % 2_001) - 1_000) / 1_000f
        }
        val allocatingTensor = dsp.stftNhwc(input)
        val reusableTensor = FloatArray(TfcTdfStreamingDsp.TENSOR_ELEMENTS)
        dsp.stftNhwcInto(input, reusableTensor)
        assertMaxAbsDifference(allocatingTensor, reusableTensor, 0f)

        val allocatingOutput = dsp.istftInterleaved(allocatingTensor)
        val reusableOutput = FloatArray(
            TfcTdfStreamingDsp.INPUT_SAMPLES * TfcTdfStreamingDsp.CHANNELS,
        )
        dsp.istftInterleavedInto(reusableTensor, reusableOutput)
        assertMaxAbsDifference(allocatingOutput, reusableOutput, 0f)

        val secondInput = FloatArray(input.size) { index -> input[index] * 0.37f }
        dsp.stftNhwcInto(secondInput, reusableTensor)
        val secondReference = dsp.stftNhwc(secondInput)
        assertMaxAbsDifference(secondReference, reusableTensor, 0f)
        assertTrue(reusableTensor !== secondReference)
        assertEquals(TfcTdfStreamingDsp.TENSOR_ELEMENTS, reusableTensor.size)
    }

    private fun assertMaxAbsDifference(expected: FloatArray, actual: FloatArray, tolerance: Float) {
        assertEquals(expected.size, actual.size)
        var maximum = 0f
        for (index in expected.indices) {
            maximum = maxOf(maximum, abs(expected[index] - actual[index]))
        }
        assertTrue("maximum absolute difference was $maximum", maximum <= tolerance)
    }
}
