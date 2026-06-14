package com.example.musicsourceseparation.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

class MdxSpectrogramTest {
    @Test
    fun waveformToTensorUsesModelShape() {
        val config = MdxDspConfig()
        val spectrogram = MdxSpectrogram(config)
        val waveform = stereoSineChunk(config)

        val tensor = spectrogram.waveformToTensor(waveform)

        assertArrayEquals(intArrayOf(4, 2048, 256), spectrogram.tensorShape())
        assertEquals(4 * 2048 * 256, tensor.size)
    }

    @Test
    fun fullBandRoundTripReconstructsWaveform() {
        val baseConfig = MdxDspConfig()
        val fullBandConfig = baseConfig.copy(dimF = baseConfig.nBins)
        val spectrogram = MdxSpectrogram(fullBandConfig)
        val waveform = stereoSineChunk(fullBandConfig)

        val reconstructed = spectrogram.tensorToWaveform(spectrogram.waveformToTensor(waveform))

        val stats = errorStats(waveform, reconstructed)
        assertEquals(0.0, stats.meanAbsError, 1e-4)
        assertEquals(0.0, stats.maxAbsError, 2e-3)
    }

    private fun stereoSineChunk(config: MdxDspConfig): Array<FloatArray> {
        return Array(2) { channel ->
            FloatArray(config.chunkSize) { index ->
                val t = index.toDouble() / config.sampleRate
                val frequency = if (channel == 0) 220.0 else 330.0
                (0.15 * sin(2.0 * PI * frequency * t)).toFloat()
            }
        }
    }

    private fun errorStats(expected: Array<FloatArray>, actual: Array<FloatArray>): ErrorStats {
        var max = 0.0
        var sum = 0.0
        var count = 0L
        for (channel in expected.indices) {
            for (index in expected[channel].indices) {
                val error = abs((expected[channel][index] - actual[channel][index]).toDouble())
                max = maxOf(max, error)
                sum += error
                count += 1
            }
        }
        return ErrorStats(maxAbsError = max, meanAbsError = sum / count)
    }

    private data class ErrorStats(
        val maxAbsError: Double,
        val meanAbsError: Double,
    )
}
