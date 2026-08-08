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

    @Test
    fun parallelStftAndIstftAreBitExactWithSerialExecution() {
        val configs = listOf(
            MdxDspConfig(nFft = 6144, dimF = 2048),
            MdxDspConfig(nFft = 7680, dimF = 3072),
            MdxDspConfig(nFft = 5120, dimF = 2560),
        )
        for (config in configs) {
            val waveform = Array(2) { channel ->
                FloatArray(config.chunkSize) { index ->
                    (((index * 37 + channel * 17) % 101) - 50) / 101f
                }
            }

            val serialTensor = MdxSpectrogram(config, workerCount = 1).use { serial ->
                serial.waveformToTensor(waveform)
            }
            val parallelTensor = MdxSpectrogram(config, workerCount = 4).use { parallel ->
                parallel.waveformToTensor(waveform)
            }
            assertArrayEquals(serialTensor, parallelTensor, 0f)

            val serialWaveform = MdxSpectrogram(config, workerCount = 1).use { serial ->
                serial.tensorToWaveform(serialTensor)
            }
            val parallelWaveform = MdxSpectrogram(config, workerCount = 4).use { parallel ->
                parallel.tensorToWaveform(serialTensor)
            }
            for (channel in serialWaveform.indices) {
                assertArrayEquals(serialWaveform[channel], parallelWaveform[channel], 0f)
            }
        }
    }

    @Test
    fun reusableNhwcPathMatchesLegacyForAllSentinels() {
        val configs = listOf(
            MdxDspConfig(nFft = 6144, dimF = 2048),
            MdxDspConfig(nFft = 7680, dimF = 3072),
            MdxDspConfig(nFft = 5120, dimF = 2560),
        )
        for (config in configs) {
            val waveform = stereoSineChunk(config)
            MdxSpectrogram(config, workerCount = 4).use { spectrogram ->
                val legacy = spectrogram.waveformToTensor(waveform)
                val directNhwc = FloatArray(config.tensorElementCount)
                spectrogram.waveformToNhwcTensorInto(waveform, directNhwc)
                assertArrayEquals(nchwToNhwc(legacy, config), directNhwc, 0f)

                val legacyWaveform = spectrogram.tensorToWaveform(legacy)
                val reusableWaveform = Array(2) { FloatArray(config.chunkSize) }
                spectrogram.nhwcTensorToWaveformInto(directNhwc, reusableWaveform)
                for (channel in legacyWaveform.indices) {
                    assertArrayEquals(legacyWaveform[channel], reusableWaveform[channel], 0f)
                }
            }
        }
    }

    private fun nchwToNhwc(input: FloatArray, config: MdxDspConfig): FloatArray {
        val output = FloatArray(input.size)
        for (channel in 0 until 4) {
            for (frequency in 0 until config.dimF) {
                for (frame in 0 until config.dimT) {
                    output[(frequency * config.dimT + frame) * 4 + channel] =
                        input[(channel * config.dimF + frequency) * config.dimT + frame]
                }
            }
        }
        return output
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
