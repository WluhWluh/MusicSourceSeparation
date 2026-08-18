package com.example.musicsourceseparation.benchmark

import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.PI
import kotlin.math.cos

/** TFC-TDF centered STFT/iSTFT contract used by the Phase 4 device benchmark. */
internal class TfcTdfStreamingDsp {
    private val fft = FloatFFT_1D(N_FFT.toLong())
    private val hann = FloatArray(N_FFT) { index ->
        (0.5 - 0.5 * cos(2.0 * PI * index / N_FFT)).toFloat()
    }
    private val padded = Array(CHANNELS) { FloatArray(PADDED_SAMPLES) }
    private val forwardBuffer = FloatArray(N_FFT * 2)
    private val inverseBuffer = FloatArray(N_FFT * 2)
    private val inversePadded = Array(CHANNELS) { FloatArray(PADDED_SAMPLES) }
    private val windowSum = FloatArray(PADDED_SAMPLES).also { sum ->
        for (frame in 0 until FRAMES) {
            val start = frame * HOP_LENGTH
            for (sample in 0 until N_FFT) {
                sum[start + sample] += hann[sample] * hann[sample]
            }
        }
    }

    fun stftNhwc(inputInterleaved: FloatArray): FloatArray {
        val tensor = FloatArray(TENSOR_ELEMENTS)
        stftNhwcInto(inputInterleaved, tensor)
        return tensor
    }

    fun stftNhwcInto(inputInterleaved: FloatArray, tensor: FloatArray) {
        require(inputInterleaved.size == INPUT_SAMPLES * CHANNELS)
        require(tensor.size == TENSOR_ELEMENTS)
        for (channel in 0 until CHANNELS) {
            val channelPadded = padded[channel]
            for (index in channelPadded.indices) {
                val sourceIndex = reflectIndex(index - CENTER_PAD, INPUT_SAMPLES)
                channelPadded[index] = inputInterleaved[sourceIndex * CHANNELS + channel]
            }
        }

        for (channel in 0 until CHANNELS) {
            val realChannel = channel
            val imaginaryChannel = channel + CHANNELS
            val channelPadded = padded[channel]
            for (frame in 0 until FRAMES) {
                forwardBuffer.fill(0f)
                val start = frame * HOP_LENGTH
                for (sample in 0 until N_FFT) {
                    forwardBuffer[sample] = channelPadded[start + sample] * hann[sample]
                }
                fft.realForwardFull(forwardBuffer)
                for (frequency in 0 until FREQUENCIES) {
                    val complexIndex = frequency * 2
                    tensor[tensorIndex(frequency, frame, realChannel)] =
                        forwardBuffer[complexIndex]
                    tensor[tensorIndex(frequency, frame, imaginaryChannel)] =
                        forwardBuffer[complexIndex + 1]
                }
            }
        }
    }

    fun istftInterleaved(tensorNhwc: FloatArray): FloatArray {
        val output = FloatArray(INPUT_SAMPLES * CHANNELS)
        istftInterleavedInto(tensorNhwc, output)
        return output
    }

    fun istftInterleavedInto(tensorNhwc: FloatArray, output: FloatArray) {
        require(tensorNhwc.size == TENSOR_ELEMENTS)
        require(output.size == INPUT_SAMPLES * CHANNELS)
        inversePadded.forEach { it.fill(0f) }
        for (channel in 0 until CHANNELS) {
            val realChannel = channel
            val imaginaryChannel = channel + CHANNELS
            val channelOutput = inversePadded[channel]
            for (frame in 0 until FRAMES) {
                inverseBuffer.fill(0f)
                for (frequency in 0 until FREQUENCIES) {
                    val real = tensorNhwc[tensorIndex(frequency, frame, realChannel)]
                    val imaginary = tensorNhwc[tensorIndex(frequency, frame, imaginaryChannel)]
                    inverseBuffer[frequency * 2] = real
                    inverseBuffer[frequency * 2 + 1] = imaginary
                    if (frequency in 1 until FREQUENCIES - 1) {
                        val mirror = N_FFT - frequency
                        inverseBuffer[mirror * 2] = real
                        inverseBuffer[mirror * 2 + 1] = -imaginary
                    }
                }
                fft.complexInverse(inverseBuffer, true)
                val start = frame * HOP_LENGTH
                for (sample in 0 until N_FFT) {
                    channelOutput[start + sample] +=
                        inverseBuffer[sample * 2] * hann[sample]
                }
            }
        }

        for (sample in 0 until INPUT_SAMPLES) {
            val paddedIndex = CENTER_PAD + sample
            val divisor = windowSum[paddedIndex]
            for (channel in 0 until CHANNELS) {
                output[sample * CHANNELS + channel] = if (divisor > 1e-8f) {
                    inversePadded[channel][paddedIndex] / divisor
                } else {
                    0f
                }
            }
        }
    }

    private fun tensorIndex(frequency: Int, frame: Int, channel: Int): Int {
        return (frequency * FRAMES + frame) * COMPLEX_CHANNELS + channel
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

    companion object {
        const val SAMPLE_RATE = 44_100
        const val CHANNELS = 2
        const val INPUT_SAMPLES = 130_048
        const val TRIM_SAMPLES = 5_120
        const val USEFUL_SAMPLES = 119_808
        const val N_FFT = 2_048
        const val HOP_LENGTH = 1_024
        const val FRAMES = 128
        const val FREQUENCIES = 1_025
        const val COMPLEX_CHANNELS = 4
        const val TENSOR_ELEMENTS = FREQUENCIES * FRAMES * COMPLEX_CHANNELS
        private const val CENTER_PAD = N_FFT / 2
        private const val PADDED_SAMPLES = INPUT_SAMPLES + N_FFT
    }
}
