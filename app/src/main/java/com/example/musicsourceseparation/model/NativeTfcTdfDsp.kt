package com.example.musicsourceseparation.model

/** Native DSP for the frozen compact TFC-TDF streaming tensor contract. */
class NativeTfcTdfDsp(
    val workerCount: Int,
    val mode: Mode,
) : AutoCloseable {
    init {
        require(workerCount in 1..MAX_WORKERS)
    }

    private var handle = nativeCreate(workerCount, mode == Mode.PACKED_REAL).also {
        require(it != 0L) { "Native TFC-TDF DSP plan creation failed: ${nativeLastError()}" }
    }

    fun stftNhwcInto(inputInterleaved: FloatArray, tensorNhwc: FloatArray) {
        check(handle != 0L) { "Native TFC-TDF DSP is closed." }
        require(inputInterleaved.size == INPUT_ELEMENTS)
        require(tensorNhwc.size == TENSOR_ELEMENTS)
        check(nativeStft(handle, inputInterleaved, tensorNhwc)) { nativeLastError() }
    }

    fun istftInterleavedInto(tensorNhwc: FloatArray, outputInterleaved: FloatArray) {
        check(handle != 0L) { "Native TFC-TDF DSP is closed." }
        require(tensorNhwc.size == TENSOR_ELEMENTS)
        require(outputInterleaved.size == INPUT_ELEMENTS)
        check(nativeIstft(handle, tensorNhwc, outputInterleaved)) { nativeLastError() }
    }

    fun istftResidualInto(
        tensorNhwc: FloatArray,
        inputInterleaved: FloatArray,
        trimSamples: Int,
        actualSamples: Int,
        outputInterleaved: FloatArray,
    ) {
        check(handle != 0L) { "Native TFC-TDF DSP is closed." }
        require(tensorNhwc.size == TENSOR_ELEMENTS)
        require(inputInterleaved.size == INPUT_ELEMENTS)
        require(trimSamples >= 0)
        require(actualSamples >= 0)
        require(trimSamples + actualSamples <= INPUT_SAMPLES)
        require(outputInterleaved.size == actualSamples * CHANNELS)
        check(
            nativeIstftResidual(
                handle,
                tensorNhwc,
                inputInterleaved,
                trimSamples,
                actualSamples,
                outputInterleaved,
            ),
        ) { nativeLastError() }
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeCreate(workerCount: Int, packedReal: Boolean): Long

    private external fun nativeStft(
        handle: Long,
        inputInterleaved: FloatArray,
        tensorNhwc: FloatArray,
    ): Boolean

    private external fun nativeIstft(
        handle: Long,
        tensorNhwc: FloatArray,
        outputInterleaved: FloatArray,
    ): Boolean

    private external fun nativeIstftResidual(
        handle: Long,
        tensorNhwc: FloatArray,
        inputInterleaved: FloatArray,
        trimSamples: Int,
        actualSamples: Int,
        outputInterleaved: FloatArray,
    ): Boolean

    private external fun nativeDestroy(handle: Long)

    private external fun nativeLastError(): String

    enum class Mode(val profileId: String) {
        FULL_COMPLEX("native-pocketfft-full-complex"),
        PACKED_REAL("native-pocketfft-packed-real"),
    }

    companion object {
        const val SAMPLE_RATE = 44_100
        const val CHANNELS = 2
        const val INPUT_SAMPLES = 130_048
        const val N_FFT = 2_048
        const val HOP_LENGTH = 1_024
        const val FRAMES = 128
        const val FREQUENCIES = 1_025
        const val COMPLEX_CHANNELS = 4
        const val INPUT_ELEMENTS = INPUT_SAMPLES * CHANNELS
        const val TENSOR_ELEMENTS = FREQUENCIES * FRAMES * COMPLEX_CHANNELS
        const val MAX_WORKERS = 4

        init {
            System.loadLibrary("mss_tfc_tdf_dsp")
        }
    }
}
