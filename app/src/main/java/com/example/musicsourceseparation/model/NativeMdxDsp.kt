package com.example.musicsourceseparation.model

class NativeMdxDsp(
    private val config: MdxDspConfig,
    workerCount: Int,
) : AutoCloseable {
    private var handle = nativeCreate(
        config.nFft,
        config.hopLength,
        config.dimF,
        config.dimT,
        config.chunkSize,
        workerCount,
    ).also { require(it != 0L) { "Native MDX DSP plan creation failed." } }

    fun waveformToNhwcTensorInto(waveform: Array<FloatArray>, tensor: FloatArray) {
        check(handle != 0L)
        require(waveform.size == 2 && waveform.all { it.size == config.chunkSize })
        require(tensor.size == config.tensorElementCount)
        check(nativePreprocess(handle, waveform[0], waveform[1], tensor))
    }

    fun nhwcTensorToWaveformInto(tensor: FloatArray, waveform: Array<FloatArray>) {
        check(handle != 0L)
        require(tensor.size == config.tensorElementCount)
        require(waveform.size == 2 && waveform.all { it.size == config.chunkSize })
        check(nativePostprocess(handle, tensor, waveform[0], waveform[1]))
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeCreate(
        nFft: Int,
        hopLength: Int,
        dimF: Int,
        dimT: Int,
        chunkSize: Int,
        workerCount: Int,
    ): Long

    private external fun nativePreprocess(
        handle: Long,
        left: FloatArray,
        right: FloatArray,
        tensorNhwc: FloatArray,
    ): Boolean

    private external fun nativePostprocess(
        handle: Long,
        tensorNhwc: FloatArray,
        left: FloatArray,
        right: FloatArray,
    ): Boolean

    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("mss_mdx_dsp")
        }
    }
}
