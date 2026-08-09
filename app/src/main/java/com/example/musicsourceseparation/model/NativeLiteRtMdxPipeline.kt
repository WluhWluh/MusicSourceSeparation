package com.example.musicsourceseparation.model

class NativeLiteRtMdxPipeline(
    private val config: MdxDspConfig,
    modelPath: String,
    cpuThreads: Int = 4,
    workerCount: Int = 4,
    val backend: Backend = Backend.CPU,
) : AutoCloseable {
    private var handle = nativeCreate(
        modelPath = modelPath,
        nFft = config.nFft,
        hopLength = config.hopLength,
        dimF = config.dimF,
        dimT = config.dimT,
        chunkSize = config.chunkSize,
        workerCount = workerCount,
        cpuThreads = cpuThreads,
        boundedGpu = backend == Backend.BOUNDED_GPU,
    ).also {
        check(it != 0L) {
            "Native LiteRT MDX pipeline creation failed: ${nativeLastError()}"
        }
    }

    fun preprocessInput(waveform: Array<FloatArray>) {
        checkOpen()
        require(waveform.size == MdxDspConfig.STEREO_CHANNELS)
        require(waveform.all { it.size == config.chunkSize })
        check(nativePreprocessInput(handle, waveform[0], waveform[1])) { nativeLastError() }
    }

    fun run() {
        checkOpen()
        check(nativeRun(handle)) { nativeLastError() }
    }

    fun postprocessOutputInto(waveform: Array<FloatArray>) {
        checkOpen()
        require(waveform.size == MdxDspConfig.STEREO_CHANNELS)
        require(waveform.all { it.size == config.chunkSize })
        check(nativePostprocessOutput(handle, waveform[0], waveform[1])) { nativeLastError() }
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkOpen() {
        check(handle != 0L) { "Native LiteRT MDX pipeline is closed." }
    }

    private external fun nativeCreate(
        modelPath: String,
        nFft: Int,
        hopLength: Int,
        dimF: Int,
        dimT: Int,
        chunkSize: Int,
        workerCount: Int,
        cpuThreads: Int,
        boundedGpu: Boolean,
    ): Long

    private external fun nativePreprocessInput(
        handle: Long,
        left: FloatArray,
        right: FloatArray,
    ): Boolean

    private external fun nativeRun(handle: Long): Boolean

    private external fun nativePostprocessOutput(
        handle: Long,
        left: FloatArray,
        right: FloatArray,
    ): Boolean

    private external fun nativeDestroy(handle: Long)

    private external fun nativeLastError(): String

    enum class Backend {
        CPU,
        BOUNDED_GPU,
    }

    companion object {
        init {
            System.loadLibrary("mss_mdx_dsp")
        }
    }
}
