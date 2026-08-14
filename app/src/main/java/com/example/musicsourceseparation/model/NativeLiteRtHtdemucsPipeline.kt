package com.example.musicsourceseparation.model

class NativeLiteRtHtdemucsPipeline(
    val contract: HtdemucsNativeDspContract,
    modelPath: String,
    cpuThreads: Int = 4,
    workerCount: Int = 4,
) : HtdemucsDspSession {
    override val frameCount: Int = contract.spectrumFrames

    private var handle = nativeCreate(
        modelPath = modelPath,
        sourceCount = contract.sourceCount,
        windowSamples = contract.windowSamples,
        spectrumFrames = contract.spectrumFrames,
        fftSize = contract.fftSize,
        hopSize = contract.hopSize,
        dimF = contract.dimF,
        outerPadLeft = contract.normalization.outerPadLeft,
        framePadLeft = contract.normalization.framePadLeft,
        framePadRight = contract.normalization.framePadRight,
        centerTrim = contract.normalization.centerTrim,
        cpuThreads = cpuThreads,
        workerCount = workerCount,
    ).also {
        check(it != 0L) {
            "Native LiteRT HTDemucs pipeline creation failed: ${nativeLastError()}"
        }
    }

    override fun waveformToSpectrum(planarStereoWaveform: FloatArray): FloatArray =
        FloatArray(contract.spectrumInputElements).also { output ->
            waveformToSpectrumInto(planarStereoWaveform, output)
        }

    fun waveformToSpectrumInto(planarStereoWaveform: FloatArray, output: FloatArray) {
        checkOpen()
        require(planarStereoWaveform.size == contract.waveformInputElements)
        require(output.size == contract.spectrumInputElements)
        check(nativePreprocess(handle, planarStereoWaveform, output)) { nativeLastError() }
    }

    fun writeInputs(planarStereoWaveform: FloatArray, spectrum: FloatArray) {
        checkOpen()
        require(planarStereoWaveform.size == contract.waveformInputElements)
        require(spectrum.size == contract.spectrumInputElements)
        check(nativeWriteInputs(handle, planarStereoWaveform, spectrum)) { nativeLastError() }
    }

    fun run() {
        checkOpen()
        check(nativeRun(handle)) { nativeLastError() }
    }

    fun postprocessOutputsInto(output: FloatArray) {
        checkOpen()
        require(output.size == contract.waveformOutputElements)
        check(nativePostprocessOutputs(handle, output)) { nativeLastError() }
    }

    override fun frequencyToWaveform(
        packedFrequency: FloatArray,
        stemCount: Int,
    ): FloatArray = error(
        "The LiteRT C pipeline postprocesses its bound native output buffers directly.",
    )

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun checkOpen() {
        check(handle != 0L) { "Native LiteRT HTDemucs pipeline is closed." }
    }

    private external fun nativeCreate(
        modelPath: String,
        sourceCount: Int,
        windowSamples: Int,
        spectrumFrames: Int,
        fftSize: Int,
        hopSize: Int,
        dimF: Int,
        outerPadLeft: Int,
        framePadLeft: Int,
        framePadRight: Int,
        centerTrim: Int,
        cpuThreads: Int,
        workerCount: Int,
    ): Long

    private external fun nativePreprocess(
        handle: Long,
        waveform: FloatArray,
        spectrum: FloatArray,
    ): Boolean

    private external fun nativeWriteInputs(
        handle: Long,
        waveform: FloatArray,
        spectrum: FloatArray,
    ): Boolean

    private external fun nativeRun(handle: Long): Boolean

    private external fun nativePostprocessOutputs(handle: Long, output: FloatArray): Boolean

    private external fun nativeDestroy(handle: Long)

    private external fun nativeLastError(): String

    companion object {
        init {
            System.loadLibrary("mss_htdemucs_dsp")
        }
    }
}
