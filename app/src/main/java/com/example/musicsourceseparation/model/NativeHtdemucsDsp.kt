package com.example.musicsourceseparation.model

class NativeHtdemucsDsp(
    val contract: HtdemucsNativeDspContract,
    workerCount: Int = 4,
) : HtdemucsDspSession {
    override val frameCount: Int = contract.spectrumFrames

    private var handle = nativeCreate(
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
        workerCount = workerCount,
    ).also { require(it != 0L) { "Native HTDemucs DSP plan creation failed." } }

    override fun waveformToSpectrum(planarStereoWaveform: FloatArray): FloatArray =
        FloatArray(contract.spectrumInputElements).also { output ->
            waveformToSpectrumInto(planarStereoWaveform, output)
        }

    fun waveformToSpectrumInto(planarStereoWaveform: FloatArray, output: FloatArray) {
        check(handle != 0L) { "Native HTDemucs DSP is closed." }
        require(planarStereoWaveform.size == contract.waveformInputElements)
        require(output.size == contract.spectrumInputElements)
        check(nativePreprocess(handle, planarStereoWaveform, output))
    }

    override fun frequencyToWaveform(
        packedFrequency: FloatArray,
        stemCount: Int,
    ): FloatArray {
        require(stemCount == contract.sourceCount)
        return FloatArray(contract.waveformOutputElements).also { output ->
            frequencyToWaveformInto(packedFrequency, output)
        }
    }

    fun frequencyToWaveformInto(packedFrequency: FloatArray, output: FloatArray) {
        check(handle != 0L) { "Native HTDemucs DSP is closed." }
        require(packedFrequency.size == contract.frequencyOutputElements)
        require(output.size == contract.waveformOutputElements)
        check(nativePostprocess(handle, packedFrequency, null, output))
    }

    fun postprocessInto(
        packedFrequency: FloatArray,
        timeWaveform: FloatArray,
        output: FloatArray,
    ) {
        check(handle != 0L) { "Native HTDemucs DSP is closed." }
        require(packedFrequency.size == contract.frequencyOutputElements)
        require(timeWaveform.size == contract.waveformOutputElements)
        require(output.size == contract.waveformOutputElements)
        check(nativePostprocess(handle, packedFrequency, timeWaveform, output))
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeCreate(
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
        workerCount: Int,
    ): Long

    private external fun nativePreprocess(
        handle: Long,
        waveform: FloatArray,
        spectrum: FloatArray,
    ): Boolean

    private external fun nativePostprocess(
        handle: Long,
        frequency: FloatArray,
        timeWaveform: FloatArray?,
        output: FloatArray,
    ): Boolean

    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("mss_htdemucs_dsp")
        }
    }
}
