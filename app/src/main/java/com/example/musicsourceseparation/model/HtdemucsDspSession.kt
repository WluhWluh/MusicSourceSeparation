package com.example.musicsourceseparation.model

interface HtdemucsDspSession : AutoCloseable {
    val frameCount: Int

    fun waveformToSpectrum(planarStereoWaveform: FloatArray): FloatArray

    fun frequencyToWaveform(packedFrequency: FloatArray, stemCount: Int): FloatArray
}
