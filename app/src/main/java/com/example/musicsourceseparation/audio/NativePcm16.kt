package com.example.musicsourceseparation.audio

object NativePcm16 {
    init {
        System.loadLibrary("mss_pcm16")
    }

    external fun encode(input: FloatArray, sampleCount: Int, output: ByteArray): Int

    external fun implementation(): String
}
