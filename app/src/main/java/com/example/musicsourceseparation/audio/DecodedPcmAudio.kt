package com.example.musicsourceseparation.audio

data class DecodedPcmAudio(
    val sampleRate: Int,
    val channelCount: Int,
    val pcm16: ByteArray,
) {
    val frameCount: Int = pcm16.size / (channelCount * Short.SIZE_BYTES)

    fun toStereoFloat(maxFrames: Int? = null): Array<FloatArray> {
        val frames = minOf(frameCount, maxFrames ?: frameCount)
        val left = FloatArray(frames)
        val right = FloatArray(frames)

        var byteIndex = 0
        for (frame in 0 until frames) {
            val leftSample = readLittleEndianShort(byteIndex)
            left[frame] = leftSample / 32768f

            val rightSample = if (channelCount > 1) {
                readLittleEndianShort(byteIndex + Short.SIZE_BYTES)
            } else {
                leftSample
            }
            right[frame] = rightSample / 32768f

            byteIndex += channelCount * Short.SIZE_BYTES
        }

        return arrayOf(left, right)
    }

    private fun readLittleEndianShort(byteIndex: Int): Short {
        val low = pcm16[byteIndex].toInt() and 0xFF
        val high = pcm16[byteIndex + 1].toInt()
        return ((high shl 8) or low).toShort()
    }
}
