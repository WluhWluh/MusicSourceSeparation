package com.example.musicsourceseparation.audio

data class DecodedPcmAudio(
    val sampleRate: Int,
    val channelCount: Int,
    val pcm16: ByteArray,
) {
    val frameCount: Int = pcm16.size / (channelCount * Short.SIZE_BYTES)

    fun toStereoFloat(startFrame: Int = 0, maxFrames: Int? = null): Array<FloatArray> {
        require(startFrame >= 0) { "Start frame must not be negative." }
        require(startFrame <= frameCount) { "Start frame exceeds decoded audio length." }

        val frames = minOf(frameCount - startFrame, maxFrames ?: (frameCount - startFrame))
        val left = FloatArray(frames)
        val right = FloatArray(frames)

        var byteIndex = startFrame * channelCount * Short.SIZE_BYTES
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
