package com.example.musicsourceseparation.benchmark

internal object HtdemucsCanonicalAudioInputPolicy {
    fun selectFrames(
        availableFrames: Int,
        frameLimit: Int?,
        durationSeconds: Int?,
        sampleRate: Int,
    ): Int {
        require(availableFrames > 1) { "Decoded audio must contain at least two frames." }
        require(sampleRate > 0) { "Sample rate must be positive." }
        require(frameLimit == null || durationSeconds == null) {
            "Frame and duration limits are mutually exclusive."
        }
        val requestedFrames = when {
            frameLimit != null -> {
                require(frameLimit > 1) { "Frame limit must exceed one frame." }
                frameLimit
            }
            durationSeconds != null -> {
                require(durationSeconds > 0) { "Duration limit must be positive." }
                Math.multiplyExact(durationSeconds, sampleRate)
            }
            else -> availableFrames
        }
        require(requestedFrames <= availableFrames) {
            "Decoded audio has $availableFrames frames, fewer than requested $requestedFrames."
        }
        return requestedFrames
    }

    fun stereoPcm16Chunk(
        sourcePcm16: ByteArray,
        sourceChannelCount: Int,
        startFrame: Int,
        frameCount: Int,
    ): ByteArray {
        require(sourceChannelCount > 0) { "Source channel count must be positive." }
        val sourceBytesPerFrame = Math.multiplyExact(sourceChannelCount, Short.SIZE_BYTES)
        require(sourcePcm16.size % sourceBytesPerFrame == 0) {
            "Source PCM must contain complete frames."
        }
        val sourceFrameCount = sourcePcm16.size / sourceBytesPerFrame
        require(startFrame >= 0 && frameCount >= 0 && startFrame <= sourceFrameCount - frameCount) {
            "Requested PCM frame range is outside the decoded source."
        }

        val output = ByteArray(Math.multiplyExact(frameCount, STEREO_BYTES_PER_FRAME))
        repeat(frameCount) { frame ->
            val sourceOffset = (startFrame + frame) * sourceBytesPerFrame
            val outputOffset = frame * STEREO_BYTES_PER_FRAME
            sourcePcm16.copyInto(output, outputOffset, sourceOffset, sourceOffset + Short.SIZE_BYTES)
            val rightOffset = if (sourceChannelCount == 1) {
                sourceOffset
            } else {
                sourceOffset + Short.SIZE_BYTES
            }
            sourcePcm16.copyInto(
                output,
                outputOffset + Short.SIZE_BYTES,
                rightOffset,
                rightOffset + Short.SIZE_BYTES,
            )
        }
        return output
    }

    fun channelMapping(sourceChannelCount: Int): String = when (sourceChannelCount) {
        1 -> "mono-duplicated-to-stereo"
        2 -> "stereo-passthrough"
        else -> {
            require(sourceChannelCount > 0) { "Source channel count must be positive." }
            "first-two-of-$sourceChannelCount-channels"
        }
    }

    private const val STEREO_BYTES_PER_FRAME = 2 * Short.SIZE_BYTES
}
