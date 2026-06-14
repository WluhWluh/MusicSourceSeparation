package com.example.musicsourceseparation.audio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.math.sin

class DecodedPcmAudioTest {
    @Test
    fun resampleToConvertsSampleRateAndPreservesDuration() {
        val source = sineAudio(sampleRate = 48_000, frameCount = 48_000)

        val resampled = source.resampleTo(44_100)

        assertEquals(44_100, resampled.sampleRate)
        assertEquals(2, resampled.channelCount)
        assertEquals(44_100, resampled.frameCount)
        assertTrue(resampled.toStereoFloat()[0].any { kotlin.math.abs(it) > 0.01f })
    }

    @Test
    fun resampleToSameSampleRateReturnsSameInstance() {
        val source = sineAudio(sampleRate = 44_100, frameCount = 1_024)

        assertSame(source, source.resampleTo(44_100))
    }

    private fun sineAudio(sampleRate: Int, frameCount: Int): DecodedPcmAudio {
        val bytes = ByteArray(frameCount * 2 * Short.SIZE_BYTES)
        var offset = 0
        for (frame in 0 until frameCount) {
            val t = frame.toDouble() / sampleRate
            val left = (sin(2.0 * PI * 440.0 * t) * Short.MAX_VALUE * 0.25).roundToInt()
            val right = (sin(2.0 * PI * 660.0 * t) * Short.MAX_VALUE * 0.25).roundToInt()
            offset = writeLittleEndianShort(bytes, offset, left)
            offset = writeLittleEndianShort(bytes, offset, right)
        }
        return DecodedPcmAudio(
            sampleRate = sampleRate,
            channelCount = 2,
            pcm16 = bytes,
        )
    }

    private fun writeLittleEndianShort(bytes: ByteArray, offset: Int, value: Int): Int {
        bytes[offset] = (value and 0xFF).toByte()
        bytes[offset + 1] = ((value ushr 8) and 0xFF).toByte()
        return offset + Short.SIZE_BYTES
    }
}
