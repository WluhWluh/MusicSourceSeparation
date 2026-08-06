package com.example.musicsourceseparation.benchmark

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class HtdemucsCanonicalAudioInputPolicyTest {
    @Test
    fun fullLengthSelectionPreservesArbitraryDecodedFrameCount() {
        assertEquals(
            515_971,
            HtdemucsCanonicalAudioInputPolicy.selectFrames(
                availableFrames = 515_971,
                frameLimit = null,
                durationSeconds = null,
                sampleRate = 44_100,
            ),
        )
        assertEquals(
            123_457,
            HtdemucsCanonicalAudioInputPolicy.selectFrames(
                availableFrames = 515_971,
                frameLimit = 123_457,
                durationSeconds = null,
                sampleRate = 44_100,
            ),
        )
        assertEquals(
            1_323_000,
            HtdemucsCanonicalAudioInputPolicy.selectFrames(
                availableFrames = 2_000_000,
                frameLimit = null,
                durationSeconds = 30,
                sampleRate = 44_100,
            ),
        )
        assertEquals(
            7_938_000,
            HtdemucsCanonicalAudioInputPolicy.selectFrames(
                availableFrames = 8_000_000,
                frameLimit = null,
                durationSeconds = 180,
                sampleRate = 44_100,
            ),
        )
    }

    @Test
    fun selectionRejectsConflictingOrOversizedLimits() {
        assertThrows(IllegalArgumentException::class.java) {
            HtdemucsCanonicalAudioInputPolicy.selectFrames(100, 50, 1, 44_100)
        }
        assertThrows(IllegalArgumentException::class.java) {
            HtdemucsCanonicalAudioInputPolicy.selectFrames(100, 101, null, 44_100)
        }
    }

    @Test
    fun pcmChunkDuplicatesMonoAndSelectsFirstTwoMultichannelInputs() {
        val mono = byteArrayOf(1, 2, 3, 4)
        assertArrayEquals(
            byteArrayOf(1, 2, 1, 2, 3, 4, 3, 4),
            HtdemucsCanonicalAudioInputPolicy.stereoPcm16Chunk(mono, 1, 0, 2),
        )

        val threeChannel = byteArrayOf(
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
        )
        assertArrayEquals(
            byteArrayOf(7, 8, 9, 10),
            HtdemucsCanonicalAudioInputPolicy.stereoPcm16Chunk(threeChannel, 3, 1, 1),
        )
        assertEquals(
            "first-two-of-3-channels",
            HtdemucsCanonicalAudioInputPolicy.channelMapping(3),
        )
    }
}
