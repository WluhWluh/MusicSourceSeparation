package com.example.musicsourceseparation.audio

import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlin.math.roundToInt
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativePcm16InstrumentedTest {
    @Test
    fun nativeEncoderMatchesKotlinScalarQuantization() {
        val edges = floatArrayOf(
            -2f, -1.0001f, -1f, -0.99999f, -0.5f, -1f / 32767f,
            -0.5f / 32767f, 0f, 0.5f / 32767f, 1f / 32767f,
            0.5f, 0.99999f, 1f, 1.0001f, 2f,
        )
        var state = 0x13579bdf
        val input = FloatArray(100_003) { index ->
            if (index < edges.size) {
                edges[index]
            } else {
                state = state * 1_664_525 + 1_013_904_223
                ((state ushr 8) / 16_777_215f) * 3f - 1.5f
            }
        }
        val expected = ByteArray(input.size * Short.SIZE_BYTES)
        input.forEachIndexed { index, value ->
            val pcm = (value.coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            expected[index * 2] = (pcm and 0xff).toByte()
            expected[index * 2 + 1] = ((pcm ushr 8) and 0xff).toByte()
        }
        val actual = ByteArray(expected.size)

        assertEquals(expected.size, NativePcm16.encode(input, input.size, actual))
        assertArrayEquals(expected, actual)
        val expectedImplementation = if (Build.SUPPORTED_ABIS.first() == "arm64-v8a") {
            "neon-aarch64"
        } else {
            "scalar-native-fallback"
        }
        assertEquals(expectedImplementation, NativePcm16.implementation())
    }
}
