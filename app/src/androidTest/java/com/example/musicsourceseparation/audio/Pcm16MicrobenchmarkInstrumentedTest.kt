package com.example.musicsourceseparation.audio

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.roundToInt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Pcm16MicrobenchmarkInstrumentedTest {
    @Test
    fun compareScalarDirectAndNeonQuantization() {
        var state = 0x2468ace1
        val input = FloatArray(SAMPLE_COUNT) {
            state = state * 1_664_525 + 1_013_904_223
            ((state ushr 8) / 16_777_215f) * 2.4f - 1.2f
        }
        val scalar = ByteArray(SAMPLE_COUNT * Short.SIZE_BYTES)
        val native = ByteArray(scalar.size)
        val direct = ByteBuffer.allocateDirect(scalar.size).order(ByteOrder.LITTLE_ENDIAN)
        val samples = linkedMapOf(
            "scalar-byte-array" to mutableListOf<Double>(),
            "direct-byte-buffer" to mutableListOf<Double>(),
            "jni-neon" to mutableListOf<Double>(),
        )

        repeat(WARMUP_ROUNDS + MEASURED_ROUNDS) { round ->
            val order = MODES.drop(round % MODES.size) + MODES.take(round % MODES.size)
            order.forEach { mode ->
                val started = SystemClock.elapsedRealtimeNanos()
                when (mode) {
                    "scalar-byte-array" -> encodeScalar(input, scalar)
                    "direct-byte-buffer" -> encodeDirect(input, direct)
                    "jni-neon" -> NativePcm16.encode(input, input.size, native)
                }
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
                if (round >= WARMUP_ROUNDS) samples.getValue(mode) += elapsedMs
            }
        }

        assertArrayEquals(scalar, native)
        val directBytes = ByteArray(scalar.size)
        direct.duplicate().apply {
            position(0)
            limit(capacity())
            get(directBytes)
        }
        assertArrayEquals(scalar, directBytes)

        val report = JSONObject()
            .put("sampleCount", input.size)
            .put("byteCount", scalar.size)
            .put("warmupRounds", WARMUP_ROUNDS)
            .put("measuredRounds", MEASURED_ROUNDS)
            .put("order", "rotating")
            .put("nativeImplementation", NativePcm16.implementation())
            .put("modes", JSONObject().also { modes ->
                samples.forEach { (mode, values) -> modes.put(mode, summary(values)) }
            })
        Log.i(LOG_TAG, report.toString())
    }

    private fun encodeScalar(input: FloatArray, output: ByteArray) {
        input.forEachIndexed { index, value ->
            val pcm = quantize(value)
            output[index * 2] = (pcm and 0xff).toByte()
            output[index * 2 + 1] = ((pcm ushr 8) and 0xff).toByte()
        }
    }

    private fun encodeDirect(input: FloatArray, output: ByteBuffer) {
        output.clear()
        input.forEach { value -> output.putShort(quantize(value).toShort()) }
    }

    private fun quantize(value: Float): Int =
        (value.coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt()
            .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())

    private fun summary(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        return JSONObject()
            .put("samplesMs", JSONArray(values))
            .put("meanMs", values.average())
            .put("medianMs", sorted[sorted.size / 2])
            .put("p95NearestRankMs", sorted[ceil(sorted.size * 0.95).toInt() - 1])
            .put("minimumMs", sorted.first())
            .put("maximumMs", sorted.last())
    }

    private companion object {
        const val LOG_TAG = "MSS-Pcm16-Micro"
        const val SAMPLE_COUNT = 257_985 * 2
        const val WARMUP_ROUNDS = 2
        const val MEASURED_ROUNDS = 20
        val MODES = listOf("scalar-byte-array", "direct-byte-buffer", "jni-neon")
    }
}
