package com.example.musicsourceseparation.model

import java.io.File
import java.io.FileInputStream
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test

class HtdemucsDspFixtureTest {
    @Test
    fun waveformToSpectrumMatchesCanonicalTorchFixture() {
        assumeCanonicalFixturesPresent()
        val dsp = HtdemucsDsp(WINDOW_SAMPLES)
        val waveform = readRawFixture(
            fileName = "waveform_input.f32le.raw",
            elementCount = HtdemucsDsp.CHANNEL_COUNT * WINDOW_SAMPLES,
            expectedSha256 = WAVEFORM_INPUT_SHA256,
        )

        val actual = dsp.waveformToSpectrum(waveform)

        assertEquals(FRAME_COUNT, dsp.frameCount)
        assertEquals(
            HtdemucsDsp.FEATURE_COUNT * HtdemucsDsp.FREQUENCY_BINS * FRAME_COUNT,
            actual.size,
        )
        val stats = compareWithRawFixture(
            actual = actual,
            fileName = "spectrum_input.f32le.raw",
            expectedSha256 = SPECTRUM_GOLDEN_SHA256,
        )
        printStats("waveform_input -> spectrum_golden", stats)
        assertQualityGate(stats)
    }

    @Test
    fun frequencyToWaveformMatchesCanonicalTorchFixture() {
        assumeCanonicalFixturesPresent()
        val dsp = HtdemucsDsp(WINDOW_SAMPLES)
        val frequency = readRawFixture(
            fileName = "frequency_golden.f32le.raw",
            elementCount = HtdemucsDsp.DEFAULT_STEM_COUNT * HtdemucsDsp.FEATURE_COUNT *
                HtdemucsDsp.FREQUENCY_BINS * FRAME_COUNT,
            expectedSha256 = FREQUENCY_GOLDEN_SHA256,
        )

        val actual = dsp.frequencyToWaveform(frequency)

        assertEquals(
            HtdemucsDsp.DEFAULT_STEM_COUNT * HtdemucsDsp.CHANNEL_COUNT * WINDOW_SAMPLES,
            actual.size,
        )
        val stats = compareWithRawFixture(
            actual = actual,
            fileName = "frequency_waveform_golden.f32le.raw",
            expectedSha256 = FREQUENCY_WAVEFORM_GOLDEN_SHA256,
        )
        printStats("frequency_output_golden -> frequency_waveform_golden", stats)
        assertQualityGate(stats)
    }

    @Test
    fun parallelLaneIstftIsBitExactForFourAndSixStems() {
        listOf(4, HtdemucsDsp.DEFAULT_STEM_COUNT).forEach { stemCount ->
            val frequency = FloatArray(
                stemCount * HtdemucsDsp.FEATURE_COUNT * HtdemucsDsp.FREQUENCY_BINS *
                    SYNTHETIC_FRAME_COUNT,
            ) { index ->
                (((index * 37L + 11L) % 257L).toInt() - 128) * 1e-5f
            }
            val serial = reconstructFrequency(
                frequency = frequency,
                stemCount = stemCount,
                mode = HtdemucsDsp.IstftMode.SERIAL,
                workers = 1,
                windowSamples = SYNTHETIC_WINDOW_SAMPLES,
            )
            listOf(2, 4).forEach { workers ->
                HtdemucsDsp(
                    windowSamples = SYNTHETIC_WINDOW_SAMPLES,
                    istftMode = HtdemucsDsp.IstftMode.PARALLEL_LANES,
                    istftWorkers = workers,
                ).use { dsp ->
                    repeat(2) { repetition ->
                        val parallel = dsp.frequencyToWaveform(frequency, stemCount)
                        assertFloatBitsEqual(
                            expected = serial,
                            actual = parallel,
                            label = "$stemCount stems, $workers workers, repetition $repetition",
                        )
                    }
                }
            }
        }
        val leakedWorkers = Thread.getAllStackTraces().keys.filter { thread ->
            thread.isAlive && thread.name.startsWith("htdemucs-istft-")
        }
        assertTrue("Parallel iSTFT workers leaked after close: $leakedWorkers", leakedWorkers.isEmpty())
    }

    @Test
    fun closedParallelDspRejectsReuseAndPreservesInterruptStatus() {
        val dsp = HtdemucsDsp(
            windowSamples = SYNTHETIC_WINDOW_SAMPLES,
            istftMode = HtdemucsDsp.IstftMode.PARALLEL_LANES,
            istftWorkers = 2,
        )
        try {
            Thread.currentThread().interrupt()
            dsp.close()
            assertTrue("close() cleared the caller interrupt status", Thread.interrupted())
            assertThrows(IllegalStateException::class.java) {
                dsp.frequencyToWaveform(
                    FloatArray(
                        4 * HtdemucsDsp.FEATURE_COUNT * HtdemucsDsp.FREQUENCY_BINS *
                            SYNTHETIC_FRAME_COUNT,
                    ),
                    4,
                )
            }
        } finally {
            Thread.interrupted()
            dsp.close()
        }
        val leakedWorkers = Thread.getAllStackTraces().keys.filter { thread ->
            thread.isAlive && thread.name.startsWith("htdemucs-istft-")
        }
        assertTrue("Parallel iSTFT workers leaked after close: $leakedWorkers", leakedWorkers.isEmpty())
    }

    @Test
    fun branchCombinationMatchesCanonicalTorchFixture() {
        assumeCanonicalFixturesPresent()
        val elementCount = HtdemucsDsp.DEFAULT_STEM_COUNT * HtdemucsDsp.CHANNEL_COUNT *
            WINDOW_SAMPLES
        val frequencyWaveform = readRawFixture(
            fileName = "frequency_waveform_golden.f32le.raw",
            elementCount = elementCount,
            expectedSha256 = FREQUENCY_WAVEFORM_GOLDEN_SHA256,
        )
        val timeWaveform = readRawFixture(
            fileName = "waveform_golden.f32le.raw",
            elementCount = elementCount,
            expectedSha256 = TIME_WAVEFORM_GOLDEN_SHA256,
        )

        val actual = HtdemucsDsp(WINDOW_SAMPLES).combineBranches(
            frequencyWaveform,
            timeWaveform,
        )

        val stats = compareWithRawFixture(
            actual = actual,
            fileName = "combined_golden.f32le.raw",
            expectedSha256 = COMBINED_GOLDEN_SHA256,
        )
        printStats("frequency_waveform + time_waveform -> combined_golden", stats)
        assertQualityGate(stats)
    }

    private fun readRawFixture(
        fileName: String,
        elementCount: Int,
        expectedSha256: String,
    ): FloatArray {
        val file = fixtureFile(fileName)
        assertEquals(elementCount.toLong() * Float.SIZE_BYTES, file.length())
        assertEquals(expectedSha256, sha256(file))
        FileInputStream(file).channel.use { channel ->
            val floats = channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()
            return FloatArray(elementCount).also(floats::get)
        }
    }

    private fun compareWithRawFixture(
        actual: FloatArray,
        fileName: String,
        expectedSha256: String,
    ): ErrorStats {
        val file = fixtureFile(fileName)
        assertEquals(actual.size.toLong() * Float.SIZE_BYTES, file.length())
        assertEquals(expectedSha256, sha256(file))
        FileInputStream(file).channel.use { channel ->
            val expected = channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()
            var finite = true
            var maxAbsoluteError = 0.0
            var sumAbsoluteError = 0.0
            var signalSquareSum = 0.0
            var candidateSquareSum = 0.0
            var errorSquareSum = 0.0
            var dotProduct = 0.0
            actual.forEach { candidateFloat ->
                val reference = expected.get().toDouble()
                val candidate = candidateFloat.toDouble()
                finite = finite && reference.isFinite() && candidate.isFinite()
                val delta = candidate - reference
                val absoluteError = abs(delta)
                maxAbsoluteError = maxOf(maxAbsoluteError, absoluteError)
                sumAbsoluteError += absoluteError
                signalSquareSum += reference * reference
                candidateSquareSum += candidate * candidate
                errorSquareSum += delta * delta
                dotProduct += reference * candidate
            }
            assertEquals(0, expected.remaining())
            val count = actual.size.toDouble()
            val referenceRms = sqrt(signalSquareSum / count)
            val candidateRms = sqrt(candidateSquareSum / count)
            val errorRms = sqrt(errorSquareSum / count)
            return ErrorStats(
                finite = finite,
                maxAbsoluteError = maxAbsoluteError,
                meanAbsoluteError = sumAbsoluteError / count,
                rootMeanSquareError = errorRms,
                signalToNoiseDb = 20.0 * log10(
                    maxOf(referenceRms, 1e-30) / maxOf(errorRms, 1e-30),
                ),
                referenceRms = referenceRms,
                candidateRms = candidateRms,
                rmsRatio = candidateRms / maxOf(referenceRms, 1e-30),
                leastSquaresGain = dotProduct / maxOf(signalSquareSum, 1e-30),
            )
        }
    }

    private fun reconstructFrequency(
        frequency: FloatArray,
        stemCount: Int,
        mode: HtdemucsDsp.IstftMode,
        workers: Int,
        windowSamples: Int = WINDOW_SAMPLES,
    ): FloatArray = HtdemucsDsp(
        windowSamples = windowSamples,
        istftMode = mode,
        istftWorkers = workers,
    ).use { dsp ->
        dsp.frequencyToWaveform(frequency, stemCount)
    }

    private fun assertFloatBitsEqual(
        expected: FloatArray,
        actual: FloatArray,
        label: String,
    ) {
        assertEquals("$label element count", expected.size, actual.size)
        var firstNonFinite = -1
        for (index in expected.indices) {
            if (expected[index].toRawBits() != actual[index].toRawBits()) {
                assertEquals(
                    "$label first raw-float mismatch at index $index",
                    expected[index].toRawBits(),
                    actual[index].toRawBits(),
                )
            }
            if (firstNonFinite < 0 && !actual[index].isFinite()) firstNonFinite = index
        }
        assertEquals("$label first non-finite output index", -1, firstNonFinite)
    }

    private fun printStats(label: String, stats: ErrorStats) {
        println(
            "$label: finite=${stats.finite}, SNR=${stats.signalToNoiseDb} dB, " +
                "maxAbs=${stats.maxAbsoluteError}, RMSE=${stats.rootMeanSquareError}, " +
                "MAE=${stats.meanAbsoluteError}, referenceRms=${stats.referenceRms}, " +
                "candidateRms=${stats.candidateRms}, rmsRatio=${stats.rmsRatio}, " +
                "leastSquaresGain=${stats.leastSquaresGain}",
        )
    }

    private fun assertQualityGate(stats: ErrorStats) {
        assertTrue("DSP output contains NaN or infinity: $stats", stats.finite)
        assertTrue(
            "DSP SNR ${stats.signalToNoiseDb} dB is below $MINIMUM_SNR_DB dB: $stats",
            stats.signalToNoiseDb >= MINIMUM_SNR_DB,
        )
        assertTrue(
            "DSP max absolute error ${stats.maxAbsoluteError} exceeds " +
                "$MAXIMUM_ABSOLUTE_ERROR: $stats",
            stats.maxAbsoluteError <= MAXIMUM_ABSOLUTE_ERROR,
        )
    }

    private fun assumeCanonicalFixturesPresent() {
        assumeTrue(
            "Canonical HTDemucs raw fixtures are a local integration-test dependency.",
            fixtureDirectory() != null,
        )
    }

    private fun fixtureFile(fileName: String): File = File(
        requireNotNull(fixtureDirectory()) { "Canonical HTDemucs fixture directory is missing." },
        fileName,
    )

    private fun fixtureDirectory(): File? {
        val relative = "models/demucs/generated/$MODEL_ID/fixtures"
        return sequenceOf(File(relative), File("../$relative"))
            .firstOrNull(File::isDirectory)
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private data class ErrorStats(
        val finite: Boolean,
        val maxAbsoluteError: Double,
        val meanAbsoluteError: Double,
        val rootMeanSquareError: Double,
        val signalToNoiseDb: Double,
        val referenceRms: Double,
        val candidateRms: Double,
        val rmsRatio: Double,
        val leastSquaresGain: Double,
    )

    companion object {
        private const val MODEL_ID = "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0"
        private const val WINDOW_SAMPLES = 343_980
        private const val FRAME_COUNT = 336
        private const val SYNTHETIC_WINDOW_SAMPLES = 4_096
        private const val SYNTHETIC_FRAME_COUNT = 4
        private const val MINIMUM_SNR_DB = 80.0
        private const val MAXIMUM_ABSOLUTE_ERROR = 1e-3
        private const val WAVEFORM_INPUT_SHA256 =
            "9515d42e72e96036e34f0193d1302e8cccbf80c4e9e7084d1fcba26e17fea40e"
        private const val SPECTRUM_GOLDEN_SHA256 =
            "9d01b6cf6943c61724f6bb7edefc8f6156f9749f1ae9bc4f84508a62738ee5e0"
        private const val FREQUENCY_GOLDEN_SHA256 =
            "86e916da6041bf0038adcd770ed986ddbe54e52e264d934f11d8186b7616dcbb"
        private const val FREQUENCY_WAVEFORM_GOLDEN_SHA256 =
            "5bb82c18f2f778b6cf66c28c663d76921f37c4bb69128351e3ce0b4229498ba8"
        private const val TIME_WAVEFORM_GOLDEN_SHA256 =
            "be0934e5be603d5fd17a69cf92d3786e21053128cdfb4fa0f4f78166aebea474"
        private const val COMBINED_GOLDEN_SHA256 =
            "7d5fd3585cbdbb46551e9e05883da9aa880702a72efaa71402072957d037264a"
    }
}
