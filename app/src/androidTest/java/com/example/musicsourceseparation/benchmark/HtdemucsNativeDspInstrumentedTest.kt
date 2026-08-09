package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.model.HtdemucsDsp
import com.example.musicsourceseparation.model.HtdemucsNativeDspContract
import com.example.musicsourceseparation.model.HtdemucsNativeDspContracts
import com.example.musicsourceseparation.model.NativeHtdemucsDsp
import java.io.File
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class HtdemucsNativeDspInstrumentedTest {
    @Test
    fun validateAllPublishedContracts() {
        val arguments = InstrumentationRegistry.getArguments()
        val measuredRuns = arguments.getString(ARG_MEASURED_RUNS)?.toIntOrNull() ?: 3
        require(measuredRuns in 1..20)
        val runId = arguments.getString(ARG_RUN_ID) ?: "run-${System.currentTimeMillis()}"
        require(RUN_ID.matches(runId))
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark/htdemucs-native-dsp/$runId")
        require(root.deleteRecursively() && root.mkdirs())

        val waveform = deterministicWaveform(HtdemucsNativeDspContracts.OFFICIAL_6S)
        val kotlinDsp = HtdemucsDsp(
            windowSamples = HtdemucsNativeDspContracts.OFFICIAL_6S.windowSamples,
            istftMode = HtdemucsDsp.IstftMode.PARALLEL_LANES,
            istftWorkers = 4,
            reuseIoWorkspaces = true,
        )
        val kotlinSpectrum = kotlinDsp.waveformToSpectrum(waveform).copyOf()
        val results = JSONArray()
        try {
            HtdemucsNativeDspContracts.ALL.forEach { contract ->
                results.put(validateContract(contract, waveform, kotlinSpectrum, kotlinDsp, measuredRuns))
                System.gc()
            }
        } finally {
            kotlinDsp.close()
        }
        val report = JSONObject()
            .put("schemaVersion", 1)
            .put("status", "complete")
            .put("runId", runId)
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL)
                .put("device", Build.DEVICE)
                .put("socManufacturer", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else JSONObject.NULL)
                .put("socModel", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else JSONObject.NULL)
                .put("sdk", Build.VERSION.SDK_INT)
                .put("abi", Build.SUPPORTED_ABIS.firstOrNull() ?: JSONObject.NULL))
            .put("gate", JSONObject()
                .put("minimumSnrDb", MINIMUM_SNR_DB)
                .put("maximumAbsoluteError", MAXIMUM_ABSOLUTE_ERROR))
            .put("measuredRuns", measuredRuns)
            .put("contracts", results)
            .put("process", JSONObject()
                .put("pssKb", Debug.getPss())
                .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize()))
        File(root, "report.json").writeText(report.toString(2), Charsets.UTF_8)
    }

    private fun validateContract(
        contract: HtdemucsNativeDspContract,
        waveform: FloatArray,
        kotlinSpectrum: FloatArray,
        kotlinDsp: HtdemucsDsp,
        measuredRuns: Int,
    ): JSONObject {
        NativeHtdemucsDsp(contract, workerCount = 4).use { native ->
            val nativeSpectrum = FloatArray(contract.spectrumInputElements)
            native.waveformToSpectrumInto(waveform, nativeSpectrum)
            val stftMetrics = compare(kotlinSpectrum, nativeSpectrum)
            require(stftMetrics.accepted) { "${contract.variant} native STFT parity failed: $stftMetrics" }

            val frequency = FloatArray(contract.frequencyOutputElements)
            repeat(contract.sourceCount) { source ->
                val scale = 0.35f + source * 0.05f
                val destinationOffset = source * contract.spectrumInputElements
                kotlinSpectrum.indices.forEach { index ->
                    frequency[destinationOffset + index] = kotlinSpectrum[index] * scale
                }
            }
            val kotlinFrequencyWaveform = kotlinDsp
                .frequencyToWaveform(frequency, contract.sourceCount)
                .copyOf()
            val nativeFrequencyWaveform = FloatArray(contract.waveformOutputElements)
            native.frequencyToWaveformInto(frequency, nativeFrequencyWaveform)
            val istftMetrics = compare(kotlinFrequencyWaveform, nativeFrequencyWaveform)
            require(istftMetrics.accepted) {
                "${contract.variant} native iSTFT parity failed: $istftMetrics"
            }

            val timeWaveform = deterministicTimeOutput(contract)
            val expectedCombined = FloatArray(contract.waveformOutputElements) { index ->
                kotlinFrequencyWaveform[index] + timeWaveform[index]
            }
            val nativeCombined = FloatArray(contract.waveformOutputElements)
            native.postprocessInto(frequency, timeWaveform, nativeCombined)
            val combinedMetrics = compare(expectedCombined, nativeCombined)
            require(combinedMetrics.accepted) {
                "${contract.variant} native branch-combine parity failed: $combinedMetrics"
            }

            val stftSamples = measure(measuredRuns) {
                native.waveformToSpectrumInto(waveform, nativeSpectrum)
            }
            val postprocessSamples = measure(measuredRuns) {
                native.postprocessInto(frequency, timeWaveform, nativeCombined)
            }
            return JSONObject()
                .put("variant", contract.variant)
                .put("executableContractId", contract.executableContractId)
                .put("modelId", contract.modelId)
                .put("artifactSha256", contract.artifactSha256)
                .put("sourceCount", contract.sourceCount)
                .put("sourceOrder", JSONArray(contract.sourceOrder))
                .put("researchOnly", contract.researchOnly)
                .put("inputShapes", JSONArray(contract.inputShapes.map(::JSONArray)))
                .put("outputShapes", JSONArray(contract.outputShapes.map(::JSONArray)))
                .put("stftParity", stftMetrics.evidence())
                .put("istftParity", istftMetrics.evidence())
                .put("combinedParity", combinedMetrics.evidence())
                .put("stftWallMs", summarize(stftSamples))
                .put("postprocessWallMs", summarize(postprocessSamples))
                .put("frequencyRawSha256", sha256(frequency))
                .put("combinedRawSha256", sha256(nativeCombined))
        }
    }

    private fun deterministicWaveform(contract: HtdemucsNativeDspContract): FloatArray =
        FloatArray(contract.waveformInputElements) { index ->
            val channel = index / contract.windowSamples
            val sample = index % contract.windowSamples
            (0.35 * sin(2.0 * PI * (173.0 + channel * 37.0) * sample / contract.sampleRate) +
                0.11 * cos(2.0 * PI * 2_311.0 * sample / contract.sampleRate)).toFloat()
        }

    private fun deterministicTimeOutput(contract: HtdemucsNativeDspContract): FloatArray =
        FloatArray(contract.waveformOutputElements) { index ->
            val sample = index % contract.windowSamples
            val plane = index / contract.windowSamples
            (0.01 * sin(2.0 * PI * (41.0 + plane) * sample / contract.sampleRate)).toFloat()
        }

    private fun measure(count: Int, action: () -> Unit): List<Double> {
        action()
        return List(count) {
            val started = SystemClock.elapsedRealtimeNanos()
            action()
            (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
        }
    }

    private fun compare(reference: FloatArray, candidate: FloatArray): Metrics {
        require(reference.size == candidate.size)
        var signalSquare = 0.0
        var errorSquare = 0.0
        var maxAbsoluteError = 0.0
        var finite = true
        reference.indices.forEach { index ->
            val expected = reference[index].toDouble()
            val actual = candidate[index].toDouble()
            if (!expected.isFinite() || !actual.isFinite()) finite = false
            val error = actual - expected
            signalSquare += expected * expected
            errorSquare += error * error
            maxAbsoluteError = maxOf(maxAbsoluteError, kotlin.math.abs(error))
        }
        val snr = 10.0 * log10((signalSquare + 1e-30) / (errorSquare + 1e-30))
        return Metrics(
            elementCount = reference.size,
            finite = finite,
            signalToNoiseDb = snr,
            maxAbsoluteError = maxAbsoluteError,
            rootMeanSquareError = sqrt(errorSquare / reference.size),
        )
    }

    private fun summarize(samples: List<Double>): JSONObject {
        val sorted = samples.sorted()
        return JSONObject()
            .put("samples", JSONArray(samples))
            .put("minimum", sorted.first())
            .put("median", sorted[sorted.size / 2])
            .put("mean", samples.sum() / samples.size)
            .put("maximum", sorted.last())
    }

    private fun sha256(values: FloatArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = ByteArray(4)
        values.forEach { value ->
            val bits = value.toRawBits()
            bytes[0] = bits.toByte()
            bytes[1] = (bits ushr 8).toByte()
            bytes[2] = (bits ushr 16).toByte()
            bytes[3] = (bits ushr 24).toByte()
            digest.update(bytes)
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
    }

    private data class Metrics(
        val elementCount: Int,
        val finite: Boolean,
        val signalToNoiseDb: Double,
        val maxAbsoluteError: Double,
        val rootMeanSquareError: Double,
    ) {
        val accepted: Boolean = finite && signalToNoiseDb >= MINIMUM_SNR_DB &&
            maxAbsoluteError <= MAXIMUM_ABSOLUTE_ERROR

        fun evidence(): JSONObject = JSONObject()
            .put("accepted", accepted)
            .put("finite", finite)
            .put("elementCount", elementCount)
            .put("signalToNoiseDb", signalToNoiseDb)
            .put("maxAbsoluteError", maxAbsoluteError)
            .put("rootMeanSquareError", rootMeanSquareError)
    }

    companion object {
        const val ARG_RUN_ID = "nativeDspRunId"
        const val ARG_MEASURED_RUNS = "nativeDspMeasuredRuns"
        private const val MINIMUM_SNR_DB = 80.0
        private const val MAXIMUM_ABSOLUTE_ERROR = 1e-3
        private val RUN_ID = Regex("^[A-Za-z0-9._-]{1,80}$")
    }
}
