package com.example.musicsourceseparation.benchmark

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import android.os.SystemClock
import android.util.Log
import java.io.File
import kotlin.math.abs
import kotlin.math.log10
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class TfcTdfStreamingDspInstrumentedTest {
    @Test
    fun reusableForwardAndInverseWorkspacesMatchAllocatingContract() {
        val dsp = TfcTdfStreamingDsp()
        val input = FloatArray(TfcTdfStreamingDsp.INPUT_SAMPLES * TfcTdfStreamingDsp.CHANNELS) {
            index -> (((index * 17) % 2_001) - 1_000) / 1_000f
        }
        val allocatingTensor = dsp.stftNhwc(input)
        val reusableTensor = FloatArray(TfcTdfStreamingDsp.TENSOR_ELEMENTS)
        dsp.stftNhwcInto(input, reusableTensor)
        assertMaxAbsDifference(allocatingTensor, reusableTensor, 0f)

        val allocatingOutput = dsp.istftInterleaved(allocatingTensor)
        val reusableOutput = FloatArray(
            TfcTdfStreamingDsp.INPUT_SAMPLES * TfcTdfStreamingDsp.CHANNELS,
        )
        dsp.istftInterleavedInto(reusableTensor, reusableOutput)
        assertMaxAbsDifference(allocatingOutput, reusableOutput, 0f)

        val secondInput = FloatArray(input.size) { index -> input[index] * 0.37f }
        dsp.stftNhwcInto(secondInput, reusableTensor)
        val secondReference = dsp.stftNhwc(secondInput)
        assertMaxAbsDifference(secondReference, reusableTensor, 0f)
        assertTrue(reusableTensor !== secondReference)
        assertEquals(TfcTdfStreamingDsp.TENSOR_ELEMENTS, reusableTensor.size)
    }

    @Test
    fun nativeFullAndPackedRealMatchKotlinContract() {
        val input = deterministicInput()
        val kotlinDsp = TfcTdfStreamingDsp()
        val referenceTensor = kotlinDsp.stftNhwc(input)
        val modelTensor = referenceTensor.copyOf().also { tensor ->
            tensor.indices.forEach { index -> tensor[index] *= 0.7f }
        }
        val referenceOutput = kotlinDsp.istftInterleaved(modelTensor)
        val referenceResidual = FloatArray(17 * TfcTdfStreamingDsp.CHANNELS)
        kotlinDsp.istftResidualInto(
            modelTensor,
            input,
            trimSamples = TfcTdfStreamingDsp.TRIM_SAMPLES,
            actualSamples = 17,
            output = referenceResidual,
        )

        for (mode in com.example.musicsourceseparation.model.NativeTfcTdfDsp.Mode.entries) {
            NativeTfcTdfStreamingDsp(workerCount = 1, mode = mode).use { native ->
                val nativeTensor = FloatArray(TfcTdfStreamingDsp.TENSOR_ELEMENTS)
                native.stftNhwcInto(input, nativeTensor)
                assertSignalClose(
                    "STFT ${mode.profileId}",
                    referenceTensor,
                    nativeTensor,
                    maxError = 3e-3f,
                    minSnrDb = 105.0,
                )
                val nativeOutput = FloatArray(referenceOutput.size)
                native.istftInterleavedInto(modelTensor, nativeOutput)
                assertSignalClose(
                    "iSTFT ${mode.profileId}",
                    referenceOutput,
                    nativeOutput,
                    maxError = 3e-5f,
                    minSnrDb = 105.0,
                )
                val nativeResidual = FloatArray(referenceResidual.size)
                native.istftResidualInto(
                    modelTensor,
                    input,
                    trimSamples = TfcTdfStreamingDsp.TRIM_SAMPLES,
                    actualSamples = 17,
                    output = nativeResidual,
                )
                assertSignalClose(
                    "residual ${mode.profileId}",
                    referenceResidual,
                    nativeResidual,
                    maxError = 3e-5f,
                    minSnrDb = 100.0,
                )
            }
        }
    }

    @Test
    fun nativePackedRealWorkerLanesMatchSingleLane() {
        val input = deterministicInput()
        val reference = NativeTfcTdfStreamingDsp(
            workerCount = 1,
            mode = com.example.musicsourceseparation.model.NativeTfcTdfDsp.Mode.PACKED_REAL,
        )
        reference.use {
            val referenceTensor = FloatArray(TfcTdfStreamingDsp.TENSOR_ELEMENTS)
            it.stftNhwcInto(input, referenceTensor)
            for (workers in listOf(2, 4)) {
                NativeTfcTdfStreamingDsp(
                    workerCount = workers,
                    mode = com.example.musicsourceseparation.model.NativeTfcTdfDsp.Mode.PACKED_REAL,
                ).use { native ->
                    val tensor = FloatArray(referenceTensor.size)
                    native.stftNhwcInto(input, tensor)
                    assertSignalClose(
                        "STFT workers=$workers",
                        referenceTensor,
                        tensor,
                        maxError = 1e-5f,
                        minSnrDb = 120.0,
                    )
                    val output = FloatArray(TfcTdfStreamingDsp.INPUT_SAMPLES * 2)
                    native.istftInterleavedInto(referenceTensor, output)
                    val referenceOutput = FloatArray(output.size)
                    it.istftInterleavedInto(referenceTensor, referenceOutput)
                    assertSignalClose(
                        "iSTFT workers=$workers",
                        referenceOutput,
                        output,
                        maxError = 2e-5f,
                        minSnrDb = 115.0,
                    )
                }
            }
        }
    }

    @Test
    fun benchmarkDspProfiles() {
        val input = deterministicInput()
        val baseline = TfcTdfStreamingDsp()
        val modelTensor = baseline.stftNhwc(input).also { tensor ->
            tensor.indices.forEach { index -> tensor[index] *= 0.7f }
        }
        baseline.close()
        val profileFactories: List<Pair<String, () -> TfcTdfDspSession>> = listOf(
            "kotlin-jtransforms-full-complex" to { TfcTdfStreamingDsp() },
            "native-pocketfft-full-complex-workers-1" to {
                NativeTfcTdfStreamingDsp(
                    1,
                    com.example.musicsourceseparation.model.NativeTfcTdfDsp.Mode.FULL_COMPLEX,
                )
            },
            "native-pocketfft-packed-real-workers-1" to {
                NativeTfcTdfStreamingDsp(
                    1,
                    com.example.musicsourceseparation.model.NativeTfcTdfDsp.Mode.PACKED_REAL,
                )
            },
            "native-pocketfft-packed-real-workers-2" to {
                NativeTfcTdfStreamingDsp(
                    2,
                    com.example.musicsourceseparation.model.NativeTfcTdfDsp.Mode.PACKED_REAL,
                )
            },
            "native-pocketfft-packed-real-workers-4" to {
                NativeTfcTdfStreamingDsp(
                    4,
                    com.example.musicsourceseparation.model.NativeTfcTdfDsp.Mode.PACKED_REAL,
                )
            },
        )
        val requestedProfile = InstrumentationRegistry.getArguments()
            .getString("profile", "all")!!
        val selectedFactories = if (requestedProfile == "all") {
            profileFactories
        } else {
            profileFactories.filter { it.first == requestedProfile }.also {
                require(it.size == 1) { "Unknown DSP benchmark profile: $requestedProfile" }
            }
        }
        val profiles = selectedFactories.map { it.second() }
        val samples = profiles.associateWith { ProfileSamples() }
        val tensors = profiles.associateWith { FloatArray(TfcTdfStreamingDsp.TENSOR_ELEMENTS) }
        val outputs = profiles.associateWith {
            FloatArray(TfcTdfStreamingDsp.INPUT_SAMPLES * TfcTdfStreamingDsp.CHANNELS)
        }
        val residualOutputs = profiles.associateWith {
            FloatArray(TfcTdfStreamingDsp.USEFUL_SAMPLES * TfcTdfStreamingDsp.CHANNELS)
        }
        try {
            repeat(2) {
                profiles.forEach { profile ->
                    profile.stftNhwcInto(input, requireNotNull(tensors[profile]))
                    profile.istftInterleavedInto(modelTensor, requireNotNull(outputs[profile]))
                    profile.istftResidualInto(
                        modelTensor,
                        input,
                        TfcTdfStreamingDsp.TRIM_SAMPLES,
                        TfcTdfStreamingDsp.USEFUL_SAMPLES,
                        requireNotNull(residualOutputs[profile]),
                    )
                }
            }
            repeat(12) { round ->
                val offset = round % profiles.size
                val ordered = profiles.drop(offset) + profiles.take(offset)
                ordered.forEach { profile ->
                    val stftStarted = SystemClock.elapsedRealtimeNanos()
                    profile.stftNhwcInto(input, requireNotNull(tensors[profile]))
                    requireNotNull(samples[profile]).stftNanos +=
                        SystemClock.elapsedRealtimeNanos() - stftStarted
                    val istftStarted = SystemClock.elapsedRealtimeNanos()
                    profile.istftInterleavedInto(modelTensor, requireNotNull(outputs[profile]))
                    requireNotNull(samples[profile]).istftNanos +=
                        SystemClock.elapsedRealtimeNanos() - istftStarted
                    val fusedStarted = SystemClock.elapsedRealtimeNanos()
                    profile.istftResidualInto(
                        modelTensor,
                        input,
                        TfcTdfStreamingDsp.TRIM_SAMPLES,
                        TfcTdfStreamingDsp.USEFUL_SAMPLES,
                        requireNotNull(residualOutputs[profile]),
                    )
                    requireNotNull(samples[profile]).fusedNanos +=
                        SystemClock.elapsedRealtimeNanos() - fusedStarted
                }
            }
            val report = JSONObject()
                .put("schemaVersion", 1)
                .put("device", JSONObject()
                    .put("model", android.os.Build.MODEL)
                    .put("device", android.os.Build.DEVICE)
                    .put("soc", android.os.Build.SOC_MODEL)
                    .put("sdk", android.os.Build.VERSION.SDK_INT))
                .put("warmups", 2)
                .put("runs", 12)
            val profileReports = JSONArray()
            profiles.forEach { profile ->
                val timing = requireNotNull(samples[profile])
                profileReports.put(
                    JSONObject()
                        .put("profile", profile.profileId)
                        .put("workers", profile.workerCount)
                        .put("stftMedianMs", medianMs(timing.stftNanos))
                        .put("stftP95Ms", percentileMs(timing.stftNanos, 0.95))
                        .put("istftMedianMs", medianMs(timing.istftNanos))
                        .put("istftP95Ms", percentileMs(timing.istftNanos, 0.95))
                        .put("fusedResidualMedianMs", medianMs(timing.fusedNanos))
                        .put("fusedResidualP95Ms", percentileMs(timing.fusedNanos, 0.95))
                        .put(
                            "combinedMedianMs",
                            medianMs(timing.stftNanos.zip(timing.istftNanos) { stft, istft ->
                                stft + istft
                            }),
                        ),
                )
            }
            report.put("profiles", profileReports)
            val runId = InstrumentationRegistry.getArguments().getString("runId", "latest")!!
            val resultFile = File(
                requireNotNull(ApplicationProvider.getApplicationContext<android.content.Context>()
                    .getExternalFilesDir(null)),
                "benchmark/tfc-tdf-dsp/results/$runId.json",
            ).apply {
                parentFile?.mkdirs()
            }
            report
                .put("runId", runId)
                .put("outputFile", resultFile.absolutePath)
            resultFile.writeText(report.toString(2))
            Log.i("TFC_TDF_DSP", report.toString())
            println("TFC_TDF_DSP_REPORT=${report}")
        } finally {
            profiles.forEach(TfcTdfDspSession::close)
        }
    }

    private data class ProfileSamples(
        val stftNanos: MutableList<Long> = ArrayList(),
        val istftNanos: MutableList<Long> = ArrayList(),
        val fusedNanos: MutableList<Long> = ArrayList(),
    )

    private fun medianMs(values: List<Long>): Double = percentileMs(values, 0.5)

    private fun percentileMs(values: List<Long>, fraction: Double): Double {
        val sorted = values.sorted()
        val index = (kotlin.math.ceil(sorted.size * fraction).toInt() - 1).coerceIn(sorted.indices)
        return sorted[index] / 1_000_000.0
    }

    private fun deterministicInput(): FloatArray = FloatArray(
        TfcTdfStreamingDsp.INPUT_SAMPLES * TfcTdfStreamingDsp.CHANNELS,
    ) { index -> (((index * 17) % 2_001) - 1_000) / 1_000f }

    private fun assertSignalClose(
        label: String,
        expected: FloatArray,
        actual: FloatArray,
        maxError: Float,
        minSnrDb: Double,
    ) {
        assertEquals(expected.size, actual.size)
        var signalEnergy = 0.0
        var errorEnergy = 0.0
        var maximum = 0f
        for (index in expected.indices) {
            val difference = expected[index] - actual[index]
            signalEnergy += expected[index].toDouble() * expected[index]
            errorEnergy += difference.toDouble() * difference
            maximum = maxOf(maximum, abs(difference))
        }
        val snrDb = if (errorEnergy == 0.0) Double.POSITIVE_INFINITY else
            10.0 * log10(signalEnergy / errorEnergy)
        assertTrue("$label max error was $maximum", maximum <= maxError)
        assertTrue("$label SNR was $snrDb dB", snrDb >= minSnrDb)
    }

    private fun assertMaxAbsDifference(expected: FloatArray, actual: FloatArray, tolerance: Float) {
        assertEquals(expected.size, actual.size)
        var maximum = 0f
        for (index in expected.indices) {
            maximum = maxOf(maximum, abs(expected[index] - actual[index]))
        }
        assertTrue("maximum absolute difference was $maximum", maximum <= tolerance)
    }
}
