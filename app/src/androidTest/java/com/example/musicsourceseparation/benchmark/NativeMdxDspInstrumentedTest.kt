package com.example.musicsourceseparation.benchmark

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.musicsourceseparation.model.MdxDspConfig
import com.example.musicsourceseparation.model.MdxSpectrogram
import com.example.musicsourceseparation.model.NativeMdxDsp
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.PI
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class NativeMdxDspInstrumentedTest {
    @Test
    fun nativeFullComplexParityGate() {
        val configs = listOf(
            "9662" to MdxDspConfig(nFft = 6144, dimF = 2048),
            "kim_inst" to MdxDspConfig(nFft = 7680, dimF = 3072),
            "hq4" to MdxDspConfig(nFft = 5120, dimF = 2560),
        )
        val results = JSONArray()
        var allBitExact = true
        for ((name, config) in configs) {
            val waveform = fixture(config)
            val kotlinTensor: FloatArray
            val kotlinWaveform: Array<FloatArray>
            MdxSpectrogram(config, workerCount = 4).use { kotlinDsp ->
                kotlinTensor = FloatArray(config.tensorElementCount).also {
                    kotlinDsp.waveformToNhwcTensorInto(waveform, it)
                }
                kotlinWaveform = Array(2) { FloatArray(config.chunkSize) }.also {
                    kotlinDsp.nhwcTensorToWaveformInto(kotlinTensor, it)
                }
            }
            val nativeTensor = FloatArray(config.tensorElementCount)
            val nativeWaveform = Array(2) { FloatArray(config.chunkSize) }
            NativeMdxDsp(config, workerCount = 4).use { nativeDsp ->
                nativeDsp.waveformToNhwcTensorInto(waveform, nativeTensor)
                nativeDsp.nhwcTensorToWaveformInto(kotlinTensor, nativeWaveform)
            }
            val stft = errorStats(kotlinTensor, nativeTensor)
            val istft = errorStats(kotlinWaveform.flatten(), nativeWaveform.flatten())
            allBitExact = allBitExact && stft.bitExact && istft.bitExact
            results.put(
                JSONObject()
                    .put("model", name)
                    .put("nFft", config.nFft)
                    .put("dimF", config.dimF)
                    .put("stft", stft.toJson())
                    .put("iStft", istft.toJson()),
            )
            check(stft.snrDb >= 80.0 && stft.maxAbs <= 1e-3)
            check(istft.snrDb >= 80.0 && istft.maxAbs <= 1e-3)
        }
        val context = ApplicationProvider.getApplicationContext<Context>()
        val output = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark/native-mdx-parity")
            .apply { mkdirs() }
        File(output, "full-complex.json").writeText(
            JSONObject()
                .put("status", if (allBitExact) "strict-pass" else "numerical-pass-bitexact-fail")
                .put("strictBitExactRequired", true)
                .put("strictBitExactPassed", allBitExact)
                .put("numericalGatePassed", true)
                .put("numericalGateSnrDb", 80.0)
                .put("numericalGateMaxAbs", 1e-3)
                .put("results", results)
                .toString(2),
        )
    }

    @Test
    fun nativePackedRealParityGate() {
        val configs = listOf(
            "9662" to MdxDspConfig(nFft = 6144, dimF = 2048),
            "kim_inst" to MdxDspConfig(nFft = 7680, dimF = 3072),
            "hq4" to MdxDspConfig(nFft = 5120, dimF = 2560),
        )
        val results = JSONArray()
        for ((name, config) in configs) {
            val waveform = fixture(config)
            val fullTensor = FloatArray(config.tensorElementCount)
            val packedTensor = FloatArray(config.tensorElementCount)
            val fullWaveform = Array(2) { FloatArray(config.chunkSize) }
            val packedWaveform = Array(2) { FloatArray(config.chunkSize) }
            NativeMdxDsp(config, 4, NativeMdxDsp.Mode.FULL_COMPLEX).use { full ->
                full.waveformToNhwcTensorInto(waveform, fullTensor)
                full.nhwcTensorToWaveformInto(fullTensor, fullWaveform)
            }
            NativeMdxDsp(config, 4, NativeMdxDsp.Mode.PACKED_REAL).use { packed ->
                packed.waveformToNhwcTensorInto(waveform, packedTensor)
                packed.nhwcTensorToWaveformInto(fullTensor, packedWaveform)
            }
            val stft = errorStats(fullTensor, packedTensor)
            val istft = errorStats(fullWaveform.flatten(), packedWaveform.flatten())
            results.put(
                JSONObject()
                    .put("model", name)
                    .put("nFft", config.nFft)
                    .put("dimF", config.dimF)
                    .put("stft", stft.toJson())
                    .put("iStft", istft.toJson()),
            )
            check(stft.snrDb >= 80.0 && stft.maxAbs <= 1e-3)
            check(istft.snrDb >= 80.0 && istft.maxAbs <= 1e-3)
        }
        val context = ApplicationProvider.getApplicationContext<Context>()
        val output = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark/native-mdx-parity")
            .apply { mkdirs() }
        File(output, "packed-real.json").writeText(
            JSONObject()
                .put("status", "numerical-pass")
                .put("reference", "native-full-complex")
                .put("numericalGateSnrDb", 80.0)
                .put("numericalGateMaxAbs", 1e-3)
                .put("results", results)
                .toString(2),
        )
    }

    private fun fixture(config: MdxDspConfig): Array<FloatArray> = Array(2) { channel ->
        FloatArray(config.chunkSize) { index ->
            (0.1 * sin(2.0 * PI * (220 + channel * 37) * index / config.sampleRate) +
                0.01 * sin(index * 0.013)).toFloat()
        }
    }

    private fun Array<FloatArray>.flatten(): FloatArray {
        val output = FloatArray(sumOf { it.size })
        var offset = 0
        for (values in this) {
            values.copyInto(output, offset)
            offset += values.size
        }
        return output
    }

    private fun errorStats(reference: FloatArray, candidate: FloatArray): ErrorStats {
        require(reference.size == candidate.size)
        var signal = 0.0
        var error = 0.0
        var maxAbs = 0.0
        var bitExact = true
        for (index in reference.indices) {
            val expected = reference[index].toDouble()
            val delta = candidate[index].toDouble() - expected
            signal += expected * expected
            error += delta * delta
            maxAbs = maxOf(maxAbs, kotlin.math.abs(delta))
            if (reference[index].toRawBits() != candidate[index].toRawBits()) bitExact = false
        }
        val snr = if (error == 0.0) Double.POSITIVE_INFINITY else 10.0 * log10(signal / error)
        return ErrorStats(snr, maxAbs, sqrt(error / reference.size), bitExact)
    }

    private data class ErrorStats(
        val snrDb: Double,
        val maxAbs: Double,
        val rmsError: Double,
        val bitExact: Boolean,
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("snrDb", if (snrDb.isFinite()) snrDb else "Infinity")
            .put("maxAbs", maxAbs)
            .put("rmsError", rmsError)
            .put("bitExact", bitExact)
    }
}
