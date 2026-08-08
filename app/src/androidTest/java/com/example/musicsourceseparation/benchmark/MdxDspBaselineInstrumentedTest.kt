package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.BuildConfig
import com.example.musicsourceseparation.model.MdxDspConfig
import com.example.musicsourceseparation.model.MdxSpectrogram
import com.example.musicsourceseparation.model.NativeMdxDsp
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.PI
import kotlin.math.sin

/** Short, contract-driven MDX DSP baseline. One invocation measures one model/backend/session. */
@RunWith(AndroidJUnit4::class)
class MdxDspBaselineInstrumentedTest {
    @Test
    fun runBaseline() {
        val args = InstrumentationRegistry.getArguments()
        val modelId = requireNotNull(args.getString("modelId"))
        val modelFileName = requireNotNull(args.getString("modelFile"))
        val contractFileName = requireNotNull(args.getString("contractFile"))
        val backend = args.getString("backend", "cpu")!!
        val threads = args.getString("threads", "4")!!.toInt().coerceIn(1, 16)
        val warmups = args.getString("warmups", "2")!!.toInt().coerceIn(0, 10)
        val measured = args.getString("measuredRuns", "10")!!.toInt().coerceIn(1, 100)
        val sessionIndex = args.getString("sessionIndex", "1")!!.toInt()
        val dspWorkers = args.getString("dspWorkers", "1")!!.toInt().coerceIn(1, 8)
        val dspProfile = args.getString("dspProfile", "legacy")!!
        val rotateFixture = args.getString("rotateFixture", "false")!!.toBooleanStrictOrNull() ?: false
        require(dspProfile in setOf("legacy", "reuse-nhwc", "native-full", "native-packed"))
        val context = ApplicationProvider.getApplicationContext<Context>()
        val powerManager = context.getSystemService(PowerManager::class.java)
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val model = File(root, "models/$modelFileName")
        val contractFile = File(root, "contracts/$contractFileName")
        require(model.isFile) { "Missing model: $model" }
        require(contractFile.isFile) { "Missing contract: $contractFile" }
        val contract = JSONObject(contractFile.readText())
        val artifact = contract.getJSONObject("artifact")
        val expectedSha = artifact.getString("sha256")
        require(sha256(model) == expectedSha) { "Model SHA mismatch for $modelId" }
        val dsp = contract.getJSONObject("dsp")
        val config = MdxDspConfig(
            sampleRate = dsp.getInt("sampleRate"),
            nFft = dsp.getInt("nFft"),
            hopLength = dsp.getInt("hopLength"),
            dimF = dsp.getInt("dimF"),
            dimTPower = dsp.getInt("dimTPower"),
        )
        val waveform = fixture(config)
        val waveformSha = sha256Floats(waveform)
        val profile = if (backend == "gpu-bounded") "gpu-opencl-bounded-fp32-v1" else "cpu-xnnpack"
        val resultDir = File(root, "mdx-dsp-baseline/$modelId/$backend/$dspProfile/workers-$dspWorkers/session-$sessionIndex").apply {
            deleteRecursively(); mkdirs()
        }
        val report = JSONObject()
            .put("status", "complete")
            .put("modelId", modelId)
            .put("modelFile", modelFileName)
            .put("modelSha256", expectedSha)
            .put("contractFile", contractFileName)
            .put("contractSha256", sha256(contractFile))
            .put("contractId", contract.getString("contractId"))
            .put("runtime", "LiteRT CompiledModel 2.1.5")
            .put("runtimeId", BuildConfig.BENCHMARK_RUNTIME_ID)
            .put("runtimeArtifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256)
            .put("sourceRevision", BuildConfig.BENCHMARK_SOURCE_REVISION)
            .put("sourceDirty", BuildConfig.BENCHMARK_SOURCE_DIRTY)
            .put("backend", backend)
            .put("profile", profile)
            .put("device", JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
                .put("socManufacturer", Build.SOC_MANUFACTURER).put("socModel", Build.SOC_MODEL)
                .put("sdk", Build.VERSION.SDK_INT).put("abis", Build.SUPPORTED_ABIS.toList()))
            .put("config", JSONObject().put("nFft", config.nFft).put("hopLength", config.hopLength)
                .put("dimF", config.dimF).put("frames", config.dimT).put("chunkSize", config.chunkSize))
            .put("tensorContract", contract.getJSONObject("tensorContract"))
            .put("stemContract", contract.getJSONObject("stemContract"))
            .put("fixture", JSONObject().put("waveformSha256", waveformSha).put("samples", config.chunkSize))
            .put("warmups", warmups).put("measuredRuns", measured).put("threads", threads)
            .put("dspWorkers", dspWorkers)
            .put("dspProfile", dspProfile)
            .put("rotateFixture", rotateFixture)
            .put("nativeFftLibrary", if (dspProfile.startsWith("native-")) "pocketfft@c90e55b3" else JSONObject.NULL)
            .put("iStftOlaCombined", true)
            .put("thermalStatusStart", powerManager.currentThermalStatus)
        val sessions = JSONArray()
        try {
            val boundedRuntime = if (backend == "gpu-bounded") BoundedGpuRuntime.loadAndValidate() else null
            val options = if (backend == "gpu-bounded") {
                CompiledModel.Options(Accelerator.GPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                        numStepsOfCommandBufferPreparations = 1,
                    )
                }
            } else {
                CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(numThreads = threads, xnnPackFlags = null, xnnPackWeightCachePath = null)
                }
            }
            val environment = Environment.create()
            val setupStart = SystemClock.elapsedRealtimeNanos()
            val compiled = CompiledModel.create(model.absolutePath, options, environment)
            val input = compiled.createInputBuffers().single()
            val output = compiled.createOutputBuffers().single()
            val setupMs = (SystemClock.elapsedRealtimeNanos() - setupStart) / 1_000_000.0
            val warmupTimes = JSONArray()
            val dspBuffers = if (dspProfile != "legacy") ReusableDspBuffers(config) else null
            val nativeDsp = when (dspProfile) {
                "native-full" -> NativeMdxDsp(config, dspWorkers, NativeMdxDsp.Mode.FULL_COMPLEX)
                "native-packed" -> NativeMdxDsp(config, dspWorkers, NativeMdxDsp.Mode.PACKED_REAL)
                else -> null
            }
            val runtimeBefore: Map<String, Long>
            val runtimeAfter: Map<String, Long>
            val memorySamples = JSONArray()
            MdxSpectrogram(config, workerCount = dspWorkers).use { spectrogram ->
                repeat(warmups) { warmupTimes.put(runWindow(compiled, input, output, spectrogram, waveform, config, dsp.getDouble("modelOutputScale"), null, boundedRuntime, dspBuffers, nativeDsp).getDouble("totalMs")) }
                boundedRuntime?.resetInferenceCounters()
                runtimeBefore = runtimeStats()
                repeat(measured) { runIndex ->
                    if (rotateFixture) rotateFixture(waveform, runIndex)
                    sessions.put(runWindow(compiled, input, output, spectrogram, waveform, config, dsp.getDouble("modelOutputScale"), resultDir, boundedRuntime, dspBuffers, nativeDsp))
                    if ((runIndex + 1) % 10 == 0 || runIndex + 1 == measured) {
                        memorySamples.put(memoryEvidence().put("completedRuns", runIndex + 1))
                    }
                }
                runtimeAfter = runtimeStats()
            }
            nativeDsp?.close()
            report.put("setupMs", setupMs).put("warmupMs", warmupTimes)
                .put("runs", sessions).put("memory", memoryEvidence())
                .put("memorySamples", memorySamples)
                .put("runtimeStatsDelta", runtimeStatsDelta(runtimeBefore, runtimeAfter))
                .put("thermalStatusEnd", powerManager.currentThermalStatus)
            boundedRuntime?.let { report.put("boundedGpuEvidence", it.evidence()) }
            compiled.close()
            environment.close()
        } catch (error: Throwable) {
            report.put("status", "error").put("errorClass", error::class.java.name).put("message", error.message.orEmpty())
                .put("stack", error.stackTraceToString()).put("memory", memoryEvidence())
                .put("thermalStatusEnd", powerManager.currentThermalStatus)
        }
        File(resultDir, "report.json").writeText(report.toString(2))
        println(report)
        check(report.getString("status") == "complete") { report.toString() }
    }

    private fun runWindow(
        model: CompiledModel,
        input: TensorBuffer,
        output: TensorBuffer,
        spectrogram: MdxSpectrogram,
        waveform: Array<FloatArray>,
        config: MdxDspConfig,
        modelOutputScale: Double,
        resultDir: File?,
        boundedRuntime: BoundedGpuRuntime?,
        reusable: ReusableDspBuffers?,
        nativeDsp: NativeMdxDsp?,
    ): JSONObject {
        fun now() = SystemClock.elapsedRealtimeNanos()
        val total = now()
        val stftStart = now()
        val spectrogramOutput = if (nativeDsp != null) {
            nativeDsp.waveformToNhwcTensorInto(waveform, requireNotNull(reusable).inputNhwc)
            reusable.inputNhwc
        } else if (reusable != null) {
            spectrogram.waveformToNhwcTensorInto(waveform, reusable.inputNhwc)
            reusable.inputNhwc
        } else {
            spectrogram.waveformToTensor(waveform)
        }
        val stftMs = (now() - stftStart) / 1e6
        val layoutStart = now()
        val nhwc = if (reusable != null) spectrogramOutput else nchwToNhwc(spectrogramOutput, config.dimF, config.dimT)
        val layoutMs = (now() - layoutStart) / 1e6
        val writeStart = now(); input.writeFloat(nhwc); val writeMs = (now() - writeStart) / 1e6
        val invokeStart = now(); boundedRuntime?.beginInference(); try { model.run(listOf(input), listOf(output)) } finally { boundedRuntime?.endInference() }; val invokeMs = (now() - invokeStart) / 1e6
        val readStart = now(); val outNhwc = output.readFloat(); val readMs = (now() - readStart) / 1e6
        val outputLayoutStart = now()
        val inverseInput = if (reusable != null) outNhwc else nhwcToNchw(outNhwc, config.dimF, config.dimT)
        val outputLayoutMs = (now() - outputLayoutStart) / 1e6
        val istftStart = now()
        val separated = if (nativeDsp != null) {
            nativeDsp.nhwcTensorToWaveformInto(inverseInput, requireNotNull(reusable).separated)
            reusable.separated
        } else if (reusable != null) {
            spectrogram.nhwcTensorToWaveformInto(inverseInput, reusable.separated)
            reusable.separated
        } else {
            spectrogram.tensorToWaveform(inverseInput)
        }
        val istftOlaMs = (now() - istftStart) / 1e6
        val residualStart = now()
        val residual = reusable?.residual ?: Array(2) { FloatArray(config.chunkSize) }
        for (c in 0 until 2) for (i in 0 until config.chunkSize) residual[c][i] = waveform[c][i] - separated[c][i] * modelOutputScale.toFloat()
        val residualMs = (now() - residualStart) / 1e6
        val pcmStart = now(); val pcmBytes = reusable?.pcm16 ?: ByteArray(config.chunkSize * 4); var offset = 0
        for (i in 0 until config.chunkSize) for (c in 0 until 2) { val sample = (separated[c][i].coerceIn(-1f, 1f) * 32767f).toInt().coerceIn(-32768, 32767); pcmBytes[offset++] = sample.toByte(); pcmBytes[offset++] = (sample ushr 8).toByte() }
        val pcmMs = (now() - pcmStart) / 1e6
        if (resultDir != null && sessionsWritten++ == 0) File(resultDir, "output-preview.pcm16le").writeBytes(pcmBytes)
        return JSONObject().put("stftMs", stftMs).put("tensorLayoutWriteMs", layoutMs).put("inputBufferWriteMs", writeMs)
            .put("invokeMs", invokeMs).put("outputReadMs", readMs).put("tensorLayoutReadMs", outputLayoutMs)
            .put("iStftOlaMs", istftOlaMs).put("residualMs", residualMs).put("pcm16Ms", pcmMs)
            .put("totalMs", (now() - total) / 1e6).put("finite", separated.all { ch -> ch.all { it.isFinite() } })
    }

    private fun fixture(config: MdxDspConfig): Array<FloatArray> = Array(2) { c ->
        FloatArray(config.chunkSize) { i -> (0.1 * sin(2.0 * PI * (220 + c * 37) * i / config.sampleRate) + 0.01 * sin(i * 0.013)).toFloat() }
    }

    private fun rotateFixture(waveform: Array<FloatArray>, runIndex: Int) {
        val delta = (runIndex + 1) * 0.0001f
        for (channel in waveform.indices) {
            for (sample in waveform[channel].indices) {
                waveform[channel][sample] = (waveform[channel][sample] + delta).coerceIn(-0.2f, 0.2f)
            }
        }
    }

    private fun nchwToNhwc(input: FloatArray, f: Int, t: Int): FloatArray {
        val out = FloatArray(input.size)
        for (c in 0 until 4) for (freq in 0 until f) for (frame in 0 until t) out[(freq * t + frame) * 4 + c] = input[(c * f + freq) * t + frame]
        return out
    }
    private fun nhwcToNchw(input: FloatArray, f: Int, t: Int): FloatArray {
        val out = FloatArray(input.size)
        for (c in 0 until 4) for (freq in 0 until f) for (frame in 0 until t) out[(c * f + freq) * t + frame] = input[(freq * t + frame) * 4 + c]
        return out
    }
    private fun memoryEvidence(): JSONObject { val m = Debug.MemoryInfo().also(Debug::getMemoryInfo); return JSONObject().put("pssKb", m.totalPss).put("privateDirtyKb", m.totalPrivateDirty).put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize()) }
    private fun runtimeStats(): Map<String, Long> = listOf(
        "art.gc.gc-count",
        "art.gc.bytes-allocated",
        "art.gc.bytes-freed",
        "art.gc.blocking-gc-count",
    ).mapNotNull { key -> runCatching { Debug.getRuntimeStat(key) }.getOrNull()?.toLongOrNull()?.let { key to it } }.toMap()
    private fun runtimeStatsDelta(before: Map<String, Long>, after: Map<String, Long>): JSONObject = JSONObject().also { out ->
        (before.keys + after.keys).filter { it.contains("gc", ignoreCase = true) || it.contains("alloc", ignoreCase = true) }.sorted().forEach { key -> out.put(key, (after[key] ?: 0L) - (before[key] ?: 0L)) }
    }
    private fun sha256(file: File): String = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }
    private fun sha256Floats(value: Array<FloatArray>): String = sha256Bytes(value.flatMap { it.toList() }.toFloatArray())
    private fun sha256Bytes(value: FloatArray): String = MessageDigest.getInstance("SHA-256").digest(floatBytes(value)).joinToString("") { "%02x".format(it) }
    private fun floatBytes(value: FloatArray): ByteArray { val b = ByteBuffer.allocate(value.size * 4).order(ByteOrder.LITTLE_ENDIAN); value.forEach { b.putFloat(it) }; return b.array() }
    private class ReusableDspBuffers(config: MdxDspConfig) {
        val inputNhwc = FloatArray(config.tensorElementCount)
        val separated = Array(2) { FloatArray(config.chunkSize) }
        val residual = Array(2) { FloatArray(config.chunkSize) }
        val pcm16 = ByteArray(config.chunkSize * 4)
    }
    private class BoundedGpuRuntime private constructor(private val runtimeClass: Class<*>, private val capability: Any) {
        fun resetInferenceCounters() { invoke("resetInferenceCounters") }
        fun beginInference() { invoke("beginInference") }
        fun endInference() { invoke("endInference") }
        fun evidence(): JSONObject = JSONObject().put("artifactVersion", value("getArtifactVersion")).put("profileId", value("getProfileId"))
            .put("kernelBatchSize", value("getKernelBatchSize")).put("commandQueueWindowSize", value("getCommandQueueWindowSize"))
            .put("dispatchCount", invoke("getDispatchCount")).put("eventWaitCount", invoke("getEventWaitCount"))
        private fun invoke(name: String): Any? = runtimeClass.getMethod(name).invoke(null)
        private fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)
        companion object {
            fun loadAndValidate(): BoundedGpuRuntime {
                val clazz = Class.forName("io.github.wluhwluh.bss.litert.BssLiteRtRuntime")
                val capability = requireNotNull(clazz.getMethod("queryCapability").invoke(null))
                fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)
                require(value("isAvailable") == true && value("getArtifactVersion") == "2.1.5-bss.2")
                require(value("getProfileId") == "gpu-opencl-bounded-fp32-v1")
                require(value("getKernelBatchSize") == 1 && value("getCommandQueueWindowSize") == 1)
                return BoundedGpuRuntime(clazz, capability)
            }
        }
    }
    companion object { private var sessionsWritten = 0 }
}
