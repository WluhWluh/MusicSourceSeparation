package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.BuildConfig
import com.example.musicsourceseparation.audio.AudioPcmDecoder
import com.example.musicsourceseparation.audio.DecodedPcmAudio
import com.example.musicsourceseparation.audio.WavFileWriter
import com.example.musicsourceseparation.model.MdxDspConfig
import com.example.musicsourceseparation.model.NativeLiteRtMdxPipeline
import com.example.musicsourceseparation.model.NativeMdxDsp
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.ceil
import kotlin.math.roundToInt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/** Full-song counterpart of MdxDoubleBufferInstrumentedTest, including output writers. */
@RunWith(AndroidJUnit4::class)
class MdxDoubleBufferFullSongInstrumentedTest {
    @Test
    fun separateFullSong() {
        val args = InstrumentationRegistry.getArguments()
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val modelId = requireNotNull(args.getString("modelId"))
        val modelFile = File(root, "models/${requireNotNull(args.getString("modelFile"))}")
        val contractFile = File(root, "contracts/${requireNotNull(args.getString("contractFile"))}")
        val audioFile = File(root, "audio-input/${requireNotNull(args.getString("audioFile"))}")
        val backend = args.getString("backend", "gpu-bounded")!!
        require(backend == "gpu-bounded" || backend == "qnn")
        val tensorBoundary = args.getString("tensorBoundary", "java-tensor")!!
        require(tensorBoundary == "java-tensor" || tensorBoundary == "native-managed")
        if (tensorBoundary == "native-managed") require(backend == "gpu-bounded")
        val contract = JSONObject(contractFile.readText())
        require(sha256(modelFile) == contract.getJSONObject("artifact").getString("sha256"))
        val dspJson = contract.getJSONObject("dsp")
        val config = MdxDspConfig(
            sampleRate = dspJson.getInt("sampleRate"), nFft = dspJson.getInt("nFft"),
            hopLength = dspJson.getInt("hopLength"), dimF = dspJson.getInt("dimF"),
            dimTPower = dspJson.getInt("dimTPower"),
        )
        val resultLeaf = if (tensorBoundary == "native-managed") "$backend/$tensorBoundary" else backend
        val outputDir = File(root, "mdx-double-buffer-full-song/$modelId/$resultLeaf").apply {
            deleteRecursively(); mkdirs()
        }
        val report = JSONObject().put("status", "running").put("modelId", modelId)
            .put("modelSha256", sha256(modelFile)).put("contractSha256", sha256(contractFile))
            .put("sourceSha256", sha256(audioFile)).put("backend", backend)
            .put("tensorBoundary", tensorBoundary)
            .put("runtimeId", BuildConfig.BENCHMARK_RUNTIME_ID)
            .put("runtimeArtifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256)
            .put("sourceRevision", BuildConfig.BENCHMARK_SOURCE_REVISION)
            .put("sourceDirty", BuildConfig.BENCHMARK_SOURCE_DIRTY)
            .put("dspProfile", "native-packed-w4").put("doubleBufferSlots", 2)
            .put("device", JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL))
        val thermal = context.getSystemService(android.os.PowerManager::class.java)
        report.put("thermalStatusStart", thermal.currentThermalStatus)
        if (tensorBoundary == "native-managed") {
            try {
                runNativeManagedFullSong(
                    context, report, outputDir, modelFile, audioFile, dspJson, config,
                )
            } catch (error: Throwable) {
                report.put("status", "error").put("errorClass", error.javaClass.name)
                    .put("message", error.message ?: JSONObject.NULL)
                    .put("stack", error.stackTraceToString())
            }
            report.put("thermalStatusEnd", thermal.currentThermalStatus)
            File(outputDir, "report.json").writeText(report.toString(2))
            check(report.getString("status") == "complete") { report.toString() }
            return
        }
        var environment: Environment? = null
        var model: CompiledModel? = null
        val inputs = mutableListOf<TensorBuffer>()
        val outputs = mutableListOf<TensorBuffer>()
        val executor = Executors.newSingleThreadExecutor()
        val dsps = Array(2) { NativeMdxDsp(config, 4, NativeMdxDsp.Mode.PACKED_REAL) }
        try {
            val decodeStarted = now()
            val decoded = AudioPcmDecoder(context).decode(Uri.fromFile(audioFile))
            require(decoded.sampleRate == config.sampleRate && decoded.channelCount == 2)
            val decodeMs = elapsed(decodeStarted)
            val provider = if (backend == "qnn") BuiltinNpuAcceleratorProvider(
                context, QnnDeviceCompatibility.checker,
            ).also { require(Build.VERSION.SDK_INT >= 31 && it.isDeviceSupported() && it.isLibraryReady()) } else null
            environment = provider?.let { Environment.create(it) } ?: Environment.create()
            val options = if (backend == "qnn") {
                CompiledModel.Options(Accelerator.NPU).apply {
                    qualcommOptions = CompiledModel.QualcommOptions(
                        logLevel = CompiledModel.QualcommOptions.LogLevel.INFO,
                        useHtpPreference = true,
                        htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.SUSTAINED_HIGH_PERFORMANCE,
                        profiling = CompiledModel.QualcommOptions.Profiling.OFF,
                        irJsonDir = File(outputDir, "qnn-ir").apply { mkdirs() }.absolutePath,
                        optimizationLevel = CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_INFERENCE,
                    )
                }
            } else CompiledModel.Options(Accelerator.GPU).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    precision = CompiledModel.GpuOptions.Precision.FP32,
                    backend = CompiledModel.GpuOptions.Backend.OPENCL,
                    numStepsOfCommandBufferPreparations = 1,
                )
            }
            val setupStarted = now()
            model = CompiledModel.create(modelFile.absolutePath, options, environment!!)
            repeat(2) { inputs += model!!.createInputBuffers().single(); outputs += model!!.createOutputBuffers().single() }
            val setupMs = elapsed(setupStarted)
            val slots = Array(2) { Slot(config) }
            val modelWriter = WavFileWriter(File(outputDir, "model-output.wav"), config.sampleRate, 2)
            val residualWriter = WavFileWriter(File(outputDir, "residual.wav"), config.sampleRate, 2)
            val bounded = if (backend == "gpu-bounded") BoundedGpuRuntime.loadAndValidate() else null
            val windowCount = ceil(decoded.frameCount.toDouble() / config.generationSize).toInt()
            val stage = linkedMapOf<String, Long>()
            fun <T> timed(name: String, action: () -> T): T {
                val start = now(); return try { action() } finally { stage[name] = (stage[name] ?: 0L) + (now() - start) }
            }
            fun prepare(slot: Int, index: Int) {
                val start = index * config.generationSize
                slots[slot].writeFrames = minOf(config.generationSize, decoded.frameCount - start)
                slots[slot].mix = decoded.toStereoFloatContextWindow(start - config.trim, config.chunkSize)
                dsps[slot].waveformToNhwcTensorInto(slots[slot].mix, slots[slot].tensor)
                inputs[slot].writeFloat(slots[slot].tensor)
            }
            fun run(slot: Int) { bounded?.beginInference(); try { model!!.run(listOf(inputs[slot]), listOf(outputs[slot])) } finally { bounded?.endInference() } }
            bounded?.resetInferenceCounters()
            val countersBefore = runtimeCounters()
            val processStarted = now()
            timed("initialPrepare") { prepare(0, 0) }
            var previous: Future<*> = executor.submit { run(0) }
            for (index in 0 until windowCount) {
                val completed = index and 1
                val next = (index + 1) and 1
                var current: Future<*>? = null
                if (index + 1 < windowCount) {
                    timed("nextPrepare") { prepare(next, index + 1) }
                    current = executor.submit { run(next) }
                }
                timed("inferenceWait") { previous.get() }
                val outputTensor = timed("outputRead") { outputs[completed].readFloat() }
                timed("iStft") { dsps[completed].nhwcTensorToWaveformInto(outputTensor, slots[completed].separated) }
                timed("residual") {
                    val scale = dspJson.getDouble("modelOutputScale").toFloat()
                    for (channel in 0..1) for (sample in 0 until config.chunkSize) {
                        slots[completed].separated[channel][sample] *= scale
                        slots[completed].residual[channel][sample] = slots[completed].mix[channel][sample] - slots[completed].separated[channel][sample]
                    }
                }
                timed("pcmWrite") {
                    fillPcm16(slots[completed].separated, config.trim, slots[completed].writeFrames, slots[completed].pcm)
                    modelWriter.writePcm16(slots[completed].pcm, 0, slots[completed].writeFrames * 4)
                    fillPcm16(slots[completed].residual, config.trim, slots[completed].writeFrames, slots[completed].pcm)
                    residualWriter.writePcm16(slots[completed].pcm, 0, slots[completed].writeFrames * 4)
                }
                if (current != null) previous = current
            }
            modelWriter.close(); residualWriter.close()
            val processingMs = elapsed(processStarted)
            report.put("status", "complete").put("decodeMs", decodeMs).put("setupMs", setupMs)
                .put("processingMs", processingMs).put("audioSeconds", decoded.frameCount.toDouble() / config.sampleRate)
                .put("windowCount", windowCount).put("realtimeFactor", processingMs / 1000.0 / (decoded.frameCount.toDouble() / config.sampleRate))
                .put("stagesMs", JSONObject(stage.mapValues { it.value / 1e6 }))
                .put("modelOutput", fileEvidence(File(outputDir, "model-output.wav")))
                .put("residualOutput", fileEvidence(File(outputDir, "residual.wav")))
                .put("artRuntimeDelta", counterDelta(countersBefore, runtimeCounters()))
                .put("memory", memoryEvidence()).put("availableAccelerators", JSONArray(environment!!.getAvailableAccelerators().map { it.name }.sorted()))
            bounded?.let { report.put("boundedGpuEvidence", it.evidence()) }
            val qnnIr = File(outputDir, "qnn-ir").walkTopDown().filter { it.isFile && it.length() > 0 }.map { JSONObject().put("path", it.relativeTo(outputDir).invariantSeparatorsPath).put("bytes", it.length()) }.toList()
            report.put("qnnIrFiles", JSONArray(qnnIr)).put("backendQualification", if (backend == "qnn" && qnnIr.isEmpty()) "fallback-only-no-qnn-ir" else "qualified")
        } catch (error: Throwable) {
            report.put("status", "error").put("errorClass", error.javaClass.name).put("message", error.message ?: JSONObject.NULL).put("stack", error.stackTraceToString())
        } finally {
            executor.shutdownNow(); inputs.forEach { it.close() }; outputs.forEach { it.close() }; model?.close(); environment?.close(); dsps.forEach { it.close() }
        }
        report.put("thermalStatusEnd", thermal.currentThermalStatus)
        File(outputDir, "report.json").writeText(report.toString(2))
        check(report.getString("status") == "complete") { report.toString() }
    }

    private fun runNativeManagedFullSong(
        context: Context,
        report: JSONObject,
        outputDir: File,
        modelFile: File,
        audioFile: File,
        dspJson: JSONObject,
        config: MdxDspConfig,
    ) {
        val decodeStarted = now()
        val decoded = AudioPcmDecoder(context).decode(Uri.fromFile(audioFile))
        require(decoded.sampleRate == config.sampleRate && decoded.channelCount == 2)
        val decodeMs = elapsed(decodeStarted)
        val setupStarted = now()
        val pipeline = NativeLiteRtMdxPipeline(
            config = config,
            modelPath = modelFile.absolutePath,
            workerCount = 4,
            backend = NativeLiteRtMdxPipeline.Backend.BOUNDED_GPU,
            slotCount = 2,
        )
        val setupMs = elapsed(setupStarted)
        val executor = Executors.newSingleThreadExecutor()
        val bounded = BoundedGpuRuntime.loadAndValidate()
        val slots = Array(2) { NativeSlot(config) }
        val modelOutput = File(outputDir, "model-output.wav")
        val residualOutput = File(outputDir, "residual.wav")
        try {
            WavFileWriter(modelOutput, config.sampleRate, 2).use { modelWriter ->
                WavFileWriter(residualOutput, config.sampleRate, 2).use { residualWriter ->
                    val windowCount = ceil(decoded.frameCount.toDouble() / config.generationSize).toInt()
                    val stage = linkedMapOf<String, Long>()
                    fun <T> timed(name: String, action: () -> T): T {
                        val start = now()
                        return try { action() } finally {
                            stage[name] = (stage[name] ?: 0L) + (now() - start)
                        }
                    }
                    fun prepare(slot: Int, index: Int) {
                        val start = index * config.generationSize
                        slots[slot].writeFrames = minOf(
                            config.generationSize, decoded.frameCount - start,
                        )
                        slots[slot].mix = decoded.toStereoFloatContextWindow(
                            start - config.trim, config.chunkSize,
                        )
                        pipeline.preprocessInput(slots[slot].mix, slot)
                    }
                    fun submit(slot: Int): Future<*> = executor.submit {
                        bounded.beginInference()
                        try { pipeline.run(slot) } finally { bounded.endInference() }
                    }
                    bounded.resetInferenceCounters()
                    val countersBefore = runtimeCounters()
                    val processStarted = now()
                    timed("initialPrepare") { prepare(0, 0) }
                    var previous = submit(0)
                    for (index in 0 until windowCount) {
                        val completed = index and 1
                        val next = (index + 1) and 1
                        var current: Future<*>? = null
                        if (index + 1 < windowCount) {
                            timed("nextPrepare") { prepare(next, index + 1) }
                            current = submit(next)
                        }
                        timed("inferenceWait") { previous.get() }
                        timed("iStft") {
                            pipeline.postprocessOutputInto(slots[completed].separated, completed)
                        }
                        check(slots[completed].separated.all { channel ->
                            channel.take(256).all { it.isFinite() }
                        })
                        timed("residual") {
                            val scale = dspJson.getDouble("modelOutputScale").toFloat()
                            for (channel in 0..1) for (sample in 0 until config.chunkSize) {
                                slots[completed].separated[channel][sample] *= scale
                                slots[completed].residual[channel][sample] =
                                    slots[completed].mix[channel][sample] -
                                        slots[completed].separated[channel][sample]
                            }
                        }
                        timed("pcmWrite") {
                            fillPcm16(
                                slots[completed].separated, config.trim,
                                slots[completed].writeFrames, slots[completed].pcm,
                            )
                            modelWriter.writePcm16(
                                slots[completed].pcm, 0, slots[completed].writeFrames * 4,
                            )
                            fillPcm16(
                                slots[completed].residual, config.trim,
                                slots[completed].writeFrames, slots[completed].pcm,
                            )
                            residualWriter.writePcm16(
                                slots[completed].pcm, 0, slots[completed].writeFrames * 4,
                            )
                        }
                        if (current != null) previous = current
                    }
                    val processingMs = elapsed(processStarted)
                    report.put("status", "complete")
                        .put("dspProfile", "native-packed-litert-c-w4")
                        .put("decodeMs", decodeMs).put("setupMs", setupMs)
                        .put("processingMs", processingMs)
                        .put("audioSeconds", decoded.frameCount.toDouble() / config.sampleRate)
                        .put("windowCount", windowCount)
                        .put("realtimeFactor", processingMs / 1000.0 /
                            (decoded.frameCount.toDouble() / config.sampleRate))
                        .put("stagesMs", JSONObject(stage.mapValues { it.value / 1e6 }))
                        .put("artRuntimeDelta", counterDelta(countersBefore, runtimeCounters()))
                        .put("memory", memoryEvidence())
                        .put("boundedGpuEvidence", bounded.evidence())
                        .put("backendQualification", "qualified")
                }
            }
            report.put("modelOutput", fileEvidence(modelOutput))
                .put("residualOutput", fileEvidence(residualOutput))
        } finally {
            executor.shutdownNow()
            pipeline.close()
        }
    }

    private class Slot(config: MdxDspConfig) {
        var mix = Array(2) { FloatArray(config.chunkSize) }
        val tensor = FloatArray(config.tensorElementCount)
        val separated = Array(2) { FloatArray(config.chunkSize) }
        val residual = Array(2) { FloatArray(config.chunkSize) }
        val pcm = ByteArray(config.generationSize * 4)
        var writeFrames = 0
    }

    private class NativeSlot(config: MdxDspConfig) {
        var mix = Array(2) { FloatArray(config.chunkSize) }
        val separated = Array(2) { FloatArray(config.chunkSize) }
        val residual = Array(2) { FloatArray(config.chunkSize) }
        val pcm = ByteArray(config.generationSize * 4)
        var writeFrames = 0
    }

    private fun DecodedPcmAudio.toStereoFloatContextWindow(start: Int, frames: Int): Array<FloatArray> {
        val copyStart = maxOf(0, start); val copyEnd = minOf(frameCount, start + frames); val result = Array(2) { FloatArray(frames) }
        if (copyEnd > copyStart) { val source = toStereoFloat(copyStart, copyEnd - copyStart); val offset = copyStart - start; for (channel in 0..1) source[channel].copyInto(result[channel], offset) }
        return result
    }
    private fun fillPcm16(waveform: Array<FloatArray>, start: Int, frames: Int, output: ByteArray) { var offset = 0; for (sample in start until start + frames) for (channel in 0..1) { val value = (waveform[channel][sample].coerceIn(-1f, 1f) * 32767f).roundToInt().coerceIn(-32768, 32767); output[offset++] = value.toByte(); output[offset++] = (value ushr 8).toByte() } }
    private fun fileEvidence(file: File): JSONObject = JSONObject().put("path", file.absolutePath).put("bytes", file.length()).put("sha256", sha256(file))
    private fun memoryEvidence(): JSONObject = Debug.MemoryInfo().also(Debug::getMemoryInfo).let { JSONObject().put("pssKb", it.totalPss).put("nativeHeapBytes", Debug.getNativeHeapAllocatedSize()) }
    private fun runtimeCounters(): Map<String, Long> = RUNTIME_COUNTERS.mapNotNull { key ->
        runCatching { Debug.getRuntimeStat(key) }.getOrNull()?.toLongOrNull()?.let { key to it }
    }.toMap()
    private fun counterDelta(before: Map<String, Long>, after: Map<String, Long>): JSONObject =
        JSONObject(RUNTIME_COUNTERS.associateWith { key ->
            if (before[key] != null && after[key] != null) after.getValue(key) - before.getValue(key)
            else JSONObject.NULL
        })
    private fun now() = SystemClock.elapsedRealtimeNanos()
    private fun elapsed(start: Long) = (now() - start) / 1e6
    private fun sha256(file: File): String = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }
    private class BoundedGpuRuntime private constructor(private val runtimeClass: Class<*>, private val capability: Any) {
        fun resetInferenceCounters() { invoke("resetInferenceCounters") }; fun beginInference() { invoke("beginInference") }; fun endInference() { invoke("endInference") }
        fun evidence(): JSONObject = JSONObject().put("artifactVersion", value("getArtifactVersion")).put("profileId", value("getProfileId")).put("dispatchCount", invoke("getDispatchCount")).put("eventWaitCount", invoke("getEventWaitCount"))
        private fun invoke(name: String): Any? = runtimeClass.getMethod(name).invoke(null); private fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)
        companion object {
            fun loadAndValidate(): BoundedGpuRuntime {
                val clazz = Class.forName("io.github.wluhwluh.bss.litert.BssLiteRtRuntime")
                val cap = requireNotNull(clazz.getMethod("queryCapability").invoke(null))
                fun value(name: String): Any? = cap.javaClass.getMethod(name).invoke(cap)
                require(value("isAvailable") == true)
                require(value("getArtifactVersion") == "2.1.5-bss.2")
                require(value("getProfileId") == "gpu-opencl-bounded-fp32-v1")
                return BoundedGpuRuntime(clazz, cap)
            }
        }
    }
    companion object {
        private val RUNTIME_COUNTERS = listOf(
            "art.gc.gc-count", "art.gc.bytes-allocated", "art.gc.bytes-freed",
            "art.gc.blocking-gc-count",
        )
    }
}
