package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.BuildConfig
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
import kotlin.math.PI
import kotlin.math.sin
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/** MDX STFT/input preparation overlapped with one CompiledModel invocation. */
@RunWith(AndroidJUnit4::class)
class MdxDoubleBufferInstrumentedTest {
    @Test
    fun runDoubleBuffer() {
        val args = InstrumentationRegistry.getArguments()
        val modelId = requireNotNull(args.getString("modelId"))
        val modelName = requireNotNull(args.getString("modelFile"))
        val contractName = requireNotNull(args.getString("contractFile"))
        val backend = args.getString("backend", "gpu-bounded")!!
        require(backend == "gpu-bounded" || backend == "qnn")
        val tensorBoundary = args.getString("tensorBoundary", "java-tensor")!!
        require(tensorBoundary == "java-tensor" || tensorBoundary == "native-managed")
        if (tensorBoundary == "native-managed") require(backend == "gpu-bounded")
        val runs = args.getString("runs", "5")!!.toInt().coerceIn(2, 20)
        val warmups = args.getString("warmups", "1")!!.toInt().coerceIn(0, 5)
        val runId = args.getString("runId") ?: "run-${System.currentTimeMillis()}"
        require(RUN_ID.matches(runId))
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val modelFile = File(root, "models/$modelName")
        val contractFile = File(root, "contracts/$contractName")
        require(modelFile.isFile && contractFile.isFile)
        val contract = JSONObject(contractFile.readText())
        val artifact = contract.getJSONObject("artifact")
        require(sha256(modelFile) == artifact.getString("sha256"))
        val dspJson = contract.getJSONObject("dsp")
        val tensorContract = contract.getJSONObject("tensorContract")
        val inputName = tensorContract.getJSONObject("input").getString("name")
        val outputName = tensorContract.getJSONObject("output").getString("name")
        val config = MdxDspConfig(
            sampleRate = dspJson.getInt("sampleRate"), nFft = dspJson.getInt("nFft"),
            hopLength = dspJson.getInt("hopLength"), dimF = dspJson.getInt("dimF"),
            dimTPower = dspJson.getInt("dimTPower"),
        )
        val resultDir = File(root, "mdx-double-buffer/$modelId/$backend/$runId").apply {
            deleteRecursively(); mkdirs()
        }
        val report = JSONObject().put("schemaVersion", 1).put("status", "complete")
            .put("modelId", modelId).put("modelFile", modelName)
            .put("modelSha256", artifact.getString("sha256"))
            .put("contractFile", contractName).put("contractSha256", sha256(contractFile))
            .put("contractId", contract.getString("contractId"))
            .put("runtime", "LiteRT CompiledModel 2.1.5")
            .put("runtimeId", BuildConfig.BENCHMARK_RUNTIME_ID)
            .put("sourceRevision", BuildConfig.BENCHMARK_SOURCE_REVISION)
            .put("sourceDirty", BuildConfig.BENCHMARK_SOURCE_DIRTY)
            .put("backend", backend).put("tensorBoundary", tensorBoundary)
            .put("runs", runs).put("warmups", warmups)
            .put("device", JSONObject().put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL).put("socManufacturer", Build.SOC_MANUFACTURER)
                .put("socModel", Build.SOC_MODEL).put("sdk", Build.VERSION.SDK_INT)
                .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())))
            .put("doubleBuffer", JSONObject().put("slots", 2)
                .put("modelInstances", 1)
                .put("overlap", "STFT+input write window n+1 with invoke window n"))
        if (tensorBoundary == "native-managed") {
            try {
                runNativeManaged(report, modelFile, config, runs, warmups)
            } catch (error: Throwable) {
                report.put("status", "error").put("errorClass", error.javaClass.name)
                    .put("message", error.message ?: JSONObject.NULL)
                    .put("stack", error.stackTraceToString())
            }
            File(resultDir, "report.json").writeText(report.toString(2))
            check(report.getString("status") == "complete") { report.toString() }
            return
        }
        var environment: Environment? = null
        var model: CompiledModel? = null
        val inputs = mutableListOf<TensorBuffer>()
        val outputs = mutableListOf<TensorBuffer>()
        val executor = Executors.newSingleThreadExecutor()
        val dsps = Array(2) { NativeMdxDsp(config, 4, NativeMdxDsp.Mode.PACKED_REAL) }
        val inputTensors = Array(2) { FloatArray(config.tensorElementCount) }
        val separated = Array(2) { Array(2) { FloatArray(config.chunkSize) } }
        val boundedRuntime = if (backend == "gpu-bounded") BoundedGpuRuntime.loadAndValidate() else null
        try {
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
                        irJsonDir = File(resultDir, "qnn-ir").apply { mkdirs() }.absolutePath,
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
            repeat(2) {
                inputs += model!!.createInputBuffer(inputName)
                outputs += model!!.createOutputBuffer(outputName)
            }
            report.put("setupMs", elapsed(setupStarted))
            val waveform = fixture(config)
            repeat(warmups) { index ->
                val slot = index and 1
                prepare(dsps[slot], waveform, inputTensors[slot], inputs[slot])
                runModel(model!!, inputs[slot], outputs[slot], boundedRuntime)
                dsps[slot].nhwcTensorToWaveformInto(outputs[slot].readFloat(), separated[slot])
            }
            boundedRuntime?.resetInferenceCounters()
            val countersBefore = runtimeCounters()
            report.put("dspProfile", "native-packed-w4")
            report.put("sequential", sequential(runs, model!!, dsps, waveform, inputTensors, separated, inputs, outputs, boundedRuntime))
            report.put("doubleBuffered", doubleBuffered(runs, model!!, dsps, waveform, inputTensors, separated, inputs, outputs, executor, boundedRuntime))
            report.put("artRuntimeDelta", counterDelta(countersBefore, runtimeCounters()))
                .put("memory", memoryEvidence())
            report.put("availableAccelerators", JSONArray(environment!!.getAvailableAccelerators().map { it.name }.sorted()))
            boundedRuntime?.let { report.put("boundedGpuEvidence", it.evidence()) }
            val qnnIr = File(resultDir, "qnn-ir").takeIf { it.isDirectory }?.walkTopDown()
                ?.filter { it.isFile && it.length() > 0L }
                ?.map { JSONObject().put("path", it.relativeTo(resultDir).invariantSeparatorsPath).put("bytes", it.length()) }
                ?.toList() ?: emptyList()
            report.put("qnnIrFiles", JSONArray(qnnIr))
            report.put("backendQualification", if (backend == "qnn" && qnnIr.isEmpty()) "fallback-only-no-qnn-ir" else "qualified")
        } catch (error: Throwable) {
            report.put("status", "error").put("errorClass", error.javaClass.name)
                .put("message", error.message ?: JSONObject.NULL).put("stack", error.stackTraceToString())
        } finally {
            executor.shutdownNow(); outputs.forEach { it.close() }; inputs.forEach { it.close() }
            model?.close(); environment?.close(); dsps.forEach { it.close() }
        }
        File(resultDir, "report.json").writeText(report.toString(2))
        check(report.getString("status") == "complete") { report.toString() }
    }

    private fun runNativeManaged(
        report: JSONObject,
        modelFile: File,
        config: MdxDspConfig,
        runs: Int,
        warmups: Int,
    ) {
        val bounded = BoundedGpuRuntime.loadAndValidate()
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
        val waveform = fixture(config)
        val separated = Array(2) { Array(2) { FloatArray(config.chunkSize) } }
        try {
            repeat(warmups) { index ->
                val slot = index and 1
                pipeline.preprocessInput(waveform, slot)
                bounded.beginInference()
                try { pipeline.run(slot) } finally { bounded.endInference() }
                pipeline.postprocessOutputInto(separated[slot], slot)
            }
            bounded.resetInferenceCounters()
            val countersBefore = runtimeCounters()
            val sequential = nativeSequential(runs, pipeline, waveform, separated, bounded)
            val doubleBuffered = nativeDoubleBuffered(
                runs, pipeline, waveform, separated, executor, bounded,
            )
            report.put("dspProfile", "native-packed-litert-c-w4")
                .put("setupMs", setupMs)
                .put("sequential", sequential)
                .put("doubleBuffered", doubleBuffered)
                .put("artRuntimeDelta", counterDelta(countersBefore, runtimeCounters()))
                .put("memory", memoryEvidence())
                .put("boundedGpuEvidence", bounded.evidence())
                .put("backendQualification", "qualified")
        } finally {
            executor.shutdownNow()
            pipeline.close()
        }
    }

    private fun nativeSequential(
        runs: Int,
        pipeline: NativeLiteRtMdxPipeline,
        waveform: Array<FloatArray>,
        separated: Array<Array<FloatArray>>,
        bounded: BoundedGpuRuntime,
    ): JSONObject {
        val samples = JSONArray()
        repeat(runs) { index ->
            val slot = index and 1
            val start = now()
            val prep = now(); pipeline.preprocessInput(waveform, slot); val prepMs = elapsed(prep)
            val invoke = now()
            bounded.beginInference()
            try { pipeline.run(slot) } finally { bounded.endInference() }
            val invokeMs = elapsed(invoke)
            val inverse = now(); pipeline.postprocessOutputInto(separated[slot], slot)
            val inverseMs = elapsed(inverse)
            check(separated[slot].all { channel -> channel.take(256).all { it.isFinite() } })
            samples.put(JSONObject().put("stftInputMs", prepMs)
                .put("inferenceMs", invokeMs).put("outputReadMs", 0.0)
                .put("iStftOlaMs", inverseMs).put("totalMs", elapsed(start)))
        }
        return summary(samples, "totalMs")
    }

    private fun nativeDoubleBuffered(
        runs: Int,
        pipeline: NativeLiteRtMdxPipeline,
        waveform: Array<FloatArray>,
        separated: Array<Array<FloatArray>>,
        executor: java.util.concurrent.ExecutorService,
        bounded: BoundedGpuRuntime,
    ): JSONObject {
        fun submit(slot: Int): Future<*> = executor.submit {
            bounded.beginInference()
            try { pipeline.run(slot) } finally { bounded.endInference() }
        }
        val samples = JSONArray()
        pipeline.preprocessInput(waveform, 0)
        var previous = submit(0)
        val pipelineStart = now()
        repeat(runs) { index ->
            val completed = index and 1
            val next = (index + 1) and 1
            val prepMs: Double
            val current: Future<*>?
            if (index + 1 < runs) {
                val prep = now(); pipeline.preprocessInput(waveform, next); prepMs = elapsed(prep)
                current = submit(next)
            } else {
                prepMs = 0.0
                current = null
            }
            val wait = now(); previous.get(); val waitMs = elapsed(wait)
            val inverse = now(); pipeline.postprocessOutputInto(separated[completed], completed)
            val inverseMs = elapsed(inverse)
            check(separated[completed].all { channel -> channel.take(256).all { it.isFinite() } })
            samples.put(JSONObject().put("nextStftInputMs", prepMs)
                .put("inferenceWaitMs", waitMs).put("outputReadMs", 0.0)
                .put("previousIStftOlaMs", inverseMs))
            if (current != null) previous = current
        }
        val pipelineMs = elapsed(pipelineStart)
        val stageValues = (0 until samples.length()).map { index ->
            samples.getJSONObject(index).let {
                it.getDouble("nextStftInputMs") + it.getDouble("inferenceWaitMs") +
                    it.getDouble("outputReadMs") + it.getDouble("previousIStftOlaMs")
            }
        }
        return JSONObject().put("count", runs).put("samples", samples)
            .put("pipelineTotalMs", pipelineMs).put("meanMs", pipelineMs / runs)
            .put("stageSumMeanMs", stageValues.average())
    }

    private fun prepare(dsp: NativeMdxDsp, waveform: Array<FloatArray>, tensor: FloatArray, input: TensorBuffer) {
        dsp.waveformToNhwcTensorInto(waveform, tensor)
        input.writeFloat(tensor)
    }

    private fun sequential(runs: Int, model: CompiledModel, dsps: Array<NativeMdxDsp>, waveform: Array<FloatArray>, tensors: Array<FloatArray>, separated: Array<Array<FloatArray>>, inputs: List<TensorBuffer>, outputs: List<TensorBuffer>, bounded: BoundedGpuRuntime?): JSONObject {
        val samples = JSONArray()
        repeat(runs) { index ->
            val slot = index and 1; val start = now(); val prep = now(); prepare(dsps[slot], waveform, tensors[slot], inputs[slot]); val prepMs = elapsed(prep)
            val invoke = now(); runModel(model, inputs[slot], outputs[slot], bounded); val invokeMs = elapsed(invoke)
            val read = now(); val values = outputs[slot].readFloat(); val readMs = elapsed(read); check(values.take(256).all { it.isFinite() })
            val inverse = now(); dsps[slot].nhwcTensorToWaveformInto(values, separated[slot]); val inverseMs = elapsed(inverse)
            samples.put(JSONObject().put("stftInputMs", prepMs).put("inferenceMs", invokeMs).put("outputReadMs", readMs).put("iStftOlaMs", inverseMs).put("totalMs", elapsed(start)))
        }
        return summary(samples, "totalMs")
    }

    private fun doubleBuffered(runs: Int, model: CompiledModel, dsps: Array<NativeMdxDsp>, waveform: Array<FloatArray>, tensors: Array<FloatArray>, separated: Array<Array<FloatArray>>, inputs: List<TensorBuffer>, outputs: List<TensorBuffer>, executor: java.util.concurrent.ExecutorService, bounded: BoundedGpuRuntime?): JSONObject {
        val samples = JSONArray(); prepare(dsps[0], waveform, tensors[0], inputs[0])
        var previous: Future<*> = executor.submit { runModel(model, inputs[0], outputs[0], bounded) }
        val pipelineStart = now()
        repeat(runs) { index ->
            val completedSlot = index and 1
            val nextSlot = (index + 1) and 1
            val prepMs: Double
            val current: Future<*>?
            if (index + 1 < runs) {
                val prep = now(); prepare(dsps[nextSlot], waveform, tensors[nextSlot], inputs[nextSlot]); prepMs = elapsed(prep)
                current = executor.submit { runModel(model, inputs[nextSlot], outputs[nextSlot], bounded) }
            } else {
                prepMs = 0.0
                current = null
            }
            val wait = now(); previous.get(); val waitMs = elapsed(wait)
            val read = now(); val values = outputs[completedSlot].readFloat(); val readMs = elapsed(read); check(values.take(256).all { it.isFinite() })
            val inverse = now(); dsps[completedSlot].nhwcTensorToWaveformInto(values, separated[completedSlot]); val inverseMs = elapsed(inverse)
            samples.put(JSONObject().put("nextStftInputMs", prepMs).put("inferenceWaitMs", waitMs).put("outputReadMs", readMs).put("previousIStftOlaMs", inverseMs))
            if (current != null) previous = current
        }
        val pipelineMs = elapsed(pipelineStart)
        val values = (0 until samples.length()).map { samples.getJSONObject(it).getDouble("nextStftInputMs") + samples.getJSONObject(it).getDouble("inferenceWaitMs") + samples.getJSONObject(it).getDouble("outputReadMs") + samples.getJSONObject(it).getDouble("previousIStftOlaMs") }
        return JSONObject().put("count", runs).put("samples", samples).put("pipelineTotalMs", pipelineMs).put("meanMs", pipelineMs / runs).put("stageSumMeanMs", values.average())
    }

    private fun summary(samples: JSONArray, key: String): JSONObject { val values = (0 until samples.length()).map { samples.getJSONObject(it).getDouble(key) }.sorted(); return JSONObject().put("count", values.size).put("samples", samples).put("medianMs", values[values.size / 2]).put("meanMs", values.average()).put("minMs", values.first()).put("maxMs", values.last()) }
    private fun runModel(model: CompiledModel, input: TensorBuffer, output: TensorBuffer, bounded: BoundedGpuRuntime?) { bounded?.beginInference(); try { model.run(listOf(input), listOf(output)) } finally { bounded?.endInference() } }
    private fun fixture(config: MdxDspConfig): Array<FloatArray> = Array(2) { c -> FloatArray(config.chunkSize) { i -> (0.1 * sin(2.0 * PI * (220 + c * 37) * i / config.sampleRate) + 0.01 * sin(i * 0.013)).toFloat() } }
    private fun now() = SystemClock.elapsedRealtimeNanos()
    private fun elapsed(start: Long) = (now() - start) / 1_000_000.0
    private fun memoryEvidence(): JSONObject = Debug.MemoryInfo().also(Debug::getMemoryInfo).let {
        JSONObject().put("pssKb", it.totalPss)
            .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
    }
    private fun runtimeCounters(): Map<String, Long> = RUNTIME_COUNTERS.mapNotNull { key ->
        runCatching { Debug.getRuntimeStat(key) }.getOrNull()?.toLongOrNull()?.let { key to it }
    }.toMap()
    private fun counterDelta(before: Map<String, Long>, after: Map<String, Long>): JSONObject =
        JSONObject(RUNTIME_COUNTERS.associateWith { key ->
            if (before[key] != null && after[key] != null) after.getValue(key) - before.getValue(key)
            else JSONObject.NULL
        })
    private fun sha256(file: File): String = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }
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
    companion object {
        private val RUN_ID = Regex("[A-Za-z0-9._-]{1,80}")
        private val RUNTIME_COUNTERS = listOf(
            "art.gc.gc-count", "art.gc.bytes-allocated", "art.gc.bytes-freed",
            "art.gc.blocking-gc-count",
        )
    }
}
