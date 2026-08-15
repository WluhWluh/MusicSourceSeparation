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
import com.example.musicsourceseparation.model.NativeLiteRtMdxPipeline
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/** Real-model two-slot managed-buffer gate for every known MDX DSP shape. */
@RunWith(AndroidJUnit4::class)
class MdxManagedBufferShapeMatrixInstrumentedTest {
    @Test
    fun runAllShapes() {
        val args = InstrumentationRegistry.getArguments()
        val matrixName = requireNotNull(args.getString("matrixFile"))
        val warmups = args.getString("warmups", "1")!!.toInt().also { require(it in 1..3) }
        val measuredWindows = args.getString("runs", "2")!!.toInt().also { require(it in 2..5) }
        val runId = args.getString("runId") ?: "run-${System.currentTimeMillis()}"
        require(SAFE_NAME.matches(matrixName) && SAFE_NAME.matches(runId))

        val context = ApplicationProvider.getApplicationContext<Context>()
        val thermal = context.getSystemService(PowerManager::class.java)
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val matrixFile = File(root, "matrices/$matrixName")
        require(matrixFile.isFile)
        val matrix = JSONObject(matrixFile.readText())
        require(matrix.getInt("schemaVersion") == 1)
        require(matrix.getInt("contractSchemaVersion") == 2)
        val entries = matrix.getJSONArray("models")
        require(entries.length() == EXPECTED_SHAPE_COUNT)
        validateUniqueMatrix(entries)
        val resultDir = File(root, "mdx-managed-buffer-shape-matrix/$runId").apply {
            deleteRecursively()
            mkdirs()
        }
        val report = JSONObject()
            .put("schemaVersion", 1)
            .put("status", "running")
            .put("matrixId", matrix.getString("matrixId"))
            .put("matrixFile", matrixName)
            .put("matrixSha256", sha256(matrixFile))
            .put("runtimeId", BuildConfig.BENCHMARK_RUNTIME_ID)
            .put("runtimeArtifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256)
            .put("sourceRevision", BuildConfig.BENCHMARK_SOURCE_REVISION)
            .put("sourceDirty", BuildConfig.BENCHMARK_SOURCE_DIRTY)
            .put("backend", "gpu-bounded")
            .put("tensorBoundary", "native-managed")
            .put("dspProfile", "native-packed-litert-c-w4")
            .put("slotCount", 2)
            .put("warmupsPerShape", warmups)
            .put("measuredWindowsPerShape", measuredWindows)
            .put("device", deviceEvidence())
            .put("thermalStatusStart", thermal.currentThermalStatus)
        val executor = Executors.newSingleThreadExecutor()
        val shapeResults = JSONArray()
        val beforeMatrix = memoryEvidence()
        try {
            repeat(entries.length()) { index ->
                shapeResults.put(runShape(
                    entry = entries.getJSONObject(index),
                    root = root,
                    warmups = warmups,
                    measuredWindows = measuredWindows,
                    executor = executor,
                    thermal = thermal,
                ))
            }
            val allQualified = (0 until shapeResults.length()).all {
                shapeResults.getJSONObject(it).getString("qualification") == "qualified"
            }
            val firstAfterClose = shapeResults.getJSONObject(0).getJSONObject("afterClose")
            val lastAfterClose = shapeResults.getJSONObject(shapeResults.length() - 1)
                .getJSONObject("afterClose")
            val pssGrowthKb = lastAfterClose.getInt("pssKb") - firstAfterClose.getInt("pssKb")
            val nativeHeapGrowthBytes = lastAfterClose.getLong("nativeHeapAllocatedBytes") -
                firstAfterClose.getLong("nativeHeapAllocatedBytes")
            val memoryQualified = pssGrowthKb <= MAX_PSS_GROWTH_KB &&
                nativeHeapGrowthBytes <= MAX_NATIVE_HEAP_GROWTH_BYTES
            report.put("shapeResults", shapeResults)
                .put("completedShapes", shapeResults.length())
                .put("completedMeasuredWindows", shapeResults.length() * measuredWindows)
                .put("allShapesQualified", allQualified)
                .put("afterClosePssGrowthKb", pssGrowthKb)
                .put("afterCloseNativeHeapGrowthBytes", nativeHeapGrowthBytes)
                .put("memoryThresholds", JSONObject()
                    .put("maximumPssGrowthKb", MAX_PSS_GROWTH_KB)
                    .put("maximumNativeHeapGrowthBytes", MAX_NATIVE_HEAP_GROWTH_BYTES))
                .put("qualification", if (allQualified && memoryQualified) {
                    "qualified"
                } else {
                    "rejected"
                })
                .put("status", "complete")
        } catch (error: Throwable) {
            report.put("status", "error")
                .put("errorClass", error.javaClass.name)
                .put("message", error.message ?: JSONObject.NULL)
                .put("stack", error.stackTraceToString())
                .put("shapeResults", shapeResults)
        } finally {
            executor.shutdownNow()
            report.put("thermalStatusEnd", thermal.currentThermalStatus)
                .put("beforeMatrix", beforeMatrix)
                .put("afterMatrix", memoryEvidence())
            File(resultDir, "report.json").writeText(report.toString(2))
        }
        check(report.getString("status") == "complete" &&
            report.getString("qualification") == "qualified") { report.toString() }
    }

    private fun runShape(
        entry: JSONObject,
        root: File,
        warmups: Int,
        measuredWindows: Int,
        executor: ExecutorService,
        thermal: PowerManager,
    ): JSONObject {
        val modelId = entry.getString("modelId")
        val modelFile = File(root, "models/${entry.getString("modelFile")}")
        val contractFile = File(root, "contracts/${entry.getString("contractFile")}")
        require(modelFile.isFile && contractFile.isFile)
        val contract = JSONObject(contractFile.readText())
        val artifact = contract.getJSONObject("artifact")
        val dsp = contract.getJSONObject("dsp")
        require(contract.getInt("contractSchemaVersion") == 2)
        require(contract.getString("modelId") == modelId)
        require(artifact.getString("fileName") == modelFile.name)
        require(artifact.getString("sha256") == entry.getString("sha256"))
        require(sha256(modelFile) == entry.getString("sha256"))
        require(dsp.getInt("nFft") == entry.getInt("nFft"))
        require(dsp.getInt("hopLength") == entry.getInt("hopLength"))
        require(dsp.getInt("dimF") == entry.getInt("dimF"))
        require(dsp.getInt("dimTPower") == entry.getInt("dimTPower"))
        val config = MdxDspConfig(
            sampleRate = dsp.getInt("sampleRate"),
            nFft = dsp.getInt("nFft"),
            hopLength = dsp.getInt("hopLength"),
            dimF = dsp.getInt("dimF"),
            dimTPower = dsp.getInt("dimTPower"),
        )
        val waveform = fixture(config)
        val separated = Array(2) { Array(2) { FloatArray(config.chunkSize) } }
        val beforeCreate = memoryEvidence()
        val countersBefore = runtimeCounters()
        val thermalStart = thermal.currentThermalStatus
        val setupStarted = now()
        val pipeline = NativeLiteRtMdxPipeline(
            config = config,
            modelPath = modelFile.absolutePath,
            workerCount = 4,
            backend = NativeLiteRtMdxPipeline.Backend.BOUNDED_GPU,
            slotCount = 2,
        )
        val setupMs = elapsed(setupStarted)
        val bounded = BoundedGpuRuntime.loadAndValidate()
        val hashes = mutableListOf<String>()
        var finite = true
        var peakAbsolute = 0f
        var preprocessMs = 0.0
        var waitMs = 0.0
        var postprocessMs = 0.0
        var totalMs = 0.0
        var dispatch = JSONObject()
        var duringRun = JSONObject()
        try {
            repeat(warmups) { index ->
                val slot = index and 1
                pipeline.preprocessInput(waveform, slot)
                bounded.beginInference()
                try { pipeline.run(slot) } finally { bounded.endInference() }
                pipeline.postprocessOutputInto(separated[slot], slot)
            }
            bounded.resetInferenceCounters()
            val totalStarted = now()
            val initialPreprocess = now()
            pipeline.preprocessInput(waveform, 0)
            preprocessMs += elapsed(initialPreprocess)
            var previous = submit(executor, pipeline, bounded, 0)
            repeat(measuredWindows) { index ->
                val completed = index and 1
                val next = (index + 1) and 1
                val current: Future<*>? = if (index + 1 < measuredWindows) {
                    val preprocessStarted = now()
                    pipeline.preprocessInput(waveform, next)
                    preprocessMs += elapsed(preprocessStarted)
                    submit(executor, pipeline, bounded, next)
                } else {
                    null
                }
                val waitStarted = now()
                previous.get()
                waitMs += elapsed(waitStarted)
                val postprocessStarted = now()
                pipeline.postprocessOutputInto(separated[completed], completed)
                postprocessMs += elapsed(postprocessStarted)
                val analysis = analyze(separated[completed])
                finite = finite && analysis.finite
                peakAbsolute = maxOf(peakAbsolute, analysis.peakAbsolute)
                hashes += analysis.hash
                if (current != null) previous = current
            }
            totalMs = elapsed(totalStarted)
            dispatch = bounded.evidence()
            duringRun = memoryEvidence()
        } finally {
            pipeline.close()
        }
        Runtime.getRuntime().gc()
        System.runFinalization()
        Thread.sleep(100)
        val afterClose = memoryEvidence()
        val dispatchQualified = dispatch.getLong("dispatchCount") > 0L &&
            dispatch.getLong("dispatchCount") == dispatch.getLong("eventWaitCount")
        val slotsStable = hashes.size == measuredWindows && hashes.distinct().size == 1
        return JSONObject()
            .put("modelId", modelId)
            .put("modelFile", modelFile.name)
            .put("modelSha256", entry.getString("sha256"))
            .put("contractFile", contractFile.name)
            .put("contractSha256", sha256(contractFile))
            .put("shape", JSONObject()
                .put("nFft", config.nFft).put("hopLength", config.hopLength)
                .put("dimF", config.dimF).put("dimT", config.dimT)
                .put("chunkSize", config.chunkSize))
            .put("thermalStatusStart", thermalStart)
            .put("thermalStatusEnd", thermal.currentThermalStatus)
            .put("setupMs", setupMs)
            .put("preprocessMs", preprocessMs)
            .put("inferenceWaitMs", waitMs)
            .put("postprocessMs", postprocessMs)
            .put("totalMs", totalMs)
            .put("meanMsPerWindow", totalMs / measuredWindows)
            .put("finite", finite)
            .put("peakAbsolute", peakAbsolute.toDouble())
            .put("perWindowHashFnv1a64", JSONArray(hashes))
            .put("slotsStable", slotsStable)
            .put("beforeCreate", beforeCreate)
            .put("duringRun", duringRun)
            .put("afterClose", afterClose)
            .put("artRuntimeDelta", counterDelta(countersBefore, runtimeCounters()))
            .put("boundedGpuEvidence", dispatch)
            .put("qualification", if (finite && slotsStable && dispatchQualified) {
                "qualified"
            } else {
                "rejected"
            })
    }

    private fun validateUniqueMatrix(entries: JSONArray) {
        val modelIds = mutableSetOf<String>()
        val shapes = mutableSetOf<String>()
        repeat(entries.length()) { index ->
            val entry = entries.getJSONObject(index)
            require(modelIds.add(entry.getString("modelId")))
            val shape = listOf("nFft", "hopLength", "dimF", "dimTPower")
                .joinToString("/") { entry.getInt(it).toString() }
            require(shapes.add(shape)) { "Duplicate matrix shape $shape" }
            require(SHA256.matches(entry.getString("sha256")))
            require(SAFE_NAME.matches(entry.getString("modelFile")))
            require(SAFE_NAME.matches(entry.getString("contractFile")))
        }
        require(shapes.size == EXPECTED_SHAPE_COUNT)
    }

    private fun submit(
        executor: ExecutorService,
        pipeline: NativeLiteRtMdxPipeline,
        bounded: BoundedGpuRuntime,
        slot: Int,
    ): Future<*> = executor.submit {
        bounded.beginInference()
        try { pipeline.run(slot) } finally { bounded.endInference() }
    }

    private data class OutputAnalysis(
        val finite: Boolean,
        val peakAbsolute: Float,
        val hash: String,
    )

    private fun analyze(waveform: Array<FloatArray>): OutputAnalysis {
        var finite = true
        var peakAbsolute = 0f
        var hash = 0xcbf29ce484222325UL
        for (channel in waveform) for (value in channel) {
            finite = finite && value.isFinite()
            peakAbsolute = maxOf(peakAbsolute, abs(value))
            hash = (hash xor value.toRawBits().toUInt().toULong()) * 0x100000001b3UL
        }
        return OutputAnalysis(finite, peakAbsolute, hash.toString(16).padStart(16, '0'))
    }

    private fun fixture(config: MdxDspConfig): Array<FloatArray> = Array(2) { channel ->
        FloatArray(config.chunkSize) { sample ->
            (0.1 * sin(2.0 * PI * (220 + channel * 37) * sample / config.sampleRate) +
                0.01 * sin(sample * 0.013)).toFloat()
        }
    }

    private fun deviceEvidence(): JSONObject = JSONObject()
        .put("manufacturer", Build.MANUFACTURER)
        .put("model", Build.MODEL)
        .put("socManufacturer", Build.SOC_MANUFACTURER)
        .put("socModel", Build.SOC_MODEL)
        .put("sdk", Build.VERSION.SDK_INT)
        .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))

    private fun memoryEvidence(): JSONObject = Debug.MemoryInfo().also(Debug::getMemoryInfo).let {
        JSONObject().put("pssKb", it.totalPss)
            .put("privateDirtyKb", it.totalPrivateDirty)
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

    private fun now() = SystemClock.elapsedRealtimeNanos()
    private fun elapsed(start: Long) = (now() - start) / 1_000_000.0
    private fun sha256(file: File): String = MessageDigest.getInstance("SHA-256")
        .digest(file.readBytes()).joinToString("") { "%02x".format(it) }

    private class BoundedGpuRuntime private constructor(
        private val runtimeClass: Class<*>,
        private val capability: Any,
    ) {
        fun resetInferenceCounters() { invoke("resetInferenceCounters") }
        fun beginInference() { invoke("beginInference") }
        fun endInference() { invoke("endInference") }
        fun evidence(): JSONObject = JSONObject()
            .put("artifactVersion", value("getArtifactVersion"))
            .put("profileId", value("getProfileId"))
            .put("kernelBatchSize", value("getKernelBatchSize"))
            .put("commandQueueWindowSize", value("getCommandQueueWindowSize"))
            .put("dispatchCount", invoke("getDispatchCount"))
            .put("eventWaitCount", invoke("getEventWaitCount"))
        private fun invoke(name: String): Any? = runtimeClass.getMethod(name).invoke(null)
        private fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)

        companion object {
            fun loadAndValidate(): BoundedGpuRuntime {
                val clazz = Class.forName("io.github.wluhwluh.bss.litert.BssLiteRtRuntime")
                val capability = requireNotNull(clazz.getMethod("queryCapability").invoke(null))
                fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)
                require(value("isAvailable") == true)
                require(value("getArtifactVersion") == "2.1.5-bss.2")
                require(value("getProfileId") == "gpu-opencl-bounded-fp32-v1")
                require(value("getKernelBatchSize") == 1)
                require(value("getCommandQueueWindowSize") == 1)
                return BoundedGpuRuntime(clazz, capability)
            }
        }
    }

    companion object {
        private const val EXPECTED_SHAPE_COUNT = 13
        private const val MAX_PSS_GROWTH_KB = 131_072
        private const val MAX_NATIVE_HEAP_GROWTH_BYTES = 32L * 1024 * 1024
        private val SAFE_NAME = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
        private val SHA256 = Regex("[0-9a-f]{64}")
        private val RUNTIME_COUNTERS = listOf(
            "art.gc.gc-count", "art.gc.bytes-allocated", "art.gc.bytes-freed",
            "art.gc.blocking-gc-count",
        )
    }
}
