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
import kotlin.math.sin
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith

/** Repeated two-slot CompiledModel/buffer lifecycle gate with 100 measured windows. */
@RunWith(AndroidJUnit4::class)
class MdxManagedBufferStabilityInstrumentedTest {
    @Test
    fun runCreateRunCloseStability() {
        val args = InstrumentationRegistry.getArguments()
        val modelId = requireNotNull(args.getString("modelId"))
        val modelName = requireNotNull(args.getString("modelFile"))
        val contractName = requireNotNull(args.getString("contractFile"))
        val cycles = args.getString("cycles", "5")!!.toInt().also { require(it in 2..10) }
        val windowsPerCycle = args.getString("windowsPerCycle", "20")!!.toInt()
            .also { require(it in 5..50) }
        val warmups = args.getString("warmups", "2")!!.toInt().also { require(it in 1..5) }
        require(cycles * windowsPerCycle == 100) { "Stability gate must measure 100 windows." }
        val runId = args.getString("runId") ?: "run-${System.currentTimeMillis()}"
        require(RUN_ID.matches(runId))

        val context = ApplicationProvider.getApplicationContext<Context>()
        val thermal = context.getSystemService(PowerManager::class.java)
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val modelFile = File(root, "models/$modelName")
        val contractFile = File(root, "contracts/$contractName")
        require(modelFile.isFile && contractFile.isFile)
        val contract = JSONObject(contractFile.readText())
        val artifact = contract.getJSONObject("artifact")
        require(sha256(modelFile) == artifact.getString("sha256"))
        val dsp = contract.getJSONObject("dsp")
        val config = MdxDspConfig(
            sampleRate = dsp.getInt("sampleRate"),
            nFft = dsp.getInt("nFft"),
            hopLength = dsp.getInt("hopLength"),
            dimF = dsp.getInt("dimF"),
            dimTPower = dsp.getInt("dimTPower"),
        )
        val resultDir = File(root, "mdx-managed-buffer-stability/$modelId/$runId").apply {
            deleteRecursively()
            mkdirs()
        }
        val report = JSONObject()
            .put("schemaVersion", 1)
            .put("status", "running")
            .put("modelId", modelId)
            .put("modelFile", modelName)
            .put("modelSha256", artifact.getString("sha256"))
            .put("contractFile", contractName)
            .put("contractSha256", sha256(contractFile))
            .put("contractId", contract.getString("contractId"))
            .put("runtimeId", BuildConfig.BENCHMARK_RUNTIME_ID)
            .put("runtimeArtifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256)
            .put("sourceRevision", BuildConfig.BENCHMARK_SOURCE_REVISION)
            .put("sourceDirty", BuildConfig.BENCHMARK_SOURCE_DIRTY)
            .put("backend", "gpu-bounded")
            .put("tensorBoundary", "native-managed")
            .put("dspProfile", "native-packed-litert-c-w4")
            .put("slotCount", 2)
            .put("cycles", cycles)
            .put("windowsPerCycle", windowsPerCycle)
            .put("measuredWindows", cycles * windowsPerCycle)
            .put("warmupsPerCycle", warmups)
            .put("device", deviceEvidence())
            .put("thermalStatusStart", thermal.currentThermalStatus)
        val executor = Executors.newSingleThreadExecutor()
        val waveform = fixture(config)
        val separated = Array(2) { Array(2) { FloatArray(config.chunkSize) } }
        val cycleReports = JSONArray()
        try {
            repeat(cycles) { cycleIndex ->
                cycleReports.put(runCycle(
                    cycleIndex = cycleIndex,
                    modelFile = modelFile,
                    config = config,
                    waveform = waveform,
                    separated = separated,
                    warmups = warmups,
                    measuredWindows = windowsPerCycle,
                    executor = executor,
                    thermal = thermal,
                ))
            }
            val firstAfterClose = cycleReports.getJSONObject(0).getJSONObject("afterClose")
            val lastAfterClose = cycleReports.getJSONObject(cycles - 1).getJSONObject("afterClose")
            val pssGrowthKb = lastAfterClose.getInt("pssKb") - firstAfterClose.getInt("pssKb")
            val nativeHeapGrowthBytes = lastAfterClose.getLong("nativeHeapAllocatedBytes") -
                firstAfterClose.getLong("nativeHeapAllocatedBytes")
            val allFinite = (0 until cycles).all {
                cycleReports.getJSONObject(it).getBoolean("finite")
            }
            val outputHashes = (0 until cycles).map {
                cycleReports.getJSONObject(it).getString("outputHashFnv1a64")
            }
            val outputStable = outputHashes.distinct().size == 1
            val allDispatchQualified = (0 until cycles).all {
                val evidence = cycleReports.getJSONObject(it).getJSONObject("boundedGpuEvidence")
                evidence.getLong("dispatchCount") > 0L &&
                    evidence.getLong("dispatchCount") == evidence.getLong("eventWaitCount")
            }
            val memoryQualified = pssGrowthKb <= MAX_PSS_GROWTH_KB &&
                nativeHeapGrowthBytes <= MAX_NATIVE_HEAP_GROWTH_BYTES
            report.put("cycleResults", cycleReports)
                .put("completedWindows", cycles * windowsPerCycle)
                .put("allFinite", allFinite)
                .put("outputStableAcrossCycles", outputStable)
                .put("allDispatchQualified", allDispatchQualified)
                .put("afterClosePssGrowthKb", pssGrowthKb)
                .put("afterCloseNativeHeapGrowthBytes", nativeHeapGrowthBytes)
                .put("memoryThresholds", JSONObject()
                    .put("maximumPssGrowthKb", MAX_PSS_GROWTH_KB)
                    .put("maximumNativeHeapGrowthBytes", MAX_NATIVE_HEAP_GROWTH_BYTES))
                .put("qualification", if (
                    allFinite && outputStable && allDispatchQualified && memoryQualified
                ) {
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
                .put("cycleResults", cycleReports)
        } finally {
            executor.shutdownNow()
            report.put("thermalStatusEnd", thermal.currentThermalStatus)
                .put("finalMemory", memoryEvidence())
            File(resultDir, "report.json").writeText(report.toString(2))
        }
        check(report.getString("status") == "complete" &&
            report.getString("qualification") == "qualified") { report.toString() }
    }

    private fun runCycle(
        cycleIndex: Int,
        modelFile: File,
        config: MdxDspConfig,
        waveform: Array<FloatArray>,
        separated: Array<Array<FloatArray>>,
        warmups: Int,
        measuredWindows: Int,
        executor: ExecutorService,
        thermal: PowerManager,
    ): JSONObject {
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
        var runMs = 0.0
        var finite = true
        var outputHash = 0xcbf29ce484222325UL
        var peakAbsolute = 0f
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
            val runStarted = now()
            pipeline.preprocessInput(waveform, 0)
            var previous = submit(executor, pipeline, bounded, 0)
            repeat(measuredWindows) { index ->
                val completed = index and 1
                val next = (index + 1) and 1
                val current: Future<*>? = if (index + 1 < measuredWindows) {
                    pipeline.preprocessInput(waveform, next)
                    submit(executor, pipeline, bounded, next)
                } else {
                    null
                }
                previous.get()
                pipeline.postprocessOutputInto(separated[completed], completed)
                for (channel in separated[completed]) {
                    for (value in channel) {
                        finite = finite && value.isFinite()
                        peakAbsolute = maxOf(peakAbsolute, kotlin.math.abs(value))
                        outputHash = (outputHash xor value.toRawBits().toUInt().toULong()) *
                            0x100000001b3UL
                    }
                }
                if (current != null) previous = current
            }
            runMs = elapsed(runStarted)
            dispatch = bounded.evidence()
            duringRun = memoryEvidence()
        } finally {
            pipeline.close()
        }
        Runtime.getRuntime().gc()
        System.runFinalization()
        Thread.sleep(100)
        val afterClose = memoryEvidence()
        return JSONObject()
            .put("cycleIndex", cycleIndex)
            .put("thermalStatusStart", thermalStart)
            .put("thermalStatusEnd", thermal.currentThermalStatus)
            .put("setupMs", setupMs)
            .put("runMs", runMs)
            .put("meanMsPerWindow", runMs / measuredWindows)
            .put("finite", finite)
            .put("peakAbsolute", peakAbsolute.toDouble())
            .put("outputHashFnv1a64", outputHash.toString(16).padStart(16, '0'))
            .put("beforeCreate", beforeCreate)
            .put("duringRun", duringRun)
            .put("afterClose", afterClose)
            .put("artRuntimeDelta", counterDelta(countersBefore, runtimeCounters()))
            .put("boundedGpuEvidence", dispatch)
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
        private const val MAX_PSS_GROWTH_KB = 65_536
        private const val MAX_NATIVE_HEAP_GROWTH_BYTES = 16L * 1024 * 1024
        private val RUN_ID = Regex("[A-Za-z0-9._-]{1,80}")
        private val RUNTIME_COUNTERS = listOf(
            "art.gc.gc-count", "art.gc.bytes-allocated", "art.gc.bytes-freed",
            "art.gc.blocking-gc-count",
        )
    }
}
