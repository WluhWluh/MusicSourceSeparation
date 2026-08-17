package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.BuildConfig
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
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class TfcTdfDefaultLiteRt220InstrumentedTest {
    @Test
    fun runTensorBenchmark() {
        val args = InstrumentationRegistry.getArguments()
        val backend = args.getString("backend", "cpu")!!
        require(backend == "cpu" || backend == "gpu-bounded")
        val threads = args.getString("threads", "4")!!.toInt().coerceIn(1, 16)
        val warmups = args.getString("warmups", "2")!!.toInt().coerceIn(0, 20)
        val measuredRuns = args.getString("measuredRuns", "10")!!.toInt().coerceIn(1, 100)
        val runId = requireSafeName(args.getString("runId", "$backend-$threads")!!)
        val fixtureDirName = requireSafeName(args.getString("fixtureDir", FIXTURE_DIRECTORY)!!)
        val expectedManifestSha256 = requireSha256(
            requireNotNull(args.getString("manifestSha256")) { "manifestSha256 is required" },
        )

        val context = ApplicationProvider.getApplicationContext<Context>()
        val benchmarkRoot = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val fixtureRoot = File(benchmarkRoot, fixtureDirName).canonicalFile
        require(fixtureRoot.parentFile == benchmarkRoot.canonicalFile)
        val manifestFile = File(fixtureRoot, "android-fixture-manifest.json")
        require(manifestFile.isFile && sha256(manifestFile) == expectedManifestSha256)
        val manifest = JSONObject(manifestFile.readText())
        require(manifest.getInt("schemaVersion") == 1)
        require(manifest.getString("fixtureId") == "tfc_tdf_default_real_window_0@1")

        require(BuildConfig.BENCHMARK_RUNTIME_VERSION == EXPECTED_LITERT_VERSION)
        require(BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256 == EXPECTED_RUNTIME_SHA256)
        val modelIdentity = manifest.getJSONObject("model")
        val modelFile = verifiedFile(fixtureRoot, modelIdentity)
        require(modelIdentity.getString("sha256") == EXPECTED_MODEL_SHA256)
        val inputIdentity = manifest.getJSONObject("input")
        val inputFile = verifiedFile(fixtureRoot, inputIdentity)
        val goldens = manifest.getJSONObject("goldens")
        val onnxGoldenIdentity = goldens.getJSONObject("onnx")
        val hostTfliteGoldenIdentity = goldens.getJSONObject("hostTflite")
        val onnxGoldenFile = verifiedFile(fixtureRoot, onnxGoldenIdentity)
        val hostTfliteGoldenFile = verifiedFile(fixtureRoot, hostTfliteGoldenIdentity)
        requireShape(inputIdentity, NCHW_SHAPE)
        requireShape(onnxGoldenIdentity, NCHW_SHAPE)
        requireShape(hostTfliteGoldenIdentity, NCHW_SHAPE)

        val inputNchw = readFloats(inputFile, ELEMENT_COUNT)
        val inputNhwc = nchwToNhwc(inputNchw)
        val onnxGolden = readFloats(onnxGoldenFile, ELEMENT_COUNT)
        val hostTfliteGolden = readFloats(hostTfliteGoldenFile, ELEMENT_COUNT)
        val resultDir = File(fixtureRoot, "results/$runId").apply {
            deleteRecursively()
            mkdirs()
        }
        val powerManager = context.getSystemService(PowerManager::class.java)
        val report = baseReport(
            backend = backend,
            threads = threads,
            warmups = warmups,
            measuredRuns = measuredRuns,
            runId = runId,
            manifestFile = manifestFile,
            manifest = manifest,
            modelFile = modelFile,
            inputFile = inputFile,
            context = context,
            powerManager = powerManager,
        )
        var environment: Environment? = null
        var compiledModel: CompiledModel? = null
        var inputBuffer: TensorBuffer? = null
        var outputBuffer: TensorBuffer? = null
        var boundedRuntime: BoundedGpuRuntime? = null
        try {
            val setupStarted = timedStart()
            boundedRuntime = if (backend == "gpu-bounded") {
                BoundedGpuRuntime.loadAndValidate()
            } else {
                null
            }
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
                    cpuOptions = CompiledModel.CpuOptions(
                        numThreads = threads,
                        xnnPackFlags = null,
                        xnnPackWeightCachePath = null,
                    )
                }
            }
            val activeEnvironment = Environment.create()
            environment = activeEnvironment
            val availableAccelerators = activeEnvironment.getAvailableAccelerators()
                .map { it.name }
                .sorted()
            require(if (backend == "gpu-bounded") "GPU" in availableAccelerators else "CPU" in availableAccelerators)
            val activeModel = CompiledModel.create(modelFile.absolutePath, options, activeEnvironment)
            compiledModel = activeModel
            val inputs = activeModel.createInputBuffers()
            val outputs = activeModel.createOutputBuffers()
            require(inputs.size == 1 && outputs.size == 1)
            val activeInput = inputs.single()
            val activeOutput = outputs.single()
            inputBuffer = activeInput
            outputBuffer = activeOutput
            val inputShape = requireNotNull(
                activeModel.getInputTensorType(modelIdentity.getString("inputName")).layout,
            ).dimensions
            val outputShape = requireNotNull(
                activeModel.getOutputTensorType(modelIdentity.getString("outputName")).layout,
            ).dimensions
            require(inputShape == NHWC_SHAPE && outputShape == NHWC_SHAPE) {
                "Unexpected runtime shapes input=$inputShape output=$outputShape"
            }
            activeInput.writeFloat(inputNhwc)
            val setup = setupStarted.elapsed()

            val warmupWallMs = mutableListOf<Double>()
            repeat(warmups) {
                val started = timedStart()
                runModel(activeModel, activeInput, activeOutput, boundedRuntime)
                activeOutput.readFloat()
                warmupWallMs += started.elapsed().wallMs
            }

            boundedRuntime?.resetInferenceCounters()
            val invokeWallMs = mutableListOf<Double>()
            val invokeCpuMs = mutableListOf<Long>()
            val readWallMs = mutableListOf<Double>()
            val totalWallMs = mutableListOf<Double>()
            val thermalSamples = JSONArray()
            var finalOutputNhwc: FloatArray? = null
            repeat(measuredRuns) { index ->
                val totalStarted = timedStart()
                val invokeStarted = timedStart()
                runModel(activeModel, activeInput, activeOutput, boundedRuntime)
                invokeStarted.elapsed().also { timing ->
                    invokeWallMs += timing.wallMs
                    invokeCpuMs += timing.cpuMs
                }
                val readStarted = timedStart()
                finalOutputNhwc = activeOutput.readFloat()
                readWallMs += readStarted.elapsed().wallMs
                totalWallMs += totalStarted.elapsed().wallMs
                thermalSamples.put(
                    JSONObject()
                        .put("run", index + 1)
                        .put("thermalStatus", powerManager.currentThermalStatus)
                        .put("batteryTemperatureDeciC", batteryTemperatureDeciC(context)),
                )
            }
            val outputNchw = nhwcToNchw(requireNotNull(finalOutputNhwc))
            val outputFile = File(resultDir, "device-output-nchw-f32.bin")
            writeFloats(outputFile, outputNchw)
            val comparisonToOnnx = compare(onnxGolden, outputNchw)
            val comparisonToHostTflite = compare(hostTfliteGolden, outputNchw)
            require(comparisonPassed(comparisonToOnnx)) {
                "Device output failed ONNX parity: $comparisonToOnnx"
            }
            require(comparisonPassed(comparisonToHostTflite)) {
                "Device output failed host TFLite parity: $comparisonToHostTflite"
            }
            val boundedEvidence = boundedRuntime?.evidence()
            if (boundedEvidence != null) {
                val dispatchCount = boundedEvidence.getLong("dispatchCount")
                val waitCount = boundedEvidence.getLong("eventWaitCount")
                require(dispatchCount > 0L && dispatchCount == waitCount) {
                    "GPU execution was not verified: $boundedEvidence"
                }
            }
            report
                .put("status", "complete")
                .put("setup", timingJson(setup))
                .put("availableAccelerators", JSONArray(availableAccelerators))
                .put("runtimeInputShape", JSONArray(inputShape))
                .put("runtimeOutputShape", JSONArray(outputShape))
                .put("warmupWallMs", JSONArray(warmupWallMs))
                .put("invokeWallMs", JSONArray(invokeWallMs))
                .put("invokeCpuMs", JSONArray(invokeCpuMs))
                .put("outputReadWallMs", JSONArray(readWallMs))
                .put("totalWallMs", JSONArray(totalWallMs))
                .put("invokeSummary", timingSummary(invokeWallMs))
                .put("totalSummary", timingSummary(totalWallMs))
                .put("thermalSamples", thermalSamples)
                .put("thermalStatusEnd", powerManager.currentThermalStatus)
                .put("batteryTemperatureDeciCEnd", batteryTemperatureDeciC(context))
                .put("memoryBeforeClose", memoryEvidence())
                .put("output", fileIdentity(outputFile).put("finite", outputNchw.all { it.isFinite() }))
                .put("comparisonToOnnx", comparisonToOnnx)
                .put("comparisonToHostTflite", comparisonToHostTflite)
                .put("gate", JSONObject().put("minimumSnrDb", 90.0).put("maximumAbsoluteError", 1e-4))
                .put("boundedGpuEvidence", boundedEvidence ?: JSONObject.NULL)
        } catch (error: Throwable) {
            report
                .put("status", "error")
                .put("errorClass", error::class.java.name)
                .put("message", error.message.orEmpty())
                .put("stack", error.stackTraceToString())
                .put("thermalStatusEnd", powerManager.currentThermalStatus)
                .put("batteryTemperatureDeciCEnd", batteryTemperatureDeciC(context))
                .put("memoryBeforeClose", memoryEvidence())
            File(resultDir, "report.json").writeText(report.toString(2))
            throw error
        } finally {
            inputBuffer?.close()
            outputBuffer?.close()
            compiledModel?.close()
            environment?.close()
            report.put("memoryAfterClose", memoryEvidence())
            File(resultDir, "report.json").writeText(report.toString(2))
        }
        println(report)
    }

    private fun baseReport(
        backend: String,
        threads: Int,
        warmups: Int,
        measuredRuns: Int,
        runId: String,
        manifestFile: File,
        manifest: JSONObject,
        modelFile: File,
        inputFile: File,
        context: Context,
        powerManager: PowerManager,
    ): JSONObject = JSONObject()
        .put("schemaVersion", 1)
        .put("status", "running")
        .put("runId", runId)
        .put("candidateId", manifest.getString("candidateId"))
        .put("fixtureId", manifest.getString("fixtureId"))
        .put("manifest", fileIdentity(manifestFile))
        .put("model", fileIdentity(modelFile))
        .put("input", fileIdentity(inputFile))
        .put("backend", backend)
        .put("profile", if (backend == "gpu-bounded") GPU_PROFILE else "cpu-xnnpack")
        .put("threads", threads)
        .put("warmups", warmups)
        .put("measuredRuns", measuredRuns)
        .put("runtime", JSONObject()
            .put("id", BuildConfig.BENCHMARK_RUNTIME_ID)
            .put("version", BuildConfig.BENCHMARK_RUNTIME_VERSION)
            .put("artifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256))
        .put("source", JSONObject()
            .put("revision", BuildConfig.BENCHMARK_SOURCE_REVISION)
            .put("dirty", BuildConfig.BENCHMARK_SOURCE_DIRTY))
        .put("device", deviceIdentity())
        .put("thermalStatusStart", powerManager.currentThermalStatus)
        .put("batteryTemperatureDeciCStart", batteryTemperatureDeciC(context))
        .put("memoryStart", memoryEvidence())

    private fun runModel(
        model: CompiledModel,
        input: TensorBuffer,
        output: TensorBuffer,
        boundedRuntime: BoundedGpuRuntime?,
    ) {
        boundedRuntime?.beginInference()
        try {
            model.run(listOf(input), listOf(output))
        } finally {
            boundedRuntime?.endInference()
        }
    }

    private fun nchwToNhwc(input: FloatArray): FloatArray = FloatArray(input.size).also { output ->
        for (channel in 0 until CHANNELS) {
            for (frequency in 0 until FREQUENCIES) {
                for (frame in 0 until FRAMES) {
                    output[(frequency * FRAMES + frame) * CHANNELS + channel] =
                        input[(channel * FREQUENCIES + frequency) * FRAMES + frame]
                }
            }
        }
    }

    private fun nhwcToNchw(input: FloatArray): FloatArray = FloatArray(input.size).also { output ->
        for (channel in 0 until CHANNELS) {
            for (frequency in 0 until FREQUENCIES) {
                for (frame in 0 until FRAMES) {
                    output[(channel * FREQUENCIES + frequency) * FRAMES + frame] =
                        input[(frequency * FRAMES + frame) * CHANNELS + channel]
                }
            }
        }
    }

    private fun compare(reference: FloatArray, candidate: FloatArray): JSONObject {
        require(reference.size == candidate.size)
        var maxAbs = 0.0
        var absSum = 0.0
        var errorPower = 0.0
        var referencePower = 0.0
        var candidatePower = 0.0
        var dot = 0.0
        var finite = true
        for (index in reference.indices) {
            val expected = reference[index].toDouble()
            val actual = candidate[index].toDouble()
            finite = finite && actual.isFinite()
            val error = actual - expected
            val absolute = kotlin.math.abs(error)
            maxAbs = maxOf(maxAbs, absolute)
            absSum += absolute
            errorPower += error * error
            referencePower += expected * expected
            candidatePower += actual * actual
            dot += expected * actual
        }
        val bitExact = errorPower == 0.0
        val snr = if (bitExact) 300.0 else {
            10.0 * ln(referencePower / errorPower) / ln(10.0)
        }
        val cosine = dot / sqrt(referencePower * candidatePower)
        return JSONObject()
            .put("finite", finite)
            .put("elements", reference.size)
            .put("maxAbsError", maxAbs)
            .put("meanAbsError", absSum / reference.size)
            .put("rmse", sqrt(errorPower / reference.size))
            .put("signalRms", sqrt(referencePower / reference.size))
            .put("snrDb", snr)
            .put("cosineSimilarity", cosine)
            .put("bitExact", bitExact)
    }

    private fun comparisonPassed(comparison: JSONObject): Boolean =
        comparison.getBoolean("finite") &&
            comparison.getDouble("snrDb") >= 90.0 &&
            comparison.getDouble("maxAbsError") <= 1e-4

    private fun timingSummary(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        fun percentile(fraction: Double): Double {
            val index = (ceil(fraction * sorted.size).toInt() - 1).coerceIn(sorted.indices)
            return sorted[index]
        }
        return JSONObject()
            .put("count", sorted.size)
            .put("minimumMs", sorted.first())
            .put("medianMs", percentile(0.5))
            .put("p95Ms", percentile(0.95))
            .put("maximumMs", sorted.last())
            .put("meanMs", sorted.average())
    }

    private fun requireShape(identity: JSONObject, expected: List<Int>) {
        val actual = identity.getJSONArray("shape")
        require(actual.length() == expected.size && expected.indices.all { actual.getInt(it) == expected[it] })
    }

    private fun verifiedFile(root: File, identity: JSONObject): File {
        val name = requireSafeName(identity.getString("file"))
        val file = File(root, name).canonicalFile
        require(file.parentFile == root.canonicalFile)
        require(file.isFile && file.length() == identity.getLong("bytes"))
        require(sha256(file) == requireSha256(identity.getString("sha256")))
        return file
    }

    private fun readFloats(file: File, elements: Int): FloatArray {
        require(file.length() == elements * Float.SIZE_BYTES.toLong())
        val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(elements).also(buffer::get)
    }

    private fun writeFloats(file: File, values: FloatArray) {
        val buffer = ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buffer.putFloat(it) }
        file.writeBytes(buffer.array())
    }

    private fun timedStart(): TimedStart = TimedStart(
        elapsedNanos = SystemClock.elapsedRealtimeNanos(),
        cpuMs = Process.getElapsedCpuTime(),
    )

    private fun timingJson(timing: Timing): JSONObject = JSONObject()
        .put("wallMs", timing.wallMs)
        .put("cpuMs", timing.cpuMs)

    private fun deviceIdentity(): JSONObject = JSONObject()
        .put("manufacturer", Build.MANUFACTURER)
        .put("model", Build.MODEL)
        .put("device", Build.DEVICE)
        .put("socManufacturer", Build.SOC_MANUFACTURER)
        .put("socModel", Build.SOC_MODEL)
        .put("sdk", Build.VERSION.SDK_INT)
        .put("release", Build.VERSION.RELEASE)
        .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))

    private fun memoryEvidence(): JSONObject {
        val info = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        return JSONObject()
            .put("pssKb", info.totalPss)
            .put("privateDirtyKb", info.totalPrivateDirty)
            .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun batteryTemperatureDeciC(context: Context): Int? = context.registerReceiver(
        null,
        IntentFilter(Intent.ACTION_BATTERY_CHANGED),
    )?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        ?.takeUnless { it == Int.MIN_VALUE }

    private fun fileIdentity(file: File): JSONObject = JSONObject()
        .put("file", file.name)
        .put("bytes", file.length())
        .put("sha256", sha256(file))

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
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun requireSafeName(value: String): String = value.also {
        require(it.matches(Regex("[A-Za-z0-9._-]+")) && it != "." && it != "..")
    }

    private fun requireSha256(value: String): String = value.also {
        require(it.matches(Regex("[0-9a-f]{64}")))
    }

    private data class TimedStart(val elapsedNanos: Long, val cpuMs: Long) {
        fun elapsed(): Timing = Timing(
            wallMs = (SystemClock.elapsedRealtimeNanos() - elapsedNanos) / 1_000_000.0,
            cpuMs = Process.getElapsedCpuTime() - cpuMs,
        )
    }

    private data class Timing(val wallMs: Double, val cpuMs: Long)

    private class BoundedGpuRuntime private constructor(
        private val runtimeClass: Class<*>,
        private val capability: Any,
    ) {
        fun resetInferenceCounters() = invoke("resetInferenceCounters")
        fun beginInference() = invoke("beginInference")
        fun endInference() = invoke("endInference")
        fun evidence(): JSONObject = JSONObject()
            .put("artifactVersion", value("getArtifactVersion"))
            .put("profileId", value("getProfileId"))
            .put("schemaVersion", value("getSchemaVersion"))
            .put("kernelBatchSize", value("getKernelBatchSize"))
            .put("commandQueueWindowSize", value("getCommandQueueWindowSize"))
            .put("dispatchCount", invoke("getDispatchCount"))
            .put("eventWaitCount", invoke("getEventWaitCount"))

        private fun invoke(name: String): Any? = runtimeClass.getMethod(name).invoke(null)
        private fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)

        companion object {
            fun loadAndValidate(): BoundedGpuRuntime {
                val runtimeClass = Class.forName(RUNTIME_CLASS)
                val capability = requireNotNull(runtimeClass.getMethod("queryCapability").invoke(null))
                fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)
                require(value("isAvailable") == true)
                require(value("getArtifactVersion") == EXPECTED_RUNTIME_ARTIFACT)
                require(value("getProfileId") == GPU_PROFILE)
                require(value("getKernelBatchSize") == 1)
                require(value("getCommandQueueWindowSize") == 1)
                return BoundedGpuRuntime(runtimeClass, capability)
            }
        }
    }

    companion object {
        private const val FIXTURE_DIRECTORY = "tfc-tdf-default"
        private const val EXPECTED_LITERT_VERSION = "2.2.0"
        private const val EXPECTED_RUNTIME_ARTIFACT = "2.2.0-bss.2"
        private const val EXPECTED_RUNTIME_SHA256 =
            "35b55a0ef9a6d28e56271a9bc3b6b6cc8a84b16732b17b34b2a6b51ee7be3124"
        private const val EXPECTED_MODEL_SHA256 =
            "0ee7bbc0bd5a1194745ebf4df1753ba6ef32a256cbb55fca8098a43912591f5e"
        private const val GPU_PROFILE = "gpu-opencl-bounded-fp32-v1"
        private const val RUNTIME_CLASS = "io.github.wluhwluh.bss.litert.BssLiteRtRuntime"
        private const val CHANNELS = 4
        private const val FREQUENCIES = 1025
        private const val FRAMES = 128
        private const val ELEMENT_COUNT = CHANNELS * FREQUENCIES * FRAMES
        private val NCHW_SHAPE = listOf(1, CHANNELS, FREQUENCIES, FRAMES)
        private val NHWC_SHAPE = listOf(1, FREQUENCIES, FRAMES, CHANNELS)
    }
}
