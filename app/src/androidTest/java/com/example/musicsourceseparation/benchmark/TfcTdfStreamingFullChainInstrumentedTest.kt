package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.BuildConfig
import com.example.musicsourceseparation.streaming.MediaCodecStreamingAudioReader
import com.example.musicsourceseparation.streaming.NonCausalStreamingSeparatedPlaybackEngine
import com.example.musicsourceseparation.streaming.StreamingAccelerator
import com.example.musicsourceseparation.streaming.StreamingInferenceSession
import com.example.musicsourceseparation.streaming.StreamingInferenceSessionFactory
import com.example.musicsourceseparation.streaming.StreamingModelConfig
import com.example.musicsourceseparation.streaming.StreamingPlaybackMode
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
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

@RunWith(AndroidJUnit4::class)
class TfcTdfStreamingFullChainInstrumentedTest {
    @Test
    fun runFullChainBenchmark() {
        val args = InstrumentationRegistry.getArguments()
        val backend = args.getString("backend", "cpu")!!
        require(backend == "cpu" || backend == "gpu-bounded")
        val threads = args.getString("threads", "4")!!.toInt().coerceIn(1, 16)
        val runId = safeName(args.getString("runId", backend)!!)
        val sourceName = safeName(requireNotNull(args.getString("sourceFile")))
        val modelName = safeName(requireNotNull(args.getString("modelFile")))
        val sourceSha = args.getString("sourceSha256")?.let(::sha256Value)
        val modelSha = args.getString("modelSha256")?.let(::sha256Value)
        val playbackSeconds = args.getString("playbackSeconds", "30")!!.toDouble()
            .coerceIn(5.0, 120.0)
        val seekAtSeconds = args.getString("seekAtSeconds", "12")!!.toDouble()
            .coerceIn(0.0, playbackSeconds - 1.0)
        val seekToSeconds = args.getString("seekToSeconds", "45")!!.toDouble()
            .coerceAtLeast(0.0)
        val blockSamples = args.getString("blockSamples", "1024")!!.toInt()
            .coerceIn(256, 4096)

        val context = ApplicationProvider.getApplicationContext<Context>()
        val benchmarkRoot = File(
            requireNotNull(context.getExternalFilesDir(null)),
            "benchmark/tfc-tdf-streaming",
        ).canonicalFile
        val sourceFile = File(benchmarkRoot, sourceName).canonicalFile
        val modelFile = File(benchmarkRoot, modelName).canonicalFile
        require(sourceFile.parentFile == benchmarkRoot && modelFile.parentFile == benchmarkRoot)
        require(sourceFile.isFile && modelFile.isFile)
        sourceSha?.let { require(sha256(sourceFile) == it) }
        modelSha?.let { require(sha256(modelFile) == it) }

        val resultDir = File(benchmarkRoot, "results/$runId").apply {
            deleteRecursively()
            mkdirs()
        }
        val powerManager = context.getSystemService(PowerManager::class.java)
        val report = JSONObject()
            .put("schemaVersion", 1)
            .put("status", "running")
            .put("runId", runId)
            .put("backend", backend)
            .put("threads", threads)
            .put("model", fileIdentity(modelFile))
            .put("source", fileIdentity(sourceFile))
            .put("runtime", JSONObject()
                .put("id", BuildConfig.BENCHMARK_RUNTIME_ID)
                .put("version", BuildConfig.BENCHMARK_RUNTIME_VERSION)
                .put("artifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256))
            .put("device", deviceIdentity())
            .put("thermalStatusStart", powerManager.currentThermalStatus)
            .put("batteryTemperatureDeciCStart", batteryTemperatureDeciC(context))
            .put("memoryStart", memoryEvidence())
            .put("gcStart", gcEvidence())

        var playbackReader: MediaCodecStreamingAudioReader? = null
        var analysisReader: MediaCodecStreamingAudioReader? = null
        var engine: NonCausalStreamingSeparatedPlaybackEngine? = null
        var factory: MeasuredLiteRtSessionFactory? = null
        val processCpuStart = Process.getElapsedCpuTime()
        val wallStart = SystemClock.elapsedRealtimeNanos()
        try {
            playbackReader = MediaCodecStreamingAudioReader(context, android.net.Uri.fromFile(sourceFile))
            analysisReader = MediaCodecStreamingAudioReader(context, android.net.Uri.fromFile(sourceFile))
            require(playbackReader.frameCount == analysisReader.frameCount)
            factory = MeasuredLiteRtSessionFactory(
                modelFile = modelFile,
                threads = threads,
                backend = backend,
            )
            engine = NonCausalStreamingSeparatedPlaybackEngine(
                reader = analysisReader,
                sessionFactory = factory,
                maxInputWindows = 3,
                maxWetWindows = 4,
            )
            engine.start(
                startSample = 0,
                model = StreamingModelConfig("tfc-tdf-default-v1"),
                accelerator = if (backend == "gpu-bounded") {
                    StreamingAccelerator.GPU
                } else {
                    StreamingAccelerator.CPU
                },
            )

            val trackFrames = playbackReader.frameCount
            val requestedFrames = (playbackSeconds * TfcTdfStreamingDsp.SAMPLE_RATE).toLong()
            val seekAtFrame = (seekAtSeconds * TfcTdfStreamingDsp.SAMPLE_RATE).toLong()
            val seekTargetFrame = (seekToSeconds * TfcTdfStreamingDsp.SAMPLE_RATE).toLong()
            require(seekTargetFrame < trackFrames) {
                "seekToSeconds exceeds source duration: $seekToSeconds"
            }
            val maxAfterSeek = trackFrames - seekTargetFrame
            val logicalFrames = min(
                requestedFrames,
                seekAtFrame + maxAfterSeek,
            )
            val dryBlocks = AtomicLong(0)
            val wetBlocks = AtomicLong(0)
            val selectionWallNanos = AtomicLong(0)
            val decodeWallNanos = AtomicLong(0)
            val leadSamples = ArrayList<Long>()
            val resourceSamples = JSONArray()
            val outputDigest = MessageDigest.getInstance("SHA-256")
            var sourcePosition = 0L
            var logicalPosition = 0L
            var firstWetWallMs: Double? = null
            var firstWetReadyWallMs: Double? = null
            var seekRequestWallMs: Double? = null
            var seekWetWallMs: Double? = null
            var seekDone = false
            var lastResourceNanos = 0L

            while (logicalPosition < logicalFrames) {
                if (!seekDone && logicalPosition >= seekAtFrame) {
                    val seekStarted = SystemClock.elapsedRealtimeNanos()
                    engine.seek(seekTargetFrame)
                    sourcePosition = seekTargetFrame
                    seekRequestWallMs = elapsedMs(wallStart, seekStarted)
                    seekDone = true
                }
                val frameCount = min(blockSamples.toLong(), logicalFrames - logicalPosition).toInt()
                val decodeStarted = SystemClock.elapsedRealtimeNanos()
                val dry = playbackReader.read(sourcePosition, frameCount)
                decodeWallNanos.addAndGet(SystemClock.elapsedRealtimeNanos() - decodeStarted)
                val output = FloatArray(dry.size)
                val selectStarted = SystemClock.elapsedRealtimeNanos()
                val mode = engine.selectBlock(sourcePosition, dry, output)
                selectionWallNanos.addAndGet(SystemClock.elapsedRealtimeNanos() - selectStarted)
                require(output.all { it.isFinite() }) {
                    "Streaming output contained a non-finite sample at $sourcePosition"
                }
                if (mode == StreamingPlaybackMode.WET) {
                    wetBlocks.incrementAndGet()
                    if (firstWetWallMs == null) {
                        firstWetWallMs = elapsedMs(wallStart, SystemClock.elapsedRealtimeNanos())
                    }
                    if (seekDone && seekWetWallMs == null) {
                        seekWetWallMs = elapsedMs(wallStart, SystemClock.elapsedRealtimeNanos())
                    }
                } else {
                    dryBlocks.incrementAndGet()
                }
                val snapshot = engine.snapshot()
                if (snapshot.publishedWindowCount > 0 && firstWetReadyWallMs == null) {
                    firstWetReadyWallMs = elapsedMs(wallStart, SystemClock.elapsedRealtimeNanos())
                }
                leadSamples += snapshot.wetLeadSamples
                updateDigest(outputDigest, output)
                sourcePosition += frameCount
                logicalPosition += frameCount

                val now = SystemClock.elapsedRealtimeNanos()
                if (now - lastResourceNanos >= 500_000_000L) {
                    resourceSamples.put(
                        JSONObject()
                            .put("logicalPositionSamples", logicalPosition)
                            .put("elapsedWallMs", elapsedMs(wallStart, now))
                            .put("thermalStatus", powerManager.currentThermalStatus)
                            .put("batteryTemperatureDeciC", batteryTemperatureDeciC(context))
                            .put("memory", memoryEvidence())
                            .put("processCpuMs", Process.getElapsedCpuTime() - processCpuStart),
                    )
                    lastResourceNanos = now
                }
                val targetNanos = wallStart + logicalPosition * 1_000_000_000L /
                    TfcTdfStreamingDsp.SAMPLE_RATE
                val sleepNanos = targetNanos - SystemClock.elapsedRealtimeNanos()
                if (sleepNanos > 0) {
                    Thread.sleep(sleepNanos / 1_000_000L, (sleepNanos % 1_000_000L).toInt())
                }
            }

            val finalSnapshot = engine.snapshot()
            engine.close()
            engine = null
            playbackReader.close()
            playbackReader = null
            val factoryReport = requireNotNull(factory).report()
            val elapsedWallMs = elapsedMs(wallStart, SystemClock.elapsedRealtimeNanos())
            val audioSeconds = logicalFrames.toDouble() / TfcTdfStreamingDsp.SAMPLE_RATE
            val fullChainWallMs = factoryReport.getDouble("fullChainWallMs")
            val positiveLeadSamples = leadSamples.filter { it > 0L }
            report
                .put("status", "complete")
                .put("playback", JSONObject()
                    .put("requestedSeconds", playbackSeconds)
                    .put("processedSeconds", audioSeconds)
                    .put("blockSamples", blockSamples)
                    .put("dryBlockCount", dryBlocks.get())
                    .put("wetBlockCount", wetBlocks.get())
                    .put("firstWetBlockWallMs", firstWetWallMs ?: JSONObject.NULL)
                    .put("firstWetReadyWallMs", firstWetReadyWallMs ?: JSONObject.NULL)
                    .put("seekRequestWallMs", seekRequestWallMs ?: JSONObject.NULL)
                    .put("seekWetBlockWallMs", seekWetWallMs ?: JSONObject.NULL)
                    .put("seekToSeconds", seekToSeconds)
                    .put("seekRewetLatencyMs", if (seekRequestWallMs != null && seekWetWallMs != null) {
                        seekWetWallMs!! - seekRequestWallMs!!
                    } else JSONObject.NULL)
                    .put("wallElapsedMs", elapsedWallMs)
                    .put("wallRtf", elapsedWallMs / 1_000.0 / audioSeconds)
                    .put("outputSha256", outputDigest.digest().joinToString("") { "%02x".format(it) }))
                .put("timing", JSONObject()
                    .put("playbackDecodeWallMs", decodeWallNanos.get() / 1_000_000.0)
                    .put("blockSelectionWallMs", selectionWallNanos.get() / 1_000_000.0)
                    .put("engine", factoryReport)
                    .put("fullChainRtf", fullChainWallMs / 1_000.0 / audioSeconds))
                .put("wetLead", JSONObject()
                    .put("samples", JSONArray(leadSamples))
                    .put("minimumSamples", leadSamples.minOrNull() ?: 0L)
                    .put("medianSamples", percentile(leadSamples, 0.5))
                    .put("maximumSamples", leadSamples.maxOrNull() ?: 0L)
                    .put("positiveMinimumSamples", positiveLeadSamples.minOrNull() ?: 0L)
                    .put("positiveMedianSamples", percentile(positiveLeadSamples, 0.5))
                    .put("positiveMaximumSamples", positiveLeadSamples.maxOrNull() ?: 0L)
                    .put("positiveMinimumMs", (positiveLeadSamples.minOrNull() ?: 0L) * 1000.0 / TfcTdfStreamingDsp.SAMPLE_RATE)
                    .put("positiveMedianMs", percentile(positiveLeadSamples, 0.5) * 1000.0 / TfcTdfStreamingDsp.SAMPLE_RATE)
                    .put("positiveMaximumMs", (positiveLeadSamples.maxOrNull() ?: 0L) * 1000.0 / TfcTdfStreamingDsp.SAMPLE_RATE)
                    .put("minimumMs", (leadSamples.minOrNull() ?: 0L) * 1000.0 / TfcTdfStreamingDsp.SAMPLE_RATE)
                    .put("maximumMs", (leadSamples.maxOrNull() ?: 0L) * 1000.0 / TfcTdfStreamingDsp.SAMPLE_RATE))
                .put("engineFinal", JSONObject()
                    .put("epoch", finalSnapshot.epoch)
                    .put("lateWindowCount", finalSnapshot.lateWindowCount)
                    .put("discardedEpochOutputCount", finalSnapshot.discardedEpochOutputCount)
                    .put("publishedWindowCount", finalSnapshot.publishedWindowCount)
                    .put("readCallCount", finalSnapshot.readCallCount)
                    .put("readFrameCount", finalSnapshot.readFrameCount)
                    .put("maxInputRingSamples", finalSnapshot.maxInputRingSamples)
                    .put("wetWindowCount", finalSnapshot.wetWindowCount)
                    .put("readAheadDecodeWallMs", finalSnapshot.readAheadDecodeWallNanos / 1_000_000.0))
                .put("resources", resourceSamples)
                .put("thermalStatusEnd", powerManager.currentThermalStatus)
                .put("batteryTemperatureDeciCEnd", batteryTemperatureDeciC(context))
                .put("memoryEnd", memoryEvidence())
                .put("gcEnd", gcEvidence())
                .put("processCpuMs", Process.getElapsedCpuTime() - processCpuStart)
            File(resultDir, "report.json").writeText(report.toString(2))
        } catch (error: Throwable) {
            report
                .put("status", "error")
                .put("errorClass", error::class.java.name)
                .put("message", error.message.orEmpty())
                .put("stack", error.stackTraceToString())
                .put("thermalStatusEnd", powerManager.currentThermalStatus)
                .put("batteryTemperatureDeciCEnd", batteryTemperatureDeciC(context))
                .put("memoryEnd", memoryEvidence())
            File(resultDir, "report.json").writeText(report.toString(2))
            throw error
        } finally {
            engine?.close()
            playbackReader?.close()
            analysisReader?.close()
        }
        println(report)
    }

    private class MeasuredLiteRtSessionFactory(
        private val modelFile: File,
        private val threads: Int,
        private val backend: String,
    ) : StreamingInferenceSessionFactory {
        private val lock = Any()
        private val sessions = ArrayList<MeasuredLiteRtSession>()
        private val openCount = AtomicInteger(0)
        private val processCount = AtomicInteger(0)

        override fun open(
            model: StreamingModelConfig,
            accelerator: StreamingAccelerator,
        ): StreamingInferenceSession {
            val session = MeasuredLiteRtSession(
                modelFile = modelFile,
                model = model,
                threads = threads,
                accelerator = accelerator,
                boundedGpu = backend == "gpu-bounded",
                processCount = processCount,
            )
            synchronized(lock) { sessions += session }
            openCount.incrementAndGet()
            return session
        }

        fun report(): JSONObject {
            val result = JSONObject()
                .put("sessionOpenCount", openCount.get())
                .put("processedWindowCount", processCount.get())
            val reports = JSONArray()
            var setupWallMs = 0.0
            var computeWallMs = 0.0
            var stftWallMs = 0.0
            var inferenceWallMs = 0.0
            var istftWallMs = 0.0
            var workerCpuMs = 0.0
            var workerThreadCpuMs = 0.0
            var setupThreadCpuMs = 0.0
            synchronized(lock) {
                sessions.forEach {
                    val report = it.report()
                    reports.put(report)
                    setupWallMs += report.optDouble("setupWallMs", 0.0)
                    computeWallMs += report.optDouble("totalWallMs", 0.0)
                    stftWallMs += report.optDouble("stftWallMs", 0.0)
                    inferenceWallMs += report.optDouble("inferenceWallMs", 0.0)
                    istftWallMs += report.optDouble("istftWallMs", 0.0)
                    workerCpuMs += report.optDouble("workerCpuMs", 0.0)
                    workerThreadCpuMs += report.optDouble("workerThreadCpuMs", 0.0)
                    setupThreadCpuMs += report.optDouble("setupThreadCpuMs", 0.0)
                }
            }
            return result
                .put("sessions", reports)
                .put("setupWallMs", setupWallMs)
                .put("computeWallMs", computeWallMs)
                .put("fullChainWallMs", setupWallMs + computeWallMs)
                .put("stftWallMs", stftWallMs)
                .put("inferenceWallMs", inferenceWallMs)
                .put("istftWallMs", istftWallMs)
                .put("workerCpuMs", workerCpuMs)
                .put("workerThreadCpuMs", workerThreadCpuMs)
                .put("setupThreadCpuMs", setupThreadCpuMs)
        }
    }

    private class MeasuredLiteRtSession(
        modelFile: File,
        private val model: StreamingModelConfig,
        threads: Int,
        accelerator: StreamingAccelerator,
        boundedGpu: Boolean,
        private val processCount: AtomicInteger,
    ) : StreamingInferenceSession {
        private val setupStart = SystemClock.elapsedRealtimeNanos()
        private val setupCpuStart = Process.getElapsedCpuTime()
        private val setupThreadCpuStart = Debug.threadCpuTimeNanos()
        private val environment: Environment
        private val compiledModel: CompiledModel
        private val inputBuffer: TensorBuffer
        private val outputBuffer: TensorBuffer
        private val dsp = TfcTdfStreamingDsp()
        private val gpuRuntime = if (boundedGpu && accelerator == StreamingAccelerator.GPU) {
            BoundedGpuRuntime.loadAndValidate()
        } else null
        private val windows = AtomicInteger(0)
        private val stftNanos = AtomicLong(0)
        private val inferenceNanos = AtomicLong(0)
        private val istftNanos = AtomicLong(0)
        private val totalNanos = AtomicLong(0)
        private val workerCpuMillis = AtomicLong(0)
        private val workerThreadCpuNanos = AtomicLong(0)
        private val setupWallMs: Double
        private val setupCpuMs: Long
        private val setupThreadCpuMs: Double
        private var closed = false

        init {
            val options = if (accelerator == StreamingAccelerator.GPU) {
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
            environment = Environment.create()
            val accelerators = environment.getAvailableAccelerators().map { it.name }
            require(if (accelerator == StreamingAccelerator.GPU) "GPU" in accelerators else "CPU" in accelerators)
            compiledModel = CompiledModel.create(modelFile.absolutePath, options, environment)
            val inputs = compiledModel.createInputBuffers()
            val outputs = compiledModel.createOutputBuffers()
            require(inputs.size == 1 && outputs.size == 1)
            inputBuffer = inputs.single()
            outputBuffer = outputs.single()
            setupWallMs = (SystemClock.elapsedRealtimeNanos() - setupStart) / 1_000_000.0
            setupCpuMs = Process.getElapsedCpuTime() - setupCpuStart
            setupThreadCpuMs = threadCpuElapsedMs(setupThreadCpuStart)
        }

        override fun process(inputPcm: FloatArray, actualSamples: Int): FloatArray {
            check(!closed)
            val totalStart = SystemClock.elapsedRealtimeNanos()
            val cpuStart = Process.getElapsedCpuTime()
            val threadCpuStart = Debug.threadCpuTimeNanos()
            val stftStart = SystemClock.elapsedRealtimeNanos()
            val tensor = dsp.stftNhwc(inputPcm)
            stftNanos.addAndGet(SystemClock.elapsedRealtimeNanos() - stftStart)
            inputBuffer.writeFloat(tensor)
            val inferenceStart = SystemClock.elapsedRealtimeNanos()
            gpuRuntime?.beginInference()
            try {
                compiledModel.run(listOf(inputBuffer), listOf(outputBuffer))
            } finally {
                gpuRuntime?.endInference()
            }
            inferenceNanos.addAndGet(SystemClock.elapsedRealtimeNanos() - inferenceStart)
            val outputTensor = outputBuffer.readFloat()
            val istftStart = SystemClock.elapsedRealtimeNanos()
            val reconstructed = dsp.istftInterleaved(outputTensor)
            istftNanos.addAndGet(SystemClock.elapsedRealtimeNanos() - istftStart)
            val valid = FloatArray(actualSamples * TfcTdfStreamingDsp.CHANNELS)
            val sourceOffset = model.trimSamples * TfcTdfStreamingDsp.CHANNELS
            for (index in valid.indices) {
                valid[index] = inputPcm[sourceOffset + index] -
                    reconstructed[sourceOffset + index]
            }
            totalNanos.addAndGet(SystemClock.elapsedRealtimeNanos() - totalStart)
            workerCpuMillis.addAndGet(Process.getElapsedCpuTime() - cpuStart)
            workerThreadCpuNanos.addAndGet(threadCpuElapsedNanos(threadCpuStart))
            windows.incrementAndGet()
            processCount.incrementAndGet()
            return valid
        }

        fun report(): JSONObject = JSONObject()
            .put("setupWallMs", setupWallMs)
            .put("setupCpuMs", setupCpuMs)
            .put("setupThreadCpuMs", setupThreadCpuMs)
            .put("windowCount", windows.get())
            .put("stftWallMs", stftNanos.get() / 1_000_000.0)
            .put("inferenceWallMs", inferenceNanos.get() / 1_000_000.0)
            .put("istftWallMs", istftNanos.get() / 1_000_000.0)
            .put("totalWallMs", totalNanos.get() / 1_000_000.0)
            .put("workerCpuMs", workerCpuMillis.get().toDouble())
            .put("workerThreadCpuMs", workerThreadCpuNanos.get() / 1_000_000.0)
            .put("inferenceRtf", inferenceNanos.get() / 1_000_000_000.0 /
                max(1, windows.get()) / (model.usefulSamples.toDouble() / TfcTdfStreamingDsp.SAMPLE_RATE))
            .put("gpuEvidence", gpuRuntime?.evidence() ?: JSONObject.NULL)

        override fun close() {
            if (closed) return
            closed = true
            inputBuffer.close()
            outputBuffer.close()
            compiledModel.close()
            environment.close()
        }

        private companion object {
            fun threadCpuElapsedNanos(start: Long): Long {
                val end = Debug.threadCpuTimeNanos()
                return if (start >= 0L && end >= start) end - start else 0L
            }

            fun threadCpuElapsedMs(start: Long): Double =
                threadCpuElapsedNanos(start) / 1_000_000.0
        }
    }

    private class BoundedGpuRuntime private constructor(
        private val runtimeClass: Class<*>,
        private val capability: Any,
    ) {
        fun beginInference() = invoke("beginInference")
        fun endInference() = invoke("endInference")

        fun evidence(): JSONObject = JSONObject()
            .put("artifactVersion", value("getArtifactVersion"))
            .put("profileId", value("getProfileId"))
            .put("dispatchCount", invoke("getDispatchCount"))
            .put("eventWaitCount", invoke("getEventWaitCount"))

        private fun invoke(name: String): Any? = runtimeClass.getMethod(name).invoke(null)
        private fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)

        companion object {
            fun loadAndValidate(): BoundedGpuRuntime {
                val runtimeClass = Class.forName("io.github.wluhwluh.bss.litert.BssLiteRtRuntime")
                val capability = requireNotNull(runtimeClass.getMethod("queryCapability").invoke(null))
                require(capability.javaClass.getMethod("isAvailable").invoke(capability) == true)
                require(capability.javaClass.getMethod("getProfileId").invoke(capability) == "gpu-opencl-bounded-fp32-v1")
                return BoundedGpuRuntime(runtimeClass, capability)
            }
        }
    }

    private fun updateDigest(digest: MessageDigest, values: FloatArray) {
        val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach(buffer::putFloat)
        digest.update(buffer.array())
    }

    private fun percentile(values: List<Long>, fraction: Double): Long {
        if (values.isEmpty()) return 0L
        val sorted = values.sorted()
        val index = (ceil(fraction * sorted.size).toInt() - 1).coerceIn(sorted.indices)
        return sorted[index]
    }

    private fun elapsedMs(startNanos: Long, endNanos: Long): Double =
        (endNanos - startNanos) / 1_000_000.0

    private fun deviceIdentity(): JSONObject = JSONObject()
        .put("model", android.os.Build.MODEL)
        .put("device", android.os.Build.DEVICE)
        .put("socModel", android.os.Build.SOC_MODEL)
        .put("sdk", android.os.Build.VERSION.SDK_INT)
        .put("abis", JSONArray(android.os.Build.SUPPORTED_ABIS.toList()))

    private fun memoryEvidence(): JSONObject {
        val info = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        return JSONObject()
            .put("pssKb", info.totalPss)
            .put("privateDirtyKb", info.totalPrivateDirty)
            .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun gcEvidence(): JSONObject {
        val stats = Debug.getRuntimeStats()
        return JSONObject().apply {
            stats.filterKeys {
                it.contains("gc-count") || it.contains("blocking-gc-count") || it.contains("gc-time")
            }.forEach { (key, value) -> put(key, value) }
        }
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

    private fun sha256Value(value: String): String = value.also {
        require(it.matches(Regex("[0-9a-fA-F]{64}")))
    }.lowercase()

    private fun safeName(value: String): String = value.also {
        require(it.matches(Regex("[A-Za-z0-9._-]+")) && it != "." && it != "..")
    }
}
