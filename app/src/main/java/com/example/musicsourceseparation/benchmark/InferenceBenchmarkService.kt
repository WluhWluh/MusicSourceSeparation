package com.example.musicsourceseparation.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.IBinder
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Locale
import java.util.Random
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.ln
import kotlin.math.sqrt

class InferenceBenchmarkService : Service() {
    private val running = AtomicBoolean(false)

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, notification("Preparing inference benchmark"))
        if (!running.compareAndSet(false, true)) {
            writeErrorReport("A benchmark is already running.", intent)
            stopSelf(startId)
            return START_NOT_STICKY
        }

        thread(name = "inference-benchmark") {
            try {
                runBenchmark(intent ?: Intent())
            } catch (error: Throwable) {
                writeErrorReport(error.stackTraceToString(), intent)
            } finally {
                running.set(false)
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf(startId)
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun runBenchmark(intent: Intent) {
        if (intent.getStringExtra(EXTRA_BACKEND) == BACKEND_INIT) {
            initializeBenchmarkDirectories(intent)
            return
        }
        val backend = BenchmarkBackend.from(intent.getStringExtra(EXTRA_BACKEND))
        val iterations = intent.getIntExtra(EXTRA_ITERATIONS, DEFAULT_ITERATIONS).coerceIn(1, 100)
        val warmups = intent.getIntExtra(EXTRA_WARMUPS, DEFAULT_WARMUPS).coerceIn(0, 20)
        val threads = intent.getIntExtra(EXTRA_THREADS, DEFAULT_THREADS).coerceIn(1, 16)
        val seed = intent.getLongExtra(EXTRA_SEED, DEFAULT_SEED)
        val height = intent.getIntExtra(EXTRA_HEIGHT, DEFAULT_HEIGHT)
        val width = intent.getIntExtra(EXTRA_WIDTH, DEFAULT_WIDTH)
        require(height in 1..4096 && width in 1..4096) {
            "Invalid model input dimensions: $height x $width"
        }
        val tag = intent.getStringExtra(EXTRA_TAG)?.sanitizeTag()
            ?.takeIf { it.isNotBlank() }
            ?: "${backend.id}-${System.currentTimeMillis()}"
        val modelsDir = File(benchmarkDir(), "models").apply { mkdirs() }
        val modelFile = File(
            modelsDir,
            when (backend) {
                BenchmarkBackend.ORT -> intent.getStringExtra(EXTRA_ONNX_MODEL)
                    ?: DEFAULT_ONNX_MODEL
                BenchmarkBackend.LITERT_CPU,
                BenchmarkBackend.LITERT_GPU,
                BenchmarkBackend.LITERT_GPU_FP32 -> intent.getStringExtra(EXTRA_LITERT_MODEL)
                    ?: DEFAULT_LITERT_MODEL
            },
        )
        require(modelFile.isFile && modelFile.length() > 0L) {
            "Model file is missing: ${modelFile.absolutePath}"
        }
        val modelId = intent.getStringExtra(EXTRA_MODEL_ID)?.sanitizeTag()
            ?.takeIf { it.isNotBlank() }
            ?: modelFile.nameWithoutExtension.sanitizeTag()
        val inputFile = intent.getStringExtra(EXTRA_INPUT_FILE)
            ?.sanitizeTag()
            ?.takeIf { it.isNotBlank() }
            ?.let { File(inputsDir(), it) }
        require(inputFile == null || inputFile.isFile) {
            "Input file is missing: ${inputFile?.absolutePath}"
        }

        val reportFile = File(reportsDir(), "$tag.json")
        File(reportsDir(), LATEST_REPORT).writeText(
            baseReport(
                tag = tag,
                backend = backend,
                modelId = modelId,
                modelFile = modelFile,
                iterations = iterations,
                warmups = warmups,
                threads = threads,
                seed = seed,
                inputSource = inputFile?.absolutePath ?: "generated",
                height = height,
                width = width,
            )
                .put("status", "running")
                .put("reportPath", reportFile.absolutePath)
                .toString(2),
        )
        updateNotification("Running ${backend.id}")

        val elementCount = elementCount(height, width)
        val inputNchw = inputFile?.let { readInput(it, elementCount) }
            ?: createInput(seed, elementCount)
        val processStart = processSnapshot()
        val deviceStart = deviceSnapshot()
        val result = when (backend) {
            BenchmarkBackend.ORT -> runOrt(
                modelFile,
                inputNchw,
                iterations,
                warmups,
                threads,
                height,
                width,
            )
            BenchmarkBackend.LITERT_CPU -> runLiteRt(
                modelFile = modelFile,
                inputNchw = inputNchw,
                iterations = iterations,
                warmups = warmups,
                threads = threads,
                gpuPrecision = null,
                height = height,
                width = width,
            )
            BenchmarkBackend.LITERT_GPU -> runLiteRt(
                modelFile = modelFile,
                inputNchw = inputNchw,
                iterations = iterations,
                warmups = warmups,
                threads = threads,
                gpuPrecision = CompiledModel.GpuOptions.Precision.FP16,
                height = height,
                width = width,
            )
            BenchmarkBackend.LITERT_GPU_FP32 -> runLiteRt(
                modelFile = modelFile,
                inputNchw = inputNchw,
                iterations = iterations,
                warmups = warmups,
                threads = threads,
                gpuPrecision = CompiledModel.GpuOptions.Precision.FP32,
                height = height,
                width = width,
            )
        }
        val processEnd = processSnapshot()
        val deviceEnd = deviceSnapshot()
        val output = result.output
        val outputStats = outputStats(output)
        val referenceKey = inputFile?.nameWithoutExtension ?: "generated"
        val referenceFile = File(
            referenceDir(),
            "ort-$modelId-$seed-$referenceKey-fp32.bin",
        )
        val comparison = if (backend == BenchmarkBackend.ORT) {
            writeFloatArray(referenceFile, output)
            null
        } else if (referenceFile.isFile) {
            compareOutputs(readFloatArray(referenceFile), output)
        } else {
            null
        }

        val report = baseReport(
            tag = tag,
            backend = backend,
            modelId = modelId,
            modelFile = modelFile,
            iterations = iterations,
            warmups = warmups,
            threads = threads,
            seed = seed,
            inputSource = inputFile?.absolutePath ?: "generated",
            height = height,
            width = width,
        )
            .put("status", "complete")
            .put("reportPath", reportFile.absolutePath)
            .put("referencePath", referenceFile.absolutePath)
            .put("setupWallMs", result.setupWallMs)
            .put("setupCpuMs", result.setupCpuMs)
            .put("availableAccelerators", JSONArray(result.availableAccelerators))
            .put("inputShapeNchw", JSONArray(listOf(BATCH, CHANNELS, height, width)))
            .put("runtimeInputShape", JSONArray(result.runtimeInputShape))
            .put("runtimeOutputShape", JSONArray(result.runtimeOutputShape))
            .put("warmupWallMs", JSONArray(result.warmupWallMs))
            .put("warmupCpuMs", JSONArray(result.warmupCpuMs))
            .put("inferenceWallMs", JSONArray(result.inferenceWallMs))
            .put("inferenceCpuMs", JSONArray(result.inferenceCpuMs))
            .put("dispatchWallMs", JSONArray(result.dispatchWallMs))
            .put("dispatchCpuMs", JSONArray(result.dispatchCpuMs))
            .put("outputReadWallMs", result.outputReadWallMs)
            .put("outputReadCpuMs", result.outputReadCpuMs)
            .put("inferenceSummary", timingSummary(result.inferenceWallMs, result.inferenceCpuMs))
            .put("output", outputStats)
            .put("comparisonToOrt", comparison ?: JSONObject.NULL)
            .put("processStart", processStart)
            .put("processEnd", processEnd)
            .put("deviceStart", deviceStart)
            .put("deviceEnd", deviceEnd)

        reportFile.writeText(report.toString(2))
        File(reportsDir(), LATEST_REPORT).writeText(report.toString(2))
        updateNotification("Completed ${backend.id}")
    }

    private fun initializeBenchmarkDirectories(intent: Intent) {
        val tag = intent.getStringExtra(EXTRA_TAG)?.sanitizeTag()
            ?.takeIf { it.isNotBlank() }
            ?: BACKEND_INIT
        val root = benchmarkDir()
        val models = File(root, "models").apply { mkdirs() }
        val reports = reportsDir()
        val reference = referenceDir()
        inputsDir().mkdirs()
        val report = JSONObject()
            .put("schemaVersion", 1)
            .put("status", "complete")
            .put("tag", tag)
            .put("backend", BACKEND_INIT)
            .put("rootPath", root.absolutePath)
            .put("modelsPath", models.absolutePath)
            .put("reportsPath", reports.absolutePath)
            .put("referencePath", reference.absolutePath)
            .put("inputsPath", inputsDir().absolutePath)
        File(reports, "$tag.json").writeText(report.toString(2))
        File(reports, LATEST_REPORT).writeText(report.toString(2))
        updateNotification("Benchmark directories ready")
    }

    private fun runOrt(
        modelFile: File,
        inputNchw: FloatArray,
        iterations: Int,
        warmups: Int,
        threads: Int,
        height: Int,
        width: Int,
    ): BackendResult {
        val environment = OrtEnvironment.getEnvironment()
        val setupStarted = timedStart()
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setInterOpNumThreads(1)
            setIntraOpNumThreads(threads)
        }
        val session = environment.createSession(modelFile.absolutePath, options)
        options.close()
        val setup = setupStarted.elapsed()
        val inputName = session.inputNames.first()
        val outputName = session.outputNames.first()
        val shape = longArrayOf(
            BATCH.toLong(),
            CHANNELS.toLong(),
            height.toLong(),
            width.toLong(),
        )
        val inputBuffer = directFloatBuffer(inputNchw)
        val tensor = OnnxTensor.createTensor(environment, inputBuffer, shape)
        var output: FloatArray? = null
        val warmupWall = mutableListOf<Double>()
        val warmupCpu = mutableListOf<Long>()
        val inferenceWall = mutableListOf<Double>()
        val inferenceCpu = mutableListOf<Long>()
        var readWallMs = 0.0
        var readCpuMs = 0L

        try {
            repeat(warmups) {
                val started = timedStart()
                session.run(mapOf(inputName to tensor)).use { result ->
                    started.elapsed().also {
                        warmupWall += it.wallMs
                        warmupCpu += it.cpuMs
                    }
                    val outputTensor = result[outputName].orElseThrow() as OnnxTensor
                    output = outputTensor.floatBuffer.toFloatArray()
                }
            }
            repeat(iterations) {
                val started = timedStart()
                session.run(mapOf(inputName to tensor)).use { result ->
                    started.elapsed().also {
                        inferenceWall += it.wallMs
                        inferenceCpu += it.cpuMs
                    }
                    val readStarted = timedStart()
                    val outputTensor = result[outputName].orElseThrow() as OnnxTensor
                    output = outputTensor.floatBuffer.toFloatArray()
                    readStarted.elapsed().also {
                        readWallMs += it.wallMs
                        readCpuMs += it.cpuMs
                    }
                }
            }
        } finally {
            tensor.close()
            session.close()
        }

        return BackendResult(
            setupWallMs = setup.wallMs,
            setupCpuMs = setup.cpuMs,
            availableAccelerators = listOf("CPU"),
            runtimeInputShape = listOf(BATCH, CHANNELS, height, width),
            runtimeOutputShape = listOf(BATCH, CHANNELS, height, width),
            warmupWallMs = warmupWall,
            warmupCpuMs = warmupCpu,
            inferenceWallMs = inferenceWall,
            inferenceCpuMs = inferenceCpu,
            dispatchWallMs = emptyList(),
            dispatchCpuMs = emptyList(),
            outputReadWallMs = readWallMs,
            outputReadCpuMs = readCpuMs,
            output = requireNotNull(output) { "ORT produced no output." },
        )
    }

    private fun runLiteRt(
        modelFile: File,
        inputNchw: FloatArray,
        iterations: Int,
        warmups: Int,
        threads: Int,
        gpuPrecision: CompiledModel.GpuOptions.Precision?,
        height: Int,
        width: Int,
    ): BackendResult {
        val setupStarted = timedStart()
        val environment = Environment.create()
        val availableAccelerators = environment.getAvailableAccelerators().map { it.name }.sorted()
        val options = if (gpuPrecision != null) {
            CompiledModel.Options(Accelerator.GPU).apply {
                gpuOptions = CompiledModel.GpuOptions(
                    precision = gpuPrecision,
                    backend = CompiledModel.GpuOptions.Backend.AUTOMATIC,
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
        val model = CompiledModel.create(modelFile.absolutePath, options, environment)
        val inputBuffers = model.createInputBuffers()
        val outputBuffers = model.createOutputBuffers()
        require(inputBuffers.size == 1 && outputBuffers.size == 1) {
            "Expected one input and one output, got ${inputBuffers.size}/${outputBuffers.size}."
        }
        val inputShape = requireNotNull(model.getInputTensorType("input").layout) {
            "LiteRT input tensor has no layout."
        }.dimensions
        val outputShape = requireNotNull(model.getOutputTensorType("output").layout) {
            "LiteRT output tensor has no layout."
        }.dimensions
        val expectedShape = listOf(BATCH, height, width, CHANNELS)
        require(inputShape == expectedShape && outputShape == expectedShape) {
            "Unexpected LiteRT shapes: input=$inputShape output=$outputShape expected=$expectedShape"
        }
        val inputNhwc = nchwToNhwc(inputNchw, height, width)
        inputBuffers.single().writeFloat(inputNhwc)
        val setup = setupStarted.elapsed()
        val warmupWall = mutableListOf<Double>()
        val warmupCpu = mutableListOf<Long>()
        val inferenceWall = mutableListOf<Double>()
        val inferenceCpu = mutableListOf<Long>()
        val dispatchWall = mutableListOf<Double>()
        val dispatchCpu = mutableListOf<Long>()
        var outputNhwc: FloatArray? = null
        var readWallMs = 0.0
        var readCpuMs = 0L

        try {
            repeat(warmups) {
                val totalStarted = timedStart()
                model.run(inputBuffers, outputBuffers)
                outputNhwc = outputBuffers.single().readFloat()
                totalStarted.elapsed().also {
                    warmupWall += it.wallMs
                    warmupCpu += it.cpuMs
                }
            }
            repeat(iterations) {
                val totalStarted = timedStart()
                val dispatchStarted = timedStart()
                model.run(inputBuffers, outputBuffers)
                dispatchStarted.elapsed().also {
                    dispatchWall += it.wallMs
                    dispatchCpu += it.cpuMs
                }
                val readStarted = timedStart()
                outputNhwc = outputBuffers.single().readFloat()
                readStarted.elapsed().also {
                    readWallMs += it.wallMs
                    readCpuMs += it.cpuMs
                }
                totalStarted.elapsed().also {
                    inferenceWall += it.wallMs
                    inferenceCpu += it.cpuMs
                }
            }
        } finally {
            inputBuffers.closeAll()
            outputBuffers.closeAll()
            model.close()
            environment.close()
        }

        return BackendResult(
            setupWallMs = setup.wallMs,
            setupCpuMs = setup.cpuMs,
            availableAccelerators = availableAccelerators,
            runtimeInputShape = inputShape,
            runtimeOutputShape = outputShape,
            warmupWallMs = warmupWall,
            warmupCpuMs = warmupCpu,
            inferenceWallMs = inferenceWall,
            inferenceCpuMs = inferenceCpu,
            dispatchWallMs = dispatchWall,
            dispatchCpuMs = dispatchCpu,
            outputReadWallMs = readWallMs,
            outputReadCpuMs = readCpuMs,
            output = nhwcToNchw(
                requireNotNull(outputNhwc) { "LiteRT produced no output." },
                height,
                width,
            ),
        )
    }

    private fun baseReport(
        tag: String,
        backend: BenchmarkBackend,
        modelId: String,
        modelFile: File,
        iterations: Int,
        warmups: Int,
        threads: Int,
        seed: Long,
        inputSource: String,
        height: Int,
        width: Int,
    ): JSONObject = JSONObject()
        .put("schemaVersion", 1)
        .put("tag", tag)
        .put("backend", backend.id)
        .put("modelId", modelId)
        .put("modelPath", modelFile.absolutePath)
        .put("modelBytes", modelFile.length())
        .put("iterations", iterations)
        .put("warmups", warmups)
        .put("threads", threads)
        .put("seed", seed)
        .put("inputSource", inputSource)
        .put("inputShapeNchw", JSONArray(listOf(BATCH, CHANNELS, height, width)))
        .put("device", JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("device", Build.DEVICE)
            .put("sdk", Build.VERSION.SDK_INT)
            .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())))

    private fun timingSummary(wallMs: List<Double>, cpuMs: List<Long>): JSONObject {
        val sorted = wallMs.sorted()
        return JSONObject()
            .put("wallMeanMs", wallMs.average())
            .put("wallMedianMs", sorted.percentile(0.5))
            .put("wallMinMs", sorted.firstOrNull() ?: 0.0)
            .put("wallMaxMs", sorted.lastOrNull() ?: 0.0)
            .put("cpuMeanMs", cpuMs.average())
            .put("cpuToWallRatio", cpuMs.sum().toDouble() / wallMs.sum().coerceAtLeast(0.001))
    }

    private fun outputStats(values: FloatArray): JSONObject {
        var sum = 0.0
        var sumSquares = 0.0
        var min = Double.POSITIVE_INFINITY
        var max = Double.NEGATIVE_INFINITY
        var nonFinite = 0L
        for (value in values) {
            val doubleValue = value.toDouble()
            if (!doubleValue.isFinite()) {
                nonFinite++
                continue
            }
            sum += kotlin.math.abs(doubleValue)
            sumSquares += doubleValue * doubleValue
            min = minOf(min, doubleValue)
            max = maxOf(max, doubleValue)
        }
        val finiteCount = (values.size - nonFinite).coerceAtLeast(1)
        return JSONObject()
            .put("elementCount", values.size)
            .put("nonFiniteCount", nonFinite)
            .put("absMean", sum / finiteCount)
            .put("rms", sqrt(sumSquares / finiteCount))
            .put("min", min)
            .put("max", max)
    }

    private fun compareOutputs(reference: FloatArray, candidate: FloatArray): JSONObject {
        require(reference.size == candidate.size) {
            "Output size mismatch: ${reference.size} != ${candidate.size}"
        }
        var signalSquares = 0.0
        var errorSquares = 0.0
        var errorAbs = 0.0
        var maxAbsError = 0.0
        var dot = 0.0
        var candidateSquares = 0.0
        for (index in reference.indices) {
            val expected = reference[index].toDouble()
            val actual = candidate[index].toDouble()
            val error = actual - expected
            val absError = kotlin.math.abs(error)
            signalSquares += expected * expected
            candidateSquares += actual * actual
            errorSquares += error * error
            errorAbs += absError
            dot += expected * actual
            maxAbsError = maxOf(maxAbsError, absError)
        }
        val count = reference.size.coerceAtLeast(1)
        val snrDb = if (errorSquares == 0.0) {
            Double.POSITIVE_INFINITY
        } else {
            10.0 * ln(signalSquares / errorSquares) / ln(10.0)
        }
        return JSONObject()
            .put("maxAbsError", maxAbsError)
            .put("meanAbsError", errorAbs / count)
            .put("rmse", sqrt(errorSquares / count))
            .put("signalRms", sqrt(signalSquares / count))
            .put("snrDb", snrDb)
            .put("cosineSimilarity", dot / sqrt(signalSquares * candidateSquares))
    }

    private fun processSnapshot(): JSONObject {
        val memory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        return JSONObject()
            .put("elapsedRealtimeMs", SystemClock.elapsedRealtime())
            .put("processCpuMs", Process.getElapsedCpuTime())
            .put("totalPssKb", memory.totalPss)
            .put("totalPrivateDirtyKb", memory.totalPrivateDirty)
            .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun deviceSnapshot(): JSONObject {
        val batteryManager = getSystemService(BatteryManager::class.java)
        val powerManager = getSystemService(PowerManager::class.java)
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return JSONObject()
            .put("elapsedRealtimeMs", SystemClock.elapsedRealtime())
            .put("thermalStatus", if (Build.VERSION.SDK_INT >= 29) powerManager.currentThermalStatus else -1)
            .put("batteryLevelPercent", battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1)
            .put("batteryTemperatureDeciC", battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1)
            .put("chargeCounterUah", batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER))
            .put("energyCounterNwh", batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER))
            .put("currentNowUa", batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW))
    }

    private fun createInput(seed: Long, elementCount: Int): FloatArray {
        val random = Random(seed)
        return FloatArray(elementCount) { (random.nextFloat() * 0.2f) - 0.1f }
    }

    private fun readInput(file: File, elementCount: Int): FloatArray {
        val expectedBytes = elementCount.toLong() * Float.SIZE_BYTES
        require(file.length() == expectedBytes) {
            "Expected $expectedBytes bytes in ${file.name}, got ${file.length()}."
        }
        return file.inputStream().use { input ->
            val bytes = input.readBytes()
            ByteBuffer.wrap(bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()
                .let { buffer -> FloatArray(elementCount).also(buffer::get) }
        }
    }

    private fun nchwToNhwc(input: FloatArray, height: Int, width: Int): FloatArray {
        require(input.size == elementCount(height, width))
        val output = FloatArray(input.size)
        for (channel in 0 until CHANNELS) {
            for (heightIndex in 0 until height) {
                for (widthIndex in 0 until width) {
                    val nchwIndex = (channel * height + heightIndex) * width + widthIndex
                    val nhwcIndex = (heightIndex * width + widthIndex) * CHANNELS + channel
                    output[nhwcIndex] = input[nchwIndex]
                }
            }
        }
        return output
    }

    private fun nhwcToNchw(input: FloatArray, height: Int, width: Int): FloatArray {
        require(input.size == elementCount(height, width))
        val output = FloatArray(input.size)
        for (heightIndex in 0 until height) {
            for (widthIndex in 0 until width) {
                for (channel in 0 until CHANNELS) {
                    val nhwcIndex = (heightIndex * width + widthIndex) * CHANNELS + channel
                    val nchwIndex = (channel * height + heightIndex) * width + widthIndex
                    output[nchwIndex] = input[nhwcIndex]
                }
            }
        }
        return output
    }

    private fun elementCount(height: Int, width: Int): Int = BATCH * CHANNELS * height * width

    private fun directFloatBuffer(values: FloatArray): FloatBuffer = ByteBuffer
        .allocateDirect(values.size * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
        .apply {
            put(values)
            rewind()
        }

    private fun writeFloatArray(file: File, values: FloatArray) {
        file.parentFile?.mkdirs()
        file.outputStream().channel.use { channel ->
            val buffer = ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
                .order(ByteOrder.LITTLE_ENDIAN)
            buffer.asFloatBuffer().put(values)
            channel.write(buffer)
        }
    }

    private fun readFloatArray(file: File): FloatArray {
        require(file.length() % Float.SIZE_BYTES == 0L) { "Invalid reference file: ${file.absolutePath}" }
        val bytes = file.readBytes()
        val floats = FloatArray(bytes.size / Float.SIZE_BYTES)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(floats)
        return floats
    }

    private fun writeErrorReport(message: String, intent: Intent?) {
        val backend = intent?.getStringExtra(EXTRA_BACKEND).orEmpty().ifBlank { "unknown" }
        val tag = intent?.getStringExtra(EXTRA_TAG)?.sanitizeTag()
            ?.takeIf { it.isNotBlank() }
            ?: "error-${System.currentTimeMillis()}"
        val report = JSONObject()
            .put("schemaVersion", 1)
            .put("status", "error")
            .put("tag", tag)
            .put("backend", backend)
            .put("message", message)
            .put("device", JSONObject().put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT))
        val reportFile = File(reportsDir(), "$tag.json")
        reportFile.writeText(report.toString(2))
        File(reportsDir(), LATEST_REPORT).writeText(report.toString(2))
    }

    private fun benchmarkDir(): File = File(getExternalFilesDir(null) ?: filesDir, "benchmark")
        .apply { mkdirs() }

    private fun reportsDir(): File = File(benchmarkDir(), "reports").apply { mkdirs() }

    private fun referenceDir(): File = File(benchmarkDir(), "reference").apply { mkdirs() }

    private fun inputsDir(): File = File(benchmarkDir(), "inputs").apply { mkdirs() }

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Inference benchmark",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    private fun notification(text: String): Notification = Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
        .setSmallIcon(android.R.drawable.stat_notify_sync)
        .setContentTitle("MDX inference benchmark")
        .setContentText(text)
        .setOngoing(true)
        .build()

    private fun updateNotification(text: String) {
        startForeground(NOTIFICATION_ID, notification(text))
    }

    private fun timedStart(): TimedStart = TimedStart(
        elapsedNanos = SystemClock.elapsedRealtimeNanos(),
        cpuMs = Process.getElapsedCpuTime(),
    )

    private fun FloatBuffer.toFloatArray(): FloatArray {
        val duplicate = duplicate().apply { rewind() }
        return FloatArray(duplicate.remaining()).also(duplicate::get)
    }

    private fun List<TensorBuffer>.closeAll() {
        forEach { it.close() }
    }

    private fun List<Double>.percentile(fraction: Double): Double {
        if (isEmpty()) return 0.0
        val index = ((size - 1) * fraction).toInt().coerceIn(indices)
        return this[index]
    }

    private fun String.sanitizeTag(): String = lowercase(Locale.US)
        .replace(Regex("[^a-z0-9._-]+"), "-")
        .trim('-')

    private data class TimedStart(val elapsedNanos: Long, val cpuMs: Long) {
        fun elapsed(): Timing = Timing(
            wallMs = (SystemClock.elapsedRealtimeNanos() - elapsedNanos) / 1_000_000.0,
            cpuMs = Process.getElapsedCpuTime() - cpuMs,
        )
    }

    private data class Timing(val wallMs: Double, val cpuMs: Long)

    private data class BackendResult(
        val setupWallMs: Double,
        val setupCpuMs: Long,
        val availableAccelerators: List<String>,
        val runtimeInputShape: List<Int>,
        val runtimeOutputShape: List<Int>,
        val warmupWallMs: List<Double>,
        val warmupCpuMs: List<Long>,
        val inferenceWallMs: List<Double>,
        val inferenceCpuMs: List<Long>,
        val dispatchWallMs: List<Double>,
        val dispatchCpuMs: List<Long>,
        val outputReadWallMs: Double,
        val outputReadCpuMs: Long,
        val output: FloatArray,
    )

    private enum class BenchmarkBackend(val id: String) {
        ORT("ort"),
        LITERT_CPU("litert_cpu"),
        LITERT_GPU("litert_gpu"),
        LITERT_GPU_FP32("litert_gpu_fp32");

        companion object {
            fun from(value: String?): BenchmarkBackend = entries.firstOrNull { it.id == value }
                ?: error("Unknown backend: $value")
        }
    }

    companion object {
        const val ACTION_RUN = "com.example.musicsourceseparation.RUN_INFERENCE_BENCHMARK"

        const val EXTRA_BACKEND = "backend"
        const val EXTRA_ITERATIONS = "iterations"
        const val EXTRA_WARMUPS = "warmups"
        const val EXTRA_THREADS = "threads"
        const val EXTRA_SEED = "seed"
        const val EXTRA_TAG = "tag"
        const val EXTRA_MODEL_ID = "modelId"
        const val EXTRA_HEIGHT = "height"
        const val EXTRA_WIDTH = "width"
        const val EXTRA_INPUT_FILE = "inputFile"
        const val EXTRA_ONNX_MODEL = "onnxModel"
        const val EXTRA_LITERT_MODEL = "litertModel"

        const val DEFAULT_ONNX_MODEL = "UVR_MDXNET_9482.onnx"
        const val DEFAULT_LITERT_MODEL = "UVR_MDXNET_9482_float32.tflite"

        private const val BACKEND_INIT = "init"

        private const val DEFAULT_ITERATIONS = 5
        private const val DEFAULT_WARMUPS = 1
        private const val DEFAULT_THREADS = 8
        private const val DEFAULT_SEED = 9482L
        private const val LATEST_REPORT = "latest.json"
        private const val NOTIFICATION_CHANNEL_ID = "inference_benchmark"
        private const val NOTIFICATION_ID = 9482

        private const val BATCH = 1
        private const val CHANNELS = 4
        private const val DEFAULT_HEIGHT = 2048
        private const val DEFAULT_WIDTH = 256
    }
}
