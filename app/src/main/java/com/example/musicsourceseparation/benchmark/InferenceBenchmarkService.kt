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
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.NpuCompatibilityChecker
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
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
        if (backend == BenchmarkBackend.LITERT_QNN_AUDIO) {
            runQnnAudioBenchmark(intent, backend)
            return
        }
        val iterations = intent.getIntExtra(EXTRA_ITERATIONS, DEFAULT_ITERATIONS)
            .coerceIn(1, MAX_ITERATIONS)
        val warmups = intent.getIntExtra(EXTRA_WARMUPS, DEFAULT_WARMUPS).coerceIn(0, 20)
        val threads = intent.getIntExtra(EXTRA_THREADS, DEFAULT_THREADS).coerceIn(1, 16)
        val seed = intent.getLongExtra(EXTRA_SEED, DEFAULT_SEED)
        val qnnProfiling = intent.getBooleanExtra(EXTRA_QNN_PROFILING, false)
        val exportOutputTensor = intent.getBooleanExtra(EXTRA_EXPORT_OUTPUT_TENSOR, false)
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
                BenchmarkBackend.LITERT_GPU_FP32,
                BenchmarkBackend.LITERT_GPU_BOUNDED,
                BenchmarkBackend.LITERT_QNN,
                BenchmarkBackend.LITERT_QNN_AUDIO -> intent.getStringExtra(EXTRA_LITERT_MODEL)
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
        publishReport(
            File(reportsDir(), LATEST_REPORT),
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
                .put("reportPath", reportFile.absolutePath),
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
                useQnn = false,
                boundedGpu = false,
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
                useQnn = false,
                boundedGpu = false,
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
                useQnn = false,
                boundedGpu = false,
                height = height,
                width = width,
            )
            BenchmarkBackend.LITERT_GPU_BOUNDED -> runLiteRt(
                modelFile = modelFile,
                inputNchw = inputNchw,
                iterations = iterations,
                warmups = warmups,
                threads = threads,
                gpuPrecision = CompiledModel.GpuOptions.Precision.FP32,
                useQnn = false,
                boundedGpu = true,
                height = height,
                width = width,
            )
            BenchmarkBackend.LITERT_QNN -> runLiteRt(
                modelFile = modelFile,
                inputNchw = inputNchw,
                iterations = iterations,
                warmups = warmups,
                threads = threads,
                gpuPrecision = null,
                useQnn = true,
                boundedGpu = false,
                qnnProfiling = qnnProfiling,
                qnnEvidenceDir = File(qnnEvidenceRoot(), tag).apply { mkdirs() },
                height = height,
                width = width,
            )
            BenchmarkBackend.LITERT_QNN_AUDIO -> error("QNN audio is handled before tensor benchmarks.")
        }
        val processEnd = processSnapshot()
        val deviceEnd = deviceSnapshot()
        val output = result.output
        val outputStats = outputStats(output)
        val outputTensorFile = if (exportOutputTensor) {
            File(tensorOutputDir(), "$tag-nchw-f32.bin").also { writeFloatArray(it, output) }
        } else {
            null
        }
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
            .put("outputTensor", outputTensorFile?.let { file ->
                JSONObject()
                    .put("path", file.absolutePath)
                    .put("bytes", file.length())
                    .put("sha256", sha256(file))
            } ?: JSONObject.NULL)
            .put("comparisonToOrt", comparison ?: JSONObject.NULL)
            .put("backendEvidence", result.backendEvidence ?: JSONObject.NULL)
            .put("processStart", processStart)
            .put("processEnd", processEnd)
            .put("deviceStart", deviceStart)
            .put("deviceEnd", deviceEnd)

        publishReport(File(reportsDir(), LATEST_REPORT), report)
        publishReport(reportFile, report)
        updateNotification("Completed ${backend.id}")
    }

    private fun runQnnAudioBenchmark(intent: Intent, backend: BenchmarkBackend) {
        val tag = intent.getStringExtra(EXTRA_TAG)?.sanitizeTag()
            ?.takeIf { it.isNotBlank() }
            ?: "${backend.id}-${System.currentTimeMillis()}"
        val modelId = intent.getStringExtra(EXTRA_MODEL_ID)?.sanitizeTag()
            ?.takeIf { it.isNotBlank() }
            ?: "uvr_mdxnet_3_9662"
        require(modelId == QNN_AUDIO_MODEL_ID) {
            "QNN audio benchmark is frozen to $QNN_AUDIO_MODEL_ID, got $modelId."
        }
        val modelName = intent.getStringExtra(EXTRA_LITERT_MODEL)?.validatedFileName()
            ?.takeIf { it.isNotBlank() }
            ?: DEFAULT_LITERT_MODEL
        val audioName = intent.getStringExtra(EXTRA_AUDIO_FILE)?.validatedFileName()
            ?.takeIf { it.isNotBlank() }
            ?: error("QNN audio benchmark requires an audio file name.")
        val modelOutputScale = intent.getFloatExtra(EXTRA_MODEL_OUTPUT_SCALE, DEFAULT_MODEL_OUTPUT_SCALE)
        require(modelOutputScale == DEFAULT_MODEL_OUTPUT_SCALE) {
            "QNN audio benchmark is frozen to model output scale $DEFAULT_MODEL_OUTPUT_SCALE."
        }
        val modelFile = File(File(benchmarkDir(), "models"), modelName)
        val audioFile = File(audioInputsDir(), audioName)
        val outputDir = File(audioOutputsDir(), tag)
        val evidenceDir = File(qnnEvidenceRoot(), tag).apply {
            deleteRecursively()
            mkdirs()
        }
        val reportFile = File(reportsDir(), "$tag.json")
        val reportBase = JSONObject()
            .put("schemaVersion", 1)
            .put("tag", tag)
            .put("backend", backend.id)
            .put("modelId", modelId)
            .put("modelPath", modelFile.absolutePath)
            .put("modelBytes", modelFile.length())
            .put("audioPath", audioFile.absolutePath)
            .put("audioBytes", audioFile.length())
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("model", Build.MODEL)
                .put("device", Build.DEVICE)
                .put("sdk", Build.VERSION.SDK_INT)
                .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())))
            .put("reportPath", reportFile.absolutePath)
        publishReport(
            File(reportsDir(), LATEST_REPORT),
            JSONObject(reportBase.toString()).put("status", "running"),
        )

        val processStart = processSnapshot()
        val deviceStart = deviceSnapshot()
        val result = QnnMdxAudioBenchmark(this).run(
            modelFile = modelFile,
            audioFile = audioFile,
            outputDir = outputDir,
            evidenceDir = evidenceDir,
            modelOutputScale = modelOutputScale,
            onPhase = ::updateNotification,
            onProgress = { completed, total ->
                updateNotification("Separating audio $completed/$total")
            },
        )
        val report = JSONObject(reportBase.toString())
            .put("status", "complete")
            .put("audio", result)
            .put("processStart", processStart)
            .put("processEnd", processSnapshot())
            .put("deviceStart", deviceStart)
            .put("deviceEnd", deviceSnapshot())
        publishReport(File(reportsDir(), LATEST_REPORT), report)
        publishReport(reportFile, report)
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
        val tensorOutput = tensorOutputDir()
        val audioInput = audioInputsDir()
        val audioOutput = audioOutputsDir()
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
            .put("tensorOutputPath", tensorOutput.absolutePath)
            .put("audioInputPath", audioInput.absolutePath)
            .put("audioOutputPath", audioOutput.absolutePath)
            .put("inputsPath", inputsDir().absolutePath)
        publishReport(File(reports, LATEST_REPORT), report)
        publishReport(File(reports, "$tag.json"), report)
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
        useQnn: Boolean,
        boundedGpu: Boolean,
        qnnProfiling: Boolean = false,
        qnnEvidenceDir: File? = null,
        height: Int,
        width: Int,
    ): BackendResult {
        val setupStarted = timedStart()
        require(!useQnn || gpuPrecision == null) { "QNN and GPU modes are mutually exclusive." }
        require(!boundedGpu || gpuPrecision == CompiledModel.GpuOptions.Precision.FP32) {
            "The bounded GPU profile requires FP32."
        }
        val boundedGpuRuntime = if (boundedGpu) BoundedGpuRuntime.loadAndValidate() else null
        val npuProvider = if (useQnn) {
            require(Build.VERSION.SDK_INT >= 31) { "QNN v79 requires Android API 31 or newer." }
            require(Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a") {
                "QNN v79 requires an arm64-v8a process; ABIs=${Build.SUPPORTED_ABIS.toList()}"
            }
            BuiltinNpuAcceleratorProvider(this, NpuCompatibilityChecker.Qualcomm).also { provider ->
                require(provider.isDeviceSupported()) {
                    "LiteRT does not recognize ${Build.SOC_MANUFACTURER}/${Build.SOC_MODEL} as a supported Qualcomm NPU."
                }
                require(provider.isLibraryReady()) { "The bundled Qualcomm NPU runtime is not ready." }
                Log.i(
                    QNN_LOG_TAG,
                    "QNN preflight soc=${Build.SOC_MANUFACTURER}/${Build.SOC_MODEL} " +
                        "libraryDir=${provider.getLibraryDir()}",
                )
            }
        } else {
            null
        }
        val environment = npuProvider?.let { Environment.create(it) } ?: Environment.create()
        var modelToClose: CompiledModel? = null
        var inputBuffers: List<TensorBuffer> = emptyList()
        var outputBuffers: List<TensorBuffer> = emptyList()
        return try {
            val availableAccelerators = environment.getAvailableAccelerators().map { it.name }.sorted()
            if (useQnn) {
                require(Accelerator.NPU.name in availableAccelerators) {
                    "NPU was not registered; available accelerators=$availableAccelerators"
                }
            }
            val options = if (useQnn) {
                val evidenceDir = requireNotNull(qnnEvidenceDir) { "QNN evidence directory is required." }
                CompiledModel.Options(Accelerator.NPU).apply {
                    qualcommOptions = CompiledModel.QualcommOptions(
                        logLevel = CompiledModel.QualcommOptions.LogLevel.INFO,
                        useHtpPreference = true,
                        htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode
                            .SUSTAINED_HIGH_PERFORMANCE,
                        profiling = if (qnnProfiling) {
                            CompiledModel.QualcommOptions.Profiling.DETAILED
                        } else {
                            CompiledModel.QualcommOptions.Profiling.OFF
                        },
                        irJsonDir = evidenceDir.absolutePath,
                        optimizationLevel = CompiledModel.QualcommOptions.OptimizationLevel
                            .HTP_OPTIMIZE_FOR_INFERENCE,
                    )
                }
            } else if (gpuPrecision != null) {
                CompiledModel.Options(Accelerator.GPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        precision = gpuPrecision,
                        backend = if (boundedGpu) {
                            CompiledModel.GpuOptions.Backend.OPENCL
                        } else {
                            CompiledModel.GpuOptions.Backend.AUTOMATIC
                        },
                        numStepsOfCommandBufferPreparations = if (boundedGpu) 1 else null,
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
            val model = CompiledModel.create(modelFile.absolutePath, options, environment).also {
                modelToClose = it
            }
            inputBuffers = model.createInputBuffers()
            outputBuffers = model.createOutputBuffers()
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
            boundedGpuRuntime?.resetInferenceCounters()

            repeat(warmups) {
                val totalStarted = timedStart()
                runLiteRtModel(model, inputBuffers, outputBuffers, boundedGpuRuntime)
                outputNhwc = outputBuffers.single().readFloat()
                totalStarted.elapsed().also {
                    warmupWall += it.wallMs
                    warmupCpu += it.cpuMs
                }
            }
            repeat(iterations) {
                val totalStarted = timedStart()
                val dispatchStarted = timedStart()
                runLiteRtModel(model, inputBuffers, outputBuffers, boundedGpuRuntime)
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

            BackendResult(
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
                backendEvidence = when {
                    useQnn -> qnnEvidence(
                        requireNotNull(npuProvider),
                        requireNotNull(qnnEvidenceDir),
                        qnnProfiling,
                    )
                    boundedGpu -> requireNotNull(boundedGpuRuntime).evidence()
                    else -> null
                },
                output = nhwcToNchw(
                    requireNotNull(outputNhwc) { "LiteRT produced no output." },
                    height,
                    width,
                ),
            )
        } finally {
            inputBuffers.closeAll()
            outputBuffers.closeAll()
            modelToClose?.close()
            environment.close()
        }
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
        return digest.digest().joinToString("") { byte -> "%02x".format(byte) }
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
        publishReport(File(reportsDir(), LATEST_REPORT), report)
        publishReport(reportFile, report)
    }

    private fun publishReport(file: File, report: JSONObject) {
        val partial = File(requireNotNull(file.parentFile), "${file.name}.partial")
        partial.writeText(report.toString(2))
        Files.move(
            partial.toPath(),
            file.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }

    private fun benchmarkDir(): File = File(getExternalFilesDir(null) ?: filesDir, "benchmark")
        .apply { mkdirs() }

    private fun reportsDir(): File = File(benchmarkDir(), "reports").apply { mkdirs() }

    private fun referenceDir(): File = File(benchmarkDir(), "reference").apply { mkdirs() }

    private fun tensorOutputDir(): File = File(benchmarkDir(), "tensor-output").apply { mkdirs() }

    private fun audioInputsDir(): File = File(benchmarkDir(), "audio-input").apply { mkdirs() }

    private fun audioOutputsDir(): File = File(benchmarkDir(), "audio-output").apply { mkdirs() }

    private fun inputsDir(): File = File(benchmarkDir(), "inputs").apply { mkdirs() }

    private fun qnnEvidenceRoot(): File = File(benchmarkDir(), "qnn").apply { mkdirs() }

    private fun qnnEvidence(
        provider: BuiltinNpuAcceleratorProvider,
        evidenceDir: File,
        detailedProfiling: Boolean,
    ): JSONObject = JSONObject()
        .put("provider", "BuiltinNpuAcceleratorProvider")
        .put("compatibilityChecker", "Qualcomm")
        .put("deviceSupported", provider.isDeviceSupported())
        .put("libraryReady", provider.isLibraryReady())
        .put("libraryDir", provider.getLibraryDir())
        .put("socManufacturer", Build.SOC_MANUFACTURER)
        .put("socModel", Build.SOC_MODEL)
        .put("htpPerformanceMode", "SUSTAINED_HIGH_PERFORMANCE")
        .put("optimizationLevel", "HTP_OPTIMIZE_FOR_INFERENCE")
        .put("profiling", if (detailedProfiling) "DETAILED" else "OFF")
        .put("irJsonDir", evidenceDir.absolutePath)
        .put("irFiles", JSONArray(
            evidenceDir.walkTopDown()
                .filter { it.isFile }
                .map { file ->
                    JSONObject()
                        .put("path", file.relativeTo(evidenceDir).invariantSeparatorsPath)
                        .put("bytes", file.length())
                }
                .toList(),
        ))

    private fun runLiteRtModel(
        model: CompiledModel,
        inputBuffers: List<TensorBuffer>,
        outputBuffers: List<TensorBuffer>,
        boundedGpuRuntime: BoundedGpuRuntime?,
    ) {
        boundedGpuRuntime?.beginInference()
        try {
            model.run(inputBuffers, outputBuffers)
        } finally {
            boundedGpuRuntime?.endInference()
        }
    }

    private class BoundedGpuRuntime private constructor(
        private val runtimeClass: Class<*>,
        private val capability: Any,
    ) {
        fun resetInferenceCounters() = invokeStatic("resetInferenceCounters")

        fun beginInference() = invokeStatic("beginInference")

        fun endInference() = invokeStatic("endInference")

        fun evidence(): JSONObject = JSONObject()
            .put("artifactVersion", capabilityValue<String>("getArtifactVersion"))
            .put("profileId", capabilityValue<String>("getProfileId"))
            .put("schemaVersion", capabilityValue<Int>("getSchemaVersion"))
            .put("kernelBatchSize", capabilityValue<Int>("getKernelBatchSize"))
            .put("commandQueueWindowSize", capabilityValue<Int>("getCommandQueueWindowSize"))
            .put("dispatchCount", invokeStatic("getDispatchCount") as Long)
            .put("eventWaitCount", invokeStatic("getEventWaitCount") as Long)

        private fun invokeStatic(name: String): Any? = runtimeClass.getMethod(name).invoke(null)

        @Suppress("UNCHECKED_CAST")
        private fun <T> capabilityValue(name: String): T =
            capability.javaClass.getMethod(name).invoke(capability) as T

        companion object {
            private const val RUNTIME_CLASS = "io.github.wluhwluh.bss.litert.BssLiteRtRuntime"

            fun loadAndValidate(): BoundedGpuRuntime {
                val runtimeClass = runCatching { Class.forName(RUNTIME_CLASS) }.getOrElse { error ->
                    throw IllegalStateException(
                        "The bounded GPU backend requires the 2.1.5-bss.2 runtime AAR.",
                        error,
                    )
                }
                val capability = requireNotNull(
                    runtimeClass.getMethod("queryCapability").invoke(null),
                ) { "The bounded GPU runtime returned no capability." }
                fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)
                require(value("isAvailable") == true) { "The bounded GPU native runtime is unavailable." }
                require(value("getArtifactVersion") == "2.1.5-bss.2") {
                    "Unexpected bounded runtime artifact: ${value("getArtifactVersion")}"
                }
                require(value("getProfileId") == "gpu-opencl-bounded-fp32-v1") {
                    "Unexpected bounded GPU profile: ${value("getProfileId")}"
                }
                require(value("getKernelBatchSize") == 1 && value("getCommandQueueWindowSize") == 1) {
                    "The bounded GPU runtime does not expose the required N=1 contract."
                }
                return BoundedGpuRuntime(runtimeClass, capability)
            }
        }
    }

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

    private fun String.validatedFileName(): String = also { value ->
        require(value != "." && value != ".." && value.matches(Regex("[A-Za-z0-9._-]+"))) {
            "Benchmark file names may contain only ASCII letters, digits, dot, underscore, or hyphen: $value"
        }
    }

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
        val backendEvidence: JSONObject? = null,
        val output: FloatArray,
    )

    private enum class BenchmarkBackend(val id: String) {
        ORT("ort"),
        LITERT_CPU("litert_cpu"),
        LITERT_GPU("litert_gpu"),
        LITERT_GPU_FP32("litert_gpu_fp32"),
        LITERT_GPU_BOUNDED("litert_gpu_bounded"),
        LITERT_QNN("litert_qnn"),
        LITERT_QNN_AUDIO("litert_qnn_audio");

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
        const val EXTRA_QNN_PROFILING = "qnnProfiling"
        const val EXTRA_EXPORT_OUTPUT_TENSOR = "exportOutputTensor"
        const val EXTRA_AUDIO_FILE = "audioFile"
        const val EXTRA_MODEL_OUTPUT_SCALE = "modelOutputScale"

        const val DEFAULT_ONNX_MODEL = "UVR_MDXNET_9482.onnx"
        const val DEFAULT_LITERT_MODEL = "UVR_MDXNET_9482_float32.tflite"

        private const val BACKEND_INIT = "init"

        private const val DEFAULT_ITERATIONS = 5
        private const val MAX_ITERATIONS = 1_000
        private const val DEFAULT_WARMUPS = 1
        private const val DEFAULT_THREADS = 8
        private const val DEFAULT_SEED = 9482L
        private const val DEFAULT_MODEL_OUTPUT_SCALE = 1.035f
        private const val QNN_AUDIO_MODEL_ID = "uvr_mdxnet_3_9662"
        private const val LATEST_REPORT = "latest.json"
        private const val NOTIFICATION_CHANNEL_ID = "inference_benchmark"
        private const val NOTIFICATION_ID = 9482
        private const val QNN_LOG_TAG = "MSS-QNN"

        private const val BATCH = 1
        private const val CHANNELS = 4
        private const val DEFAULT_HEIGHT = 2048
        private const val DEFAULT_WIDTH = 256
    }
}
