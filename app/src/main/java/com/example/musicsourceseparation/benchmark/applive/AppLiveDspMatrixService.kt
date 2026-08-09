package com.example.musicsourceseparation.benchmark.applive

import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.IBinder
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import com.example.musicsourceseparation.MainActivity
import com.example.musicsourceseparation.model.MdxDspConfig
import com.example.musicsourceseparation.model.MdxSpectrogram
import com.example.musicsourceseparation.model.NativeMdxDsp
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.log10
import kotlin.math.sin
import kotlin.math.sqrt

class AppLiveDspMatrixService : Service() {
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action != ACTION_RUN) return START_NOT_STICKY
        startForeground(NOTIFICATION_ID, notification("Preparing DSP matrix"))
        if (!running.compareAndSet(false, true)) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        thread(name = "app-live-dsp-matrix") {
            try {
                execute()
            } finally {
                running.set(false)
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf(startId)
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun execute() {
        var bundle: AppLiveDspMatrixBundle? = null
        var relay: AppLiveRelayClient? = null
        var runId = ""
        var runDirectory: File? = null
        var logger: RunLogger? = null
        var logcat: AppLiveStageLogcatCapture? = null
        try {
            bundle = AppLiveDspMatrixBundle.load(this)
            relay = AppLiveRelayClient(bundle.relay)
            val origin = AppLiveValidationState.snapshot(this).origin
            runId = createRunId(origin, bundle.bundleId)
            runDirectory = File(appLiveRoot(), runId).apply {
                deleteRecursively()
                require(mkdirs()) { "Could not create run directory: $absolutePath" }
            }
            logger = RunLogger(File(runDirectory, "app.log"))
            logcat = AppLiveStageLogcatCapture.start(
                "dsp-matrix",
                File(runDirectory, "dsp-matrix-logcat.txt"),
            )
            logger.log("run=$runId bundle=${bundle.bundleId} origin=${origin.id}")
            updateState(runId, "Running contract-driven DSP matrix")
            val manifestFile = File(runDirectory, "artifact-manifest.json").apply {
                writeText(bundle.manifestText)
            }
            val identity = identity(runId, origin, bundle)
            val identityFile = File(runDirectory, "identity.json").apply {
                writeJsonAtomic(this, identity)
            }
            val report = runMatrix(runId, origin, bundle, logger)
            report.put("logcat", logcat.finish("dsp-matrix"))
            val reportFile = File(runDirectory, "dsp-matrix-report.json").apply {
                writeJsonAtomic(this, report)
            }
            val summary = AppLiveDspMatrixSummary.create(identity, report)
            require(summary.getBoolean("allRowsQualified")) { "DSP summary contains failed rows" }
            val summaryJsonFile = File(runDirectory, "dsp-matrix-summary.json").apply {
                writeJsonAtomic(this, summary)
            }
            val summaryCsvFile = File(runDirectory, "dsp-matrix-summary.csv").apply {
                writeText(AppLiveDspMatrixSummary.toCsv(summary), Charsets.UTF_8)
            }
            logger.log("matrix complete; beginning deferred relay upload")
            val completeFile = File(runDirectory, "complete.json").apply {
                writeJsonAtomic(
                    this,
                    JSONObject()
                        .put("schemaVersion", 1)
                        .put("status", "complete")
                        .put("profile", AppLiveValidationProfile.DSP_MATRIX.id)
                        .put("runId", runId)
                        .put("bundleId", bundle.bundleId)
                        .put("summaryRows", summary.getInt("rowCount"))
                        .put("uniformNativeWinner", summary.getString("uniformNativeWinner"))
                        .put("summaryJson", fileEvidence(summaryJsonFile))
                        .put("summaryCsv", fileEvidence(summaryCsvFile))
                        .put("completedAt", isoNow()),
                )
            }
            val uploads = listOf(
                manifestFile to "application/json",
                identityFile to "application/json",
                summaryJsonFile to "application/json",
                summaryCsvFile to "text/csv",
                reportFile to "application/json",
                logcat.file to "text/plain",
                logger.file to "text/plain",
                completeFile to "application/json",
            )
            uploads.forEach { (file, contentType) ->
                updateState(runId, "Uploading ${file.name}")
                relay.upload(runId, file.name, file, contentType)
            }
            AppLiveValidationState.update(
                this,
                state = "complete",
                running = false,
                profile = AppLiveValidationProfile.DSP_MATRIX,
                runId = runId,
                message = "DSP matrix complete and uploaded",
            )
            updateNotification("DSP matrix complete")
        } catch (error: Throwable) {
            logger?.log("failure=${error::class.java.name}: ${error.message}")
            val directory = runDirectory
            if (directory != null) {
                val logcatEvidence = runCatching { logcat?.finish("dsp-matrix") }.getOrNull()
                val failureFile = File(directory, "failure.json").apply {
                    writeJsonAtomic(
                        this,
                        JSONObject()
                            .put("schemaVersion", 1)
                            .put("status", "error")
                            .put("profile", AppLiveValidationProfile.DSP_MATRIX.id)
                            .put("runId", runId)
                            .put("bundleId", bundle?.bundleId ?: JSONObject.NULL)
                            .put("failedAt", isoNow())
                            .put("errorClass", error::class.java.name)
                            .put("message", error.message.orEmpty())
                            .put("stack", error.stackTraceToString())
                            .put("logcat", logcatEvidence ?: JSONObject.NULL),
                    )
                }
                if (relay != null && runId.isNotBlank()) {
                    listOfNotNull(
                        File(directory, "artifact-manifest.json").takeIf(File::isFile),
                        File(directory, "identity.json").takeIf(File::isFile),
                        File(directory, "dsp-matrix-summary.json").takeIf(File::isFile),
                        File(directory, "dsp-matrix-summary.csv").takeIf(File::isFile),
                        File(directory, "dsp-matrix-report.json").takeIf(File::isFile),
                        logger?.file?.takeIf(File::isFile),
                        logcat?.file?.takeIf(File::isFile),
                        failureFile,
                    ).forEach { file ->
                        runCatching { relay.upload(runId, file.name, file, contentType(file)) }
                    }
                }
            }
            AppLiveValidationState.update(
                this,
                state = "error",
                running = false,
                profile = AppLiveValidationProfile.DSP_MATRIX,
                runId = runId,
                message = error.message ?: error::class.java.simpleName,
            )
            updateNotification("DSP matrix failed")
        }
    }

    private fun runMatrix(
        runId: String,
        origin: AppLiveRunOrigin,
        bundle: AppLiveDspMatrixBundle,
        logger: RunLogger,
    ): JSONObject {
        val processStart = processEvidence()
        val deviceStart = deviceEvidence()
        val started = SystemClock.elapsedRealtimeNanos()
        val shapeResults = JSONArray()
        val totalShapeRuns = bundle.shapes.size * bundle.workerCounts.size
        var shapeRunIndex = 0
        bundle.workerCounts.forEach { workerCount ->
            bundle.shapes.forEach { shape ->
                shapeRunIndex++
                updateState(
                    "",
                    "Running ${shape.id} workers=$workerCount ($shapeRunIndex/$totalShapeRuns)",
                )
                logger.log("shape=${shape.id} workers=$workerCount starting")
                shapeResults.put(runShape(shape, workerCount, bundle, logger))
                logger.log("shape=${shape.id} workers=$workerCount complete")
            }
        }
        return JSONObject()
            .put("schemaVersion", 1)
            .put("status", "complete")
            .put("runId", runId)
            .put("origin", origin.id)
            .put("contractVersion", bundle.contractVersion)
            .put("bundleId", bundle.bundleId)
            .put("pocketfftRevision", POCKETFFT_REVISION)
            .put("profiles", JSONArray(bundle.profiles))
            .put("workerCounts", JSONArray(bundle.workerCounts))
            .put("warmups", bundle.warmups)
            .put("measuredRunsPerProfile", bundle.measuredRuns)
            .put("measurementOrder", "balanced-alternating-rounds")
            .put("minimumSnrDb", bundle.minimumSnrDb)
            .put("maximumAbsoluteError", bundle.maximumAbsoluteError)
            .put("elapsedMs", elapsedMs(started))
            .put("processStart", processStart)
            .put("processEnd", processEvidence())
            .put("deviceStart", deviceStart)
            .put("deviceEnd", deviceEvidence())
            .put("shapes", shapeResults)
    }

    private fun runShape(
        shape: AppLiveDspShape,
        workerCount: Int,
        bundle: AppLiveDspMatrixBundle,
        logger: RunLogger,
    ): JSONObject {
        val config = MdxDspConfig(
            nFft = shape.nFft,
            hopLength = shape.hopLength,
            dimF = shape.dimF,
            dimTPower = shape.dimTPower,
        )
        val waveform = fixture(config)
        val referenceTensor = FloatArray(config.tensorElementCount)
        val referenceWaveform = Array(2) { FloatArray(config.chunkSize) }
        val shapeStartMemory = processEvidence()
        val thermalStart = deviceEvidence()
        val runtimeBefore = runtimeStats()
        val engines = bundle.profiles.associateWithTo(linkedMapOf()) { profile ->
            when (profile) {
                PROFILE_KOTLIN -> KotlinDspEngine(config, workerCount)
                PROFILE_NATIVE_FULL -> NativeDspEngine(
                    config,
                    workerCount,
                    NativeMdxDsp.Mode.FULL_COMPLEX,
                )
                PROFILE_NATIVE_PACKED -> NativeDspEngine(
                    config,
                    workerCount,
                    NativeMdxDsp.Mode.PACKED_REAL,
                )
                else -> error("Unsupported DSP profile: $profile")
            }
        }
        try {
            val kotlin = engines.getValue(PROFILE_KOTLIN)
            kotlin.stft(waveform, referenceTensor)
            kotlin.iStft(referenceTensor, referenceWaveform)
            val parity = linkedMapOf<String, JSONObject>()
            val samples = bundle.profiles.associateWith { mutableListOf<RunSample>() }
            val candidateTensor = FloatArray(config.tensorElementCount)
            val candidateWaveform = Array(2) { FloatArray(config.chunkSize) }

            bundle.profiles.forEach { profile ->
                val engine = engines.getValue(profile)
                repeat(bundle.warmups) {
                    engine.stft(waveform, candidateTensor)
                    engine.iStft(referenceTensor, candidateWaveform)
                }
                engine.stft(waveform, candidateTensor)
                val stftStats = errorStats(referenceTensor, candidateTensor)
                engine.iStft(referenceTensor, candidateWaveform)
                val iStftStats = errorStats(referenceWaveform, candidateWaveform)
                require(stftStats.finite && stftStats.snrDb >= bundle.minimumSnrDb &&
                    stftStats.maxAbs <= bundle.maximumAbsoluteError
                ) { "$profile STFT parity failed for ${shape.id}: $stftStats" }
                require(iStftStats.finite && iStftStats.snrDb >= bundle.minimumSnrDb &&
                    iStftStats.maxAbs <= bundle.maximumAbsoluteError
                ) { "$profile iSTFT parity failed for ${shape.id}: $iStftStats" }
                parity[profile] = JSONObject()
                    .put("stft", stftStats.toJson())
                    .put("iStft", iStftStats.toJson())
            }

            repeat(bundle.measuredRuns) { cycle ->
                val order = if (cycle % 2 == 0) bundle.profiles else bundle.profiles.reversed()
                order.forEach { profile ->
                    val engine = engines.getValue(profile)
                    val sample = measuredRun(engine, waveform, referenceTensor, candidateTensor, candidateWaveform)
                    samples.getValue(profile).add(sample)
                    logger.log(
                        "shape=${shape.id} profile=$profile cycle=$cycle " +
                            "stftMs=${sample.stftWallMs} iStftMs=${sample.iStftWallMs}",
                    )
                }
            }
            val profiles = JSONObject()
            bundle.profiles.forEach { profile ->
                val values = samples.getValue(profile)
                require(values.size == bundle.measuredRuns)
                profiles.put(
                    profile,
                    JSONObject()
                        .put("parity", parity.getValue(profile))
                        .put("stft", timingSummary(values.map { it.stftWallMs }))
                        .put("iStft", timingSummary(values.map { it.iStftWallMs }))
                        .put("combined", timingSummary(values.map { it.totalWallMs }))
                        .put("processCpu", timingSummary(values.map { it.processCpuMs }))
                        .put("samples", JSONArray(values.map(RunSample::toJson))),
                )
            }
            val fastest = bundle.profiles.minBy { profile ->
                median(samples.getValue(profile).map { it.totalWallMs })
            }
            return JSONObject()
                .put("id", shape.id)
                .put("workerCount", workerCount)
                .put("config", JSONObject()
                    .put("nFft", config.nFft)
                    .put("hopLength", config.hopLength)
                    .put("dimF", config.dimF)
                    .put("dimT", config.dimT)
                    .put("chunkSize", config.chunkSize)
                    .put("tensorElements", config.tensorElementCount))
                .put("fixture", JSONObject()
                    .put("kind", "deterministic-stereo-multisine-v1")
                    .put("waveformSha256", sha256(waveform))
                    .put("referenceTensorSha256", sha256(referenceTensor)))
                .put("fastestCombinedMedianProfile", fastest)
                .put("profiles", profiles)
                .put("runtimeStatsDelta", runtimeStatsDelta(runtimeBefore, runtimeStats()))
                .put("processStart", shapeStartMemory)
                .put("processEnd", processEvidence())
                .put("deviceStart", thermalStart)
                .put("deviceEnd", deviceEvidence())
        } finally {
            engines.values.reversed().forEach(DspEngine::close)
        }
    }

    private fun measuredRun(
        engine: DspEngine,
        waveform: Array<FloatArray>,
        referenceTensor: FloatArray,
        candidateTensor: FloatArray,
        candidateWaveform: Array<FloatArray>,
    ): RunSample {
        val totalStarted = SystemClock.elapsedRealtimeNanos()
        val cpuStarted = Process.getElapsedCpuTime()
        val stftStarted = SystemClock.elapsedRealtimeNanos()
        engine.stft(waveform, candidateTensor)
        val stftMs = elapsedMs(stftStarted)
        val iStftStarted = SystemClock.elapsedRealtimeNanos()
        engine.iStft(referenceTensor, candidateWaveform)
        val iStftMs = elapsedMs(iStftStarted)
        return RunSample(
            stftWallMs = stftMs,
            iStftWallMs = iStftMs,
            totalWallMs = elapsedMs(totalStarted),
            processCpuMs = (Process.getElapsedCpuTime() - cpuStarted).toDouble(),
        )
    }

    private fun identity(
        runId: String,
        origin: AppLiveRunOrigin,
        bundle: AppLiveDspMatrixBundle,
    ): JSONObject {
        val packageInfo = packageManager.getPackageInfo(packageName, 0)
        val memory = ActivityManager.MemoryInfo().also {
            getSystemService(ActivityManager::class.java).getMemoryInfo(it)
        }
        val nativeLibraries = File(applicationInfo.nativeLibraryDir).listFiles().orEmpty()
            .filter { it.isFile && it.extension == "so" }
            .sortedBy(File::getName)
            .map { file -> fileEvidence(file) }
        val apkFiles = buildList {
            add(applicationInfo.sourceDir)
            addAll(applicationInfo.splitSourceDirs.orEmpty())
        }.distinct().map { fileEvidence(File(it)) }
        return JSONObject()
            .put("schemaVersion", 1)
            .put("capturedAt", isoNow())
            .put("runId", runId)
            .put("origin", origin.id)
            .put("bundleId", bundle.bundleId)
            .put("contractVersion", bundle.contractVersion)
            .put("sourceCommit", bundle.sourceCommit)
            .put("sourceDirty", bundle.sourceDirty)
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("brand", Build.BRAND)
                .put("model", Build.MODEL)
                .put("product", Build.PRODUCT)
                .put("device", Build.DEVICE)
                .put("board", Build.BOARD)
                .put("hardware", Build.HARDWARE)
                .put("socManufacturer", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else "")
                .put("socModel", if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "")
                .put("fingerprint", Build.FINGERPRINT)
                .put("androidRelease", Build.VERSION.RELEASE)
                .put("sdk", Build.VERSION.SDK_INT)
                .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
                .put("processAbi", processAbi())
                .put("is64Bit", Process.is64Bit())
                .put("totalMemoryBytes", memory.totalMem)
                .put("lowRamDevice", getSystemService(ActivityManager::class.java).isLowRamDevice))
            .put("application", JSONObject()
                .put("package", packageName)
                .put("versionName", packageInfo.versionName)
                .put("versionCode", packageInfo.longVersionCode)
                .put("apkFiles", JSONArray(apkFiles))
                .put("nativeLibraryDir", applicationInfo.nativeLibraryDir)
                .put("nativeLibraries", JSONArray(nativeLibraries)))
            .put("nativeDsp", JSONObject()
                .put("library", "mss_mdx_dsp")
                .put("fft", "pocketfft")
                .put("pocketfftRevision", POCKETFFT_REVISION)
                .put("profiles", JSONArray(bundle.profiles)))
    }

    private fun processEvidence(): JSONObject {
        val memory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        return JSONObject()
            .put("elapsedRealtimeMs", SystemClock.elapsedRealtime())
            .put("processCpuMs", Process.getElapsedCpuTime())
            .put("totalPssKb", memory.totalPss)
            .put("totalPrivateDirtyKb", memory.totalPrivateDirty)
            .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun processAbi(): String {
        val candidates = if (Process.is64Bit()) Build.SUPPORTED_64_BIT_ABIS else Build.SUPPORTED_32_BIT_ABIS
        return candidates.firstOrNull().orEmpty()
    }

    private fun deviceEvidence(): JSONObject {
        val power = getSystemService(PowerManager::class.java)
        val batteryManager = getSystemService(BatteryManager::class.java)
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return JSONObject()
            .put("elapsedRealtimeMs", SystemClock.elapsedRealtime())
            .put("thermalStatus", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else -1)
            .put("batteryLevelPercent", battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1)
            .put(
                "batteryTemperatureDeciC",
                battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1,
            )
            .put(
                "chargeCounterUah",
                batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            )
            .put(
                "energyCounterNwh",
                batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER),
            )
    }

    private fun runtimeStats(): Map<String, Long> = RUNTIME_STATS.associateWith { key ->
        Debug.getRuntimeStat(key)?.toLongOrNull() ?: 0L
    }

    private fun runtimeStatsDelta(before: Map<String, Long>, after: Map<String, Long>): JSONObject =
        JSONObject(after.mapValues { (key, value) -> value - before.getValue(key) })

    private fun timingSummary(values: List<Double>): JSONObject {
        require(values.isNotEmpty())
        val sorted = values.sorted()
        val p95 = sorted[(ceil(sorted.size * 0.95).toInt() - 1).coerceIn(sorted.indices)]
        return JSONObject()
            .put("count", values.size)
            .put("meanMs", values.average())
            .put("medianMs", median(values))
            .put("p95Ms", p95)
            .put("minMs", sorted.first())
            .put("maxMs", sorted.last())
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    private fun errorStats(reference: FloatArray, candidate: FloatArray): ErrorStats {
        require(reference.size == candidate.size)
        var signal = 0.0
        var error = 0.0
        var maxAbs = 0.0
        var finite = true
        var bitExact = true
        for (index in reference.indices) {
            val expected = reference[index].toDouble()
            val actual = candidate[index].toDouble()
            finite = finite && expected.isFinite() && actual.isFinite()
            val delta = actual - expected
            signal += expected * expected
            error += delta * delta
            maxAbs = maxOf(maxAbs, abs(delta))
            bitExact = bitExact && reference[index].toRawBits() == candidate[index].toRawBits()
        }
        val snr = if (error == 0.0) Double.POSITIVE_INFINITY else 10.0 * log10(signal / error)
        return ErrorStats(finite, snr, maxAbs, sqrt(error / reference.size), bitExact)
    }

    private fun errorStats(
        reference: Array<FloatArray>,
        candidate: Array<FloatArray>,
    ): ErrorStats {
        require(reference.size == candidate.size)
        var signal = 0.0
        var error = 0.0
        var maxAbs = 0.0
        var finite = true
        var bitExact = true
        var count = 0L
        for (channel in reference.indices) {
            require(reference[channel].size == candidate[channel].size)
            for (index in reference[channel].indices) {
                val expected = reference[channel][index].toDouble()
                val actual = candidate[channel][index].toDouble()
                finite = finite && expected.isFinite() && actual.isFinite()
                val delta = actual - expected
                signal += expected * expected
                error += delta * delta
                maxAbs = maxOf(maxAbs, abs(delta))
                bitExact = bitExact &&
                    reference[channel][index].toRawBits() == candidate[channel][index].toRawBits()
                count += 1
            }
        }
        val snr = if (error == 0.0) Double.POSITIVE_INFINITY else 10.0 * log10(signal / error)
        return ErrorStats(finite, snr, maxAbs, sqrt(error / count), bitExact)
    }

    private fun fixture(config: MdxDspConfig): Array<FloatArray> = Array(2) { channel ->
        FloatArray(config.chunkSize) { index ->
            val first = sin(2.0 * PI * (220 + channel * 37) * index / config.sampleRate)
            val second = sin(2.0 * PI * (997 + channel * 53) * index / config.sampleRate)
            val deterministicNoise = (((index * 37L + channel * 17L) % 101L) - 50L) / 50.0
            (0.10 * first + 0.03 * second + 0.002 * deterministicNoise).toFloat()
        }
    }

    private fun sha256(values: Array<FloatArray>): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        val bytes = ByteArray(4)
        values.forEach { channel ->
            channel.forEach { value ->
                val bits = value.toRawBits()
                bytes[0] = bits.toByte()
                bytes[1] = (bits ushr 8).toByte()
                bytes[2] = (bits ushr 16).toByte()
                bytes[3] = (bits ushr 24).toByte()
                digest.update(bytes)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sha256(values: FloatArray): String = sha256(arrayOf(values))

    private fun fileEvidence(file: File): JSONObject = JSONObject()
        .put("name", file.name)
        .put("bytes", file.length())
        .put("sha256", AppLiveHashing.sha256(file))

    private fun updateState(runId: String, message: String) {
        val currentRunId = runId.ifBlank { AppLiveValidationState.snapshot(this).runId }
        AppLiveValidationState.update(
            this,
            state = "running",
            running = true,
            profile = AppLiveValidationProfile.DSP_MATRIX,
            runId = currentRunId,
            message = message,
        )
        updateNotification(message)
    }

    private fun createRunId(origin: AppLiveRunOrigin, bundleId: String): String {
        val model = Build.MODEL.lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), "-")
            .trim('-')
            .ifBlank { "android" }
        val formatter = SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        return "${origin.id}-$model-dsp-matrix-${formatter.format(Date())}-${bundleId.take(8)}".take(96)
    }

    private fun appLiveRoot(): File = File(getExternalFilesDir(null) ?: filesDir, "app-live-dsp")
        .apply { mkdirs() }

    private fun writeJsonAtomic(file: File, value: JSONObject) {
        file.parentFile?.mkdirs()
        val partial = File(requireNotNull(file.parentFile), "${file.name}.partial")
        partial.writeText(value.toString(2))
        Files.move(
            partial.toPath(),
            file.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }

    private fun contentType(file: File): String = when (file.extension.lowercase(Locale.US)) {
        "json" -> "application/json"
        "txt", "log" -> "text/plain"
        else -> "application/octet-stream"
    }

    private fun elapsedMs(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    private fun isoNow(): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        return formatter.format(Date())
    }

    private fun notification(message: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("App Live DSP matrix")
            .setContentText(message)
            .setContentIntent(pendingIntent)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(message: String) {
        startForeground(NOTIFICATION_ID, notification(message))
    }

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "App Live DSP matrix",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    private interface DspEngine : AutoCloseable {
        fun stft(waveform: Array<FloatArray>, tensor: FloatArray)
        fun iStft(tensor: FloatArray, waveform: Array<FloatArray>)
    }

    private class KotlinDspEngine(config: MdxDspConfig, workers: Int) : DspEngine {
        private val dsp = MdxSpectrogram(config, workers)
        override fun stft(waveform: Array<FloatArray>, tensor: FloatArray) =
            dsp.waveformToNhwcTensorInto(waveform, tensor)
        override fun iStft(tensor: FloatArray, waveform: Array<FloatArray>) =
            dsp.nhwcTensorToWaveformInto(tensor, waveform)
        override fun close() = dsp.close()
    }

    private class NativeDspEngine(
        config: MdxDspConfig,
        workers: Int,
        mode: NativeMdxDsp.Mode,
    ) : DspEngine {
        private val dsp = NativeMdxDsp(config, workers, mode)
        override fun stft(waveform: Array<FloatArray>, tensor: FloatArray) =
            dsp.waveformToNhwcTensorInto(waveform, tensor)
        override fun iStft(tensor: FloatArray, waveform: Array<FloatArray>) =
            dsp.nhwcTensorToWaveformInto(tensor, waveform)
        override fun close() = dsp.close()
    }

    private data class RunSample(
        val stftWallMs: Double,
        val iStftWallMs: Double,
        val totalWallMs: Double,
        val processCpuMs: Double,
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("stftWallMs", stftWallMs)
            .put("iStftWallMs", iStftWallMs)
            .put("totalWallMs", totalWallMs)
            .put("processCpuMs", processCpuMs)
    }

    private data class ErrorStats(
        val finite: Boolean,
        val snrDb: Double,
        val maxAbs: Double,
        val rmsError: Double,
        val bitExact: Boolean,
    ) {
        fun toJson(): JSONObject = JSONObject()
            .put("finite", finite)
            .put("snrDb", if (snrDb.isFinite()) snrDb else "Infinity")
            .put("maxAbs", maxAbs)
            .put("rmsError", rmsError)
            .put("bitExact", bitExact)
    }

    private class RunLogger(val file: File) {
        @Synchronized
        fun log(message: String) {
            file.parentFile?.mkdirs()
            file.appendText("${System.currentTimeMillis()} $message\n")
        }
    }

    companion object {
        private const val ACTION_RUN = "com.example.musicsourceseparation.APP_LIVE_DSP_MATRIX"
        private const val NOTIFICATION_CHANNEL_ID = "app_live_dsp_matrix"
        private const val NOTIFICATION_ID = 6144
        private const val POCKETFFT_REVISION = "c90e55b3d529f8efa40ed01a20de22405f45fc65"
        private const val PROFILE_KOTLIN = "kotlin-jtransforms"
        private const val PROFILE_NATIVE_FULL = "native-full"
        private const val PROFILE_NATIVE_PACKED = "native-packed"
        private val RUNTIME_STATS = listOf(
            "art.gc.gc-count",
            "art.gc.blocking-gc-count",
            "art.gc.bytes-allocated",
            "art.gc.bytes-freed",
        )
        private val running = AtomicBoolean(false)

        internal fun isRunning(): Boolean = running.get()

        internal fun start(context: Context) {
            context.startForegroundService(
                Intent(ACTION_RUN, null, context, AppLiveDspMatrixService::class.java),
            )
        }
    }
}
