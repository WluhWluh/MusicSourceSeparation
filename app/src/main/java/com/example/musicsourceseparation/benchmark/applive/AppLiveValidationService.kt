package com.example.musicsourceseparation.benchmark.applive

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Debug
import android.os.IBinder
import android.os.Process
import com.example.musicsourceseparation.MainActivity
import com.example.musicsourceseparation.benchmark.InferenceBenchmarkService
import com.example.musicsourceseparation.benchmark.QnnDelegationAssessment
import com.example.musicsourceseparation.benchmark.QnnDelegationEvidence
import com.example.musicsourceseparation.benchmark.QnnDelegationStatus
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class AppLiveValidationService : Service() {
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val profile = when (intent?.action) {
            ACTION_QUICK -> AppLiveValidationProfile.QUICK
            ACTION_FULL -> AppLiveValidationProfile.FULL
            else -> return START_NOT_STICKY
        }
        startForeground(NOTIFICATION_ID, notification("Preparing ${profile.displayName}"))
        if (!running.compareAndSet(false, true)) {
            stopSelf(startId)
            return START_NOT_STICKY
        }
        thread(name = "app-live-${profile.id}") {
            try {
                execute(profile)
            } finally {
                running.set(false)
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf(startId)
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun execute(profile: AppLiveValidationProfile) {
        var relay: AppLiveRelayClient? = null
        var runId = ""
        var runDirectory: File? = null
        var logger: RunLogger? = null
        val uploadedFiles = mutableListOf<String>()
        try {
            val loadedBundle = AppLiveValidationBundle.load(this)
            val activeRelay = AppLiveRelayClient(loadedBundle.relay)
            relay = activeRelay
            val origin = AppLiveValidationState.snapshot(this).origin
            val uploadLargeArtifacts = origin == AppLiveRunOrigin.BROWSERSTACK
            runId = createRunId(origin, profile, loadedBundle.bundleId)
            val activeRunDirectory = File(appLiveRoot(), runId).apply {
                deleteRecursively()
                require(mkdirs()) { "Could not create run directory: $absolutePath" }
            }
            runDirectory = activeRunDirectory
            val activeLogger = RunLogger(File(activeRunDirectory, "app.log"))
            logger = activeLogger
            updateState(profile, runId, "Staging and verifying packaged fixtures")
            activeLogger.log("run=$runId profile=${profile.id} bundle=${loadedBundle.bundleId}")
            val staged = stageFixtures(loadedBundle, profile, activeRunDirectory, activeLogger)

            val manifestFile = File(activeRunDirectory, "artifact-manifest.json").apply {
                writeText(loadedBundle.manifestText)
            }
            val identityFile = File(activeRunDirectory, "identity.json").apply {
                writeJsonAtomic(
                    this,
                    identity(profile, runId, loadedBundle, staged, uploadLargeArtifacts),
                )
            }
            upload(
                activeRelay,
                runId,
                "identity.json",
                identityFile,
                "application/json",
                uploadedFiles,
                activeLogger,
            )
            upload(
                activeRelay,
                runId,
                "artifact-manifest.json",
                manifestFile,
                "application/json",
                uploadedFiles,
                activeLogger,
            )
            upload(
                activeRelay,
                runId,
                "runtime-manifest.json",
                staged.getValue("runtimeManifest"),
                "application/json",
                uploadedFiles,
                activeLogger,
            )
            checkpoint(
                activeRelay,
                runId,
                activeRunDirectory,
                "checkpoint-01-fixtures.json",
                JSONObject()
                    .put("status", "complete")
                    .put("stage", "fixtures")
                    .put("files", stagedEvidence(staged)),
                uploadedFiles,
                activeLogger,
            )

            when (profile) {
                AppLiveValidationProfile.QUICK -> runQuickSuite(
                    runId,
                    activeRunDirectory,
                    activeRelay,
                    loadedBundle,
                    uploadLargeArtifacts,
                    uploadedFiles,
                    activeLogger,
                )
                AppLiveValidationProfile.FULL -> runFullSuite(
                    runId,
                    activeRunDirectory,
                    activeRelay,
                    loadedBundle,
                    uploadLargeArtifacts,
                    uploadedFiles,
                    activeLogger,
                )
                AppLiveValidationProfile.DSP_MATRIX -> error(
                    "DSP matrix must run in AppLiveDspMatrixService",
                )
            }

            activeLogger.log("validation stages complete")
            upload(
                activeRelay,
                runId,
                "app.log",
                activeLogger.file,
                "text/plain",
                uploadedFiles,
                activeLogger,
                logSuccess = false,
            )
            val completeFile = File(activeRunDirectory, "complete.json").apply {
                writeJsonAtomic(
                    this,
                    JSONObject()
                        .put("schemaVersion", 1)
                        .put("status", "complete")
                        .put("profile", profile.id)
                        .put("runId", runId)
                        .put("bundleId", loadedBundle.bundleId)
                        .put("completedAt", isoNow())
                        .put("uploadedFiles", JSONArray(uploadedFiles)),
                )
            }
            upload(
                activeRelay,
                runId,
                "complete.json",
                completeFile,
                "application/json",
                uploadedFiles,
                activeLogger,
                logSuccess = false,
            )
            AppLiveValidationState.update(
                this,
                state = "complete",
                running = false,
                profile = profile,
                runId = runId,
                message = "Completed and uploaded ${profile.displayName}",
            )
            updateNotification("Completed ${profile.displayName}")
        } catch (error: Throwable) {
            logger?.log(error.stackTraceToString())
            val failureDirectory = runDirectory
            val failureRelay = relay
            if (failureDirectory != null && failureRelay != null && runId.isNotBlank()) {
                val failureFile = File(failureDirectory, "failure.json").apply {
                    writeJsonAtomic(
                        this,
                        JSONObject()
                            .put("schemaVersion", 1)
                            .put("status", "error")
                            .put("profile", profile.id)
                            .put("runId", runId)
                            .put("failedAt", isoNow())
                            .put("message", error.message ?: error::class.java.name)
                            .put("stackTrace", error.stackTraceToString()),
                    )
                }
                runCatching {
                    logger?.file?.let { logFile ->
                        failureRelay.upload(runId, "app-failure.log", logFile, "text/plain")
                    }
                    failureRelay.upload(runId, "failure.json", failureFile, "application/json")
                }
            }
            AppLiveValidationState.update(
                this,
                state = "error",
                running = false,
                profile = profile,
                runId = runId,
                message = error.message ?: error::class.java.simpleName,
            )
            updateNotification("Validation failed")
        }
    }

    private fun runQuickSuite(
        runId: String,
        runDirectory: File,
        relay: AppLiveRelayClient,
        bundle: AppLiveValidationBundle,
        uploadLargeArtifacts: Boolean,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
    ) {
        runTensorStage(
            profile = AppLiveValidationProfile.QUICK,
            runId = runId,
            runDirectory = runDirectory,
            relay = relay,
            stage = "stage-10-qnn-window",
            backend = "litert_qnn",
            iterations = 3,
            exportOutput = uploadLargeArtifacts,
            uploadedFiles = uploadedFiles,
            logger = logger,
        )
        runTensorStage(
            profile = AppLiveValidationProfile.QUICK,
            runId = runId,
            runDirectory = runDirectory,
            relay = relay,
            stage = "stage-20-cpu-window",
            backend = "litert_cpu",
            iterations = 3,
            exportOutput = false,
            uploadedFiles = uploadedFiles,
            logger = logger,
        )
        runTensorStage(
            profile = AppLiveValidationProfile.QUICK,
            runId = runId,
            runDirectory = runDirectory,
            relay = relay,
            stage = "stage-30-bounded-gpu-window",
            backend = "litert_gpu_bounded",
            iterations = 5,
            exportOutput = false,
            uploadedFiles = uploadedFiles,
            logger = logger,
        )
        runAudioStage(
            profile = AppLiveValidationProfile.QUICK,
            runId = runId,
            runDirectory = runDirectory,
            relay = relay,
            stage = "stage-40-qnn-audio-12s",
            audioName = bundle.requireFixture("quickAudio").targetName,
            uploadedFiles = uploadedFiles,
            logger = logger,
            timeoutMinutes = 10,
            uploadAudioOutputs = true,
            expectedFrames = 529_200L,
            expectedWindows = 3,
        )
    }

    private fun runFullSuite(
        runId: String,
        runDirectory: File,
        relay: AppLiveRelayClient,
        bundle: AppLiveValidationBundle,
        uploadLargeArtifacts: Boolean,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
    ) {
        runTensorStage(
            profile = AppLiveValidationProfile.FULL,
            runId = runId,
            runDirectory = runDirectory,
            relay = relay,
            stage = "stage-10-qnn-window",
            backend = "litert_qnn",
            iterations = 3,
            exportOutput = uploadLargeArtifacts,
            uploadedFiles = uploadedFiles,
            logger = logger,
        )
        runAudioStage(
            profile = AppLiveValidationProfile.FULL,
            runId = runId,
            runDirectory = runDirectory,
            relay = relay,
            stage = "stage-50-qnn-audio-full",
            audioName = bundle.requireFixture("fullAudio").targetName,
            uploadedFiles = uploadedFiles,
            logger = logger,
            timeoutMinutes = 20,
            uploadAudioOutputs = uploadLargeArtifacts,
            expectedFrames = 12_070_130L,
            expectedWindows = 48,
        )
    }

    private fun runTensorStage(
        profile: AppLiveValidationProfile,
        runId: String,
        runDirectory: File,
        relay: AppLiveRelayClient,
        stage: String,
        backend: String,
        iterations: Int,
        exportOutput: Boolean,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
    ): JSONObject {
        updateState(profile, runId, "Running $backend")
        updateNotification("Running $backend")
        logger.log("starting stage=$stage backend=$backend")
        val tag = "$runId-${stage.removePrefix("stage-")}".take(120)
        val logcatCapture = AppLiveStageLogcatCapture.start(
            stage,
            File(runDirectory, "$stage-logcat.txt"),
        )
        var execution: BenchmarkExecution? = null
        var report: JSONObject? = null
        var delegation: QnnDelegationAssessment? = null
        var logcatEvidence: JSONObject? = null
        val request = Intent(InferenceBenchmarkService.ACTION_RUN, null, this, InferenceBenchmarkService::class.java)
            .putExtra(InferenceBenchmarkService.EXTRA_BACKEND, backend)
            .putExtra(InferenceBenchmarkService.EXTRA_ITERATIONS, iterations)
            .putExtra(InferenceBenchmarkService.EXTRA_WARMUPS, 1)
            .putExtra(InferenceBenchmarkService.EXTRA_THREADS, 4)
            .putExtra(InferenceBenchmarkService.EXTRA_SEED, 9482L)
            .putExtra(InferenceBenchmarkService.EXTRA_TAG, tag)
            .putExtra(InferenceBenchmarkService.EXTRA_MODEL_ID, "uvr_mdxnet_3_9662")
            .putExtra(InferenceBenchmarkService.EXTRA_HEIGHT, 2048)
            .putExtra(InferenceBenchmarkService.EXTRA_WIDTH, 256)
            .putExtra(
                InferenceBenchmarkService.EXTRA_INPUT_FILE,
                "uvr_mdxnet_3_9662_coast_town_window0_nchw_f32.bin",
            )
            .putExtra(
                InferenceBenchmarkService.EXTRA_LITERT_MODEL,
                "UVR_MDXNET_3_9662_static_float32.tflite",
            )
            .putExtra(InferenceBenchmarkService.EXTRA_QNN_PROFILING, false)
            .putExtra(InferenceBenchmarkService.EXTRA_EXPORT_OUTPUT_TENSOR, exportOutput)
        try {
            val completedExecution = executeBenchmark(request, tag, timeoutMinutes = 10)
            execution = completedExecution
            val completedReport = completedExecution.report
            report = completedReport
            val completedLogcatEvidence = logcatCapture.finish(stage)
            logcatEvidence = completedLogcatEvidence
            val completedDelegation = if (backend == "litert_qnn") {
                assessQnnDelegation(completedReport.optJSONObject("backendEvidence"))
            } else {
                null
            }
            delegation = completedDelegation
            annotateAppLiveExecution(completedReport, completedExecution, completedDelegation)
            publishBenchmarkArtifacts(
                runId,
                runDirectory,
                relay,
                stage,
                tag,
                completedReport,
                uploadedFiles,
                logger,
            )
            uploadStageLogcat(
                relay,
                runId,
                stage,
                logcatCapture.file,
                uploadedFiles,
                logger,
            )
            requireBenchmarkSucceeded(completedExecution)
            validateTensorReport(completedReport, backend)
            publishStageCheckpoint(
                relay = relay,
                runId = runId,
                runDirectory = runDirectory,
                stage = stage,
                backend = backend,
                status = "validated",
                execution = completedExecution,
                report = completedReport,
                delegation = completedDelegation,
                logcat = completedLogcatEvidence,
                failure = null,
                uploadedFiles = uploadedFiles,
                logger = logger,
            )
            return completedReport
        } catch (error: Throwable) {
            logcatEvidence = logcatEvidence ?: logcatCapture.finish(stage)
            runCatching {
                uploadStageLogcat(
                    relay,
                    runId,
                    stage,
                    logcatCapture.file,
                    uploadedFiles,
                    logger,
                )
            }.onFailure { logger.log("stage=$stage logcat upload failed: ${it.message}") }
            publishFailedStageCheckpoint(
                relay,
                runId,
                runDirectory,
                stage,
                backend,
                execution,
                report,
                delegation,
                requireNotNull(logcatEvidence),
                error,
                uploadedFiles,
                logger,
            )
            throw error
        }
    }

    private fun runAudioStage(
        profile: AppLiveValidationProfile,
        runId: String,
        runDirectory: File,
        relay: AppLiveRelayClient,
        stage: String,
        audioName: String,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
        timeoutMinutes: Long,
        uploadAudioOutputs: Boolean,
        expectedFrames: Long,
        expectedWindows: Int,
    ): JSONObject {
        updateState(profile, runId, "Running QNN audio separation")
        updateNotification("Running QNN audio separation")
        logger.log("starting stage=$stage audio=$audioName")
        val tag = "$runId-${stage.removePrefix("stage-")}".take(120)
        val backend = "litert_qnn_audio"
        val logcatCapture = AppLiveStageLogcatCapture.start(
            stage,
            File(runDirectory, "$stage-logcat.txt"),
        )
        var execution: BenchmarkExecution? = null
        var report: JSONObject? = null
        var delegation: QnnDelegationAssessment? = null
        var logcatEvidence: JSONObject? = null
        val request = Intent(InferenceBenchmarkService.ACTION_RUN, null, this, InferenceBenchmarkService::class.java)
            .putExtra(InferenceBenchmarkService.EXTRA_BACKEND, backend)
            .putExtra(InferenceBenchmarkService.EXTRA_TAG, tag)
            .putExtra(InferenceBenchmarkService.EXTRA_MODEL_ID, "uvr_mdxnet_3_9662")
            .putExtra(
                InferenceBenchmarkService.EXTRA_LITERT_MODEL,
                "UVR_MDXNET_3_9662_static_float32.tflite",
            )
            .putExtra(InferenceBenchmarkService.EXTRA_AUDIO_FILE, audioName)
            .putExtra(InferenceBenchmarkService.EXTRA_MODEL_OUTPUT_SCALE, 1.035f)
        try {
            val completedExecution = executeBenchmark(request, tag, timeoutMinutes)
            execution = completedExecution
            val completedReport = completedExecution.report
            report = completedReport
            val completedLogcatEvidence = logcatCapture.finish(stage)
            logcatEvidence = completedLogcatEvidence
            val completedDelegation = assessQnnDelegation(
                completedReport.optJSONObject("audio")?.optJSONObject("backendEvidence"),
            )
            delegation = completedDelegation
            annotateAppLiveExecution(completedReport, completedExecution, completedDelegation)
            publishBenchmarkArtifacts(
                runId,
                runDirectory,
                relay,
                stage,
                tag,
                completedReport,
                uploadedFiles,
                logger,
                uploadAudioOutputs = uploadAudioOutputs,
            )
            uploadStageLogcat(
                relay,
                runId,
                stage,
                logcatCapture.file,
                uploadedFiles,
                logger,
            )
            requireBenchmarkSucceeded(completedExecution)
            validateAudioReport(completedReport, expectedFrames, expectedWindows)
            publishStageCheckpoint(
                relay = relay,
                runId = runId,
                runDirectory = runDirectory,
                stage = stage,
                backend = backend,
                status = "validated",
                execution = completedExecution,
                report = completedReport,
                delegation = completedDelegation,
                logcat = completedLogcatEvidence,
                failure = null,
                uploadedFiles = uploadedFiles,
                logger = logger,
            )
            return completedReport
        } catch (error: Throwable) {
            logcatEvidence = logcatEvidence ?: logcatCapture.finish(stage)
            runCatching {
                uploadStageLogcat(
                    relay,
                    runId,
                    stage,
                    logcatCapture.file,
                    uploadedFiles,
                    logger,
                )
            }.onFailure { logger.log("stage=$stage logcat upload failed: ${it.message}") }
            publishFailedStageCheckpoint(
                relay,
                runId,
                runDirectory,
                stage,
                backend,
                execution,
                report,
                delegation,
                requireNotNull(logcatEvidence),
                error,
                uploadedFiles,
                logger,
            )
            throw error
        }
    }

    @SuppressLint("UnspecifiedRegisterReceiverFlag")
    private fun executeBenchmark(request: Intent, tag: String, timeoutMinutes: Long): BenchmarkExecution {
        val reportFile = File(reportsDirectory(), "$tag.json").apply { delete() }
        val latch = CountDownLatch(1)
        var broadcastSucceeded = false
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.getStringExtra(InferenceBenchmarkService.EXTRA_TAG) == tag) {
                    broadcastSucceeded = intent.getBooleanExtra(
                        InferenceBenchmarkService.EXTRA_SUCCEEDED,
                        false,
                    )
                    latch.countDown()
                }
            }
        }
        val filter = IntentFilter(InferenceBenchmarkService.ACTION_FINISHED)
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(
                receiver,
                filter,
                InferenceBenchmarkService.INTERNAL_BENCHMARK_PERMISSION,
                null,
                RECEIVER_NOT_EXPORTED,
            )
        } else {
            @Suppress("DEPRECATION")
            registerReceiver(
                receiver,
                filter,
                InferenceBenchmarkService.INTERNAL_BENCHMARK_PERMISSION,
                null,
            )
        }
        try {
            startForegroundService(request)
            if (!latch.await(timeoutMinutes, TimeUnit.MINUTES)) {
                stopService(Intent(this, InferenceBenchmarkService::class.java))
                error("Benchmark timed out after $timeoutMinutes minutes: $tag")
            }
        } finally {
            runCatching { unregisterReceiver(receiver) }
        }
        require(reportFile.isFile) { "Benchmark did not publish its report: $tag" }
        val report = JSONObject(reportFile.readText())
        return BenchmarkExecution(report, broadcastSucceeded)
    }

    private fun annotateAppLiveExecution(
        report: JSONObject,
        execution: BenchmarkExecution,
        delegation: QnnDelegationAssessment?,
    ) {
        report.put(
            "appLiveExecution",
            JSONObject()
                .put("broadcastSucceeded", execution.broadcastSucceeded)
                .put("benchmarkStatus", report.optString("status", "missing"))
                .put("delegation", delegation?.toJson() ?: JSONObject.NULL),
        )
    }

    private fun assessQnnDelegation(evidence: JSONObject?): QnnDelegationAssessment =
        evidence?.let(QnnDelegationEvidence::annotate) ?: QnnDelegationEvidence.assess(
            deviceSupported = null,
            libraryReady = null,
            irFilesDeclared = false,
            irFiles = emptyList(),
        )

    private fun requireBenchmarkSucceeded(execution: BenchmarkExecution) {
        require(execution.succeeded) {
            val message = execution.report.optString("message", execution.report.optString("tag", "unknown"))
                .lineSequence()
                .firstOrNull()
                .orEmpty()
                .take(1_000)
            "Benchmark failed: $message"
        }
    }

    private fun uploadStageLogcat(
        relay: AppLiveRelayClient,
        runId: String,
        stage: String,
        file: File,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
    ) {
        val name = "$stage-logcat.txt"
        if (name in uploadedFiles) return
        upload(relay, runId, name, file, "text/plain", uploadedFiles, logger)
    }

    private fun publishStageCheckpoint(
        relay: AppLiveRelayClient,
        runId: String,
        runDirectory: File,
        stage: String,
        backend: String,
        status: String,
        execution: BenchmarkExecution?,
        report: JSONObject?,
        delegation: QnnDelegationAssessment?,
        logcat: JSONObject,
        failure: Throwable?,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
        failureCheckpoint: Boolean = false,
    ) {
        val suffix = if (failureCheckpoint) "-failure" else ""
        val value = JSONObject()
            .put("schemaVersion", 1)
            .put("capturedAt", isoNow())
            .put("status", status)
            .put("stage", stage)
            .put("backend", backend)
            .put("broadcastSucceeded", execution?.broadcastSucceeded ?: JSONObject.NULL)
            .put("benchmarkStatus", report?.optString("status", "missing") ?: JSONObject.NULL)
            .put("delegation", delegation?.toJson() ?: JSONObject.NULL)
            .put("logcat", logcat)
        if (failure != null) {
            value.put(
                "failure",
                JSONObject()
                    .put("type", failure::class.java.name)
                    .put("message", (failure.message ?: failure::class.java.simpleName).take(4_000)),
            )
        }
        checkpoint(
            relay,
            runId,
            runDirectory,
            "checkpoint-${stage.removePrefix("stage-")}$suffix.json",
            value,
            uploadedFiles,
            logger,
        )
    }

    private fun publishFailedStageCheckpoint(
        relay: AppLiveRelayClient,
        runId: String,
        runDirectory: File,
        stage: String,
        backend: String,
        execution: BenchmarkExecution?,
        report: JSONObject?,
        delegation: QnnDelegationAssessment?,
        logcat: JSONObject,
        failure: Throwable,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
    ) {
        runCatching {
            publishStageCheckpoint(
                relay = relay,
                runId = runId,
                runDirectory = runDirectory,
                stage = stage,
                backend = backend,
                status = "error",
                execution = execution,
                report = report,
                delegation = delegation,
                logcat = logcat,
                failure = failure,
                uploadedFiles = uploadedFiles,
                logger = logger,
                failureCheckpoint = true,
            )
        }.onFailure { logger.log("stage=$stage failure checkpoint upload failed: ${it.message}") }
    }

    private fun publishBenchmarkArtifacts(
        runId: String,
        runDirectory: File,
        relay: AppLiveRelayClient,
        stage: String,
        tag: String,
        report: JSONObject,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
        uploadAudioOutputs: Boolean = true,
    ) {
        val reportCopy = File(runDirectory, "$stage-report.json").apply {
            writeJsonAtomic(this, report)
        }
        upload(
            relay,
            runId,
            "$stage-report.json",
            reportCopy,
            "application/json",
            uploadedFiles,
            logger,
        )
        val outputTensor = report.optJSONObject("outputTensor")
        if (outputTensor != null) {
            val path = outputTensor.optString("path")
            if (path.isNotBlank()) {
                upload(
                    relay,
                    runId,
                    "$stage-output.bin",
                    File(path),
                    "application/octet-stream",
                    uploadedFiles,
                    logger,
                )
            }
        }
        val evidenceDirectory = File(qnnEvidenceDirectory(), tag)
        evidenceDirectory.walkTopDown()
            .filter { it.isFile }
            .sortedBy { it.name }
            .forEach { file ->
                val evidenceName = "$stage-${file.name.replace('_', '-')}".take(96)
                upload(
                    relay,
                    runId,
                    evidenceName,
                    file,
                    contentType(file),
                    uploadedFiles,
                    logger,
                )
            }
        if (uploadAudioOutputs) {
            val audio = report.optJSONObject("audio")
            val outputs = audio?.optJSONObject("outputs")
            listOf("vocals", "instrumental").forEach { stem ->
                val path = outputs?.optJSONObject(stem)?.optString("path").orEmpty()
                if (path.isNotBlank()) {
                    upload(
                        relay,
                        runId,
                        "$stage-$stem.wav",
                        File(path),
                        "audio/wav",
                        uploadedFiles,
                        logger,
                    )
                }
            }
        }
    }

    private fun validateTensorReport(report: JSONObject, backend: String) {
        require(report.getJSONObject("output").getInt("nonFiniteCount") == 0) {
            "$backend produced non-finite output"
        }
        val comparison = requireNotNull(report.optJSONObject("comparisonToOrt")) {
            "$backend did not compare against the packaged ORT reference"
        }
        val minimumSnr = if (backend == "litert_qnn") 34.0 else 90.0
        require(comparison.getDouble("snrDb") >= minimumSnr) {
            "$backend SNR ${comparison.getDouble("snrDb")} dB is below $minimumSnr dB"
        }
        if (backend == "litert_qnn") {
            require(comparison.getDouble("maxAbsError") <= 0.11) { "QNN maximum error is too large" }
            val evidence = report.getJSONObject("backendEvidence")
            require(evidence.getBoolean("deviceSupported") && evidence.getBoolean("libraryReady")) {
                "Qualcomm QNN provider is not ready"
            }
            val delegation = QnnDelegationEvidence.assess(evidence)
            require(delegation.status == QnnDelegationStatus.DELEGATED) {
                "QNN delegation was not verified (${delegation.status.id}): ${delegation.reason}"
            }
        }
        if (backend == "litert_gpu_bounded") {
            val evidence = report.getJSONObject("backendEvidence")
            require(evidence.getInt("kernelBatchSize") == 1) { "Bounded GPU batch size is not N=1" }
            require(evidence.getInt("commandQueueWindowSize") == 1) {
                "Bounded GPU queue window is not N=1"
            }
        }
    }

    private fun validateAudioReport(report: JSONObject, expectedFrames: Long, expectedWindows: Int) {
        val audio = report.getJSONObject("audio")
        require(audio.getLong("outputFrames") == expectedFrames) { "Unexpected audio output frame count" }
        require(audio.getInt("windowCount") == expectedWindows) { "Unexpected audio window count" }
        require(audio.getInt("completedWindows") == expectedWindows) { "Audio separation did not finish" }
        require(audio.getJSONObject("session").getBoolean("cleanupComplete")) { "QNN session cleanup failed" }
        val delegation = QnnDelegationEvidence.assess(audio.getJSONObject("backendEvidence"))
        require(delegation.status == QnnDelegationStatus.DELEGATED) {
            "QNN audio delegation was not verified (${delegation.status.id}): ${delegation.reason}"
        }
        val outputs = audio.getJSONObject("outputs")
        listOf("vocals", "instrumental").forEach { stem ->
            val output = outputs.getJSONObject(stem)
            require(output.getLong("frames") == expectedFrames) { "$stem frame count is incorrect" }
            require(output.getLong("nonFiniteSamples") == 0L) { "$stem contains non-finite samples" }
        }
    }

    private fun stageFixtures(
        bundle: AppLiveValidationBundle,
        profile: AppLiveValidationProfile,
        runDirectory: File,
        logger: RunLogger,
    ): Map<String, File> {
        val fixtureIds = when (profile) {
            AppLiveValidationProfile.QUICK -> setOf(
                "model",
                "windowInput",
                "windowReference",
                "quickAudio",
                "runtimeManifest",
            )
            AppLiveValidationProfile.FULL -> setOf(
                "model",
                "windowInput",
                "windowReference",
                "fullAudio",
                "runtimeManifest",
            )
            AppLiveValidationProfile.DSP_MATRIX -> error(
                "DSP matrix does not use QNN validation fixtures",
            )
        }
        return fixtureIds.associateWith { id ->
            val fixture = bundle.requireFixture(id)
            val parent = when (fixture.targetDirectory) {
                "models" -> File(benchmarkDirectory(), "models")
                "inputs" -> File(benchmarkDirectory(), "inputs")
                "reference" -> File(benchmarkDirectory(), "reference")
                "audio-input" -> File(benchmarkDirectory(), "audio-input")
                "evidence" -> runDirectory
                else -> error("Unexpected target directory: ${fixture.targetDirectory}")
            }.apply { mkdirs() }
            val target = File(parent, fixture.targetName)
            if (target.isFile && target.length() == fixture.bytes &&
                AppLiveHashing.sha256(target) == fixture.sha256
            ) {
                logger.log("fixture reused id=$id bytes=${fixture.bytes}")
                target
            } else {
                val partial = File(parent, "${fixture.targetName}.partial")
                partial.delete()
                assets.open(fixture.assetPath).buffered().use { input ->
                    partial.outputStream().buffered().use { output -> input.copyTo(output) }
                }
                require(partial.length() == fixture.bytes) { "Fixture size mismatch after copy: $id" }
                require(AppLiveHashing.sha256(partial) == fixture.sha256) {
                    "Fixture SHA-256 mismatch after copy: $id"
                }
                Files.move(
                    partial.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
                logger.log("fixture staged id=$id bytes=${fixture.bytes}")
                target
            }
        }
    }

    private fun identity(
        profile: AppLiveValidationProfile,
        runId: String,
        bundle: AppLiveValidationBundle,
        staged: Map<String, File>,
        uploadLargeArtifacts: Boolean,
    ): JSONObject {
        val memory = ActivityManager.MemoryInfo().also {
            getSystemService(ActivityManager::class.java).getMemoryInfo(it)
        }
        val packageInfo = packageManager.getPackageInfo(packageName, 0)
        val socManufacturer = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER else ""
        val socModel = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else ""
        @Suppress("DEPRECATION")
        val versionCode = if (Build.VERSION.SDK_INT >= 28) {
            packageInfo.longVersionCode
        } else {
            packageInfo.versionCode.toLong()
        }
        val nativeLibraries = File(applicationInfo.nativeLibraryDir).listFiles()
            .orEmpty()
            .filter { file -> file.isFile && file.extension == "so" }
            .sortedBy { it.name }
            .map { file ->
                JSONObject()
                    .put("name", file.name)
                    .put("bytes", file.length())
                    .put("sha256", AppLiveHashing.sha256(file))
            }
        val apkFiles = buildList {
            add(applicationInfo.sourceDir)
            addAll(applicationInfo.splitSourceDirs.orEmpty())
        }.distinct().map { path ->
            val file = File(path)
            JSONObject()
                .put("name", file.name)
                .put("bytes", file.length())
                .put("sha256", AppLiveHashing.sha256(file))
        }
        val processMemory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        return JSONObject()
            .put("schemaVersion", 1)
            .put("capturedAt", isoNow())
            .put("runId", runId)
            .put("profile", profile.id)
            .put("bundleId", bundle.bundleId)
            .put("diagnosticContractVersion", bundle.diagnosticContractVersion)
            .put("sourceCommit", bundle.sourceCommit)
            .put("sourceDirty", bundle.sourceDirty)
            .put("artifactPolicy", JSONObject()
                .put("uploadTensorOutputs", uploadLargeArtifacts)
                .put("uploadFullAudioOutputs", uploadLargeArtifacts)
                .put("uploadQuickAudioOutputs", true))
            .put("device", JSONObject()
                .put("manufacturer", Build.MANUFACTURER)
                .put("brand", Build.BRAND)
                .put("model", Build.MODEL)
                .put("product", Build.PRODUCT)
                .put("device", Build.DEVICE)
                .put("board", Build.BOARD)
                .put("hardware", Build.HARDWARE)
                .put("socManufacturer", socManufacturer)
                .put("socModel", socModel)
                .put("fingerprint", Build.FINGERPRINT)
                .put("buildId", Build.ID)
                .put("buildDisplay", Build.DISPLAY)
                .put("buildType", Build.TYPE)
                .put("buildTags", Build.TAGS)
                .put("buildTimeMs", Build.TIME)
                .put("incremental", Build.VERSION.INCREMENTAL)
                .put("securityPatch", Build.VERSION.SECURITY_PATCH)
                .put("baseOs", Build.VERSION.BASE_OS)
                .put("bootloader", Build.BOOTLOADER)
                .put("baseband", runCatching { Build.getRadioVersion() }.getOrNull() ?: JSONObject.NULL)
                .put("androidRelease", Build.VERSION.RELEASE)
                .put("sdk", Build.VERSION.SDK_INT)
                .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))
                .put("supported32BitAbis", JSONArray(Build.SUPPORTED_32_BIT_ABIS.toList()))
                .put("supported64BitAbis", JSONArray(Build.SUPPORTED_64_BIT_ABIS.toList()))
                .put("is64Bit", Process.is64Bit())
                .put("totalMemoryBytes", memory.totalMem)
                .put("lowMemory", memory.lowMemory)
                .put("lowRamDevice", getSystemService(ActivityManager::class.java).isLowRamDevice)
                .put("systemProperties", diagnosticSystemProperties()))
            .put("application", JSONObject()
                .put("package", packageName)
                .put("versionName", packageInfo.versionName)
                .put("versionCode", versionCode)
                .put("apkBytes", File(applicationInfo.sourceDir).length())
                .put("apkSha256", AppLiveHashing.sha256(File(applicationInfo.sourceDir)))
                .put("apkFiles", JSONArray(apkFiles))
                .put("nativeLibraryDir", applicationInfo.nativeLibraryDir)
                .put("nativeLibraries", JSONArray(nativeLibraries)))
            .put("process", JSONObject()
                .put("pid", Process.myPid())
                .put("totalPssKb", processMemory.totalPss)
                .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize()))
            .put("fixtures", stagedEvidence(staged))
    }

    private fun diagnosticSystemProperties(): JSONObject = JSONObject().apply {
        DIAGNOSTIC_SYSTEM_PROPERTIES.forEach { name ->
            put(name, readSystemProperty(name) ?: JSONObject.NULL)
        }
    }

    private fun readSystemProperty(name: String): String? = runCatching {
        val propertyProcess = ProcessBuilder("/system/bin/getprop", name)
            .redirectErrorStream(true)
            .start()
        if (!propertyProcess.waitFor(2, TimeUnit.SECONDS)) {
            propertyProcess.destroyForcibly()
            return@runCatching null
        }
        propertyProcess.inputStream.bufferedReader().use { it.readText() }
            .trim()
            .takeIf { it.isNotEmpty() }
            ?.take(4_096)
    }.getOrNull()

    private fun stagedEvidence(staged: Map<String, File>): JSONArray = JSONArray(
        staged.entries.sortedBy { it.key }.map { (id, file) ->
            JSONObject()
                .put("id", id)
                .put("bytes", file.length())
                .put("sha256", AppLiveHashing.sha256(file))
        },
    )

    private fun checkpoint(
        relay: AppLiveRelayClient,
        runId: String,
        runDirectory: File,
        name: String,
        value: JSONObject,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
    ) {
        val file = File(runDirectory, name).apply { writeJsonAtomic(this, value) }
        upload(relay, runId, name, file, "application/json", uploadedFiles, logger)
    }

    private fun upload(
        relay: AppLiveRelayClient,
        runId: String,
        name: String,
        file: File,
        contentType: String,
        uploadedFiles: MutableList<String>,
        logger: RunLogger,
        logSuccess: Boolean = true,
    ) {
        relay.upload(runId, name, file, contentType)
        uploadedFiles += name
        if (logSuccess) logger.log("uploaded name=$name bytes=${file.length()}")
    }

    private fun updateState(profile: AppLiveValidationProfile, runId: String, message: String) {
        AppLiveValidationState.update(
            this,
            state = "running",
            running = true,
            profile = profile,
            runId = runId,
            message = message,
        )
    }

    private fun updateNotification(message: String) {
        startForeground(NOTIFICATION_ID, notification(message))
    }

    private fun notification(message: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("App Live validation")
            .setContentText(message)
            .setContentIntent(pendingIntent)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .build()
    }

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "App Live validation",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    private fun createRunId(
        origin: AppLiveRunOrigin,
        profile: AppLiveValidationProfile,
        bundleId: String,
    ): String {
        val model = Build.MODEL.lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), "-")
            .trim('-')
            .ifBlank { "android" }
        val formatter = SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        return "${origin.id}-$model-${profile.id}-${formatter.format(Date())}-${bundleId.take(8)}".take(96)
    }

    private fun appLiveRoot(): File = File(getExternalFilesDir(null) ?: filesDir, "app-live")
        .apply { mkdirs() }

    private fun benchmarkDirectory(): File = File(getExternalFilesDir(null) ?: filesDir, "benchmark")
        .apply { mkdirs() }

    private fun reportsDirectory(): File = File(benchmarkDirectory(), "reports").apply { mkdirs() }

    private fun qnnEvidenceDirectory(): File = File(benchmarkDirectory(), "qnn").apply { mkdirs() }

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
        "wav" -> "audio/wav"
        else -> "application/octet-stream"
    }

    private fun isoNow(): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        return formatter.format(Date())
    }

    private class RunLogger(val file: File) {
        @Synchronized
        fun log(message: String) {
            file.parentFile?.mkdirs()
            file.appendText("${System.currentTimeMillis()} $message\n")
        }
    }

    private data class BenchmarkExecution(
        val report: JSONObject,
        val broadcastSucceeded: Boolean,
    ) {
        val succeeded: Boolean
            get() = broadcastSucceeded && report.optString("status") == "complete"
    }

    companion object {
        const val ACTION_QUICK = "com.example.musicsourceseparation.APP_LIVE_QUICK"
        const val ACTION_FULL = "com.example.musicsourceseparation.APP_LIVE_FULL"

        private const val NOTIFICATION_CHANNEL_ID = "app_live_validation"
        private const val NOTIFICATION_ID = 9662
        private val DIAGNOSTIC_SYSTEM_PROPERTIES = listOf(
            "ro.build.characteristics",
            "ro.build.version.incremental",
            "ro.build.version.security_patch",
            "ro.product.vendor.brand",
            "ro.product.vendor.manufacturer",
            "ro.product.vendor.model",
            "ro.product.vendor.device",
            "ro.product.vendor.name",
            "ro.vendor.build.fingerprint",
            "ro.vendor.build.id",
            "ro.vendor.build.version.incremental",
            "ro.vendor.build.version.security_patch",
            "ro.board.platform",
            "ro.hardware",
            "ro.boot.hardware",
            "ro.boot.hardware.platform",
            "ro.boot.hardware.sku",
            "ro.boot.product.hardware.sku",
            "ro.boot.bootloader",
            "ro.boot.baseband",
            "ro.soc.manufacturer",
            "ro.soc.model",
            "ro.vendor.qti.soc_id",
            "ro.vendor.qti.soc_model",
        )
        private val running = AtomicBoolean(false)

        internal fun isRunning(): Boolean = running.get()

        internal fun start(context: Context, profile: AppLiveValidationProfile) {
            val action = when (profile) {
                AppLiveValidationProfile.QUICK -> ACTION_QUICK
                AppLiveValidationProfile.FULL -> ACTION_FULL
                AppLiveValidationProfile.DSP_MATRIX -> error("DSP matrix uses AppLiveDspMatrixService")
            }
            context.startForegroundService(Intent(action, null, context, AppLiveValidationService::class.java))
        }
    }
}
