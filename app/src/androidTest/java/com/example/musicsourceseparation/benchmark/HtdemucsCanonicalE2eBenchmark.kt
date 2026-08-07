package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import com.example.musicsourceseparation.BuildConfig
import com.example.musicsourceseparation.audio.AudioPcmDecoder
import com.example.musicsourceseparation.audio.DecodedPcmAudio
import com.example.musicsourceseparation.audio.WavFileWriter
import com.example.musicsourceseparation.model.HtdemucsDsp
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlin.math.roundToInt
import kotlin.math.sqrt
import org.jtransforms.utils.CommonUtils
import org.json.JSONArray
import org.json.JSONObject
import pl.edu.icm.jlargearrays.ConcurrencyUtils

internal class HtdemucsCanonicalE2eBenchmark(
    private val context: Context,
) {
    data class Config(
        val audioFileName: String,
        val durationSeconds: Int? = null,
        val frameLimit: Int? = null,
        val threads: Int,
        val istftMode: HtdemucsDsp.IstftMode = HtdemucsDsp.IstftMode.SERIAL,
        val istftWorkers: Int = 1,
        val validateIstftFloatParity: Boolean = false,
        val coreWarmupRuns: Int = 0,
        val coreMeasuredRuns: Int = 0,
        val postprocessMode: PostprocessMode = PostprocessMode.LEGACY,
        val runId: String,
        val modelVariant: String = MODEL_VARIANT_OFFICIAL,
        val expectedAudioSha256: String? = null,
        val cancelAfterWindows: Int = 0,
        val resumeAfterCancel: Boolean = false,
        val exportCanonicalInput: Boolean = false,
    )

    private data class ModelIdentity(
        val variant: String,
        val modelId: String,
        val fileName: String,
        val byteSize: Long,
        val sha256: String,
        val diagnosticOnly: Boolean,
        val researchOnly: Boolean,
        val hostAdmissionStatus: String,
        val stemOrder: List<String>,
    )

    data class Result(
        val report: JSONObject,
        val reportFile: File,
    )

    enum class PostprocessMode(val wireValue: String) {
        LEGACY("legacy"),
        FUSED_REUSE("fused-reuse"),
        ;

        companion object {
            fun fromWireValue(value: String): PostprocessMode = entries.firstOrNull {
                it.wireValue == value
            } ?: error("Unknown postprocess mode '$value'.")
        }
    }

    fun run(config: Config): Result {
        validateConfig(config)
        val modelIdentity = MODEL_IDENTITIES.getValue(config.modelVariant)
        val outputAbi = outputAbi(modelIdentity.stemOrder.size)
        val benchmarkRoot = File(
            requireNotNull(context.getExternalFilesDir(null)),
            "benchmark",
        )
        val modelFile = File(benchmarkRoot, "models/${modelIdentity.fileName}")
        require(modelFile.isFile && modelFile.length() == modelIdentity.byteSize) {
            "Missing ${modelIdentity.variant} model or byte-size mismatch: " +
                modelFile.absolutePath
        }
        val modelHash = timed { sha256(modelFile) }
        require(modelHash.value == modelIdentity.sha256) {
            "${modelIdentity.variant} model SHA-256 mismatch: ${modelHash.value}"
        }

        val audioFile = File(File(benchmarkRoot, "audio"), config.audioFileName).canonicalFile
        val audioRoot = File(benchmarkRoot, "audio").canonicalFile
        require(audioFile.parentFile == audioRoot && audioFile.isFile) {
            "Audio input must be a file directly under ${audioRoot.absolutePath}."
        }
        val audioHash = timed { sha256(audioFile) }
        config.expectedAudioSha256?.let { expected ->
            require(audioHash.value == expected) {
                "Audio SHA-256 mismatch: ${audioHash.value}"
            }
        }
        val selectionLabel = when {
            config.frameLimit != null -> "${config.frameLimit}frames"
            config.durationSeconds != null && config.durationSeconds != 0 ->
                "${config.durationSeconds}s"
            else -> "full-song"
        }
        val runFamily = if (modelIdentity.variant == MODEL_VARIANT_OFFICIAL) {
            "htdemucs-canonical-e2e/cpu"
        } else {
            "htdemucs-canonical-e2e/cpu/${modelIdentity.variant}"
        }
        val executionFamily = if (config.istftMode == HtdemucsDsp.IstftMode.SERIAL) {
            runFamily
        } else {
            "$runFamily/istft-${config.istftMode.wireValue}-w${config.istftWorkers}"
        }
        val profileFamily = "$executionFamily/postprocess-${config.postprocessMode.wireValue}"
        val runRoot = File(benchmarkRoot, "$profileFamily/$selectionLabel/${config.runId}")
        require(runRoot.deleteRecursively() && runRoot.mkdirs()) {
            "Could not prepare E2E run directory: ${runRoot.absolutePath}"
        }
        val temporaryInputRoot = File(runRoot, TEMPORARY_INPUT_DIRECTORY)
        val preparedAudio = try {
            prepareCanonicalAudio(audioFile, temporaryInputRoot)
        } catch (error: Throwable) {
            cleanupTemporaryInputAfterFailure(temporaryInputRoot, error)
            throw error
        }

        val report = try {
            val wav = preparedAudio.wav
            // Zero is an explicit full-song sentinel; the policy's null duration
            // path preserves the complete decoded frame count without multiplication.
            val durationLimitSeconds = config.durationSeconds?.takeUnless { it == 0 }
            val selectedFrames = HtdemucsCanonicalAudioInputPolicy.selectFrames(
                availableFrames = preparedAudio.canonicalFrames,
                frameLimit = config.frameLimit,
                durationSeconds = durationLimitSeconds,
                sampleRate = SAMPLE_RATE,
            )
            val normalizationRead = timed { wav.scanNormalization(selectedFrames) }
            val normalization = normalizationRead.value
            if (selectedFrames == preparedAudio.canonicalFrames) {
                require(normalization.selectedPcmSha256 == preparedAudio.canonicalPcmSha256) {
                    "Selected PCM SHA-256 does not match the complete canonical PCM."
                }
            }
            val expectedWindowCount = windowPlans(selectedFrames).size
            require(config.cancelAfterWindows <= expectedWindowCount) {
                "cancelAfterWindows exceeds the $expectedWindowCount-window run."
            }

            val sourceEvidence = JSONObject()
                .put("path", audioFile.absolutePath)
                .put("fileName", audioFile.name)
                .put("fileBytes", audioFile.length())
                .put("fileSha256", audioHash.value)
                .put("mediaFormat", preparedAudio.mediaFormat)
                .put("decoded", preparedAudio.decodedEvidence)
                .put("canonicalPcm", preparedAudio.canonicalEvidence())
                .put("selectedPcmSha256", normalization.selectedPcmSha256)
                .put("sampleRate", wav.sampleRate)
                .put("channelCount", wav.channelCount)
                .put("bitsPerSample", wav.bitsPerSample)
                .put("sourceFrames", preparedAudio.decodedEvidence.getInt("frames"))
                .put("canonicalFrames", wav.frameCount)
                .put("selectedStartFrame", 0)
                .put("selectedFrames", selectedFrames)
                .put("requestedFrameLimit", config.frameLimit ?: JSONObject.NULL)
                .put("requestedDurationSeconds", config.durationSeconds ?: JSONObject.NULL)
                .put("selectionMode", if (selectedFrames == preparedAudio.canonicalFrames) {
                    "full-song"
                } else {
                    "bounded-prefix"
                })
                .put("durationNumeratorFrames", selectedFrames)
                .put("durationDenominatorSampleRate", SAMPLE_RATE)
                .put("durationSeconds", selectedFrames.toDouble() / SAMPLE_RATE)
                .put("globalMean", normalization.mean)
                .put("sampleStandardDeviation", normalization.sampleStandardDeviation)
                .put("normalizationScale", normalization.scale)
                .put("normalizationEpsilon", NORMALIZATION_EPSILON)

            val attempts = JSONArray()
            var terminalStatus = "complete"
            if (config.cancelAfterWindows > 0) {
                val cancellation = runAttempt(
                    config = config,
                    wav = wav,
                    selectedFrames = selectedFrames,
                    normalization = normalization,
                    modelFile = modelFile,
                    modelIdentity = modelIdentity,
                    attemptRoot = File(runRoot, "cancel-probe"),
                    cancelAfterWindows = config.cancelAfterWindows,
                )
                require(cancellation.getString("status") == "cancelled")
                require(!File(File(runRoot, "cancel-probe"), OUTPUT_STAGING_NAME).exists()) {
                    "Cancellation left a partial output directory behind."
                }
                attempts.put(cancellation)
                if (config.resumeAfterCancel) {
                    attempts.put(
                        runAttempt(
                            config = config,
                            wav = wav,
                            selectedFrames = selectedFrames,
                            normalization = normalization,
                            modelFile = modelFile,
                            modelIdentity = modelIdentity,
                            attemptRoot = File(runRoot, "resume-from-zero"),
                            cancelAfterWindows = 0,
                        ),
                    )
                } else {
                    terminalStatus = "cancelled"
                }
            } else {
                attempts.put(
                    runAttempt(
                        config = config,
                        wav = wav,
                        selectedFrames = selectedFrames,
                        normalization = normalization,
                        modelFile = modelFile,
                        modelIdentity = modelIdentity,
                        attemptRoot = File(runRoot, "full"),
                        cancelAfterWindows = 0,
                    ),
                )
            }

            JSONObject()
                .put("schemaVersion", 3)
                .put("status", terminalStatus)
                .put("runId", config.runId)
                .put("backend", "litert-compiled-model-cpu")
                .put("execution", istftExecutionEvidence(config, modelIdentity.stemOrder.size))
                .put("runtime", runtimeEvidence())
                .put("model", JSONObject()
                    .put("modelId", modelIdentity.modelId)
                    .put("path", modelFile.absolutePath)
                    .put("fileName", modelIdentity.fileName)
                    .put("byteSize", modelIdentity.byteSize)
                    .put("sha256", modelIdentity.sha256)
                    .put("hashTiming", modelHash.timing.evidence())
                    .put("signatureKey", SIGNATURE_KEY)
                    .put("inputs", tensorAbiEvidence(INPUT_ABI))
                    .put("outputs", tensorAbiEvidence(outputAbi))
                    .also { modelEvidence ->
                        if (modelIdentity.variant != MODEL_VARIANT_OFFICIAL) {
                            modelEvidence
                                .put("variant", modelIdentity.variant)
                                .put("diagnosticOnly", modelIdentity.diagnosticOnly)
                                .put("researchOnly", modelIdentity.researchOnly)
                                .put("hostAdmissionStatus", modelIdentity.hostAdmissionStatus)
                        }
                    })
                .put("source", sourceEvidence)
                .put("sourceHashTiming", audioHash.timing.evidence())
                .put("inputPreparation", preparedAudio.timingEvidence)
                .put("normalizationScanTiming", normalizationRead.timing.evidence())
                .put("contract", contractEvidence(selectedFrames, modelIdentity.stemOrder))
                .put("cancelAfterWindows", config.cancelAfterWindows)
                .put("resumeAfterCancel", config.resumeAfterCancel)
                .put("exportCanonicalInput", config.exportCanonicalInput)
                .put(
                    "resumeStrategy",
                    if (config.cancelAfterWindows > 0 && config.resumeAfterCancel) {
                        "validated-cleanup-then-restart-from-window-zero"
                    } else {
                        JSONObject.NULL
                    },
                )
                .put("attempts", attempts)
        } catch (error: Throwable) {
            cleanupTemporaryInputAfterFailure(temporaryInputRoot, error)
            throw error
        }

        val exportedCanonicalInput = File(runRoot, EXPORTED_CANONICAL_INPUT_NAME)
        val canonicalInputExport = try {
            if (config.exportCanonicalInput && report.getString("status") == "complete") {
                timed {
                    exportCanonicalInput(
                        source = preparedAudio.wav,
                        target = exportedCanonicalInput,
                        expectedFrames = preparedAudio.canonicalFrames,
                        expectedPcmSha256 = preparedAudio.canonicalPcmSha256,
                    )
                }
            } else {
                null
            }
        } catch (error: Throwable) {
            try {
                require(!exportedCanonicalInput.exists() || exportedCanonicalInput.delete()) {
                    "Could not remove failed canonical input export."
                }
            } catch (cleanupFailure: Throwable) {
                error.addSuppressed(cleanupFailure)
            }
            cleanupTemporaryInputAfterFailure(temporaryInputRoot, error)
            throw error
        }
        report.put(
            "canonicalInputExport",
            canonicalInputExport?.value ?: JSONObject()
                .put("requested", config.exportCanonicalInput)
                .put("retained", false)
                .put(
                    "reason",
                    if (config.exportCanonicalInput) "run-not-complete" else "not-requested",
                ),
        )
        report.put(
            "canonicalInputExportTiming",
            canonicalInputExport?.timing?.evidence() ?: JSONObject.NULL,
        )

        val cleanup = timed {
            require(temporaryInputRoot.deleteRecursively() && !temporaryInputRoot.exists()) {
                "Could not clean temporary canonical audio input."
            }
        }
        report.put(
            "temporaryInputCleanup",
            cleanup.timing.evidence()
                .put("complete", true)
                .put("temporaryWavDeleted", !preparedAudio.wav.file.exists()),
        )
        val reportFile = File(runRoot, "report.json")
        publishJson(reportFile, report)
        return Result(report, reportFile)
    }

    private fun prepareCanonicalAudio(
        sourceFile: File,
        temporaryInputRoot: File,
    ): PreparedCanonicalAudio {
        require(!temporaryInputRoot.exists() && temporaryInputRoot.mkdirs()) {
            "Could not prepare temporary audio directory: ${temporaryInputRoot.absolutePath}"
        }
        val totalStarted = StageStart()
        val mediaInspection = timed { inspectSourceMedia(sourceFile) }
        val decoded = decodeAndResample(sourceFile)
        val temporaryWav = File(temporaryInputRoot, TEMPORARY_WAV_NAME)
        val wavWrite = timed { writeCanonicalWav(decoded.resampledAudio, temporaryWav) }
        val wavValidation = timed { CanonicalPcm16Wav.open(temporaryWav) }
        val wav = wavValidation.value
        require(wav.frameCount <= Int.MAX_VALUE.toLong()) {
            "Canonical WAV has ${wav.frameCount} frames; this harness uses Int-indexed " +
                "window plans and supports at most ${Int.MAX_VALUE}."
        }
        require(wav.frameCount == wavWrite.value.frames.toLong()) {
            "Temporary WAV frame count does not match decoded canonical PCM."
        }
        require(wav.dataBytes == wavWrite.value.pcmBytes) {
            "Temporary WAV PCM size does not match decoded canonical PCM."
        }

        return PreparedCanonicalAudio(
            wav = wav,
            canonicalFrames = wavWrite.value.frames,
            canonicalPcmSha256 = wavWrite.value.pcmSha256,
            mediaFormat = mediaInspection.value,
            decodedEvidence = decoded.decodedEvidence,
            channelMapping = wavWrite.value.channelMapping,
            resampled = decoded.resampled,
            sourceSampleRate = decoded.sourceSampleRate,
            sourceChannelCount = decoded.sourceChannelCount,
            canonicalWavBytes = temporaryWav.length(),
            timingEvidence = JSONObject()
                .put("mediaInspection", mediaInspection.timing.evidence())
                .put("decode", decoded.decodeTiming.evidence())
                .put("decodedPcmHash", decoded.decodedPcmHashTiming.evidence())
                .put("resample", decoded.resampleTiming.evidence())
                .put("canonicalWavWrite", wavWrite.timing.evidence())
                .put("canonicalWavValidation", wavValidation.timing.evidence())
                .put("total", totalStarted.elapsed().evidence())
                .put("memoryPolicy", "decode-resample-write-release-before-model-prepare")
                .put("wavReadMode", "bounded-random-access-streaming"),
        )
    }

    private fun exportCanonicalInput(
        source: CanonicalPcm16Wav,
        target: File,
        expectedFrames: Int,
        expectedPcmSha256: String,
    ): JSONObject {
        require(source.file.isFile) { "Canonical input WAV is missing before export." }
        require(!target.exists()) { "Canonical input export already exists: ${target.absolutePath}" }
        requireNotNull(target.parentFile).mkdirs()
        Files.move(
            source.file.toPath(),
            target.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
        )
        val exported = CanonicalPcm16Wav.open(target)
        require(exported.frameCount == expectedFrames.toLong()) {
            "Exported canonical WAV frame count mismatch: ${exported.frameCount} != $expectedFrames"
        }
        require(exported.dataBytes == expectedFrames.toLong() * BYTES_PER_FRAME)
        return JSONObject()
            .put("requested", true)
            .put("retained", true)
            .put("path", target.absolutePath)
            .put("fileName", target.name)
            .put("byteSize", target.length())
            .put("sha256", sha256(target))
            .put("pcmSha256", expectedPcmSha256)
            .put("frames", exported.frameCount)
            .put("durationNumeratorFrames", exported.frameCount)
            .put("durationDenominatorSampleRate", SAMPLE_RATE)
            .put("durationSeconds", exported.frameCount.toDouble() / SAMPLE_RATE)
            .put("sampleRate", exported.sampleRate)
            .put("channelCount", exported.channelCount)
            .put("bitsPerSample", exported.bitsPerSample)
    }

    private fun decodeAndResample(sourceFile: File): DecodedStage {
        val decode = timed {
            AudioPcmDecoder(context).decode(Uri.fromFile(sourceFile))
        }
        val decodedAudio = decode.value
        require(decodedAudio.frameCount > 1) { "Decoder produced fewer than two PCM frames." }
        val decodedPcmHash = timed { sha256(decodedAudio.pcm16) }
        val decodedEvidence = JSONObject()
            .put("sampleRate", decodedAudio.sampleRate)
            .put("channelCount", decodedAudio.channelCount)
            .put("frames", decodedAudio.frameCount)
            .put("pcmBytes", decodedAudio.pcm16.size)
            .put("pcmEncoding", "signed-pcm16-le")
            .put("pcmSha256", decodedPcmHash.value)
        val resample = timed { decodedAudio.resampleTo(SAMPLE_RATE) }
        val resampledAudio = resample.value
        require(resampledAudio.sampleRate == SAMPLE_RATE)
        require(resampledAudio.frameCount > 1)
        return DecodedStage(
            resampledAudio = resampledAudio,
            decodedEvidence = decodedEvidence,
            sourceSampleRate = decodedAudio.sampleRate,
            sourceChannelCount = decodedAudio.channelCount,
            resampled = resampledAudio !== decodedAudio,
            decodeTiming = decode.timing,
            decodedPcmHashTiming = decodedPcmHash.timing,
            resampleTiming = resample.timing,
        )
    }

    private fun writeCanonicalWav(
        audio: DecodedPcmAudio,
        target: File,
    ): CanonicalWavWrite {
        require(audio.sampleRate == SAMPLE_RATE)
        require(audio.frameCount > 1)
        require(!target.exists())
        val partial = File(requireNotNull(target.parentFile), "${target.name}.partial")
        require(!partial.exists())
        val digest = MessageDigest.getInstance("SHA-256")
        var complete = false
        try {
            WavFileWriter(partial, SAMPLE_RATE, CHANNEL_COUNT).use { writer ->
                var startFrame = 0
                while (startFrame < audio.frameCount) {
                    val frameCount = minOf(CANONICAL_WRITE_CHUNK_FRAMES, audio.frameCount - startFrame)
                    val chunk = HtdemucsCanonicalAudioInputPolicy.stereoPcm16Chunk(
                        sourcePcm16 = audio.pcm16,
                        sourceChannelCount = audio.channelCount,
                        startFrame = startFrame,
                        frameCount = frameCount,
                    )
                    digest.update(chunk)
                    writer.writePcm16(chunk)
                    startFrame += frameCount
                }
            }
            require(partial.length() == WAV_HEADER_BYTES + audio.frameCount.toLong() * BYTES_PER_FRAME)
            require(partial.renameTo(target)) { "Could not finalize temporary canonical WAV." }
            complete = true
            return CanonicalWavWrite(
                frames = audio.frameCount,
                pcmBytes = audio.frameCount.toLong() * BYTES_PER_FRAME,
                pcmSha256 = digest.digest().toHex(),
                channelMapping = HtdemucsCanonicalAudioInputPolicy.channelMapping(audio.channelCount),
            )
        } finally {
            if (!complete) {
                partial.delete()
                target.delete()
            }
        }
    }

    private fun inspectSourceMedia(sourceFile: File): JSONObject {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(sourceFile.absolutePath)
            val audioTracks = JSONArray()
            var selectedTrack: JSONObject? = null
            repeat(extractor.trackCount) { index ->
                val format = extractor.getTrackFormat(index)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: return@repeat
                if (!mime.startsWith("audio/")) return@repeat
                val evidence = JSONObject()
                    .put("trackIndex", index)
                    .put("mime", mime)
                    .put(
                        "sampleRate",
                        format.optionalInteger(MediaFormat.KEY_SAMPLE_RATE) ?: JSONObject.NULL,
                    )
                    .put(
                        "channelCount",
                        format.optionalInteger(MediaFormat.KEY_CHANNEL_COUNT) ?: JSONObject.NULL,
                    )
                    .put(
                        "durationUs",
                        format.optionalLong(MediaFormat.KEY_DURATION) ?: JSONObject.NULL,
                    )
                    .put(
                        "bitRate",
                        format.optionalInteger(MediaFormat.KEY_BIT_RATE) ?: JSONObject.NULL,
                    )
                audioTracks.put(evidence)
                if (selectedTrack == null) selectedTrack = evidence
            }
            return JSONObject()
                .put("fileExtension", sourceFile.extension.lowercase())
                .put("extractorTrackCount", extractor.trackCount)
                .put("audioTrackCount", audioTracks.length())
                .put("selectedAudioTrack", requireNotNull(selectedTrack) {
                    "Source media has no audio track."
                })
                .put("audioTracks", audioTracks)
        } finally {
            extractor.release()
        }
    }

    private fun cleanupTemporaryInputAfterFailure(root: File, failure: Throwable) {
        try {
            require(root.deleteRecursively() && !root.exists()) {
                "Could not clean temporary canonical audio after failure."
            }
        } catch (cleanupFailure: Throwable) {
            failure.addSuppressed(cleanupFailure)
        }
    }

    private fun runAttempt(
        config: Config,
        wav: CanonicalPcm16Wav,
        selectedFrames: Int,
        normalization: Normalization,
        modelFile: File,
        modelIdentity: ModelIdentity,
        attemptRoot: File,
        cancelAfterWindows: Int,
    ): JSONObject {
        val stemOrder = modelIdentity.stemOrder
        val stemCount = stemOrder.size
        val planeCount = stemCount * CHANNEL_COUNT
        val frequencyOutputElements =
            stemCount * HtdemucsDsp.FEATURE_COUNT * HtdemucsDsp.FREQUENCY_BINS * SPECTRUM_FRAMES
        val timeOutputElements = stemCount * CHANNEL_COUNT * WINDOW_SAMPLES
        val outputAbi = outputAbi(stemCount)
        require(attemptRoot.deleteRecursively() && attemptRoot.mkdirs())
        val stagingDir = File(attemptRoot, OUTPUT_STAGING_NAME)
        val outputDir = File(attemptRoot, OUTPUT_FINAL_NAME)
        require(stagingDir.mkdirs())
        val windows = JSONArray()
        val totalStarted = StageStart()
        var dsp: HtdemucsDsp? = null
        var parityDsp: HtdemucsDsp? = null
        val triangleWeight = triangleWeight()
        val carry = FloatArray(planeCount * OVERLAP_SAMPLES)
        val carryWeight = FloatArray(OVERLAP_SAMPLES)
        val reuseWorkspaces = config.postprocessMode == PostprocessMode.FUSED_REUSE
        val waveformWorkspace = if (reuseWorkspaces) {
            FloatArray(CHANNEL_COUNT * WINDOW_SAMPLES)
        } else {
            null
        }
        val pcmWorkspaces = if (reuseWorkspaces) {
            Array(stemCount) { ByteArray(WINDOW_SAMPLES * BYTES_PER_FRAME) }
        } else {
            emptyArray()
        }
        val stemStats = Array(stemCount) { StemStats() }
        val writers = mutableListOf<WavFileWriter>()
        var environment: Environment? = null
        var model: CompiledModel? = null
        val inputBuffers = linkedMapOf<String, TensorBuffer>()
        val outputBuffers = linkedMapOf<String, TensorBuffer>()
        var completedWindows = 0
        var carryLength = 0
        var outputFrames = 0
        var prepareEvidence = JSONObject()
        var coreBenchmarkEvidence: Any = JSONObject.NULL
        var e2eStarted: StageStart? = null
        var failure: Throwable? = null
        var cancelled = false

        try {
            val sessionPrepare = timed {
                val dspSetup = timed {
                    HtdemucsDsp(
                        windowSamples = WINDOW_SAMPLES,
                        istftMode = config.istftMode,
                        istftWorkers = config.istftWorkers,
                        reuseIoWorkspaces = reuseWorkspaces,
                    )
                }
                dsp = dspSetup.value
                require(dspSetup.value.frameCount == SPECTRUM_FRAMES)
                val parityDspSetup = if (config.validateIstftFloatParity) {
                    timed { HtdemucsDsp(WINDOW_SAMPLES) }
                } else {
                    null
                }
                parityDsp = parityDspSetup?.value
                val createdEnvironment = Environment.create()
                environment = createdEnvironment
                val availableAccelerators = createdEnvironment.getAvailableAccelerators()
                    .map { it.name }
                    .sorted()
                require(Accelerator.CPU.name in availableAccelerators) {
                    "CPU accelerator is unavailable: $availableAccelerators"
                }
                val options = CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(
                        numThreads = config.threads,
                        xnnPackFlags = null,
                        xnnPackWeightCachePath = null,
                    )
                }
                val compile = timed {
                    CompiledModel.create(modelFile.absolutePath, options, createdEnvironment)
                }
                val compiled = compile.value
                model = compiled
                val allocation = timed {
                    INPUT_ABI.forEach { tensor ->
                        checkTensorType(
                            compiled.getInputTensorType(tensor.name, SIGNATURE_KEY),
                            tensor,
                        )
                        inputBuffers[tensor.name] =
                            compiled.createInputBuffer(tensor.name, SIGNATURE_KEY)
                    }
                    outputAbi.forEach { tensor ->
                        checkTensorType(
                            compiled.getOutputTensorType(tensor.name, SIGNATURE_KEY),
                            tensor,
                        )
                        outputBuffers[tensor.name] =
                            compiled.createOutputBuffer(tensor.name, SIGNATURE_KEY)
                    }
                }
                val writerSetup = timed {
                    stemOrder.forEach { stem ->
                        writers += WavFileWriter(
                            File(stagingDir, "$stem.wav.partial"),
                            SAMPLE_RATE,
                            CHANNEL_COUNT,
                        )
                    }
                }
                JSONObject()
                    .put("dspSetup", dspSetup.timing.evidence())
                    .put(
                        "parityDspSetup",
                        parityDspSetup?.timing?.evidence() ?: JSONObject.NULL,
                    )
                    .put("dspIstft", istftExecutionEvidence(config, stemCount))
                    .put("availableAccelerators", JSONArray(availableAccelerators))
                    .put("compile", compile.timing.evidence())
                    .put("bufferAllocation", allocation.timing.evidence())
                    .put("writerSetup", writerSetup.timing.evidence())
                    .put("process", processSnapshot())
            }
            prepareEvidence = sessionPrepare.value
                .put("total", sessionPrepare.timing.evidence())

            RandomAccessFile(wav.file, "r").use { audio ->
                if (config.coreWarmupRuns > 0 || config.coreMeasuredRuns > 0) {
                    val calibrationPlan = windowPlans(selectedFrames).first()
                    val calibrationSetup = timed {
                        val waveform = readNormalizedWindow(
                            audio,
                            wav,
                            calibrationPlan,
                            normalization,
                            waveformWorkspace,
                        )
                        val spectrum = requireNotNull(dsp).waveformToSpectrum(waveform)
                        inputBuffers.getValue(WAVEFORM_INPUT_NAME).writeFloat(waveform)
                        inputBuffers.getValue(SPECTRUM_INPUT_NAME).writeFloat(spectrum)
                    }
                    repeat(config.coreWarmupRuns) {
                        requireNotNull(model).run(inputBuffers, outputBuffers, SIGNATURE_KEY)
                    }
                    val samples = JSONArray()
                    val before = processSnapshot()
                    repeat(config.coreMeasuredRuns) { sampleIndex ->
                        val sample = timed {
                            requireNotNull(model).run(inputBuffers, outputBuffers, SIGNATURE_KEY)
                        }
                        samples.put(
                            JSONObject()
                                .put("index", sampleIndex)
                                .put("timing", sample.timing.evidence())
                                .put("thermalStatus", thermalStatus()),
                        )
                    }
                    coreBenchmarkEvidence = JSONObject()
                        .put("fixedCanonicalWindowIndex", 0)
                        .put("warmupRuns", config.coreWarmupRuns)
                        .put("measuredRuns", config.coreMeasuredRuns)
                        .put("setup", calibrationSetup.timing.evidence())
                        .put("samples", samples)
                        .put("summary", summarizeTimings(samples))
                        .put("before", before)
                        .put("after", processSnapshot())
                        .put(
                            "perOpProfiling",
                            JSONObject()
                                .put("status", "unsupported-by-litert-2.1.5-java-api")
                                .put(
                                    "detail",
                                    "CompiledModel CPU options expose threads and XNNPACK flags, " +
                                        "but no per-op profiler callback or result API.",
                                ),
                        )
                }
                e2eStarted = StageStart()
                windowPlans(selectedFrames).forEachIndexed { windowIndex, plan ->
                    val stage = JSONObject()
                    val windowStarted = StageStart()
                    var waveformInput: FloatArray? = null
                    val prepareStarted = StageStart()
                    waveformInput = readNormalizedWindow(
                        audio,
                        wav,
                        plan,
                        normalization,
                        waveformWorkspace,
                    )
                    stage.put("prepare", prepareStarted.elapsed().evidence())

                    var spectrumInput: FloatArray? = null
                    val stftStarted = StageStart()
                    spectrumInput = requireNotNull(dsp)
                        .waveformToSpectrum(requireNotNull(waveformInput))
                    stage.put("stft", stftStarted.elapsed().evidence())

                    val inputWrite = timed {
                        inputBuffers.getValue(WAVEFORM_INPUT_NAME)
                            .writeFloat(requireNotNull(waveformInput))
                        inputBuffers.getValue(SPECTRUM_INPUT_NAME)
                            .writeFloat(requireNotNull(spectrumInput))
                    }
                    stage.put("inputWrite", inputWrite.timing.evidence())
                    waveformInput = null
                    spectrumInput = null

                    val inference = timed {
                        requireNotNull(model).run(inputBuffers, outputBuffers, SIGNATURE_KEY)
                    }
                    stage.put("inference", inference.timing.evidence())

                    var frequencyOutput: FloatArray? = null
                    val frequencyReadStarted = StageStart()
                    frequencyOutput = outputBuffers.getValue(FREQUENCY_OUTPUT_NAME).readFloat()
                    require(requireNotNull(frequencyOutput).size == frequencyOutputElements)
                    val frequencyRead = frequencyReadStarted.elapsed()

                    val inverse = timed {
                        requireNotNull(dsp).frequencyToWaveform(
                            requireNotNull(frequencyOutput),
                            stemCount,
                        )
                    }
                    val combined = inverse.value
                    stage.put("iSTFT", inverse.timing.evidence())
                    if (config.validateIstftFloatParity && windowIndex == 0) {
                        val parityReference = timed {
                            requireNotNull(parityDsp).frequencyToWaveform(
                                requireNotNull(frequencyOutput),
                                stemCount,
                            )
                        }
                        stage.put("iSTFTParityReference", parityReference.timing.evidence())
                        val parityComparison = timed {
                            compareFloatBits(combined, parityReference.value)
                        }
                        require(parityComparison.value.mismatchCount == 0L) {
                            "S10 iSTFT raw-float parity failed: ${parityComparison.value}"
                        }
                        require(
                            parityComparison.value.candidateNonFiniteCount == 0L &&
                                parityComparison.value.referenceNonFiniteCount == 0L,
                        ) { "S10 iSTFT raw-float parity contains non-finite values." }
                        require(
                            parityComparison.value.candidateRawSha256 ==
                                parityComparison.value.referenceRawSha256,
                        ) { "Equal iSTFT float arrays produced different raw-byte hashes." }
                        stage.put(
                            "iSTFTParityComparison",
                            parityComparison.timing.evidence()
                                .put("result", parityComparison.value.evidence()),
                        )
                    }
                    frequencyOutput = null

                    var timeOutput: FloatArray? = null
                    val timeReadStarted = StageStart()
                    timeOutput = outputBuffers.getValue(TIME_OUTPUT_NAME).readFloat()
                    require(requireNotNull(timeOutput).size == timeOutputElements)
                    val timeRead = timeReadStarted.elapsed()
                    stage.put(
                        "outputRead",
                        (frequencyRead + timeRead).evidence()
                            .put("frequency", frequencyRead.evidence())
                            .put("time", timeRead.evidence()),
                    )

                    val finalized = if (config.postprocessMode == PostprocessMode.FUSED_REUSE) {
                        val fused = timed {
                            fusedPostprocessAndWrite(
                                frequencyWaveform = combined,
                                timeWaveform = requireNotNull(timeOutput),
                                plan = plan,
                                trackFrames = selectedFrames,
                                normalization = normalization,
                                triangleWeight = triangleWeight,
                                carry = carry,
                                carryWeight = carryWeight,
                                carryLength = carryLength,
                                stemCount = stemCount,
                                stats = stemStats,
                                writers = writers,
                                pcmWorkspaces = pcmWorkspaces,
                            )
                        }
                        stage.put("fusedPostprocessWrite", fused.timing.evidence())
                        fused.value
                    } else {
                        val branchCombine = timed {
                            val time = requireNotNull(timeOutput)
                            combined.indices.forEach { index -> combined[index] += time[index] }
                        }
                        stage.put("branchCombine", branchCombine.timing.evidence())
                        val ola = timed {
                            applyStreamingOla(
                                combined = combined,
                                plan = plan,
                                trackFrames = selectedFrames,
                                normalization = normalization,
                                triangleWeight = triangleWeight,
                                carry = carry,
                                carryWeight = carryWeight,
                                carryLength = carryLength,
                                planeCount = planeCount,
                            )
                        }
                        val pcm = timed {
                            encodePcm16(
                                combined = combined,
                                frames = ola.value.frames,
                                stats = stemStats,
                                stemCount = stemCount,
                            )
                        }
                        stage.put("branchCombine", branchCombine.timing.evidence())
                        stage.put("ola", ola.timing.evidence())
                        stage.put("pcm", pcm.timing.evidence())
                        val write = timed {
                            pcm.value.forEachIndexed { stem, bytes ->
                                writers[stem].writePcm16(bytes)
                            }
                        }
                        stage.put("write", write.timing.evidence())
                        ola.value
                    }
                    timeOutput = null
                    carryLength = finalized.nextCarryLength
                    outputFrames += finalized.frames
                    completedWindows = windowIndex + 1

                    windows.put(
                        JSONObject()
                            .put("index", windowIndex)
                            .put("plan", plan.evidence())
                            .put("finalizedFrames", finalized.frames)
                            .put("outputFramesAfterWindow", outputFrames)
                            .put("stages", stage)
                            .put("total", windowStarted.elapsed().evidence())
                            .put("process", processSnapshot())
                            .put("thermalStatus", thermalStatus()),
                    )
                    publishJson(
                        File(attemptRoot, "progress.json"),
                        JSONObject()
                            .put("status", "running")
                            .put("completedWindows", completedWindows)
                            .put("outputFrames", outputFrames)
                            .put("lastWindow", windows.getJSONObject(windows.length() - 1)),
                    )
                    if (cancelAfterWindows > 0 && completedWindows >= cancelAfterWindows) {
                        throw IntentionalCancellation(completedWindows)
                    }
                }
            }
            require(outputFrames == selectedFrames) {
                "OLA emitted $outputFrames frames; expected $selectedFrames."
            }
            require(carryLength == 0) { "OLA retained $carryLength frames after EOF." }
        } catch (error: Throwable) {
            failure = error
            cancelled = error is IntentionalCancellation
        } finally {
            val closeFailure = closeResources(
                writers = writers,
                inputBuffers = inputBuffers.values,
                outputBuffers = outputBuffers.values,
                model = model,
                environment = environment,
                dsp = dsp,
                parityDsp = parityDsp,
            )
            if (closeFailure != null) {
                if (failure == null) failure = closeFailure else failure?.addSuppressed(closeFailure)
            }
        }

        val terminalFailure = failure
        if (terminalFailure != null) {
            require(stagingDir.deleteRecursively()) {
                "Could not clean partial E2E outputs after failure."
            }
            val failureReport = attemptReport(
                status = if (cancelled) "cancelled" else "failed",
                config = config,
                selectedFrames = selectedFrames,
                completedWindows = completedWindows,
                outputFrames = outputFrames,
                prepare = prepareEvidence,
                windows = windows,
                total = totalStarted.elapsed(),
                e2eTotal = e2eStarted?.elapsed(),
                coreBenchmark = coreBenchmarkEvidence,
                outputs = JSONObject.NULL,
                failure = terminalFailure,
                stagingCleanupComplete = !stagingDir.exists(),
                cancelAfterWindows = cancelAfterWindows,
            )
            publishJson(File(attemptRoot, "attempt-report.json"), failureReport)
            if (cancelled) return failureReport
            throw terminalFailure
        }

        val commit = timed {
            stemOrder.forEach { stem ->
                val partial = File(stagingDir, "$stem.wav.partial")
                val complete = File(stagingDir, "$stem.wav")
                require(partial.renameTo(complete)) { "Could not finalize ${partial.name}." }
            }
            require(!outputDir.exists())
            require(stagingDir.renameTo(outputDir)) { "Could not commit E2E output directory." }
        }
        val outputEvidence = timed {
            JSONObject().also { result ->
                stemOrder.forEachIndexed { index, stem ->
                    val file = File(outputDir, "$stem.wav")
                    val stats = stemStats[index]
                    require(file.length() == WAV_HEADER_BYTES + selectedFrames.toLong() * BYTES_PER_FRAME)
                    require(stats.sampleCount == selectedFrames.toLong() * CHANNEL_COUNT)
                    require(stats.nonFiniteCount == 0L)
                    result.put(
                        stem,
                        stats.evidence()
                            .put("path", file.absolutePath)
                            .put("byteSize", file.length())
                            .put("sha256", sha256(file)),
                    )
                }
            }
        }
        val completedReport = attemptReport(
            status = "complete",
            config = config,
            selectedFrames = selectedFrames,
            completedWindows = completedWindows,
            outputFrames = outputFrames,
            prepare = prepareEvidence
                .put("outputCommit", commit.timing.evidence())
                .put("outputHashAndValidation", outputEvidence.timing.evidence()),
            windows = windows,
            total = totalStarted.elapsed(),
            e2eTotal = requireNotNull(e2eStarted).elapsed(),
            coreBenchmark = coreBenchmarkEvidence,
            outputs = outputEvidence.value,
            failure = null,
            stagingCleanupComplete = !stagingDir.exists(),
            cancelAfterWindows = cancelAfterWindows,
        )
        publishJson(File(attemptRoot, "attempt-report.json"), completedReport)
        return completedReport
    }

    private fun attemptReport(
        status: String,
        config: Config,
        selectedFrames: Int,
        completedWindows: Int,
        outputFrames: Int,
        prepare: JSONObject,
        windows: JSONArray,
        total: StageTiming,
        e2eTotal: StageTiming?,
        coreBenchmark: Any,
        outputs: Any,
        failure: Throwable?,
        stagingCleanupComplete: Boolean,
        cancelAfterWindows: Int,
    ): JSONObject {
        val duration = selectedFrames.toDouble() / SAMPLE_RATE
        return JSONObject()
            .put("status", status)
            .put("backend", "CPU")
            .put("threads", config.threads)
            .put("litertCpuThreads", config.threads)
            .put("dspIstft", istftExecutionEvidence(config, stemCountFor(config.modelVariant)))
            .put("selectedFrames", selectedFrames)
            .put("durationNumeratorFrames", selectedFrames)
            .put("durationDenominatorSampleRate", SAMPLE_RATE)
            .put("durationSeconds", duration)
            .put("expectedWindows", windowPlans(selectedFrames).size)
            .put("completedWindows", completedWindows)
            .put("outputFrames", outputFrames)
            .put("cancelAfterWindows", cancelAfterWindows)
            .put("prepare", prepare)
            .put("windows", windows)
            .put("stageSummary", summarizeWindowStages(windows))
            .put("coreBenchmark", coreBenchmark)
            .put("total", total.evidence())
            .put("e2eTotal", e2eTotal?.evidence() ?: JSONObject.NULL)
            .put(
                "realtimeFactor",
                if (status == "complete") {
                    requireNotNull(e2eTotal).wallMs / 1000.0 / duration
                } else {
                    JSONObject.NULL
                },
            )
            .put("outputs", outputs)
            .put("stagingCleanupComplete", stagingCleanupComplete)
            .put("failure", failure?.let(::failureEvidence) ?: JSONObject.NULL)
            .put("finalProcess", processSnapshot())
            .put("finalThermalStatus", thermalStatus())
    }

    private fun compareFloatBits(
        candidate: FloatArray,
        reference: FloatArray,
    ): FloatBitComparison {
        require(candidate.size == reference.size)
        val candidateDigest = MessageDigest.getInstance("SHA-256")
        val referenceDigest = MessageDigest.getInstance("SHA-256")
        val candidateBuffer = ByteBuffer.allocate(RAW_FLOAT_HASH_BUFFER_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        val referenceBuffer = ByteBuffer.allocate(RAW_FLOAT_HASH_BUFFER_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
        var mismatchCount = 0L
        var firstMismatchIndex = -1
        var candidateNonFiniteCount = 0L
        var referenceNonFiniteCount = 0L
        candidate.indices.forEach { index ->
            if (candidateBuffer.remaining() < Int.SIZE_BYTES) {
                candidateDigest.update(candidateBuffer.array(), 0, candidateBuffer.position())
                referenceDigest.update(referenceBuffer.array(), 0, referenceBuffer.position())
                candidateBuffer.clear()
                referenceBuffer.clear()
            }
            val candidateValue = candidate[index]
            val referenceValue = reference[index]
            val candidateBits = candidateValue.toRawBits()
            val referenceBits = referenceValue.toRawBits()
            candidateBuffer.putInt(candidateBits)
            referenceBuffer.putInt(referenceBits)
            if (candidateBits != referenceBits) {
                mismatchCount++
                if (firstMismatchIndex < 0) firstMismatchIndex = index
            }
            if (!candidateValue.isFinite()) candidateNonFiniteCount++
            if (!referenceValue.isFinite()) referenceNonFiniteCount++
        }
        candidateDigest.update(candidateBuffer.array(), 0, candidateBuffer.position())
        referenceDigest.update(referenceBuffer.array(), 0, referenceBuffer.position())
        return FloatBitComparison(
            elementCount = candidate.size,
            mismatchCount = mismatchCount,
            firstMismatchIndex = firstMismatchIndex,
            candidateNonFiniteCount = candidateNonFiniteCount,
            referenceNonFiniteCount = referenceNonFiniteCount,
            candidateRawSha256 = candidateDigest.digest().toHex(),
            referenceRawSha256 = referenceDigest.digest().toHex(),
        )
    }

    private fun readNormalizedWindow(
        input: RandomAccessFile,
        wav: CanonicalPcm16Wav,
        plan: WindowPlan,
        normalization: Normalization,
        reusableOutput: FloatArray? = null,
    ): FloatArray {
        val output = reusableOutput ?: FloatArray(CHANNEL_COUNT * WINDOW_SAMPLES)
        require(output.size == CHANNEL_COUNT * WINDOW_SAMPLES)
        output.fill(0f)
        val copyFrames = plan.sourceEnd - plan.sourceStart
        require(copyFrames + plan.padLeft + plan.padRight == WINDOW_SAMPLES)
        input.seek(wav.dataOffset + plan.sourceStart.toLong() * BYTES_PER_FRAME)
        val bytes = ByteArray(minOf(IO_BUFFER_BYTES, copyFrames * BYTES_PER_FRAME))
        var frame = 0
        while (frame < copyFrames) {
            val framesThisRead = minOf(bytes.size / BYTES_PER_FRAME, copyFrames - frame)
            val byteCount = framesThisRead * BYTES_PER_FRAME
            input.readFully(bytes, 0, byteCount)
            var byteIndex = 0
            repeat(framesThisRead) {
                val destination = plan.padLeft + frame
                val left = littleEndianShort(bytes, byteIndex).toFloat() / PCM_SCALE_FLOAT
                val right = littleEndianShort(bytes, byteIndex + 2).toFloat() / PCM_SCALE_FLOAT
                output[destination] = (left - normalization.mean) / normalization.scale
                output[WINDOW_SAMPLES + destination] =
                    (right - normalization.mean) / normalization.scale
                byteIndex += BYTES_PER_FRAME
                frame++
            }
        }
        return output
    }

    private fun applyStreamingOla(
        combined: FloatArray,
        plan: WindowPlan,
        trackFrames: Int,
        normalization: Normalization,
        triangleWeight: FloatArray,
        carry: FloatArray,
        carryWeight: FloatArray,
        carryLength: Int,
        planeCount: Int,
    ): FinalizedChunk {
        require(combined.size == planeCount * WINDOW_SAMPLES)
        if (plan.offset == 0) {
            require(carryLength == 0)
        } else {
            require(carryLength in 1..OVERLAP_SAMPLES)
        }
        val hasNext = plan.offset + STRIDE_SAMPLES < trackFrames
        val finalizedFrames = if (hasNext) STRIDE_SAMPLES else plan.actualSamples
        val nextCarryLength = if (hasNext) plan.actualSamples - STRIDE_SAMPLES else 0
        require(finalizedFrames <= plan.actualSamples)
        require(nextCarryLength in 0..OVERLAP_SAMPLES)
        repeat(planeCount) { plane ->
            val sourceBase = plane * WINDOW_SAMPLES + plan.cropLeft
            val destinationBase = plane * WINDOW_SAMPLES
            val carryBase = plane * OVERLAP_SAMPLES
            repeat(finalizedFrames) { frame ->
                var numerator = combined[sourceBase + frame] * triangleWeight[frame]
                var denominator = triangleWeight[frame]
                if (frame < carryLength) {
                    numerator += carry[carryBase + frame]
                    denominator += carryWeight[frame]
                }
                require(denominator > 0f)
                val normalized = numerator / denominator
                combined[destinationBase + frame] =
                    normalized * normalization.scale + normalization.mean
            }
            if (hasNext) {
                repeat(nextCarryLength) { frame ->
                    val activeFrame = STRIDE_SAMPLES + frame
                    carry[carryBase + frame] =
                        combined[sourceBase + activeFrame] * triangleWeight[activeFrame]
                }
                carry.fill(0f, carryBase + nextCarryLength, carryBase + OVERLAP_SAMPLES)
            }
        }
        if (hasNext) {
            repeat(nextCarryLength) { frame ->
                carryWeight[frame] = triangleWeight[STRIDE_SAMPLES + frame]
            }
            carryWeight.fill(0f, nextCarryLength, OVERLAP_SAMPLES)
        } else {
            carry.fill(0f)
            carryWeight.fill(0f)
        }
        return FinalizedChunk(
            frames = finalizedFrames,
            nextCarryLength = nextCarryLength,
        )
    }

    private fun encodePcm16(
        combined: FloatArray,
        frames: Int,
        stats: Array<StemStats>,
        stemCount: Int,
    ): Array<ByteArray> = Array(stemCount) { stem ->
        ByteArray(frames * BYTES_PER_FRAME).also { bytes ->
            val leftBase = (stem * CHANNEL_COUNT) * WINDOW_SAMPLES
            val rightBase = leftBase + WINDOW_SAMPLES
            var byteIndex = 0
            repeat(frames) { frame ->
                byteIndex = encodeSample(combined[leftBase + frame], bytes, byteIndex, stats[stem])
                byteIndex = encodeSample(combined[rightBase + frame], bytes, byteIndex, stats[stem])
            }
        }
    }

    private fun fusedPostprocessAndWrite(
        frequencyWaveform: FloatArray,
        timeWaveform: FloatArray,
        plan: WindowPlan,
        trackFrames: Int,
        normalization: Normalization,
        triangleWeight: FloatArray,
        carry: FloatArray,
        carryWeight: FloatArray,
        carryLength: Int,
        stemCount: Int,
        stats: Array<StemStats>,
        writers: List<WavFileWriter>,
        pcmWorkspaces: Array<ByteArray>,
    ): FinalizedChunk {
        val planeCount = stemCount * CHANNEL_COUNT
        require(frequencyWaveform.size == planeCount * WINDOW_SAMPLES)
        require(timeWaveform.size == frequencyWaveform.size)
        require(pcmWorkspaces.size == stemCount && writers.size == stemCount)
        if (plan.offset == 0) require(carryLength == 0) else require(carryLength in 1..OVERLAP_SAMPLES)
        val hasNext = plan.offset + STRIDE_SAMPLES < trackFrames
        val finalizedFrames = if (hasNext) STRIDE_SAMPLES else plan.actualSamples
        val nextCarryLength = if (hasNext) plan.actualSamples - STRIDE_SAMPLES else 0
        require(finalizedFrames <= plan.actualSamples)
        require(nextCarryLength in 0..OVERLAP_SAMPLES)

        repeat(stemCount) { stem ->
            val leftPlane = stem * CHANNEL_COUNT
            val rightPlane = leftPlane + 1
            val leftSource = leftPlane * WINDOW_SAMPLES + plan.cropLeft
            val rightSource = rightPlane * WINDOW_SAMPLES + plan.cropLeft
            val leftCarry = leftPlane * OVERLAP_SAMPLES
            val rightCarry = rightPlane * OVERLAP_SAMPLES
            val bytes = pcmWorkspaces[stem]
            var byteIndex = 0
            repeat(finalizedFrames) { frame ->
                val weight = triangleWeight[frame]
                var denominator = weight
                var left = (frequencyWaveform[leftSource + frame] + timeWaveform[leftSource + frame]) * weight
                var right = (frequencyWaveform[rightSource + frame] + timeWaveform[rightSource + frame]) * weight
                if (frame < carryLength) {
                    left += carry[leftCarry + frame]
                    right += carry[rightCarry + frame]
                    denominator += carryWeight[frame]
                }
                require(denominator > 0f)
                val leftOutput = (left / denominator) * normalization.scale + normalization.mean
                val rightOutput = (right / denominator) * normalization.scale + normalization.mean
                byteIndex = encodeSample(leftOutput, bytes, byteIndex, stats[stem])
                byteIndex = encodeSample(rightOutput, bytes, byteIndex, stats[stem])
            }
            if (hasNext) {
                repeat(nextCarryLength) { frame ->
                    val activeFrame = STRIDE_SAMPLES + frame
                    val weight = triangleWeight[activeFrame]
                    carry[leftCarry + frame] =
                        (frequencyWaveform[leftSource + activeFrame] +
                            timeWaveform[leftSource + activeFrame]) * weight
                    carry[rightCarry + frame] =
                        (frequencyWaveform[rightSource + activeFrame] +
                            timeWaveform[rightSource + activeFrame]) * weight
                }
                carry.fill(0f, leftCarry + nextCarryLength, leftCarry + OVERLAP_SAMPLES)
                carry.fill(0f, rightCarry + nextCarryLength, rightCarry + OVERLAP_SAMPLES)
            }
            writers[stem].writePcm16(bytes, 0, byteIndex)
        }
        if (hasNext) {
            repeat(nextCarryLength) { frame ->
                carryWeight[frame] = triangleWeight[STRIDE_SAMPLES + frame]
            }
            carryWeight.fill(0f, nextCarryLength, OVERLAP_SAMPLES)
        } else {
            carry.fill(0f)
            carryWeight.fill(0f)
        }
        return FinalizedChunk(finalizedFrames, nextCarryLength)
    }

    private fun encodeSample(
        value: Float,
        destination: ByteArray,
        offset: Int,
        stats: StemStats,
    ): Int {
        stats.add(value)
        require(value.isFinite()) { "Separated PCM contains NaN or infinity." }
        val clipped = value.coerceIn(-1f, 1f)
        val pcm = (clipped * Short.MAX_VALUE).roundToInt()
            .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
        destination[offset] = (pcm and 0xff).toByte()
        destination[offset + 1] = ((pcm ushr 8) and 0xff).toByte()
        return offset + Short.SIZE_BYTES
    }

    private fun windowPlans(trackFrames: Int): List<WindowPlan> = buildList {
        var offset = 0
        while (offset < trackFrames) {
            val actual = minOf(WINDOW_SAMPLES, trackFrames - offset)
            val delta = WINDOW_SAMPLES - actual
            val contextStart = offset - delta / 2
            val contextEnd = contextStart + WINDOW_SAMPLES
            val sourceStart = maxOf(0, contextStart)
            val sourceEnd = minOf(trackFrames, contextEnd)
            add(
                WindowPlan(
                    offset = offset,
                    actualSamples = actual,
                    contextStart = contextStart,
                    contextEnd = contextEnd,
                    sourceStart = sourceStart,
                    sourceEnd = sourceEnd,
                    padLeft = sourceStart - contextStart,
                    padRight = contextEnd - sourceEnd,
                    cropLeft = delta / 2,
                    cropRight = delta - delta / 2,
                ),
            )
            offset += STRIDE_SAMPLES
        }
    }

    private fun triangleWeight(): FloatArray {
        val half = WINDOW_SAMPLES / 2
        return FloatArray(WINDOW_SAMPLES) { index ->
            if (index < half) {
                (index + 1).toFloat() / half
            } else {
                (WINDOW_SAMPLES - index).toFloat() / half
            }
        }
    }

    private fun tensorAbiEvidence(abi: List<TensorAbi>): JSONArray = JSONArray().also { array ->
        abi.forEach { tensor ->
            array.put(
                JSONObject()
                    .put("name", tensor.name)
                    .put("dtype", "float32")
                    .put("shape", JSONArray(tensor.shape))
                    .put("elementCount", tensor.elementCount),
            )
        }
    }

    private fun checkTensorType(type: TensorType, expected: TensorAbi) {
        require(type.elementType == TensorType.ElementType.FLOAT)
        require(requireNotNull(type.layout).dimensions == expected.shape) {
            "${expected.name} shape mismatch: ${type.layout?.dimensions} != ${expected.shape}"
        }
    }

    private fun contractEvidence(trackFrames: Int, stemOrder: List<String>): JSONObject = JSONObject()
        .put("sampleRate", SAMPLE_RATE)
        .put("trackFrames", trackFrames)
        .put("durationNumeratorFrames", trackFrames)
        .put("durationDenominatorSampleRate", SAMPLE_RATE)
        .put("durationSeconds", trackFrames.toDouble() / SAMPLE_RATE)
        .put("windowSamples", WINDOW_SAMPLES)
        .put("strideSamples", STRIDE_SAMPLES)
        .put("overlapSamples", OVERLAP_SAMPLES)
        .put("overlap", 0.25)
        .put("transitionPower", 1.0)
        .put("windowCount", windowPlans(trackFrames).size)
        .put("windowOrder", "ascending-offset")
        .put("tailPadding", "official-demucs-tensor-chunk-centered-zero-pad")
        .put("tailCrop", "center-crop-to-actual-samples")
        .put("tailWeightRule", "triangle-prefix")
        .put("globalNormalizationReference", "mean-across-stereo-channels")
        .put("globalNormalizationCorrection", 1)
        .put("normalizationBeforeChunkPadding", true)
        .put("inverseNormalizationAfterOla", true)
        .put("stemOrder", JSONArray(stemOrder))
        .put("streamingCarryFrames", OVERLAP_SAMPLES)
        .put(
            "streamingCarryFloatBytes",
            stemOrder.size.toLong() * CHANNEL_COUNT * OVERLAP_SAMPLES * Float.SIZE_BYTES,
        )

    private fun runtimeEvidence(): JSONObject = JSONObject()
        .put("litertVersion", BuildConfig.BENCHMARK_RUNTIME_VERSION)
        .put("runtimeId", BuildConfig.BENCHMARK_RUNTIME_ID)
        .put("runtimeArtifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256)
        .put("sourceRevision", BuildConfig.BENCHMARK_SOURCE_REVISION)
        .put("sourceDirty", BuildConfig.BENCHMARK_SOURCE_DIRTY)
        .put("device", Build.DEVICE)
        .put("model", Build.MODEL)
        .put("manufacturer", Build.MANUFACTURER)
        .put("socManufacturer", Build.SOC_MANUFACTURER)
        .put("socModel", Build.SOC_MODEL)
        .put("sdk", Build.VERSION.SDK_INT)
        .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList()))

    private fun processSnapshot(): JSONObject {
        val runtime = Runtime.getRuntime()
        val memory = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        return JSONObject()
            .put("elapsedRealtimeMs", SystemClock.elapsedRealtime())
            .put("processCpuMs", android.os.Process.getElapsedCpuTime())
            .put("pssKb", memory.totalPss)
            .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
            .put("javaHeapUsedBytes", runtime.totalMemory() - runtime.freeMemory())
            .put("javaHeapCommittedBytes", runtime.totalMemory())
            .put("javaHeapMaxBytes", runtime.maxMemory())
    }

    private fun thermalStatus(): Int? = if (Build.VERSION.SDK_INT >= 29) {
        context.getSystemService(PowerManager::class.java)?.currentThermalStatus
    } else {
        null
    }

    private fun summarizeWindowStages(windows: JSONArray): JSONObject {
        val values = linkedMapOf<String, MutableList<Double>>()
        repeat(windows.length()) { index ->
            val stages = windows.getJSONObject(index).getJSONObject("stages")
            stages.keys().forEach { key ->
                values.getOrPut(key) { mutableListOf() }
                    .add(stages.getJSONObject(key).getDouble("wallMs"))
            }
        }
        return JSONObject().also { result ->
            values.forEach { (stage, samples) ->
                val sorted = samples.sorted()
                result.put(
                    stage,
                    JSONObject()
                        .put("count", sorted.size)
                        .put("totalWallMs", sorted.sum())
                        .put("minWallMs", sorted.first())
                        .put("medianWallMs", sorted[sorted.size / 2])
                        .put("maxWallMs", sorted.last()),
                )
            }
        }
    }

    private fun summarizeTimings(samples: JSONArray): JSONObject {
        if (samples.length() == 0) return JSONObject().put("count", 0)
        val wall = mutableListOf<Double>()
        val processCpu = mutableListOf<Long>()
        val threadCpu = mutableListOf<Double>()
        repeat(samples.length()) { index ->
            val timing = samples.getJSONObject(index).getJSONObject("timing")
            wall += timing.getDouble("wallMs")
            processCpu += timing.getLong("processCpuMs")
            threadCpu += timing.getDouble("threadCpuMs")
        }
        fun summary(values: List<Double>): JSONObject {
            val sorted = values.sorted()
            return JSONObject()
                .put("minimum", sorted.first())
                .put("median", sorted[sorted.size / 2])
                .put("mean", sorted.sum() / sorted.size)
                .put("maximum", sorted.last())
        }
        return JSONObject()
            .put("count", samples.length())
            .put("wallMs", summary(wall))
            .put("processCpuMs", summary(processCpu.map(Long::toDouble)))
            .put("threadCpuMs", summary(threadCpu))
    }

    private fun closeResources(
        writers: List<WavFileWriter>,
        inputBuffers: Collection<TensorBuffer>,
        outputBuffers: Collection<TensorBuffer>,
        model: CompiledModel?,
        environment: Environment?,
        dsp: HtdemucsDsp?,
        parityDsp: HtdemucsDsp?,
    ): Throwable? {
        var failure: Throwable? = null
        fun close(action: () -> Unit) {
            try {
                action()
            } catch (error: Throwable) {
                if (failure == null) failure = error else failure?.addSuppressed(error)
            }
        }
        writers.asReversed().forEach { writer -> close(writer::close) }
        inputBuffers.forEach { buffer -> close(buffer::close) }
        outputBuffers.forEach { buffer -> close(buffer::close) }
        model?.let { close(it::close) }
        environment?.let { close(it::close) }
        dsp?.let { close(it::close) }
        parityDsp?.let { close(it::close) }
        return failure
    }

    private fun istftExecutionEvidence(config: Config, stemCount: Int): JSONObject = JSONObject()
        .put("mode", config.istftMode.wireValue)
        .put("requestedWorkers", config.istftWorkers)
        .put("effectiveWorkers", minOf(config.istftWorkers, stemCount * CHANNEL_COUNT))
        .put("laneCount", stemCount * CHANNEL_COUNT)
        .put("fftImplementation", "JTransforms-3.1-FloatFFT_1D-complexInverse")
        .put("jTransformsInternalThreadPolicy", "library-default-global")
        .put("jTransformsGlobalThreads", ConcurrencyUtils.getNumberOfThreads())
        .put("jTransformsProcessorCount", ConcurrencyUtils.getNumberOfProcessors())
        .put("jTransforms1dFft2ThreadsThreshold", CommonUtils.getThreadsBeginN_1D_FFT_2Threads())
        .put("jTransforms1dFft4ThreadsThreshold", CommonUtils.getThreadsBeginN_1D_FFT_4Threads())
        .put("complexTransformArrayLength", HtdemucsDsp.N_FFT * 2)
        .put("executorOwned", config.istftMode == HtdemucsDsp.IstftMode.PARALLEL_LANES)
        .put("floatParityCheckRequested", config.validateIstftFloatParity)
        .put("postprocessMode", config.postprocessMode.wireValue)
        .put("reuseWaveformWorkspace", config.postprocessMode == PostprocessMode.FUSED_REUSE)
        .put("reuseDspIoWorkspaces", config.postprocessMode == PostprocessMode.FUSED_REUSE)
        .put("reusePcmByteBuffers", config.postprocessMode == PostprocessMode.FUSED_REUSE)
        .put("fusedBranchOlaPcmWrite", config.postprocessMode == PostprocessMode.FUSED_REUSE)
        .put("tensorBufferReadIntoAvailable", false)

    private fun stemCountFor(modelVariant: String): Int =
        MODEL_IDENTITIES.getValue(modelVariant).stemOrder.size

    private fun failureEvidence(error: Throwable): JSONObject = JSONObject()
        .put("type", error.javaClass.name)
        .put("message", error.message ?: JSONObject.NULL)
        .put("stackTrace", error.stackTraceToString())

    private fun validateConfig(config: Config) {
        require(config.audioFileName == File(config.audioFileName).name)
        require(config.audioFileName.isNotBlank())
        require(config.durationSeconds == null || config.durationSeconds >= 0)
        require(config.frameLimit == null || config.frameLimit > 1)
        require(config.durationSeconds == null || config.frameLimit == null) {
            "durationSeconds and frameLimit are mutually exclusive."
        }
        require(config.threads in 1..16)
        require(config.coreWarmupRuns in 0..100)
        require(config.coreMeasuredRuns in 0..100)
        require((config.coreWarmupRuns == 0) == (config.coreMeasuredRuns == 0)) {
            "Core warmup and measured run counts must both be zero or both be positive."
        }
        require(RUN_ID.matches(config.runId)) { "runId contains unsupported characters." }
        require(config.modelVariant in MODEL_IDENTITIES) {
            "Unsupported model variant '${config.modelVariant}'; expected one of " +
                MODEL_IDENTITIES.keys.sorted().joinToString()
        }
        val laneCount = stemCountFor(config.modelVariant) * CHANNEL_COUNT
        require(config.istftWorkers in 1..minOf(HtdemucsDsp.MAX_ISTFT_WORKERS, laneCount))
        require(
            config.istftMode != HtdemucsDsp.IstftMode.SERIAL || config.istftWorkers == 1,
        ) { "Serial iSTFT requires exactly one worker." }
        require(
            config.istftMode != HtdemucsDsp.IstftMode.PARALLEL_LANES ||
                config.istftWorkers >= 2,
        ) { "Parallel-lanes iSTFT requires at least two workers." }
        require(
            !config.validateIstftFloatParity ||
                config.istftMode == HtdemucsDsp.IstftMode.PARALLEL_LANES,
        ) { "Raw-float iSTFT parity validation requires parallel-lanes mode." }
        config.expectedAudioSha256?.let { require(SHA256.matches(it)) }
        require(config.cancelAfterWindows >= 0)
        require(!config.resumeAfterCancel || config.cancelAfterWindows > 0)
    }

    private fun publishJson(file: File, value: JSONObject) {
        file.parentFile?.mkdirs()
        val partial = File(requireNotNull(file.parentFile), "${file.name}.partial")
        partial.writeText(value.toString(2), Charsets.UTF_8)
        Files.move(
            partial.toPath(),
            file.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
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
        return digest.digest().toHex()
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .toHex()

    private fun ByteArray.toHex(): String = joinToString("") { byte -> "%02x".format(byte) }

    private fun MediaFormat.optionalInteger(key: String): Int? =
        if (containsKey(key)) getInteger(key) else null

    private fun MediaFormat.optionalLong(key: String): Long? =
        if (containsKey(key)) getLong(key) else null

    private fun <T> timed(block: () -> T): TimedValue<T> {
        val start = StageStart()
        return TimedValue(block(), start.elapsed())
    }

    private class StageStart {
        private val wallNanos = SystemClock.elapsedRealtimeNanos()
        private val processCpuMs = android.os.Process.getElapsedCpuTime()
        private val threadCpuNanos = Debug.threadCpuTimeNanos()
        private val allocatedBytes = runtimeCounter("art.gc.bytes-allocated")
        private val gcCount = runtimeCounter("art.gc.gc-count")

        fun elapsed(): StageTiming = StageTiming(
            wallMs = (SystemClock.elapsedRealtimeNanos() - wallNanos) / 1_000_000.0,
            processCpuMs = android.os.Process.getElapsedCpuTime() - processCpuMs,
            threadCpuMs = (Debug.threadCpuTimeNanos() - threadCpuNanos) / 1_000_000.0,
            allocatedBytesDelta = counterDelta(allocatedBytes, runtimeCounter("art.gc.bytes-allocated")),
            gcCountDelta = counterDelta(gcCount, runtimeCounter("art.gc.gc-count")),
        )
    }

    private data class StageTiming(
        val wallMs: Double,
        val processCpuMs: Long,
        val threadCpuMs: Double,
        val allocatedBytesDelta: Long?,
        val gcCountDelta: Long?,
    ) {
        operator fun plus(other: StageTiming): StageTiming = StageTiming(
            wallMs = wallMs + other.wallMs,
            processCpuMs = processCpuMs + other.processCpuMs,
            threadCpuMs = threadCpuMs + other.threadCpuMs,
            allocatedBytesDelta = nullableSum(allocatedBytesDelta, other.allocatedBytesDelta),
            gcCountDelta = nullableSum(gcCountDelta, other.gcCountDelta),
        )

        fun evidence(): JSONObject = JSONObject()
            .put("wallMs", wallMs)
            .put("cpuMs", processCpuMs)
            .put("processCpuMs", processCpuMs)
            .put("threadCpuMs", threadCpuMs)
            .put("allocatedBytesDelta", allocatedBytesDelta ?: JSONObject.NULL)
            .put("gcCountDelta", gcCountDelta ?: JSONObject.NULL)
    }

    private data class TimedValue<T>(
        val value: T,
        val timing: StageTiming,
    )

    private data class FloatBitComparison(
        val elementCount: Int,
        val mismatchCount: Long,
        val firstMismatchIndex: Int,
        val candidateNonFiniteCount: Long,
        val referenceNonFiniteCount: Long,
        val candidateRawSha256: String,
        val referenceRawSha256: String,
    ) {
        fun evidence(): JSONObject = JSONObject()
            .put("elementCount", elementCount)
            .put("byteCount", elementCount.toLong() * Float.SIZE_BYTES)
            .put("byteOrder", "little-endian-raw-float-bits")
            .put("mismatchCount", mismatchCount)
            .put(
                "firstMismatchIndex",
                if (firstMismatchIndex >= 0) firstMismatchIndex else JSONObject.NULL,
            )
            .put("candidateNonFiniteCount", candidateNonFiniteCount)
            .put("referenceNonFiniteCount", referenceNonFiniteCount)
            .put("candidateRawSha256", candidateRawSha256)
            .put("referenceRawSha256", referenceRawSha256)
    }

    private data class PreparedCanonicalAudio(
        val wav: CanonicalPcm16Wav,
        val canonicalFrames: Int,
        val canonicalPcmSha256: String,
        val mediaFormat: JSONObject,
        val decodedEvidence: JSONObject,
        val channelMapping: String,
        val resampled: Boolean,
        val sourceSampleRate: Int,
        val sourceChannelCount: Int,
        val canonicalWavBytes: Long,
        val timingEvidence: JSONObject,
    ) {
        fun canonicalEvidence(): JSONObject = JSONObject()
            .put("sampleRate", SAMPLE_RATE)
            .put("channelCount", CHANNEL_COUNT)
            .put("bitsPerSample", 16)
            .put("frames", canonicalFrames)
            .put("pcmBytes", canonicalFrames.toLong() * BYTES_PER_FRAME)
            .put("pcmEncoding", "signed-pcm16-le")
            .put("pcmSha256", canonicalPcmSha256)
            .put("sourceSampleRate", sourceSampleRate)
            .put("sourceChannelCount", sourceChannelCount)
            .put("resampled", resampled)
            .put("channelMapping", channelMapping)
            .put("temporaryWavPath", wav.file.absolutePath)
            .put("temporaryWavBytes", canonicalWavBytes)
            .put("temporaryWavRetainedAfterRun", false)
    }

    private data class DecodedStage(
        val resampledAudio: DecodedPcmAudio,
        val decodedEvidence: JSONObject,
        val sourceSampleRate: Int,
        val sourceChannelCount: Int,
        val resampled: Boolean,
        val decodeTiming: StageTiming,
        val decodedPcmHashTiming: StageTiming,
        val resampleTiming: StageTiming,
    )

    private data class CanonicalWavWrite(
        val frames: Int,
        val pcmBytes: Long,
        val pcmSha256: String,
        val channelMapping: String,
    )

    private data class FinalizedChunk(
        val frames: Int,
        val nextCarryLength: Int,
    )

    private data class WindowPlan(
        val offset: Int,
        val actualSamples: Int,
        val contextStart: Int,
        val contextEnd: Int,
        val sourceStart: Int,
        val sourceEnd: Int,
        val padLeft: Int,
        val padRight: Int,
        val cropLeft: Int,
        val cropRight: Int,
    ) {
        fun evidence(): JSONObject = JSONObject()
            .put("offset", offset)
            .put("actualSamples", actualSamples)
            .put("contextStart", contextStart)
            .put("contextEnd", contextEnd)
            .put("sourceStart", sourceStart)
            .put("sourceEnd", sourceEnd)
            .put("padLeft", padLeft)
            .put("padRight", padRight)
            .put("cropLeft", cropLeft)
            .put("cropRight", cropRight)
    }

    private data class Normalization(
        val mean: Float,
        val sampleStandardDeviation: Float,
        val scale: Float,
        val selectedPcmSha256: String,
    )

    private class StemStats {
        var sampleCount = 0L
            private set
        var nonFiniteCount = 0L
            private set
        private var clippedCount = 0L
        private var squareSum = 0.0
        private var absoluteSum = 0.0
        private var min = Double.POSITIVE_INFINITY
        private var max = Double.NEGATIVE_INFINITY

        fun add(value: Float) {
            sampleCount++
            val number = value.toDouble()
            if (!number.isFinite()) {
                nonFiniteCount++
                return
            }
            if (number < -1.0 || number > 1.0) clippedCount++
            squareSum += number * number
            absoluteSum += kotlin.math.abs(number)
            min = minOf(min, number)
            max = maxOf(max, number)
        }

        fun evidence(): JSONObject {
            val finiteCount = sampleCount - nonFiniteCount
            return JSONObject()
                .put("sampleCount", sampleCount)
                .put("nonFiniteCount", nonFiniteCount)
                .put("clippedSampleCount", clippedCount)
                .put("absoluteMean", if (finiteCount > 0) absoluteSum / finiteCount else JSONObject.NULL)
                .put("rms", if (finiteCount > 0) sqrt(squareSum / finiteCount) else JSONObject.NULL)
                .put("min", if (finiteCount > 0) min else JSONObject.NULL)
                .put("max", if (finiteCount > 0) max else JSONObject.NULL)
        }
    }

    private data class TensorAbi(
        val name: String,
        val shape: List<Int>,
    ) {
        val elementCount: Long = shape.fold(1L) { product, dimension -> product * dimension }
    }

    private class IntentionalCancellation(windows: Int) :
        RuntimeException("Intentional cancellation after $windows completed windows.")

    private data class CanonicalPcm16Wav(
        val file: File,
        val sampleRate: Int,
        val channelCount: Int,
        val bitsPerSample: Int,
        val dataOffset: Long,
        val dataBytes: Long,
    ) {
        val frameCount: Long = dataBytes / BYTES_PER_FRAME

        fun scanNormalization(frames: Int): Normalization {
            val digest = MessageDigest.getInstance("SHA-256")
            var referenceSum = 0.0
            RandomAccessFile(file, "r").use { input ->
                input.seek(dataOffset)
                val buffer = ByteArray(IO_BUFFER_BYTES)
                var remainingFrames = frames
                while (remainingFrames > 0) {
                    val readFrames = minOf(buffer.size / BYTES_PER_FRAME, remainingFrames)
                    val byteCount = readFrames * BYTES_PER_FRAME
                    input.readFully(buffer, 0, byteCount)
                    digest.update(buffer, 0, byteCount)
                    var byteIndex = 0
                    repeat(readFrames) {
                        val left = littleEndianShort(buffer, byteIndex) / PCM_SCALE
                        val right = littleEndianShort(buffer, byteIndex + 2) / PCM_SCALE
                        referenceSum += (left + right) * 0.5
                        byteIndex += BYTES_PER_FRAME
                    }
                    remainingFrames -= readFrames
                }
            }
            require(frames > 1)
            val mean = referenceSum / frames
            var squaredDeviationSum = 0.0
            RandomAccessFile(file, "r").use { input ->
                input.seek(dataOffset)
                val buffer = ByteArray(IO_BUFFER_BYTES)
                var remainingFrames = frames
                while (remainingFrames > 0) {
                    val readFrames = minOf(buffer.size / BYTES_PER_FRAME, remainingFrames)
                    val byteCount = readFrames * BYTES_PER_FRAME
                    input.readFully(buffer, 0, byteCount)
                    var byteIndex = 0
                    repeat(readFrames) {
                        val left = littleEndianShort(buffer, byteIndex) / PCM_SCALE
                        val right = littleEndianShort(buffer, byteIndex + 2) / PCM_SCALE
                        val deviation = (left + right) * 0.5 - mean
                        squaredDeviationSum += deviation * deviation
                        byteIndex += BYTES_PER_FRAME
                    }
                    remainingFrames -= readFrames
                }
            }
            val meanFloat = mean.toFloat()
            val std = sqrt(squaredDeviationSum / (frames - 1)).toFloat()
            val scale = std + NORMALIZATION_EPSILON
            require(meanFloat.isFinite() && scale.isFinite() && scale > 0f)
            return Normalization(
                mean = meanFloat,
                sampleStandardDeviation = std,
                scale = scale,
                selectedPcmSha256 = digest.digest().joinToString("") { byte ->
                    "%02x".format(byte)
                },
            )
        }

        companion object {
            fun open(file: File): CanonicalPcm16Wav {
                RandomAccessFile(file, "r").use { input ->
                    require(input.readFourCc() == "RIFF") { "Input is not a RIFF file." }
                    input.readUnsignedIntLe()
                    require(input.readFourCc() == "WAVE") { "Input is not a WAVE file." }
                    var format: Int? = null
                    var channels: Int? = null
                    var sampleRate: Int? = null
                    var byteRate: Long? = null
                    var blockAlign: Int? = null
                    var bits: Int? = null
                    var dataOffset: Long? = null
                    var dataBytes: Long? = null
                    while (input.filePointer + 8 <= input.length()) {
                        val id = input.readFourCc()
                        val size = input.readUnsignedIntLe()
                        val payload = input.filePointer
                        require(payload + size <= input.length()) { "Truncated WAV chunk $id." }
                        when (id) {
                            "fmt " -> {
                                require(size >= 16)
                                format = input.readUnsignedShortLe()
                                channels = input.readUnsignedShortLe()
                                sampleRate = input.readUnsignedIntLe().toInt()
                                byteRate = input.readUnsignedIntLe()
                                blockAlign = input.readUnsignedShortLe()
                                bits = input.readUnsignedShortLe()
                            }
                            "data" -> {
                                require(dataOffset == null) { "Multiple WAV data chunks are unsupported." }
                                dataOffset = payload
                                dataBytes = size
                            }
                        }
                        input.seek(payload + size + (size and 1L))
                    }
                    require(format == 1) { "WAV must use integer PCM encoding." }
                    require(channels == CHANNEL_COUNT) { "WAV must be stereo." }
                    require(sampleRate == SAMPLE_RATE) { "WAV must be 44.1 kHz." }
                    require(bits == 16) { "WAV must use signed PCM16 samples." }
                    require(blockAlign == BYTES_PER_FRAME)
                    require(byteRate == SAMPLE_RATE.toLong() * BYTES_PER_FRAME)
                    val offset = requireNotNull(dataOffset) { "WAV data chunk is missing." }
                    val bytes = requireNotNull(dataBytes)
                    require(bytes > 0 && bytes % BYTES_PER_FRAME == 0L)
                    return CanonicalPcm16Wav(
                        file = file,
                        sampleRate = requireNotNull(sampleRate),
                        channelCount = requireNotNull(channels),
                        bitsPerSample = requireNotNull(bits),
                        dataOffset = offset,
                        dataBytes = bytes,
                    )
                }
            }
        }
    }

    companion object {
        const val ARG_AUDIO_FILE = "canonicalE2eAudioFile"
        const val ARG_AUDIO_SHA256 = "canonicalE2eAudioSha256"
        const val ARG_DURATION_SECONDS = "canonicalE2eDurationSeconds"
        const val ARG_FRAME_LIMIT = "canonicalE2eFrameLimit"
        const val ARG_THREADS = "canonicalE2eThreads"
        const val ARG_ISTFT_MODE = "canonicalE2eIstftMode"
        const val ARG_ISTFT_WORKERS = "canonicalE2eIstftWorkers"
        const val ARG_VALIDATE_ISTFT_FLOAT_PARITY =
            "canonicalE2eValidateIstftFloatParity"
        const val ARG_CORE_WARMUP_RUNS = "canonicalE2eCoreWarmupRuns"
        const val ARG_CORE_MEASURED_RUNS = "canonicalE2eCoreMeasuredRuns"
        const val ARG_POSTPROCESS_MODE = "canonicalE2ePostprocessMode"
        const val ARG_RUN_ID = "canonicalE2eRunId"
        const val ARG_CANCEL_AFTER_WINDOWS = "canonicalE2eCancelAfterWindows"
        const val ARG_RESUME_AFTER_CANCEL = "canonicalE2eResumeAfterCancel"
        const val ARG_EXPORT_CANONICAL_INPUT = "canonicalE2eExportCanonicalInput"
        const val ARG_MODEL_VARIANT = "canonicalE2eModelVariant"

        const val MODEL_VARIANT_OFFICIAL = "official"
        const val MODEL_VARIANT_OFFICIAL_4S = "official-4s"
        const val MODEL_VARIANT_GUITAR_FT = "guitar-ft"

        const val MODEL_ID = "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0"
        const val MODEL_FILE_NAME = "htdemucs_6s.core.canonical_7p8s.fp32.tflite"
        const val MODEL_BYTE_SIZE = 117_624_880L
        const val MODEL_SHA256 =
            "8b19e919dd17c6a93d862ca9b1158ed72f09feb4c52745819346369506ba4ed7"

        private const val GUITAR_FT_MODEL_ID =
            "htdemucs_6s_guitar_ft_core_canonical_7p8s_fp32_v1_0_0"
        private const val GUITAR_FT_MODEL_FILE_NAME =
            "htdemucs_6s_guitar_ft.core.canonical_7p8s.fp32.tflite"
        private const val GUITAR_FT_MODEL_BYTE_SIZE = 117_729_544L
        private const val GUITAR_FT_MODEL_SHA256 =
            "ab632a5a024033d557eabb716f8829230532e8e5b4cd7ba146812a301f89b9a5"

        private const val OFFICIAL_4S_MODEL_ID =
            "htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0"
        private const val OFFICIAL_4S_MODEL_FILE_NAME =
            "htdemucs_4s.core.canonical_7p8s.fp32.tflite"
        private const val OFFICIAL_4S_MODEL_BYTE_SIZE = 178_042_000L
        private const val OFFICIAL_4S_MODEL_SHA256 =
            "9855718072ee819bacacdb6b670bd6257feca172bf27ac1d72dff994cdbeed81"

        private val FOUR_STEM_ORDER = listOf("drums", "bass", "other", "vocals")
        private val SIX_STEM_ORDER = FOUR_STEM_ORDER + listOf("guitar", "piano")

        private fun runtimeCounter(name: String): Long? = runCatching {
            Debug.getRuntimeStat(name)?.toLongOrNull()
        }.getOrNull()

        private fun counterDelta(before: Long?, after: Long?): Long? =
            if (before != null && after != null && after >= before) after - before else null

        private fun nullableSum(left: Long?, right: Long?): Long? =
            if (left != null && right != null) left + right else null

        private val MODEL_IDENTITIES = listOf(
            ModelIdentity(
                variant = MODEL_VARIANT_OFFICIAL,
                modelId = MODEL_ID,
                fileName = MODEL_FILE_NAME,
                byteSize = MODEL_BYTE_SIZE,
                sha256 = MODEL_SHA256,
                diagnosticOnly = false,
                researchOnly = false,
                hostAdmissionStatus = "admitted",
                stemOrder = SIX_STEM_ORDER,
            ),
            ModelIdentity(
                variant = MODEL_VARIANT_OFFICIAL_4S,
                modelId = OFFICIAL_4S_MODEL_ID,
                fileName = OFFICIAL_4S_MODEL_FILE_NAME,
                byteSize = OFFICIAL_4S_MODEL_BYTE_SIZE,
                sha256 = OFFICIAL_4S_MODEL_SHA256,
                diagnosticOnly = false,
                researchOnly = false,
                hostAdmissionStatus = "admitted",
                stemOrder = FOUR_STEM_ORDER,
            ),
            ModelIdentity(
                variant = MODEL_VARIANT_GUITAR_FT,
                modelId = GUITAR_FT_MODEL_ID,
                fileName = GUITAR_FT_MODEL_FILE_NAME,
                byteSize = GUITAR_FT_MODEL_BYTE_SIZE,
                sha256 = GUITAR_FT_MODEL_SHA256,
                diagnosticOnly = true,
                researchOnly = true,
                hostAdmissionStatus = "not-admitted",
                stemOrder = SIX_STEM_ORDER,
            ),
        ).associateBy(ModelIdentity::variant)

        private const val SIGNATURE_KEY = "serving_default"
        private const val WAVEFORM_INPUT_NAME = "args_0"
        private const val SPECTRUM_INPUT_NAME = "args_1"
        private const val FREQUENCY_OUTPUT_NAME = "output_0"
        private const val TIME_OUTPUT_NAME = "output_1"
        private const val SAMPLE_RATE = 44_100
        private const val WINDOW_SAMPLES = 343_980
        private const val SPECTRUM_FRAMES = 336
        private const val STRIDE_SAMPLES = 257_985
        private const val OVERLAP_SAMPLES = 85_995
        private const val CHANNEL_COUNT = 2
        private const val BYTES_PER_FRAME = CHANNEL_COUNT * Short.SIZE_BYTES
        private const val WAV_HEADER_BYTES = 44L
        private const val PCM_SCALE = 32768.0
        private const val PCM_SCALE_FLOAT = 32768f
        private const val NORMALIZATION_EPSILON = 1e-8f
        private const val IO_BUFFER_BYTES = 64 * 1024
        private const val RAW_FLOAT_HASH_BUFFER_BYTES = 64 * 1024
        private const val CANONICAL_WRITE_CHUNK_FRAMES = IO_BUFFER_BYTES / BYTES_PER_FRAME
        private const val TEMPORARY_INPUT_DIRECTORY = "input-tmp"
        private const val TEMPORARY_WAV_NAME = "decoded-canonical-44100-stereo-pcm16.wav"
        private const val EXPORTED_CANONICAL_INPUT_NAME = "canonical-input-44100-stereo-pcm16.wav"
        private const val OUTPUT_STAGING_NAME = "outputs.partial"
        private const val OUTPUT_FINAL_NAME = "outputs"

        private val INPUT_ABI = listOf(
            TensorAbi(WAVEFORM_INPUT_NAME, listOf(1, 2, WINDOW_SAMPLES)),
            TensorAbi(SPECTRUM_INPUT_NAME, listOf(1, 4, 2_048, SPECTRUM_FRAMES)),
        )
        private val SHA256 = Regex("^[0-9a-f]{64}$")
        private val RUN_ID = Regex("^[A-Za-z0-9._-]{1,80}$")

        private fun outputAbi(stemCount: Int) = listOf(
            TensorAbi(
                FREQUENCY_OUTPUT_NAME,
                listOf(1, stemCount, 4, 2_048, SPECTRUM_FRAMES),
            ),
            TensorAbi(TIME_OUTPUT_NAME, listOf(1, stemCount, 2, WINDOW_SAMPLES)),
        )

        private fun littleEndianShort(bytes: ByteArray, offset: Int): Short {
            val low = bytes[offset].toInt() and 0xff
            val high = bytes[offset + 1].toInt()
            return ((high shl 8) or low).toShort()
        }

        private fun RandomAccessFile.readFourCc(): String {
            val bytes = ByteArray(4)
            readFully(bytes)
            return String(bytes, StandardCharsets.US_ASCII)
        }

        private fun RandomAccessFile.readUnsignedShortLe(): Int {
            val low = readUnsignedByte()
            return low or (readUnsignedByte() shl 8)
        }

        private fun RandomAccessFile.readUnsignedIntLe(): Long {
            val low = readUnsignedShortLe().toLong()
            return low or (readUnsignedShortLe().toLong() shl 16)
        }
    }
}
