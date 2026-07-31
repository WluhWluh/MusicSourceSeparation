package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.util.Log
import com.example.musicsourceseparation.audio.AudioPcmDecoder
import com.example.musicsourceseparation.audio.DecodedPcmAudio
import com.example.musicsourceseparation.audio.WavFileWriter
import com.example.musicsourceseparation.model.MdxDspConfig
import com.example.musicsourceseparation.model.MdxSpectrogram
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.roundToInt

internal class QnnMdxAudioBenchmark(
    private val context: Context,
    private val config: MdxDspConfig = MdxDspConfig(),
) {
    fun run(
        modelFile: File,
        audioFile: File,
        outputDir: File,
        evidenceDir: File,
        modelOutputScale: Float,
        onPhase: (String) -> Unit = {},
        onProgress: (completed: Int, total: Int) -> Unit = { _, _ -> },
    ): JSONObject {
        require(Build.VERSION.SDK_INT >= 31) { "QNN requires Android API 31 or newer." }
        require(Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a") {
            "QNN requires an arm64-v8a process; ABIs=${Build.SUPPORTED_ABIS.toList()}"
        }
        require(modelFile.isFile && modelFile.length() > 0L) {
            "Model file is missing: ${modelFile.absolutePath}"
        }
        require(audioFile.isFile && audioFile.length() > 0L) {
            "Audio file is missing: ${audioFile.absolutePath}"
        }
        require(modelOutputScale == MODEL_OUTPUT_SCALE) {
            "9662 model output scale must be $MODEL_OUTPUT_SCALE."
        }
        val inputWav = validateCanonicalPcm16Wav(audioFile)

        val outputParent = requireNotNull(outputDir.parentFile) {
            "QNN audio output directory has no parent: ${outputDir.absolutePath}"
        }
        val stagingDir = File(outputParent, "${outputDir.name}.partial")
        require(outputDir.deleteRecursively()) {
            "Could not remove stale QNN output directory: ${outputDir.absolutePath}"
        }
        require(stagingDir.deleteRecursively() && stagingDir.mkdirs()) {
            "Could not prepare QNN staging directory: ${stagingDir.absolutePath}"
        }
        evidenceDir.mkdirs()
        val totalStarted = SystemClock.elapsedRealtimeNanos()
        val stageNanos = linkedMapOf<String, Long>()
        fun <T> measured(stage: String, block: () -> T): T {
            val started = SystemClock.elapsedRealtimeNanos()
            return try {
                block()
            } finally {
                stageNanos[stage] = (stageNanos[stage] ?: 0L) +
                    (SystemClock.elapsedRealtimeNanos() - started)
            }
        }

        val modelSha256 = measured("modelHash") { sha256(modelFile) }
        require(modelSha256 == MODEL_SHA256) {
            "QNN audio benchmark requires the frozen 9662 artifact, got $modelSha256."
        }
        val sourceSha256 = measured("sourceHash") { sha256(audioFile) }

        onPhase("Decoding source audio")
        val source = measured("decode") {
            AudioPcmDecoder(context).decode(Uri.fromFile(audioFile))
        }
        require(source.sampleRate == config.sampleRate) {
            "QNN audio benchmark requires ${config.sampleRate} Hz PCM, got ${source.sampleRate} Hz."
        }
        require(source.channelCount == MdxDspConfig.STEREO_CHANNELS) {
            "QNN audio benchmark requires stereo PCM, got ${source.channelCount} channels."
        }
        require(source.frameCount.toLong() == inputWav.frameCount) {
            "Decoded frame count ${source.frameCount} did not match WAV header ${inputWav.frameCount}."
        }
        val decoded = source
        require(decoded.frameCount > 0) { "Decoded audio is empty." }
        val windowCount = ceil(decoded.frameCount.toDouble() / config.generationSize).toInt()
        val vocalsFile = File(outputDir, "vocals.wav")
        val instrumentalFile = File(outputDir, "instrumental.wav")
        val vocalsPartialFile = File(stagingDir, "vocals.wav.partial")
        val instrumentalPartialFile = File(stagingDir, "instrumental.wav.partial")
        val spectrogram = MdxSpectrogram(config)
        val inferenceWallMs = mutableListOf<Double>()
        val inferenceCpuMs = mutableListOf<Long>()
        val vocalsPcmStats = PcmStats()
        val instrumentalPcmStats = PcmStats()
        var completedWindows = 0

        onPhase("Preparing Qualcomm HTP graph")
        val provider = BuiltinNpuAcceleratorProvider(context, QnnDeviceCompatibility.checker)
        require(provider.isDeviceSupported()) {
            "LiteRT does not recognize ${Build.SOC_MANUFACTURER}/${Build.SOC_MODEL} as a supported Qualcomm NPU."
        }
        require(provider.isLibraryReady()) { "The bundled Qualcomm NPU runtime is not ready." }
        Log.i(
            LOG_TAG,
            "QNN audio preflight soc=${Build.SOC_MANUFACTURER}/${Build.SOC_MODEL} " +
                "libraryDir=${provider.getLibraryDir()}",
        )
        val environment = Environment.create(provider)
        var modelToClose: CompiledModel? = null
        var inputBuffers: List<TensorBuffer> = emptyList()
        var outputBuffers: List<TensorBuffer> = emptyList()
        var setupWallMs = 0.0
        var setupCpuMs = 0L
        val availableAccelerators: List<String>
        var primaryFailure: Throwable? = null
        try {
            availableAccelerators = environment.getAvailableAccelerators().map { it.name }.sorted()
            require(Accelerator.NPU.name in availableAccelerators) {
                "NPU was not registered; available accelerators=$availableAccelerators"
            }
            val setupStartedNanos = SystemClock.elapsedRealtimeNanos()
            val setupStartedCpuMs = android.os.Process.getElapsedCpuTime()
            val options = CompiledModel.Options(Accelerator.NPU).apply {
                qualcommOptions = CompiledModel.QualcommOptions(
                    logLevel = CompiledModel.QualcommOptions.LogLevel.INFO,
                    useHtpPreference = true,
                    htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode
                        .SUSTAINED_HIGH_PERFORMANCE,
                    profiling = CompiledModel.QualcommOptions.Profiling.OFF,
                    irJsonDir = evidenceDir.absolutePath,
                    optimizationLevel = CompiledModel.QualcommOptions.OptimizationLevel
                        .HTP_OPTIMIZE_FOR_INFERENCE,
                )
            }
            val model = CompiledModel.create(modelFile.absolutePath, options, environment).also {
                modelToClose = it
            }
            inputBuffers = model.createInputBuffers()
            outputBuffers = model.createOutputBuffers()
            require(inputBuffers.size == 1 && outputBuffers.size == 1) {
                "Expected one input and one output, got ${inputBuffers.size}/${outputBuffers.size}."
            }
            val expectedShape = listOf(1, config.dimF, config.dimT, MdxDspConfig.STEM_COMPLEX_CHANNELS)
            val inputShape = requireNotNull(model.getInputTensorType(INPUT_NAME).layout) {
                "LiteRT input tensor has no layout."
            }.dimensions
            val outputShape = requireNotNull(model.getOutputTensorType(OUTPUT_NAME).layout) {
                "LiteRT output tensor has no layout."
            }.dimensions
            require(inputShape == expectedShape && outputShape == expectedShape) {
                "Unexpected LiteRT shapes: input=$inputShape output=$outputShape expected=$expectedShape"
            }
            setupWallMs = nanosToMs(SystemClock.elapsedRealtimeNanos() - setupStartedNanos)
            setupCpuMs = android.os.Process.getElapsedCpuTime() - setupStartedCpuMs

            onPhase("Separating audio")
            WavFileWriter(
                vocalsPartialFile,
                config.sampleRate,
                MdxDspConfig.STEREO_CHANNELS,
            ).use { vocalsWriter ->
                WavFileWriter(
                    instrumentalPartialFile,
                    config.sampleRate,
                    MdxDspConfig.STEREO_CHANNELS,
                ).use { instrumentalWriter ->
                    for (windowIndex in 0 until windowCount) {
                        val generationStartFrame = windowIndex * config.generationSize
                        val remainingFrames = decoded.frameCount - generationStartFrame
                        val writeFrames = minOf(config.generationSize, remainingFrames)
                        val mixWindow = measured("windowInput") {
                            decoded.toStereoFloatContextWindow(
                                windowStartFrame = generationStartFrame - config.trim,
                                frames = config.chunkSize,
                            )
                        }
                        val inputNchw = measured("stft") {
                            spectrogram.waveformToTensor(mixWindow)
                        }
                        val inputNhwc = measured("inputLayout") {
                            nchwToNhwc(inputNchw, config.dimF, config.dimT)
                        }
                        measured("inputWrite") {
                            inputBuffers.single().writeFloat(inputNhwc)
                        }
                        val inferenceStartedNanos = SystemClock.elapsedRealtimeNanos()
                        val inferenceStartedCpuMs = android.os.Process.getElapsedCpuTime()
                        model.run(inputBuffers, outputBuffers)
                        inferenceWallMs += nanosToMs(
                            SystemClock.elapsedRealtimeNanos() - inferenceStartedNanos,
                        )
                        inferenceCpuMs += android.os.Process.getElapsedCpuTime() - inferenceStartedCpuMs
                        val outputNhwc = measured("outputRead") {
                            outputBuffers.single().readFloat()
                        }
                        val outputNchw = measured("outputLayout") {
                            nhwcToNchw(outputNhwc, config.dimF, config.dimT)
                        }
                        val vocalsWindow = measured("istft") {
                            spectrogram.tensorToWaveform(outputNchw)
                        }
                        measured("scaleAndSubtract") {
                            for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
                                for (sample in vocalsWindow[channel].indices) {
                                    vocalsWindow[channel][sample] *= modelOutputScale
                                }
                            }
                        }
                        val instrumentalWindow = measured("scaleAndSubtract") {
                            subtract(mixWindow, vocalsWindow)
                        }
                        val vocalsPcm = measured("pcmConvert") {
                            stereoFloatToPcm16(
                                vocalsWindow,
                                config.trim,
                                writeFrames,
                                vocalsPcmStats,
                            )
                        }
                        val instrumentalPcm = measured("pcmConvert") {
                            stereoFloatToPcm16(
                                instrumentalWindow,
                                config.trim,
                                writeFrames,
                                instrumentalPcmStats,
                            )
                        }
                        measured("wavWrite") {
                            vocalsWriter.writePcm16(vocalsPcm)
                            instrumentalWriter.writePcm16(instrumentalPcm)
                        }
                        completedWindows = windowIndex + 1
                        onProgress(windowIndex + 1, windowCount)
                    }
                }
            }
        } catch (error: Throwable) {
            primaryFailure = error
            throw error
        } finally {
            closeLiteRtResources(
                primaryFailure = primaryFailure,
                inputBuffers = inputBuffers,
                outputBuffers = outputBuffers,
                model = modelToClose,
                environment = environment,
            )
        }
        measured("outputCommit") {
            commitOutput(vocalsPartialFile, File(stagingDir, vocalsFile.name))
            commitOutput(instrumentalPartialFile, File(stagingDir, instrumentalFile.name))
            commitOutput(stagingDir, outputDir)
        }

        val processingWallMs = nanosToMs(SystemClock.elapsedRealtimeNanos() - totalStarted)
        val audioDurationSeconds = decoded.frameCount.toDouble() / decoded.sampleRate
        val vocalsEvidence = measured("outputHash") { fileEvidence(vocalsFile, vocalsPcmStats) }
        val instrumentalEvidence = measured("outputHash") {
            fileEvidence(instrumentalFile, instrumentalPcmStats)
        }
        val endToEndWallMs = nanosToMs(SystemClock.elapsedRealtimeNanos() - totalStarted)
        val backendEvidence = JSONObject()
            .put("provider", "BuiltinNpuAcceleratorProvider")
            .put("compatibilityChecker", QnnDeviceCompatibility.CHECKER_NAME)
            .put("deviceSupported", provider.isDeviceSupported())
            .put("libraryReady", provider.isLibraryReady())
            .put("libraryDir", provider.getLibraryDir())
            .put("socManufacturer", Build.SOC_MANUFACTURER)
            .put("socModel", Build.SOC_MODEL)
            .put("htpPerformanceMode", "SUSTAINED_HIGH_PERFORMANCE")
            .put("optimizationLevel", "HTP_OPTIMIZE_FOR_INFERENCE")
            .put("profiling", "OFF")
            .put("irJsonDir", evidenceDir.absolutePath)
            .put("irFiles", JSONArray(evidenceDir.walkTopDown()
                .filter { it.isFile }
                .map { file -> JSONObject()
                    .put("path", file.relativeTo(evidenceDir).invariantSeparatorsPath)
                    .put("bytes", file.length())
                    .put("sha256", sha256(file)) }
                .toList()))
        QnnDelegationEvidence.annotate(backendEvidence)
        return JSONObject()
            .put("contract", JSONObject()
                .put("contractId", CONTRACT_ID)
                .put("modelOutputStem", "vocals")
                .put("modelOutputScale", modelOutputScale.toDouble())
                .put("logicalShapeNchw", JSONArray(listOf(
                    1,
                    MdxDspConfig.STEM_COMPLEX_CHANNELS,
                    config.dimF,
                    config.dimT,
                )))
                .put("runtimeShapeNhwc", JSONArray(listOf(
                    1,
                    config.dimF,
                    config.dimT,
                    MdxDspConfig.STEM_COMPLEX_CHANNELS,
                ))))
            .put("model", JSONObject()
                .put("path", modelFile.absolutePath)
                .put("bytes", modelFile.length())
                .put("sha256", modelSha256))
            .put("source", JSONObject()
                .put("path", audioFile.absolutePath)
                .put("bytes", audioFile.length())
                .put("sha256", sourceSha256)
                .put("sampleRate", source.sampleRate)
                .put("channelCount", source.channelCount)
                .put("decodedFrames", source.frameCount)
                .put("resampled", false)
                .put("container", "canonical-riff-wave")
                .put("encoding", "signed-pcm16-le")
                .put("pcmDataBytes", inputWav.dataBytes))
            .put("outputSampleRate", decoded.sampleRate)
            .put("outputFrames", decoded.frameCount)
            .put("audioDurationSeconds", audioDurationSeconds)
            .put("windowCount", windowCount)
            .put("completedWindows", completedWindows)
            .put("lastWindowFrames", decoded.frameCount - (windowCount - 1) * config.generationSize)
            .put("dsp", JSONObject()
                .put("nFft", config.nFft)
                .put("hopLength", config.hopLength)
                .put("dimF", config.dimF)
                .put("dimT", config.dimT)
                .put("trim", config.trim)
                .put("generationSize", config.generationSize)
                .put("modelOutputStem", "vocals")
                .put("modelOutputScale", modelOutputScale.toDouble()))
            .put("availableAccelerators", JSONArray(availableAccelerators))
            .put("session", JSONObject()
                .put("count", 1)
                .put("cleanupComplete", true)
                .put("setupWallMs", setupWallMs)
                .put("setupCpuMs", setupCpuMs))
            .put("inferenceWallMs", JSONArray(inferenceWallMs))
            .put("inferenceCpuMs", JSONArray(inferenceCpuMs))
            .put("inferenceSummary", timingSummary(inferenceWallMs, inferenceCpuMs))
            .put("stageWallMs", JSONObject(stageNanos.mapValues { nanosToMs(it.value) }))
            .put("processingWallMs", processingWallMs)
            .put("endToEndWallMs", endToEndWallMs)
            .put("realtimeFactor", endToEndWallMs / 1000.0 / audioDurationSeconds)
            .put("outputs", JSONObject()
                .put("vocals", vocalsEvidence)
                .put("instrumental", instrumentalEvidence))
            .put("backendEvidence", backendEvidence)
    }

    private fun DecodedPcmAudio.toStereoFloatContextWindow(
        windowStartFrame: Int,
        frames: Int,
    ): Array<FloatArray> {
        val windowEndFrame = windowStartFrame + frames
        val copyStartFrame = maxOf(0, windowStartFrame)
        val copyEndFrame = minOf(frameCount, windowEndFrame)
        val window = Array(MdxDspConfig.STEREO_CHANNELS) { FloatArray(frames) }
        if (copyEndFrame <= copyStartFrame) return window
        val source = toStereoFloat(copyStartFrame, copyEndFrame - copyStartFrame)
        val destinationOffset = copyStartFrame - windowStartFrame
        for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
            source[channel].copyInto(window[channel], destinationOffset)
        }
        return window
    }

    private fun nchwToNhwc(input: FloatArray, height: Int, width: Int): FloatArray {
        val output = FloatArray(input.size)
        for (channel in 0 until MdxDspConfig.STEM_COMPLEX_CHANNELS) {
            for (row in 0 until height) {
                for (column in 0 until width) {
                    val nchwIndex = (channel * height + row) * width + column
                    val nhwcIndex = (row * width + column) * MdxDspConfig.STEM_COMPLEX_CHANNELS + channel
                    output[nhwcIndex] = input[nchwIndex]
                }
            }
        }
        return output
    }

    private fun nhwcToNchw(input: FloatArray, height: Int, width: Int): FloatArray {
        val output = FloatArray(input.size)
        for (channel in 0 until MdxDspConfig.STEM_COMPLEX_CHANNELS) {
            for (row in 0 until height) {
                for (column in 0 until width) {
                    val nchwIndex = (channel * height + row) * width + column
                    val nhwcIndex = (row * width + column) * MdxDspConfig.STEM_COMPLEX_CHANNELS + channel
                    output[nchwIndex] = input[nhwcIndex]
                }
            }
        }
        return output
    }

    private fun subtract(mix: Array<FloatArray>, vocals: Array<FloatArray>): Array<FloatArray> =
        Array(MdxDspConfig.STEREO_CHANNELS) { channel ->
            FloatArray(config.chunkSize) { sample -> mix[channel][sample] - vocals[channel][sample] }
        }

    private fun stereoFloatToPcm16(
        waveform: Array<FloatArray>,
        startFrame: Int,
        frames: Int,
        stats: PcmStats,
    ): ByteArray {
        val bytes = ByteArray(frames * MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES)
        var peak = stats.peak
        var positiveSaturatedSamples = 0L
        var negativeSaturatedSamples = 0L
        val endFrame = startFrame + frames
        for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
            val samples = waveform[channel]
            var frame = startFrame
            var offset = channel * Short.SIZE_BYTES
            while (frame < endFrame) {
                val sample = samples[frame]
                require(sample.isFinite()) { "Non-finite audio sample before PCM conversion." }
                peak = max(peak, abs(sample))
                if (sample >= 1f) positiveSaturatedSamples += 1
                if (sample <= -1f) negativeSaturatedSamples += 1
                val value = (sample.coerceIn(-1f, 1f) * Short.MAX_VALUE)
                    .roundToInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                bytes[offset] = (value and 0xFF).toByte()
                bytes[offset + 1] = ((value ushr 8) and 0xFF).toByte()
                frame += 1
                offset += MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES
            }
        }
        stats.sampleCount += frames.toLong() * MdxDspConfig.STEREO_CHANNELS
        stats.peak = peak
        stats.positiveSaturatedSamples += positiveSaturatedSamples
        stats.negativeSaturatedSamples += negativeSaturatedSamples
        return bytes
    }

    private fun commitOutput(partial: File, destination: File) {
        Files.move(
            partial.toPath(),
            destination.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
    }

    private fun validateCanonicalPcm16Wav(file: File): WavInputContract {
        require(file.length() >= WAV_HEADER_BYTES) { "WAV input is shorter than its header." }
        val header = ByteArray(WAV_HEADER_BYTES.toInt())
        file.inputStream().buffered().use { input ->
            var offset = 0
            while (offset < header.size) {
                val count = input.read(header, offset, header.size - offset)
                require(count > 0) { "Could not read the complete WAV header." }
                offset += count
            }
        }
        fun requireMarker(offset: Int, value: String) {
            val actual = header.copyOfRange(offset, offset + value.length)
                .toString(Charsets.US_ASCII)
            require(actual == value) { "Expected WAV marker $value at byte $offset, got $actual." }
        }
        requireMarker(0, "RIFF")
        requireMarker(8, "WAVE")
        requireMarker(12, "fmt ")
        requireMarker(36, "data")
        val values = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
        val riffBytes = Integer.toUnsignedLong(values.getInt(4)) + 8L
        val formatBytes = Integer.toUnsignedLong(values.getInt(16))
        val format = values.getShort(20).toInt() and 0xFFFF
        val channels = values.getShort(22).toInt() and 0xFFFF
        val sampleRate = values.getInt(24)
        val byteRate = values.getInt(28)
        val blockAlign = values.getShort(32).toInt() and 0xFFFF
        val bitsPerSample = values.getShort(34).toInt() and 0xFFFF
        val dataBytes = Integer.toUnsignedLong(values.getInt(40))
        require(riffBytes == file.length() && dataBytes + WAV_HEADER_BYTES == file.length()) {
            "QNN audio benchmark requires a canonical 44-byte WAV header."
        }
        require(formatBytes == 16L && format == WAV_FORMAT_PCM) {
            "QNN audio benchmark requires uncompressed PCM WAV input."
        }
        require(
            channels == MdxDspConfig.STEREO_CHANNELS &&
                sampleRate == config.sampleRate &&
                bitsPerSample == PCM_BITS_PER_SAMPLE &&
                blockAlign == PCM_BLOCK_ALIGN &&
                byteRate == config.sampleRate * PCM_BLOCK_ALIGN,
        ) {
            "QNN audio benchmark requires 44.1 kHz stereo PCM16 WAV input."
        }
        require(dataBytes in 1..MAX_DECODED_PCM_BYTES && dataBytes % PCM_BLOCK_ALIGN == 0L) {
            "WAV PCM data must be aligned and no larger than " +
                "${MAX_DECODED_PCM_BYTES / (1024 * 1024)} MiB."
        }
        return WavInputContract(dataBytes, dataBytes / PCM_BLOCK_ALIGN)
    }

    private fun closeLiteRtResources(
        primaryFailure: Throwable?,
        inputBuffers: List<TensorBuffer>,
        outputBuffers: List<TensorBuffer>,
        model: CompiledModel?,
        environment: Environment,
    ) {
        var closeFailure: Throwable? = null
        fun close(action: () -> Unit) {
            try {
                action()
            } catch (error: Throwable) {
                if (closeFailure == null) {
                    closeFailure = error
                } else {
                    closeFailure?.addSuppressed(error)
                }
            }
        }
        outputBuffers.asReversed().forEach { buffer -> close(buffer::close) }
        inputBuffers.asReversed().forEach { buffer -> close(buffer::close) }
        model?.let { close(it::close) }
        close(environment::close)
        closeFailure?.let { failure ->
            if (primaryFailure != null) {
                primaryFailure.addSuppressed(failure)
            } else {
                throw failure
            }
        }
    }

    private fun fileEvidence(file: File, stats: PcmStats): JSONObject = JSONObject()
        .put("path", file.absolutePath)
        .put("bytes", file.length())
        .put("sha256", sha256(file))
        .put("sampleRate", config.sampleRate)
        .put("channelCount", MdxDspConfig.STEREO_CHANNELS)
        .put("frames", (file.length() - WAV_HEADER_BYTES) /
            (MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES))
        .put("floatPeak", stats.peak.toDouble())
        .put("sampleCount", stats.sampleCount)
        .put("positiveSaturatedSamples", stats.positiveSaturatedSamples)
        .put("negativeSaturatedSamples", stats.negativeSaturatedSamples)
        .put("nonFiniteSamples", 0)
        .put("quantization", "round-to-nearest-pcm16")

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

    private fun timingSummary(wallMs: List<Double>, cpuMs: List<Long>): JSONObject {
        val sorted = wallMs.sorted()
        return JSONObject()
            .put("wallMeanMs", wallMs.average())
            .put("wallMedianMs", sorted.percentile(0.5))
            .put("wallP95Ms", sorted.percentile(0.95))
            .put("wallMinMs", sorted.firstOrNull() ?: 0.0)
            .put("wallMaxMs", sorted.lastOrNull() ?: 0.0)
            .put("cpuMeanMs", cpuMs.average())
            .put("cpuToWallRatio", if (wallMs.sum() > 0.0) cpuMs.sum() / wallMs.sum() else 0.0)
    }

    private fun List<Double>.percentile(fraction: Double): Double {
        if (isEmpty()) return 0.0
        val index = ((size - 1) * fraction).roundToInt().coerceIn(indices)
        return this[index]
    }

    private fun nanosToMs(nanos: Long): Double = nanos / 1_000_000.0

    private data class PcmStats(
        var sampleCount: Long = 0,
        var peak: Float = 0f,
        var positiveSaturatedSamples: Long = 0,
        var negativeSaturatedSamples: Long = 0,
    )

    private data class WavInputContract(
        val dataBytes: Long,
        val frameCount: Long,
    )

    private companion object {
        const val INPUT_NAME = "input"
        const val OUTPUT_NAME = "output"
        const val LOG_TAG = "MSS-QNN"
        const val CONTRACT_ID = "uvr_mdxnet_3_9662@2"
        const val MODEL_OUTPUT_SCALE = 1.035f
        const val MODEL_SHA256 = "f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378"
        const val MAX_DECODED_PCM_BYTES = 128L * 1024 * 1024
        const val WAV_HEADER_BYTES = 44L
        const val WAV_FORMAT_PCM = 1
        const val PCM_BITS_PER_SAMPLE = 16
        const val PCM_BLOCK_ALIGN = MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES
    }
}
