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
import com.google.ai.edge.litert.NpuCompatibilityChecker
import com.google.ai.edge.litert.TensorBuffer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import kotlin.math.ceil
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
        require(Build.VERSION.SDK_INT >= 31) { "QNN v79 requires Android API 31 or newer." }
        require(Build.SUPPORTED_ABIS.firstOrNull() == "arm64-v8a") {
            "QNN v79 requires an arm64-v8a process; ABIs=${Build.SUPPORTED_ABIS.toList()}"
        }
        require(modelFile.isFile && modelFile.length() > 0L) {
            "Model file is missing: ${modelFile.absolutePath}"
        }
        require(audioFile.isFile && audioFile.length() > 0L) {
            "Audio file is missing: ${audioFile.absolutePath}"
        }
        require(modelOutputScale.isFinite() && modelOutputScale > 0f) {
            "Model output scale must be finite and positive."
        }

        outputDir.mkdirs()
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

        onPhase("Decoding source audio")
        val source = measured("decode") {
            AudioPcmDecoder(context).decode(Uri.fromFile(audioFile))
        }
        val decoded = measured("resample") { source.resampleTo(config.sampleRate) }
        require(decoded.frameCount > 0) { "Decoded audio is empty." }
        val windowCount = ceil(decoded.frameCount.toDouble() / config.generationSize).toInt()
        val vocalsFile = File(outputDir, "vocals.wav")
        val instrumentalFile = File(outputDir, "instrumental.wav")
        val spectrogram = MdxSpectrogram(config)
        val inferenceWallMs = mutableListOf<Double>()
        val inferenceCpuMs = mutableListOf<Long>()

        onPhase("Preparing Qualcomm HTP graph")
        val provider = BuiltinNpuAcceleratorProvider(context, NpuCompatibilityChecker.Qualcomm)
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
            WavFileWriter(vocalsFile, config.sampleRate, MdxDspConfig.STEREO_CHANNELS).use { vocalsWriter ->
                WavFileWriter(
                    instrumentalFile,
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
                            stereoFloatToPcm16(vocalsWindow, config.trim, writeFrames)
                        }
                        val instrumentalPcm = measured("pcmConvert") {
                            stereoFloatToPcm16(instrumentalWindow, config.trim, writeFrames)
                        }
                        measured("wavWrite") {
                            vocalsWriter.writePcm16(vocalsPcm)
                            instrumentalWriter.writePcm16(instrumentalPcm)
                        }
                        onProgress(windowIndex + 1, windowCount)
                    }
                }
            }
        } finally {
            inputBuffers.closeAll()
            outputBuffers.closeAll()
            modelToClose?.close()
            environment.close()
        }

        val totalWallMs = nanosToMs(SystemClock.elapsedRealtimeNanos() - totalStarted)
        val audioDurationSeconds = decoded.frameCount.toDouble() / decoded.sampleRate
        return JSONObject()
            .put("source", JSONObject()
                .put("path", audioFile.absolutePath)
                .put("bytes", audioFile.length())
                .put("sha256", sha256(audioFile))
                .put("sampleRate", source.sampleRate)
                .put("channelCount", source.channelCount)
                .put("decodedFrames", source.frameCount))
            .put("outputSampleRate", decoded.sampleRate)
            .put("outputFrames", decoded.frameCount)
            .put("audioDurationSeconds", audioDurationSeconds)
            .put("windowCount", windowCount)
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
            .put("setupWallMs", setupWallMs)
            .put("setupCpuMs", setupCpuMs)
            .put("inferenceWallMs", JSONArray(inferenceWallMs))
            .put("inferenceCpuMs", JSONArray(inferenceCpuMs))
            .put("inferenceSummary", timingSummary(inferenceWallMs, inferenceCpuMs))
            .put("stageWallMs", JSONObject(stageNanos.mapValues { nanosToMs(it.value) }))
            .put("totalWallMs", totalWallMs)
            .put("realtimeFactor", totalWallMs / 1000.0 / audioDurationSeconds)
            .put("outputs", JSONObject()
                .put("vocals", fileEvidence(vocalsFile))
                .put("instrumental", fileEvidence(instrumentalFile)))
            .put("backendEvidence", JSONObject()
                .put("provider", "BuiltinNpuAcceleratorProvider")
                .put("compatibilityChecker", "Qualcomm")
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
                        .put("bytes", file.length()) }
                    .toList())))
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
    ): ByteArray {
        val bytes = ByteArray(frames * MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES)
        var offset = 0
        for (frame in startFrame until startFrame + frames) {
            for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
                val value = (waveform[channel][frame].coerceIn(-1f, 1f) * Short.MAX_VALUE)
                    .roundToInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                bytes[offset++] = (value and 0xFF).toByte()
                bytes[offset++] = ((value ushr 8) and 0xFF).toByte()
            }
        }
        return bytes
    }

    private fun List<TensorBuffer>.closeAll() = forEach { it.close() }

    private fun fileEvidence(file: File): JSONObject = JSONObject()
        .put("path", file.absolutePath)
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

    private companion object {
        const val INPUT_NAME = "input"
        const val OUTPUT_NAME = "output"
        const val LOG_TAG = "MSS-QNN"
    }
}
