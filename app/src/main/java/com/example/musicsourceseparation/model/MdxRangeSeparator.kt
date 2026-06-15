package com.example.musicsourceseparation.model

import android.content.Context
import android.net.Uri
import android.os.Environment
import android.os.SystemClock
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.musicsourceseparation.audio.AudioPcmDecoder
import com.example.musicsourceseparation.audio.DecodedPcmAudio
import com.example.musicsourceseparation.audio.WavFileWriter
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.roundToInt

class MdxRangeSeparator(
    private val context: Context,
    private val config: MdxDspConfig = MdxDspConfig(),
    private val runtimeSettings: MdxRuntimeSettings = MdxRuntimeSettings(),
    private val modelVariant: MdxModelVariant = MdxModelVariant.INST_MAIN,
) {
    fun separate(
        uri: Uri,
        displayName: String,
        startMs: Long,
        endMs: Long?,
        runtimeSettings: MdxRuntimeSettings = this.runtimeSettings,
        modelVariant: MdxModelVariant = this.modelVariant,
        onProgress: (MdxRangeProgress) -> Unit = {},
    ): MdxRangeSeparationResult {
        val timing = MdxRangeTimingAccumulator()
        val totalStartedAt = SystemClock.elapsedRealtime()
        val source = measureElapsed(timing, "Decode") {
            AudioPcmDecoder(context).decode(uri)
        }
        val decoded = measureElapsed(timing, "Resample") {
            source.resampleTo(config.sampleRate)
        }

        val startFrame = msToFrame(startMs).coerceIn(0, decoded.frameCount)
        val requestedEndFrame = endMs?.let { msToFrame(it) } ?: decoded.frameCount
        val endFrame = requestedEndFrame.coerceIn(startFrame, decoded.frameCount)
        val targetFrames = endFrame - startFrame
        require(targetFrames > 0) { "Selected range is empty." }

        val modelFile = measureElapsed(timing, "Model file") {
            MdxModelFile.get(context, modelVariant)
        }

        val outputSetupStartedAt = SystemClock.elapsedRealtime()
        val outputDir = File(
            context.getExternalFilesDir(Environment.DIRECTORY_MUSIC) ?: context.filesDir,
            "separated",
        ).apply { mkdirs() }

        val baseName = safeBaseName(displayName)
        val rangeTag = "${modelVariant.outputTag}_${frameToMs(startFrame)}ms_${frameToMs(endFrame)}ms"
        val vocalsFile = uniqueOutputFile(outputDir, "${baseName}_${rangeTag}_vocals.wav")
        val instrumentalFile = uniqueOutputFile(outputDir, "${baseName}_${rangeTag}_instrumental.wav")
        val timingFile = uniqueOutputFile(outputDir, "${baseName}_${rangeTag}_timing.txt")
        timing.add("Output setup", SystemClock.elapsedRealtime() - outputSetupStartedAt)

        val windowCount = ceil(targetFrames.toDouble() / config.generationSize.toDouble()).toInt()
        val environment = OrtEnvironment.getEnvironment()
        val spectrogram = MdxSpectrogram(config)

        WavFileWriter(vocalsFile, config.sampleRate, MdxDspConfig.STEREO_CHANNELS).use { vocalsWriter ->
            WavFileWriter(instrumentalFile, config.sampleRate, MdxDspConfig.STEREO_CHANNELS).use { instrumentalWriter ->
                val session = measureElapsed(timing, "Session setup") {
                    runtimeSettings.createSessionOptions().use { options ->
                        environment.createSession(modelFile.absolutePath, options)
                    }
                }
                session.use {
                    val inputName = session.inputInfo.keys.first()
                    val outputName = session.outputInfo.keys.first()
                    for (windowIndex in 0 until windowCount) {
                        val generationStartFrame = startFrame + windowIndex * config.generationSize
                        val remainingFrames = endFrame - generationStartFrame
                        val writeFrames = minOf(config.generationSize, remainingFrames)
                        val mixWindow = measureElapsed(timing, "Window input") {
                            decoded.toStereoFloatContextWindow(
                                windowStartFrame = generationStartFrame - config.trim,
                                frames = config.chunkSize,
                            )
                        }

                        val modelOutputWindow = runWindow(
                            session = session,
                            inputName = inputName,
                            outputName = outputName,
                            spectrogram = spectrogram,
                            mixWindow = mixWindow,
                            timing = timing,
                        )
                        val residualWindow = measureElapsed(timing, "Stem subtract") {
                            subtract(mixWindow, modelOutputWindow)
                        }
                        val vocalsWindow = when (modelVariant.modelOutputStem) {
                            MdxStem.VOCALS -> modelOutputWindow
                            MdxStem.INSTRUMENTAL -> residualWindow
                        }
                        val instrumentalWindow = when (modelVariant.modelOutputStem) {
                            MdxStem.VOCALS -> residualWindow
                            MdxStem.INSTRUMENTAL -> modelOutputWindow
                        }

                        val vocalsPcm = measureElapsed(timing, "PCM convert") {
                            stereoFloatToPcm16(vocalsWindow, startFrame = config.trim, frames = writeFrames)
                        }
                        val instrumentalPcm = measureElapsed(timing, "PCM convert") {
                            stereoFloatToPcm16(instrumentalWindow, startFrame = config.trim, frames = writeFrames)
                        }
                        measureElapsed(timing, "WAV write") {
                            vocalsWriter.writePcm16(vocalsPcm)
                            instrumentalWriter.writePcm16(instrumentalPcm)
                        }
                        onProgress(MdxRangeProgress(windowIndex + 1, windowCount))
                    }
                }
            }
        }

        val elapsedMs = SystemClock.elapsedRealtime() - totalStartedAt
        val timingReport = timing.toReport(
            audioDurationSeconds = targetFrames.toDouble() / config.sampleRate,
            windowCount = windowCount,
            totalMs = elapsedMs,
            runtimeSettings = runtimeSettings,
            modelVariant = modelVariant,
        )
        timingFile.writeText(
            timingReport.toFileText(
                vocalsFile = vocalsFile,
                instrumentalFile = instrumentalFile,
            ),
            Charsets.UTF_8,
        )
        return MdxRangeSeparationResult(
            vocalsFile = vocalsFile,
            instrumentalFile = instrumentalFile,
            timingFile = timingFile,
            startMs = frameToMs(startFrame),
            endMs = frameToMs(endFrame),
            frames = targetFrames,
            windowCount = windowCount,
            elapsedMs = elapsedMs,
            sourceSampleRate = source.sampleRate,
            outputSampleRate = decoded.sampleRate,
            timingReport = timingReport,
            modelVariant = modelVariant,
        )
    }

    private fun runWindow(
        session: OrtSession,
        inputName: String,
        outputName: String,
        spectrogram: MdxSpectrogram,
        mixWindow: Array<FloatArray>,
        timing: MdxRangeTimingAccumulator,
    ): Array<FloatArray> {
        val modelInput = measureElapsed(timing, "STFT") {
            spectrogram.waveformToTensor(mixWindow)
        }
        val shape = longArrayOf(1, 4, config.dimF.toLong(), config.dimT.toLong())

        measureElapsed(timing, "Tensor create") {
            OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), FloatBuffer.wrap(modelInput), shape)
        }.use { tensor ->
            measureElapsed(timing, "ONNX inference") {
                session.run(mapOf(inputName to tensor))
            }.use { outputs ->
                val output = outputs[outputName].orElseThrow {
                    IllegalStateException("Missing ONNX output: $outputName")
                }.value
                @Suppress("UNCHECKED_CAST")
                val outputArray = output as Array<Array<Array<FloatArray>>>
                val flatOutput = measureElapsed(timing, "Output flatten") {
                    flattenOutput(outputArray)
                }
                return measureElapsed(timing, "ISTFT") {
                    spectrogram.tensorToWaveform(flatOutput)
                }
            }
        }
    }

    private inline fun <T> measureElapsed(timing: MdxRangeTimingAccumulator, stage: String, block: () -> T): T {
        val startedAt = SystemClock.elapsedRealtime()
        return try {
            block()
        } finally {
            timing.add(stage, SystemClock.elapsedRealtime() - startedAt)
        }
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

        val source = toStereoFloat(
            startFrame = copyStartFrame,
            maxFrames = copyEndFrame - copyStartFrame,
        )
        val destinationOffset = copyStartFrame - windowStartFrame
        for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
            source[channel].copyInto(
                destination = window[channel],
                destinationOffset = destinationOffset,
            )
        }
        return window
    }

    private fun stereoFloatToPcm16(waveform: Array<FloatArray>, startFrame: Int, frames: Int): ByteArray {
        val bytes = ByteArray(frames * MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES)
        var offset = 0
        val endFrame = startFrame + frames
        for (frame in startFrame until endFrame) {
            for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
                val value = (waveform[channel][frame].coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                bytes[offset++] = (value and 0xFF).toByte()
                bytes[offset++] = ((value ushr 8) and 0xFF).toByte()
            }
        }
        return bytes
    }

    private fun flattenOutput(output: Array<Array<Array<FloatArray>>>): FloatArray {
        require(output.size == 1) { "Expected batch size 1, got ${output.size}." }
        val flat = FloatArray(config.tensorElementCount)
        var offset = 0
        for (channel in output[0]) {
            for (frequency in channel) {
                for (value in frequency) {
                    flat[offset++] = value
                }
            }
        }
        return flat
    }

    private fun subtract(mix: Array<FloatArray>, instrumental: Array<FloatArray>): Array<FloatArray> {
        return Array(MdxDspConfig.STEREO_CHANNELS) { channel ->
            FloatArray(config.chunkSize) { index ->
                mix[channel][index] - instrumental[channel][index]
            }
        }
    }

    private fun msToFrame(ms: Long): Int {
        return ((ms.coerceAtLeast(0) * config.sampleRate) / 1000L).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    }

    private fun frameToMs(frame: Int): Long {
        return (frame.toLong() * 1000L) / config.sampleRate
    }

    private fun safeBaseName(displayName: String): String {
        return displayName.substringBeforeLast('.')
            .replace(Regex("[^A-Za-z0-9._-]+"), "_")
            .ifBlank { "audio" }
    }

    private fun uniqueOutputFile(outputDir: File, name: String): File {
        val extensionIndex = name.lastIndexOf('.')
        val base = if (extensionIndex > 0) name.substring(0, extensionIndex) else name
        val extension = if (extensionIndex > 0) name.substring(extensionIndex) else ""
        var candidate = File(outputDir, name)
        var suffix = 2
        while (candidate.exists()) {
            candidate = File(outputDir, "${base}_$suffix$extension")
            suffix += 1
        }
        return candidate
    }
}

data class MdxRangeProgress(
    val completedWindows: Int,
    val totalWindows: Int,
) {
    val percent: Int = ((completedWindows * 100.0) / totalWindows).roundToInt()
}

data class MdxRangeSeparationResult(
    val vocalsFile: File,
    val instrumentalFile: File,
    val timingFile: File,
    val startMs: Long,
    val endMs: Long,
    val frames: Int,
    val windowCount: Int,
    val elapsedMs: Long,
    val sourceSampleRate: Int,
    val outputSampleRate: Int,
    val timingReport: MdxRangeTimingReport,
    val modelVariant: MdxModelVariant,
) {
    fun toDisplayText(): String {
        val durationSeconds = frames.toDouble() / outputSampleRate
        return buildString {
            appendLine("Range separation complete")
            appendLine("Range: ${formatMs(startMs)} - ${formatMs(endMs)}")
            appendLine("Model: ${modelVariant.displayName}")
            appendLine("Duration: %.2f seconds".format(durationSeconds))
            appendLine("Windows: $windowCount")
            appendLine("Elapsed: %.2f seconds".format(elapsedMs / 1000.0))
            if (sourceSampleRate != outputSampleRate) {
                appendLine("Resampled: $sourceSampleRate Hz -> $outputSampleRate Hz")
            }
            appendLine("Vocals: ${vocalsFile.absolutePath}")
            appendLine("Instrumental: ${instrumentalFile.absolutePath}")
            appendLine("Timing report: ${timingFile.absolutePath}")
            append(timingReport.toDisplayText())
        }
    }

    private fun formatMs(ms: Long): String {
        val minutes = ms / 60_000
        val seconds = (ms % 60_000) / 1000
        val millis = ms % 1000
        return "%d:%02d.%03d".format(minutes, seconds, millis)
    }
}
