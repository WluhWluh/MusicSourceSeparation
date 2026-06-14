package com.example.musicsourceseparation.model

import android.content.Context
import android.net.Uri
import android.os.Environment
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
) {
    fun separate(
        uri: Uri,
        displayName: String,
        startMs: Long,
        endMs: Long?,
        onProgress: (MdxRangeProgress) -> Unit = {},
    ): MdxRangeSeparationResult {
        val decoded = AudioPcmDecoder(context).decode(uri)
        require(decoded.sampleRate == config.sampleRate) {
            "Expected ${config.sampleRate} Hz audio, got ${decoded.sampleRate} Hz."
        }

        val startFrame = msToFrame(startMs).coerceIn(0, decoded.frameCount)
        val requestedEndFrame = endMs?.let { msToFrame(it) } ?: decoded.frameCount
        val endFrame = requestedEndFrame.coerceIn(startFrame, decoded.frameCount)
        val targetFrames = endFrame - startFrame
        require(targetFrames > 0) { "Selected range is empty." }

        val modelFile = File(context.filesDir, MODEL_FILE_NAME)
        require(modelFile.isFile) {
            "Model file missing: ${modelFile.absolutePath}"
        }

        val outputDir = File(
            context.getExternalFilesDir(Environment.DIRECTORY_MUSIC) ?: context.filesDir,
            "separated",
        ).apply { mkdirs() }

        val baseName = safeBaseName(displayName)
        val rangeTag = "${frameToMs(startFrame)}ms_${frameToMs(endFrame)}ms"
        val vocalsFile = uniqueOutputFile(outputDir, "${baseName}_${rangeTag}_vocals.wav")
        val instrumentalFile = uniqueOutputFile(outputDir, "${baseName}_${rangeTag}_instrumental.wav")

        val windowCount = ceil(targetFrames.toDouble() / config.chunkSize.toDouble()).toInt()
        val environment = OrtEnvironment.getEnvironment()
        val spectrogram = MdxSpectrogram(config)
        val startedAt = System.currentTimeMillis()

        WavFileWriter(vocalsFile, config.sampleRate, MdxDspConfig.STEREO_CHANNELS).use { vocalsWriter ->
            WavFileWriter(instrumentalFile, config.sampleRate, MdxDspConfig.STEREO_CHANNELS).use { instrumentalWriter ->
                environment.createSession(modelFile.absolutePath, OrtSession.SessionOptions()).use { session ->
                    val inputName = session.inputInfo.keys.first()
                    val outputName = session.outputInfo.keys.first()
                    for (windowIndex in 0 until windowCount) {
                        val windowStartFrame = startFrame + windowIndex * config.chunkSize
                        val remainingFrames = endFrame - windowStartFrame
                        val writeFrames = minOf(config.chunkSize, remainingFrames)
                        val mixWindow = decoded.toStereoFloatPadded(
                            startFrame = windowStartFrame,
                            frames = writeFrames,
                            paddedFrames = config.chunkSize,
                        )

                        val instrumentalWindow = runWindow(
                            session = session,
                            inputName = inputName,
                            outputName = outputName,
                            spectrogram = spectrogram,
                            mixWindow = mixWindow,
                        )
                        val vocalsWindow = subtract(mixWindow, instrumentalWindow)

                        vocalsWriter.writePcm16(stereoFloatToPcm16(vocalsWindow, writeFrames))
                        instrumentalWriter.writePcm16(stereoFloatToPcm16(instrumentalWindow, writeFrames))
                        onProgress(MdxRangeProgress(windowIndex + 1, windowCount))
                    }
                }
            }
        }

        val elapsedMs = System.currentTimeMillis() - startedAt
        return MdxRangeSeparationResult(
            vocalsFile = vocalsFile,
            instrumentalFile = instrumentalFile,
            startMs = frameToMs(startFrame),
            endMs = frameToMs(endFrame),
            frames = targetFrames,
            windowCount = windowCount,
            elapsedMs = elapsedMs,
        )
    }

    private fun runWindow(
        session: OrtSession,
        inputName: String,
        outputName: String,
        spectrogram: MdxSpectrogram,
        mixWindow: Array<FloatArray>,
    ): Array<FloatArray> {
        val modelInput = spectrogram.waveformToTensor(mixWindow)
        val shape = longArrayOf(1, 4, config.dimF.toLong(), config.dimT.toLong())

        OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), FloatBuffer.wrap(modelInput), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { outputs ->
                val output = outputs[outputName].orElseThrow {
                    IllegalStateException("Missing ONNX output: $outputName")
                }.value
                @Suppress("UNCHECKED_CAST")
                val outputArray = output as Array<Array<Array<FloatArray>>>
                return spectrogram.tensorToWaveform(flattenOutput(outputArray))
            }
        }
    }

    private fun DecodedPcmAudio.toStereoFloatPadded(
        startFrame: Int,
        frames: Int,
        paddedFrames: Int,
    ): Array<FloatArray> {
        val source = toStereoFloat(startFrame = startFrame, maxFrames = frames)
        return Array(MdxDspConfig.STEREO_CHANNELS) { channel ->
            FloatArray(paddedFrames).also { padded ->
                source[channel].copyInto(padded)
            }
        }
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

    private fun stereoFloatToPcm16(waveform: Array<FloatArray>, frames: Int): ByteArray {
        val bytes = ByteArray(frames * MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES)
        var offset = 0
        for (frame in 0 until frames) {
            for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
                val value = (waveform[channel][frame].coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                bytes[offset++] = (value and 0xFF).toByte()
                bytes[offset++] = ((value ushr 8) and 0xFF).toByte()
            }
        }
        return bytes
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
        val base = name.substringBeforeLast(".wav")
        var candidate = File(outputDir, name)
        var suffix = 2
        while (candidate.exists()) {
            candidate = File(outputDir, "${base}_$suffix.wav")
            suffix += 1
        }
        return candidate
    }

    private companion object {
        const val MODEL_FILE_NAME = "UVR-MDX-NET-Inst_Main.onnx"
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
    val startMs: Long,
    val endMs: Long,
    val frames: Int,
    val windowCount: Int,
    val elapsedMs: Long,
) {
    fun toDisplayText(): String {
        val durationSeconds = frames / 44_100.0
        return buildString {
            appendLine("Range separation complete")
            appendLine("Range: ${formatMs(startMs)} - ${formatMs(endMs)}")
            appendLine("Duration: %.2f seconds".format(durationSeconds))
            appendLine("Windows: $windowCount")
            appendLine("Elapsed: %.2f seconds".format(elapsedMs / 1000.0))
            appendLine("Vocals: ${vocalsFile.absolutePath}")
            append("Instrumental: ${instrumentalFile.absolutePath}")
        }
    }

    private fun formatMs(ms: Long): String {
        val minutes = ms / 60_000
        val seconds = (ms % 60_000) / 1000
        val millis = ms % 1000
        return "%d:%02d.%03d".format(minutes, seconds, millis)
    }
}
