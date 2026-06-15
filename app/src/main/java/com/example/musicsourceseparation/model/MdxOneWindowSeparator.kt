package com.example.musicsourceseparation.model

import android.content.Context
import android.net.Uri
import android.os.Environment
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.musicsourceseparation.audio.AudioPcmDecoder
import com.example.musicsourceseparation.audio.WavFileWriter
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.roundToInt

class MdxOneWindowSeparator(
    private val context: Context,
    private val config: MdxDspConfig = MdxDspConfig(),
    private val modelVariant: MdxModelVariant = MdxModelVariant.INST_MAIN,
) {
    fun separate(
        uri: Uri,
        displayName: String,
        startSeconds: Double = 0.0,
        modelVariant: MdxModelVariant = this.modelVariant,
    ): MdxOneWindowSeparationResult {
        val source = AudioPcmDecoder(context).decode(uri)
        val decoded = source.resampleTo(config.sampleRate)

        val startFrame = (startSeconds * config.sampleRate).toInt().coerceAtLeast(0)
        val requiredFrames = startFrame + config.chunkSize
        require(decoded.frameCount >= requiredFrames) {
            "Audio is too short. Need at least $requiredFrames frames for this start offset."
        }

        val modelFile = MdxModelFile.get(context, modelVariant)

        val waveform = decoded.toStereoFloat(startFrame = startFrame, maxFrames = config.chunkSize)
        val spectrogram = MdxSpectrogram(config)
        val modelInput = spectrogram.waveformToTensor(waveform)
        val modelOutput = runModel(modelFile, modelInput, spectrogram)
        val residual = subtract(waveform, modelOutput)
        val vocals = when (modelVariant.modelOutputStem) {
            MdxStem.VOCALS -> modelOutput
            MdxStem.INSTRUMENTAL -> residual
        }
        val instrumental = when (modelVariant.modelOutputStem) {
            MdxStem.VOCALS -> residual
            MdxStem.INSTRUMENTAL -> modelOutput
        }

        val outputDir = File(
            context.getExternalFilesDir(Environment.DIRECTORY_MUSIC) ?: context.filesDir,
            "separated",
        ).apply { mkdirs() }
        val baseName = safeBaseName(displayName)
        val offsetTag = if (startFrame == 0) {
            "${modelVariant.outputTag}_one_window"
        } else {
            "${modelVariant.outputTag}_one_window_${startSeconds.roundTag()}s"
        }
        val vocalsFile = uniqueOutputFile(outputDir, "${baseName}_${offsetTag}_vocals.wav")
        val instrumentalFile = uniqueOutputFile(outputDir, "${baseName}_${offsetTag}_instrumental.wav")

        writeStereoWav(vocalsFile, vocals)
        writeStereoWav(instrumentalFile, instrumental)

        return MdxOneWindowSeparationResult(
            vocalsFile = vocalsFile,
            instrumentalFile = instrumentalFile,
            frames = config.chunkSize,
            startSeconds = startSeconds,
            seconds = config.chunkSize.toDouble() / config.sampleRate,
            sourceSampleRate = source.sampleRate,
            outputSampleRate = decoded.sampleRate,
            modelVariant = modelVariant,
        )
    }

    private fun runModel(
        modelFile: File,
        modelInput: FloatArray,
        spectrogram: MdxSpectrogram,
    ): Array<FloatArray> {
        val environment = OrtEnvironment.getEnvironment()
        environment.createSession(modelFile.absolutePath, OrtSession.SessionOptions()).use { session ->
            val inputName = session.inputInfo.keys.first()
            val outputName = session.outputInfo.keys.first()
            val shape = longArrayOf(1, 4, config.dimF.toLong(), config.dimT.toLong())

            OnnxTensor.createTensor(environment, FloatBuffer.wrap(modelInput), shape).use { tensor ->
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

    private fun writeStereoWav(file: File, waveform: Array<FloatArray>) {
        val bytes = ByteArray(config.chunkSize * MdxDspConfig.STEREO_CHANNELS * Short.SIZE_BYTES)
        var offset = 0
        for (frame in 0 until config.chunkSize) {
            for (channel in 0 until MdxDspConfig.STEREO_CHANNELS) {
                val value = (waveform[channel][frame].coerceIn(-1f, 1f) * Short.MAX_VALUE).roundToInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                bytes[offset++] = (value and 0xFF).toByte()
                bytes[offset++] = ((value ushr 8) and 0xFF).toByte()
            }
        }

        WavFileWriter(file, config.sampleRate, MdxDspConfig.STEREO_CHANNELS).use { writer ->
            writer.writePcm16(bytes)
        }
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

    private fun Double.roundTag(): String {
        val rounded = roundToInt()
        return if (kotlin.math.abs(this - rounded) < 0.001) {
            rounded.toString()
        } else {
            "%.2f".format(this).replace('.', '_')
        }
    }
}

data class MdxOneWindowSeparationResult(
    val vocalsFile: File,
    val instrumentalFile: File,
    val frames: Int,
    val startSeconds: Double,
    val seconds: Double,
    val sourceSampleRate: Int,
    val outputSampleRate: Int,
    val modelVariant: MdxModelVariant,
) {
    fun toDisplayText(): String {
        return buildString {
            appendLine("One-window separation complete")
            appendLine("Model: ${modelVariant.displayName}")
            appendLine("Start: %.2f seconds".format(startSeconds))
            appendLine("Duration: %.2f seconds".format(seconds))
            if (sourceSampleRate != outputSampleRate) {
                appendLine("Resampled: $sourceSampleRate Hz -> $outputSampleRate Hz")
            }
            appendLine("Vocals: ${vocalsFile.absolutePath}")
            append("Instrumental: ${instrumentalFile.absolutePath}")
        }
    }
}
