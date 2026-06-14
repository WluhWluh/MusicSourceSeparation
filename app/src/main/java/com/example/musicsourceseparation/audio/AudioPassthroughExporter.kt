package com.example.musicsourceseparation.audio

import android.content.Context
import android.net.Uri
import android.os.Environment
import java.io.File

class AudioPassthroughExporter(private val context: Context) {
    fun exportToWav(uri: Uri, displayName: String): AudioExportResult {
        val outputDir = File(
            context.getExternalFilesDir(Environment.DIRECTORY_MUSIC) ?: context.filesDir,
            "exports",
        ).apply { mkdirs() }
        val outputFile = uniqueOutputFile(outputDir, displayName)
        val decoded = AudioPcmDecoder(context).decode(uri)
        writeToWav(decoded, outputFile)
        return AudioExportResult(outputFile = outputFile)
    }

    private fun writeToWav(decoded: DecodedPcmAudio, outputFile: File) {
        var writer: WavFileWriter? = null
        try {
            writer = WavFileWriter(outputFile, decoded.sampleRate, decoded.channelCount)
            writer.writePcm16(decoded.pcm16)
        } catch (error: Throwable) {
            writer?.close()
            outputFile.delete()
            throw error
        } finally {
            writer?.close()
        }
    }

    private fun uniqueOutputFile(outputDir: File, displayName: String): File {
        val baseName = displayName.substringBeforeLast('.')
            .replace(Regex("[^A-Za-z0-9._-]+"), "_")
            .ifBlank { "audio" }
        var candidate = File(outputDir, "${baseName}_decoded.wav")
        var suffix = 2
        while (candidate.exists()) {
            candidate = File(outputDir, "${baseName}_decoded_$suffix.wav")
            suffix += 1
        }
        return candidate
    }

}

data class AudioExportResult(
    val outputFile: File,
)
