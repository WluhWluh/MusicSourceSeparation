package com.example.musicsourceseparation.audio

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Environment
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

class AudioPassthroughExporter(private val context: Context) {
    fun exportToWav(uri: Uri, displayName: String): AudioExportResult {
        val outputDir = File(
            context.getExternalFilesDir(Environment.DIRECTORY_MUSIC) ?: context.filesDir,
            "exports",
        ).apply { mkdirs() }
        val outputFile = uniqueOutputFile(outputDir, displayName)
        decodeToWav(uri, outputFile)
        return AudioExportResult(outputFile = outputFile)
    }

    private fun decodeToWav(uri: Uri, outputFile: File) {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        var writer: WavFileWriter? = null
        var codecStarted = false

        try {
            extractor.setDataSource(context, uri, null)
            val trackIndex = findAudioTrack(extractor)
            if (trackIndex < 0) error("No audio track was found.")

            extractor.selectTrack(trackIndex)
            val inputFormat = extractor.getTrackFormat(trackIndex)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)
                ?: error("Audio track has no MIME type.")

            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(inputFormat, null, null, 0)
            codec.start()
            codecStarted = true

            val bufferInfo = MediaCodec.BufferInfo()
            var inputEnded = false
            var outputEnded = false
            var outputFormat = codec.outputFormat

            while (!outputEnded) {
                if (!inputEnded) {
                    val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inputIndex >= 0) {
                        val inputBuffer = codec.getInputBuffer(inputIndex)
                            ?: error("Decoder returned a null input buffer.")
                        val sampleSize = extractor.readSampleData(inputBuffer, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(
                                inputIndex,
                                0,
                                0,
                                0L,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM,
                            )
                            inputEnded = true
                        } else {
                            codec.queueInputBuffer(
                                inputIndex,
                                0,
                                sampleSize,
                                extractor.sampleTime,
                                0,
                            )
                            extractor.advance()
                        }
                    }
                }

                when (val outputIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)) {
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        outputFormat = codec.outputFormat
                        if (writer != null) error("Decoder output format changed after writing began.")
                    }
                    else -> {
                        if (outputIndex >= 0) {
                            val outputBuffer = codec.getOutputBuffer(outputIndex)
                            if (bufferInfo.size > 0 && outputBuffer != null) {
                                val activeWriter = writer ?: createWriter(outputFile, outputFormat).also {
                                    writer = it
                                }
                                writePcmAs16BitWav(activeWriter, outputBuffer, bufferInfo, outputFormat)
                            }
                            outputEnded = bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            codec.releaseOutputBuffer(outputIndex, false)
                        }
                    }
                }
            }

            writer ?: error("Decoder produced no PCM output.")
        } catch (error: Throwable) {
            writer?.close()
            outputFile.delete()
            throw error
        } finally {
            writer?.close()
            if (codecStarted) {
                codec?.stop()
            }
            codec?.release()
            extractor.release()
        }
    }

    private fun findAudioTrack(extractor: MediaExtractor): Int {
        for (index in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(index)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return index
        }
        return -1
    }

    private fun createWriter(outputFile: File, format: MediaFormat): WavFileWriter {
        val sampleRate = format.optionalInteger(MediaFormat.KEY_SAMPLE_RATE)
            ?: error("Decoder output has no sample rate.")
        val channelCount = format.optionalInteger(MediaFormat.KEY_CHANNEL_COUNT)
            ?: error("Decoder output has no channel count.")
        return WavFileWriter(outputFile, sampleRate, channelCount)
    }

    private fun writePcmAs16BitWav(
        writer: WavFileWriter,
        buffer: ByteBuffer,
        info: MediaCodec.BufferInfo,
        format: MediaFormat,
    ) {
        val encoding = format.optionalInteger(MediaFormat.KEY_PCM_ENCODING)
            ?: AudioFormat.ENCODING_PCM_16BIT
        val duplicate = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        duplicate.position(info.offset)
        duplicate.limit(info.offset + info.size)

        when (encoding) {
            AudioFormat.ENCODING_PCM_16BIT -> {
                val bytes = ByteArray(duplicate.remaining())
                duplicate.get(bytes)
                writer.writePcm16(bytes)
            }
            AudioFormat.ENCODING_PCM_FLOAT -> {
                val bytes = ByteArray((duplicate.remaining() / Float.SIZE_BYTES) * Short.SIZE_BYTES)
                var out = 0
                while (duplicate.remaining() >= Float.SIZE_BYTES) {
                    val sample = duplicate.float.coerceIn(-1f, 1f)
                    val value = (sample * Short.MAX_VALUE).roundToInt()
                        .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                    bytes[out++] = (value and 0xFF).toByte()
                    bytes[out++] = ((value ushr 8) and 0xFF).toByte()
                }
                writer.writePcm16(bytes)
            }
            AudioFormat.ENCODING_PCM_8BIT -> {
                val bytes = ByteArray(duplicate.remaining() * Short.SIZE_BYTES)
                var out = 0
                while (duplicate.hasRemaining()) {
                    val unsigned = duplicate.get().toInt() and 0xFF
                    val value = (unsigned - 128) shl 8
                    bytes[out++] = (value and 0xFF).toByte()
                    bytes[out++] = ((value ushr 8) and 0xFF).toByte()
                }
                writer.writePcm16(bytes)
            }
            else -> error("Unsupported decoder PCM encoding: $encoding")
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

    private fun MediaFormat.optionalInteger(key: String): Int? {
        return if (containsKey(key)) getInteger(key) else null
    }

    private companion object {
        const val TIMEOUT_US = 10_000L
    }
}

data class AudioExportResult(
    val outputFile: File,
)
