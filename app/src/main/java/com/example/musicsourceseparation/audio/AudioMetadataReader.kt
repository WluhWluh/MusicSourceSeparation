package com.example.musicsourceseparation.audio

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns

class AudioMetadataReader(private val context: Context) {
    fun read(uri: Uri): AudioMetadata {
        val resolver = context.contentResolver
        val displayName = queryDisplayName(uri) ?: uri.lastPathSegment ?: uri.toString()
        val sizeBytes = querySize(uri)
        val mimeType = resolver.getType(uri)
        val durationMs = readDuration(uri)
        val trackFormat = readAudioTrackFormat(uri)

        return AudioMetadata(
            displayName = displayName,
            mimeType = mimeType,
            sizeBytes = sizeBytes,
            durationMs = durationMs,
            sampleRate = trackFormat?.sampleRate,
            channelCount = trackFormat?.channelCount,
        )
    }

    private fun queryDisplayName(uri: Uri): String? {
        return queryOpenableColumn(uri, OpenableColumns.DISPLAY_NAME) as? String
    }

    private fun querySize(uri: Uri): Long? {
        return queryOpenableColumn(uri, OpenableColumns.SIZE) as? Long
    }

    private fun queryOpenableColumn(uri: Uri, column: String): Any? {
        context.contentResolver.query(uri, arrayOf(column), null, null, null)?.use { cursor ->
            if (!cursor.moveToFirst()) return null
            val index = cursor.getColumnIndex(column)
            if (index < 0 || cursor.isNull(index)) return null
            return when (cursor.getType(index)) {
                android.database.Cursor.FIELD_TYPE_INTEGER -> cursor.getLong(index)
                android.database.Cursor.FIELD_TYPE_STRING -> cursor.getString(index)
                else -> null
            }
        }
        return null
    }

    private fun readDuration(uri: Uri): Long? {
        return runCatching {
            val retriever = MediaMetadataRetriever()
            retriever.use {
                it.setDataSource(context, uri)
                it.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
            }
        }.getOrNull()
    }

    private fun readAudioTrackFormat(uri: Uri): TrackFormat? {
        val extractor = MediaExtractor()
        return try {
            extractor.setDataSource(context, uri, null)
            for (index in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(index)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (!mime.startsWith("audio/")) continue
                return TrackFormat(
                    sampleRate = format.optionalInteger(MediaFormat.KEY_SAMPLE_RATE),
                    channelCount = format.optionalInteger(MediaFormat.KEY_CHANNEL_COUNT),
                )
            }
            null
        } catch (_: RuntimeException) {
            null
        } finally {
            extractor.release()
        }
    }

    private fun MediaFormat.optionalInteger(key: String): Int? {
        return if (containsKey(key)) getInteger(key) else null
    }

    private data class TrackFormat(
        val sampleRate: Int?,
        val channelCount: Int?,
    )
}
