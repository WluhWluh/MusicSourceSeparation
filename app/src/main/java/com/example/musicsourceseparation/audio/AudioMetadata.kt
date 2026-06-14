package com.example.musicsourceseparation.audio

data class AudioMetadata(
    val displayName: String,
    val mimeType: String?,
    val sizeBytes: Long?,
    val durationMs: Long?,
    val sampleRate: Int?,
    val channelCount: Int?,
) {
    fun toDisplayText(): String {
        val lines = mutableListOf(displayName)
        durationMs?.let { lines += "Duration: ${formatDuration(it)}" }
        sampleRate?.let { lines += "Sample rate: $it Hz" }
        channelCount?.let { lines += "Channels: $it" }
        sizeBytes?.let { lines += "Size: ${formatBytes(it)}" }
        mimeType?.let { lines += "Type: $it" }
        return lines.joinToString(separator = "\n")
    }

    private fun formatDuration(durationMs: Long): String {
        val totalSeconds = durationMs / 1000
        val minutes = totalSeconds / 60
        val seconds = totalSeconds % 60
        return "%d:%02d".format(minutes, seconds)
    }

    private fun formatBytes(bytes: Long): String {
        val mib = bytes.toDouble() / (1024.0 * 1024.0)
        return "%.1f MiB".format(mib)
    }
}
