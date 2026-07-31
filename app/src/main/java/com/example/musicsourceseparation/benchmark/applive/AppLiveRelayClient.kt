package com.example.musicsourceseparation.benchmark.applive

import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

internal class AppLiveRelayClient(
    private val config: AppLiveRelayConfig,
) {
    fun upload(runId: String, fileName: String, file: File, contentType: String): JSONObject {
        require(runId.matches(safeName)) { "Invalid relay run ID: $runId" }
        require(fileName.matches(safeName)) { "Invalid relay file name: $fileName" }
        require(file.isFile) { "Upload file is missing: ${file.absolutePath}" }
        val sha256 = AppLiveHashing.sha256(file)
        var lastFailure: Throwable? = null
        repeat(MAX_ATTEMPTS) { attempt ->
            try {
                return uploadOnce(runId, fileName, file, contentType, sha256)
            } catch (error: Throwable) {
                lastFailure = error
                if (attempt + 1 < MAX_ATTEMPTS) {
                    Thread.sleep(1_000L shl attempt)
                }
            }
        }
        throw IllegalStateException("Relay upload failed after $MAX_ATTEMPTS attempts: $fileName", lastFailure)
    }

    private fun uploadOnce(
        runId: String,
        fileName: String,
        file: File,
        contentType: String,
        sha256: String,
    ): JSONObject {
        val url = URL(
            "${config.baseUrl}/v1/campaigns/${config.campaign}/runs/$runId/files/$fileName",
        )
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "PUT"
            connectTimeout = 15_000
            readTimeout = 60 * 60_000
            doOutput = true
            useCaches = false
            setFixedLengthStreamingMode(file.length())
            setRequestProperty("Authorization", "Bearer ${config.uploadToken}")
            setRequestProperty("Content-Type", contentType)
            setRequestProperty("X-BSS-Expected-Size", file.length().toString())
            setRequestProperty("X-BSS-Expected-SHA256", sha256)
        }
        try {
            connection.outputStream.buffered().use { output ->
                file.inputStream().buffered().use { input -> input.copyTo(output) }
            }
            val responseCode = connection.responseCode
            val responseText = (if (responseCode in 200..299) {
                connection.inputStream
            } else {
                connection.errorStream
            })?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (responseCode !in 200..299) {
                throw IllegalStateException("Relay returned HTTP $responseCode for $fileName: $responseText")
            }
            return JSONObject(responseText)
        } finally {
            connection.disconnect()
        }
    }

    private companion object {
        const val MAX_ATTEMPTS = 3
        val safeName = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
    }
}
