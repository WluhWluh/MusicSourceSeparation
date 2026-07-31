package com.example.musicsourceseparation.runtime

import org.json.JSONObject

internal data class DownloadableLiteRtNativeLoaderReport(
    val className: String,
    val configuredAbsolutePath: String,
    val loaded: Boolean,
    val loadWallMs: Double,
    val samePathReconfigurationAccepted: Boolean,
    val conflictingPathRejected: Boolean,
    val conflictingPathErrorType: String,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("className", className)
        .put("configuredAbsolutePath", configuredAbsolutePath)
        .put("loaded", loaded)
        .put("loadWallMs", loadWallMs)
        .put("samePathReconfigurationAccepted", samePathReconfigurationAccepted)
        .put("conflictingPathRejected", conflictingPathRejected)
        .put("conflictingPathErrorType", conflictingPathErrorType)
}
