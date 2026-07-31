package com.example.musicsourceseparation.runtime

import android.os.SystemClock
import com.google.ai.edge.litert.LiteRtNativeLibraryLoader
import java.io.File

internal object DownloadableLiteRtNativeLoader {
    private const val LOADER_CLASS_NAME =
        "com.google.ai.edge.litert.LiteRtNativeLibraryLoader"

    fun configureAndLoad(library: File): DownloadableLiteRtNativeLoaderReport {
        val path = library.canonicalPath
        LiteRtNativeLibraryLoader.configureAbsolutePath(path)
        check(LiteRtNativeLibraryLoader.configuredAbsolutePath() == path) {
            "LiteRT loader did not retain the configured path"
        }

        val loadStarted = SystemClock.elapsedRealtimeNanos()
        LiteRtNativeLibraryLoader.load()
        val loadWallMs = (SystemClock.elapsedRealtimeNanos() - loadStarted) / 1_000_000.0
        check(LiteRtNativeLibraryLoader.isLoaded()) { "LiteRT loader did not report loaded" }

        LiteRtNativeLibraryLoader.configureAbsolutePath(path)
        val conflictingPath = File(library.parentFile, "conflicting-${library.name}").canonicalPath
        val conflict = runCatching {
            LiteRtNativeLibraryLoader.configureAbsolutePath(conflictingPath)
        }.exceptionOrNull()
        check(conflict is IllegalStateException) {
            "LiteRT loader did not reject a conflicting native-library path"
        }

        return DownloadableLiteRtNativeLoaderReport(
            className = LOADER_CLASS_NAME,
            configuredAbsolutePath = requireNotNull(
                LiteRtNativeLibraryLoader.configuredAbsolutePath()
            ),
            loaded = LiteRtNativeLibraryLoader.isLoaded(),
            loadWallMs = loadWallMs,
            samePathReconfigurationAccepted = true,
            conflictingPathRejected = true,
            conflictingPathErrorType = conflict::class.java.name,
        )
    }
}
