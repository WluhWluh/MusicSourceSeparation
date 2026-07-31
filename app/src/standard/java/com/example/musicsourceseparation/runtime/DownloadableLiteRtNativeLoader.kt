package com.example.musicsourceseparation.runtime

import java.io.File

internal object DownloadableLiteRtNativeLoader {
    fun configureAndLoad(library: File): DownloadableLiteRtNativeLoaderReport {
        error("Downloadable LiteRT loading is unavailable in the standard runtime flavor")
    }
}
