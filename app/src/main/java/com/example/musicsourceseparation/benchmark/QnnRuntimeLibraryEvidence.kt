package com.example.musicsourceseparation.benchmark

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

internal object QnnRuntimeLibraryEvidence {
    fun collect(libraryDirectory: String): JSONArray {
        val root = File(libraryDirectory)
        val libraries = root.listFiles().orEmpty()
            .filter { file ->
                file.isFile &&
                    file.extension == "so" &&
                    (file.name.startsWith("libLiteRt") || file.name.startsWith("libQnn"))
            }
            .sortedBy(File::getName)
        return JSONArray(libraries.map { file ->
            JSONObject()
                .put("fileName", file.name)
                .put("bytes", file.length())
                .put("sha256", sha256(file))
        })
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }
    }
}
