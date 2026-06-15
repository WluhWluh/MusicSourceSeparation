package com.example.musicsourceseparation.model

import android.content.Context
import java.io.File
import java.io.IOException

object MdxModelFile {
    const val FILE_NAME = "UVR-MDX-NET-Inst_Main.onnx"

    fun get(context: Context): File {
        val target = File(context.filesDir, FILE_NAME)
        if (target.isFile && target.length() > 0L) return target

        copyBundledAsset(context, target)
        require(target.isFile && target.length() > 0L) {
            "Model file missing: ${target.absolutePath}"
        }
        return target
    }

    private fun copyBundledAsset(context: Context, target: File) {
        val temp = File(context.filesDir, "$FILE_NAME.tmp")
        try {
            context.assets.open(FILE_NAME).use { input ->
                temp.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        } catch (error: IOException) {
            temp.delete()
            throw IllegalStateException(
                "Bundled model asset missing. Build the APK with models/uvr-mdx/$FILE_NAME available.",
                error,
            )
        }

        if (target.exists() && !target.delete()) {
            temp.delete()
            error("Could not replace model file: ${target.absolutePath}")
        }
        if (!temp.renameTo(target)) {
            temp.copyTo(target, overwrite = true)
            temp.delete()
        }
    }
}
