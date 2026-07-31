package com.example.musicsourceseparation.benchmark.applive

import android.os.Process
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

internal class AppLiveStageLogcatCapture private constructor(
    val file: File,
    private val command: List<String>,
    private val process: java.lang.Process?,
    private val startError: String?,
) {
    private var finished: JSONObject? = null

    @Synchronized
    fun finish(stage: String): JSONObject {
        finished?.let { return it }
        Log.i(LOG_TAG, "stage=$stage logcat-capture=stopping")
        var exitCode: Int? = null
        val activeProcess = process
        var terminated = activeProcess == null
        if (activeProcess != null) {
            activeProcess.destroy()
            terminated = runCatching { activeProcess.waitFor(STOP_TIMEOUT_SECONDS, TimeUnit.SECONDS) }
                .getOrDefault(false)
            if (!terminated) {
                activeProcess.destroyForcibly()
                terminated = runCatching { activeProcess.waitFor(STOP_TIMEOUT_SECONDS, TimeUnit.SECONDS) }
                    .getOrDefault(false)
            }
            if (terminated) {
                exitCode = runCatching { activeProcess.exitValue() }.getOrNull()
            }
        }
        if (!file.exists()) {
            file.parentFile?.mkdirs()
            file.writeText(startError?.let { "logcat capture failed: $it\n" }.orEmpty())
        }
        return JSONObject()
            .put("requested", true)
            .put("started", process != null)
            .put("pid", Process.myPid())
            .put("command", JSONArray(command))
            .put("terminated", terminated)
            .put("exitCode", exitCode ?: JSONObject.NULL)
            .put("startError", startError ?: JSONObject.NULL)
            .put("file", file.name)
            .put("bytes", file.length())
            .put("sha256", AppLiveHashing.sha256(file))
            .also { finished = it }
    }

    companion object {
        private const val LOG_TAG = "BSS-AppLive"
        private const val STOP_TIMEOUT_SECONDS = 2L

        fun start(stage: String, file: File): AppLiveStageLogcatCapture {
            file.parentFile?.mkdirs()
            file.delete()
            val command = listOf(
                "/system/bin/logcat",
                "--pid=${Process.myPid()}",
                "-v",
                "threadtime",
                "-T",
                "1",
            )
            val started = runCatching {
                ProcessBuilder(command)
                    .redirectErrorStream(true)
                    .redirectOutput(file)
                    .start()
            }
            val capture = AppLiveStageLogcatCapture(
                file = file,
                command = command,
                process = started.getOrNull(),
                startError = started.exceptionOrNull()?.stackTraceToString(),
            )
            Log.i(
                LOG_TAG,
                "stage=$stage logcat-capture=${if (started.isSuccess) "started" else "failed"} " +
                    "pid=${Process.myPid()}",
            )
            return capture
        }
    }
}
