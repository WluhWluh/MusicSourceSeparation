package com.example.musicsourceseparation.model

import ai.onnxruntime.OrtSession

data class MdxRuntimeSettings(
    val cpuThreads: Int = 0,
) {
    fun createSessionOptions(): OrtSession.SessionOptions {
        return OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setInterOpNumThreads(1)
            if (cpuThreads > 0) {
                setIntraOpNumThreads(cpuThreads)
            }
        }
    }

    fun toDisplayText(): String {
        val threadText = if (cpuThreads > 0) cpuThreads.toString() else "ORT default"
        return "Runtime: CPU threads=$threadText, optimization=ALL_OPT, execution=SEQUENTIAL"
    }
}
