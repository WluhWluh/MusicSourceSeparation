package com.example.musicsourceseparation.model

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.abs

class MdxOnnxSmokeTester(
    private val context: Context,
    private val modelVariant: MdxModelVariant = MdxModelVariant.INST_MAIN,
) {
    fun run(modelVariant: MdxModelVariant = this.modelVariant): MdxSmokeTestResult {
        val modelFile = MdxModelFile.get(context, modelVariant)

        val environment = OrtEnvironment.getEnvironment()
        environment.createSession(modelFile.absolutePath, OrtSession.SessionOptions()).use { session ->
            val inputInfo = session.inputInfo.keys.first()
            val outputInfo = session.outputInfo.keys.first()
            val shape = longArrayOf(1, 4, 2048, 256)
            val inputElementCount = shape.fold(1L) { total, dim -> total * dim }.toInt()
            val zeros = FloatArray(inputElementCount)

            OnnxTensor.createTensor(environment, FloatBuffer.wrap(zeros), shape).use { tensor ->
                session.run(mapOf(inputInfo to tensor)).use { outputs ->
                    val output = outputs[outputInfo].orElseThrow {
                        IllegalStateException("Missing ONNX output: $outputInfo")
                    }.value
                    @Suppress("UNCHECKED_CAST")
                    val outputArray = output as Array<Array<Array<FloatArray>>>
                    return MdxSmokeTestResult(
                        modelPath = modelFile.absolutePath,
                        modelVariant = modelVariant,
                        inputName = inputInfo,
                        outputName = outputInfo,
                        outputShape = listOf(
                            outputArray.size,
                            outputArray.firstOrNull()?.size ?: 0,
                            outputArray.firstOrNull()?.firstOrNull()?.size ?: 0,
                            outputArray.firstOrNull()?.firstOrNull()?.firstOrNull()?.size ?: 0,
                        ),
                        outputAbsMean = outputArray.absMean(),
                    )
                }
            }
        }
    }

    private fun Array<Array<Array<FloatArray>>>.absMean(): Float {
        var sum = 0.0
        var count = 0L
        for (batch in this) {
            for (channel in batch) {
                for (row in channel) {
                    for (value in row) {
                        sum += abs(value.toDouble())
                        count += 1
                    }
                }
            }
        }
        return if (count == 0L) 0f else (sum / count).toFloat()
    }
}

data class MdxSmokeTestResult(
    val modelPath: String,
    val modelVariant: MdxModelVariant,
    val inputName: String,
    val outputName: String,
    val outputShape: List<Int>,
    val outputAbsMean: Float,
) {
    fun toDisplayText(): String {
        return buildString {
            appendLine("ONNX smoke test complete")
            appendLine("Model: ${modelVariant.displayName}")
            appendLine("Model: $modelPath")
            appendLine("Input: $inputName")
            appendLine("Output: $outputName")
            appendLine("Output shape: $outputShape")
            append("Output abs mean: $outputAbsMean")
        }
    }
}
