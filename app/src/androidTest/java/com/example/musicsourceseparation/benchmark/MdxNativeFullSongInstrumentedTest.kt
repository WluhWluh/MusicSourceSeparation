package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.audio.AudioPcmDecoder
import com.example.musicsourceseparation.audio.DecodedPcmAudio
import com.example.musicsourceseparation.audio.WavFileWriter
import com.example.musicsourceseparation.model.MdxDspConfig
import com.example.musicsourceseparation.model.NativeMdxDsp
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest
import kotlin.math.ceil
import kotlin.math.roundToInt

@RunWith(AndroidJUnit4::class)
class MdxNativeFullSongInstrumentedTest {
    @Test
    fun separateCanonicalSong() {
        val args = InstrumentationRegistry.getArguments()
        val context = ApplicationProvider.getApplicationContext<Context>()
        val root = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val modelId = requireNotNull(args.getString("modelId"))
        val modelFile = File(root, "models/${requireNotNull(args.getString("modelFile"))}")
        val contractFile = File(root, "contracts/${requireNotNull(args.getString("contractFile"))}")
        val audioFile = File(root, "audio-input/${requireNotNull(args.getString("audioFile"))}")
        val backend = args.getString("backend", "gpu-bounded")!!
        val profile = args.getString("dspProfile", "native-full")!!
        require(profile == "native-full" || profile == "native-packed")
        val contract = JSONObject(contractFile.readText())
        require(sha256(modelFile) == contract.getJSONObject("artifact").getString("sha256"))
        val dsp = contract.getJSONObject("dsp")
        val config = MdxDspConfig(
            sampleRate = dsp.getInt("sampleRate"),
            nFft = dsp.getInt("nFft"),
            hopLength = dsp.getInt("hopLength"),
            dimF = dsp.getInt("dimF"),
            dimTPower = dsp.getInt("dimTPower"),
        )
        val outputDir = File(root, "native-full-song/$modelId/$backend/$profile").apply {
            deleteRecursively(); mkdirs()
        }
        val report = JSONObject()
            .put("status", "running")
            .put("modelId", modelId)
            .put("modelSha256", sha256(modelFile))
            .put("contractSha256", sha256(contractFile))
            .put("sourceSha256", sha256(audioFile))
            .put("backend", backend)
            .put("dspProfile", profile)
            .put("device", JSONObject().put("model", Build.MODEL).put("soc", Build.SOC_MODEL))
        val powerManager = context.getSystemService(PowerManager::class.java)
        report.put("thermalStatusStart", powerManager.currentThermalStatus)
        try {
            val started = SystemClock.elapsedRealtimeNanos()
            val decoded = AudioPcmDecoder(context).decode(Uri.fromFile(audioFile))
            require(decoded.sampleRate == config.sampleRate && decoded.channelCount == 2)
            val decodeMs = elapsedMs(started)
            val options = if (backend == "gpu-bounded") {
                CompiledModel.Options(Accelerator.GPU).apply {
                    gpuOptions = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP32,
                        backend = CompiledModel.GpuOptions.Backend.OPENCL,
                        numStepsOfCommandBufferPreparations = 1,
                    )
                }
            } else {
                CompiledModel.Options(Accelerator.CPU).apply {
                    cpuOptions = CompiledModel.CpuOptions(4, null, null)
                }
            }
            val environment = Environment.create()
            val boundedRuntime = if (backend == "gpu-bounded") BoundedGpuRuntime.loadAndValidate() else null
            val setupStarted = SystemClock.elapsedRealtimeNanos()
            val model = CompiledModel.create(modelFile.absolutePath, options, environment)
            val input = model.createInputBuffers().single()
            val output = model.createOutputBuffers().single()
            val setupMs = elapsedMs(setupStarted)
            val nativeMode = if (profile == "native-packed") {
                NativeMdxDsp.Mode.PACKED_REAL
            } else {
                NativeMdxDsp.Mode.FULL_COMPLEX
            }
            val inputTensor = FloatArray(config.tensorElementCount)
            val separated = Array(2) { FloatArray(config.chunkSize) }
            val residual = Array(2) { FloatArray(config.chunkSize) }
            val pcm = ByteArray(config.generationSize * 4)
            val modelStemFile = File(outputDir, "model-output.wav")
            val residualFile = File(outputDir, "residual.wav")
            val stageNanos = linkedMapOf<String, Long>()
            fun <T> timed(name: String, action: () -> T): T {
                val stageStarted = SystemClock.elapsedRealtimeNanos()
                return try { action() } finally {
                    stageNanos[name] = (stageNanos[name] ?: 0L) +
                        SystemClock.elapsedRealtimeNanos() - stageStarted
                }
            }
            val windowCount = ceil(decoded.frameCount.toDouble() / config.generationSize).toInt()
            val processingStarted = SystemClock.elapsedRealtimeNanos()
            boundedRuntime?.resetInferenceCounters()
            NativeMdxDsp(config, 4, nativeMode).use { nativeDsp ->
                WavFileWriter(modelStemFile, config.sampleRate, 2).use { modelWriter ->
                    WavFileWriter(residualFile, config.sampleRate, 2).use { residualWriter ->
                        for (windowIndex in 0 until windowCount) {
                            val generationStart = windowIndex * config.generationSize
                            val writeFrames = minOf(config.generationSize, decoded.frameCount - generationStart)
                            val mix = timed("windowInput") {
                                decoded.toStereoFloatContextWindow(generationStart - config.trim, config.chunkSize)
                            }
                            timed("stft") { nativeDsp.waveformToNhwcTensorInto(mix, inputTensor) }
                            timed("inputWrite") { input.writeFloat(inputTensor) }
                            timed("invoke") {
                                boundedRuntime?.beginInference()
                                try { model.run(listOf(input), listOf(output)) } finally {
                                    boundedRuntime?.endInference()
                                }
                            }
                            val outputTensor = timed("outputRead") { output.readFloat() }
                            timed("iStftOla") { nativeDsp.nhwcTensorToWaveformInto(outputTensor, separated) }
                            timed("residual") {
                                val scale = dsp.getDouble("modelOutputScale").toFloat()
                                for (channel in 0 until 2) for (sample in 0 until config.chunkSize) {
                                    separated[channel][sample] *= scale
                                    residual[channel][sample] = mix[channel][sample] - separated[channel][sample]
                                }
                            }
                            timed("pcmWrite") {
                                fillPcm16(separated, config.trim, writeFrames, pcm)
                                modelWriter.writePcm16(pcm, 0, writeFrames * 4)
                                fillPcm16(residual, config.trim, writeFrames, pcm)
                                residualWriter.writePcm16(pcm, 0, writeFrames * 4)
                            }
                        }
                    }
                }
            }
            val processingMs = elapsedMs(processingStarted)
            input.close(); output.close(); model.close(); environment.close()
            val stages = JSONObject()
            stageNanos.forEach { (name, nanos) -> stages.put(name, nanos / 1e6) }
            report.put("status", "complete")
                .put("decodeMs", decodeMs)
                .put("setupMs", setupMs)
                .put("processingMs", processingMs)
                .put("audioSeconds", decoded.frameCount.toDouble() / config.sampleRate)
                .put("windowCount", windowCount)
                .put("stagesMs", stages)
                .put("modelOutput", fileEvidence(modelStemFile))
                .put("residualOutput", fileEvidence(residualFile))
                .put("memory", memoryEvidence())
            boundedRuntime?.let { report.put("boundedGpuEvidence", it.evidence()) }
        } catch (error: Throwable) {
            report.put("status", "error").put("errorClass", error::class.java.name)
                .put("message", error.message.orEmpty()).put("stack", error.stackTraceToString())
        }
        report.put("thermalStatusEnd", powerManager.currentThermalStatus)
        File(outputDir, "report.json").writeText(report.toString(2))
        check(report.getString("status") == "complete") { report.toString() }
    }

    private fun DecodedPcmAudio.toStereoFloatContextWindow(start: Int, frames: Int): Array<FloatArray> {
        val copyStart = maxOf(0, start)
        val copyEnd = minOf(frameCount, start + frames)
        val result = Array(2) { FloatArray(frames) }
        if (copyEnd <= copyStart) return result
        val source = toStereoFloat(copyStart, copyEnd - copyStart)
        val offset = copyStart - start
        for (channel in 0 until 2) source[channel].copyInto(result[channel], offset)
        return result
    }

    private fun fillPcm16(waveform: Array<FloatArray>, start: Int, frames: Int, output: ByteArray) {
        var offset = 0
        for (sample in start until start + frames) for (channel in 0 until 2) {
            val value = (waveform[channel][sample].coerceIn(-1f, 1f) * 32767f).roundToInt()
                .coerceIn(-32768, 32767)
            output[offset++] = value.toByte()
            output[offset++] = (value ushr 8).toByte()
        }
    }

    private fun fileEvidence(file: File): JSONObject = JSONObject()
        .put("path", file.absolutePath).put("bytes", file.length()).put("sha256", sha256(file))

    private fun memoryEvidence(): JSONObject = Debug.MemoryInfo().also(Debug::getMemoryInfo).let {
        JSONObject().put("pssKb", it.totalPss).put("nativeHeapBytes", Debug.getNativeHeapAllocatedSize())
    }

    private fun elapsedMs(started: Long): Double = (SystemClock.elapsedRealtimeNanos() - started) / 1e6

    private fun sha256(file: File): String = MessageDigest.getInstance("SHA-256")
        .digest(file.readBytes()).joinToString("") { "%02x".format(it) }

    private class BoundedGpuRuntime private constructor(
        private val runtimeClass: Class<*>,
        private val capability: Any,
    ) {
        fun resetInferenceCounters() { invoke("resetInferenceCounters") }
        fun beginInference() { invoke("beginInference") }
        fun endInference() { invoke("endInference") }
        fun evidence(): JSONObject = JSONObject()
            .put("artifactVersion", value("getArtifactVersion"))
            .put("profileId", value("getProfileId"))
            .put("dispatchCount", invoke("getDispatchCount"))
            .put("eventWaitCount", invoke("getEventWaitCount"))
        private fun invoke(name: String): Any? = runtimeClass.getMethod(name).invoke(null)
        private fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)

        companion object {
            fun loadAndValidate(): BoundedGpuRuntime {
                val clazz = Class.forName("io.github.wluhwluh.bss.litert.BssLiteRtRuntime")
                val capability = requireNotNull(clazz.getMethod("queryCapability").invoke(null))
                fun value(name: String): Any? = capability.javaClass.getMethod(name).invoke(capability)
                require(value("isAvailable") == true)
                require(value("getArtifactVersion") == "2.1.5-bss.2")
                require(value("getProfileId") == "gpu-opencl-bounded-fp32-v1")
                return BoundedGpuRuntime(clazz, capability)
            }
        }
    }
}
