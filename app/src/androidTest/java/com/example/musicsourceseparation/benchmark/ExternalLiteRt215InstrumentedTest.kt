package com.example.musicsourceseparation.benchmark

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.musicsourceseparation.BuildConfig
import com.example.musicsourceseparation.benchmark.contract.DerivedLiteRtCandidateContractLoader
import com.example.musicsourceseparation.benchmark.contract.ExternalLiteRtCandidateContract
import com.example.musicsourceseparation.benchmark.contract.ExternalLiteRtCandidateContractLoader
import com.example.musicsourceseparation.benchmark.contract.ExternalLiteRtFlatBuffer
import com.example.musicsourceseparation.benchmark.contract.ExternalLiteRtSignature
import com.example.musicsourceseparation.benchmark.contract.ExternalLiteRtTensor
import com.example.musicsourceseparation.benchmark.contract.GeneratedLiteRtCandidateContractLoader
import com.example.musicsourceseparation.benchmark.contract.GeneratedLiteRtHostCandidateManifestLoader
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class ExternalLiteRt215InstrumentedTest {
    @Test
    fun probeExternalLiteRtCandidate() {
        val arguments = InstrumentationRegistry.getArguments()
        val candidate = ProbeCandidate.fromWireValue(
            requireNotNull(arguments.getString(ARG_CANDIDATE)) {
                "Pass -e $ARG_CANDIDATE <candidate>."
            },
        )
        val phase = ProbePhase.fromWireValue(arguments.getString(ARG_PHASE) ?: "prepare")
        val threads = (arguments.getString(ARG_THREADS)?.toIntOrNull() ?: 4).coerceIn(1, 16)
        val exportOutputs = arguments.getString(ARG_EXPORT_OUTPUTS)?.toBooleanStrictOrNull() ?: false
        val warmupRuns = boundedCount(arguments.getString(ARG_WARMUP_RUNS), 0, 0, 10, ARG_WARMUP_RUNS)
        val measuredRuns = boundedCount(arguments.getString(ARG_MEASURED_RUNS), 1, 1, 100, ARG_MEASURED_RUNS)
        if (candidate.engine == ProbeEngine.INTERPRETER) {
            check(warmupRuns == 0 && measuredRuns == 1) {
                "Interpreter probes remain single-run; use CompiledModel for sustained measurements."
            }
        }
        val context = ApplicationProvider.getApplicationContext<Context>()
        val benchmarkRoot = File(requireNotNull(context.getExternalFilesDir(null)), "benchmark")
        val contractFileName = arguments.getString(ARG_CONTRACT) ?: CONTRACT_FILE_NAME
        val contractFile = File(benchmarkRoot, "contracts/$contractFileName")
        val loaded = loadProbeContract(contractFileName, contractFile)
        val contract = loaded.view
        val artifact = loadProbeArtifact(
            benchmarkRoot = benchmarkRoot,
            manifestFileName = arguments.getString(ARG_DERIVATION_MANIFEST),
            contract = loaded,
            contractFile = contractFile,
            contractSha256 = loaded.sidecarSha256,
        )
        val modelFile = File(benchmarkRoot, "models/${artifact.fileName}")
        check(modelFile.isFile && modelFile.length() == artifact.byteSize) {
            "Missing verified external model: $modelFile"
        }
        check(sha256(modelFile) == artifact.sha256) { "External model SHA-256 mismatch." }

        val resultRoot = File(
            benchmarkRoot,
            "litert215-probe/${artifact.modelId}/${candidate.wireValue}/${phase.wireValue}",
        ).apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        val cacheRoot = File(
            context.codeCacheDir,
            "litert215-probe-${artifact.sha256.take(12)}-${candidate.wireValue}",
        ).apply {
            if (exists()) deleteRecursively()
            check(mkdirs())
        }
        val base = baseEvidence(
            context = context,
            candidate = candidate,
            phase = phase,
            threads = threads,
            exportOutputs = exportOutputs,
            warmupRuns = warmupRuns,
            measuredRuns = measuredRuns,
            contract = contract,
            contractFile = contractFile,
            contractSha256 = loaded.sidecarSha256,
            modelFile = modelFile,
            artifact = artifact,
        )
        checkpoint(resultRoot, base, "validated")

        val result = runCatching {
            when (candidate.engine) {
                ProbeEngine.INTERPRETER -> runInterpreter(
                    modelFile = modelFile,
                    contract = contract,
                    fixtureRoot = File(File(benchmarkRoot, "inputs"), artifact.modelId),
                    phase = phase,
                    threads = threads,
                    exportOutputs = exportOutputs,
                    resultRoot = resultRoot,
                    base = base,
                )

                ProbeEngine.COMPILED_MODEL -> runCompiledModel(
                    context = context,
                    modelFile = modelFile,
                    contract = contract,
                    fixtureRoot = File(File(benchmarkRoot, "inputs"), artifact.modelId),
                    candidate = candidate,
                    phase = phase,
                    threads = threads,
                    exportOutputs = exportOutputs,
                    warmupRuns = warmupRuns,
                    measuredRuns = measuredRuns,
                    cacheRoot = cacheRoot,
                    resultRoot = resultRoot,
                    base = base,
                    modelSha256 = artifact.sha256,
                )
            }
        }
        result.onSuccess { evidence ->
            val report = JSONObject(base.toString())
                .put("status", "complete")
                .put("stage", "complete")
                .put("result", evidence)
                .put("process", processSnapshot())
            publishJson(File(resultRoot, "result.json"), report)
            Log.i(LOG_TAG, report.toString())
        }
        result.onFailure { error ->
            val report = JSONObject(base.toString())
                .put("status", "error")
                .put("stage", "caught-error")
                .put("errorClass", error::class.java.name)
                .put("message", error.message.orEmpty())
                .put("stack", error.stackTraceToString())
                .put("process", processSnapshot())
            publishJson(File(resultRoot, "failure.json"), report)
            Log.e(LOG_TAG, report.toString(), error)
        }
        result.getOrThrow()
    }

    private fun runInterpreter(
        modelFile: File,
        contract: ProbeContractView,
        fixtureRoot: File,
        phase: ProbePhase,
        threads: Int,
        exportOutputs: Boolean,
        resultRoot: File,
        base: JSONObject,
    ): JSONObject {
        val signature = contract.flatBuffer.signatures.single()
        val inputs = signatureInputTensors(contract, signature)
        val outputs = signatureOutputTensors(contract, signature)
        val started = TimedStart()
        checkpoint(resultRoot, base, "interpreter-create-start")
        val interpreter = Interpreter(
            modelFile,
            Interpreter.Options()
                .setUseXNNPACK(true)
                .setNumThreads(threads),
        )
        return try {
            checkpoint(resultRoot, base, "interpreter-created")
            check(interpreter.signatureKeys.toList() == listOf(signature.key))
            val inputBuffers = linkedMapOf<String, Any>()
            val outputBuffers = linkedMapOf<String, Any>()
            val tensorTypes = JSONObject()
            inputs.forEachIndexed { index, (bindingName, tensor) ->
                val runtimeTensor = interpreter.getInputTensorFromSignature(bindingName, signature.key)
                check(runtimeTensor.dataType().toString() == "FLOAT32")
                check(runtimeTensor.shape().toList() == tensor.shape)
                val buffer = directBuffer(tensor.byteSize)
                if (phase == ProbePhase.RUN) {
                    fillFixtureOrDeterministic(
                        buffer = buffer,
                        tensor = tensor,
                        bindingName = bindingName,
                        index = index,
                        fixtureRoot = fixtureRoot,
                        contract = contract,
                    )
                }
                inputBuffers[bindingName] = buffer
                tensorTypes.put("input:$bindingName", tensorTypeEvidence(tensor.shape, "FLOAT32"))
            }
            outputs.forEach { (bindingName, tensor) ->
                val runtimeTensor = interpreter.getOutputTensorFromSignature(bindingName, signature.key)
                check(runtimeTensor.dataType().toString() == "FLOAT32")
                check(runtimeTensor.shape().toList() == tensor.shape)
                outputBuffers[bindingName] = directBuffer(tensor.byteSize)
                tensorTypes.put("output:$bindingName", tensorTypeEvidence(tensor.shape, "FLOAT32"))
            }
            val prepared = started.elapsed()
            checkpoint(
                resultRoot,
                base,
                "buffers-created",
                JSONObject().put("prepareWallMs", prepared.wallMs),
            )
            if (phase == ProbePhase.PREPARE) {
                return JSONObject()
                    .put("prepareWallMs", prepared.wallMs)
                    .put("prepareCpuMs", prepared.cpuMs)
                    .put("tensorTypes", tensorTypes)
                    .put("inference", JSONObject.NULL)
            }

            checkpoint(resultRoot, base, "inference-start")
            val inferenceStarted = TimedStart()
            interpreter.runSignature(inputBuffers, outputBuffers, signature.key)
            val inference = inferenceStarted.elapsed()
            checkpoint(
                resultRoot,
                base,
                "inference-complete",
                JSONObject().put("inferenceWallMs", inference.wallMs),
            )
            val outputEvidence = JSONObject()
            outputs.forEach { (bindingName, tensor) ->
                checkpoint(resultRoot, base, "read-output-$bindingName")
                val buffer = outputBuffers.getValue(bindingName) as ByteBuffer
                val outputFile = if (exportOutputs) {
                    File(resultRoot, "$bindingName.f32le")
                } else {
                    null
                }
                outputEvidence.put(
                    bindingName,
                    summarize(
                        buffer.duplicate().order(ByteOrder.nativeOrder()).apply { clear() }.asFloatBuffer(),
                        tensor,
                        outputFile,
                    ),
                )
            }
            JSONObject()
                .put("prepareWallMs", prepared.wallMs)
                .put("prepareCpuMs", prepared.cpuMs)
                .put("tensorTypes", tensorTypes)
                .put("inference", JSONObject()
                    .put("wallMs", inference.wallMs)
                    .put("cpuMs", inference.cpuMs)
                    .put(
                        "nativeMs",
                        interpreter.lastNativeInferenceDurationNanoseconds?.div(1_000_000.0),
                    ))
                .put("outputs", outputEvidence)
        } finally {
            interpreter.close()
        }
    }

    private fun runCompiledModel(
        context: Context,
        modelFile: File,
        contract: ProbeContractView,
        fixtureRoot: File,
        candidate: ProbeCandidate,
        phase: ProbePhase,
        threads: Int,
        exportOutputs: Boolean,
        warmupRuns: Int,
        measuredRuns: Int,
        cacheRoot: File,
        resultRoot: File,
        base: JSONObject,
        modelSha256: String,
    ): JSONObject {
        val signature = contract.flatBuffer.signatures.single()
        val inputs = signatureInputTensors(contract, signature)
        val outputs = signatureOutputTensors(contract, signature)
        val qnnEvidenceDir = File(resultRoot, "qnn-ir").apply { mkdirs() }
        val provider = if (Accelerator.NPU in candidate.accelerators) {
            BuiltinNpuAcceleratorProvider(context, QnnDeviceCompatibility.checker).also {
                check(it.isDeviceSupported()) { "LiteRT does not support this device for NPU." }
                check(it.isLibraryReady()) { "The LiteRT Qualcomm runtime is not ready." }
            }
        } else {
            null
        }
        val coldPrepareStarted = TimedStart()
        checkpoint(resultRoot, base, "environment-create-start")
        val environmentStarted = TimedStart()
        val environment = provider?.let { Environment.create(it) } ?: Environment.create()
        val environmentTiming = environmentStarted.elapsed()
        var model: CompiledModel? = null
        val inputBuffers = linkedMapOf<String, TensorBuffer>()
        val outputBuffers = linkedMapOf<String, TensorBuffer>()
        return try {
            val availableAccelerators = environment.getAvailableAccelerators().map { it.name }.sorted()
            checkpoint(
                resultRoot,
                base,
                "compiled-model-create-start",
                JSONObject().put("availableAccelerators", JSONArray(availableAccelerators)),
            )
            val options = CompiledModel.Options(*candidate.accelerators.toTypedArray()).apply {
                cpuOptions = CompiledModel.CpuOptions(
                    numThreads = threads,
                    xnnPackFlags = null,
                    xnnPackWeightCachePath = null,
                )
                candidate.gpuProfile?.let { profile ->
                    gpuOptions = CompiledModel.GpuOptions(
                        constantTensorSharing = profile.constantTensorSharing,
                        precision = profile.precision,
                        bufferStorageType = profile.bufferStorageType,
                        preferTextureWeights = profile.preferTextureWeights,
                        serializationDir = cacheRoot.absolutePath,
                        modelCacheKey = "mss-${candidate.wireValue}-$modelSha256",
                        serializeProgramCache = false,
                        backend = profile.backend,
                        priority = CompiledModel.GpuOptions.Priority.HIGH,
                        numStepsOfCommandBufferPreparations = profile.commandPreparationSteps,
                    )
                }
                candidate.qnnOptimization?.let { optimization ->
                    qualcommOptions = CompiledModel.QualcommOptions(
                        logLevel = CompiledModel.QualcommOptions.LogLevel.INFO,
                        useHtpPreference = true,
                        useConvHmx = candidate.qnnUseConvHmx,
                        htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode
                            .SUSTAINED_HIGH_PERFORMANCE,
                        profiling = CompiledModel.QualcommOptions.Profiling.OFF,
                        irJsonDir = qnnEvidenceDir.absolutePath,
                        numHvxThreads = candidate.qnnNumHvxThreads,
                        optimizationLevel = optimization,
                    )
                }
            }
            val compileStarted = TimedStart()
            val compiledModel = CompiledModel.create(modelFile.absolutePath, options, environment)
            val compileTiming = compileStarted.elapsed()
            model = compiledModel
            checkpoint(resultRoot, base, "compiled-model-created")

            val tensorTypes = JSONObject()
            val bufferAllocationStarted = TimedStart()
            inputs.forEach { (bindingName, tensor) ->
                val type = compiledModel.getInputTensorType(bindingName, signature.key)
                checkCompiledTensorType(type, tensor)
                inputBuffers[bindingName] = compiledModel.createInputBuffer(bindingName, signature.key)
                tensorTypes.put("input:$bindingName", tensorTypeEvidence(type))
            }
            outputs.forEach { (bindingName, tensor) ->
                val type = compiledModel.getOutputTensorType(bindingName, signature.key)
                checkCompiledTensorType(type, tensor)
                outputBuffers[bindingName] = compiledModel.createOutputBuffer(bindingName, signature.key)
                tensorTypes.put("output:$bindingName", tensorTypeEvidence(type))
            }
            val bufferAllocationTiming = bufferAllocationStarted.elapsed()
            val prepared = coldPrepareStarted.elapsed()
            val prepareStages = JSONObject()
                .put("environmentCreate", environmentTiming.evidence())
                .put("compiledModelCreate", compileTiming.evidence())
                .put("bufferAllocation", bufferAllocationTiming.evidence())
            checkpoint(
                resultRoot,
                base,
                "buffers-created",
                JSONObject()
                    .put("prepareWallMs", prepared.wallMs)
                    .put("qnnIr", qnnEvidence(qnnEvidenceDir)),
            )
            if (phase == ProbePhase.PREPARE) {
                return JSONObject()
                    .put("prepareWallMs", prepared.wallMs)
                    .put("prepareCpuMs", prepared.cpuMs)
                    .put("prepareStages", prepareStages)
                    .put("availableAccelerators", JSONArray(availableAccelerators))
                    .put("tensorTypes", tensorTypes)
                    .put("qnnIr", qnnEvidence(qnnEvidenceDir))
                    .put("inference", JSONObject.NULL)
            }

            val inputEvidence = JSONObject()
            inputs.forEachIndexed { index, (bindingName, tensor) ->
                checkpoint(resultRoot, base, "write-input-$bindingName")
                val loadStarted = TimedStart()
                val values = fixtureOrDeterministic(
                    tensor = tensor,
                    bindingName = bindingName,
                    index = index,
                    fixtureRoot = fixtureRoot,
                    contract = contract,
                )
                val loadTiming = loadStarted.elapsed()
                val writeStarted = TimedStart()
                inputBuffers.getValue(bindingName).writeFloat(values)
                val writeTiming = writeStarted.elapsed()
                inputEvidence.put(
                    bindingName,
                    JSONObject()
                        .put("loadAndVerify", loadTiming.evidence())
                        .put("bufferWrite", writeTiming.evidence())
                        .put("bytes", tensor.byteSize),
                )
            }

            val warmupTimings = mutableListOf<TimedResult>()
            repeat(warmupRuns) { index ->
                checkpoint(resultRoot, base, "warmup-" + index + "-start")
                val timing = runCompiledInference(compiledModel, inputBuffers, outputBuffers, signature.key)
                warmupTimings += timing
                checkpoint(
                    resultRoot,
                    base,
                    "warmup-" + index + "-complete",
                    timing.evidence(),
                )
            }
            val measuredTimings = mutableListOf<TimedResult>()
            repeat(measuredRuns) { index ->
                checkpoint(resultRoot, base, "measured-" + index + "-start")
                val timing = runCompiledInference(compiledModel, inputBuffers, outputBuffers, signature.key)
                measuredTimings += timing
                checkpoint(
                    resultRoot,
                    base,
                    "measured-" + index + "-complete",
                    timing.evidence(),
                )
            }
            val outputEvidence = JSONObject()
            outputs.forEach { (bindingName, tensor) ->
                checkpoint(resultRoot, base, "read-output-$bindingName")
                val readStarted = TimedStart()
                val values = outputBuffers.getValue(bindingName).readFloat()
                val read = readStarted.elapsed()
                check(values.size.toLong() == tensor.elementCount)
                val outputFile = if (exportOutputs) {
                    File(resultRoot, "$bindingName.f32le")
                } else {
                    null
                }
                val summarizeStarted = TimedStart()
                val summary = summarize(values, tensor, outputFile)
                val summarizeTiming = summarizeStarted.elapsed()
                outputEvidence.put(
                    bindingName,
                    summary
                        .put("read", read.evidence())
                        .put("summarizeAndExport", summarizeTiming.evidence())
                        .put("exportedFileBytes", outputFile?.length() ?: 0L),
                )
            }
            JSONObject()
                .put("prepareWallMs", prepared.wallMs)
                .put("prepareCpuMs", prepared.cpuMs)
                .put("prepareStages", prepareStages)
                .put("availableAccelerators", JSONArray(availableAccelerators))
                .put("tensorTypes", tensorTypes)
                .put("qnnIr", qnnEvidence(qnnEvidenceDir))
                .put("inputs", inputEvidence)
                .put("inference", JSONObject()
                    .put("warmupRuns", timingEvidence(warmupTimings))
                    .put("measuredRuns", timingEvidence(measuredTimings))
                    .put("measuredSummary", timingSummary(measuredTimings)))
                .put("outputs", outputEvidence)
        } finally {
            inputBuffers.values.forEach(TensorBuffer::close)
            outputBuffers.values.forEach(TensorBuffer::close)
            model?.close()
            environment.close()
        }
    }

    private fun loadProbeContract(fileName: String, file: File): LoadedProbeContract {
        check(fileName == File(fileName).name && fileName.endsWith(".json")) {
            "Probe contract must be a JSON base name."
        }
        val identity = PROBE_CONTRACT_IDENTITIES[fileName]
            ?: error("Unknown LiteRT probe contract: $fileName")
        return when (identity.kind) {
            ProbeContractKind.GENERATED_SIDECAR -> {
                val loaded = GeneratedLiteRtCandidateContractLoader.load(
                    file,
                    identity.byteSize,
                    identity.sha256,
                )
                val contract = loaded.contract
                LoadedProbeContract(
                    view = ProbeContractView(
                        contractId = contract.contractId,
                        modelId = contract.modelId,
                        artifactFileName = contract.artifact.fileName,
                        artifactByteSize = contract.artifact.byteSize,
                        artifactSha256 = contract.artifact.sha256,
                        flatBuffer = contract.flatBuffer,
                        inputFixtures = contract.fixtures.inputs.map { fixture ->
                            ProbeInputFixture(
                                bindingName = fixture.bindingName,
                                fileName = fixture.fileName,
                                byteSize = fixture.byteSize,
                                sha256 = fixture.sha256,
                                shape = fixture.shape,
                            )
                        },
                    ),
                    sidecarSha256 = loaded.sidecarSha256,
                )
            }

            ProbeContractKind.EXTERNAL_SIDECAR -> {
                val loaded = ExternalLiteRtCandidateContractLoader.load(
                    file,
                    identity.byteSize,
                    identity.sha256,
                )
                val contract = loaded.contract
                LoadedProbeContract(
                    view = ProbeContractView(
                        contractId = contract.contractId,
                        modelId = contract.modelId,
                        artifactFileName = contract.artifact.fileName,
                        artifactByteSize = contract.artifact.byteSize,
                        artifactSha256 = contract.artifact.sha256,
                        flatBuffer = contract.flatBuffer,
                    ),
                    sidecarSha256 = loaded.sidecarSha256,
                    externalContract = contract,
                )
            }

            ProbeContractKind.GENERATED_HOST_MANIFEST -> {
                val loaded = GeneratedLiteRtHostCandidateManifestLoader.load(
                    file,
                    identity.byteSize,
                    identity.sha256,
                )
                val manifest = loaded.manifest
                LoadedProbeContract(
                    view = ProbeContractView(
                        contractId = manifest.candidateId,
                        modelId = manifest.modelId,
                        artifactFileName = manifest.artifact.fileName,
                        artifactByteSize = manifest.artifact.byteSize,
                        artifactSha256 = manifest.artifact.sha256,
                        flatBuffer = manifest.flatBuffer,
                        inputFixtures = manifest.inputFixtures.map { fixture ->
                            ProbeInputFixture(
                                bindingName = requireNotNull(fixture.bindingName),
                                fileName = fixture.fileName,
                                byteSize = fixture.byteSize,
                                sha256 = fixture.sha256,
                                shape = fixture.shape,
                            )
                        },
                        hostGate = ProbeHostGate(
                            minimumSignalToNoiseDb = manifest.qualityGate.minimumSignalToNoiseDb,
                            maximumAbsoluteError = manifest.qualityGate.maximumAbsoluteError,
                            lowSignalReferenceRms = manifest.qualityGate.lowSignalReferenceRms,
                            lowSignalMaximumAbsoluteError =
                                manifest.qualityGate.lowSignalMaximumAbsoluteError,
                            uniformTensorGatePassed =
                                manifest.qualityGate.uniformTensorGatePassed,
                            hostPipelineGatePassed =
                                manifest.qualityGate.hostPipelineGatePassed,
                            acceptedForDeviceTesting =
                                manifest.qualityGate.acceptedForDeviceTesting,
                        ),
                    ),
                    sidecarSha256 = loaded.manifestSha256,
                )
            }
        }
    }

    private fun loadProbeArtifact(
        benchmarkRoot: File,
        manifestFileName: String?,
        contract: LoadedProbeContract,
        contractFile: File,
        contractSha256: String,
    ): ProbeArtifact {
        val view = contract.view
        if (manifestFileName == null) {
            return ProbeArtifact(
                contractId = view.contractId,
                modelId = view.modelId,
                fileName = view.artifactFileName,
                byteSize = view.artifactByteSize,
                sha256 = view.artifactSha256,
            )
        }
        val externalContract = requireNotNull(contract.externalContract) {
            "Derived manifests are only supported for the external BandBuddy contract."
        }
        check(manifestFileName == DERIVED_MANIFEST_FILE_NAME) {
            "Unknown LiteRT derivation manifest: $manifestFileName"
        }
        val manifestRoot = File(benchmarkRoot, "contracts").canonicalFile
        val manifestFile = File(manifestRoot, manifestFileName).canonicalFile
        check(manifestFile.parentFile == manifestRoot && manifestFile.isFile) {
            "Missing derivation manifest: $manifestFile"
        }
        val loaded = DerivedLiteRtCandidateContractLoader.load(manifestFile, contractFile)
        check(loaded.base.sidecarSha256 == contractSha256)
        check(loaded.base.contract.contractId == externalContract.contractId)
        check(loaded.base.contract.modelId == externalContract.modelId)
        val derived = loaded.contract

        return ProbeArtifact(
            contractId = derived.contractId,
            modelId = derived.modelId,
            fileName = derived.artifact.fileName,
            byteSize = derived.artifact.byteSize,
            sha256 = derived.artifact.sha256,
            manifestFile = manifestFile,
            manifestSha256 = sha256(manifestFile),
        )
    }

    private fun baseEvidence(
        context: Context,
        candidate: ProbeCandidate,
        phase: ProbePhase,
        threads: Int,
        exportOutputs: Boolean,
        warmupRuns: Int,
        measuredRuns: Int,
        contract: ProbeContractView,
        contractFile: File,
        contractSha256: String,
        modelFile: File,
        artifact: ProbeArtifact,
    ): JSONObject = JSONObject()
        .put("schemaVersion", 1)
        .put("status", "running")
        .put("candidate", candidate.wireValue)
        .put("phase", phase.wireValue)
        .put("threads", threads)
        .put("exportOutputs", exportOutputs)
        .put("warmupRuns", warmupRuns)
        .put("measuredRuns", measuredRuns)
        .put("inputFixtureMode", if (contract.inputFixtures.isEmpty()) "generated-formula" else "contract-frozen-raw")
        .put("hostQualityGate", contract.hostGate?.evidence() ?: JSONObject.NULL)
        .put("candidateOptions", candidate.evidence())
        .put("contract", JSONObject()
            .put("id", artifact.contractId)
            .put("baseId", contract.contractId)
            .put("path", contractFile.absolutePath)
            .put("bytes", contractFile.length())
            .put("sha256", contractSha256))
        .put("derivationManifest", artifact.manifestFile?.let { manifestFile ->
            JSONObject()
                .put("path", manifestFile.absolutePath)
                .put("bytes", manifestFile.length())
                .put("sha256", artifact.manifestSha256)
        } ?: JSONObject.NULL)
        .put("model", JSONObject()
            .put("id", artifact.modelId)
            .put("path", modelFile.absolutePath)
            .put("bytes", modelFile.length())
            .put("sha256", artifact.sha256)
            .put("signature", contract.flatBuffer.signatures.single().key))
        .put("runtime", JSONObject()
            .put("api", candidate.engine.wireValue)
            .put("id", BuildConfig.BENCHMARK_RUNTIME_ID)
            .put("version", BuildConfig.BENCHMARK_RUNTIME_VERSION)
            .put("artifactSha256", BuildConfig.BENCHMARK_RUNTIME_ARTIFACT_SHA256)
            .put("acceleratorBundleSha256", BuildConfig.BENCHMARK_ACCELERATOR_BUNDLE_SHA256)
            .put("nativeLibraries", nativeLibraryEvidence(context)))
        .put("app", JSONObject()
            .put("applicationId", BuildConfig.APPLICATION_ID)
            .put("flavor", BuildConfig.FLAVOR)
            .put("buildType", BuildConfig.BUILD_TYPE)
            .put("sourceRevision", BuildConfig.BENCHMARK_SOURCE_REVISION)
            .put("sourceDirty", BuildConfig.BENCHMARK_SOURCE_DIRTY))
        .put("device", JSONObject()
            .put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL)
            .put("socManufacturer", Build.SOC_MANUFACTURER)
            .put("socModel", Build.SOC_MODEL)
            .put("sdk", Build.VERSION.SDK_INT)
            .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())))

    private fun signatureInputTensors(
        contract: ProbeContractView,
        signature: ExternalLiteRtSignature,
    ): List<Pair<String, ExternalLiteRtTensor>> = signature.inputs.map { binding ->
        binding.name to contract.flatBuffer.inputs.single { it.tensorIndex == binding.tensorIndex }
    }

    private fun signatureOutputTensors(
        contract: ProbeContractView,
        signature: ExternalLiteRtSignature,
    ): List<Pair<String, ExternalLiteRtTensor>> = signature.outputs.map { binding ->
        binding.name to contract.flatBuffer.outputs.single { it.tensorIndex == binding.tensorIndex }
    }

    private fun checkCompiledTensorType(type: TensorType, expected: ExternalLiteRtTensor) {
        check(type.elementType == TensorType.ElementType.FLOAT)
        check(requireNotNull(type.layout).dimensions == expected.shape) {
            "Unexpected tensor shape ${type.layout?.dimensions}; expected ${expected.shape}."
        }
    }

    private fun tensorTypeEvidence(type: TensorType): JSONObject = JSONObject()
        .put("elementType", type.elementType.name)
        .put("shape", JSONArray(requireNotNull(type.layout).dimensions))

    private fun tensorTypeEvidence(shape: List<Int>, elementType: String): JSONObject = JSONObject()
        .put("elementType", elementType)
        .put("shape", JSONArray(shape))

    private fun directBuffer(byteSize: Long): ByteBuffer {
        check(byteSize in 1..Int.MAX_VALUE.toLong())
        return ByteBuffer.allocateDirect(byteSize.toInt()).order(ByteOrder.nativeOrder())
    }

    private fun boundedCount(
        raw: String?,
        defaultValue: Int,
        minimum: Int,
        maximum: Int,
        argumentName: String,
    ): Int {
        val value = raw?.toIntOrNull() ?: defaultValue
        check(value in minimum..maximum) {
            "$argumentName must be in $minimum..$maximum."
        }
        return value
    }

    private fun fixtureOrDeterministic(
        tensor: ExternalLiteRtTensor,
        bindingName: String,
        index: Int,
        fixtureRoot: File,
        contract: ProbeContractView,
    ): FloatArray {
        val fixture = contract.inputFixtures.singleOrNull { it.bindingName == bindingName }
            ?: return deterministicInput(tensor.elementCount.toInt(), index)
        val file = verifiedFixtureFile(fixtureRoot, fixture, tensor)
        val source = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(source.remaining()).also(source::get)
    }

    private fun fillFixtureOrDeterministic(
        buffer: ByteBuffer,
        tensor: ExternalLiteRtTensor,
        bindingName: String,
        index: Int,
        fixtureRoot: File,
        contract: ProbeContractView,
    ) {
        val fixture = contract.inputFixtures.singleOrNull { it.bindingName == bindingName }
        if (fixture == null) {
            fillDeterministicInput(buffer.asFloatBuffer(), index)
            return
        }
        val file = verifiedFixtureFile(fixtureRoot, fixture, tensor)
        val bytes = file.readBytes()
        check(bytes.size == buffer.capacity())
        buffer.clear()
        buffer.put(bytes)
        buffer.rewind()
    }

    private fun verifiedFixtureFile(
        fixtureRoot: File,
        fixture: ProbeInputFixture,
        tensor: ExternalLiteRtTensor,
    ): File {
        check(fixture.shape == tensor.shape) {
            "Fixture ${fixture.bindingName} shape ${fixture.shape} does not match ${tensor.shape}."
        }
        check(fixture.byteSize == tensor.byteSize)
        val canonicalRoot = fixtureRoot.canonicalFile
        val file = File(canonicalRoot, fixture.fileName).canonicalFile
        check(file.parentFile == canonicalRoot && file.isFile && file.length() == fixture.byteSize) {
            "Missing verified input fixture: $file"
        }
        check(sha256(file) == fixture.sha256) { "Input fixture SHA-256 mismatch: $file" }
        return file
    }

    private fun deterministicInput(size: Int, inputIndex: Int): FloatArray = FloatArray(size).also {
        it.indices.forEach { index ->
            it[index] = deterministicValue(inputIndex, index)
        }
    }

    private fun fillDeterministicInput(buffer: FloatBuffer, inputIndex: Int) {
        repeat(buffer.capacity()) { index ->
            buffer.put(index, deterministicValue(inputIndex, index))
        }
        buffer.rewind()
    }

    private fun deterministicValue(inputIndex: Int, index: Int): Float = when (inputIndex) {
        0 -> (((index * 17) % 257) - 128) / 512f
        1 -> (((index * 29) % 251) - 125) / 512f
        else -> error("No deterministic fixture for input index $inputIndex.")
    }

    private fun summarize(
        values: FloatArray,
        tensor: ExternalLiteRtTensor,
        outputFile: File?,
    ): JSONObject {
        check(values.size.toLong() == tensor.elementCount)
        val accumulator = SummaryAccumulator(outputFile)
        values.forEach(accumulator::add)
        return accumulator.finish(tensor)
    }

    private fun summarize(
        values: FloatBuffer,
        tensor: ExternalLiteRtTensor,
        outputFile: File?,
    ): JSONObject {
        val source = values.duplicate().apply { clear() }
        check(source.remaining().toLong() == tensor.elementCount)
        val accumulator = SummaryAccumulator(outputFile)
        while (source.hasRemaining()) accumulator.add(source.get())
        return accumulator.finish(tensor)
    }

    private class SummaryAccumulator(outputFile: File?) {
        private val digest = MessageDigest.getInstance("SHA-256")
        private val bytes = ByteArray(64 * 1024)
        private val byteBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        private val output = outputFile?.also { it.parentFile?.mkdirs() }?.outputStream()?.buffered()
        private var elementCount = 0L
        private var nonFiniteCount = 0L
        private var absSum = 0.0
        private var squareSum = 0.0
        private var min = Double.POSITIVE_INFINITY
        private var max = Double.NEGATIVE_INFINITY

        fun add(value: Float) {
            if (byteBuffer.remaining() < Float.SIZE_BYTES) flush()
            byteBuffer.putFloat(value)
            elementCount++
            val number = value.toDouble()
            if (!number.isFinite()) {
                nonFiniteCount++
                return
            }
            absSum += abs(number)
            squareSum += number * number
            min = minOf(min, number)
            max = maxOf(max, number)
        }

        fun finish(tensor: ExternalLiteRtTensor): JSONObject {
            flush()
            output?.close()
            val finiteCount = elementCount - nonFiniteCount
            return JSONObject()
                .put("shape", JSONArray(tensor.shape))
                .put("elementCount", elementCount)
                .put("byteSize", elementCount * Float.SIZE_BYTES)
                .put("nonFiniteCount", nonFiniteCount)
                .put("absMean", if (finiteCount > 0) absSum / finiteCount else JSONObject.NULL)
                .put("rms", if (finiteCount > 0) sqrt(squareSum / finiteCount) else JSONObject.NULL)
                .put("min", if (finiteCount > 0) min else JSONObject.NULL)
                .put("max", if (finiteCount > 0) max else JSONObject.NULL)
                .put("sha256F32Le", digest.digest().joinToString("") { byte -> "%02x".format(byte) })
        }

        private fun flush() {
            val size = byteBuffer.position()
            if (size == 0) return
            digest.update(bytes, 0, size)
            output?.write(bytes, 0, size)
            byteBuffer.clear()
        }
    }

    private fun checkpoint(
        resultRoot: File,
        base: JSONObject,
        stage: String,
        details: JSONObject = JSONObject(),
    ) {
        val report = JSONObject(base.toString())
            .put("stage", stage)
            .put("stageElapsedRealtimeMs", SystemClock.elapsedRealtime())
            .put("details", details)
            .put("process", processSnapshot())
        publishJson(File(resultRoot, "progress.json"), report)
        Log.i(LOG_TAG, "candidate=${base.getString("candidate")} stage=$stage")
    }

    private fun processSnapshot(): JSONObject {
        val runtime = Runtime.getRuntime()
        val memoryInfo = Debug.MemoryInfo().also(Debug::getMemoryInfo)
        return JSONObject()
            .put("elapsedRealtimeMs", SystemClock.elapsedRealtime())
            .put("processCpuMs", android.os.Process.getElapsedCpuTime())
            .put("pssKb", memoryInfo.totalPss)
            .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
            .put("javaHeapUsedBytes", runtime.totalMemory() - runtime.freeMemory())
            .put("javaHeapCommittedBytes", runtime.totalMemory())
            .put("javaHeapMaxBytes", runtime.maxMemory())
    }

    private fun runCompiledInference(
        model: CompiledModel,
        inputBuffers: Map<String, TensorBuffer>,
        outputBuffers: Map<String, TensorBuffer>,
        signatureKey: String,
    ): TimedResult {
        val started = TimedStart()
        model.run(inputBuffers, outputBuffers, signatureKey)
        return started.elapsed().copy(process = processSnapshot())
    }

    private fun timingEvidence(timings: List<TimedResult>): JSONArray = JSONArray().also { array ->
        timings.forEachIndexed { index, timing ->
            array.put(timing.evidence().put("index", index))
        }
    }

    private fun timingSummary(timings: List<TimedResult>): JSONObject {
        check(timings.isNotEmpty())
        return JSONObject()
            .put("count", timings.size)
            .put("wallMs", numericSummary(timings.map(TimedResult::wallMs)))
            .put("cpuMs", numericSummary(timings.map { it.cpuMs.toDouble() }))
    }

    private fun numericSummary(values: List<Double>): JSONObject {
        val sorted = values.sorted()
        val midpoint = sorted.size / 2
        val median = if (sorted.size % 2 == 0) {
            (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
        } else {
            sorted[midpoint]
        }
        val p95Index = (ceil(sorted.size * 0.95).toInt() - 1).coerceIn(sorted.indices)
        return JSONObject()
            .put("min", sorted.first())
            .put("median", median)
            .put("p95NearestRank", sorted[p95Index])
            .put("max", sorted.last())
            .put("mean", values.average())
    }

    private fun nativeLibraryEvidence(context: Context): JSONArray {
        val root = File(context.applicationInfo.nativeLibraryDir)
        val names = listOf(
            "libLiteRt.so",
            "libLiteRtClGlAccelerator.so",
            "libLiteRtCompilerPlugin_Qualcomm.so",
            "libLiteRtDispatch_Qualcomm.so",
            "libQnnHtp.so",
            "libQnnHtpPrepare.so",
            "libQnnHtpV79Skel.so",
        )
        return JSONArray(names.mapNotNull { name ->
            File(root, name).takeIf(File::isFile)?.let { file ->
                JSONObject()
                    .put("name", name)
                    .put("bytes", file.length())
                    .put("sha256", sha256(file))
            }
        })
    }

    private fun qnnEvidence(root: File): JSONArray = JSONArray(
        root.walkTopDown()
            .filter { it.isFile }
            .map { file ->
                JSONObject()
                    .put("path", file.relativeTo(root).invariantSeparatorsPath)
                    .put("bytes", file.length())
                    .put("sha256", sha256(file))
            }
            .toList(),
    )

    private fun publishJson(file: File, value: JSONObject) {
        val partial = File(requireNotNull(file.parentFile), "${file.name}.partial")
        partial.writeText(value.toString(2), Charsets.UTF_8)
        Files.move(
            partial.toPath(),
            file.toPath(),
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING,
        )
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
        return digest.digest().toHex()
    }

    private fun ByteArray.toHex(): String = joinToString("") { byte -> "%02x".format(byte) }

    private data class TimedResult(
        val wallMs: Double,
        val cpuMs: Long,
        val process: JSONObject? = null,
    ) {
        fun evidence(): JSONObject = JSONObject()
            .put("wallMs", wallMs)
            .put("cpuMs", cpuMs)
            .put("process", process ?: JSONObject.NULL)
    }

    private data class ProbeArtifact(
        val contractId: String,
        val modelId: String,
        val fileName: String,
        val byteSize: Long,
        val sha256: String,
        val manifestFile: File? = null,
        val manifestSha256: String? = null,
    )

    private data class ProbeInputFixture(
        val bindingName: String,
        val fileName: String,
        val byteSize: Long,
        val sha256: String,
        val shape: List<Int>,
    )

    private data class ProbeHostGate(
        val minimumSignalToNoiseDb: Double,
        val maximumAbsoluteError: Double,
        val lowSignalReferenceRms: Double,
        val lowSignalMaximumAbsoluteError: Double,
        val uniformTensorGatePassed: Boolean,
        val hostPipelineGatePassed: Boolean,
        val acceptedForDeviceTesting: Boolean,
    ) {
        fun evidence(): JSONObject = JSONObject()
            .put("minimumSignalToNoiseDb", minimumSignalToNoiseDb)
            .put("maximumAbsoluteError", maximumAbsoluteError)
            .put("lowSignalReferenceRms", lowSignalReferenceRms)
            .put("lowSignalMaximumAbsoluteError", lowSignalMaximumAbsoluteError)
            .put("uniformTensorGatePassed", uniformTensorGatePassed)
            .put("hostPipelineGatePassed", hostPipelineGatePassed)
            .put("acceptedForDeviceTesting", acceptedForDeviceTesting)
    }

    private data class ProbeContractView(
        val contractId: String,
        val modelId: String,
        val artifactFileName: String,
        val artifactByteSize: Long,
        val artifactSha256: String,
        val flatBuffer: ExternalLiteRtFlatBuffer,
        val inputFixtures: List<ProbeInputFixture> = emptyList(),
        val hostGate: ProbeHostGate? = null,
    )

    private data class LoadedProbeContract(
        val view: ProbeContractView,
        val sidecarSha256: String,
        val externalContract: ExternalLiteRtCandidateContract? = null,
    )

    private enum class ProbeContractKind {
        EXTERNAL_SIDECAR,
        GENERATED_SIDECAR,
        GENERATED_HOST_MANIFEST,
    }

    private data class ProbeContractIdentity(
        val kind: ProbeContractKind,
        val byteSize: Long,
        val sha256: String,
    )

    private class TimedStart {
        private val wallNanos = System.nanoTime()
        private val cpuMs = android.os.Process.getElapsedCpuTime()

        fun elapsed(): TimedResult = TimedResult(
            wallMs = (System.nanoTime() - wallNanos) / 1_000_000.0,
            cpuMs = android.os.Process.getElapsedCpuTime() - cpuMs,
        )
    }

    private enum class ProbeEngine(val wireValue: String) {
        INTERPRETER("org.tensorflow.lite.Interpreter"),
        COMPILED_MODEL("com.google.ai.edge.litert.CompiledModel"),
    }

    private enum class ProbePhase(val wireValue: String) {
        PREPARE("prepare"),
        RUN("run");

        companion object {
            fun fromWireValue(value: String): ProbePhase = entries.firstOrNull {
                it.wireValue == value
            } ?: error("Unknown probe phase: $value")
        }
    }

    private data class GpuProfile(
        val precision: CompiledModel.GpuOptions.Precision,
        val backend: CompiledModel.GpuOptions.Backend,
        val bufferStorageType: CompiledModel.GpuOptions.BufferStorageType? = null,
        val preferTextureWeights: Boolean? = null,
        val constantTensorSharing: Boolean? = null,
        val commandPreparationSteps: Int? = null,
    )

    private enum class ProbeCandidate(
        val wireValue: String,
        val engine: ProbeEngine,
        val accelerators: List<Accelerator> = emptyList(),
        val gpuProfile: GpuProfile? = null,
        val qnnOptimization: CompiledModel.QualcommOptions.OptimizationLevel? = null,
        val qnnUseConvHmx: Boolean = true,
        val qnnNumHvxThreads: Int? = null,
    ) {
        INTERPRETER_CPU_XNNPACK(
            "interpreter-cpu-xnnpack",
            ProbeEngine.INTERPRETER,
        ),
        COMPILED_CPU_XNNPACK(
            "compiled-cpu-xnnpack",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.CPU),
        ),
        COMPILED_GPU_AUTO_FP16(
            "compiled-gpu-auto-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.AUTOMATIC,
            ),
        ),
        COMPILED_GPU_AUTO_FP32(
            "compiled-gpu-auto-fp32",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.AUTOMATIC,
            ),
        ),
        COMPILED_GPU_OPENCL_FP16(
            "compiled-gpu-opencl-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.OPENCL,
            ),
        ),
        COMPILED_GPU_OPENCL_FP32(
            "compiled-gpu-opencl-fp32",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.OPENCL,
            ),
        ),
        COMPILED_GPU_OPENCL_BUFFER_FP16(
            "compiled-gpu-opencl-buffer-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.OPENCL,
                CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                preferTextureWeights = false,
                constantTensorSharing = true,
                commandPreparationSteps = 1,
            ),
        ),
        COMPILED_GPU_OPENCL_BUFFER_FP32(
            "compiled-gpu-opencl-buffer-fp32",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.OPENCL,
                CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                preferTextureWeights = false,
                constantTensorSharing = true,
                commandPreparationSteps = 1,
            ),
        ),
        COMPILED_GPU_OPENGL_FP16(
            "compiled-gpu-opengl-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.OPENGL,
            ),
        ),
        COMPILED_GPU_WEBGPU_FP16(
            "compiled-gpu-webgpu-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.WEBGPU,
            ),
        ),
        COMPILED_GPU_CPU_OPENCL_BUFFER_FP16(
            "compiled-gpu-cpu-opencl-buffer-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.OPENCL,
                CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                preferTextureWeights = false,
                constantTensorSharing = true,
                commandPreparationSteps = 1,
            ),
        ),
        COMPILED_GPU_CPU_AUTO_FP16(
            "compiled-gpu-cpu-auto-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.AUTOMATIC,
            ),
        ),
        COMPILED_GPU_CPU_AUTO_FP32(
            "compiled-gpu-cpu-auto-fp32",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.AUTOMATIC,
            ),
        ),
        COMPILED_GPU_CPU_OPENCL_FP16(
            "compiled-gpu-cpu-opencl-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.OPENCL,
            ),
        ),
        COMPILED_GPU_CPU_OPENCL_FP32(
            "compiled-gpu-cpu-opencl-fp32",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.OPENCL,
            ),
        ),
        COMPILED_GPU_CPU_OPENGL_FP16(
            "compiled-gpu-cpu-opengl-fp16",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP16,
                CompiledModel.GpuOptions.Backend.OPENGL,
            ),
        ),
        COMPILED_GPU_CPU_OPENGL_FP32(
            "compiled-gpu-cpu-opengl-fp32",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.OPENGL,
            ),
        ),
        COMPILED_GPU_CPU_OPENCL_BUFFER_FP32(
            "compiled-gpu-cpu-opencl-buffer-fp32",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.OPENCL,
                CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                preferTextureWeights = false,
                constantTensorSharing = true,
                commandPreparationSteps = 1,
            ),
        ),
        COMPILED_NPU_QNN_PREPARE(
            "compiled-npu-qnn-prepare-opt",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU),
            qnnOptimization = CompiledModel.QualcommOptions.OptimizationLevel
                .HTP_OPTIMIZE_FOR_PREPARE,
        ),
        COMPILED_NPU_QNN_INFERENCE(
            "compiled-npu-qnn-inference-opt",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU),
            qnnOptimization = CompiledModel.QualcommOptions.OptimizationLevel
                .HTP_OPTIMIZE_FOR_INFERENCE,
        ),
        COMPILED_NPU_QNN_INFERENCE_O3(
            "compiled-npu-qnn-inference-o3",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU),
            qnnOptimization = CompiledModel.QualcommOptions.OptimizationLevel
                .HTP_OPTIMIZE_FOR_INFERENCE_O3,
        ),
        COMPILED_NPU_QNN_PREPARE_NO_HMX(
            "compiled-npu-qnn-prepare-opt-no-hmx",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU),
            qnnOptimization = CompiledModel.QualcommOptions.OptimizationLevel
                .HTP_OPTIMIZE_FOR_PREPARE,
            qnnUseConvHmx = false,
        ),
        COMPILED_NPU_CPU_QNN_PREPARE_NO_HMX(
            "compiled-npu-cpu-qnn-prepare-opt-no-hmx",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU, Accelerator.CPU),
            qnnOptimization = CompiledModel.QualcommOptions.OptimizationLevel
                .HTP_OPTIMIZE_FOR_PREPARE,
            qnnUseConvHmx = false,
        ),
        COMPILED_NPU_CPU_QNN_PREPARE_NO_HMX_HVX1(
            "compiled-npu-cpu-qnn-prepare-opt-no-hmx-hvx1",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU, Accelerator.CPU),
            qnnOptimization = CompiledModel.QualcommOptions.OptimizationLevel
                .HTP_OPTIMIZE_FOR_PREPARE,
            qnnUseConvHmx = false,
            qnnNumHvxThreads = 1,
        ),
        COMPILED_NPU_CPU_QNN_PREPARE(
            "compiled-npu-cpu-qnn-prepare-opt",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU, Accelerator.CPU),
            qnnOptimization = CompiledModel.QualcommOptions.OptimizationLevel
                .HTP_OPTIMIZE_FOR_PREPARE,
        ),
        COMPILED_NPU_GPU_CPU_PREPARE(
            "compiled-npu-gpu-cpu-prepare-opt",
            ProbeEngine.COMPILED_MODEL,
            listOf(Accelerator.NPU, Accelerator.GPU, Accelerator.CPU),
            GpuProfile(
                CompiledModel.GpuOptions.Precision.FP32,
                CompiledModel.GpuOptions.Backend.OPENCL,
                CompiledModel.GpuOptions.BufferStorageType.BUFFER,
                preferTextureWeights = false,
                constantTensorSharing = true,
                commandPreparationSteps = 1,
            ),
            CompiledModel.QualcommOptions.OptimizationLevel.HTP_OPTIMIZE_FOR_PREPARE,
        );

        fun evidence(): JSONObject = JSONObject()
            .put("engine", engine.wireValue)
            .put("accelerators", JSONArray(accelerators.map(Accelerator::name)))
            .put("gpu", gpuProfile?.let { profile ->
                JSONObject()
                    .put("precision", profile.precision.name)
                    .put("backend", profile.backend.name)
                    .put("bufferStorageType", profile.bufferStorageType?.name)
                    .put("preferTextureWeights", profile.preferTextureWeights)
                    .put("constantTensorSharing", profile.constantTensorSharing)
                    .put("priority", "HIGH")
                    .put("numStepsOfCommandBufferPreparations", profile.commandPreparationSteps)
            } ?: JSONObject.NULL)
            .put("qnnOptimization", qnnOptimization?.name ?: JSONObject.NULL)
            .put(
                "qnnUseConvHmx",
                if (qnnOptimization == null) JSONObject.NULL else qnnUseConvHmx,
            )
            .put("qnnNumHvxThreads", qnnNumHvxThreads ?: JSONObject.NULL)

        companion object {
            fun fromWireValue(value: String): ProbeCandidate = entries.firstOrNull {
                it.wireValue == value
            } ?: error("Unknown LiteRT 2.1.5 probe candidate: $value")
        }
    }

    private companion object {
        const val ARG_CANDIDATE = "probeCandidate"
        const val ARG_CONTRACT = "probeContract"
        const val ARG_PHASE = "probePhase"
        const val ARG_THREADS = "probeThreads"
        const val ARG_EXPORT_OUTPUTS = "probeExportOutputs"
        const val ARG_WARMUP_RUNS = "probeWarmupRuns"
        const val ARG_MEASURED_RUNS = "probeMeasuredRuns"
        const val ARG_DERIVATION_MANIFEST = "probeDerivationManifest"
        const val LOG_TAG = "MSS-LiteRt215-Probe"
        const val CONTRACT_FILE_NAME = "bandbuddy_htdemucs_6s_core_v1_0_0.json"
        const val GENERATED_CONTRACT_FILE_NAME =
            "htdemucs_6s_core_smoke_2s_fp32_v1_0_0.json"
        const val CANONICAL_HOST_MANIFEST_FILE_NAME =
            "htdemucs_6s_core_canonical_7p8s_fp32_v1_0_0.host.json"
        const val FOUR_STEM_CANONICAL_HOST_MANIFEST_FILE_NAME =
            "htdemucs_4s_core_canonical_7p8s_fp32_v1_0_0.host.json"
        const val DERIVED_MANIFEST_FILE_NAME =
            "bandbuddy_htdemucs_6s_core_gather_reshape_v1_0_0.json"
        val PROBE_CONTRACT_IDENTITIES = mapOf(
            CONTRACT_FILE_NAME to ProbeContractIdentity(
                ProbeContractKind.EXTERNAL_SIDECAR,
                5_022,
                "ce0dfb4481eeb97a5d8dc32554ce90bcd6d2786bfbd63766909f836794288e09",
            ),
            GENERATED_CONTRACT_FILE_NAME to ProbeContractIdentity(
                ProbeContractKind.GENERATED_SIDECAR,
                15_303,
                "a14474d308734c84003cfc83e2ccd746564022d34ad79d6066c87d50a3dd8d0c",
            ),
            CANONICAL_HOST_MANIFEST_FILE_NAME to ProbeContractIdentity(
                ProbeContractKind.GENERATED_HOST_MANIFEST,
                47_505,
                "e22708ecbb1e43f528a3f1ff2ab33a8062c42fc36ffed1134f837426865f33e2",
            ),
            FOUR_STEM_CANONICAL_HOST_MANIFEST_FILE_NAME to ProbeContractIdentity(
                ProbeContractKind.GENERATED_HOST_MANIFEST,
                38_207,
                "134642ea71cfbb8174cb8f27a6696f3620b03d6556afcfade1457a4245b8a879",
            ),
        )
    }
}
