package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class GeneratedLiteRtHostArtifact(
    val fileName: String,
    val localPath: String,
    val byteSize: Long,
    val sha256: String,
    val format: String,
)

data class GeneratedLiteRtHostFixture(
    val role: String,
    val bindingName: String?,
    val fileName: String,
    val localPath: String,
    val byteSize: Long,
    val sha256: String,
    val dtype: String,
    val shape: List<Int>,
)

data class GeneratedLiteRtHostQualityGate(
    val minimumSignalToNoiseDb: Double,
    val maximumAbsoluteError: Double,
    val lowSignalReferenceRms: Double,
    val lowSignalMaximumAbsoluteError: Double,
    val uniformTensorGatePassed: Boolean,
    val hostPipelineGatePassed: Boolean,
    val acceptedForDeviceTesting: Boolean,
)

data class GeneratedLiteRtHostGlobalNormalization(
    val reference: String,
    val standardDeviationCorrection: Int,
    val epsilon: Double,
)

data class GeneratedLiteRtHostStft(
    val nFft: Int,
    val hopLength: Int,
    val window: String,
    val normalized: Boolean,
    val center: Boolean,
    val padMode: String,
    val outerPadLeft: Int,
    val outerPadRight: Int,
    val dropNyquistBin: Boolean,
    val frameCropLeft: Int,
    val frameCropRight: Int,
)

data class GeneratedLiteRtHostIstft(
    val normalized: Boolean,
    val center: Boolean,
    val restoreZeroNyquistBin: Boolean,
    val framePadLeft: Int,
    val framePadRight: Int,
    val reconstructionLength: Int,
    val cropStart: Int,
    val cropEnd: Int,
)

data class GeneratedLiteRtHostOla(
    val sampleRate: Int,
    val windowSamples: Int,
    val strideSamples: Int,
    val overlapSamples: Int,
    val overlap: Double,
    val transitionPower: Double,
    val shifts: Int,
    val accumulationDtype: String,
    val windowOrder: String,
    val tailWeightRule: String,
    val trackSamples: Int,
)

data class GeneratedLiteRtHostTailWindowPlan(
    val offset: Int,
    val actualSamples: Int,
    val contextStart: Int,
    val contextEnd: Int,
    val sourceStart: Int,
    val sourceEnd: Int,
    val padLeft: Int,
    val padRight: Int,
    val cropLeft: Int,
    val cropRight: Int,
)

data class GeneratedLiteRtHostDspContract(
    val branchCombination: String,
    val globalNormalization: GeneratedLiteRtHostGlobalNormalization,
    val stft: GeneratedLiteRtHostStft,
    val istft: GeneratedLiteRtHostIstft,
    val ola: GeneratedLiteRtHostOla,
    val tailWindowPlans: List<GeneratedLiteRtHostTailWindowPlan>,
)

data class GeneratedLiteRtHostCandidateManifest(
    val candidateId: String,
    val modelId: String,
    val artifact: GeneratedLiteRtHostArtifact,
    val flatBuffer: ExternalLiteRtFlatBuffer,
    val fixtures: List<GeneratedLiteRtHostFixture>,
    val sampleRate: Int,
    val windowSamples: Int,
    val spectrumFrameCount: Int,
    val stemOrder: List<BenchmarkStemSemantic>,
    val hostDspContract: GeneratedLiteRtHostDspContract,
    val qualityGate: GeneratedLiteRtHostQualityGate,
) {
    val inputFixtures: List<GeneratedLiteRtHostFixture>
        get() = fixtures.filter { it.bindingName != null }
}

data class LoadedGeneratedLiteRtHostCandidateManifest(
    val manifest: GeneratedLiteRtHostCandidateManifest,
    val manifestFile: File,
    val manifestByteSize: Long,
    val manifestSha256: String,
)

object GeneratedLiteRtHostCandidateManifestLoader {
    fun load(
        manifestFile: File,
        expectedManifestByteSize: Long,
        expectedManifestSha256: String,
    ): LoadedGeneratedLiteRtHostCandidateManifest {
        expect(expectedManifestByteSize > 0L, "Expected host manifest byte size must be positive.")
        expect(SHA256.matches(expectedManifestSha256), "Expected host manifest SHA-256 is invalid.")
        if (!manifestFile.isFile) {
            invalid("Generated LiteRT host manifest is not readable: " + manifestFile.absolutePath)
        }
        val bytes = manifestFile.readBytes()
        expect(bytes.size.toLong() == expectedManifestByteSize, "Host manifest byte-size mismatch.")
        val actualSha256 = sha256(bytes)
        expect(
            MessageDigest.isEqual(
                expectedManifestSha256.toByteArray(StandardCharsets.US_ASCII),
                actualSha256.toByteArray(StandardCharsets.US_ASCII),
            ),
            "Host manifest SHA-256 mismatch.",
        )
        val json = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (error: Exception) {
            throw BenchmarkModelContractException("Host manifest is not valid UTF-8.", error)
        }
        return LoadedGeneratedLiteRtHostCandidateManifest(
            manifest = parse(json),
            manifestFile = manifestFile.absoluteFile,
            manifestByteSize = bytes.size.toLong(),
            manifestSha256 = actualSha256,
        )
    }

    fun parse(json: String): GeneratedLiteRtHostCandidateManifest {
        val root = ObjectReader(StrictJsonParser(json).parse().asObject("$"), "$")
        root.requireExactKeys(
            "artifact",
            "candidateId",
            "conversion",
            "fixtures",
            "flatBuffer",
            "hostDspContract",
            "hostValidation",
            "manifestKind",
            "manifestSchemaVersion",
            "modelId",
            "modelSemantics",
            "provenance",
            "scope",
            "status",
        )
        expect(root.int("manifestSchemaVersion") == 1, "$.manifestSchemaVersion must equal 1.")
        expect(
            root.string("manifestKind") == "generated-litert-host-candidate",
            "$.manifestKind mismatch.",
        )
        val modelId = root.nonEmptyString("modelId")
        expect(MODEL_ID.matches(modelId), "$.modelId is invalid.")
        val candidateId = root.nonEmptyString("candidateId")
        expect(candidateId == modelId + "@host-1", "$.candidateId does not match modelId.")
        expect(root.string("scope") == "host-only-not-a-device-result", "$.scope mismatch.")
        expect(root.string("status") == "host-pipeline-passed", "$.status mismatch.")
        root.obj("conversion")
        root.obj("provenance")

        val artifact = parseArtifact(root.obj("artifact"), modelId)
        val flatBuffer = parseFlatBuffer(root.obj("flatBuffer"))
        val fixtures = parseFixtures(root.array("fixtures"))
        val semantics = parseSemantics(root.obj("modelSemantics"))
        val hostDspContract = parseHostDspContract(root.obj("hostDspContract"))
        val qualityGate = parseHostValidation(root.obj("hostValidation"))
        val result = GeneratedLiteRtHostCandidateManifest(
            candidateId = candidateId,
            modelId = modelId,
            artifact = artifact,
            flatBuffer = flatBuffer,
            fixtures = fixtures,
            sampleRate = semantics.sampleRate,
            windowSamples = semantics.windowSamples,
            spectrumFrameCount = semantics.spectrumFrameCount,
            stemOrder = semantics.stemOrder,
            hostDspContract = hostDspContract,
            qualityGate = qualityGate,
        )
        validateCrossFields(result, semantics.featureOrder)
        return result
    }

    private data class Semantics(
        val sampleRate: Int,
        val windowSamples: Int,
        val spectrumFrameCount: Int,
        val featureOrder: List<String>,
        val stemOrder: List<BenchmarkStemSemantic>,
    )

    private fun parseArtifact(value: JsonObject, modelId: String): GeneratedLiteRtHostArtifact {
        val path = "$.artifact"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("byteSize", "fileName", "format", "localPath", "sha256")
        val fileName = baseName(reader.string("fileName"), "$path.fileName")
        val localPath = normalizedPath(reader.string("localPath"), "$path.localPath")
        expect(
            localPath == "models/demucs/generated/" + modelId + "/" + fileName,
            "$path.localPath mismatch.",
        )
        val format = reader.string("format")
        expect(format == "tflite-flatbuffer", "$path.format mismatch.")
        return GeneratedLiteRtHostArtifact(
            fileName = fileName,
            localPath = localPath,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = hash(reader.string("sha256"), "$path.sha256"),
            format = format,
        )
    }

    private fun parseFlatBuffer(value: JsonObject): ExternalLiteRtFlatBuffer {
        val path = "$.flatBuffer"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "bufferCount",
            "customOperatorCount",
            "description",
            "fileIdentifier",
            "inputs",
            "inspectionReport",
            "keepStablehloConstant",
            "minimumRuntimeVersion",
            "operatorCodeCount",
            "operatorCount",
            "operatorHistogram",
            "outputs",
            "schemaVersion",
            "signature",
            "subgraphCount",
            "tensorCount",
        )
        expect(reader.string("fileIdentifier") == "TFL3", "$path.fileIdentifier mismatch.")
        expect(reader.string("description") == "MLIR Converted.", "$path.description mismatch.")
        expect(reader.string("keepStablehloConstant") == "true", "$path.keepStablehloConstant mismatch.")
        expect(reader.string("minimumRuntimeVersion") == "2.16.0", "$path.minimumRuntimeVersion mismatch.")
        val operatorCount = reader.positiveInt("operatorCount")
        val tensorCount = reader.positiveInt("tensorCount")
        expect(reader.int("customOperatorCount") == 0, "$path.customOperatorCount must equal zero.")
        expect(reader.positiveInt("operatorCodeCount") > 0, "$path.operatorCodeCount must be positive.")
        expect(reader.positiveInt("bufferCount") >= tensorCount, "$path.bufferCount is inconsistent.")
        val histogram = reader.obj("operatorHistogram")
        val histogramTotal = histogram.values.entries.sumOf { entry ->
            entry.value.asPositiveInt("$path.operatorHistogram." + entry.key)
        }
        expect(histogramTotal == operatorCount, "$path.operatorHistogram total mismatch.")
        validatePinnedReport(reader.obj("inspectionReport"), "$path.inspectionReport")
        return ExternalLiteRtFlatBuffer(
            fileIdentifier = "TFL3",
            schemaVersion = reader.positiveInt("schemaVersion"),
            subgraphCount = reader.positiveInt("subgraphCount"),
            operatorCount = operatorCount,
            tensorCount = tensorCount,
            customOperatorCount = 0,
            inputs = parseTensors(reader.array("inputs"), "$path.inputs"),
            outputs = parseTensors(reader.array("outputs"), "$path.outputs"),
            signatures = listOf(parseSignature(reader.obj("signature"), "$path.signature")),
        )
    }

    private fun validatePinnedReport(value: JsonObject, path: String) {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("byteSize", "fileName", "format", "localPath", "sha256")
        reader.positiveLong("byteSize")
        baseName(reader.string("fileName"), "$path.fileName")
        normalizedPath(reader.string("localPath"), "$path.localPath")
        hash(reader.string("sha256"), "$path.sha256")
        expect(reader.string("format") == "json", "$path.format mismatch.")
    }

    private fun parseTensors(value: JsonArray, path: String): List<ExternalLiteRtTensor> =
        value.values.mapIndexed { index, item ->
            val itemPath = "$path[" + index + "]"
            val reader = ObjectReader(item.asObject(itemPath), itemPath)
            reader.requireExactKeys("axes", "dtype", "index", "name", "shape", "tensorIndex")
            val logicalIndex = reader.int("index")
            val tensorIndex = reader.int("tensorIndex")
            expect(logicalIndex == index, "$itemPath.index mismatch.")
            expect(tensorIndex >= 0, "$itemPath.tensorIndex must be non-negative.")
            val shape = positiveIntList(reader.array("shape"), "$itemPath.shape")
            val elementCount = multiplyShape(shape, "$itemPath.shape")
            ExternalLiteRtTensor(
                index = logicalIndex,
                tensorIndex = tensorIndex,
                name = reader.nonEmptyString("name"),
                dtype = reader.string("dtype").also {
                    expect(it == "float32", "$itemPath.dtype mismatch.")
                },
                shape = shape,
                axes = stringList(reader.array("axes"), "$itemPath.axes"),
                elementCount = elementCount,
                byteSize = Math.multiplyExact(elementCount, 4L),
            )
        }

    private fun parseSignature(value: JsonObject, path: String): ExternalLiteRtSignature {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("inputs", "key", "outputs", "subgraphIndex")
        return ExternalLiteRtSignature(
            key = reader.string("key"),
            subgraphIndex = reader.int("subgraphIndex"),
            inputs = parseSignatureTensors(reader.array("inputs"), "$path.inputs"),
            outputs = parseSignatureTensors(reader.array("outputs"), "$path.outputs"),
        )
    }

    private fun parseSignatureTensors(
        value: JsonArray,
        path: String,
    ): List<ExternalLiteRtSignatureTensor> = value.values.mapIndexed { index, item ->
        val itemPath = "$path[" + index + "]"
        val reader = ObjectReader(item.asObject(itemPath), itemPath)
        reader.requireExactKeys("name", "tensorIndex")
        ExternalLiteRtSignatureTensor(
            name = reader.nonEmptyString("name"),
            tensorIndex = reader.int("tensorIndex").also {
                expect(it >= 0, "$itemPath.tensorIndex must be non-negative.")
            },
        )
    }

    private fun parseFixtures(value: JsonArray): List<GeneratedLiteRtHostFixture> =
        value.values.mapIndexed { index, item ->
            val path = "$.fixtures[" + index + "]"
            val reader = ObjectReader(item.asObject(path), path)
            reader.requireExactKeys("byteSize", "dtype", "fileName", "localPath", "role", "sha256", "shape")
            val role = reader.nonEmptyString("role")
            GeneratedLiteRtHostFixture(
                role = role,
                bindingName = INPUT_BINDINGS[role],
                fileName = baseName(reader.string("fileName"), "$path.fileName"),
                localPath = normalizedPath(reader.string("localPath"), "$path.localPath"),
                byteSize = reader.positiveLong("byteSize"),
                sha256 = hash(reader.string("sha256"), "$path.sha256"),
                dtype = reader.string("dtype").also {
                    expect(it == "float32-le", "$path.dtype mismatch.")
                },
                shape = positiveIntList(reader.array("shape"), "$path.shape"),
            )
        }.also { fixtures ->
            expect(fixtures.map { it.role }.toSet().size == fixtures.size, "$.fixtures roles must be unique.")
            expect(fixtures.map { it.role }.toSet() == EXPECTED_FIXTURE_ROLES, "$.fixtures roles mismatch.")
        }

    private fun parseSemantics(value: JsonObject): Semantics {
        val path = "$.modelSemantics"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "channelCount",
            "featureOrder",
            "sampleRate",
            "spectrumFrameCount",
            "stemOrder",
            "windowSamples",
        )
        expect(reader.positiveInt("channelCount") == 2, "$path.channelCount must equal 2.")
        return Semantics(
            sampleRate = reader.positiveInt("sampleRate"),
            windowSamples = reader.positiveInt("windowSamples"),
            spectrumFrameCount = reader.positiveInt("spectrumFrameCount"),
            featureOrder = stringList(reader.array("featureOrder"), "$path.featureOrder"),
            stemOrder = reader.array("stemOrder").values.mapIndexed { index, item ->
                val itemPath = "$path.stemOrder[" + index + "]"
                BenchmarkStemSemantic.fromWireValue(item.asString(itemPath), itemPath)
            },
        )
    }

    private fun parseHostDspContract(value: JsonObject): GeneratedLiteRtHostDspContract {
        val path = "$.hostDspContract"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "branchCombination",
            "globalNormalization",
            "istft",
            "ola",
            "stft",
            "tailWindowPlans",
        )
        val normalizationPath = "$path.globalNormalization"
        val normalization = ObjectReader(reader.obj("globalNormalization"), normalizationPath)
        normalization.requireExactKeys("epsilon", "reference", "standardDeviationCorrection")
        val stftPath = "$path.stft"
        val stft = ObjectReader(reader.obj("stft"), stftPath)
        stft.requireExactKeys(
            "center",
            "dropNyquistBin",
            "frameCropLeft",
            "frameCropRight",
            "hopLength",
            "nFft",
            "normalized",
            "outerPadLeft",
            "outerPadRight",
            "padMode",
            "window",
        )
        val istftPath = "$path.istft"
        val istft = ObjectReader(reader.obj("istft"), istftPath)
        istft.requireExactKeys(
            "center",
            "cropEnd",
            "cropStart",
            "framePadLeft",
            "framePadRight",
            "normalized",
            "reconstructionLength",
            "restoreZeroNyquistBin",
        )
        val olaPath = "$path.ola"
        val ola = ObjectReader(reader.obj("ola"), olaPath)
        ola.requireExactKeys(
            "accumulationDtype",
            "overlap",
            "overlapSamples",
            "sampleRate",
            "shifts",
            "strideSamples",
            "tailWeightRule",
            "trackSamples",
            "transitionPower",
            "windowOrder",
            "windowSamples",
        )
        val tailWindowPlans = reader.array("tailWindowPlans").values.mapIndexed { index, item ->
            val itemPath = "$path.tailWindowPlans[$index]"
            val plan = ObjectReader(item.asObject(itemPath), itemPath)
            plan.requireExactKeys(
                "actualSamples",
                "contextEnd",
                "contextStart",
                "cropLeft",
                "cropRight",
                "offset",
                "padLeft",
                "padRight",
                "sourceEnd",
                "sourceStart",
            )
            GeneratedLiteRtHostTailWindowPlan(
                offset = plan.nonNegativeInt("offset", itemPath),
                actualSamples = plan.positiveInt("actualSamples"),
                contextStart = plan.int("contextStart"),
                contextEnd = plan.positiveInt("contextEnd"),
                sourceStart = plan.nonNegativeInt("sourceStart", itemPath),
                sourceEnd = plan.positiveInt("sourceEnd"),
                padLeft = plan.nonNegativeInt("padLeft", itemPath),
                padRight = plan.nonNegativeInt("padRight", itemPath),
                cropLeft = plan.nonNegativeInt("cropLeft", itemPath),
                cropRight = plan.nonNegativeInt("cropRight", itemPath),
            )
        }
        expect(tailWindowPlans.isNotEmpty(), "$path.tailWindowPlans must not be empty.")
        return GeneratedLiteRtHostDspContract(
            branchCombination = reader.string("branchCombination"),
            globalNormalization = GeneratedLiteRtHostGlobalNormalization(
                reference = normalization.string("reference"),
                standardDeviationCorrection = normalization.int("standardDeviationCorrection"),
                epsilon = normalization.positiveDouble("epsilon"),
            ),
            stft = GeneratedLiteRtHostStft(
                nFft = stft.positiveInt("nFft"),
                hopLength = stft.positiveInt("hopLength"),
                window = stft.string("window"),
                normalized = stft.boolean("normalized"),
                center = stft.boolean("center"),
                padMode = stft.string("padMode"),
                outerPadLeft = stft.nonNegativeInt("outerPadLeft", stftPath),
                outerPadRight = stft.nonNegativeInt("outerPadRight", stftPath),
                dropNyquistBin = stft.boolean("dropNyquistBin"),
                frameCropLeft = stft.nonNegativeInt("frameCropLeft", stftPath),
                frameCropRight = stft.nonNegativeInt("frameCropRight", stftPath),
            ),
            istft = GeneratedLiteRtHostIstft(
                normalized = istft.boolean("normalized"),
                center = istft.boolean("center"),
                restoreZeroNyquistBin = istft.boolean("restoreZeroNyquistBin"),
                framePadLeft = istft.nonNegativeInt("framePadLeft", istftPath),
                framePadRight = istft.nonNegativeInt("framePadRight", istftPath),
                reconstructionLength = istft.positiveInt("reconstructionLength"),
                cropStart = istft.nonNegativeInt("cropStart", istftPath),
                cropEnd = istft.positiveInt("cropEnd"),
            ),
            ola = GeneratedLiteRtHostOla(
                sampleRate = ola.positiveInt("sampleRate"),
                windowSamples = ola.positiveInt("windowSamples"),
                strideSamples = ola.positiveInt("strideSamples"),
                overlapSamples = ola.positiveInt("overlapSamples"),
                overlap = ola.positiveDouble("overlap"),
                transitionPower = ola.positiveDouble("transitionPower"),
                shifts = ola.int("shifts"),
                accumulationDtype = ola.string("accumulationDtype"),
                windowOrder = ola.string("windowOrder"),
                tailWeightRule = ola.string("tailWeightRule"),
                trackSamples = ola.positiveInt("trackSamples"),
            ),
            tailWindowPlans = tailWindowPlans,
        )
    }

    private fun ObjectReader.nonNegativeInt(name: String, parentPath: String): Int = int(name).also {
        expect(it >= 0, "$parentPath.$name must not be negative.")
    }

    private fun parseHostValidation(value: JsonObject): GeneratedLiteRtHostQualityGate {
        val path = "$.hostValidation"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "canonicalOla",
            "qualityGate",
            "singleWindowGateEvaluation",
            "singleWindowMetrics",
            "torchCoreReconstruction",
        )
        val gatePath = "$path.qualityGate"
        val gate = ObjectReader(reader.obj("qualityGate"), gatePath)
        gate.requireExactKeys(
            "acceptedForDeviceTesting",
            "hostPipelineGatePassed",
            "lowSignalLatentMaximumAbsoluteError",
            "lowSignalMaximumAbsoluteError",
            "lowSignalReferenceRms",
            "maximumAbsoluteError",
            "minimumSignalToNoiseDb",
            "uniformTensorGatePassed",
        )
        val result = GeneratedLiteRtHostQualityGate(
            minimumSignalToNoiseDb = gate.positiveDouble("minimumSignalToNoiseDb"),
            maximumAbsoluteError = gate.positiveDouble("maximumAbsoluteError"),
            lowSignalReferenceRms = gate.positiveDouble("lowSignalReferenceRms"),
            lowSignalMaximumAbsoluteError = gate.positiveDouble("lowSignalMaximumAbsoluteError"),
            uniformTensorGatePassed = gate.boolean("uniformTensorGatePassed"),
            hostPipelineGatePassed = gate.boolean("hostPipelineGatePassed"),
            acceptedForDeviceTesting = gate.boolean("acceptedForDeviceTesting"),
        )
        gate.positiveDouble("lowSignalLatentMaximumAbsoluteError")
        expect(!result.uniformTensorGatePassed, "$gatePath must preserve the raw latent warning.")
        expect(result.hostPipelineGatePassed, "$gatePath host pipeline gate must pass.")
        expect(result.acceptedForDeviceTesting, "$gatePath must accept device testing.")
        val olaPath = "$path.canonicalOla"
        val ola = ObjectReader(reader.obj("canonicalOla"), olaPath)
        ola.requireExactKeys(
            "liteRtVsTorchOla",
            "paddingVsOfficialTensorChunkBitwiseEqual",
            "qualityGatePolicy",
            "status",
            "torchCoreOlaVsOfficialApplyModel",
            "uniformTensorGatePassed",
        )
        expect(ola.string("status") == "passed", "$olaPath.status must equal passed.")
        expect(ola.boolean("paddingVsOfficialTensorChunkBitwiseEqual"), "$olaPath padding parity must pass.")
        expect(!ola.boolean("uniformTensorGatePassed"), "$olaPath must preserve the latent warning.")
        return result
    }

    private fun validateCrossFields(
        manifest: GeneratedLiteRtHostCandidateManifest,
        featureOrder: List<String>,
    ) {
        expect(manifest.sampleRate == 44_100, "$.modelSemantics.sampleRate mismatch.")
        expect(
            manifest.spectrumFrameCount == (manifest.windowSamples + 1023) / 1024,
            "$.modelSemantics.spectrumFrameCount mismatch.",
        )
        expect(featureOrder == listOf("L.real", "L.imag", "R.real", "R.imag"), "$.modelSemantics.featureOrder mismatch.")
        expect(
            manifest.stemOrder == FOUR_STEM_ORDER || manifest.stemOrder == SIX_STEM_ORDER,
            "$.modelSemantics.stemOrder must use an official HTDemucs order.",
        )
        validateHostDspCrossFields(manifest)
        val flatBuffer = manifest.flatBuffer
        expect(flatBuffer.schemaVersion == 3 && flatBuffer.subgraphCount == 1, "$.flatBuffer identity mismatch.")
        expect(flatBuffer.inputs.size == 2 && flatBuffer.outputs.size == 2, "$.flatBuffer ABI count mismatch.")
        val inputShapes = listOf(
            listOf(1, 2, manifest.windowSamples),
            listOf(1, 4, 2048, manifest.spectrumFrameCount),
        )
        val outputShapes = listOf(
            listOf(1, manifest.stemOrder.size, 4, 2048, manifest.spectrumFrameCount),
            listOf(1, manifest.stemOrder.size, 2, manifest.windowSamples),
        )
        expect(flatBuffer.inputs.map { it.shape } == inputShapes, "$.flatBuffer input shapes mismatch.")
        expect(flatBuffer.outputs.map { it.shape } == outputShapes, "$.flatBuffer output shapes mismatch.")
        expect(flatBuffer.inputs.map { it.axes } == INPUT_AXES, "$.flatBuffer input axes mismatch.")
        expect(flatBuffer.outputs.map { it.axes } == OUTPUT_AXES, "$.flatBuffer output axes mismatch.")
        val signature = flatBuffer.signatures.single()
        expect(signature.key == "serving_default" && signature.subgraphIndex == 0, "$.flatBuffer.signature identity mismatch.")
        expect(signature.inputs.map { it.name } == listOf("args_0", "args_1"), "$.flatBuffer signature inputs mismatch.")
        expect(signature.outputs.map { it.name } == listOf("output_0", "output_1"), "$.flatBuffer signature outputs mismatch.")
        expect(signature.inputs.map { it.tensorIndex } == flatBuffer.inputs.map { it.tensorIndex }, "$.flatBuffer input indices mismatch.")
        expect(signature.outputs.map { it.tensorIndex } == flatBuffer.outputs.map { it.tensorIndex }, "$.flatBuffer output indices mismatch.")
        val inputs = manifest.inputFixtures
        expect(inputs.mapNotNull { it.bindingName } == listOf("args_0", "args_1"), "$.fixtures input order mismatch.")
        inputs.zip(flatBuffer.inputs).forEachIndexed { index, pair ->
            val fixture = pair.first
            val tensor = pair.second
            expect(fixture.shape == tensor.shape, "$.fixtures input[" + index + "] shape mismatch.")
            expect(fixture.byteSize == tensor.byteSize, "$.fixtures input[" + index + "] byte size mismatch.")
            val expectedPath = "models/demucs/generated/" + manifest.modelId + "/fixtures/" + fixture.fileName
            expect(fixture.localPath == expectedPath, "$.fixtures input[" + index + "] path mismatch.")
        }
    }

    private fun validateHostDspCrossFields(manifest: GeneratedLiteRtHostCandidateManifest) {
        val path = "$.hostDspContract"
        val dsp = manifest.hostDspContract
        expect(
            dsp.branchCombination == "frequency-istft-plus-time-waveform",
            "$path.branchCombination mismatch.",
        )
        expect(
            dsp.globalNormalization == GeneratedLiteRtHostGlobalNormalization(
                reference = "mean-across-stereo-channels",
                standardDeviationCorrection = 1,
                epsilon = 1e-8,
            ),
            "$path.globalNormalization mismatch.",
        )
        expect(
            dsp.stft == GeneratedLiteRtHostStft(
                nFft = 4_096,
                hopLength = 1_024,
                window = "periodic-hann",
                normalized = true,
                center = true,
                padMode = "reflect",
                outerPadLeft = 1_536,
                outerPadRight = 1_620,
                dropNyquistBin = true,
                frameCropLeft = 2,
                frameCropRight = 2,
            ),
            "$path.stft mismatch.",
        )
        expect(
            dsp.istft == GeneratedLiteRtHostIstft(
                normalized = true,
                center = true,
                restoreZeroNyquistBin = true,
                framePadLeft = 2,
                framePadRight = 2,
                reconstructionLength = 347_136,
                cropStart = 1_536,
                cropEnd = 345_516,
            ),
            "$path.istft mismatch.",
        )
        val ola = dsp.ola
        expect(ola.sampleRate == manifest.sampleRate, "$path.ola.sampleRate mismatch.")
        expect(ola.windowSamples == manifest.windowSamples, "$path.ola.windowSamples mismatch.")
        expect(ola.strideSamples + ola.overlapSamples == ola.windowSamples, "$path.ola stride mismatch.")
        expect(ola.strideSamples == 257_985 && ola.overlapSamples == 85_995, "$path.ola sample counts mismatch.")
        expect(ola.overlap == 0.25 && ola.transitionPower == 1.0, "$path.ola weighting mismatch.")
        expect(ola.shifts == 0, "$path.ola.shifts must equal zero.")
        expect(ola.accumulationDtype == "float32", "$path.ola.accumulationDtype mismatch.")
        expect(ola.windowOrder == "ascending-offset", "$path.ola.windowOrder mismatch.")
        expect(ola.tailWeightRule == "triangle-prefix", "$path.ola.tailWeightRule mismatch.")
        val olaInput = manifest.fixtures.single { it.role == "olaMixInput" }
        expect(olaInput.shape == listOf(1, 2, ola.trackSamples), "$path.ola.trackSamples mismatch.")
        val expectedOffsets = buildList {
            var offset = 0
            while (offset < ola.trackSamples) {
                add(offset)
                offset += ola.strideSamples
            }
        }
        expect(
            dsp.tailWindowPlans.map { it.offset } == expectedOffsets,
            "$path.tailWindowPlans offsets mismatch.",
        )
        dsp.tailWindowPlans.forEachIndexed { index, plan ->
            val itemPath = "$path.tailWindowPlans[$index]"
            val actual = minOf(ola.windowSamples, ola.trackSamples - plan.offset)
            val delta = ola.windowSamples - actual
            val cropLeft = delta / 2
            val cropRight = delta - cropLeft
            val contextStart = plan.offset - cropLeft
            val contextEnd = contextStart + ola.windowSamples
            val sourceStart = maxOf(0, contextStart)
            val sourceEnd = minOf(ola.trackSamples, contextEnd)
            expect(plan.actualSamples == actual, "$itemPath.actualSamples mismatch.")
            expect(plan.contextStart == contextStart && plan.contextEnd == contextEnd, "$itemPath context mismatch.")
            expect(plan.sourceStart == sourceStart && plan.sourceEnd == sourceEnd, "$itemPath source mismatch.")
            expect(plan.padLeft == sourceStart - contextStart, "$itemPath.padLeft mismatch.")
            expect(plan.padRight == contextEnd - sourceEnd, "$itemPath.padRight mismatch.")
            expect(plan.cropLeft == cropLeft && plan.cropRight == cropRight, "$itemPath crop mismatch.")
        }
    }

    private fun positiveIntList(value: JsonArray, path: String): List<Int> =
        value.values.mapIndexed { index, item -> item.asPositiveInt(path + "[" + index + "]") }

    private fun stringList(value: JsonArray, path: String): List<String> =
        value.values.mapIndexed { index, item -> item.asString(path + "[" + index + "]") }

    private fun multiplyShape(shape: List<Int>, path: String): Long = try {
        shape.fold(1L) { product, dimension -> Math.multiplyExact(product, dimension.toLong()) }
    } catch (error: ArithmeticException) {
        throw BenchmarkModelContractException(path + " shape product overflows Long.", error)
    }

    private fun baseName(value: String, path: String): String = value.also {
        expect(it.isNotEmpty() && it == File(it).name, "$path must be a file base name.")
        expect(LOCAL_PATH.matches(it), "$path contains unsupported characters.")
    }

    private fun normalizedPath(value: String, path: String): String = value.also {
        expect(LOCAL_PATH.matches(it), "$path contains unsupported characters.")
        expect(!it.startsWith('/'), "$path must be relative.")
        expect(it.split('/').none { part -> part.isEmpty() || part == "." || part == ".." }, "$path must be normalized.")
    }

    private fun hash(value: String, path: String): String = value.also {
        expect(SHA256.matches(it), "$path must be a lowercase SHA-256.")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val MODEL_ID = Regex("^[a-z0-9_]+$")
    private val LOCAL_PATH = Regex("^[A-Za-z0-9._/-]+$")
    private val INPUT_BINDINGS = linkedMapOf(
        "waveformInput" to "args_0",
        "spectrumInput" to "args_1",
    )
    private val EXPECTED_FIXTURE_ROLES = setOf(
        "waveformInput",
        "spectrumInput",
        "frequencyGolden",
        "waveformGolden",
        "frequencyWaveformGolden",
        "combinedGolden",
        "olaMixInput",
        "olaCombinedGolden",
    )
    private val FOUR_STEM_ORDER = listOf(
        BenchmarkStemSemantic.DRUMS,
        BenchmarkStemSemantic.BASS,
        BenchmarkStemSemantic.OTHER,
        BenchmarkStemSemantic.VOCALS,
    )
    private val SIX_STEM_ORDER = listOf(
        BenchmarkStemSemantic.DRUMS,
        BenchmarkStemSemantic.BASS,
        BenchmarkStemSemantic.OTHER,
        BenchmarkStemSemantic.VOCALS,
        BenchmarkStemSemantic.GUITAR,
        BenchmarkStemSemantic.PIANO,
    )
    private val INPUT_AXES = listOf(
        listOf("batch", "channel", "sample"),
        listOf("batch", "feature", "frequency", "frame"),
    )
    private val OUTPUT_AXES = listOf(
        listOf("batch", "stem", "feature", "frequency", "frame"),
        listOf("batch", "stem", "channel", "sample"),
    )
}
