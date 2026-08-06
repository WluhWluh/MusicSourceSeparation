package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.math.BigDecimal
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class GeneratedLiteRtContractReference(
    val contractId: String,
    val sidecarPath: String,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
    val role: String,
)

data class GeneratedLiteRtPinnedFile(
    val fileName: String,
    val localPath: String,
    val byteSize: Long,
    val sha256: String,
    val format: String,
)

data class GeneratedLiteRtOfficialModel(
    val hubRepository: String,
    val hubRevision: String,
    val loaderRepository: String,
    val loaderRevision: String,
    val licenseSpdx: String,
    val weight: GeneratedLiteRtPinnedFile,
    val metadata: GeneratedLiteRtPinnedFile,
    val bagManifest: GeneratedLiteRtPinnedFile,
)

data class GeneratedLiteRtNeuralCoreReference(
    val repository: String,
    val revision: String,
    val licenseSpdx: String,
    val role: String,
    val boundarySource: GeneratedLiteRtPinnedFile,
    val referenceExporter: GeneratedLiteRtPinnedFile,
)

data class GeneratedLiteRtExportRecipe(
    val script: GeneratedLiteRtPinnedFile,
    val boundaryImplementation: String,
    val sourceWeightFormat: String,
    val strictExport: Boolean,
    val lightweightConversion: Boolean,
    val enableX64: Boolean,
    val deterministicPositionalEmbedding: Boolean,
)

data class GeneratedLiteRtProvenance(
    val artifactOrigin: String,
    val officialModel: GeneratedLiteRtOfficialModel,
    val neuralCoreReference: GeneratedLiteRtNeuralCoreReference,
    val exportRecipe: GeneratedLiteRtExportRecipe,
)

data class GeneratedLiteRtProfile(
    val name: String,
    val sampleRate: Int,
    val sampleCount: Int,
    val spectrumFrameCount: Int,
    val segmentNumerator: Int,
    val segmentDenominator: Int,
    val originalSegment: String,
    val useTrainSegment: Boolean,
)

data class GeneratedLiteRtVersions(
    val python: String,
    val torch: String,
    val numpy: String,
    val safetensors: String,
    val litertTorch: String,
    val aiEdgeLiteRt: String,
)

data class GeneratedLiteRtOnnxDiagnostic(
    val artifact: GeneratedLiteRtPinnedFile,
    val role: String,
    val opset: Int,
    val producerName: String,
    val producerVersion: String,
    val operatorCount: Int,
    val externalDataTensorCount: Int,
)

data class GeneratedLiteRtConversion(
    val profile: GeneratedLiteRtProfile,
    val versions: GeneratedLiteRtVersions,
    val onnxDiagnostic: GeneratedLiteRtOnnxDiagnostic,
    val exportReport: GeneratedLiteRtPinnedFile,
)

data class GeneratedLiteRtArtifact(
    val fileName: String,
    val localPath: String,
    val byteSize: Long,
    val sha256: String,
    val format: String,
)

data class GeneratedLiteRtFlatBufferInspection(
    val report: GeneratedLiteRtPinnedFile,
    val description: String,
    val operatorCodeCount: Int,
    val bufferCount: Int,
    val minimumRuntimeVersion: String,
    val keepStablehloConstant: Boolean,
    val operatorHistogram: Map<String, Int>,
)

data class GeneratedLiteRtFixtureTensor(
    val bindingName: String,
    val fileName: String,
    val localPath: String,
    val byteSize: Long,
    val sha256: String,
    val dtype: String,
    val shape: List<Int>,
)

data class GeneratedLiteRtCombinedFixture(
    val fileName: String,
    val localPath: String,
    val byteSize: Long,
    val sha256: String,
    val dtype: String,
    val shape: List<Int>,
)

data class GeneratedLiteRtFixtures(
    val seed: Int,
    val inputs: List<GeneratedLiteRtFixtureTensor>,
    val outputs: List<GeneratedLiteRtFixtureTensor>,
    val combinedGolden: GeneratedLiteRtCombinedFixture,
)

data class GeneratedLiteRtQualityGate(
    val minimumSignalToNoiseDb: Double,
    val maximumAbsoluteError: Double,
    val acceptedForDeviceTesting: Boolean,
)

data class GeneratedLiteRtMetrics(
    val finite: Boolean,
    val bitwiseEqual: Boolean,
    val maxAbsoluteError: Double,
    val rootMeanSquareError: Double,
    val signalRootMeanSquare: Double,
    val signalToNoiseDb: Double,
)

data class GeneratedLiteRtStemMetrics(
    val stem: BenchmarkStemSemantic,
    val metrics: GeneratedLiteRtMetrics,
)

data class GeneratedLiteRtParityMetrics(
    val frequencyOutput: GeneratedLiteRtMetrics,
    val waveformOutput: GeneratedLiteRtMetrics,
    val combinedOutput: GeneratedLiteRtMetrics,
    val perStemCombined: List<GeneratedLiteRtStemMetrics>,
)

data class GeneratedLiteRtHostValidation(
    val status: String,
    val qualityGate: GeneratedLiteRtQualityGate,
    val torchCoreReconstruction: GeneratedLiteRtMetrics,
    val liteRtVsTorch: GeneratedLiteRtParityMetrics,
)

data class GeneratedLiteRtRuntimeIdentity(
    val version: String,
    val coordinate: String,
    val resolvedArtifact: ExternalLiteRtAarIdentity,
)

data class GeneratedLiteRtTargetDevice(
    val manufacturer: String,
    val model: String,
    val soc: String,
    val androidApi: Int,
    val abi: String,
    val status: String,
)

data class GeneratedLiteRtDevicePlan(
    val id: String,
    val backend: String,
    val accelerator: String,
    val precision: String,
    val fallbackPolicy: String,
    val status: String,
)

data class GeneratedLiteRtRuntimeTarget(
    val liteRt: GeneratedLiteRtRuntimeIdentity,
    val targetDevice: GeneratedLiteRtTargetDevice,
    val qnnV79BundleManifest: GeneratedLiteRtPinnedFile,
    val plans: List<GeneratedLiteRtDevicePlan>,
)

data class GeneratedLiteRtCandidateContract(
    val contractSchemaVersion: Int,
    val contractKind: String,
    val contractId: String,
    val modelId: String,
    val displayName: String,
    val sourceCandidateContract: GeneratedLiteRtContractReference,
    val provenance: GeneratedLiteRtProvenance,
    val conversion: GeneratedLiteRtConversion,
    val artifact: GeneratedLiteRtArtifact,
    val flatBuffer: ExternalLiteRtFlatBuffer,
    val flatBufferInspection: GeneratedLiteRtFlatBufferInspection,
    val modelSemantics: ExternalLiteRtModelSemantics,
    val fixtures: GeneratedLiteRtFixtures,
    val hostValidation: GeneratedLiteRtHostValidation,
    val runtimeTarget: GeneratedLiteRtRuntimeTarget,
)

data class LoadedGeneratedLiteRtCandidateContract(
    val contract: GeneratedLiteRtCandidateContract,
    val sidecarFile: File,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
)

object GeneratedLiteRtCandidateContractLoader {
    fun load(sidecarFile: File): LoadedGeneratedLiteRtCandidateContract = loadInternal(sidecarFile)

    fun load(
        sidecarFile: File,
        expectedSidecarSha256: String,
    ): LoadedGeneratedLiteRtCandidateContract = loadAfterIdentityCheck(
        sidecarFile,
        null,
        expectedSidecarSha256,
    )

    fun load(
        sidecarFile: File,
        expectedSidecarByteSize: Long,
        expectedSidecarSha256: String,
    ): LoadedGeneratedLiteRtCandidateContract {
        expect(expectedSidecarByteSize > 0L, "Expected sidecar byte size must be positive.")
        return loadAfterIdentityCheck(sidecarFile, expectedSidecarByteSize, expectedSidecarSha256)
    }

    private fun loadAfterIdentityCheck(
        sidecarFile: File,
        expectedSidecarByteSize: Long?,
        expectedSidecarSha256: String,
    ): LoadedGeneratedLiteRtCandidateContract {
        expect(SHA256.matches(expectedSidecarSha256), "Expected sidecar SHA-256 must be lowercase hexadecimal.")
        return loadInternal(sidecarFile, expectedSidecarByteSize, expectedSidecarSha256)
    }

    private fun loadInternal(
        sidecarFile: File,
        expectedSidecarByteSize: Long? = null,
        expectedSidecarSha256: String? = null,
    ): LoadedGeneratedLiteRtCandidateContract {
        if (!sidecarFile.isFile) invalid("Generated LiteRT contract is not a readable file: ${sidecarFile.absolutePath}")
        val bytes = try {
            sidecarFile.readBytes()
        } catch (error: Exception) {
            throw BenchmarkModelContractException("Could not read generated LiteRT contract.", error)
        }
        if (expectedSidecarByteSize != null) {
            expect(bytes.size.toLong() == expectedSidecarByteSize, "Generated LiteRT contract byte-size mismatch.")
        }
        val json = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (error: Exception) {
            throw BenchmarkModelContractException("Generated LiteRT contract is not valid UTF-8.", error)
        }
        val actualSha256 = sha256(bytes)
        if (expectedSidecarSha256 != null) {
            expect(
                MessageDigest.isEqual(
                    expectedSidecarSha256.toByteArray(StandardCharsets.US_ASCII),
                    actualSha256.toByteArray(StandardCharsets.US_ASCII),
                ),
                "Generated LiteRT contract SHA-256 mismatch.",
            )
        }
        return LoadedGeneratedLiteRtCandidateContract(
            contract = parse(json),
            sidecarFile = sidecarFile.absoluteFile,
            sidecarByteSize = bytes.size.toLong(),
            sidecarSha256 = actualSha256,
        )
    }

    fun parse(json: String): GeneratedLiteRtCandidateContract {
        val root = ObjectReader(StrictJsonParser(json).parse().asObject("$"), "$")
        root.requireExactKeys(
            "contractSchemaVersion", "contractKind", "contractId", "modelId", "displayName",
            "sourceCandidateContract", "provenance", "conversion", "artifact", "flatBuffer",
            "flatBufferInspection", "modelSemantics", "fixtures", "hostValidation", "runtimeTarget",
        )
        val schemaVersion = root.int("contractSchemaVersion")
        expect(schemaVersion == SCHEMA_VERSION, "$.contractSchemaVersion must equal $SCHEMA_VERSION.")
        val kind = root.string("contractKind")
        expect(kind == CONTRACT_KIND, "$.contractKind must equal '$CONTRACT_KIND'.")
        val modelId = root.string("modelId")
        expect(modelId == MODEL_ID, "$.modelId must equal '$MODEL_ID'.")
        val contractId = root.string("contractId")
        expect(contractId == "$MODEL_ID@$SCHEMA_VERSION", "$.contractId does not match modelId and schema.")
        val contract = GeneratedLiteRtCandidateContract(
            contractSchemaVersion = schemaVersion,
            contractKind = kind,
            contractId = contractId,
            modelId = modelId,
            displayName = root.nonEmptyString("displayName"),
            sourceCandidateContract = parseSourceContract(root.obj("sourceCandidateContract")),
            provenance = parseProvenance(root.obj("provenance")),
            conversion = parseConversion(root.obj("conversion")),
            artifact = parseArtifact(root.obj("artifact")),
            flatBuffer = parseFlatBuffer(root.obj("flatBuffer")),
            flatBufferInspection = parseInspection(root.obj("flatBufferInspection")),
            modelSemantics = parseSemantics(root.obj("modelSemantics")),
            fixtures = parseFixtures(root.obj("fixtures")),
            hostValidation = parseHostValidation(root.obj("hostValidation")),
            runtimeTarget = parseRuntimeTarget(root.obj("runtimeTarget")),
        )
        validateCrossFieldConsistency(contract)
        return contract
    }

    private fun parseSourceContract(value: JsonObject): GeneratedLiteRtContractReference {
        val path = "$.sourceCandidateContract"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("contractId", "sidecarPath", "sidecarByteSize", "sidecarSha256", "role")
        return GeneratedLiteRtContractReference(
            contractId = reader.string("contractId"),
            sidecarPath = normalizedPath(reader.string("sidecarPath"), "$path.sidecarPath"),
            sidecarByteSize = reader.positiveLong("sidecarByteSize"),
            sidecarSha256 = hash(reader.string("sidecarSha256"), "$path.sidecarSha256"),
            role = reader.string("role"),
        )
    }

    private fun parseProvenance(value: JsonObject): GeneratedLiteRtProvenance {
        val path = "$.provenance"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("artifactOrigin", "officialModel", "neuralCoreReference", "exportRecipe")
        return GeneratedLiteRtProvenance(
            artifactOrigin = reader.string("artifactOrigin"),
            officialModel = parseOfficialModel(reader.obj("officialModel")),
            neuralCoreReference = parseNeuralCoreReference(reader.obj("neuralCoreReference")),
            exportRecipe = parseExportRecipe(reader.obj("exportRecipe")),
        )
    }

    private fun parseOfficialModel(value: JsonObject): GeneratedLiteRtOfficialModel {
        val path = "$.provenance.officialModel"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "hubRepository", "hubRevision", "loaderRepository", "loaderRevision", "licenseSpdx", "weight", "metadata", "bagManifest",
        )
        return GeneratedLiteRtOfficialModel(
            hubRepository = reader.string("hubRepository"),
            hubRevision = revision(reader.string("hubRevision"), "$path.hubRevision"),
            loaderRepository = httpsUri(reader.string("loaderRepository"), "$path.loaderRepository"),
            loaderRevision = revision(reader.string("loaderRevision"), "$path.loaderRevision"),
            licenseSpdx = reader.string("licenseSpdx"),
            weight = parsePinnedFile(reader.obj("weight"), "$path.weight", "safetensors"),
            metadata = parsePinnedFile(reader.obj("metadata"), "$path.metadata", "json"),
            bagManifest = parsePinnedFile(reader.obj("bagManifest"), "$path.bagManifest", "yaml"),
        )
    }

    private fun parseNeuralCoreReference(value: JsonObject): GeneratedLiteRtNeuralCoreReference {
        val path = "$.provenance.neuralCoreReference"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("repository", "revision", "licenseSpdx", "role", "boundarySource", "referenceExporter")
        return GeneratedLiteRtNeuralCoreReference(
            repository = httpsUri(reader.string("repository"), "$path.repository"),
            revision = revision(reader.string("revision"), "$path.revision"),
            licenseSpdx = reader.string("licenseSpdx"),
            role = reader.string("role"),
            boundarySource = parsePinnedFile(reader.obj("boundarySource"), "$path.boundarySource", "python-source"),
            referenceExporter = parsePinnedFile(reader.obj("referenceExporter"), "$path.referenceExporter", "python-source"),
        )
    }

    private fun parseExportRecipe(value: JsonObject): GeneratedLiteRtExportRecipe {
        val path = "$.provenance.exportRecipe"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "script", "boundaryImplementation", "sourceWeightFormat", "strictExport",
            "lightweightConversion", "enableX64", "deterministicPositionalEmbedding",
        )
        return GeneratedLiteRtExportRecipe(
            script = parsePinnedFile(reader.obj("script"), "$path.script", "python-source"),
            boundaryImplementation = reader.string("boundaryImplementation"),
            sourceWeightFormat = reader.string("sourceWeightFormat"),
            strictExport = reader.boolean("strictExport"),
            lightweightConversion = reader.boolean("lightweightConversion"),
            enableX64 = reader.boolean("enableX64"),
            deterministicPositionalEmbedding = reader.boolean("deterministicPositionalEmbedding"),
        )
    }

    private fun parseConversion(value: JsonObject): GeneratedLiteRtConversion {
        val path = "$.conversion"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("profile", "versions", "onnxDiagnostic", "exportReport")
        return GeneratedLiteRtConversion(
            profile = parseProfile(reader.obj("profile")),
            versions = parseVersions(reader.obj("versions")),
            onnxDiagnostic = parseOnnxDiagnostic(reader.obj("onnxDiagnostic")),
            exportReport = parsePinnedFile(reader.obj("exportReport"), "$path.exportReport", "json"),
        )
    }

    private fun parseProfile(value: JsonObject): GeneratedLiteRtProfile {
        val path = "$.conversion.profile"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "name", "sampleRate", "sampleCount", "spectrumFrameCount", "segmentNumerator",
            "segmentDenominator", "originalSegment", "useTrainSegment",
        )
        return GeneratedLiteRtProfile(
            name = reader.string("name"),
            sampleRate = reader.positiveInt("sampleRate"),
            sampleCount = reader.positiveInt("sampleCount"),
            spectrumFrameCount = reader.positiveInt("spectrumFrameCount"),
            segmentNumerator = reader.positiveInt("segmentNumerator"),
            segmentDenominator = reader.positiveInt("segmentDenominator"),
            originalSegment = reader.string("originalSegment"),
            useTrainSegment = reader.boolean("useTrainSegment"),
        )
    }

    private fun parseVersions(value: JsonObject): GeneratedLiteRtVersions {
        val path = "$.conversion.versions"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("python", "torch", "numpy", "safetensors", "litertTorch", "aiEdgeLiteRt")
        return GeneratedLiteRtVersions(
            python = reader.string("python"),
            torch = reader.string("torch"),
            numpy = reader.string("numpy"),
            safetensors = reader.string("safetensors"),
            litertTorch = reader.string("litertTorch"),
            aiEdgeLiteRt = reader.string("aiEdgeLiteRt"),
        )
    }

    private fun parseOnnxDiagnostic(value: JsonObject): GeneratedLiteRtOnnxDiagnostic {
        val path = "$.conversion.onnxDiagnostic"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "artifact", "role", "opset", "producerName", "producerVersion", "operatorCount", "externalDataTensorCount",
        )
        return GeneratedLiteRtOnnxDiagnostic(
            artifact = parsePinnedFile(reader.obj("artifact"), "$path.artifact", "onnx-model"),
            role = reader.string("role"),
            opset = reader.positiveInt("opset"),
            producerName = reader.string("producerName"),
            producerVersion = reader.string("producerVersion"),
            operatorCount = reader.positiveInt("operatorCount"),
            externalDataTensorCount = reader.int("externalDataTensorCount").also {
                expect(it >= 0, "$path.externalDataTensorCount must not be negative.")
            },
        )
    }

    private fun parseArtifact(value: JsonObject): GeneratedLiteRtArtifact {
        val file = parsePinnedFile(value, "$.artifact", "tflite-flatbuffer")
        return GeneratedLiteRtArtifact(file.fileName, file.localPath, file.byteSize, file.sha256, file.format)
    }

    private fun parsePinnedFile(
        value: JsonObject,
        path: String,
        expectedFormat: String,
    ): GeneratedLiteRtPinnedFile {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("fileName", "localPath", "byteSize", "sha256", "format")
        val fileName = reader.nonEmptyString("fileName")
        val localPath = normalizedPath(reader.string("localPath"), "$path.localPath")
        expect(localPath.substringAfterLast('/') == fileName, "$path.localPath must end with fileName.")
        val format = reader.string("format")
        expect(format == expectedFormat, "$path.format must equal '$expectedFormat'.")
        val suffix = FORMAT_SUFFIX[format] ?: invalid("$path.format is unsupported.")
        expect(fileName.endsWith(suffix), "$path.fileName must end in '$suffix'.")
        return GeneratedLiteRtPinnedFile(
            fileName, localPath, reader.positiveLong("byteSize"), hash(reader.string("sha256"), "$path.sha256"), format,
        )
    }

    private fun parseFlatBuffer(value: JsonObject): ExternalLiteRtFlatBuffer {
        val path = "$.flatBuffer"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "fileIdentifier", "schemaVersion", "subgraphCount", "operatorCount", "tensorCount",
            "customOperatorCount", "inputs", "outputs", "signatures",
        )
        val inputs = parseTensors(reader.array("inputs"), "$path.inputs")
        val outputs = parseTensors(reader.array("outputs"), "$path.outputs")
        val signatures = parseSignatures(reader.array("signatures"), "$path.signatures")
        return ExternalLiteRtFlatBuffer(
            fileIdentifier = reader.string("fileIdentifier"),
            schemaVersion = reader.positiveInt("schemaVersion"),
            subgraphCount = reader.positiveInt("subgraphCount"),
            operatorCount = reader.positiveInt("operatorCount"),
            tensorCount = reader.positiveInt("tensorCount"),
            customOperatorCount = reader.int("customOperatorCount").also {
                expect(it >= 0, "$path.customOperatorCount must not be negative.")
            },
            inputs = inputs,
            outputs = outputs,
            signatures = signatures,
        )
    }

    private fun parseTensors(value: JsonArray, path: String): List<ExternalLiteRtTensor> =
        value.values.mapIndexed { position, item ->
            val tensorPath = "$path[$position]"
            val reader = ObjectReader(item.asObject(tensorPath), tensorPath)
            reader.requireExactKeys("index", "tensorIndex", "name", "dtype", "shape", "axes")
            val shape = positiveIntList(reader.array("shape"), "$tensorPath.shape")
            val axes = stringList(reader.array("axes"), "$tensorPath.axes")
            expect(shape.size == axes.size, "$tensorPath shape and axes ranks must match.")
            expect(shape.firstOrNull() == 1 && axes.firstOrNull() == "batch", "$tensorPath must start with batch size 1.")
            expect(axes.toSet().size == axes.size && axes.all { it in SUPPORTED_AXES }, "$tensorPath axes are invalid.")
            val elements = multiplyShape(shape, tensorPath)
            val bytes = Math.multiplyExact(elements, FLOAT32_BYTES)
            expect(bytes <= Int.MAX_VALUE, "$tensorPath exceeds one Android tensor buffer.")
            ExternalLiteRtTensor(
                index = reader.int("index").also { expect(it >= 0, "$tensorPath.index must not be negative.") },
                tensorIndex = reader.int("tensorIndex").also { expect(it >= 0, "$tensorPath.tensorIndex must not be negative.") },
                name = reader.nonEmptyString("name"),
                dtype = reader.string("dtype").also { expect(it == FLOAT32, "$tensorPath.dtype must equal '$FLOAT32'.") },
                shape = shape,
                axes = axes,
                elementCount = elements,
                byteSize = bytes,
            )
        }.also { tensors ->
            expect(tensors.map(ExternalLiteRtTensor::index) == tensors.indices.toList(), "$path indices must be ordered.")
            expect(tensors.map(ExternalLiteRtTensor::tensorIndex).toSet().size == tensors.size, "$path tensor indices must be unique.")
            expect(tensors.map(ExternalLiteRtTensor::name).toSet().size == tensors.size, "$path names must be unique.")
        }

    private fun parseSignatures(value: JsonArray, path: String): List<ExternalLiteRtSignature> =
        value.values.mapIndexed { index, item ->
            val signaturePath = "$path[$index]"
            val reader = ObjectReader(item.asObject(signaturePath), signaturePath)
            reader.requireExactKeys("key", "subgraphIndex", "inputs", "outputs")
            ExternalLiteRtSignature(
                key = reader.nonEmptyString("key"),
                subgraphIndex = reader.int("subgraphIndex").also { expect(it >= 0, "$signaturePath.subgraphIndex is invalid.") },
                inputs = parseSignatureTensors(reader.array("inputs"), "$signaturePath.inputs"),
                outputs = parseSignatureTensors(reader.array("outputs"), "$signaturePath.outputs"),
            )
        }

    private fun parseSignatureTensors(value: JsonArray, path: String): List<ExternalLiteRtSignatureTensor> =
        value.values.mapIndexed { index, item ->
            val itemPath = "$path[$index]"
            val reader = ObjectReader(item.asObject(itemPath), itemPath)
            reader.requireExactKeys("name", "tensorIndex")
            ExternalLiteRtSignatureTensor(
                reader.nonEmptyString("name"),
                reader.int("tensorIndex").also { expect(it >= 0, "$itemPath.tensorIndex must not be negative.") },
            )
        }.also { tensors ->
            expect(tensors.map(ExternalLiteRtSignatureTensor::name).toSet().size == tensors.size, "$path names must be unique.")
        }

    private fun parseInspection(value: JsonObject): GeneratedLiteRtFlatBufferInspection {
        val path = "$.flatBufferInspection"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "report", "description", "operatorCodeCount", "bufferCount", "minimumRuntimeVersion",
            "keepStablehloConstant", "operatorHistogram",
        )
        val histogramValue = reader.obj("operatorHistogram")
        expect(
            histogramValue.values.keys == EXPECTED_OPERATOR_HISTOGRAM.keys,
            "$path.operatorHistogram must contain the exact inspected operator set.",
        )
        val histogram = histogramValue.values.mapValues { (name, item) ->
            item.asPositiveInt("$path.operatorHistogram.$name")
        }
        return GeneratedLiteRtFlatBufferInspection(
            report = parsePinnedFile(reader.obj("report"), "$path.report", "json"),
            description = reader.string("description"),
            operatorCodeCount = reader.positiveInt("operatorCodeCount"),
            bufferCount = reader.positiveInt("bufferCount"),
            minimumRuntimeVersion = reader.string("minimumRuntimeVersion"),
            keepStablehloConstant = reader.boolean("keepStablehloConstant"),
            operatorHistogram = histogram,
        )
    }

    private fun parseSemantics(value: JsonObject): ExternalLiteRtModelSemantics {
        val path = "$.modelSemantics"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("workloadKind", "sampleRate", "channelCount", "sampleCount", "featureOrder", "stemOrder")
        return ExternalLiteRtModelSemantics(
            workloadKind = reader.string("workloadKind"),
            sampleRate = reader.positiveInt("sampleRate"),
            channelCount = reader.positiveInt("channelCount"),
            sampleCount = reader.positiveInt("sampleCount"),
            featureOrder = stringList(reader.array("featureOrder"), "$path.featureOrder"),
            stemOrder = reader.array("stemOrder").values.mapIndexed { index, item ->
                BenchmarkStemSemantic.fromWireValue(item.asString("$path.stemOrder[$index]"), "$path.stemOrder[$index]")
            },
        )
    }

    private fun parseFixtures(value: JsonObject): GeneratedLiteRtFixtures {
        val path = "$.fixtures"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("seed", "inputs", "outputs", "combinedGolden")
        return GeneratedLiteRtFixtures(
            seed = reader.positiveInt("seed"),
            inputs = parseFixtureTensors(reader.array("inputs"), "$path.inputs"),
            outputs = parseFixtureTensors(reader.array("outputs"), "$path.outputs"),
            combinedGolden = parseCombinedFixture(reader.obj("combinedGolden"), "$path.combinedGolden"),
        )
    }

    private fun parseFixtureTensors(value: JsonArray, path: String): List<GeneratedLiteRtFixtureTensor> =
        value.values.mapIndexed { index, item ->
            val itemPath = "$path[$index]"
            val reader = ObjectReader(item.asObject(itemPath), itemPath)
            reader.requireExactKeys("bindingName", "fileName", "localPath", "byteSize", "sha256", "dtype", "shape")
            val fileName = reader.nonEmptyString("fileName")
            val localPath = normalizedPath(reader.string("localPath"), "$itemPath.localPath")
            expect(localPath.substringAfterLast('/') == fileName, "$itemPath.localPath must end with fileName.")
            GeneratedLiteRtFixtureTensor(
                bindingName = reader.nonEmptyString("bindingName"),
                fileName = fileName,
                localPath = localPath,
                byteSize = reader.positiveLong("byteSize"),
                sha256 = hash(reader.string("sha256"), "$itemPath.sha256"),
                dtype = reader.string("dtype").also { expect(it == FLOAT32_LE, "$itemPath.dtype must equal '$FLOAT32_LE'.") },
                shape = positiveIntList(reader.array("shape"), "$itemPath.shape"),
            )
        }.also { fixtures ->
            expect(fixtures.map(GeneratedLiteRtFixtureTensor::bindingName).toSet().size == fixtures.size, "$path bindings must be unique.")
        }

    private fun parseCombinedFixture(value: JsonObject, path: String): GeneratedLiteRtCombinedFixture {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("fileName", "localPath", "byteSize", "sha256", "dtype", "shape")
        val fileName = reader.nonEmptyString("fileName")
        val localPath = normalizedPath(reader.string("localPath"), "$path.localPath")
        expect(localPath.substringAfterLast('/') == fileName, "$path.localPath must end with fileName.")
        return GeneratedLiteRtCombinedFixture(
            fileName = fileName,
            localPath = localPath,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = hash(reader.string("sha256"), "$path.sha256"),
            dtype = reader.string("dtype").also { expect(it == FLOAT32_LE, "$path.dtype must equal '$FLOAT32_LE'.") },
            shape = positiveIntList(reader.array("shape"), "$path.shape"),
        )
    }

    private fun parseHostValidation(value: JsonObject): GeneratedLiteRtHostValidation {
        val path = "$.hostValidation"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("status", "qualityGate", "torchCoreReconstruction", "liteRtVsTorch")
        return GeneratedLiteRtHostValidation(
            status = reader.string("status"),
            qualityGate = parseQualityGate(reader.obj("qualityGate")),
            torchCoreReconstruction = parseMetrics(reader.obj("torchCoreReconstruction"), "$path.torchCoreReconstruction"),
            liteRtVsTorch = parseParityMetrics(reader.obj("liteRtVsTorch")),
        )
    }

    private fun parseQualityGate(value: JsonObject): GeneratedLiteRtQualityGate {
        val path = "$.hostValidation.qualityGate"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("minimumSignalToNoiseDb", "maximumAbsoluteError", "acceptedForDeviceTesting")
        return GeneratedLiteRtQualityGate(
            minimumSignalToNoiseDb = number(value, "minimumSignalToNoiseDb", path, allowZero = false),
            maximumAbsoluteError = number(value, "maximumAbsoluteError", path, allowZero = false),
            acceptedForDeviceTesting = reader.boolean("acceptedForDeviceTesting"),
        )
    }

    private fun parseMetrics(value: JsonObject, path: String): GeneratedLiteRtMetrics {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "finite", "bitwiseEqual", "maxAbsoluteError", "rootMeanSquareError", "signalRootMeanSquare", "signalToNoiseDb",
        )
        return GeneratedLiteRtMetrics(
            finite = reader.boolean("finite"),
            bitwiseEqual = reader.boolean("bitwiseEqual"),
            maxAbsoluteError = number(value, "maxAbsoluteError", path, allowZero = true),
            rootMeanSquareError = number(value, "rootMeanSquareError", path, allowZero = true),
            signalRootMeanSquare = number(value, "signalRootMeanSquare", path, allowZero = true),
            signalToNoiseDb = number(value, "signalToNoiseDb", path, allowZero = true),
        )
    }

    private fun parseParityMetrics(value: JsonObject): GeneratedLiteRtParityMetrics {
        val path = "$.hostValidation.liteRtVsTorch"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("frequencyOutput", "waveformOutput", "combinedOutput", "perStemCombined")
        val stems = reader.array("perStemCombined").values.mapIndexed { index, item ->
            val itemPath = "$path.perStemCombined[$index]"
            val itemReader = ObjectReader(item.asObject(itemPath), itemPath)
            itemReader.requireExactKeys("stem", "metrics")
            GeneratedLiteRtStemMetrics(
                stem = BenchmarkStemSemantic.fromWireValue(itemReader.string("stem"), "$itemPath.stem"),
                metrics = parseMetrics(itemReader.obj("metrics"), "$itemPath.metrics"),
            )
        }
        return GeneratedLiteRtParityMetrics(
            frequencyOutput = parseMetrics(reader.obj("frequencyOutput"), "$path.frequencyOutput"),
            waveformOutput = parseMetrics(reader.obj("waveformOutput"), "$path.waveformOutput"),
            combinedOutput = parseMetrics(reader.obj("combinedOutput"), "$path.combinedOutput"),
            perStemCombined = stems,
        )
    }

    private fun parseRuntimeTarget(value: JsonObject): GeneratedLiteRtRuntimeTarget {
        val path = "$.runtimeTarget"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("liteRt", "targetDevice", "qnnV79BundleManifest", "plans")
        return GeneratedLiteRtRuntimeTarget(
            liteRt = parseRuntimeIdentity(reader.obj("liteRt")),
            targetDevice = parseTargetDevice(reader.obj("targetDevice")),
            qnnV79BundleManifest = parsePinnedFile(reader.obj("qnnV79BundleManifest"), "$path.qnnV79BundleManifest", "json"),
            plans = parseDevicePlans(reader.array("plans")),
        )
    }

    private fun parseRuntimeIdentity(value: JsonObject): GeneratedLiteRtRuntimeIdentity {
        val path = "$.runtimeTarget.liteRt"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("version", "coordinate", "resolvedArtifact")
        val artifactPath = "$path.resolvedArtifact"
        val artifactReader = ObjectReader(reader.obj("resolvedArtifact"), artifactPath)
        artifactReader.requireExactKeys("coordinate", "fileName", "byteSize", "sha256")
        return GeneratedLiteRtRuntimeIdentity(
            version = reader.string("version"),
            coordinate = reader.string("coordinate"),
            resolvedArtifact = ExternalLiteRtAarIdentity(
                coordinate = artifactReader.string("coordinate"),
                fileName = artifactReader.nonEmptyString("fileName"),
                byteSize = artifactReader.positiveLong("byteSize"),
                sha256 = hash(artifactReader.string("sha256"), "$artifactPath.sha256"),
            ),
        )
    }

    private fun parseTargetDevice(value: JsonObject): GeneratedLiteRtTargetDevice {
        val path = "$.runtimeTarget.targetDevice"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("manufacturer", "model", "soc", "androidApi", "abi", "status")
        return GeneratedLiteRtTargetDevice(
            reader.nonEmptyString("manufacturer"), reader.nonEmptyString("model"), reader.nonEmptyString("soc"),
            reader.positiveInt("androidApi"), reader.nonEmptyString("abi"), reader.string("status"),
        )
    }

    private fun parseDevicePlans(value: JsonArray): List<GeneratedLiteRtDevicePlan> =
        value.values.mapIndexed { index, item ->
            val path = "$.runtimeTarget.plans[$index]"
            val reader = ObjectReader(item.asObject(path), path)
            reader.requireExactKeys("id", "backend", "accelerator", "precision", "fallbackPolicy", "status")
            GeneratedLiteRtDevicePlan(
                reader.nonEmptyString("id"), reader.string("backend"), reader.nonEmptyString("accelerator"),
                reader.string("precision"), reader.string("fallbackPolicy"), reader.string("status"),
            )
        }

    private fun validateCrossFieldConsistency(contract: GeneratedLiteRtCandidateContract) {
        validateProvenance(contract)
        val profile = contract.conversion.profile
        expect(
            profile == GeneratedLiteRtProfile("smoke_2s", 44_100, 88_200, 87, 2, 1, "39/5", true),
            "$.conversion.profile must equal the generated 2-second export profile.",
        )
        expect(
            profile.sampleRate.toLong() * profile.segmentNumerator / profile.segmentDenominator == profile.sampleCount.toLong(),
            "$.conversion.profile sample count must match its exact segment duration.",
        )
        expect(
            contract.conversion.versions == GeneratedLiteRtVersions("3.12.3", "2.11.0+cpu", "2.5.1", "0.8.0", "0.9.1", "2.1.5"),
            "$.conversion.versions must match export-report.json.",
        )
        val onnx = contract.conversion.onnxDiagnostic
        expect(onnx.role == ONNX_ROLE && onnx.opset == 17 && onnx.producerName == "pytorch", "$.conversion.onnxDiagnostic is invalid.")
        expect(onnx.producerVersion == "2.11.0" && onnx.operatorCount == 3_256 && onnx.externalDataTensorCount == 0, "$.conversion.onnxDiagnostic inspection mismatch.")
        expectPinnedIdentity(onnx.artifact, ONNX_PATH, 111_392_345, ONNX_SHA, "$.conversion.onnxDiagnostic.artifact")
        expectPinnedIdentity(contract.conversion.exportReport, EXPORT_REPORT_PATH, 13_025, EXPORT_REPORT_SHA, "$.conversion.exportReport")
        expect(contract.artifact.fileName == ARTIFACT_NAME && contract.artifact.localPath == ARTIFACT_PATH, "$.artifact path is not the generated candidate.")
        expect(contract.artifact.byteSize == 112_924_120L && contract.artifact.sha256 == ARTIFACT_SHA, "$.artifact identity mismatch.")
        validateFlatBuffer(contract)
        validateFixtures(contract)
        validateHostValidation(contract)
        validateRuntimeTarget(contract)
    }

    private fun validateProvenance(contract: GeneratedLiteRtCandidateContract) {
        val source = contract.sourceCandidateContract
        expect(source.contractId == "htdemucs_6s_waveform_7p8s_onnx@3", "$.sourceCandidateContract.contractId is invalid.")
        expect(source.sidecarPath == "benchmark-contracts/htdemucs_6s_waveform_7p8s_onnx.json", "$.sourceCandidateContract.sidecarPath is invalid.")
        expect(source.sidecarByteSize == 6_604L && source.sidecarSha256 == SOURCE_SIDECAR_SHA, "$.sourceCandidateContract identity mismatch.")
        expect(source.role == "upstream-model-identity-only", "$.sourceCandidateContract.role is invalid.")
        val provenance = contract.provenance
        expect(provenance.artifactOrigin == "project-generated-from-official-safetensors", "$.provenance.artifactOrigin is invalid.")
        val official = provenance.officialModel
        expect(official.hubRepository == "adefossez/HTDemucs-6s" && official.hubRevision == HUB_REVISION, "$.provenance.officialModel hub identity mismatch.")
        expect(official.loaderRepository == "https://github.com/adefossez/demucs" && official.loaderRevision == LOADER_REVISION, "$.provenance.officialModel loader identity mismatch.")
        expect(official.licenseSpdx == "MIT", "$.provenance.officialModel.licenseSpdx must equal MIT.")
        expectPinnedIdentity(official.weight, WEIGHT_PATH, 54_885_744, WEIGHT_SHA, "$.provenance.officialModel.weight")
        expectPinnedIdentity(official.metadata, METADATA_PATH, 10_398, METADATA_SHA, "$.provenance.officialModel.metadata")
        expectPinnedIdentity(official.bagManifest, BAG_MANIFEST_PATH, 21, BAG_MANIFEST_SHA, "$.provenance.officialModel.bagManifest")
        val reference = provenance.neuralCoreReference
        expect(reference.repository == "https://github.com/JBNU-CILAB/demucs-lite" && reference.revision == DEMUCS_LITE_REVISION, "$.provenance.neuralCoreReference identity mismatch.")
        expect(reference.licenseSpdx == "MIT" && reference.role == "boundary-reference-only", "$.provenance.neuralCoreReference role/license mismatch.")
        expectPinnedIdentity(reference.boundarySource, BOUNDARY_PATH, 29_284, BOUNDARY_SHA, "$.provenance.neuralCoreReference.boundarySource")
        expectPinnedIdentity(reference.referenceExporter, REFERENCE_EXPORTER_PATH, 5_879, REFERENCE_EXPORTER_SHA, "$.provenance.neuralCoreReference.referenceExporter")
        val recipe = provenance.exportRecipe
        expectPinnedIdentity(recipe.script, EXPORT_SCRIPT_PATH, 26_003, EXPORT_SCRIPT_SHA, "$.provenance.exportRecipe.script")
        expect(recipe.boundaryImplementation == "project-owned-neural-core-rewrite", "$.provenance.exportRecipe boundary is invalid.")
        expect(recipe.sourceWeightFormat == "canonical-safetensors", "$.provenance.exportRecipe source weight is invalid.")
        expect(recipe.strictExport && recipe.lightweightConversion && !recipe.enableX64 && recipe.deterministicPositionalEmbedding, "$.provenance.exportRecipe flags must match the executed conversion.")
    }

    private fun validateFlatBuffer(contract: GeneratedLiteRtCandidateContract) {
        val flatBuffer = contract.flatBuffer
        expect(flatBuffer.fileIdentifier == "TFL3" && flatBuffer.schemaVersion == 3, "$.flatBuffer is not a TFLite schema-3 FlatBuffer.")
        expect(flatBuffer.subgraphCount == 1 && flatBuffer.operatorCount == 3_504 && flatBuffer.tensorCount == 4_301, "$.flatBuffer static counts mismatch.")
        expect(flatBuffer.customOperatorCount == 0, "$.flatBuffer.customOperatorCount must equal zero.")
        expect(flatBuffer.inputs.size == 2 && flatBuffer.outputs.size == 2 && flatBuffer.signatures.size == 1, "$.flatBuffer must expose the fixed two-input/two-output ABI.")
        val expectedInputs = listOf(
            Triple("serving_default_args_0", 0, listOf(1, 2, 88_200)),
            Triple("serving_default_args_1", 1, listOf(1, 4, 2_048, 87)),
        )
        val expectedOutputs = listOf(
            Triple("serving_default_output_0_output", 4_295, listOf(1, 6, 4, 2_048, 87)),
            Triple("serving_default_output_1_output", 4_300, listOf(1, 6, 2, 88_200)),
        )
        flatBuffer.inputs.zip(expectedInputs).forEachIndexed { index, (actual, expected) ->
            expect(actual.index == index && actual.name == expected.first && actual.tensorIndex == expected.second && actual.shape == expected.third, "$.flatBuffer.inputs[$index] does not match the fixed ABI.")
        }
        flatBuffer.outputs.zip(expectedOutputs).forEachIndexed { index, (actual, expected) ->
            expect(actual.index == index && actual.name == expected.first && actual.tensorIndex == expected.second && actual.shape == expected.third, "$.flatBuffer.outputs[$index] does not match the fixed ABI.")
        }
        expect(flatBuffer.inputs.map(ExternalLiteRtTensor::axes) == EXPECTED_INPUT_AXES, "$.flatBuffer input axes mismatch.")
        expect(flatBuffer.outputs.map(ExternalLiteRtTensor::axes) == EXPECTED_OUTPUT_AXES, "$.flatBuffer output axes mismatch.")
        val signature = flatBuffer.signatures.single()
        expect(signature.key == "serving_default" && signature.subgraphIndex == 0, "$.flatBuffer signature identity mismatch.")
        expect(signature.inputs.map(ExternalLiteRtSignatureTensor::name) == listOf("args_0", "args_1"), "$.flatBuffer signature input bindings mismatch.")
        expect(signature.outputs.map(ExternalLiteRtSignatureTensor::name) == listOf("output_0", "output_1"), "$.flatBuffer signature output bindings mismatch.")
        expect(signature.inputs.map(ExternalLiteRtSignatureTensor::tensorIndex) == flatBuffer.inputs.map(ExternalLiteRtTensor::tensorIndex), "$.flatBuffer signature input indices mismatch.")
        expect(signature.outputs.map(ExternalLiteRtSignatureTensor::tensorIndex) == flatBuffer.outputs.map(ExternalLiteRtTensor::tensorIndex), "$.flatBuffer signature output indices mismatch.")
        val inspection = contract.flatBufferInspection
        expectPinnedIdentity(inspection.report, INSPECTION_PATH, 3_766, INSPECTION_SHA, "$.flatBufferInspection.report")
        expect(inspection.description == "MLIR Converted." && inspection.operatorCodeCount == 27 && inspection.bufferCount == 4_304, "$.flatBufferInspection static metadata mismatch.")
        expect(inspection.minimumRuntimeVersion == "2.16.0" && inspection.keepStablehloConstant, "$.flatBufferInspection metadata mismatch.")
        expect(inspection.operatorHistogram == EXPECTED_OPERATOR_HISTOGRAM, "$.flatBufferInspection.operatorHistogram mismatch.")
        expect(inspection.operatorHistogram.values.sum() == flatBuffer.operatorCount, "$.flatBufferInspection histogram must sum to operatorCount.")
        val semantics = contract.modelSemantics
        expect(semantics.workloadKind == "neural-core-segment" && semantics.sampleRate == 44_100 && semantics.channelCount == 2 && semantics.sampleCount == 88_200, "$.modelSemantics workload mismatch.")
        expect(semantics.featureOrder == listOf("L.real", "L.imag", "R.real", "R.imag"), "$.modelSemantics.featureOrder mismatch.")
        expect(semantics.stemOrder == STEM_ORDER, "$.modelSemantics.stemOrder mismatch.")
    }

    private fun validateFixtures(contract: GeneratedLiteRtCandidateContract) {
        val fixtures = contract.fixtures
        expect(fixtures.seed == 20_260_803, "$.fixtures.seed must match export-report.json.")
        val signature = contract.flatBuffer.signatures.single()
        validateFixtureBindings(fixtures.inputs, signature.inputs, contract.flatBuffer.inputs, "$.fixtures.inputs")
        validateFixtureBindings(fixtures.outputs, signature.outputs, contract.flatBuffer.outputs, "$.fixtures.outputs")
        fixtures.inputs.forEachIndexed { index, fixture ->
            expect(fixture.fileName == EXPECTED_INPUT_FIXTURES[index].first && fixture.sha256 == EXPECTED_INPUT_FIXTURES[index].second, "$.fixtures.inputs[$index] identity mismatch.")
        }
        fixtures.outputs.forEachIndexed { index, fixture ->
            expect(fixture.fileName == EXPECTED_OUTPUT_FIXTURES[index].first && fixture.sha256 == EXPECTED_OUTPUT_FIXTURES[index].second, "$.fixtures.outputs[$index] identity mismatch.")
        }
        val combined = fixtures.combinedGolden
        expect(combined.fileName == "combined_golden.f32le.raw" && combined.sha256 == COMBINED_SHA, "$.fixtures.combinedGolden identity mismatch.")
        expect(combined.localPath == "$GENERATED_ROOT/fixtures/${combined.fileName}", "$.fixtures.combinedGolden.localPath mismatch.")
        expect(combined.shape == listOf(1, 6, 2, 88_200), "$.fixtures.combinedGolden shape mismatch.")
        expect(combined.byteSize == Math.multiplyExact(multiplyShape(combined.shape, "$.fixtures.combinedGolden"), FLOAT32_BYTES), "$.fixtures.combinedGolden byteSize mismatch.")
    }

    private fun validateFixtureBindings(
        fixtures: List<GeneratedLiteRtFixtureTensor>,
        bindings: List<ExternalLiteRtSignatureTensor>,
        tensors: List<ExternalLiteRtTensor>,
        path: String,
    ) {
        expect(fixtures.size == bindings.size && tensors.size == bindings.size, "$path count must match the signature.")
        fixtures.indices.forEach { index ->
            val fixture = fixtures[index]
            expect(fixture.bindingName == bindings[index].name, "$path[$index].bindingName must match the signature.")
            expect(bindings[index].tensorIndex == tensors[index].tensorIndex, "$path[$index] signature tensor index mismatch.")
            expect(fixture.shape == tensors[index].shape, "$path[$index].shape must match the FlatBuffer ABI.")
            expect(fixture.byteSize == tensors[index].byteSize, "$path[$index].byteSize must match the FP32 tensor.")
            expect(
                fixture.localPath == "$GENERATED_ROOT/fixtures/${fixture.fileName}",
                "$path[$index].localPath must identify the frozen generated fixture.",
            )
        }
    }

    private fun validateHostValidation(contract: GeneratedLiteRtCandidateContract) {
        val validation = contract.hostValidation
        val gate = validation.qualityGate
        expect(validation.status == "passed", "$.hostValidation.status must equal 'passed'.")
        expect(gate.minimumSignalToNoiseDb == 80.0 && gate.maximumAbsoluteError == 0.001, "$.hostValidation.qualityGate thresholds mismatch.")
        expect(gate.acceptedForDeviceTesting, "$.hostValidation must be accepted before device testing.")
        val torch = validation.torchCoreReconstruction
        expect(torch.finite && torch.bitwiseEqual && torch.maxAbsoluteError == 0.0 && torch.rootMeanSquareError == 0.0, "$.hostValidation.torchCoreReconstruction must be finite and bitwise equal.")
        val parity = validation.liteRtVsTorch
        val gated = listOf(parity.frequencyOutput, parity.waveformOutput, parity.combinedOutput) + parity.perStemCombined.map(GeneratedLiteRtStemMetrics::metrics)
        gated.forEachIndexed { index, metrics ->
            expect(metrics.finite, "$.hostValidation LiteRT metric[$index] must be finite.")
            expect(metrics.signalToNoiseDb >= gate.minimumSignalToNoiseDb, "$.hostValidation LiteRT metric[$index] is below the SNR gate.")
            expect(metrics.maxAbsoluteError <= gate.maximumAbsoluteError, "$.hostValidation LiteRT metric[$index] exceeds the error gate.")
        }
        expect(parity.perStemCombined.map(GeneratedLiteRtStemMetrics::stem) == STEM_ORDER, "$.hostValidation.liteRtVsTorch.perStemCombined must follow stemOrder.")
    }

    private fun validateRuntimeTarget(contract: GeneratedLiteRtCandidateContract) {
        val target = contract.runtimeTarget
        val runtime = target.liteRt
        expect(runtime.version == "2.1.5" && runtime.coordinate == LITERT_COORDINATE, "$.runtimeTarget.liteRt must target LiteRT 2.1.5.")
        expect(runtime.resolvedArtifact.coordinate == LITERT_COORDINATE && runtime.resolvedArtifact.fileName == "litert-2.1.5.aar", "$.runtimeTarget.liteRt.resolvedArtifact coordinate mismatch.")
        expect(runtime.resolvedArtifact.byteSize == 10_058_192L && runtime.resolvedArtifact.sha256 == LITERT_AAR_SHA, "$.runtimeTarget.liteRt.resolvedArtifact identity mismatch.")
        val device = target.targetDevice
        expect(device.manufacturer == "Samsung" && device.model == "SM-S9310" && device.soc == "SM8750", "$.runtimeTarget.targetDevice identity mismatch.")
        expect(device.androidApi == 35 && device.abi == "arm64-v8a", "$.runtimeTarget.targetDevice platform mismatch.")
        expect(device.status == NOT_RUN, "$.runtimeTarget.targetDevice.status must equal '$NOT_RUN'.")
        expectPinnedIdentity(target.qnnV79BundleManifest, QNN_MANIFEST_PATH, 4_270, QNN_MANIFEST_SHA, "$.runtimeTarget.qnnV79BundleManifest")
        expect(target.plans == EXPECTED_DEVICE_PLANS, "$.runtimeTarget.plans must equal the frozen S25 run plan.")
        expect(target.plans.all { it.status == NOT_RUN }, "$.runtimeTarget.plans device status must remain '$NOT_RUN'.")
    }

    private fun expectPinnedIdentity(
        file: GeneratedLiteRtPinnedFile,
        expectedPath: String,
        expectedBytes: Long,
        expectedSha256: String,
        path: String,
    ) {
        expect(file.localPath == expectedPath, "$path.localPath mismatch.")
        expect(file.byteSize == expectedBytes && file.sha256 == expectedSha256, "$path identity mismatch.")
    }

    private fun number(
        value: JsonObject,
        name: String,
        parentPath: String,
        allowZero: Boolean,
    ): Double {
        val path = "$parentPath.$name"
        val decimal = (value.values[name] as? JsonNumber)?.value ?: invalid("$path must be a number.")
        expect(if (allowZero) decimal >= BigDecimal.ZERO else decimal > BigDecimal.ZERO, "$path must be ${if (allowZero) "non-negative" else "positive"}.")
        return decimal.toDouble().also { expect(it.isFinite(), "$path must be finite.") }
    }

    private fun positiveIntList(value: JsonArray, path: String): List<Int> =
        value.values.mapIndexed { index, item -> item.asPositiveInt("$path[$index]") }

    private fun stringList(value: JsonArray, path: String): List<String> =
        value.values.mapIndexed { index, item -> item.asString("$path[$index]") }

    private fun multiplyShape(shape: List<Int>, path: String): Long = try {
        shape.fold(1L) { product, dimension -> Math.multiplyExact(product, dimension.toLong()) }
    } catch (error: ArithmeticException) {
        throw BenchmarkModelContractException("$path shape product overflows Long.", error)
    }

    private fun normalizedPath(value: String, path: String): String {
        expect(LOCAL_PATH.matches(value), "$path contains unsupported characters.")
        expect(!value.startsWith('/'), "$path must be relative.")
        expect(value.split('/').none { it.isEmpty() || it == "." || it == ".." }, "$path must be normalized.")
        return value
    }

    private fun hash(value: String, path: String): String = value.also {
        expect(SHA256.matches(it), "$path must be 64 lowercase hexadecimal characters.")
    }

    private fun revision(value: String, path: String): String = value.also {
        expect(GIT_REVISION.matches(it), "$path must be a 40-character Git revision.")
    }

    private fun httpsUri(value: String, path: String): String = value.also {
        val uri = try {
            URI(it)
        } catch (error: Exception) {
            throw BenchmarkModelContractException("$path must be a valid HTTPS URI.", error)
        }
        expect(uri.isAbsolute && !uri.isOpaque && uri.scheme == "https" && !uri.host.isNullOrBlank(), "$path must be an HTTPS URI.")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private const val SCHEMA_VERSION = 1
    private const val CONTRACT_KIND = "generated-litert-runtime-candidate"
    private const val MODEL_ID = "htdemucs_6s_core_smoke_2s_fp32_v1_0_0"
    private const val FLOAT32 = "float32"
    private const val FLOAT32_LE = "float32-le"
    private const val FLOAT32_BYTES = 4L
    private const val NOT_RUN = "not-run"
    private const val LITERT_COORDINATE = "com.google.ai.edge.litert:litert:2.1.5"
    private const val SOURCE_SIDECAR_SHA = "cc8dafb665602b6b8410ca87c6450e29ff4986990f22613e6451cf8da9f4c513"
    private const val HUB_REVISION = "053e1404489b3dc58bf718224fac4b7316de8c93"
    private const val LOADER_REVISION = "eeac1d15891af95b1288d2884b95baa3e5baa96c"
    private const val DEMUCS_LITE_REVISION = "9a2a17c7a81843c2ae49674986f9e1e8b5f6915f"
    private const val WEIGHT_PATH = "models/demucs/official-hf/htdemucs_6s/5c90dfd2.safetensors"
    private const val WEIGHT_SHA = "d2a1745f0744721f6b8ca5bf469b67c651ea5ed1b52998cab033b2158609d411"
    private const val METADATA_PATH = "models/demucs/official-hf/htdemucs_6s/5c90dfd2.json"
    private const val METADATA_SHA = "72d7b4739ba40c8ff1d697404232edd335f397cedbf1bb88eec0034bdbab153e"
    private const val BAG_MANIFEST_PATH = "models/demucs/official-hf/htdemucs_6s/htdemucs_6s.yaml"
    private const val BAG_MANIFEST_SHA = "207405151270af8fd81c2373c25d27950916682ac91dca7884a11ce13dad6f58"
    private const val BOUNDARY_PATH = ".tmp/demucs-lite-reference/demucs-for-onnx/demucs/htdemucs.py"
    private const val BOUNDARY_SHA = "fc9b1debbc2d0e61f523ccd32a22b19f471bc4404d32fadbd79f38c049abcc7b"
    private const val REFERENCE_EXPORTER_PATH = ".tmp/demucs-lite-reference/scripts/convert-pth-to-onnx-chunked.py"
    private const val REFERENCE_EXPORTER_SHA = "57b73642c1ac2d399817a8dff4438db587202a14ffa90877c7b9fb0d95f9e506"
    private const val EXPORT_SCRIPT_PATH = "tools/export_htdemucs_litert_candidate.py"
    private const val EXPORT_SCRIPT_SHA = "a1229069ebee6e48d03507e730b3a4fa428067eba4222d75b3e6e5872bcf50c5"
    private const val GENERATED_ROOT = "models/demucs/generated/$MODEL_ID"
    private const val ONNX_PATH = "$GENERATED_ROOT/htdemucs_6s.core.smoke_2s.fp32.onnx"
    private const val ONNX_SHA = "16464486d8791e673d1ee9bed9aecc7045b4042b3616c41af2b3b0276801879c"
    private const val EXPORT_REPORT_PATH = "$GENERATED_ROOT/export-report.json"
    private const val EXPORT_REPORT_SHA = "f7b8672875f4db58b8ef823bcf1f513b57ba4d878303b2bbd40dc2d38a6bc1e2"
    private const val ARTIFACT_NAME = "htdemucs_6s.core.smoke_2s.fp32.tflite"
    private const val ARTIFACT_PATH = "$GENERATED_ROOT/$ARTIFACT_NAME"
    private const val ARTIFACT_SHA = "35ec0361b8ee6b415d435dc6d35d52dbf13bf9492b3f044e3259f414dbf23e61"
    private const val INSPECTION_PATH = "$GENERATED_ROOT/flatbuffer-inspection.json"
    private const val INSPECTION_SHA = "050291ecbc0fee4480bf661cc59c0c84ba22618b581a357fd290c24ff98337b6"
    private const val COMBINED_SHA = "3f2a30b8cdff473b568aea32bbe4824c8a0fcde500e08185e0d548ba25928b40"
    private const val LITERT_AAR_SHA = "a162d1ddbdad87c002b7ec7eb31a703f2761335e693f292f94091b3569d8aa37"
    private const val QNN_MANIFEST_PATH = "app/src/qnnV79/assets/app-live/runtime-manifest.json"
    private const val QNN_MANIFEST_SHA = "8e20d10a6b27107fcb80460cc27892d5c2b797107c09ecbebeb095fbfe09c297"
    private const val ONNX_ROLE = "diagnostic-parity-intermediate-not-litert-input"

    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val GIT_REVISION = Regex("^[0-9a-f]{40}$")
    private val LOCAL_PATH = Regex("^[A-Za-z0-9._/-]+$")
    private val FORMAT_SUFFIX = mapOf(
        "safetensors" to ".safetensors",
        "json" to ".json",
        "yaml" to ".yaml",
        "python-source" to ".py",
        "onnx-model" to ".onnx",
        "tflite-flatbuffer" to ".tflite",
    )
    private val SUPPORTED_AXES = setOf("batch", "stem", "channel", "sample", "feature", "frequency", "frame")
    private val STEM_ORDER = listOf(
        BenchmarkStemSemantic.DRUMS, BenchmarkStemSemantic.BASS, BenchmarkStemSemantic.OTHER,
        BenchmarkStemSemantic.VOCALS, BenchmarkStemSemantic.GUITAR, BenchmarkStemSemantic.PIANO,
    )
    private val EXPECTED_INPUT_AXES = listOf(
        listOf("batch", "channel", "sample"),
        listOf("batch", "feature", "frequency", "frame"),
    )
    private val EXPECTED_OUTPUT_AXES = listOf(
        listOf("batch", "stem", "feature", "frequency", "frame"),
        listOf("batch", "stem", "channel", "sample"),
    )
    private val EXPECTED_INPUT_FIXTURES = listOf(
        "waveform_input.f32le.raw" to "a644f187d8c6d6644838fd637463695877e5d0c3680ba7fa7364e4b0fe6aa071",
        "spectrum_input.f32le.raw" to "c02deb000bb84026487bf9885faba4f15b5282052c36890a53fae052f3791a6d",
    )
    private val EXPECTED_OUTPUT_FIXTURES = listOf(
        "frequency_golden.f32le.raw" to "8cf015f76b00208013fcb0ba72436b0b0236d3af8de9ac0e634c6bdd0062d02f",
        "waveform_golden.f32le.raw" to "981824eb40608db997b61dfed92d0a28ed2826ab876fc287e28431a7a38903b4",
    )
    private val EXPECTED_DEVICE_PLANS = listOf(
        GeneratedLiteRtDevicePlan(
            "cpu-xnnpack-fp32", "CPU", "XNNPACK", "FP32", "not-applicable", NOT_RUN,
        ),
        GeneratedLiteRtDevicePlan(
            "gpu-opencl-buffer-fp32", "GPU", "LiteRT CompiledModel OPENCL buffer", "FP32",
            "strict-prepare-then-explicit-cpu-remainder", NOT_RUN,
        ),
        GeneratedLiteRtDevicePlan(
            "npu-qnn-v79-fp16", "NPU", "Qualcomm HTP v79", "FP16",
            "strict-non-empty-ir-and-finalize", NOT_RUN,
        ),
    )
    private val EXPECTED_OPERATOR_HISTOGRAM = linkedMapOf(
        "RESHAPE" to 941, "MUL" to 716, "ADD" to 422, "SLICE" to 282, "TRANSPOSE" to 269,
        "SUM" to 152, "GATHER_ND" to 132, "SUB" to 98, "RSQRT" to 96, "CONV_2D" to 88,
        "FULLY_CONNECTED" to 60, "GELU" to 56, "LOGISTIC" to 48, "MEAN" to 44,
        "SQUARED_DIFFERENCE" to 22, "BATCH_MATMUL" to 20, "CONCATENATION" to 16,
        "SOFTMAX" to 10, "TRANSPOSE_CONV" to 8, "BROADCAST_TO" to 5, "STABLEHLO_PAD" to 4,
        "REVERSE_V2" to 4, "PAD" to 3, "SELECT" to 3, "SQRT" to 2, "DIV" to 2, "SELECT_V2" to 1,
    )
}
