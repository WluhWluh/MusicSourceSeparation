package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class ExternalLiteRtContractReference(
    val contractId: String,
    val sidecarPath: String,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
)

data class ExternalLiteRtProvenance(
    val artifactOrigin: String,
    val publisher: String,
    val sourceRepository: String,
    val sourceRevision: String,
    val modelRepository: String,
    val modelRevision: String,
    val conversionRecipeStatus: String,
)

data class ExternalLiteRtArtifact(
    val fileName: String,
    val localPath: String,
    val url: String,
    val byteSize: Long,
    val sha256: String,
    val format: String,
)

data class ExternalLiteRtTensor(
    val index: Int,
    val tensorIndex: Int,
    val name: String,
    val dtype: String,
    val shape: List<Int>,
    val axes: List<String>,
    val elementCount: Long,
    val byteSize: Long,
)

data class ExternalLiteRtSignatureTensor(
    val name: String,
    val tensorIndex: Int,
)

data class ExternalLiteRtSignature(
    val key: String,
    val subgraphIndex: Int,
    val inputs: List<ExternalLiteRtSignatureTensor>,
    val outputs: List<ExternalLiteRtSignatureTensor>,
)

data class ExternalLiteRtFlatBuffer(
    val fileIdentifier: String,
    val schemaVersion: Int,
    val subgraphCount: Int,
    val operatorCount: Int,
    val tensorCount: Int,
    val customOperatorCount: Int,
    val inputs: List<ExternalLiteRtTensor>,
    val outputs: List<ExternalLiteRtTensor>,
    val signatures: List<ExternalLiteRtSignature>,
)

data class ExternalLiteRtModelSemantics(
    val workloadKind: String,
    val sampleRate: Int,
    val channelCount: Int,
    val sampleCount: Int,
    val featureOrder: List<String>,
    val stemOrder: List<BenchmarkStemSemantic>,
)

data class ExternalLiteRtAarIdentity(
    val coordinate: String,
    val fileName: String,
    val byteSize: Long,
    val sha256: String,
)

data class ExternalLiteRtDependency(
    val declaredCoordinate: String,
    val relocatedCoordinate: String,
    val resolvedArtifact: ExternalLiteRtAarIdentity,
)

data class ExternalLiteRtQnnDependency(
    val version: String,
    val runtimeArtifact: ExternalLiteRtAarIdentity,
    val delegateArtifact: ExternalLiteRtAarIdentity,
)

data class ExternalLiteRtCpuPlan(
    val backend: String,
    val precision: String,
    val threadCount: Int,
)

data class ExternalLiteRtNpuPlan(
    val backend: String,
    val precision: String,
    val convHmxMode: String,
    val performanceMode: String,
    val optimizationStrategy: String,
    val delegatedNodeIds: List<Int>,
    val expectedDelegatedNodeCount: Int,
    val expectedPartitionCount: Int,
    val remainingNodeBackend: String,
)

data class ExternalLiteRtRuntime(
    val liteRt: ExternalLiteRtDependency,
    val qnn: ExternalLiteRtQnnDependency,
    val cpuPlan: ExternalLiteRtCpuPlan,
    val npuPlan: ExternalLiteRtNpuPlan,
)

enum class ExternalLiteRtCandidateStatus(val wireValue: String) {
    THROUGHPUT_ONLY("throughput-only");

    companion object {
        internal fun fromWireValue(value: String, path: String): ExternalLiteRtCandidateStatus =
            entries.firstOrNull { it.wireValue == value }
                ?: invalid("$path has unsupported candidate status '$value'.")
    }
}

enum class ExternalLiteRtParityStatus(val wireValue: String) {
    NOT_ESTABLISHED("not-established");

    companion object {
        internal fun fromWireValue(value: String, path: String): ExternalLiteRtParityStatus =
            entries.firstOrNull { it.wireValue == value }
                ?: invalid("$path has unsupported parity status '$value'.")
    }
}

enum class ExternalLiteRtQualityGateStatus(val wireValue: String) {
    NOT_PASSED("not-passed");

    companion object {
        internal fun fromWireValue(value: String, path: String): ExternalLiteRtQualityGateStatus =
            entries.firstOrNull { it.wireValue == value }
                ?: invalid("$path has unsupported quality-gate status '$value'.")
    }
}

data class ExternalLiteRtSnrRange(
    val minimumDb: Double,
    val maximumDb: Double,
)

data class ExternalLiteRtValidation(
    val status: ExternalLiteRtCandidateStatus,
    val evidenceOrigin: String,
    val officialSafetensorsParityStatus: ExternalLiteRtParityStatus,
    val thirdPartyReportedSnr: ExternalLiteRtSnrRange,
    val projectSnrGateDb: Double,
    val projectSnrGateStatus: ExternalLiteRtQualityGateStatus,
)

data class ExternalLiteRtCandidateContract(
    val contractSchemaVersion: Int,
    val contractKind: String,
    val contractId: String,
    val modelId: String,
    val displayName: String,
    val sourceCandidateContract: ExternalLiteRtContractReference,
    val provenance: ExternalLiteRtProvenance,
    val artifact: ExternalLiteRtArtifact,
    val flatBuffer: ExternalLiteRtFlatBuffer,
    val modelSemantics: ExternalLiteRtModelSemantics,
    val runtime: ExternalLiteRtRuntime,
    val validation: ExternalLiteRtValidation,
)

data class LoadedExternalLiteRtCandidateContract(
    val contract: ExternalLiteRtCandidateContract,
    val sidecarFile: File,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
)

object ExternalLiteRtCandidateContractLoader {
    fun load(sidecarFile: File): LoadedExternalLiteRtCandidateContract = loadInternal(sidecarFile)

    fun load(
        sidecarFile: File,
        expectedSidecarSha256: String,
    ): LoadedExternalLiteRtCandidateContract = loadAfterIdentityCheck(
        sidecarFile = sidecarFile,
        expectedSidecarByteSize = null,
        expectedSidecarSha256 = expectedSidecarSha256,
    )

    fun load(
        sidecarFile: File,
        expectedSidecarByteSize: Long,
        expectedSidecarSha256: String,
    ): LoadedExternalLiteRtCandidateContract {
        expect(expectedSidecarByteSize > 0L, "Expected sidecar byte size must be positive.")
        return loadAfterIdentityCheck(sidecarFile, expectedSidecarByteSize, expectedSidecarSha256)
    }

    private fun loadAfterIdentityCheck(
        sidecarFile: File,
        expectedSidecarByteSize: Long?,
        expectedSidecarSha256: String,
    ): LoadedExternalLiteRtCandidateContract {
        expect(
            SHA256.matches(expectedSidecarSha256),
            "Expected sidecar SHA-256 must be 64 lowercase hexadecimal characters.",
        )
        return loadInternal(sidecarFile, expectedSidecarByteSize, expectedSidecarSha256)
    }

    private fun loadInternal(
        sidecarFile: File,
        expectedSidecarByteSize: Long? = null,
        expectedSidecarSha256: String? = null,
    ): LoadedExternalLiteRtCandidateContract {
        if (!sidecarFile.isFile) {
            invalid("External LiteRT contract sidecar is not a readable file: ${sidecarFile.absolutePath}")
        }
        val bytes = try {
            sidecarFile.readBytes()
        } catch (error: Exception) {
            throw BenchmarkModelContractException(
                "Could not read external LiteRT contract sidecar: ${sidecarFile.absolutePath}",
                error,
            )
        }
        if (expectedSidecarByteSize != null) {
            expect(
                bytes.size.toLong() == expectedSidecarByteSize,
                "External LiteRT contract sidecar byte-size mismatch for ${sidecarFile.absolutePath}.",
            )
        }
        val json = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (error: Exception) {
            throw BenchmarkModelContractException(
                "External LiteRT contract sidecar is not valid UTF-8: ${sidecarFile.absolutePath}",
                error,
            )
        }
        val actualSha256 = sha256(bytes)
        if (expectedSidecarSha256 != null) {
            expect(
                MessageDigest.isEqual(
                    expectedSidecarSha256.toByteArray(StandardCharsets.US_ASCII),
                    actualSha256.toByteArray(StandardCharsets.US_ASCII),
                ),
                "External LiteRT contract sidecar SHA-256 mismatch for ${sidecarFile.absolutePath}.",
            )
        }
        return LoadedExternalLiteRtCandidateContract(
            contract = parse(json),
            sidecarFile = sidecarFile.absoluteFile,
            sidecarByteSize = bytes.size.toLong(),
            sidecarSha256 = actualSha256,
        )
    }

    fun parse(json: String): ExternalLiteRtCandidateContract {
        val root = ObjectReader(StrictJsonParser(json).parse().asObject("$"), "$")
        root.requireExactKeys(
            "contractSchemaVersion",
            "contractKind",
            "contractId",
            "modelId",
            "displayName",
            "sourceCandidateContract",
            "provenance",
            "artifact",
            "flatBuffer",
            "modelSemantics",
            "runtime",
            "validation",
        )

        val schemaVersion = root.int("contractSchemaVersion")
        expect(schemaVersion == SCHEMA_VERSION, "$.contractSchemaVersion must equal $SCHEMA_VERSION.")
        val modelId = root.string("modelId")
        expect(MODEL_ID.matches(modelId), "$.modelId must match ${MODEL_ID.pattern}.")
        val contractId = root.string("contractId")
        expect(contractId == "$modelId@$SCHEMA_VERSION", "$.contractId must equal '$modelId@$SCHEMA_VERSION'.")
        val contractKind = root.string("contractKind")
        expect(contractKind == CONTRACT_KIND, "$.contractKind must equal '$CONTRACT_KIND'.")

        val contract = ExternalLiteRtCandidateContract(
            contractSchemaVersion = schemaVersion,
            contractKind = contractKind,
            contractId = contractId,
            modelId = modelId,
            displayName = root.nonEmptyString("displayName"),
            sourceCandidateContract = parseContractReference(root.obj("sourceCandidateContract")),
            provenance = parseProvenance(root.obj("provenance")),
            artifact = parseArtifact(root.obj("artifact")),
            flatBuffer = parseFlatBuffer(root.obj("flatBuffer")),
            modelSemantics = parseModelSemantics(root.obj("modelSemantics")),
            runtime = parseRuntime(root.obj("runtime")),
            validation = parseValidation(root.obj("validation")),
        )
        validateCrossFieldConsistency(contract)
        return contract
    }

    private fun parseContractReference(value: JsonObject): ExternalLiteRtContractReference {
        val path = "$.sourceCandidateContract"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("contractId", "sidecarPath", "sidecarByteSize", "sidecarSha256")
        val contractId = reader.string("contractId")
        expect(V3_CONTRACT_ID.matches(contractId), "$path.contractId must identify a schema v3 contract.")
        val sidecarPath = reader.nonEmptyString("sidecarPath")
        requireNormalizedRelativePath(sidecarPath, "$path.sidecarPath")
        expect(sidecarPath.endsWith(".json"), "$path.sidecarPath must end in .json.")
        return ExternalLiteRtContractReference(
            contractId = contractId,
            sidecarPath = sidecarPath,
            sidecarByteSize = reader.positiveLong("sidecarByteSize"),
            sidecarSha256 = parseSha256(reader.string("sidecarSha256"), "$path.sidecarSha256"),
        )
    }

    private fun parseProvenance(value: JsonObject): ExternalLiteRtProvenance {
        val path = "$.provenance"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "artifactOrigin",
            "publisher",
            "sourceRepository",
            "sourceRevision",
            "modelRepository",
            "modelRevision",
            "conversionRecipeStatus",
        )
        val artifactOrigin = reader.string("artifactOrigin")
        expect(artifactOrigin == THIRD_PARTY_ORIGIN, "$path.artifactOrigin must equal '$THIRD_PARTY_ORIGIN'.")
        val sourceRepository = reader.string("sourceRepository")
        requireHttpsUri(sourceRepository, "$path.sourceRepository")
        val sourceRevision = reader.string("sourceRevision")
        expect(GIT_REVISION.matches(sourceRevision), "$path.sourceRevision must be a 40-character revision.")
        val modelRepository = reader.string("modelRepository")
        expect(HUB_REPOSITORY.matches(modelRepository), "$path.modelRepository is invalid.")
        val modelRevision = reader.string("modelRevision")
        expect(RELEASE_REVISION.matches(modelRevision), "$path.modelRevision must be a fixed release tag.")
        val conversionRecipeStatus = reader.string("conversionRecipeStatus")
        expect(
            conversionRecipeStatus == INCOMPLETE_RECIPE,
            "$path.conversionRecipeStatus must equal '$INCOMPLETE_RECIPE'.",
        )
        return ExternalLiteRtProvenance(
            artifactOrigin = artifactOrigin,
            publisher = reader.nonEmptyString("publisher"),
            sourceRepository = sourceRepository,
            sourceRevision = sourceRevision,
            modelRepository = modelRepository,
            modelRevision = modelRevision,
            conversionRecipeStatus = conversionRecipeStatus,
        )
    }

    private fun parseArtifact(value: JsonObject): ExternalLiteRtArtifact {
        val path = "$.artifact"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("fileName", "localPath", "url", "byteSize", "sha256", "format")
        val fileName = reader.nonEmptyString("fileName")
        expect(fileName.endsWith(TFLITE_SUFFIX), "$path.fileName must end in $TFLITE_SUFFIX.")
        val localPath = reader.nonEmptyString("localPath")
        requireNormalizedRelativePath(localPath, "$path.localPath")
        expect(localPath.substringAfterLast('/') == fileName, "$path.localPath must end with fileName.")
        val url = reader.string("url")
        requireHttpsUri(url, "$path.url")
        val format = reader.string("format")
        expect(format == TFLITE_FORMAT, "$path.format must equal '$TFLITE_FORMAT'.")
        return ExternalLiteRtArtifact(
            fileName = fileName,
            localPath = localPath,
            url = url,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = parseSha256(reader.string("sha256"), "$path.sha256"),
            format = format,
        )
    }

    private fun parseFlatBuffer(value: JsonObject): ExternalLiteRtFlatBuffer {
        val path = "$.flatBuffer"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "fileIdentifier",
            "schemaVersion",
            "subgraphCount",
            "operatorCount",
            "tensorCount",
            "customOperatorCount",
            "inputs",
            "outputs",
            "signatures",
        )
        val fileIdentifier = reader.string("fileIdentifier")
        expect(fileIdentifier == TFLITE_IDENTIFIER, "$path.fileIdentifier must equal '$TFLITE_IDENTIFIER'.")
        val schemaVersion = reader.positiveInt("schemaVersion")
        expect(schemaVersion == TFLITE_SCHEMA_VERSION, "$path.schemaVersion must equal $TFLITE_SCHEMA_VERSION.")
        val subgraphCount = reader.positiveInt("subgraphCount")
        expect(subgraphCount == 1, "$path.subgraphCount must equal 1 for this schema.")
        val inputs = parseTensors(reader.array("inputs"), "$path.inputs")
        val outputs = parseTensors(reader.array("outputs"), "$path.outputs")
        expect(inputs.size == 2, "$path.inputs must contain exactly two tensors.")
        expect(outputs.size == 2, "$path.outputs must contain exactly two tensors.")
        validateTensorList(inputs, "$path.inputs")
        validateTensorList(outputs, "$path.outputs")
        val signatures = parseSignatures(reader.array("signatures"), "$path.signatures")
        expect(signatures.size == 1, "$path.signatures must contain exactly one signature.")
        return ExternalLiteRtFlatBuffer(
            fileIdentifier = fileIdentifier,
            schemaVersion = schemaVersion,
            subgraphCount = subgraphCount,
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
            val dtype = reader.string("dtype")
            expect(dtype == FLOAT32, "$tensorPath.dtype must equal '$FLOAT32'.")
            val shape = reader.array("shape").values.mapIndexed { index, dimension ->
                dimension.asPositiveInt("$tensorPath.shape[$index]")
            }
            expect(shape.size in 3..5, "$tensorPath.shape must contain between three and five dimensions.")
            val axes = reader.array("axes").values.mapIndexed { index, axis ->
                axis.asString("$tensorPath.axes[$index]").also {
                    expect(it in SUPPORTED_AXES, "$tensorPath.axes[$index] has unsupported axis '$it'.")
                }
            }
            expect(shape.size == axes.size, "$tensorPath.axes must have one entry per shape dimension.")
            expect(axes.toSet().size == axes.size, "$tensorPath.axes entries must be unique.")
            expect(axes.first() == BATCH_AXIS, "$tensorPath.axes must start with '$BATCH_AXIS'.")
            expect(shape.first() == 1, "$tensorPath batch dimension must equal 1.")
            val elementCount = try {
                shape.fold(1L) { total, dimension -> Math.multiplyExact(total, dimension.toLong()) }
            } catch (error: ArithmeticException) {
                throw BenchmarkModelContractException("$tensorPath element count overflows Long.", error)
            }
            val byteSize = try {
                Math.multiplyExact(elementCount, FLOAT32_BYTES)
            } catch (error: ArithmeticException) {
                throw BenchmarkModelContractException("$tensorPath byte size overflows Long.", error)
            }
            expect(byteSize <= Int.MAX_VALUE, "$tensorPath is too large for one Android tensor buffer.")
            ExternalLiteRtTensor(
                index = reader.int("index").also {
                    expect(it >= 0, "$tensorPath.index must not be negative.")
                },
                tensorIndex = reader.int("tensorIndex").also {
                    expect(it >= 0, "$tensorPath.tensorIndex must not be negative.")
                },
                name = reader.nonEmptyString("name"),
                dtype = dtype,
                shape = shape,
                axes = axes,
                elementCount = elementCount,
                byteSize = byteSize,
            )
        }

    private fun validateTensorList(tensors: List<ExternalLiteRtTensor>, path: String) {
        expect(
            tensors.map(ExternalLiteRtTensor::index) == tensors.indices.toList(),
            "$path indices must be contiguous and ordered from zero.",
        )
        expect(tensors.map(ExternalLiteRtTensor::tensorIndex).toSet().size == tensors.size, "$path tensorIndex values must be unique.")
        expect(tensors.map(ExternalLiteRtTensor::name).toSet().size == tensors.size, "$path names must be unique.")
    }

    private fun parseSignatures(value: JsonArray, path: String): List<ExternalLiteRtSignature> =
        value.values.mapIndexed { index, item ->
            val signaturePath = "$path[$index]"
            val reader = ObjectReader(item.asObject(signaturePath), signaturePath)
            reader.requireExactKeys("key", "subgraphIndex", "inputs", "outputs")
            ExternalLiteRtSignature(
                key = reader.nonEmptyString("key"),
                subgraphIndex = reader.int("subgraphIndex").also {
                    expect(it >= 0, "$signaturePath.subgraphIndex must not be negative.")
                },
                inputs = parseSignatureTensors(reader.array("inputs"), "$signaturePath.inputs"),
                outputs = parseSignatureTensors(reader.array("outputs"), "$signaturePath.outputs"),
            )
        }.also { signatures ->
            expect(signatures.isNotEmpty(), "$path must not be empty.")
            expect(signatures.map(ExternalLiteRtSignature::key).toSet().size == signatures.size, "$path keys must be unique.")
        }

    private fun parseSignatureTensors(value: JsonArray, path: String): List<ExternalLiteRtSignatureTensor> =
        value.values.mapIndexed { index, item ->
            val tensorPath = "$path[$index]"
            val reader = ObjectReader(item.asObject(tensorPath), tensorPath)
            reader.requireExactKeys("name", "tensorIndex")
            ExternalLiteRtSignatureTensor(
                name = reader.nonEmptyString("name"),
                tensorIndex = reader.int("tensorIndex").also {
                    expect(it >= 0, "$tensorPath.tensorIndex must not be negative.")
                },
            )
        }.also { tensors ->
            expect(tensors.isNotEmpty(), "$path must not be empty.")
            expect(tensors.map(ExternalLiteRtSignatureTensor::name).toSet().size == tensors.size, "$path names must be unique.")
            expect(tensors.map(ExternalLiteRtSignatureTensor::tensorIndex).toSet().size == tensors.size, "$path tensorIndex values must be unique.")
        }

    private fun parseModelSemantics(value: JsonObject): ExternalLiteRtModelSemantics {
        val path = "$.modelSemantics"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "workloadKind",
            "sampleRate",
            "channelCount",
            "sampleCount",
            "featureOrder",
            "stemOrder",
        )
        val workloadKind = reader.string("workloadKind")
        expect(workloadKind == NEURAL_CORE_WORKLOAD, "$path.workloadKind must equal '$NEURAL_CORE_WORKLOAD'.")
        val featureOrder = parseStringList(reader.array("featureOrder"), "$path.featureOrder")
        expect(featureOrder == FEATURE_ORDER, "$path.featureOrder must equal ${FEATURE_ORDER.joinToString()}.")
        val stemOrder = reader.array("stemOrder").values.mapIndexed { index, value ->
            BenchmarkStemSemantic.fromWireValue(value.asString("$path.stemOrder[$index]"), "$path.stemOrder[$index]")
        }
        expect(stemOrder == STEM_ORDER, "$path.stemOrder must use the official HTDemucs 6-stem order.")
        return ExternalLiteRtModelSemantics(
            workloadKind = workloadKind,
            sampleRate = reader.positiveInt("sampleRate").also {
                expect(it == SAMPLE_RATE, "$path.sampleRate must equal $SAMPLE_RATE.")
            },
            channelCount = reader.positiveInt("channelCount").also {
                expect(it == CHANNEL_COUNT, "$path.channelCount must equal $CHANNEL_COUNT.")
            },
            sampleCount = reader.positiveInt("sampleCount"),
            featureOrder = featureOrder,
            stemOrder = stemOrder,
        )
    }

    private fun parseRuntime(value: JsonObject): ExternalLiteRtRuntime {
        val path = "$.runtime"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("liteRt", "qnn", "cpuPlan", "npuPlan")
        return ExternalLiteRtRuntime(
            liteRt = parseLiteRtDependency(reader.obj("liteRt")),
            qnn = parseQnnDependency(reader.obj("qnn")),
            cpuPlan = parseCpuPlan(reader.obj("cpuPlan")),
            npuPlan = parseNpuPlan(reader.obj("npuPlan")),
        )
    }

    private fun parseLiteRtDependency(value: JsonObject): ExternalLiteRtDependency {
        val path = "$.runtime.liteRt"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("declaredCoordinate", "relocatedCoordinate", "resolvedArtifact")
        val declaredCoordinate = parseCoordinate(reader.string("declaredCoordinate"), "$path.declaredCoordinate")
        val relocatedCoordinate = parseCoordinate(reader.string("relocatedCoordinate"), "$path.relocatedCoordinate")
        expect(declaredCoordinate != relocatedCoordinate, "$path must preserve distinct declared and relocated coordinates.")
        val artifact = parseAarIdentity(reader.obj("resolvedArtifact"), "$path.resolvedArtifact")
        expect(artifact.coordinate == relocatedCoordinate, "$path.resolvedArtifact.coordinate must equal relocatedCoordinate.")
        return ExternalLiteRtDependency(declaredCoordinate, relocatedCoordinate, artifact)
    }

    private fun parseQnnDependency(value: JsonObject): ExternalLiteRtQnnDependency {
        val path = "$.runtime.qnn"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("version", "runtimeArtifact", "delegateArtifact")
        val version = reader.nonEmptyString("version")
        expect(VERSION.matches(version), "$path.version is invalid.")
        val runtimeArtifact = parseAarIdentity(reader.obj("runtimeArtifact"), "$path.runtimeArtifact")
        val delegateArtifact = parseAarIdentity(reader.obj("delegateArtifact"), "$path.delegateArtifact")
        expect(runtimeArtifact.coordinate.endsWith(":$version"), "$path.runtimeArtifact.coordinate must use QNN version $version.")
        expect(delegateArtifact.coordinate.endsWith(":$version"), "$path.delegateArtifact.coordinate must use QNN version $version.")
        expect(runtimeArtifact.coordinate != delegateArtifact.coordinate, "$path artifacts must have distinct coordinates.")
        return ExternalLiteRtQnnDependency(version, runtimeArtifact, delegateArtifact)
    }

    private fun parseAarIdentity(value: JsonObject, path: String): ExternalLiteRtAarIdentity {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("coordinate", "fileName", "byteSize", "sha256")
        val coordinate = parseCoordinate(reader.string("coordinate"), "$path.coordinate")
        val parts = coordinate.split(':')
        val fileName = reader.nonEmptyString("fileName")
        expect(fileName == "${parts[1]}-${parts[2]}.aar", "$path.fileName does not match its Maven coordinate.")
        return ExternalLiteRtAarIdentity(
            coordinate = coordinate,
            fileName = fileName,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = parseSha256(reader.string("sha256"), "$path.sha256"),
        )
    }

    private fun parseCpuPlan(value: JsonObject): ExternalLiteRtCpuPlan {
        val path = "$.runtime.cpuPlan"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("backend", "precision", "threadCount")
        val backend = reader.string("backend")
        expect(backend == XNNPACK, "$path.backend must equal '$XNNPACK'.")
        val precision = reader.string("precision")
        expect(precision == FP32, "$path.precision must equal '$FP32'.")
        return ExternalLiteRtCpuPlan(backend, precision, reader.positiveInt("threadCount"))
    }

    private fun parseNpuPlan(value: JsonObject): ExternalLiteRtNpuPlan {
        val path = "$.runtime.npuPlan"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "backend",
            "precision",
            "convHmxMode",
            "performanceMode",
            "optimizationStrategy",
            "delegatedNodeIds",
            "expectedDelegatedNodeCount",
            "expectedPartitionCount",
            "remainingNodeBackend",
        )
        val backend = reader.string("backend")
        expect(backend == QUALCOMM_HTP, "$path.backend must equal '$QUALCOMM_HTP'.")
        val precision = reader.string("precision")
        expect(precision == FP16, "$path.precision must equal '$FP16'.")
        val convHmxMode = reader.string("convHmxMode")
        expect(convHmxMode == HTP_CONV_HMX_ON, "$path.convHmxMode must equal '$HTP_CONV_HMX_ON'.")
        val performanceMode = reader.string("performanceMode")
        expect(performanceMode == HTP_PERFORMANCE_BURST, "$path.performanceMode must equal '$HTP_PERFORMANCE_BURST'.")
        val optimizationStrategy = reader.string("optimizationStrategy")
        expect(
            optimizationStrategy == HTP_OPTIMIZE_FOR_PREPARE,
            "$path.optimizationStrategy must equal '$HTP_OPTIMIZE_FOR_PREPARE'.",
        )
        val delegatedNodeIds = reader.array("delegatedNodeIds").values.mapIndexed { index, node ->
            node.asInt("$path.delegatedNodeIds[$index]").also {
                expect(it >= 0, "$path.delegatedNodeIds[$index] must not be negative.")
            }
        }
        expect(delegatedNodeIds.isNotEmpty(), "$path.delegatedNodeIds must not be empty.")
        expect(delegatedNodeIds == delegatedNodeIds.sorted(), "$path.delegatedNodeIds must be sorted.")
        expect(delegatedNodeIds.toSet().size == delegatedNodeIds.size, "$path.delegatedNodeIds must be unique.")
        val remainingNodeBackend = reader.string("remainingNodeBackend")
        expect(remainingNodeBackend == XNNPACK_FP32, "$path.remainingNodeBackend must equal '$XNNPACK_FP32'.")
        return ExternalLiteRtNpuPlan(
            backend = backend,
            precision = precision,
            convHmxMode = convHmxMode,
            performanceMode = performanceMode,
            optimizationStrategy = optimizationStrategy,
            delegatedNodeIds = delegatedNodeIds,
            expectedDelegatedNodeCount = reader.positiveInt("expectedDelegatedNodeCount"),
            expectedPartitionCount = reader.positiveInt("expectedPartitionCount"),
            remainingNodeBackend = remainingNodeBackend,
        )
    }

    private fun parseValidation(value: JsonObject): ExternalLiteRtValidation {
        val path = "$.validation"
        val reader = ObjectReader(value, path)
        reader.requireExactKeys(
            "status",
            "evidenceOrigin",
            "officialSafetensorsParityStatus",
            "thirdPartyReportedSnr",
            "projectSnrGateDb",
            "projectSnrGateStatus",
        )
        val evidenceOrigin = reader.string("evidenceOrigin")
        expect(evidenceOrigin == THIRD_PARTY_EVIDENCE, "$path.evidenceOrigin must equal '$THIRD_PARTY_EVIDENCE'.")
        val rangeReader = ObjectReader(reader.obj("thirdPartyReportedSnr"), "$path.thirdPartyReportedSnr")
        rangeReader.requireExactKeys("minimumDb", "maximumDb")
        val range = ExternalLiteRtSnrRange(
            minimumDb = rangeReader.positiveDouble("minimumDb"),
            maximumDb = rangeReader.positiveDouble("maximumDb"),
        )
        expect(range.maximumDb >= range.minimumDb, "$path.thirdPartyReportedSnr maximum must not be below minimum.")
        val projectSnrGateDb = reader.positiveDouble("projectSnrGateDb")
        expect(
            range.maximumDb < projectSnrGateDb,
            "$path third-party SNR evidence must remain below the unpassed project gate.",
        )
        return ExternalLiteRtValidation(
            status = ExternalLiteRtCandidateStatus.fromWireValue(reader.string("status"), "$path.status"),
            evidenceOrigin = evidenceOrigin,
            officialSafetensorsParityStatus = ExternalLiteRtParityStatus.fromWireValue(
                reader.string("officialSafetensorsParityStatus"),
                "$path.officialSafetensorsParityStatus",
            ),
            thirdPartyReportedSnr = range,
            projectSnrGateDb = projectSnrGateDb,
            projectSnrGateStatus = ExternalLiteRtQualityGateStatus.fromWireValue(
                reader.string("projectSnrGateStatus"),
                "$path.projectSnrGateStatus",
            ),
        )
    }

    private fun validateCrossFieldConsistency(contract: ExternalLiteRtCandidateContract) {
        val flatBuffer = contract.flatBuffer
        expect(
            flatBuffer.customOperatorCount <= flatBuffer.operatorCount,
            "$.flatBuffer.customOperatorCount must not exceed operatorCount.",
        )
        val allTensors = flatBuffer.inputs + flatBuffer.outputs
        expect(
            allTensors.all { it.tensorIndex < flatBuffer.tensorCount },
            "FlatBuffer input/output tensor indices must be below tensorCount.",
        )
        expect(
            allTensors.map(ExternalLiteRtTensor::tensorIndex).toSet().size == allTensors.size,
            "FlatBuffer input/output tensor indices must be globally unique.",
        )

        val semantics = contract.modelSemantics
        val expectedInputs = listOf(
            ExpectedTensor(
                name = "serving_default_args_0",
                shape = listOf(1, semantics.channelCount, semantics.sampleCount),
                axes = listOf(BATCH_AXIS, CHANNEL_AXIS, SAMPLE_AXIS),
            ),
            ExpectedTensor(
                name = "serving_default_args_1",
                shape = listOf(1, semantics.featureOrder.size, FREQUENCY_BIN_COUNT, SPECTRUM_FRAME_COUNT),
                axes = listOf(BATCH_AXIS, FEATURE_AXIS, FREQUENCY_AXIS, FRAME_AXIS),
            ),
        )
        val expectedOutputs = listOf(
            ExpectedTensor(
                name = "serving_default_output_0_output",
                shape = listOf(
                    1,
                    semantics.stemOrder.size,
                    semantics.featureOrder.size,
                    FREQUENCY_BIN_COUNT,
                    SPECTRUM_FRAME_COUNT,
                ),
                axes = listOf(BATCH_AXIS, STEM_AXIS, FEATURE_AXIS, FREQUENCY_AXIS, FRAME_AXIS),
            ),
            ExpectedTensor(
                name = "serving_default_output_1_output",
                shape = listOf(1, semantics.stemOrder.size, semantics.channelCount, semantics.sampleCount),
                axes = listOf(BATCH_AXIS, STEM_AXIS, CHANNEL_AXIS, SAMPLE_AXIS),
            ),
        )
        validateExpectedTensors(flatBuffer.inputs, expectedInputs, "$.flatBuffer.inputs")
        validateExpectedTensors(flatBuffer.outputs, expectedOutputs, "$.flatBuffer.outputs")

        val signature = flatBuffer.signatures.single()
        expect(signature.key == SERVING_DEFAULT, "$.flatBuffer.signatures[0].key must equal '$SERVING_DEFAULT'.")
        expect(signature.subgraphIndex == 0, "$.flatBuffer.signatures[0].subgraphIndex must equal 0.")
        expect(
            signature.inputs.map(ExternalLiteRtSignatureTensor::name) == listOf("args_0", "args_1"),
            "Serving signature input names must equal args_0, args_1.",
        )
        expect(
            signature.outputs.map(ExternalLiteRtSignatureTensor::name) == listOf("output_0", "output_1"),
            "Serving signature output names must equal output_0, output_1.",
        )
        expect(
            signature.inputs.map(ExternalLiteRtSignatureTensor::tensorIndex) ==
                flatBuffer.inputs.map(ExternalLiteRtTensor::tensorIndex),
            "Serving signature input tensor indices must match flatBuffer.inputs.",
        )
        expect(
            signature.outputs.map(ExternalLiteRtSignatureTensor::tensorIndex) ==
                flatBuffer.outputs.map(ExternalLiteRtTensor::tensorIndex),
            "Serving signature output tensor indices must match flatBuffer.outputs.",
        )

        val npuPlan = contract.runtime.npuPlan
        expect(
            npuPlan.expectedDelegatedNodeCount == npuPlan.delegatedNodeIds.size,
            "$.runtime.npuPlan.expectedDelegatedNodeCount must match delegatedNodeIds.",
        )
        expect(
            npuPlan.delegatedNodeIds.all { it < flatBuffer.operatorCount },
            "$.runtime.npuPlan.delegatedNodeIds must be below flatBuffer.operatorCount.",
        )
        expect(
            npuPlan.expectedPartitionCount <= npuPlan.expectedDelegatedNodeCount,
            "$.runtime.npuPlan.expectedPartitionCount must not exceed delegated node count.",
        )

        val source = contract.sourceCandidateContract
        expect(
            source.contractId.substringBefore('@') == "htdemucs_6s_waveform_7p8s_onnx",
            "$.sourceCandidateContract must reference the six-stem waveform candidate.",
        )
        val provenance = contract.provenance
        expect(
            contract.artifact.url.contains("/${provenance.modelRepository}/resolve/${provenance.modelRevision}/"),
            "$.artifact.url must use the pinned provenance model repository and revision.",
        )
    }

    private fun validateExpectedTensors(
        actual: List<ExternalLiteRtTensor>,
        expected: List<ExpectedTensor>,
        path: String,
    ) {
        actual.zip(expected).forEachIndexed { index, (tensor, expectedTensor) ->
            expect(tensor.name == expectedTensor.name, "$path[$index].name does not match the fixed model ABI.")
            expect(tensor.shape == expectedTensor.shape, "$path[$index].shape does not match the fixed model ABI.")
            expect(tensor.axes == expectedTensor.axes, "$path[$index].axes does not match the fixed model ABI.")
        }
    }

    private fun parseStringList(value: JsonArray, path: String): List<String> =
        value.values.mapIndexed { index, item ->
            item.asString("$path[$index]").also {
                expect(it.isNotEmpty(), "$path[$index] must not be empty.")
            }
        }.also { items ->
            expect(items.isNotEmpty(), "$path must not be empty.")
            expect(items.toSet().size == items.size, "$path entries must be unique.")
        }

    private fun parseCoordinate(value: String, path: String): String = value.also {
        expect(MAVEN_COORDINATE.matches(it), "$path must be a group:artifact:version Maven coordinate.")
    }

    private fun parseSha256(value: String, path: String): String = value.also {
        expect(SHA256.matches(it), "$path must be 64 lowercase hexadecimal characters.")
    }

    private fun requireNormalizedRelativePath(value: String, path: String) {
        expect(RELATIVE_PATH.matches(value), "$path contains unsupported characters.")
        expect(!value.startsWith('/'), "$path must be relative.")
        expect(value.split('/').none { it.isEmpty() || it == "." || it == ".." }, "$path is not normalized.")
    }

    private fun requireHttpsUri(value: String, path: String) {
        val uri = try {
            URI(value)
        } catch (error: Exception) {
            throw BenchmarkModelContractException("$path must be a valid HTTPS URI.", error)
        }
        expect(uri.scheme == "https" && !uri.host.isNullOrEmpty(), "$path must be a valid HTTPS URI.")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private data class ExpectedTensor(
        val name: String,
        val shape: List<Int>,
        val axes: List<String>,
    )

    private const val SCHEMA_VERSION = 4
    private const val CONTRACT_KIND = "external-litert-runtime-candidate"
    private const val TFLITE_FORMAT = "tflite-flatbuffer"
    private const val TFLITE_SUFFIX = ".tflite"
    private const val TFLITE_IDENTIFIER = "TFL3"
    private const val TFLITE_SCHEMA_VERSION = 3
    private const val FLOAT32 = "float32"
    private const val FLOAT32_BYTES = 4L
    private const val THIRD_PARTY_ORIGIN = "third-party-public-release"
    private const val INCOMPLETE_RECIPE = "incomplete-public-wrapper"
    private const val NEURAL_CORE_WORKLOAD = "neural-core-segment"
    private const val SAMPLE_RATE = 44_100
    private const val CHANNEL_COUNT = 2
    private const val FREQUENCY_BIN_COUNT = 2_048
    private const val SPECTRUM_FRAME_COUNT = 336
    private const val SERVING_DEFAULT = "serving_default"
    private const val XNNPACK = "XNNPACK"
    private const val FP32 = "FP32"
    private const val FP16 = "FP16"
    private const val QUALCOMM_HTP = "Qualcomm HTP"
    private const val HTP_CONV_HMX_ON = "HTP_CONV_HMX_ON"
    private const val HTP_PERFORMANCE_BURST = "HTP_PERFORMANCE_BURST"
    private const val HTP_OPTIMIZE_FOR_PREPARE = "HTP_OPTIMIZE_FOR_PREPARE"
    private const val XNNPACK_FP32 = "XNNPACK FP32"
    private const val THIRD_PARTY_EVIDENCE = "third-party"
    private const val BATCH_AXIS = "batch"
    private const val CHANNEL_AXIS = "channel"
    private const val SAMPLE_AXIS = "sample"
    private const val FEATURE_AXIS = "feature"
    private const val FREQUENCY_AXIS = "frequency"
    private const val FRAME_AXIS = "frame"
    private const val STEM_AXIS = "stem"
    private val FEATURE_ORDER = listOf("L.real", "L.imag", "R.real", "R.imag")
    private val STEM_ORDER = listOf(
        BenchmarkStemSemantic.DRUMS,
        BenchmarkStemSemantic.BASS,
        BenchmarkStemSemantic.OTHER,
        BenchmarkStemSemantic.VOCALS,
        BenchmarkStemSemantic.GUITAR,
        BenchmarkStemSemantic.PIANO,
    )
    private val SUPPORTED_AXES = setOf(
        BATCH_AXIS,
        CHANNEL_AXIS,
        SAMPLE_AXIS,
        FEATURE_AXIS,
        FREQUENCY_AXIS,
        FRAME_AXIS,
        STEM_AXIS,
    )
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val GIT_REVISION = Regex("^[0-9a-f]{40}$")
    private val MODEL_ID = Regex("^[a-z0-9_]+$")
    private val V3_CONTRACT_ID = Regex("^[a-z0-9_]+@3$")
    private val HUB_REPOSITORY = Regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    private val RELEASE_REVISION = Regex("^v[0-9]+\\.[0-9]+\\.[0-9]+$")
    private val VERSION = Regex("^[0-9]+\\.[0-9]+\\.[0-9]+$")
    private val MAVEN_COORDINATE = Regex("^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+:[A-Za-z0-9_.+-]+$")
    private val RELATIVE_PATH = Regex("^[A-Za-z0-9._/-]+$")
}
