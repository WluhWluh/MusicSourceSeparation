package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.net.URI
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

enum class TensorOnlyArtifactFormat(
    val wireValue: String,
    val fileSuffix: String,
) {
    ONNX_MODEL("onnx-model", ".onnx"),
    PYTORCH_CHECKPOINT("pytorch-checkpoint", ".th"),
    SAFETENSORS("safetensors", ".safetensors"),
    JSON("json", ".json"),
    YAML("yaml", ".yaml"),
    TEXT("text", ".txt"),
    MARKDOWN("markdown", ".md");

    companion object {
        internal fun fromWireValue(value: String, path: String): TensorOnlyArtifactFormat =
            entries.firstOrNull { it.wireValue == value }
                ?: invalid("$path has unsupported artifact format '$value'.")
    }
}

data class TensorOnlyArtifact(
    val fileName: String,
    val localPath: String,
    val url: String,
    val byteSize: Long,
    val sha256: String,
    val format: TensorOnlyArtifactFormat,
)

data class TensorOnlyUpstream(
    val hubRepository: String,
    val hubRevision: String,
    val loaderRepository: String,
    val loaderRevision: String,
    val licenseSpdx: String,
    val weight: TensorOnlyArtifact,
    val metadata: TensorOnlyArtifact,
    val bagManifest: TensorOnlyArtifact,
)

data class TensorOnlyConversion(
    val repository: String,
    val revision: String,
    val modelRepository: String,
    val modelRevision: String,
    val exporterInput: TensorOnlyArtifact,
    val opset: Int,
    val producerName: String,
    val producerVersion: String,
    val verificationStatus: String,
    val repositoryLicenseSpdx: String,
    val modelLicenseSpdx: String,
    val repositoryLicense: TensorOnlyArtifact,
    val modelCard: TensorOnlyArtifact,
)

data class TensorOnlyDtypeInventory(
    val dtype: String,
    val tensorCount: Int,
    val elementCount: Long,
    val byteSize: Long,
)

data class TensorOnlyPrecisionContract(
    val scope: String,
    val ioDtype: String,
    val quantization: String,
    val runtimePrecisionStatus: String,
    val initializerStorage: List<TensorOnlyDtypeInventory>,
    val constantStorage: List<TensorOnlyDtypeInventory>,
)

enum class TensorOnlyWorkloadKind(val wireValue: String) {
    WAVEFORM_SEGMENT("waveform-segment"),
    NEURAL_CORE_SEGMENT("neural-core-segment");

    companion object {
        internal fun fromWireValue(value: String, path: String): TensorOnlyWorkloadKind =
            entries.firstOrNull { it.wireValue == value }
                ?: invalid("$path has unsupported workload kind '$value'.")
    }
}

data class TensorOnlyWorkload(
    val kind: TensorOnlyWorkloadKind,
    val sampleRate: Int,
    val channelCount: Int,
    val sampleCount: Int,
    val segmentNumerator: Int,
    val segmentDenominator: Int,
)

data class TensorOnlyTensor(
    val index: Int,
    val name: String,
    val dtype: String,
    val shape: List<Int>,
    val axes: List<String>,
    val elementCount: Long,
    val byteSize: Long,
)

data class TensorOnlyTensorContract(
    val inputEncoding: String,
    val inputs: List<TensorOnlyTensor>,
    val outputs: List<TensorOnlyTensor>,
)

enum class TensorOnlyStemPackingKind(val wireValue: String) {
    AXIS_CONTIGUOUS("axis-contiguous"),
    WHOLE_TENSOR("whole-tensor");

    companion object {
        internal fun fromWireValue(value: String, path: String): TensorOnlyStemPackingKind =
            entries.firstOrNull { it.wireValue == value }
                ?: invalid("$path has unsupported stem packing '$value'.")
    }
}

data class TensorOnlyStemPacking(
    val kind: TensorOnlyStemPackingKind,
    val axis: Int?,
    val elementsPerStem: Int?,
)

data class TensorOnlyStemBinding(
    val outputIndex: Int,
    val packing: TensorOnlyStemPacking,
    val stems: List<BenchmarkStem>,
)

data class TensorOnlyStemContract(
    val derivation: String,
    val bindings: List<TensorOnlyStemBinding>,
)

data class TensorOnlyPipelineCompatibility(
    val pipelineId: String,
    val minimumVersion: Int,
    val maximumVersion: Int,
)

data class TensorOnlyCandidateContract(
    val contractSchemaVersion: Int,
    val contractKind: String,
    val contractId: String,
    val modelId: String,
    val displayName: String,
    val conversionSource: TensorOnlyArtifact,
    val upstream: TensorOnlyUpstream,
    val conversion: TensorOnlyConversion,
    val precisionContract: TensorOnlyPrecisionContract,
    val workloadContract: TensorOnlyWorkload,
    val tensorContract: TensorOnlyTensorContract,
    val stemContract: TensorOnlyStemContract,
    val pipelineCompatibility: TensorOnlyPipelineCompatibility,
) {
    val stems: List<BenchmarkStem> = when (workloadContract.kind) {
        TensorOnlyWorkloadKind.WAVEFORM_SEGMENT ->
            stemContract.bindings.flatMap(TensorOnlyStemBinding::stems)
        TensorOnlyWorkloadKind.NEURAL_CORE_SEGMENT ->
            stemContract.bindings.firstOrNull()?.stems.orEmpty()
    }
}

data class LoadedTensorOnlyCandidateContract(
    val contract: TensorOnlyCandidateContract,
    val sidecarFile: File,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
)

object TensorOnlyCandidateContractLoader {
    fun load(sidecarFile: File): LoadedTensorOnlyCandidateContract = loadInternal(sidecarFile, null)

    fun load(sidecarFile: File, expectedSidecarSha256: String): LoadedTensorOnlyCandidateContract {
        return loadInternalAfterIdentityCheck(sidecarFile, null, expectedSidecarSha256)
    }

    fun load(
        sidecarFile: File,
        expectedSidecarByteSize: Long,
        expectedSidecarSha256: String,
    ): LoadedTensorOnlyCandidateContract {
        expect(expectedSidecarByteSize > 0L, "Expected sidecar byte size must be positive.")
        return loadInternalAfterIdentityCheck(sidecarFile, expectedSidecarByteSize, expectedSidecarSha256)
    }

    private fun loadInternalAfterIdentityCheck(
        sidecarFile: File,
        expectedSidecarByteSize: Long?,
        expectedSidecarSha256: String,
    ): LoadedTensorOnlyCandidateContract {
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
    ): LoadedTensorOnlyCandidateContract {
        if (!sidecarFile.isFile) {
            invalid("Tensor-only contract sidecar is not a readable file: ${sidecarFile.absolutePath}")
        }
        val bytes = try {
            sidecarFile.readBytes()
        } catch (error: Exception) {
            throw BenchmarkModelContractException(
                "Could not read tensor-only contract sidecar: ${sidecarFile.absolutePath}",
                error,
            )
        }
        if (expectedSidecarByteSize != null) {
            expect(
                bytes.size.toLong() == expectedSidecarByteSize,
                "Tensor-only contract sidecar byte-size mismatch for ${sidecarFile.absolutePath}.",
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
                "Tensor-only contract sidecar is not valid UTF-8: ${sidecarFile.absolutePath}",
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
                "Tensor-only contract sidecar SHA-256 mismatch for ${sidecarFile.absolutePath}.",
            )
        }
        return LoadedTensorOnlyCandidateContract(
            contract = parse(json),
            sidecarFile = sidecarFile.absoluteFile,
            sidecarByteSize = bytes.size.toLong(),
            sidecarSha256 = actualSha256,
        )
    }

    fun parse(json: String): TensorOnlyCandidateContract {
        val root = ObjectReader(StrictJsonParser(json).parse().asObject("$"), "$")
        root.requireExactKeys(
            "contractSchemaVersion",
            "contractKind",
            "contractId",
            "modelId",
            "displayName",
            "conversionSource",
            "upstream",
            "conversion",
            "precisionContract",
            "workloadContract",
            "tensorContract",
            "stemContract",
            "pipelineCompatibility",
        )

        val schemaVersion = root.int("contractSchemaVersion")
        expect(schemaVersion == SCHEMA_VERSION, "$.contractSchemaVersion must equal $SCHEMA_VERSION.")
        val contractId = root.string("contractId")
        val modelId = root.string("modelId")
        expect(MODEL_ID.matches(modelId), "$.modelId must match ${MODEL_ID.pattern}.")
        expect(CONTRACT_ID.matches(contractId), "$.contractId must match ${CONTRACT_ID.pattern}.")
        expect(contractId == "$modelId@$SCHEMA_VERSION", "$.contractId must equal '$modelId@$SCHEMA_VERSION'.")

        val contractKind = root.string("contractKind")
        expect(contractKind == CONTRACT_KIND, "$.contractKind must equal '$CONTRACT_KIND'.")
        val contract = TensorOnlyCandidateContract(
            contractSchemaVersion = schemaVersion,
            contractKind = contractKind,
            contractId = contractId,
            modelId = modelId,
            displayName = root.nonEmptyString("displayName"),
            conversionSource = parseArtifact(
                root.obj("conversionSource"),
                "$.conversionSource",
                TensorOnlyArtifactFormat.ONNX_MODEL,
            ),
            upstream = parseUpstream(root.obj("upstream")),
            conversion = parseConversion(root.obj("conversion")),
            precisionContract = parsePrecisionContract(root.obj("precisionContract")),
            workloadContract = parseWorkload(root.obj("workloadContract")),
            tensorContract = parseTensorContract(root.obj("tensorContract")),
            stemContract = parseStemContract(root.obj("stemContract")),
            pipelineCompatibility = parsePipeline(root.obj("pipelineCompatibility")),
        )
        validateCrossFieldConsistency(contract)
        return contract
    }

    private fun parseArtifact(
        value: JsonObject,
        path: String,
        expectedFormat: TensorOnlyArtifactFormat? = null,
    ): TensorOnlyArtifact {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("fileName", "localPath", "url", "byteSize", "sha256", "format")
        val fileName = reader.nonEmptyString("fileName")
        val localPath = reader.nonEmptyString("localPath")
        expect(LOCAL_PATH.matches(localPath), "$path.localPath contains unsupported characters.")
        expect(!localPath.startsWith('/'), "$path.localPath must be relative.")
        val pathParts = localPath.split('/')
        expect(pathParts.none { it.isEmpty() || it == "." || it == ".." }, "$path.localPath is not normalized.")
        expect(pathParts.last() == fileName, "$path.localPath must end with fileName '$fileName'.")
        val url = reader.string("url")
        requireHttpsUri(url, "$path.url")
        val sha256 = reader.string("sha256")
        expect(SHA256.matches(sha256), "$path.sha256 must be 64 lowercase hexadecimal characters.")
        val format = TensorOnlyArtifactFormat.fromWireValue(reader.string("format"), "$path.format")
        if (expectedFormat != null) {
            expect(format == expectedFormat, "$path.format must equal '${expectedFormat.wireValue}'.")
        }
        expect(fileName.endsWith(format.fileSuffix), "$path.fileName must end in ${format.fileSuffix}.")
        return TensorOnlyArtifact(
            fileName = fileName,
            localPath = localPath,
            url = url,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = sha256,
            format = format,
        )
    }

    private fun parseUpstream(value: JsonObject): TensorOnlyUpstream {
        val reader = ObjectReader(value, "$.upstream")
        reader.requireExactKeys(
            "hubRepository",
            "hubRevision",
            "loaderRepository",
            "loaderRevision",
            "licenseSpdx",
            "weight",
            "metadata",
            "bagManifest",
        )
        val hubRepository = reader.string("hubRepository")
        expect(HUB_REPOSITORY.matches(hubRepository), "$.upstream.hubRepository is invalid.")
        val hubRevision = reader.string("hubRevision")
        expect(GIT_REVISION.matches(hubRevision), "$.upstream.hubRevision must be a 40-character revision.")
        val loaderRepository = reader.string("loaderRepository")
        requireHttpsUri(loaderRepository, "$.upstream.loaderRepository")
        val loaderRevision = reader.string("loaderRevision")
        expect(GIT_REVISION.matches(loaderRevision), "$.upstream.loaderRevision must be a 40-character revision.")
        val licenseSpdx = reader.string("licenseSpdx")
        expect(licenseSpdx == MIT, "$.upstream.licenseSpdx must equal '$MIT'.")
        return TensorOnlyUpstream(
            hubRepository = hubRepository,
            hubRevision = hubRevision,
            loaderRepository = loaderRepository,
            loaderRevision = loaderRevision,
            licenseSpdx = licenseSpdx,
            weight = parseArtifact(reader.obj("weight"), "$.upstream.weight", TensorOnlyArtifactFormat.SAFETENSORS),
            metadata = parseArtifact(reader.obj("metadata"), "$.upstream.metadata", TensorOnlyArtifactFormat.JSON),
            bagManifest = parseArtifact(
                reader.obj("bagManifest"),
                "$.upstream.bagManifest",
                TensorOnlyArtifactFormat.YAML,
            ),
        )
    }

    private fun parseConversion(value: JsonObject): TensorOnlyConversion {
        val reader = ObjectReader(value, "$.conversion")
        reader.requireExactKeys(
            "repository",
            "revision",
            "modelRepository",
            "modelRevision",
            "exporterInput",
            "opset",
            "producerName",
            "producerVersion",
            "verificationStatus",
            "repositoryLicenseSpdx",
            "modelLicenseSpdx",
            "repositoryLicense",
            "modelCard",
        )
        val repository = reader.string("repository")
        requireHttpsUri(repository, "$.conversion.repository")
        val revision = reader.string("revision")
        expect(GIT_REVISION.matches(revision), "$.conversion.revision must be a 40-character revision.")
        val modelRepository = reader.string("modelRepository")
        expect(HUB_REPOSITORY.matches(modelRepository), "$.conversion.modelRepository is invalid.")
        val modelRevision = reader.string("modelRevision")
        expect(GIT_REVISION.matches(modelRevision), "$.conversion.modelRevision must be a 40-character revision.")
        val opset = reader.positiveInt("opset")
        expect(opset in 13..18, "$.conversion.opset must be between 13 and 18.")
        val verificationStatus = reader.string("verificationStatus")
        expect(
            verificationStatus in VERIFICATION_STATUSES,
            "$.conversion.verificationStatus is unsupported.",
        )
        return TensorOnlyConversion(
            repository = repository,
            revision = revision,
            modelRepository = modelRepository,
            modelRevision = modelRevision,
            exporterInput = parseArtifact(
                reader.obj("exporterInput"),
                "$.conversion.exporterInput",
                TensorOnlyArtifactFormat.PYTORCH_CHECKPOINT,
            ),
            opset = opset,
            producerName = reader.nonEmptyString("producerName"),
            producerVersion = reader.nonEmptyString("producerVersion"),
            verificationStatus = verificationStatus,
            repositoryLicenseSpdx = reader.string("repositoryLicenseSpdx").also {
                expect(it == MIT, "$.conversion.repositoryLicenseSpdx must equal '$MIT'.")
            },
            modelLicenseSpdx = reader.string("modelLicenseSpdx").also {
                expect(it == MIT, "$.conversion.modelLicenseSpdx must equal '$MIT'.")
            },
            repositoryLicense = parseArtifact(
                reader.obj("repositoryLicense"),
                "$.conversion.repositoryLicense",
                TensorOnlyArtifactFormat.TEXT,
            ),
            modelCard = parseArtifact(
                reader.obj("modelCard"),
                "$.conversion.modelCard",
                TensorOnlyArtifactFormat.MARKDOWN,
            ),
        )
    }

    private fun parsePrecisionContract(value: JsonObject): TensorOnlyPrecisionContract {
        val reader = ObjectReader(value, "$.precisionContract")
        reader.requireExactKeys(
            "scope",
            "ioDtype",
            "quantization",
            "runtimePrecisionStatus",
            "initializerStorage",
            "constantStorage",
        )
        val scope = reader.string("scope")
        expect(scope == PRECISION_SCOPE, "$.precisionContract.scope must equal '$PRECISION_SCOPE'.")
        val ioDtype = reader.string("ioDtype")
        expect(ioDtype == FLOAT32, "$.precisionContract.ioDtype must equal '$FLOAT32'.")
        val quantization = reader.string("quantization")
        expect(quantization == NO_QUANTIZATION, "$.precisionContract.quantization must equal '$NO_QUANTIZATION'.")
        val runtimePrecisionStatus = reader.string("runtimePrecisionStatus")
        expect(
            runtimePrecisionStatus == RUNTIME_PRECISION_NOT_ESTABLISHED,
            "$.precisionContract.runtimePrecisionStatus must equal '$RUNTIME_PRECISION_NOT_ESTABLISHED'.",
        )
        return TensorOnlyPrecisionContract(
            scope = scope,
            ioDtype = ioDtype,
            quantization = quantization,
            runtimePrecisionStatus = runtimePrecisionStatus,
            initializerStorage = parseDtypeInventory(reader.array("initializerStorage"), "$.precisionContract.initializerStorage"),
            constantStorage = parseDtypeInventory(reader.array("constantStorage"), "$.precisionContract.constantStorage"),
        )
    }

    private fun parseDtypeInventory(value: JsonArray, path: String): List<TensorOnlyDtypeInventory> {
        val entries = value.values.mapIndexed { index, item ->
            val entryPath = "$path[$index]"
            val reader = ObjectReader(item.asObject(entryPath), entryPath)
            reader.requireExactKeys("dtype", "tensorCount", "elementCount", "byteSize")
            val dtype = reader.string("dtype")
            val bytesPerElement = DTYPE_BYTES[dtype]
                ?: invalid("$entryPath.dtype is unsupported.")
            val tensorCount = reader.positiveInt("tensorCount")
            val elementCount = reader.positiveLong("elementCount")
            val byteSize = reader.positiveLong("byteSize")
            val expectedByteSize = try {
                Math.multiplyExact(elementCount, bytesPerElement)
            } catch (error: ArithmeticException) {
                throw BenchmarkModelContractException("$entryPath byte size overflows Long.", error)
            }
            expect(byteSize == expectedByteSize, "$entryPath.byteSize does not match dtype element count.")
            TensorOnlyDtypeInventory(dtype, tensorCount, elementCount, byteSize)
        }
        expect(entries.isNotEmpty(), "$path must not be empty.")
        expect(entries.map(TensorOnlyDtypeInventory::dtype).toSet().size == entries.size, "$path dtypes must be unique.")
        return entries
    }

    private fun parseWorkload(value: JsonObject): TensorOnlyWorkload {
        val reader = ObjectReader(value, "$.workloadContract")
        reader.requireExactKeys(
            "kind",
            "sampleRate",
            "channelCount",
            "sampleCount",
            "segmentNumerator",
            "segmentDenominator",
        )
        val sampleRate = reader.positiveInt("sampleRate")
        expect(sampleRate == 44_100, "$.workloadContract.sampleRate must equal 44100.")
        val channelCount = reader.positiveInt("channelCount")
        expect(channelCount == 2, "$.workloadContract.channelCount must equal 2.")
        return TensorOnlyWorkload(
            kind = TensorOnlyWorkloadKind.fromWireValue(reader.string("kind"), "$.workloadContract.kind"),
            sampleRate = sampleRate,
            channelCount = channelCount,
            sampleCount = reader.positiveInt("sampleCount"),
            segmentNumerator = reader.positiveInt("segmentNumerator"),
            segmentDenominator = reader.positiveInt("segmentDenominator"),
        )
    }

    private fun parseTensorContract(value: JsonObject): TensorOnlyTensorContract {
        val reader = ObjectReader(value, "$.tensorContract")
        reader.requireExactKeys("inputEncoding", "inputs", "outputs")
        val inputEncoding = reader.string("inputEncoding")
        expect(inputEncoding == FLOAT32_LE, "$.tensorContract.inputEncoding must equal '$FLOAT32_LE'.")
        val inputs = parseTensors(reader.array("inputs"), "$.tensorContract.inputs")
        val outputs = parseTensors(reader.array("outputs"), "$.tensorContract.outputs")
        expect(inputs.isNotEmpty(), "$.tensorContract.inputs must not be empty.")
        expect(outputs.isNotEmpty(), "$.tensorContract.outputs must not be empty.")
        validateTensorIndicesAndNames(inputs, "$.tensorContract.inputs")
        validateTensorIndicesAndNames(outputs, "$.tensorContract.outputs")
        return TensorOnlyTensorContract(inputEncoding, inputs, outputs)
    }

    private fun parseTensors(value: JsonArray, path: String): List<TensorOnlyTensor> =
        value.values.mapIndexed { position, item ->
            val tensorPath = "$path[$position]"
            val reader = ObjectReader(item.asObject(tensorPath), tensorPath)
            reader.requireExactKeys("index", "name", "dtype", "shape", "axes")
            val dtype = reader.string("dtype")
            expect(dtype == FLOAT32, "$tensorPath.dtype must equal '$FLOAT32'.")
            val shape = reader.array("shape").values.mapIndexed { index, dimension ->
                dimension.asPositiveInt("$tensorPath.shape[$index]")
            }
            expect(shape.size in 2..6, "$tensorPath.shape must contain between two and six dimensions.")
            val axes = reader.array("axes").values.mapIndexed { index, axis ->
                axis.asString("$tensorPath.axes[$index]").also {
                    expect(it in AXES, "$tensorPath.axes[$index] has unsupported axis '$it'.")
                }
            }
            expect(axes.size == shape.size, "$tensorPath.axes must have one entry per shape dimension.")
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
            TensorOnlyTensor(
                index = reader.int("index").also {
                    expect(it >= 0, "$tensorPath.index must not be negative.")
                },
                name = reader.nonEmptyString("name"),
                dtype = dtype,
                shape = shape,
                axes = axes,
                elementCount = elementCount,
                byteSize = byteSize,
            )
        }

    private fun validateTensorIndicesAndNames(tensors: List<TensorOnlyTensor>, path: String) {
        expect(
            tensors.map(TensorOnlyTensor::index) == tensors.indices.toList(),
            "$path indices must be contiguous and ordered from zero.",
        )
        val names = tensors.map(TensorOnlyTensor::name)
        expect(names.toSet().size == names.size, "$path names must be unique.")
    }

    private fun parseStemContract(value: JsonObject): TensorOnlyStemContract {
        val reader = ObjectReader(value, "$.stemContract")
        reader.requireExactKeys("derivation", "bindings")
        val derivation = reader.string("derivation")
        expect(derivation == DIRECT_DERIVATION, "$.stemContract.derivation must equal '$DIRECT_DERIVATION'.")
        val bindings = reader.array("bindings").values.mapIndexed { index, item ->
            val path = "$.stemContract.bindings[$index]"
            val bindingReader = ObjectReader(item.asObject(path), path)
            bindingReader.requireExactKeys("outputIndex", "packing", "stems")
            val outputIndex = bindingReader.int("outputIndex")
            expect(outputIndex >= 0, "$path.outputIndex must not be negative.")
            val stems = bindingReader.array("stems").values.mapIndexed { stemIndex, stemValue ->
                parseStem(stemValue.asObject("$path.stems[$stemIndex]"), "$path.stems[$stemIndex]")
            }
            expect(stems.isNotEmpty(), "$path.stems must not be empty.")
            TensorOnlyStemBinding(
                outputIndex = outputIndex,
                packing = parsePacking(bindingReader.obj("packing"), "$path.packing"),
                stems = stems,
            )
        }
        expect(bindings.isNotEmpty(), "$.stemContract.bindings must not be empty.")
        return TensorOnlyStemContract(derivation, bindings)
    }

    private fun parsePacking(value: JsonObject, path: String): TensorOnlyStemPacking {
        val reader = ObjectReader(value, path)
        val kind = TensorOnlyStemPackingKind.fromWireValue(reader.string("kind"), "$path.kind")
        return when (kind) {
            TensorOnlyStemPackingKind.AXIS_CONTIGUOUS -> {
                reader.requireExactKeys("kind", "axis", "elementsPerStem")
                val axis = reader.int("axis")
                expect(axis >= 0, "$path.axis must not be negative.")
                TensorOnlyStemPacking(kind, axis, reader.positiveInt("elementsPerStem"))
            }
            TensorOnlyStemPackingKind.WHOLE_TENSOR -> {
                reader.requireExactKeys("kind")
                TensorOnlyStemPacking(kind, null, null)
            }
        }
    }

    private fun parseStem(value: JsonObject, path: String): BenchmarkStem {
        val reader = ObjectReader(value, path)
        reader.requireExactKeys("semantic", "displayLabel")
        return BenchmarkStem(
            semantic = BenchmarkStemSemantic.fromWireValue(reader.string("semantic"), "$path.semantic"),
            displayLabel = reader.nonEmptyString("displayLabel"),
        )
    }

    private fun parsePipeline(value: JsonObject): TensorOnlyPipelineCompatibility {
        val reader = ObjectReader(value, "$.pipelineCompatibility")
        reader.requireExactKeys("pipelineId", "minimumVersion", "maximumVersion")
        val pipelineId = reader.string("pipelineId")
        expect(pipelineId == PIPELINE_ID, "$.pipelineCompatibility.pipelineId must equal '$PIPELINE_ID'.")
        val minimum = reader.positiveInt("minimumVersion")
        val maximum = reader.positiveInt("maximumVersion")
        expect(maximum >= minimum, "$.pipelineCompatibility.maximumVersion must not be below minimumVersion.")
        return TensorOnlyPipelineCompatibility(pipelineId, minimum, maximum)
    }

    private fun validateCrossFieldConsistency(contract: TensorOnlyCandidateContract) {
        val workload = contract.workloadContract
        val scaledSamples = try {
            Math.multiplyExact(workload.sampleRate.toLong(), workload.segmentNumerator.toLong())
        } catch (error: ArithmeticException) {
            throw BenchmarkModelContractException("Workload segment sample calculation overflows Long.", error)
        }
        expect(
            scaledSamples % workload.segmentDenominator == 0L,
            "$.workloadContract segment duration must resolve to a whole sample count.",
        )
        expect(
            scaledSamples / workload.segmentDenominator == workload.sampleCount.toLong(),
            "$.workloadContract.sampleCount does not match the exact segment duration.",
        )

        val waveformShape = listOf(1, workload.channelCount, workload.sampleCount)
        val waveformInput = contract.tensorContract.inputs.firstOrNull {
            it.axes == listOf(BATCH_AXIS, CHANNEL_AXIS, SAMPLE_AXIS)
        }
        expect(waveformInput != null, "$.tensorContract.inputs must include a batch/channel/sample waveform tensor.")
        expect(waveformInput?.shape == waveformShape, "Waveform input shape must equal $waveformShape.")
        expect(
            contract.precisionContract.ioDtype == FLOAT32 &&
                contract.tensorContract.inputs.all { it.dtype == FLOAT32 } &&
                contract.tensorContract.outputs.all { it.dtype == FLOAT32 },
            "$.precisionContract.ioDtype must match every tensor contract dtype.",
        )

        when (workload.kind) {
            TensorOnlyWorkloadKind.WAVEFORM_SEGMENT -> {
                expect(
                    contract.tensorContract.inputs.size == 1,
                    "Waveform-segment candidates must have exactly one input.",
                )
                contract.tensorContract.outputs.forEachIndexed { index, output ->
                    val path = "$.tensorContract.outputs[$index]"
                    expect(
                        CHANNEL_AXIS in output.axes && SAMPLE_AXIS in output.axes,
                        "$path must contain channel and sample axes for a waveform-segment workload.",
                    )
                    validateWorkloadAxis(output, CHANNEL_AXIS, workload.channelCount, path)
                    validateWorkloadAxis(output, SAMPLE_AXIS, workload.sampleCount, path)
                }
            }
            TensorOnlyWorkloadKind.NEURAL_CORE_SEGMENT -> {
                expect(
                    contract.tensorContract.inputs.size >= 2,
                    "Neural-core candidates must have at least two inputs.",
                )
                expect(
                    contract.tensorContract.outputs.size >= 2,
                    "Neural-core candidates must have at least two outputs.",
                )
            }
        }

        val outputs = contract.tensorContract.outputs
        val bindings = contract.stemContract.bindings
        expect(
            bindings.map(TensorOnlyStemBinding::outputIndex) == outputs.indices.toList(),
            "$.stemContract.bindings must bind every output exactly once in output-index order.",
        )
        bindings.forEachIndexed { bindingIndex, binding ->
            val path = "$.stemContract.bindings[$bindingIndex]"
            expect(binding.outputIndex in outputs.indices, "$path.outputIndex is out of range.")
            val output = outputs[binding.outputIndex]
            when (binding.packing.kind) {
                TensorOnlyStemPackingKind.AXIS_CONTIGUOUS -> {
                    val axis = requireNotNull(binding.packing.axis)
                    val elementsPerStem = requireNotNull(binding.packing.elementsPerStem)
                    expect(axis in output.shape.indices, "$path.packing.axis is out of range.")
                    expect(output.axes[axis] == STEM_AXIS, "$path.packing.axis must select the stem axis.")
                    expect(elementsPerStem == 1, "$path.packing.elementsPerStem must equal 1 for a stem axis.")
                    val expectedElements = try {
                        Math.multiplyExact(binding.stems.size, elementsPerStem)
                    } catch (error: ArithmeticException) {
                        throw BenchmarkModelContractException("$path packed stem dimension overflows Int.", error)
                    }
                    expect(
                        output.shape[axis] == expectedElements,
                        "$path packed axis must equal stem count times elementsPerStem.",
                    )
                }
                TensorOnlyStemPackingKind.WHOLE_TENSOR -> expect(
                    binding.stems.size == 1,
                    "$path whole-tensor packing must bind exactly one stem.",
                )
            }
        }

        val stems = when (workload.kind) {
            TensorOnlyWorkloadKind.WAVEFORM_SEGMENT -> contract.stems
            TensorOnlyWorkloadKind.NEURAL_CORE_SEGMENT -> {
                val referenceStems = bindings.first().stems
                bindings.drop(1).forEachIndexed { index, binding ->
                    expect(
                        binding.stems == referenceStems,
                        "$.stemContract.bindings[${index + 1}].stems must match the first neural-core output's ordered stems.",
                    )
                }
                referenceStems
            }
        }
        expect(stems.size in setOf(4, 6), "$.stemContract must describe exactly four or six stems.")
        val semantics = stems.map(BenchmarkStem::semantic)
        expect(semantics.toSet().size == semantics.size, "$.stemContract stem semantics must be unique.")
        val labels = stems.map(BenchmarkStem::displayLabel)
        expect(labels.toSet().size == labels.size, "$.stemContract display labels must be unique.")
    }

    private fun validateWorkloadAxis(
        tensor: TensorOnlyTensor,
        axisName: String,
        expectedSize: Int,
        path: String,
    ) {
        val axis = tensor.axes.indexOf(axisName)
        if (axis >= 0) {
            expect(tensor.shape[axis] == expectedSize, "$path $axisName axis must equal $expectedSize.")
        }
    }

    private fun requireHttpsUri(value: String, path: String) {
        val uri = try {
            URI(value)
        } catch (error: Exception) {
            throw BenchmarkModelContractException("$path must be a valid HTTPS URI.", error)
        }
        expect(
            uri.isAbsolute &&
                !uri.isOpaque &&
                uri.scheme.equals("https", ignoreCase = true) &&
                !uri.host.isNullOrBlank(),
            "$path must be an absolute hierarchical HTTPS URI with a host.",
        )
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private const val SCHEMA_VERSION = 3
    private const val CONTRACT_KIND = "tensor-only-candidate"
    private const val FLOAT32 = "float32"
    private const val FLOAT32_LE = "tensor-order-float32-le"
    private const val FLOAT32_BYTES = 4L
    private const val BATCH_AXIS = "batch"
    private const val STEM_AXIS = "stem"
    private const val CHANNEL_AXIS = "channel"
    private const val SAMPLE_AXIS = "sample"
    private const val DIRECT_DERIVATION = "direct"
    private const val PIPELINE_ID = "tensor-only-multistem"
    private const val MIT = "MIT"
    private const val PRECISION_SCOPE = "conversion-source-storage"
    private const val NO_QUANTIZATION = "none"
    private const val RUNTIME_PRECISION_NOT_ESTABLISHED = "not-established"
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val GIT_REVISION = Regex("^[0-9a-f]{40}$")
    private val MODEL_ID = Regex("^[a-z0-9_]+$")
    private val CONTRACT_ID = Regex("^[a-z0-9_]+@3$")
    private val HUB_REPOSITORY = Regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    private val LOCAL_PATH = Regex("^[A-Za-z0-9._/-]+$")
    private val AXES = setOf("batch", "stem", "channel", "sample", "complex", "frequency", "frame", "feature")
    private val VERIFICATION_STATUSES = setOf("local-structure-and-zero-input", "local-oracle-parity")
    private val DTYPE_BYTES = mapOf(
        "float16" to 2L,
        "float32" to 4L,
        "float64" to 8L,
        "int64" to 8L,
    )
}
