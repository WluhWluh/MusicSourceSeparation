package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

data class DerivedLiteRtBaseReference(
    val contractId: String,
    val modelId: String,
    val sidecarFileName: String,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
    val artifactByteSize: Long,
    val artifactSha256: String,
)

data class DerivedLiteRtArtifact(
    val fileName: String,
    val localPath: String,
    val byteSize: Long,
    val sha256: String,
    val format: String,
)

data class DerivedLiteRtRewrite(
    val ruleId: String,
    val toolPath: String,
    val toolSha256: String,
    val inputByteSize: Long,
    val inputSha256: String,
    val rewrittenOperatorCount: Int,
    val rewrittenShapeTensorCount: Int,
    val expectedEquivalence: String,
    val flatBufferByteSizePreserved: Boolean,
)

data class DerivedLiteRtFlatBufferDelta(
    val operatorCount: Int,
    val tensorCount: Int,
    val gatherNdCount: Int,
    val reshapeDelta: Int,
)

data class DerivedLiteRtValidationPlan(
    val cpuParity: String,
    val strictGpuPrepare: String,
    val hybridGpuRun: String,
)

data class DerivedLiteRtCandidateContract(
    val contractSchemaVersion: Int,
    val contractKind: String,
    val contractId: String,
    val modelId: String,
    val displayName: String,
    val base: DerivedLiteRtBaseReference,
    val artifact: DerivedLiteRtArtifact,
    val rewrite: DerivedLiteRtRewrite,
    val flatBufferDelta: DerivedLiteRtFlatBufferDelta,
    val validationPlan: DerivedLiteRtValidationPlan,
)

data class LoadedDerivedLiteRtCandidateContract(
    val contract: DerivedLiteRtCandidateContract,
    val base: LoadedExternalLiteRtCandidateContract,
    val sidecarFile: File,
    val sidecarByteSize: Long,
    val sidecarSha256: String,
)

object DerivedLiteRtCandidateContractLoader {
    fun load(
        sidecarFile: File,
        baseSidecarFile: File,
    ): LoadedDerivedLiteRtCandidateContract {
        val sidecarBytes = readBytes(sidecarFile)
        val sidecarSha256 = sha256(sidecarBytes)
        val contract = parse(decodeUtf8(sidecarBytes, sidecarFile))
        expect(baseSidecarFile.name == contract.base.sidecarFileName, "Base sidecar file does not match manifest.")
        val base = ExternalLiteRtCandidateContractLoader.load(baseSidecarFile)
        validateBase(contract, base, baseSidecarFile)
        return LoadedDerivedLiteRtCandidateContract(
            contract = contract,
            base = base,
            sidecarFile = sidecarFile.absoluteFile,
            sidecarByteSize = sidecarBytes.size.toLong(),
            sidecarSha256 = sidecarSha256,
        )
    }

    fun parse(json: String): DerivedLiteRtCandidateContract {
        val root = ObjectReader(StrictJsonParser(json).parse().asObject("$"), "$")
        root.requireExactKeys(
            "contractSchemaVersion",
            "contractKind",
            "contractId",
            "modelId",
            "displayName",
            "baseContract",
            "artifact",
            "rewrite",
            "flatBufferDelta",
            "validationPlan",
        )
        val schemaVersion = root.int("contractSchemaVersion")
        expect(schemaVersion == SCHEMA_VERSION, "$.contractSchemaVersion must equal $SCHEMA_VERSION.")
        val contractKind = root.string("contractKind")
        expect(contractKind == CONTRACT_KIND, "$.contractKind must equal '$CONTRACT_KIND'.")
        val modelId = root.string("modelId")
        expect(MODEL_ID.matches(modelId), "$.modelId is invalid.")
        val contractId = root.string("contractId")
        expect(contractId == "$modelId@$SCHEMA_VERSION", "$.contractId must equal '$modelId@$SCHEMA_VERSION'.")
        val base = parseBase(root.obj("baseContract"))
        val artifact = parseArtifact(root.obj("artifact"))
        val rewrite = parseRewrite(root.obj("rewrite"))
        val delta = parseDelta(root.obj("flatBufferDelta"))
        val validationPlan = parseValidationPlan(root.obj("validationPlan"))
        expect(
            modelId == "${base.modelId.removeSuffix("_v1_0_0")}_gather_reshape_v1_0_0",
            "$.baseContract.modelId must identify the base model.",
        )
        expect(rewrite.inputByteSize == base.artifactByteSize, "$.rewrite.inputByteSize must match the base artifact.")
        expect(rewrite.inputSha256 == base.artifactSha256, "$.rewrite.inputSha256 must match the base artifact.")
        expect(artifact.byteSize == base.artifactByteSize, "$.artifact.byteSize must preserve the base artifact byte size.")
        expect(rewrite.rewrittenOperatorCount == delta.reshapeDelta, "Rewrite operator count must match reshape delta.")
        expect(delta.gatherNdCount == 0, "$.flatBufferDelta.gatherNdCount must be zero after rewrite.")
        expect(delta.operatorCount > 0 && delta.tensorCount > 0, "FlatBuffer counts must be positive.")
        return DerivedLiteRtCandidateContract(
            contractSchemaVersion = schemaVersion,
            contractKind = contractKind,
            contractId = contractId,
            modelId = modelId,
            displayName = root.nonEmptyString("displayName"),
            base = base,
            artifact = artifact,
            rewrite = rewrite,
            flatBufferDelta = delta,
            validationPlan = validationPlan,
        )
    }

    private fun parseBase(value: JsonObject): DerivedLiteRtBaseReference {
        val reader = ObjectReader(value, "$.baseContract")
        reader.requireExactKeys(
            "contractId",
            "modelId",
            "sidecarFileName",
            "sidecarByteSize",
            "sidecarSha256",
            "artifactByteSize",
            "artifactSha256",
        )
        val sidecarFileName = reader.nonEmptyString("sidecarFileName")
        requireSafeFileName(sidecarFileName, "$.baseContract.sidecarFileName", ".json")
        return DerivedLiteRtBaseReference(
            contractId = reader.nonEmptyString("contractId"),
            modelId = reader.nonEmptyString("modelId"),
            sidecarFileName = sidecarFileName,
            sidecarByteSize = reader.positiveLong("sidecarByteSize"),
            sidecarSha256 = parseSha256(reader.string("sidecarSha256"), "$.baseContract.sidecarSha256"),
            artifactByteSize = reader.positiveLong("artifactByteSize"),
            artifactSha256 = parseSha256(reader.string("artifactSha256"), "$.baseContract.artifactSha256"),
        )
    }

    private fun parseArtifact(value: JsonObject): DerivedLiteRtArtifact {
        val reader = ObjectReader(value, "$.artifact")
        reader.requireExactKeys("fileName", "localPath", "byteSize", "sha256", "format")
        val fileName = reader.nonEmptyString("fileName")
        requireSafeFileName(fileName, "$.artifact.fileName", ".tflite")
        val localPath = reader.nonEmptyString("localPath")
        requireNormalizedRelativePath(localPath, "$.artifact.localPath")
        expect(localPath.substringAfterLast('/') == fileName, "$.artifact.localPath must end with fileName.")
        expect(reader.string("format") == TFLITE_FORMAT, "$.artifact.format must equal '$TFLITE_FORMAT'.")
        return DerivedLiteRtArtifact(
            fileName = fileName,
            localPath = localPath,
            byteSize = reader.positiveLong("byteSize"),
            sha256 = parseSha256(reader.string("sha256"), "$.artifact.sha256"),
            format = TFLITE_FORMAT,
        )
    }

    private fun parseRewrite(value: JsonObject): DerivedLiteRtRewrite {
        val reader = ObjectReader(value, "$.rewrite")
        reader.requireExactKeys(
            "ruleId",
            "toolPath",
            "toolSha256",
            "inputByteSize",
            "inputSha256",
            "rewrittenOperatorCount",
            "rewrittenShapeTensorCount",
            "expectedEquivalence",
            "flatBufferByteSizePreserved",
        )
        expect(reader.string("ruleId") == RULE_ID, "$.rewrite.ruleId must equal '$RULE_ID'.")
        val toolPath = reader.nonEmptyString("toolPath")
        requireNormalizedRelativePath(toolPath, "$.rewrite.toolPath")
        expect(toolPath == TOOL_PATH, "$.rewrite.toolPath must equal '$TOOL_PATH'.")
        val preserved = reader.boolean("flatBufferByteSizePreserved")
        expect(preserved, "$.rewrite.flatBufferByteSizePreserved must be true.")
        expect(reader.string("expectedEquivalence") == BITWISE, "$.rewrite.expectedEquivalence must be bitwise.")
        return DerivedLiteRtRewrite(
            ruleId = RULE_ID,
            toolPath = toolPath,
            toolSha256 = parseSha256(reader.string("toolSha256"), "$.rewrite.toolSha256"),
            inputByteSize = reader.positiveLong("inputByteSize"),
            inputSha256 = parseSha256(reader.string("inputSha256"), "$.rewrite.inputSha256"),
            rewrittenOperatorCount = reader.positiveInt("rewrittenOperatorCount"),
            rewrittenShapeTensorCount = reader.positiveInt("rewrittenShapeTensorCount"),
            expectedEquivalence = BITWISE,
            flatBufferByteSizePreserved = preserved,
        )
    }

    private fun parseDelta(value: JsonObject): DerivedLiteRtFlatBufferDelta {
        val reader = ObjectReader(value, "$.flatBufferDelta")
        reader.requireExactKeys("operatorCount", "tensorCount", "gatherNdCount", "reshapeDelta")
        return DerivedLiteRtFlatBufferDelta(
            operatorCount = reader.positiveInt("operatorCount"),
            tensorCount = reader.positiveInt("tensorCount"),
            gatherNdCount = reader.int("gatherNdCount"),
            reshapeDelta = reader.positiveInt("reshapeDelta"),
        )
    }

    private fun parseValidationPlan(value: JsonObject): DerivedLiteRtValidationPlan {
        val reader = ObjectReader(value, "$.validationPlan")
        reader.requireExactKeys("cpuParity", "strictGpuPrepare", "hybridGpuRun")
        return DerivedLiteRtValidationPlan(
            cpuParity = reader.string("cpuParity").also { expect(it == REQUIRED_BITWISE, "$.validationPlan.cpuParity is invalid.") },
            strictGpuPrepare = reader.string("strictGpuPrepare").also { expect(it == REQUIRED, "$.validationPlan.strictGpuPrepare is invalid.") },
            hybridGpuRun = reader.string("hybridGpuRun").also { expect(it == REQUIRED, "$.validationPlan.hybridGpuRun is invalid.") },
        )
    }

    private fun validateBase(
        contract: DerivedLiteRtCandidateContract,
        base: LoadedExternalLiteRtCandidateContract,
        baseSidecarFile: File,
    ) {
        val baseContract = base.contract
        expect(baseSidecarFile.name == contract.base.sidecarFileName, "Base sidecar file name does not match derived contract.")
        expect(base.sidecarByteSize == contract.base.sidecarByteSize, "Base sidecar byte size mismatch.")
        expect(base.sidecarSha256 == contract.base.sidecarSha256, "Base sidecar SHA-256 mismatch.")
        expect(baseContract.contractId == contract.base.contractId, "Base contract ID mismatch.")
        expect(baseContract.modelId == contract.base.modelId, "Base model ID mismatch.")
        expect(baseContract.artifact.byteSize == contract.base.artifactByteSize, "Base artifact byte size mismatch.")
        expect(baseContract.artifact.sha256 == contract.base.artifactSha256, "Base artifact SHA-256 mismatch.")
        expect(contract.artifact.sha256 != contract.base.artifactSha256, "Derived artifact must have a new SHA-256.")
        expect(contract.flatBufferDelta.operatorCount == baseContract.flatBuffer.operatorCount, "Operator count changed unexpectedly.")
        expect(contract.flatBufferDelta.tensorCount == baseContract.flatBuffer.tensorCount, "Tensor count changed unexpectedly.")
    }

    private fun readBytes(file: File): ByteArray {
        if (!file.isFile) invalid("Derived LiteRT contract sidecar is not readable: ${file.absolutePath}")
        return try {
            file.readBytes()
        } catch (error: Exception) {
            throw BenchmarkModelContractException("Could not read derived LiteRT contract sidecar: ${file.absolutePath}", error)
        }
    }

    private fun decodeUtf8(bytes: ByteArray, file: File): String {
        return try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (error: Exception) {
            throw BenchmarkModelContractException("Derived LiteRT contract sidecar is not valid UTF-8: ${file.absolutePath}", error)
        }
    }

    private fun requireSafeFileName(value: String, path: String, suffix: String) {
        expect(value.endsWith(suffix), "$path must end in $suffix.")
        expect(!value.contains('/') && !value.contains('\\') && value != "." && value != "..", "$path is not a file name.")
    }

    private fun requireNormalizedRelativePath(value: String, path: String) {
        expect(RELATIVE_PATH.matches(value), "$path contains unsupported characters.")
        expect(!value.startsWith('/'), "$path must be relative.")
        expect(value.split('/').none { it.isEmpty() || it == "." || it == ".." }, "$path is not normalized.")
    }

    private fun parseSha256(value: String, path: String): String = value.also {
        expect(SHA256.matches(it), "$path must be 64 lowercase hexadecimal characters.")
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256")
        .digest(bytes)
        .joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private const val SCHEMA_VERSION = 1
    private const val CONTRACT_KIND = "derived-litert-model-candidate"
    private const val TFLITE_FORMAT = "tflite-flatbuffer"
    private const val RULE_ID = "identity-gather-nd-to-reshape-v1"
    private const val TOOL_PATH = "tools/rewrite_tflite_identity_gather_nd.py"
    private const val BITWISE = "bitwise"
    private const val REQUIRED_BITWISE = "required-bitwise"
    private const val REQUIRED = "required"
    private val MODEL_ID = Regex("^[a-z0-9_]+$")
    private val SHA256 = Regex("^[0-9a-f]{64}$")
    private val RELATIVE_PATH = Regex("^[A-Za-z0-9._/-]+$")
}
