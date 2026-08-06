package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class ExternalLiteRtCandidateContractLoaderTest {
    @Test
    fun loadsPinnedBandBuddyRuntimeCandidate() {
        val loaded = ExternalLiteRtCandidateContractLoader.load(
            sidecarFile = asset(),
            expectedSidecarByteSize = SIDECAR_BYTE_SIZE,
            expectedSidecarSha256 = SIDECAR_SHA256,
        )
        val contract = loaded.contract

        assertEquals(4, contract.contractSchemaVersion)
        assertEquals("external-litert-runtime-candidate", contract.contractKind)
        assertEquals("bandbuddy_htdemucs_6s_core_v1_0_0@4", contract.contractId)
        assertEquals(SIDECAR_BYTE_SIZE, loaded.sidecarByteSize)
        assertEquals(SIDECAR_SHA256, loaded.sidecarSha256)

        assertEquals("htdemucs_6s_waveform_7p8s_onnx@3", contract.sourceCandidateContract.contractId)
        assertEquals(6_604L, contract.sourceCandidateContract.sidecarByteSize)
        assertEquals(
            "cc8dafb665602b6b8410ca87c6450e29ff4986990f22613e6451cf8da9f4c513",
            contract.sourceCandidateContract.sidecarSha256,
        )
        assertEquals("third-party-public-release", contract.provenance.artifactOrigin)
        assertEquals(
            "010d28948607e2d9c62b56ab1038681359869736",
            contract.provenance.sourceRevision,
        )
        assertEquals("incomplete-public-wrapper", contract.provenance.conversionRecipeStatus)

        assertEquals("htdemucs_6s.core.tflite", contract.artifact.fileName)
        assertEquals(117_784_760L, contract.artifact.byteSize)
        assertEquals(
            "a9fcc89e84aa65313e0540b582e710007ed12064969a0d49a3c85e49f1ae4e3d",
            contract.artifact.sha256,
        )

        val flatBuffer = contract.flatBuffer
        assertEquals("TFL3", flatBuffer.fileIdentifier)
        assertEquals(3, flatBuffer.schemaVersion)
        assertEquals(1, flatBuffer.subgraphCount)
        assertEquals(3_504, flatBuffer.operatorCount)
        assertEquals(4_332, flatBuffer.tensorCount)
        assertEquals(0, flatBuffer.customOperatorCount)
        assertEquals(
            listOf("serving_default_args_0", "serving_default_args_1"),
            flatBuffer.inputs.map(ExternalLiteRtTensor::name),
        )
        assertEquals(listOf(0, 1), flatBuffer.inputs.map(ExternalLiteRtTensor::tensorIndex))
        assertEquals(
            listOf(listOf(1, 2, 343_980), listOf(1, 4, 2_048, 336)),
            flatBuffer.inputs.map(ExternalLiteRtTensor::shape),
        )
        assertEquals(
            listOf("serving_default_output_0_output", "serving_default_output_1_output"),
            flatBuffer.outputs.map(ExternalLiteRtTensor::name),
        )
        assertEquals(listOf(4_292, 4_297), flatBuffer.outputs.map(ExternalLiteRtTensor::tensorIndex))
        assertEquals(
            listOf(listOf(1, 6, 4, 2_048, 336), listOf(1, 6, 2, 343_980)),
            flatBuffer.outputs.map(ExternalLiteRtTensor::shape),
        )
        assertEquals(listOf("float32", "float32"), flatBuffer.inputs.map(ExternalLiteRtTensor::dtype))
        assertEquals(listOf("float32", "float32"), flatBuffer.outputs.map(ExternalLiteRtTensor::dtype))

        val signature = flatBuffer.signatures.single()
        assertEquals("serving_default", signature.key)
        assertEquals(listOf("args_0", "args_1"), signature.inputs.map(ExternalLiteRtSignatureTensor::name))
        assertEquals(listOf(0, 1), signature.inputs.map(ExternalLiteRtSignatureTensor::tensorIndex))
        assertEquals(listOf("output_0", "output_1"), signature.outputs.map(ExternalLiteRtSignatureTensor::name))
        assertEquals(listOf(4_292, 4_297), signature.outputs.map(ExternalLiteRtSignatureTensor::tensorIndex))

        assertEquals(
            listOf("L.real", "L.imag", "R.real", "R.imag"),
            contract.modelSemantics.featureOrder,
        )
        assertEquals(
            listOf(
                BenchmarkStemSemantic.DRUMS,
                BenchmarkStemSemantic.BASS,
                BenchmarkStemSemantic.OTHER,
                BenchmarkStemSemantic.VOCALS,
                BenchmarkStemSemantic.GUITAR,
                BenchmarkStemSemantic.PIANO,
            ),
            contract.modelSemantics.stemOrder,
        )

        val runtime = contract.runtime
        assertEquals("org.tensorflow:tensorflow-lite:2.17.0", runtime.liteRt.declaredCoordinate)
        assertEquals("com.google.ai.edge.litert:litert:1.0.1", runtime.liteRt.relocatedCoordinate)
        assertEquals(6_487_444L, runtime.liteRt.resolvedArtifact.byteSize)
        assertEquals(
            "467679820f836fe70418f01c9f6689f328f79fbf2cbabe197eab70b209b43e8d",
            runtime.liteRt.resolvedArtifact.sha256,
        )
        assertEquals(68_393_877L, runtime.qnn.runtimeArtifact.byteSize)
        assertEquals(
            "ef47797c557e124e18eeea32c8cd2e346233d2d28a69f85e22da9fb0b639de06",
            runtime.qnn.runtimeArtifact.sha256,
        )
        assertEquals(659_678L, runtime.qnn.delegateArtifact.byteSize)
        assertEquals(
            "b8bd8bada3a11c36add5768804a49234bbeb0253ce50b9a603dfce23e6061acb",
            runtime.qnn.delegateArtifact.sha256,
        )
        assertEquals(
            listOf(586, 867, 2_639, 2_645, 2_777, 2_784, 2_923, 2_929, 3_157, 3_164, 3_303),
            runtime.npuPlan.delegatedNodeIds,
        )
        assertEquals(11, runtime.npuPlan.expectedDelegatedNodeCount)
        assertEquals(7, runtime.npuPlan.expectedPartitionCount)

        assertEquals(ExternalLiteRtCandidateStatus.THROUGHPUT_ONLY, contract.validation.status)
        assertEquals(
            ExternalLiteRtParityStatus.NOT_ESTABLISHED,
            contract.validation.officialSafetensorsParityStatus,
        )
        assertEquals(55.0, contract.validation.thirdPartyReportedSnr.minimumDb, 0.0)
        assertEquals(58.0, contract.validation.thirdPartyReportedSnr.maximumDb, 0.0)
        assertEquals(80.0, contract.validation.projectSnrGateDb, 0.0)
        assertEquals(
            ExternalLiteRtQualityGateStatus.NOT_PASSED,
            contract.validation.projectSnrGateStatus,
        )
    }

    @Test
    fun rejectsUnknownTopLevelField() {
        val json = fixture().replaceFirst(
            "\"contractSchemaVersion\": 4,",
            "\"contractSchemaVersion\": 4, \"unexpected\": true,",
        )

        assertInvalid(json, "unknown fields")
    }

    @Test
    fun rejectsChangedTensorName() {
        assertInvalid(
            replaceRequired(
                fixture(),
                "\"serving_default_args_0\"",
                "\"serving_default_mix\"",
            ),
            "fixed model ABI",
        )
    }

    @Test
    fun rejectsChangedWaveformShape() {
        assertInvalid(
            replaceRequired(fixture(), "[1, 2, 343980]", "[1, 2, 343979]"),
            "fixed model ABI",
        )
    }

    @Test
    fun rejectsMismatchedSignatureTensorIndex() {
        assertInvalid(
            replaceRequired(
                fixture(),
                "{\"name\": \"output_0\", \"tensorIndex\": 4292}",
                "{\"name\": \"output_0\", \"tensorIndex\": 4291}",
            ),
            "must match flatBuffer.outputs",
        )
    }

    @Test
    fun rejectsUnsortedNpuNodeIds() {
        assertInvalid(
            replaceRequired(fixture(), "[586, 867, 2639", "[867, 586, 2639"),
            "must be sorted",
        )
    }

    @Test
    fun rejectsIncorrectDelegatedNodeCount() {
        assertInvalid(
            replaceRequired(
                fixture(),
                "\"expectedDelegatedNodeCount\": 11",
                "\"expectedDelegatedNodeCount\": 12",
            ),
            "must match delegatedNodeIds",
        )
    }

    @Test
    fun rejectsLiteRtRelocationMismatch() {
        assertInvalid(
            replaceRequired(
                fixture(),
                "\"relocatedCoordinate\": \"com.google.ai.edge.litert:litert:1.0.1\"",
                "\"relocatedCoordinate\": \"com.google.ai.edge.litert:litert:1.0.2\"",
            ),
            "must equal relocatedCoordinate",
        )
    }

    @Test
    fun rejectsClaimThatProjectQualityGatePassed() {
        assertInvalid(
            replaceRequired(
                fixture(),
                "\"projectSnrGateStatus\": \"not-passed\"",
                "\"projectSnrGateStatus\": \"passed\"",
            ),
            "unsupported quality-gate status",
        )
    }

    @Test
    fun rejectsClaimOfOfficialSafetensorsParity() {
        assertInvalid(
            replaceRequired(
                fixture(),
                "\"officialSafetensorsParityStatus\": \"not-established\"",
                "\"officialSafetensorsParityStatus\": \"established\"",
            ),
            "unsupported parity status",
        )
    }

    @Test
    fun rejectsWrongExpectedSidecarIdentity() {
        val error = assertThrows(BenchmarkModelContractException::class.java) {
            ExternalLiteRtCandidateContractLoader.load(asset(), "0".repeat(64))
        }
        assert(error.message.orEmpty().contains("SHA-256 mismatch"))
    }

    private fun assertInvalid(json: String, messageFragment: String) {
        val error = assertThrows(BenchmarkModelContractException::class.java) {
            ExternalLiteRtCandidateContractLoader.parse(json)
        }
        assert(error.message.orEmpty().contains(messageFragment)) {
            "Expected '${error.message}' to contain '$messageFragment'."
        }
    }

    private fun fixture(): String = asset().readText(Charsets.UTF_8)

    private fun asset(): File {
        val relative = "src/main/assets/benchmark-contracts/$ASSET_NAME"
        return sequenceOf(File(relative), File("app/$relative"))
            .firstOrNull(File::isFile)
            ?: error("Could not locate external LiteRT contract asset $ASSET_NAME.")
    }

    private fun replaceRequired(source: String, oldValue: String, newValue: String): String {
        check(source.contains(oldValue)) { "Fixture did not contain expected mutation target." }
        return source.replaceFirst(oldValue, newValue)
    }

    companion object {
        private const val ASSET_NAME = "bandbuddy_htdemucs_6s_core_v1_0_0.json"
        private const val SIDECAR_BYTE_SIZE = 5_022L
        private const val SIDECAR_SHA256 =
            "ce0dfb4481eeb97a5d8dc32554ce90bcd6d2786bfbd63766909f836794288e09"
    }
}
