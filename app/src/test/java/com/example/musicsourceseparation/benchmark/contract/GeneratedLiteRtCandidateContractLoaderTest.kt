package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class GeneratedLiteRtCandidateContractLoaderTest {
    @Test
    fun loadsGeneratedTwoSecondCandidateAndBindsInspectedAbi() {
        val loaded = GeneratedLiteRtCandidateContractLoader.load(
            asset(),
            SIDECAR_BYTE_SIZE,
            SIDECAR_SHA256,
        )
        val contract = loaded.contract

        assertEquals("generated-litert-runtime-candidate", contract.contractKind)
        assertEquals("htdemucs_6s_core_smoke_2s_fp32_v1_0_0@1", contract.contractId)
        assertEquals(112_924_120L, contract.artifact.byteSize)
        assertEquals(
            "35ec0361b8ee6b415d435dc6d35d52dbf13bf9492b3f044e3259f414dbf23e61",
            contract.artifact.sha256,
        )
        assertEquals(listOf(1, 2, 88_200), contract.flatBuffer.inputs[0].shape)
        assertEquals(listOf(1, 4, 2_048, 87), contract.flatBuffer.inputs[1].shape)
        assertEquals(listOf(4_295, 4_300), contract.flatBuffer.outputs.map(ExternalLiteRtTensor::tensorIndex))
        assertEquals(
            listOf("args_0", "args_1"),
            contract.fixtures.inputs.map(GeneratedLiteRtFixtureTensor::bindingName),
        )
        assertEquals(
            listOf("output_0", "output_1"),
            contract.fixtures.outputs.map(GeneratedLiteRtFixtureTensor::bindingName),
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
        assertEquals("2.1.5", contract.runtimeTarget.liteRt.version)
        assertEquals("not-run", contract.runtimeTarget.targetDevice.status)
        assertTrue(contract.runtimeTarget.plans.all { it.status == "not-run" })
        assertEquals(
            "STABLEHLO_PAD",
            contract.flatBufferInspection.operatorHistogram.keys.single { it == "STABLEHLO_PAD" },
        )
        assertEquals(4, contract.flatBufferInspection.operatorHistogram["STABLEHLO_PAD"])
    }

    @Test
    fun rejectsUnknownTopLevelField() {
        assertInvalid(fixture().replaceFirst("\"contractSchemaVersion\": 1,", "\"contractSchemaVersion\": 1, \"unexpected\": true,"))
    }

    @Test
    fun rejectsArtifactIdentityMutation() {
        assertInvalid(
            fixture().replaceFirst(
                "35ec0361b8ee6b415d435dc6d35d52dbf13bf9492b3f044e3259f414dbf23e61",
                "0".repeat(64),
            ),
        )
    }

    @Test
    fun rejectsFixtureThatDoesNotBindSignature() {
        assertInvalid(fixture().replaceFirst("\"bindingName\": \"args_0\"", "\"bindingName\": \"args_x\""))
    }

    @Test
    fun rejectsHostMetricBelowProjectGate() {
        assertInvalid(fixture().replaceFirst("\"signalToNoiseDb\": 116.2312098868844", "\"signalToNoiseDb\": 79.0"))
    }

    @Test
    fun rejectsDeviceStatusClaimBeforeDeviceRun() {
        assertInvalid(fixture().replaceFirst("\"status\": \"not-run\"", "\"status\": \"passed\""))
    }

    @Test
    fun rejectsStaleInspectionReportIdentity() {
        assertInvalid(fixture().replaceFirst("\"byteSize\": 3766", "\"byteSize\": 3765"))
    }

    private fun assertInvalid(json: String) {
        val error = assertThrows(BenchmarkModelContractException::class.java) {
            GeneratedLiteRtCandidateContractLoader.parse(json)
        }
        assertTrue(error.message.orEmpty().isNotBlank())
    }

    private fun fixture(): String = asset().readText(Charsets.UTF_8)

    private fun asset(): File {
        val relative = "src/main/assets/benchmark-contracts/$ASSET_NAME"
        return sequenceOf(File(relative), File("app/$relative"))
            .firstOrNull(File::isFile)
            ?: error("Could not locate generated LiteRT candidate contract asset.")
    }

    companion object {
        private const val ASSET_NAME = "htdemucs_6s_core_smoke_2s_fp32_v1_0_0.json"
        private const val SIDECAR_BYTE_SIZE = 15_303L
        private const val SIDECAR_SHA256 =
            "a14474d308734c84003cfc83e2ccd746564022d34ad79d6066c87d50a3dd8d0c"
    }
}
