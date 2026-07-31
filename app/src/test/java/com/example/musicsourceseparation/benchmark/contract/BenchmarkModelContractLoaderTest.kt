package com.example.musicsourceseparation.benchmark.contract

import java.io.File
import java.nio.charset.StandardCharsets
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class BenchmarkModelContractLoaderTest {
    @Test
    fun loadsCanonical9662ContractAndDerivesMdxDimensions() {
        val loaded = BenchmarkModelContractLoader.load(canonicalSidecar())
        val contract = loaded.contract

        assertEquals(2, contract.contractSchemaVersion)
        assertEquals("uvr_mdxnet_3_9662@2", contract.contractId)
        assertEquals("uvr_mdxnet_3_9662", contract.modelId)
        assertEquals("UVR_MDXNET_3_9662_static_float32.tflite", contract.artifact.fileName)
        assertEquals(29_700_464L, contract.artifact.byteSize)
        assertEquals(
            "f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378",
            contract.artifact.sha256,
        )
        assertEquals(listOf(1, 2048, 256, 4), contract.tensorContract.input.shape)
        assertEquals(listOf(1, 2048, 256, 4), contract.tensorContract.output.shape)
        assertEquals(listOf(1, 2048, 256, 4), contract.expectedTensorShape)
        assertEquals(6_144, contract.dsp.nFft)
        assertEquals(1.035, contract.dsp.modelOutputScale, 0.0)
        assertEquals(261_120, contract.chunkSizeSamples)
        assertEquals(3_072, contract.trimSamples)
        assertEquals(254_976, contract.generationSizeSamples)
        assertEquals(BenchmarkStemSemantic.VOCALS, contract.stemContract.modelOutput.semantic)
        assertEquals(BenchmarkStemSemantic.INSTRUMENTAL, contract.stemContract.residual.semantic)
        assertEquals("booming-ss-mdx-stft", contract.pipelineCompatibility.pipelineId)
        assertEquals(2_273L, loaded.sidecarByteSize)
        assertEquals(CANONICAL_SIDECAR_SHA256, loaded.sidecarSha256)
    }

    @Test
    fun rejectsMalformedArtifactHash() {
        val tampered = replaceRequired(
            canonicalJson(),
            "f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378",
            "f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f837",
        )

        assertRejected(tampered, "artifact.sha256")
    }

    @Test
    fun pinnedSidecarRejectsAValidButTamperedArtifactHash() {
        val tampered = replaceRequired(
            canonicalJson(),
            "f74eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378",
            "074eee1ac06845a7cf277416138b19a6203f34316a3a74b2bde19acbfb2f8378",
        )
        val sidecar = File.createTempFile("tampered-9662-", ".json")
        try {
            sidecar.writeText(tampered, StandardCharsets.UTF_8)

            val error = assertThrows(BenchmarkModelContractException::class.java) {
                BenchmarkModelContractLoader.load(sidecar, CANONICAL_SIDECAR_SHA256)
            }
            assertTrue(error.message.orEmpty().contains("SHA-256 mismatch"))
        } finally {
            sidecar.delete()
        }
    }

    @Test
    fun rejectsTensorShapeInconsistentWithDsp() {
        val tampered = replaceRequired(
            canonicalJson(),
            "        2048,\n        256,\n        4",
            "        2047,\n        256,\n        4",
        )

        assertRejected(tampered, "input.shape")
    }

    @Test
    fun rejectsInvalidDspField() {
        val tampered = replaceRequired(canonicalJson(), "\"hopLength\": 1024", "\"hopLength\": 0")

        assertRejected(tampered, "dsp.hopLength")
    }

    @Test
    fun rejectsDspTimeFramesInconsistentWithPower() {
        val tampered = replaceRequired(
            canonicalJson(),
            "\"modelTimeFrames\": 256",
            "\"modelTimeFrames\": 255",
        )

        assertRejected(tampered, "modelTimeFrames must equal 2^dimTPower")
    }

    @Test
    fun rejectsUnsupportedStemSemantic() {
        val tampered = replaceRequired(canonicalJson(), "\"semantic\": \"vocals\"", "\"semantic\": \"lead\"")

        assertRejected(tampered, "unsupported stem semantic")
    }

    @Test
    fun rejectsAdditionalTopLevelField() {
        val tampered = replaceRequired(canonicalJson(), "{\n", "{\n  \"unexpected\": true,\n")

        assertRejected(tampered, "unknown fields: unexpected")
    }

    private fun assertRejected(json: String, messageFragment: String) {
        val error = assertThrows(BenchmarkModelContractException::class.java) {
            BenchmarkModelContractLoader.parse(json)
        }
        assertTrue(
            "Expected '${error.message}' to contain '$messageFragment'.",
            error.message.orEmpty().contains(messageFragment),
        )
    }

    private fun canonicalJson(): String = canonicalSidecar().readText(StandardCharsets.UTF_8)

    private fun canonicalSidecar(): File {
        val relative = "src/main/assets/benchmark-contracts/uvr_mdxnet_3_9662.json"
        return sequenceOf(File(relative), File("app/$relative"))
            .firstOrNull(File::isFile)
            ?: error("Could not locate canonical 9662 benchmark contract asset.")
    }

    private fun replaceRequired(source: String, oldValue: String, newValue: String): String {
        check(source.contains(oldValue)) { "Canonical fixture did not contain expected mutation target." }
        return source.replaceFirst(oldValue, newValue)
    }

    companion object {
        private const val CANONICAL_SIDECAR_SHA256 =
            "edf02de52bb45c842ad65a4be8f2118ed6d212ec4590a81ae0401e760bfb4fe9"
    }
}
